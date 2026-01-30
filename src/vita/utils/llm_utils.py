import copy
import json
import re
import time
from urllib.parse import urlparse
from typing import Any, Optional, List, Dict

from loguru import logger
from openai import OpenAI


from vita.config import (
    models,
    DEFAULT_MAX_RETRIES,
    DEFAULT_LLM_TIMEOUT,
)
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def get_response_cost(usage, model) -> float:
    num_prompt_token = usage["prompt_tokens"]
    num_completion_token = usage["completion_tokens"]
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar",{}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1000000
    else:
        return 0.0


def get_response_usage(response) -> Optional[dict]:
    usage = response.get("usage", {})
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0)
    }


def format_messages(messages: list[Message]) -> list[dict]:
    messages_formatted = []
    for message in messages:
        if isinstance(message, UserMessage):
            messages_formatted.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            messages_formatted.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
            # add reasoning content if exists
            if message.raw_data is not None and message.raw_data.get("message") is not None:
                raw_message = message.raw_data["message"]

                reasoning_details = raw_message.get("reasoning_details")
                if reasoning_details is not None:
                    messages_formatted[-1]["reasoning_details"] = reasoning_details
                else:
                    reasoning_content = raw_message.get("reasoning_content")
                    if reasoning_content is not None:
                        messages_formatted[-1]["reasoning_content"] = reasoning_content
        elif isinstance(message, ToolMessage):
            messages_formatted.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            messages_formatted.append({"role": "system", "content": message.content})
    return messages_formatted


def add_anthropic_caching(messages: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    if not ("minimax" in model_name.lower() or "claude" in model_name.lower()):
        return messages

    cached_messages = copy.deepcopy(messages)

    for n in range(len(cached_messages)):
        if n >= len(cached_messages) - 3:
            msg = cached_messages[n]
            if not isinstance(msg, dict):
                continue

            content = msg.get("content")
            if isinstance(content, str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            elif isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and "type" in content_item:
                        content_item["cache_control"] = {"type": "ephemeral"}

    return cached_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    **kwargs: Any,
) -> UserMessage | AssistantMessage:
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """
    try:
        if kwargs.get("num_retries") is None:
            kwargs["num_retries"] = DEFAULT_MAX_RETRIES
        messages_formatted = format_messages(messages)
        if kwargs.get("enable_prompt_caching"):
            messages_formatted = add_anthropic_caching(messages_formatted, model)
        tools = [tool.openai_schema for tool in tools] if tools else None
        if tools and tool_choice is None:
            tool_choice = "auto"
        model_cfg = models.get(model, {}) or {}
        base_url = model_cfg.get("base_url")
        api_key = model_cfg.get("api_key")
        if base_url is None:
            raise KeyError(f"Missing base_url for model: {model}")
        if api_key is None:
            raise KeyError(f"Missing api_key for model: {model}")

        if isinstance(base_url, str):
            if base_url.endswith("/chat/completions"):
                base_url = base_url[: -len("/chat/completions")]
            elif base_url.endswith("/completions"):
                base_url = base_url[: -len("/completions")]

        parsed = urlparse(base_url)
        if parsed.path in ("", "/"):
            base_url = base_url.rstrip("/") + "/v1"

        headers = model_cfg.get("headers") or {}
        extra_headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in {"authorization", "content-type"}
        }

        temperature = kwargs.get("temperature")
        if temperature is None:
            temperature = model_cfg.get("temperature")

        max_tokens = kwargs.get("max_tokens")
        if max_tokens is None:
            max_tokens = model_cfg.get("max_tokens")
        if max_tokens is None:
            max_tokens = model_cfg.get("max_completion_tokens")

        extra_body = model_cfg.get("extra_body")

        reserved_cfg_keys = {
            "base_url",
            "api_key",
            "headers",
            "cost_1m_token_dollar",
            "extra_body",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "seed",
            "timeout",
            "name",
        }
        passthrough_body = {
            k: v for k, v in model_cfg.items() if k not in reserved_cfg_keys
        }
        if passthrough_body:
            if extra_body is None:
                extra_body = {}
            if isinstance(extra_body, dict):
                extra_body = {**extra_body, **passthrough_body}

        if kwargs.get("extra_body") is not None:
            if isinstance(extra_body, dict) and isinstance(kwargs.get("extra_body"), dict):
                extra_body = {**extra_body, **kwargs.get("extra_body")}
            else:
                extra_body = kwargs.get("extra_body")

        seed = kwargs.get("seed")
        if seed is None:
            seed = model_cfg.get("seed")

        timeout = kwargs.get("timeout")
        if timeout is None:
            timeout = model_cfg.get("timeout")
        if timeout is None:
            timeout = DEFAULT_LLM_TIMEOUT

        max_retries = int(kwargs.get("num_retries") or 0)
        retry_delay = 1
        response_dict = None
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=timeout,
                )
                response = client.chat.completions.create(
                    model=model,
                    messages=messages_formatted,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    extra_headers=extra_headers or None,
                    extra_body=extra_body or None,
                )
                response_dict = response.model_dump()
                break
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    logger.warning(
                        f"OpenAI SDK call failed, attempt {attempt + 1} retry, retrying in {retry_delay} seconds... Error: {e}"
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

        if response_dict is None:
            raise last_err if last_err is not None else RuntimeError("OpenAI SDK call failed")

        usage = get_response_usage(response_dict)
        cost = get_response_cost(usage, model)
        try:
            response = response_dict['choices'][0]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(
                f"Full response: {json.dumps(response_dict, ensure_ascii=False, indent=2) if isinstance(response_dict, dict) else response_dict}"
            )
            raise ValueError(f"Invalid API response format: {e}") from e
        assert response['message']['role'] == "assistant", (
            "The response should be an assistant message"
        )
        content = response['message'].get('content')
        tool_calls = response['message'].get('tool_calls') or []
        tool_calls = [
            ToolCall(
                id=tool_call.get('id'),
                name=tool_call.get('function', {}).get('name'),
                arguments=json.loads(tool_call.get('function', {}).get('arguments')) if tool_call.get('function', {}).get('arguments') else {},
            )
            for tool_call in tool_calls
        ]
        tool_calls = tool_calls or None
        message = AssistantMessage(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            cost=cost,
            usage=usage,
            raw_data=response,
        )
        return message
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(e)


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost
