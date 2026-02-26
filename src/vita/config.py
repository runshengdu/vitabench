import os
import yaml
from pathlib import Path

_models_yaml_path = Path(__file__).parent / "models.yaml"
if os.environ.get("VITA_MODEL_CONFIG_PATH", None):
    _models_yaml_path = Path(os.environ.get("VITA_MODEL_CONFIG_PATH"))

_evaluators_yaml_path = Path(__file__).parent / "evaluators.yaml"
if os.environ.get("VITA_EVALUATOR_CONFIG_PATH", None):
    _evaluators_yaml_path = Path(os.environ.get("VITA_EVALUATOR_CONFIG_PATH"))

if not os.path.exists(str(_models_yaml_path)):
    raise FileNotFoundError(
        f"Model configuration file ({_models_yaml_path}) dose not exists, you should create it first.")

if not os.path.exists(str(_evaluators_yaml_path)):
    raise FileNotFoundError(
        f"Evaluator configuration file ({_evaluators_yaml_path}) dose not exists, you should create it first.")


DEFAULT_LLM_TIMEOUT = 600


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value

    return result


def _resolve_env_vars(obj):
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_vars(v) for v in obj]
    if isinstance(obj, str):
        for key, val in os.environ.items():
            obj = obj.replace(f"${{{key}}}", val)
        return obj
    return obj


def _normalize_api_config(cfg: dict) -> dict:
    cfg = _resolve_env_vars(cfg)
    if not isinstance(cfg, dict):
        return cfg

    if cfg.get("timeout") is None:
        cfg = cfg.copy()
        cfg["timeout"] = DEFAULT_LLM_TIMEOUT

    if cfg.get("headers") is None and cfg.get("api_key"):
        cfg = cfg.copy()
        cfg["headers"] = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['api_key']}",
        }
    return cfg


try:
    with open(_models_yaml_path, 'r', encoding='utf-8') as f:
        models_config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    default_model_config = models_config_yaml.get('default', {})

    models = {"default": default_model_config}
    for model in models_config_yaml.get('models', []):
        model_name = model['name']
        merged_config = _deep_merge_dict(default_model_config, model)
        models[model_name] = _normalize_api_config(merged_config)

    _available_model_names = [k for k in models.keys() if k != "default"]

    print(f"Available models: {list(models.keys())}")

except FileNotFoundError:
    print(f"Warning: models.yaml not found at {_models_yaml_path}")
    models = {}
    _available_model_names = []
except Exception as e:
    print(f"Error loading models.yaml: {e}")
    models = {}
    _available_model_names = []


try:
    with open(_evaluators_yaml_path, 'r', encoding='utf-8') as f:
        evaluators_config_yaml = yaml.load(f, Loader=yaml.FullLoader)

    default_evaluator_config = evaluators_config_yaml.get('default', {})

    evaluators = {"default": default_evaluator_config}
    for model in evaluators_config_yaml.get('models', []):
        model_name = model['name']
        merged_config = _deep_merge_dict(default_evaluator_config, model)
        evaluators[model_name] = _normalize_api_config(merged_config)

    _available_evaluator_names = [k for k in evaluators.keys() if k != "default"]

except FileNotFoundError:
    print(f"Warning: evaluators.yaml not found at {_evaluators_yaml_path}")
    evaluators = {}
    _available_evaluator_names = []
except Exception as e:
    print(f"Error loading evaluators.yaml: {e}")
    evaluators = {}
    _available_evaluator_names = []

models.update(evaluators)

# SIMULATION
DEFAULT_MAX_STEPS = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 30
DEFAULT_MAX_EVALUATIONS = 30
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LANGUAGE = "english"
DEFAULT_EVALUATION_TYPE = "trajectory"

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "openai/gpt-5.2"
DEFAULT_LLM_USER = "qwen3.5-plus"
DEFAULT_LLM_EVALUATORS = [
    "doubao-seed-2-0-pro-260215",
    "qwen3.5-plus",
    "deepseek-chat",
]
