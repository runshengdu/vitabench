import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from vita.config import models, DEFAULT_LLM_EVALUATORS
from vita.data_model.simulation import RewardInfo, EvaluationType, SimulationRun, TerminationReason
from vita.data_model.tasks import Task
from vita.evaluator.evaluator_traj import TrajectoryEvaluator

class EvaluationAbortedError(RuntimeError):
    pass


def _evaluate_single_judge(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    domain: str,
    llm_evaluator: str,
    llm_args_evaluator: dict,
    language: str = None,
) -> RewardInfo:
    if evaluation_type == "trajectory":
        return TrajectoryEvaluator.calculate_reward(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    if evaluation_type == "trajectory_full_traj_rubric":
        return TrajectoryEvaluator.calculate_reward_full_traj_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    if evaluation_type == "trajectory_sliding_wo_rubric":
        return TrajectoryEvaluator.calculate_reward_sliding_wo_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    if evaluation_type == "trajectory_full_traj_wo_rubric":
        return TrajectoryEvaluator.calculate_reward_full_traj_wo_rubric(
            task=task,
            full_trajectory=simulation.messages,
            final_state=simulation.states,
            llm_evaluator=llm_evaluator,
            llm_args_evaluator=llm_args_evaluator,
            language=language,
        )
    raise ValueError(f"Unknown evaluation type: {evaluation_type}")


def _vote_from_reward(reward: float) -> int:
    return 1 if reward >= 0.5 else 0


def _call_with_retries(fn, retries: int = 3, desc: str | None = None):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(), attempt, None
        except Exception as e:
            last_exc = e
            if desc:
                logger.warning(
                    f"{desc} retry attempt={attempt}/{retries} error_type={type(e).__name__} error={e}"
                )
            else:
                logger.warning(
                    f"retry attempt={attempt}/{retries} error_type={type(e).__name__} error={e}"
                )
    return None, retries, last_exc

def evaluate_simulation(
    simulation: SimulationRun,
    task: Task,
    evaluation_type: EvaluationType,
    domain: str,
    llm_evaluators: list[str] | None = None,
    llm_args_evaluators: list[dict] | None = None,
    language: str = None,
    parallel_evaluators: bool = False,
) -> RewardInfo:
    """
    Evaluate the simulation based on the evaluation type.
    """
    if simulation.termination_reason in {
        TerminationReason.TOO_MANY_ERRORS,
        TerminationReason.MAX_STEPS,
        TerminationReason.INVALID_AGENT_MESSAGE,
    }:
        return RewardInfo(
            reward=0.0,
            info={
                "note": f"Simulation terminated prematurely. Termination reason: {simulation.termination_reason}"
            },
        )
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            info={"note": "No evaluation criteria"},
        )

    log_prefix = f"[eval:{domain}:{simulation.task_id}]"

    if llm_evaluators is None:
        llm_evaluators = DEFAULT_LLM_EVALUATORS
    if llm_args_evaluators is None:
        llm_args_evaluators = [models[name] for name in llm_evaluators]

    if len(llm_evaluators) != len(llm_args_evaluators):
        raise ValueError("llm_evaluators and llm_args_evaluators must have the same length")
    if len(llm_evaluators) < 1:
        raise ValueError("llm_evaluators must have length >= 1")
    if len(llm_evaluators) % 2 == 0:
        raise ValueError("llm_evaluators must have odd length")

    judge_records: list[dict] = []
    successes: list[tuple[str, RewardInfo]] = []
    all_evaluator_details: dict[str, dict] = {}

    def _run_one_evaluator(name: str, args: dict):
        reward_info, attempts, err = _call_with_retries(
            lambda: _evaluate_single_judge(
                simulation=simulation,
                task=task,
                evaluation_type=evaluation_type,
                domain=domain,
                llm_evaluator=name,
                llm_args_evaluator=args,
                language=language,
            ),
            retries=3,
            desc=f"{log_prefix} evaluator={name}",
        )
        return name, reward_info, attempts, err

    results_by_name: dict[str, tuple[str, RewardInfo | None, int, Exception | None]] = {}
    if parallel_evaluators and len(llm_evaluators) > 1:
        with ThreadPoolExecutor(max_workers=len(llm_evaluators)) as executor:
            futures = {
                executor.submit(_run_one_evaluator, name, args): name
                for name, args in zip(llm_evaluators, llm_args_evaluators)
            }
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    n, reward_info, attempts, err = fut.result()
                    results_by_name[n] = (n, reward_info, attempts, err)
                except Exception as e:
                    results_by_name[name] = (name, None, 3, e)
    else:
        for name, args in zip(llm_evaluators, llm_args_evaluators):
            n, reward_info, attempts, err = _run_one_evaluator(name, args)
            results_by_name[n] = (n, reward_info, attempts, err)

    for name, _args in zip(llm_evaluators, llm_args_evaluators):
        _n, reward_info, attempts, err = results_by_name.get(name, (name, None, 3, RuntimeError("missing evaluator result")))

        if err is None and reward_info is not None:
            successes.append((name, reward_info))
            judge_records.append(
                {
                    "llm_evaluator": name,
                    "status": "success",
                    "attempts": attempts,
                    "reward": reward_info.reward,
                    "vote": _vote_from_reward(reward_info.reward),
                }
            )
            all_evaluator_details[name] = {
                "status": "success",
                "attempts": attempts,
                "reward": reward_info.reward,
                "vote": _vote_from_reward(reward_info.reward),
                "reward_info": reward_info.model_dump(),
            }
            logger.info(
                f"{log_prefix} evaluator={name} status=success attempts={attempts} reward={reward_info.reward} vote={_vote_from_reward(reward_info.reward)}"
            )
        else:
            judge_records.append(
                {
                    "llm_evaluator": name,
                    "status": "failed",
                    "attempts": attempts,
                    "error": str(err),
                }
            )
            all_evaluator_details[name] = {
                "status": "failed",
                "attempts": attempts,
                "error": str(err),
            }
            logger.warning(
                f"{log_prefix} evaluator={name} status=failed attempts={attempts} error={str(err)}"
            )

    failures = [r for r in judge_records if r.get("status") == "failed"]
    failure_names = [r.get("llm_evaluator") for r in failures]

    if len(successes) == 0:
        logger.error(
            f"{log_prefix} judge_summary successes=0 failures={len(llm_evaluators)} status=aborted reason=all_evaluators_failed"
        )
        raise EvaluationAbortedError(
            f"All evaluators failed after 3 retries; aborting evaluation (n={len(llm_evaluators)})"
        )

    success_votes_by_name = {
        name: _vote_from_reward(ri.reward) for name, ri in successes
    }

    replacements: list[dict] = []
    final_votes: list[int] = []
    final_votes_by_evaluator: dict[str, int] = {}

    for record in judge_records:
        name = record.get("llm_evaluator")
        if record.get("status") == "success":
            vote = int(record.get("vote"))
            final_votes.append(vote)
            final_votes_by_evaluator[name] = vote
            continue

        picked_name, _picked_reward_info = random.choice(successes)
        vote = success_votes_by_name[picked_name]
        final_votes.append(vote)
        final_votes_by_evaluator[name] = vote

        record["replacement_picked"] = picked_name
        record["replacement_vote"] = vote
        replacements.append(
            {
                "failed": name,
                "picked": picked_name,
                "vote": vote,
            }
        )
        if name in all_evaluator_details:
            all_evaluator_details[name]["replacement_picked"] = picked_name
            all_evaluator_details[name]["replacement_vote"] = vote
        logger.warning(
            f"{log_prefix} replacement_vote failed={name} picked={picked_name}"
        )
        print(
            f"{log_prefix} replacement_vote failed={name} picked={picked_name} replacement_vote={vote}"
        )

    majority_vote = 1 if sum(final_votes) > (len(final_votes) // 2) else 0
    chosen_name, chosen_reward_info = next(
        (n, ri)
        for (n, ri) in successes
        if _vote_from_reward(ri.reward) == majority_vote
    )

    logger.info(
        f"{log_prefix} judge_summary successes={len(successes)} failures={len(failures)} majority_vote={majority_vote} majority_reward={float(majority_vote)} chosen={chosen_name}"
    )

    chosen_reward_info.reward = float(majority_vote)
    chosen_reward_info.nl_rubrics = None
    if chosen_reward_info.info is None:
        chosen_reward_info.info = {}
    chosen_reward_info.info = {
        **(chosen_reward_info.info or {}),
        "judge_mode": "majority_vote_reward",
        "llm_evaluators": llm_evaluators,
        "judge_records": judge_records,
        "replacements": replacements,
        "final_votes_by_evaluator": final_votes_by_evaluator,
        "majority_vote": majority_vote,
        "majority_reward": float(majority_vote),
        "failed_evaluators": failure_names,
        "all_evaluator_details": all_evaluator_details,
    }
    return chosen_reward_info
