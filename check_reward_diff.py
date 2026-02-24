#!/usr/bin/env python3
# this program can calculate how many simulation results have different judge rewards
import argparse
import json
from pathlib import Path
from typing import Iterable


def _iter_simulations(data: dict) -> Iterable[dict]:
    sims = data.get("simulations")
    if isinstance(sims, list):
        for sim in sims:
            if isinstance(sim, dict):
                yield sim


def _extract_votes(sim: dict) -> list[int]:
    reward_info = sim.get("reward_info") or {}
    info = reward_info.get("info") or {}
    judge_records = info.get("judge_records") or []
    votes: list[int] = []
    if isinstance(judge_records, list):
        for record in judge_records:
            if not isinstance(record, dict):
                continue
            if "vote" in record:
                try:
                    votes.append(int(record["vote"]))
                except (TypeError, ValueError):
                    continue
    return votes


def _is_inconsistent(votes: list[int]) -> bool:
    if not votes:
        return False
    unique_votes = {vote for vote in votes if vote in {0, 1}}
    return len(unique_votes) > 1


def _scan_file(path: Path, show: bool) -> int:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    inconsistent = []
    for sim in _iter_simulations(data):
        votes = _extract_votes(sim)
        if _is_inconsistent(votes):
            inconsistent.append(sim.get("task_id") or sim.get("id"))

    print(f"{path}: inconsistent_tasks={len(inconsistent)}")
    if show and inconsistent:
        for task_id in inconsistent:
            print(f"  {task_id}")
    return len(inconsistent)


def _iter_json_files(directory: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(directory.rglob(pattern)):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check tasks with inconsistent judge votes (not all 0 or all 1)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path", type=Path, help="Path to simulation JSON file")
    group.add_argument("--dir", type=Path, help="Directory containing simulation JSON files")
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for JSON files under --dir (default: *.json)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Show task IDs with inconsistent votes"
    )
    args = parser.parse_args()

    if args.path:
        _scan_file(args.path, args.show)
        return

    total = 0
    files = list(_iter_json_files(args.dir, args.pattern))
    for path in files:
        total += _scan_file(path, args.show)
    print(f"total_inconsistent_tasks={total}")


if __name__ == "__main__":
    main()
