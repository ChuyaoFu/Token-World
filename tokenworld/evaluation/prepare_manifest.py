#!/usr/bin/env python3
"""Select a fixed number of distinct Robotwin initial states per task.

This helper only prepares the closed-loop input manifest. The source replay
manifest and cache are read-only; the selected entries are re-indexed for
stable per-video output names.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=5)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def choose_group(entries: list[dict], count: int, seed: int) -> list[dict]:
    if len(entries) < count:
        raise ValueError(f"Task {entries[0]['task']!r} has only {len(entries)} entries; need {count}")
    rng = random.Random(seed)
    by_outcome: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_outcome[str(entry.get("outcome", "unknown"))].append(entry)
    for values in by_outcome.values():
        values.sort(key=lambda item: (int(item.get("episode_index", 0)), str(item.get("cache_path", ""))))
        rng.shuffle(values)

    # Keep both success/failure domains represented whenever the source has
    # both, then fill the remainder from a deterministic shuffle.
    selected: list[dict] = []
    outcomes = [key for key in ("success", "failure") if by_outcome.get(key)]
    if len(outcomes) > 1:
        target_success = (count + 1) // 2
        target_failure = count - target_success
        selected.extend(by_outcome["success"][:target_success])
        selected.extend(by_outcome["failure"][:target_failure])
    if len(selected) < count:
        remaining = [entry for entry in entries if entry not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected[:count]


def main() -> None:
    args = parse_args()
    if args.per_task <= 0 or args.expected_tasks <= 0:
        raise ValueError("--per-task and --expected-tasks must be positive")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    entries = list(source["episodes"])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["task"])].append(entry)
    tasks = sorted(grouped)
    if len(tasks) != args.expected_tasks:
        raise ValueError(f"Expected {args.expected_tasks} tasks, found {len(tasks)}")

    selected: list[dict] = []
    for task_index, task in enumerate(tasks):
        for entry in choose_group(grouped[task], args.per_task, args.seed + task_index):
            copied = dict(entry)
            copied["source_manifest_index"] = int(entry.get("manifest_index", -1))
            copied["manifest_index"] = len(selected)
            selected.append(copied)

    payload = {
        "version": 1,
        "selection": "fixed_per_task_closed_loop_initial_states",
        "seed": args.seed,
        "per_task": args.per_task,
        "num_tasks": len(tasks),
        "num_episodes": len(selected),
        "source_manifest": str(args.input.resolve()),
        "episodes": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(selected)} episodes, {len(tasks)} tasks x {args.per_task})")


if __name__ == "__main__":
    main()
