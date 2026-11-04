#!/usr/bin/env python3
"""Create one deterministic, balanced RobotWin replay manifest.

The manifest is deliberately independent of either model.  Both replay
programs consume the same cache records, which makes the RGB/latent artifacts
directly comparable episode by episode.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--all-validation",
        action="store_true",
        help="Use every validation record instead of a smaller balanced subset.",
    )
    parser.add_argument("--seed", type=int, default=3047)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 and not args.all_validation:
        raise ValueError("--episodes must be positive")
    with args.index.open(encoding="utf-8") as handle:
        records = json.load(handle)["records"]
    records = [r for r in records if str(r.get("split")) == "val"]
    grouped = {"success": [], "failure": []}
    for index, record in enumerate(records):
        outcome = str(record["outcome"])
        if outcome not in grouped:
            raise ValueError(f"Unexpected outcome {outcome!r}")
        grouped[outcome].append((index, record))
    rng = random.Random(args.seed)
    for values in grouped.values():
        rng.shuffle(values)

    if args.all_validation:
        target_success = len(grouped["success"])
        target_failure = len(grouped["failure"])
    else:
        target_success = args.episodes // 2
        target_failure = args.episodes - target_success
    if len(grouped["success"]) < target_success or len(grouped["failure"]) < target_failure:
        raise ValueError("Not enough validation records for a balanced manifest")

    selected = grouped["success"][:target_success] + grouped["failure"][:target_failure]
    rng.shuffle(selected)
    output = []
    for cache_record_index, record in selected:
        source_id = int(record["episode_index"])
        # prepare_robotwin_success_fail.py keeps success ids and offsets
        # failure ids by 2500 to avoid collisions in the flat Ctrl dataset.
        ctrl_episode_id = source_id if record["dataset"] == "robotwin_success" else 2500 + source_id
        output.append(
            {
                "manifest_index": len(output),
                "cache_record_index": int(cache_record_index),
                "dataset": str(record["dataset"]),
                "episode_index": source_id,
                "outcome": str(record["outcome"]),
                "task": str(record["task"]),
                "num_frames": int(record["num_frames"]),
                "cache_path": str(record["path"]),
                "ctrl_episode_id": int(ctrl_episode_id),
            }
        )

    payload = {
        "version": 1,
        "seed": args.seed,
        "source_index": str(args.index),
        "split": "val",
        "selection": (
            "all_validation"
            if args.all_validation
            else "balanced_success_failure"
        ),
        "episodes": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {args.output} ({len(output)} episodes)")
    for item in output:
        print(
            f"{item['manifest_index']:02d} "
            f"{item['dataset']}:{item['episode_index']:04d} "
            f"ctrl={item['ctrl_episode_id']:04d} "
            f"{item['task']} frames={item['num_frames']}"
        )


if __name__ == "__main__":
    main()
