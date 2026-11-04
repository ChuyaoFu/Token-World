"""Aggregate episode metrics within tasks, then equally across tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean

METRICS = ("nmse", "mse", "rmse", "mae", "row_cosine_similarity", "cosine")


def macro_average(episodes: list[dict]) -> dict:
    groups = defaultdict(list)
    identities = set()
    for episode in episodes:
        identity = (episode["task"], episode["episode_id"])
        if identity in identities:
            raise ValueError(f"Duplicate episode: {identity}")
        identities.add(identity)
        groups[episode["task"]].append(episode)
    if not groups:
        raise ValueError("No episodes to aggregate")
    per_task = {}
    for task, rows in sorted(groups.items()):
        report = {"episodes": len(rows)}
        for branch in ("qwen_feature_metrics", "action_metrics"):
            measures = {}
            for metric in METRICS:
                values = [row.get(branch, {}).get(metric) for row in rows]
                if all(v is None for v in values):
                    continue
                if any(v is None or not math.isfinite(float(v)) for v in values):
                    raise ValueError(f"Missing or non-finite {task}/{branch}/{metric}")
                measures[metric] = mean(values)
            report[branch] = measures
        per_task[task] = report
    result = {"method": "equal_episode_mean_within_task_then_equal_task_mean",
              "num_tasks": len(groups), "num_episodes": len(episodes), "per_task": per_task}
    for branch in ("qwen_feature_metrics", "action_metrics"):
        keys = set().union(*(row[branch] for row in per_task.values()))
        if any(set(row[branch]) != keys for row in per_task.values()):
            raise ValueError(f"Inconsistent metric coverage across tasks: {branch}")
        result[branch] = {key: mean(row[branch][key] for row in per_task.values()) for key in sorted(keys)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = []
    protocols = []
    for path in args.input:
        payload = json.loads(path.read_text())
        episodes.extend(payload["episodes"])
        protocols.append({key: payload.get(key) for key in (
            "protocol", "reference", "initial_context_frames", "sampling_steps", "views", "latent_method",
            "max_steps", "query_every", "chunk_size", "split", "world_model_checkpoint", "codec_checkpoint", "policy_bundle", "action_feature_interface", "action_reference", "qwen_model", "raw_data_root")})
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("Cannot merge different protocols/models; aggregate each experiment separately")
    result = {"protocol": protocols[0], "aggregate_macro": macro_average(episodes)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
