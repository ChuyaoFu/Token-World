"""Compare two completed open-loop reports with identical evaluation protocols.

Confidence intervals resample tasks as paired clusters; they describe task
variation for these checkpoints, not uncertainty across training seeds.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenworld.evaluation.aggregate import macro_average

PROTOCOL_FIELDS = (
    "version", "protocol", "reference", "initial_context_frames", "sampling_steps",
    "views", "latent_method", "max_steps", "query_every", "chunk_size", "split",
    "cache", "codec_checkpoint", "policy_bundle", "seed",
    "action_feature_interface", "action_reference", "qwen_model", "raw_data_root",
)


def compare(baseline, candidate, seed=3047):
    for report in (baseline, candidate):
        if report.get("status") != "complete":
            raise ValueError("Only completed reports can be compared")
        if report.get("num_episodes") != len(report["episodes"]):
            raise ValueError("Report episode count is inconsistent")
    differences = [key for key in PROTOCOL_FIELDS if baseline.get(key) != candidate.get(key)]
    if differences:
        raise ValueError(f"Evaluation protocols differ: {differences}")
    def identities(report):
        return sorted((e["task"], e["episode_id"], e["source_outcome"],
                       e["predicted_frames"], e["action_query_frames"])
                      for e in report["episodes"])
    if identities(baseline) != identities(candidate):
        raise ValueError("Episode identities, horizons, or action-query counts differ")
    old, new = (macro_average(report["episodes"]) for report in (baseline, candidate))
    tasks = sorted(old["per_task"])
    rng = random.Random(seed)
    resamples = [[rng.randrange(len(tasks)) for _ in tasks] for _ in range(10000)]
    rows = []
    for branch in ("qwen_feature_metrics", "action_metrics"):
        for metric in ("cosine", "nmse"):
            if metric not in old[branch]:
                continue
            delta = [new["per_task"][task][branch][metric] -
                     old["per_task"][task][branch][metric] for task in tasks]
            bootstrap = sorted(mean(delta[i] for i in draw) for draw in resamples)
            rows.append({
                "branch": branch, "metric": metric,
                "higher_is_better": metric == "cosine",
                "baseline": old[branch][metric], "candidate": new[branch][metric],
                "candidate_minus_baseline": mean(delta),
                "paired_task_bootstrap_95pct": [bootstrap[249], bootstrap[9749]],
            })
    outcomes = {}
    for outcome in sorted({e["source_outcome"] for e in baseline["episodes"]}):
        outcomes[outcome] = {
            name: macro_average([e for e in report["episodes"] if e["source_outcome"] == outcome])
            for name, report in (("baseline", baseline), ("candidate", candidate))
        }
    return {
        "protocol": {key: candidate.get(key) for key in PROTOCOL_FIELDS},
        "baseline_checkpoint": baseline["world_model_checkpoint"],
        "candidate_checkpoint": candidate["world_model_checkpoint"],
        "num_tasks": len(tasks), "num_episodes": len(candidate["episodes"]),
        "predicted_frames": sum(e["predicted_frames"] for e in candidate["episodes"]),
        "aggregation": old["method"], "metrics": rows,
        "baseline_macro": old, "candidate_macro": new, "by_outcome": outcomes,
        "uncertainty": "10,000 paired task bootstrap draws; no training-seed uncertainty",
        "bootstrap_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# World-model checkpoint comparison", "",
             f"{result['num_episodes']} matched episodes across {result['num_tasks']} tasks; "
             f"{result['predicted_frames']} predicted frames per checkpoint.",
             "Task macro averages; delta = candidate − baseline.", "",
             "| Metric | Baseline | Candidate | Delta | Paired task 95% interval |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in result["metrics"]:
        lo, hi = row["paired_task_bootstrap_95pct"]
        lines.append(f"| {row['branch']}/{row['metric']} | {row['baseline']:.6f} | "
                     f"{row['candidate']:.6f} | {row['candidate_minus_baseline']:+.6f} | "
                     f"[{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "These are open-loop feature/action fidelity metrics, not task-success rates.",
              "Intervals do not measure variability across training seeds.", ""]
    args.output.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
