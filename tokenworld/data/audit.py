"""Validate the fixed RoboTwin episode split and optional LeRobot files, read-only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit_manifest(manifest: dict, data_root: Path | None = None) -> dict:
    datasets = manifest["datasets"]
    if len(datasets) != 2:
        raise ValueError("Expected the two success/failure datasets")
    counts = Counter()
    task_sets = []
    file_count = 0
    bytes_total = 0
    outcomes = set()
    deviations = []
    for name, dataset in datasets.items():
        outcome = dataset["outcome"]
        outcomes.add(outcome)
        tasks = dataset["tasks"]
        task_sets.append(set(tasks))
        if len(tasks) != 50:
            raise ValueError(f"{name}: expected 50 tasks, got {len(tasks)}")
        seen = set()
        info = None
        if data_root is not None:
            info = json.loads((data_root / name / "meta/info.json").read_text())
        for task, split in tasks.items():
            train, val = split["train"], split["val"]
            if len(train) != 45 or len(val) != 5 or split.get("unused"):
                deviations.append({"dataset": name, "task": task, "train": len(train),
                                   "val": len(val), "unused": len(split.get("unused", [])),
                                   "expected_train": 45, "expected_val": 5})
            ids = train + val
            if len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError(f"Repeated episode in {name}/{task}")
            seen.update(ids)
            counts[f"{outcome}_train"] += len(train)
            counts[f"{outcome}_val"] += len(val)
            if info is not None:
                keys = [key for key, spec in info["features"].items() if spec.get("dtype") == "video"]
                if not keys:
                    raise ValueError(f"{name}: no video features in metadata")
                for index in ids:
                    fields = {"episode_index": index, "episode_chunk": index // int(info.get("chunks_size", 1000))}
                    paths = [info["data_path"].format(**fields)]
                    paths += [info["video_path"].format(**fields, video_key=key) for key in keys]
                    for relative in paths:
                        p = data_root / name / relative
                        if not p.is_file():
                            raise FileNotFoundError(p)
                        file_count += 1
                        bytes_total += p.stat().st_size
    if task_sets[0] != task_sets[1] or outcomes != {"success", "failure"}:
        raise ValueError("Success/failure task sets do not match")
    return {"tasks": 50, "episodes": sum(counts.values()), "splits": dict(counts),
            "paper_expected_episodes": 5000, "paper_count_matches": not deviations,
            "paper_count_deviations": deviations,
            "train_val_disjoint": True, "episode_ids_unique_within_dataset": True,
            "task_names": sorted(task_sets[0]), "files_checked": file_count, "referenced_file_bytes": bytes_total,
            "validation": "split_and_file_existence" if data_root else "split_only",
            "video_decode_and_parquet_contents_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_manifest(json.loads(args.manifest.read_text()), args.data_root)
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
