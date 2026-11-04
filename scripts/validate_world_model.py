"""Evaluate the checkpoint's configured validation windows without training.

Run via bash scripts/run_python.sh scripts/validate_world_model.py --help.
This measures the stochastic training objective, not autoregressive fidelity.
Use identical batch size and seed when comparing checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from tokenworld.data.dataset import RobotwinSVAETokenDataset
from tokenworld.utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_batches < 0:
        parser.error("Invalid batch size, workers, or maximum batches")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model, config = load_checkpoint(str(args.checkpoint), device=str(device))
    data = config["data"]
    if data.get("frameskip", 1) != 1 or data.get("addskip", 0) != 0:
        raise ValueError("This validation entrypoint requires frameskip=1/addskip=0")
    fields = ("num_hist", "num_pred", "latent_method", "tasks", "outcomes",
              "views", "normalize_latents", "latent_stats_scope", "normalize_action",
              "verify_action_equals_state", "tail_align", "lru_episodes", "action_schema")
    options = {key: data[key] for key in fields if key in data}
    options.update(sequence_interval=data.get("val_sequence_interval", 1),
                   window_stride=data.get("val_window_stride"),
                   random_window_per_episode=False)
    dataset = RobotwinSVAETokenDataset(args.cache, split="val", **options)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    precision = config["training"].get("mixed_precision", "no")
    if precision not in ("no", "bf16", "fp16"):
        raise ValueError(f"Unsupported precision: {precision}")
    model.eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress = args.output.with_suffix(".batches.jsonl")
    rows, sums, samples = [], {}, 0
    with progress.open("w") as stream, torch.inference_mode():
        for index, batch in enumerate(loader):
            if args.max_batches and index >= args.max_batches:
                break
            # Reset every batch so separate checkpoint runs see the same noise
            # and shortcut branch, regardless of checkpoint loading RNG use.
            seed = args.seed + index
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            count = int(batch["vlm_latents"].shape[0])
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type,
                                dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                                enabled=device.type == "cuda" and precision != "no"):
                loss, parts, *_ = model(**batch, train_decoder=False,
                                       training_mode="world_model", update_loss_ema=False)
            metrics = {"loss": float(loss)}
            for name, value in zip(parts._fields, parts):
                if value is not None:
                    metrics[name] = float(value)
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite validation batch {index}: {metrics}")
            row = {"batch": index, "samples": count, "seed": seed, **metrics}
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            samples += count
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + count * value
            print(f"[{index+1}/{len(loader)}] samples={samples} loss={metrics['loss']:.6f}", flush=True)
    if not rows:
        raise ValueError("No validation batches")
    result = {
        "status": "complete", "protocol": "fixed_seed_validation_objective_v1",
        "checkpoint": str(args.checkpoint.resolve()), "cache": str(args.cache.resolve()),
        "seed": args.seed, "rng": "Python, NumPy, Torch reset to seed+batch_index",
        "batch_size": args.batch_size, "precision": precision,
        "dataset_options": options, "available_windows": len(dataset),
        "evaluated_windows": samples, "full_validation": samples == len(dataset),
        "num_episodes": len(dataset.records), "num_batches": len(rows),
        "prob_no_shortcut": model.prob_no_shortcut_train,
        "loss_normalization": {
            "flow_exp_avg_sq": model.flow_loss_normalizer.exp_avg_sq.detach().cpu().tolist(),
            "flow_rms": model.flow_loss_normalizer.exp_avg_sq.detach().sqrt().cpu().tolist(),
            "comparison_metric": "sample_weighted_mean.flow",
            "warning": "loss uses each checkpoint's EMA scale; compare raw flow across checkpoints",
        },
        "sample_weighted_mean": {key: value / samples for key, value in sums.items()},
        "batch_mean": {key: sum(row[key] for row in rows) / len(rows) for key in sums},
        "batches": rows,
        "note": "Stochastic denoising/shortcut objective; not open-loop prediction or task success. "
                "Use raw flow for cross-checkpoint comparisons; loss uses checkpoint-specific EMA scaling.",
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["sample_weighted_mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
