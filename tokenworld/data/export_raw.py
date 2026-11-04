# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
"""Export uncompressed Qwen VLM features for the RobotWin TokenWorld ablation."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tokenworld.codec.common import VIDEO_KEYS, build_episode_records, resolve_video_paths  # noqa: E402
from tokenworld.data.cache_utils import (  # noqa: E402
    _decode_selected_frames,
    _episode_data_path,
    _finish_stats,
    _running_stats_update,
)
from tokenworld.utils.io import (  # noqa: E402
    SEQUENCE_CACHE_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    distributed_info,
    sha256_file,
)
from tokenworld.codec.train import FrozenQwenVision  # noqa: E402

RAW_METHOD = "qwen_raw"
RAW_VIDEO_KEYS = (VIDEO_KEYS[0],)
RAW_FEATURE_DIM = 2560
RAW_GRID = (9, 12)


def _cache_path(cache_root: Path, record: dict) -> Path:
    return (
        cache_root / record["split"] / record["dataset"]
        / f"episode_{int(record['episode_index']):06d}.pt"
    )


@torch.inference_mode()
def _encode_raw_frames(
    teacher: FrozenQwenVision,
    frames: list[list],
    image_batch_size: int,
) -> torch.Tensor:
    # The video decoder returns all RobotWin views. This ablation intentionally
    # retains only cam_high, matching the baseline TokenWorld configuration.
    left_images = [frame[0] for frame in frames]
    chunks = []
    for start in range(0, len(left_images), image_batch_size):
        features = teacher(left_images[start : start + image_batch_size])
        if tuple(features.shape[1:]) != (RAW_GRID[0] * RAW_GRID[1], RAW_FEATURE_DIM):
            raise ValueError(
                f"Unexpected raw Qwen feature shape {tuple(features.shape)}; "
                f"expected [B,{RAW_GRID[0] * RAW_GRID[1]},{RAW_FEATURE_DIM}]"
            )
        chunks.append(features.detach().cpu().to(torch.bfloat16))
    if not chunks:
        raise RuntimeError("No frames were encoded")
    return torch.cat(chunks, dim=0).reshape(
        len(frames), len(RAW_VIDEO_KEYS), RAW_GRID[0] * RAW_GRID[1], RAW_FEATURE_DIM
    )


def _plan(args: argparse.Namespace, records: list[dict]) -> dict:
    return {
        "version": SEQUENCE_CACHE_VERSION,
        "cache_variant": "robotwin_raw_qwen_vlm_tokenworld",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_size": [args.image_height, args.image_width],
        "qwen_feature_image_batch_size": args.image_batch_size,
        "video_keys": list(RAW_VIDEO_KEYS),
        "latent_method": RAW_METHOD,
        "feature_dim": RAW_FEATURE_DIM,
        "grid_size": list(RAW_GRID),
        "codec_config": {
            "method": RAW_METHOD,
            "input_dim": RAW_FEATURE_DIM,
            "tokens_per_view": RAW_GRID[0] * RAW_GRID[1],
            "grid_size": list(RAW_GRID),
            "latent_grid_size": list(RAW_GRID),
        },
        "records": records,
    }


def _finalize(cache_root: Path) -> None:
    plan = json.loads((cache_root / "cache_plan.json").read_text())
    task_names = sorted({row["task"] for row in plan["records"]})
    task_to_id = {task: index for index, task in enumerate(task_names)}
    accumulators = {"action": {}, "state": {}, RAW_METHOD: {}}
    latent_by_view_accumulators = {key: {} for key in RAW_VIDEO_KEYS}
    records = []
    totals = {"train": 0, "val": 0}
    latent_spec = None
    missing = []

    for row in plan["records"]:
        path = _cache_path(cache_root, row)
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        frames = int(payload["states"].shape[0])
        expected_frames = len(range(0, int(row["episode_length"]), int(plan["export_frame_stride"])) )
        if frames != expected_frames:
            raise ValueError(f"Frame mismatch in {path}: {frames} != {expected_frames}")
        if not torch.equal(payload["actions"].float(), payload["states"].float()):
            raise ValueError(f"action != observation.state in {path}")
        latents = payload["latents"][RAW_METHOD]
        current = {
            "shape_per_frame": list(latents.shape[1:]),
            "latent_count": 1,
            "latent_dim": int(latents.shape[-1]),
        }
        latent_spec = latent_spec or current
        if current != latent_spec:
            raise ValueError(f"Raw feature shape mismatch in {path}")
        if row["split"] == "train":
            _running_stats_update(accumulators["action"], payload["actions"])
            _running_stats_update(accumulators["state"], payload["states"])
            _running_stats_update(accumulators[RAW_METHOD], latents)
            for view_index, view_key in enumerate(RAW_VIDEO_KEYS):
                _running_stats_update(
                    latent_by_view_accumulators[view_key], latents[:, view_index]
                )
        totals[row["split"]] += frames
        records.append({
            **{key: row[key] for key in ("dataset", "outcome", "task", "split", "episode_index")},
            "task_id": task_to_id[row["task"]],
            "num_frames": frames,
            "num_transitions": max(frames - 1, 0),
            "path": str(path.relative_to(cache_root)),
        })

    if missing:
        raise RuntimeError(f"Raw feature cache incomplete: {len(missing)} missing shards")
    if latent_spec is None:
        raise RuntimeError("Raw feature cache contains no shards")

    stats = {
        "version": SEQUENCE_CACHE_VERSION,
        "action": _finish_stats(accumulators["action"]),
        "state": _finish_stats(accumulators["state"]),
        RAW_METHOD: _finish_stats(accumulators[RAW_METHOD]),
        "latent_by_view": {
            RAW_METHOD: {
                key: _finish_stats(value)
                for key, value in latent_by_view_accumulators.items()
            }
        },
    }
    atomic_torch_save(stats, cache_root / "train_stats.pt")
    spec = {
        **{key: plan[key] for key in (
            "version", "cache_variant", "model", "model_config_sha256",
            "data_root", "manifest", "manifest_sha256", "image_size",
            "qwen_feature_image_batch_size", "video_keys", "latent_method",
            "feature_dim", "grid_size", "codec_config",
        )},
        "task_names": task_names,
        "task_to_id": task_to_id,
        "latent_specs": {RAW_METHOD: latent_spec},
        "export_dtype": "bfloat16",
        "export_frame_stride": int(plan["export_frame_stride"]),
        "export_frame_rate_hz": 50.0 / int(plan["export_frame_stride"]),
        "source_frame_rate_hz": 50,
        "alignment": "raw Qwen feature[t] and observation.state[t] share the same frame index",
        "action_state_identity_verified": True,
    }
    atomic_json_dump(spec, cache_root / "cache_spec.json")
    atomic_json_dump({
        "version": SEQUENCE_CACHE_VERSION,
        "totals": {
            split: {
                "episodes": sum(row["split"] == split for row in records),
                "frames": totals[split],
                "transitions": sum(row["num_transitions"] for row in records if row["split"] == split),
            }
            for split in ("train", "val")
        },
        "records": records,
    }, cache_root / "index.json")
    print(json.dumps({"cache": str(cache_root), "latent_spec": latent_spec, "totals": totals}, indent=2))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--export-frame-stride", type=int, default=1)
    parser.add_argument("--expected-task-count", type=int, default=50)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument(
        "--max-records-per-split",
        type=int,
        default=0,
        help="Global smoke-test limit after manifest expansion; 0 keeps all records.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.finalize_only:
        _finalize(args.output)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Raw Qwen feature export requires CUDA")
    if args.export_frame_stride < 1:
        raise ValueError("--export-frame-stride must be positive")
    rank, world_size, local_rank = distributed_info()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    records = build_episode_records(
        args.data_root, args.manifest, train_frames_per_episode=0,
        val_frames_per_episode=0, max_episodes_per_split=args.max_episodes_per_split,
    )
    if args.max_records_per_split > 0:
        # Keep the deterministic first N records independently for each split.
        records = [
            row for split in ("train", "val")
            for row in [r for r in records if r["split"] == split][: args.max_records_per_split]
        ]
    tasks = sorted({row["task"] for row in records})
    if args.expected_task_count > 0 and len(tasks) != args.expected_task_count:
        raise ValueError(f"Manifest contains {len(tasks)} tasks, expected {args.expected_task_count}")
    plan = _plan(args, records)
    plan["export_frame_stride"] = int(args.export_frame_stride)
    if rank == 0:
        plan_path = args.output / "cache_plan.json"
        if plan_path.exists() and not args.overwrite:
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Existing raw cache plan differs; use a new output path or --overwrite")
        atomic_json_dump(plan, plan_path)
    assigned = records[rank::world_size]
    pending = [row for row in assigned if args.overwrite or not _cache_path(args.output, row).is_file()]
    print(f"[rank {rank}/{world_size}] assigned={len(assigned)} pending={len(pending)}", flush=True)
    if not pending:
        return
    teacher = FrozenQwenVision(args.model, device)
    for number, record in enumerate(pending, start=1):
        started = time.perf_counter()
        data = pd.read_parquet(_episode_data_path(args.data_root, record))
        actions = torch.from_numpy(np.stack(data["action"].to_numpy()).astype("float32"))
        states = torch.from_numpy(np.stack(data["observation.state"].to_numpy()).astype("float32"))
        if not torch.equal(actions, states):
            raise ValueError(f"action != observation.state for {record['dataset']}:{record['episode_index']}")
        timestamps = torch.as_tensor(data["timestamp"].to_numpy(), dtype=torch.float32)
        frame_indices = list(range(0, len(actions), args.export_frame_stride))
        actions, states, timestamps = actions[frame_indices], states[frame_indices], timestamps[frame_indices]
        frames_all = _decode_selected_frames(
            resolve_video_paths(args.data_root, record), frame_indices,
            (args.image_height, args.image_width),
        )
        frames = [[frame[0]] for frame in frames_all]
        latents = _encode_raw_frames(teacher, frames, args.image_batch_size)
        if tuple(latents.shape) != (len(actions), 1, 108, 2560) or not torch.isfinite(latents.float()).all():
            raise ValueError(f"Invalid raw feature tensor {tuple(latents.shape)}")
        payload = {
            "version": SEQUENCE_CACHE_VERSION,
            "dataset": record["dataset"], "outcome": record["outcome"],
            "task": record["task"], "split": record["split"],
            "episode_index": int(record["episode_index"]),
            "frame_indices": frame_indices, "timestamps": timestamps,
            "actions": actions, "states": states,
            "latents": {RAW_METHOD: latents},
            "latent_grid_hw": list(RAW_GRID),
        }
        atomic_torch_save(payload, _cache_path(args.output, record))
        print(
            f"[rank {rank}] {number}/{len(pending)} {record['split']} "
            f"{record['dataset']}:{record['episode_index']} frames={len(actions)} "
            f"{RAW_METHOD}={tuple(latents.shape)} seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
