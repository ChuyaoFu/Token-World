# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
"""Export a 50-task symmetric-Qwen S-VAE sequence cache for VLM dynamics.

The historical filename is retained so existing launch references do not
break.  Formal exports use the ``symmetric_qwen`` posterior mean produced by
the all-task S-VAE; old E2 caches remain readable by the dynamics trainer.
"""

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

from tokenworld.codec.common import (  # noqa: E402
    VIDEO_KEYS,
    build_episode_records,
    instantiate_codec,
    resolve_video_paths,
)
from tokenworld.data.cache_utils import (  # noqa: E402
    _decode_selected_frames,
    _episode_data_path,
    _finish_stats,
    _running_stats_update,
)
from tokenworld.codec.train import FrozenQwenVision  # noqa: E402
from tokenworld.utils.io import (  # noqa: E402
    SEQUENCE_CACHE_VERSION,
    atomic_json_dump,
    atomic_torch_save,
    distributed_info,
    sha256_file,
)


DEFAULT_LATENT_METHOD = "symmetric_qwen"


def _cache_path(cache_root: Path, record: dict) -> Path:
    return (
        cache_root
        / record["split"]
        / record["dataset"]
        / f"episode_{int(record['episode_index']):06d}.pt"
    )


@torch.inference_mode()
def _encode_symmetric_frames(
    teacher: FrozenQwenVision,
    codec: torch.nn.Module,
    frames: list[list],
    image_batch_size: int,
) -> torch.Tensor:
    flat_images = [image for frame in frames for image in frame]
    chunks = []
    for start in range(0, len(flat_images), image_batch_size):
        features = teacher(flat_images[start : start + image_batch_size])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, mean, _ = codec.encode_with_posterior(
                list(features.unbind(0)), sample_posterior=False
            )
        chunks.append(mean.detach().to(device="cpu", dtype=torch.bfloat16))
    if not chunks:
        raise RuntimeError("No frames were encoded")
    flat = torch.cat(chunks, dim=0)
    return flat.reshape(len(frames), len(VIDEO_KEYS), *flat.shape[1:])


def _plan(
    args: argparse.Namespace,
    records: list[dict],
    latent_method: str,
    codec_config: dict,
) -> dict:
    return {
        "version": SEQUENCE_CACHE_VERSION,
        "cache_variant": "single_latent_vlm_world_model",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "image_size": [args.image_height, args.image_width],
        "codec_encode_image_batch_size": args.image_batch_size,
        "video_keys": list(VIDEO_KEYS),
        "latent_method": latent_method,
        "codec_config": codec_config,
        "codec_checkpoints": {
            latent_method: str(args.codec_checkpoint.resolve())
        },
        "codec_checkpoint_sha256": {
            latent_method: sha256_file(args.codec_checkpoint)
        },
        "source_frame_rate_hz": 50,
        "export_frame_stride": int(args.export_frame_stride),
        "export_frame_rate_hz": 50.0 / int(args.export_frame_stride),
        "records": records,
    }


def _finalize(cache_root: Path) -> None:
    with open(cache_root / "cache_plan.json", "r") as handle:
        plan = json.load(handle)
    task_names = sorted({row["task"] for row in plan["records"]})
    task_to_id = {task: index for index, task in enumerate(task_names)}
    latent_method = str(plan["latent_method"])
    accumulators = {"action": {}, "state": {}, latent_method: {}}
    video_keys = list(plan["video_keys"])
    latent_by_view_accumulators = {key: {} for key in video_keys}
    records = []
    total_frames = {"train": 0, "val": 0}
    latent_spec = None
    missing = []
    for row in plan["records"]:
        path = _cache_path(cache_root, row)
        if not path.is_file():
            missing.append(str(path))
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload.get("version", -1)) != SEQUENCE_CACHE_VERSION:
            raise ValueError(f"Bad payload version: {path}")
        frames = int(payload["states"].shape[0])
        expected_frames = len(
            range(0, int(row["episode_length"]), int(plan["export_frame_stride"]))
        )
        if frames != expected_frames:
            raise ValueError(
                f"Frame mismatch in {path}: got {frames}, expected {expected_frames}"
            )
        if not torch.equal(payload["actions"].float(), payload["states"].float()):
            raise ValueError(f"action != observation.state in {path}")
        latents = payload["latents"][latent_method]
        current_spec = {
            "shape_per_frame": list(latents.shape[1:]),
            "latent_count": 1,
            "latent_dim": int(latents.shape[-1]),
        }
        latent_spec = latent_spec or current_spec
        if current_spec != latent_spec:
            raise ValueError(f"Latent shape mismatch in {path}")
        if row["split"] == "train":
            _running_stats_update(accumulators["action"], payload["actions"])
            _running_stats_update(accumulators["state"], payload["states"])
            _running_stats_update(accumulators[latent_method], latents)
            if int(latents.shape[1]) != len(video_keys):
                raise ValueError(
                    f"Latent view count does not match video_keys in {path}: "
                    f"{int(latents.shape[1])} != {len(video_keys)}"
                )
            for view_index, view_key in enumerate(video_keys):
                _running_stats_update(
                    latent_by_view_accumulators[view_key], latents[:, view_index]
                )
        total_frames[row["split"]] += frames
        records.append(
            {
                **{
                    key: row[key]
                    for key in (
                        "dataset",
                        "outcome",
                        "task",
                        "split",
                        "episode_index",
                    )
                },
                "task_id": task_to_id[row["task"]],
                "num_frames": frames,
                "num_transitions": max(frames - 1, 0),
                "path": str(path.relative_to(cache_root)),
            }
        )
    if missing:
        raise RuntimeError(f"Sequence cache incomplete: {len(missing)} missing shards")
    if latent_spec is None:
        raise RuntimeError("Sequence cache contains no shards")
    stats = {
        "version": SEQUENCE_CACHE_VERSION,
        "action": _finish_stats(accumulators["action"]),
        "state": _finish_stats(accumulators["state"]),
        latent_method: _finish_stats(accumulators[latent_method]),
        "latent_by_view": {
            latent_method: {
                view_key: _finish_stats(view_accumulator)
                for view_key, view_accumulator in latent_by_view_accumulators.items()
            }
        },
    }
    atomic_torch_save(stats, cache_root / "train_stats.pt")
    spec = {
        key: plan[key]
        for key in (
            "version",
            "cache_variant",
            "model",
            "model_config_sha256",
            "data_root",
            "manifest",
            "manifest_sha256",
            "image_size",
            "codec_encode_image_batch_size",
            "video_keys",
            "latent_method",
            "codec_config",
            "codec_checkpoints",
            "codec_checkpoint_sha256",
            "source_frame_rate_hz",
            "export_frame_stride",
            "export_frame_rate_hz",
        )
    }
    spec.update(
        {
            "task_names": task_names,
            "task_to_id": task_to_id,
            "latent_specs": {latent_method: latent_spec},
            "export_dtype": "bfloat16",
            "alignment": (
                "vision_latent[t] and observation.state[t] share the same frame index; "
                "state is a condition and is never a prediction target"
            ),
            "action_state_identity_verified": True,
        }
    )
    atomic_json_dump(spec, cache_root / "cache_spec.json")
    atomic_json_dump(
        {
            "version": SEQUENCE_CACHE_VERSION,
            "totals": {
                split: {
                    "episodes": sum(row["split"] == split for row in records),
                    "frames": total_frames[split],
                    "transitions": sum(
                        row["num_transitions"] for row in records if row["split"] == split
                    ),
                }
                for split in ("train", "val")
            },
            "records": records,
        },
        cache_root / "index.json",
    )
    print(
        json.dumps(
            {
                "cache": str(cache_root),
                "tasks": len(task_names),
                "frames": total_frames,
                "latent_spec": latent_spec,
            },
            indent=2,
        )
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--codec-checkpoint",
        "--e2-checkpoint",
        dest="codec_checkpoint",
        type=Path,
        default=None,
        help="Final symmetric_qwen S-VAE checkpoint; --e2-checkpoint is a deprecated alias.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--expected-task-count", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-batch-size", type=int, default=12)
    parser.add_argument(
        "--export-frame-stride",
        type=int,
        default=1,
        help="1 writes the canonical 50-Hz cache; 5 writes a compact 10-Hz debug cache.",
    )
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.finalize_only:
        _finalize(args.output)
        return
    if args.manifest is None:
        raise ValueError(
            "--manifest is required for export; pass a separately generated 50-task manifest"
        )
    if args.codec_checkpoint is None:
        raise ValueError("--codec-checkpoint is required for export")
    if args.export_frame_stride < 1:
        raise ValueError("--export-frame-stride must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("E2 sequence-cache export requires CUDA")
    if args.image_batch_size % len(VIDEO_KEYS):
        raise ValueError("--image-batch-size must be divisible by the number of camera views")
    rank, world_size, local_rank = distributed_info()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    records = build_episode_records(
        args.data_root,
        args.manifest,
        train_frames_per_episode=0,
        val_frames_per_episode=0,
        max_episodes_per_split=args.max_episodes_per_split,
    )
    observed_tasks = sorted({record["task"] for record in records})
    if args.expected_task_count > 0 and len(observed_tasks) != args.expected_task_count:
        raise ValueError(
            f"Manifest contains {len(observed_tasks)} tasks, expected "
            f"{args.expected_task_count}: {args.manifest}"
        )
    checkpoint_payload = torch.load(
        args.codec_checkpoint, map_location="cpu", weights_only=True
    )
    codec_config = dict(checkpoint_payload["codec_config"])
    latent_method = str(codec_config["method"])
    if latent_method != DEFAULT_LATENT_METHOD:
        raise ValueError(
            f"Formal ACWM export requires {DEFAULT_LATENT_METHOD}, got {latent_method}"
        )
    latent_grid = list(
        map(int, codec_config.get("latent_grid_size", codec_config["grid_size"]))
    )
    if math.prod(latent_grid) != int(codec_config["tokens_per_view"]):
        raise ValueError(
            f"Codec latent grid {latent_grid} does not match tokens_per_view="
            f"{codec_config['tokens_per_view']}"
        )
    plan = _plan(args, records, latent_method, codec_config)
    if rank == 0:
        plan_path = args.output / "cache_plan.json"
        if plan_path.is_file() and not args.overwrite:
            with open(plan_path, "r") as handle:
                existing = json.load(handle)
            if existing != plan:
                raise ValueError("Existing cache plan differs; use a new output path")
        atomic_json_dump(plan, plan_path)
    assigned = records[rank::world_size]
    pending = [
        row for row in assigned if args.overwrite or not _cache_path(args.output, row).is_file()
    ]
    print(f"[rank {rank}/{world_size}] assigned={len(assigned)} pending={len(pending)}", flush=True)
    if not pending:
        return
    codec = instantiate_codec(codec_config)
    codec.load_state_dict(checkpoint_payload["state_dict"], strict=True)
    codec = codec.to(device).eval()
    del checkpoint_payload
    teacher = FrozenQwenVision(args.model, device)
    image_size = (args.image_height, args.image_width)
    for number, record in enumerate(pending, start=1):
        started = time.perf_counter()
        data = pd.read_parquet(_episode_data_path(args.data_root, record))
        actions = torch.from_numpy(np.stack(data["action"].to_numpy()).astype("float32"))
        states = torch.from_numpy(
            np.stack(data["observation.state"].to_numpy()).astype("float32")
        )
        if not torch.equal(actions, states):
            raise ValueError(
                f"action != observation.state for {record['dataset']}:{record['episode_index']}"
            )
        timestamps = torch.as_tensor(data["timestamp"].to_numpy(), dtype=torch.float32)
        frame_indices = list(range(0, len(actions), args.export_frame_stride))
        actions = actions[frame_indices]
        states = states[frame_indices]
        timestamps = timestamps[frame_indices]
        frames = _decode_selected_frames(
            resolve_video_paths(args.data_root, record), frame_indices, image_size
        )
        latents = _encode_symmetric_frames(
            teacher,
            codec,
            frames,
            args.image_batch_size,
        )
        payload = {
            "version": SEQUENCE_CACHE_VERSION,
            "dataset": record["dataset"],
            "outcome": record["outcome"],
            "task": record["task"],
            "split": record["split"],
            "episode_index": int(record["episode_index"]),
            "frame_indices": frame_indices,
            "timestamps": timestamps,
            "actions": actions,
            "states": states,
            "latents": {latent_method: latents},
            "post_merger_grid_hw": list(map(int, codec_config["grid_size"])),
            "latent_grid_hw": list(
                map(
                    int,
                    codec_config.get("latent_grid_size", codec_config["grid_size"]),
                )
            ),
        }
        path = _cache_path(args.output, record)
        atomic_torch_save(payload, path)
        print(
            f"[rank {rank}] {number}/{len(pending)} {record['split']} "
            f"{record['dataset']}:{record['episode_index']} frames={len(actions)} "
            f"{latent_method}={tuple(latents.shape)} "
            f"seconds={time.perf_counter() - started:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
