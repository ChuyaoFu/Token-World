# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
"""Train the symmetric Qwen S-VAE from selected RoboTwin cameras (default 108x16).

Frozen Qwen merger features are produced online and discarded after each
optimizer step.  No feature cache is written.  Episode-aware sharding keeps
video decoding sequential, while padding a very small tail makes every DDP
rank execute exactly the same number of optimizer steps.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from tokenworld.codec.common import (  # noqa: E402
    VIDEO_KEYS as ALL_VIDEO_KEYS,
    atomic_json_dump,
    atomic_torch_save,
    build_episode_records,
    distributed_info,
    instantiate_codec,
    resolve_video_paths,
    sha256_file,
)
from tokenworld.data.video import _decode_selected_frames  # noqa: E402
from tokenworld.codec.fidelity import FidelityMetrics  # noqa: E402


CHECKPOINT_VERSION = 1
NATIVE_TOKENS = 108
INPUT_DIM = 2560
GRID_SIZE = [9, 12]
# Each torchrun worker selects the same camera preset before planning or decoding.
CAMERA_PRESETS = {"cam_high": ("observation.images.cam_high",), "all": ALL_VIDEO_KEYS}
VIDEO_KEYS = CAMERA_PRESETS["cam_high"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", choices=tuple(CAMERA_PRESETS), default="cam_high",
                        help="Cameras used for both training and validation: cam_high or all three views.")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-GPU image batch size across selected cameras; global batch is this times WORLD_SIZE.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument(
        "--train-frames-per-episode",
        type=int,
        default=0,
        help="0 uses every frame from every train episode.",
    )
    parser.add_argument(
        "--val-frames-per-episode",
        type=int,
        default=8,
        help="Validation-only deterministic trajectory-wide samples; 0 uses all.",
    )
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    parser.add_argument("--expected-tasks", type=int, default=50)
    parser.add_argument("--expected-world-size", type=int, default=0)
    parser.add_argument("--require-all-episodes-in-split", action="store_true")
    parser.add_argument("--token-dim", type=int, default=16)
    parser.add_argument(
        "--latent-grid-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=GRID_SIZE,
        help="Spatial S-VAE bottleneck grid; native Qwen post-merger grid is 9x12.",
    )
    parser.add_argument("--projection-hidden-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--kl-weight", type=float, default=1.0e-6)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        default=5000,
        help="Write an exact optimizer/scheduler resume checkpoint; 0 disables it.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Keep a weights-only epoch_NNNN.pt checkpoint after every validation epoch.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _episode_metadata(dataset_dir: Path) -> dict[int, dict]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[int(row["episode_index"])] = row
    return result


def validate_manifest(
    data_root: Path,
    manifest_path: Path,
    expected_tasks: int,
    require_all_episodes: bool,
) -> dict:
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    datasets = manifest.get("datasets", {})
    if not datasets:
        raise ValueError(f"Manifest has no datasets: {manifest_path}")
    task_sets = []
    summary = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "datasets": {},
    }
    for dataset_name, entry in sorted(datasets.items()):
        metadata = _episode_metadata(data_root / dataset_name)
        tasks = entry.get("tasks", {})
        task_sets.append(set(tasks))
        seen: set[int] = set()
        train_frames = 0
        val_frames = 0
        train_episodes = 0
        val_episodes = 0
        train_counts = []
        val_counts = []
        for task_name, task in sorted(tasks.items()):
            train_ids = list(map(int, task.get("train", [])))
            val_ids = list(map(int, task.get("val", [])))
            unused_ids = list(map(int, task.get("unused", [])))
            if not train_ids or not val_ids:
                raise ValueError(f"{dataset_name}/{task_name} needs nonempty train and val")
            if set(train_ids) & set(val_ids):
                raise ValueError(f"Train/val overlap in {dataset_name}/{task_name}")
            if require_all_episodes and unused_ids:
                raise ValueError(
                    f"All-data run requires unused=[]; {dataset_name}/{task_name} "
                    f"has {len(unused_ids)} unused episodes"
                )
            for split, episode_ids in (("train", train_ids), ("val", val_ids)):
                for episode_id in episode_ids:
                    if episode_id in seen:
                        raise ValueError(
                            f"Duplicate episode {episode_id} in {dataset_name} manifest"
                        )
                    seen.add(episode_id)
                    if episode_id not in metadata:
                        raise KeyError(f"Missing {dataset_name} episode {episode_id}")
                    episode_tasks = metadata[episode_id].get("tasks", [])
                    if episode_tasks and task_name not in episode_tasks:
                        raise ValueError(
                            f"{dataset_name} episode {episode_id} belongs to "
                            f"{episode_tasks}, not {task_name}"
                        )
                    length = int(metadata[episode_id]["length"])
                    if split == "train":
                        train_frames += length
                    else:
                        val_frames += length
            train_episodes += len(train_ids)
            val_episodes += len(val_ids)
            train_counts.append(len(train_ids))
            val_counts.append(len(val_ids))
        if require_all_episodes and seen != set(metadata):
            missing = sorted(set(metadata) - seen)
            extra = sorted(seen - set(metadata))
            raise ValueError(
                f"{dataset_name} manifest does not cover every episode: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )
        summary["datasets"][dataset_name] = {
            "outcome": entry.get("outcome", "unknown"),
            "tasks": len(tasks),
            "train_episodes": train_episodes,
            "val_episodes": val_episodes,
            "train_episode_count_range_per_task": [min(train_counts), max(train_counts)],
            "val_episode_count_range_per_task": [min(val_counts), max(val_counts)],
            "train_frames": train_frames,
            "val_frames": val_frames,
            "available_episodes": len(metadata),
        }
    if any(tasks != task_sets[0] for tasks in task_sets[1:]):
        raise ValueError("Dataset task sets differ in the manifest")
    task_count = len(task_sets[0])
    if task_count != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, found {task_count}")
    summary["task_count"] = task_count
    summary["uses_all_non_validation_episodes_for_train"] = bool(
        manifest.get("uses_all_non_validation_episodes_for_train", False)
    )
    return summary


def _record_views(record: dict) -> int:
    return len(record["frame_indices"]) * len(VIDEO_KEYS)


def balanced_assignments(
    records: Sequence[dict], world_size: int, seed: int
) -> tuple[list[list[dict]], list[int]]:
    """Greedily balance whole episodes so video files are decoded only once."""
    rng = random.Random(seed)
    decorated = [(rng.random(), record) for record in records]
    decorated.sort(key=lambda item: (-_record_views(item[1]), item[0]))
    bins: list[list[dict]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    for _, record in decorated:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        bins[rank].append(record)
        loads[rank] += _record_views(record)
    for rank, values in enumerate(bins):
        random.Random(seed + rank * 100003).shuffle(values)
    return bins, loads


def iter_image_batches(
    data_root: Path,
    records: Sequence[dict],
    batch_size: int,
    image_size: tuple[int, int],
    start_view: int,
    total_views: int,
) -> Iterator[list]:
    """Yield a deterministic view stream, cycling only for DDP tail padding."""
    if not records:
        raise ValueError("Cannot stream an empty record list")
    if start_view < 0 or total_views < start_view:
        raise ValueError(f"Invalid stream range {start_view}:{total_views}")
    pending = []
    cursor = 0
    record_index = 0
    while cursor < total_views:
        record = records[record_index % len(records)]
        record_index += 1
        frame_indices = list(map(int, record["frame_indices"]))
        full_views = len(frame_indices) * len(VIDEO_KEYS)
        visible_views = min(full_views, total_views - cursor)
        if cursor + visible_views <= start_view:
            cursor += full_views
            continue
        local_start = max(start_view - cursor, 0)
        local_end = visible_views
        first_frame = local_start // len(VIDEO_KEYS)
        final_frame = int(math.ceil(local_end / len(VIDEO_KEYS)))
        selected_indices = frame_indices[first_frame:final_frame]
        decoded = _decode_selected_frames(
            resolve_video_paths(data_root, record, video_keys=VIDEO_KEYS), selected_indices, image_size
        )
        flat_images = [image for frame in decoded for image in frame]
        first_view = local_start - first_frame * len(VIDEO_KEYS)
        needed = local_end - local_start
        flat_images = flat_images[first_view : first_view + needed]
        for image in flat_images:
            pending.append(image)
            if len(pending) == batch_size:
                yield pending
                pending = []
        cursor += full_views
    if pending:
        yield pending


class FrozenQwenVision:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        full_model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        self.visual = full_model.model.visual.to(device).eval().requires_grad_(False)
        del full_model
        gc.collect()

    @torch.inference_mode()
    def __call__(self, images: list) -> torch.Tensor:
        encoded = self.processor.image_processor(images=images, return_tensors="pt")
        grid = encoded["image_grid_thw"]
        merge = int(self.visual.spatial_merge_size)
        native_counts = [int(row.prod().item()) // (merge**2) for row in grid]
        if any(value != NATIVE_TOKENS for value in native_counts):
            raise ValueError(
                f"Expected {NATIVE_TOKENS} post-merger tokens at 384x288, "
                f"got {native_counts} from grid={grid.tolist()}"
            )
        pixel_values = encoded["pixel_values"].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=True
        )
        grid = grid.to(device=self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            merged, _ = self.visual(pixel_values, grid_thw=grid)
        features = torch.stack(torch.split(merged, native_counts), dim=0)
        if tuple(features.shape[1:]) != (NATIVE_TOKENS, INPUT_DIM):
            raise ValueError(f"Unexpected Qwen feature shape {tuple(features.shape)}")
        return features.detach()


class CodecObjective(nn.Module):
    def __init__(self, codec: nn.Module) -> None:
        super().__init__()
        self.codec = codec

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        _, _, losses = self.codec.forward_with_auxiliary_losses(
            list(features.detach().unbind(0))
        )
        return losses


def codec_config(args: argparse.Namespace) -> dict:
    latent_grid = list(map(int, args.latent_grid_size))
    if min(latent_grid) <= 0:
        raise ValueError(f"latent_grid_size must be positive, got {latent_grid}")
    if any(latent > native for latent, native in zip(latent_grid, GRID_SIZE)):
        raise ValueError(
            f"latent_grid_size={latent_grid} cannot exceed native grid {GRID_SIZE}"
        )
    return {
        "method": "symmetric_qwen",
        "input_dim": INPUT_DIM,
        "native_tokens": NATIVE_TOKENS,
        "tokens_per_view": math.prod(latent_grid),
        "token_dim": int(args.token_dim),
        "model_dim": 384,
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "grid_size": GRID_SIZE,
        "latent_grid_size": latent_grid,
        "projection_hidden_dim": int(args.projection_hidden_dim),
        "kl_weight": float(args.kl_weight),
        "pixel_output_size": [96, 128],
    }


def run_config(
    args: argparse.Namespace,
    world_size: int,
    manifest_summary: dict,
    steps_per_epoch: Sequence[int],
) -> dict:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "data_root": str(args.data_root.resolve()),
        "manifest_sha256": manifest_summary["manifest_sha256"],
        "world_size": world_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size_per_gpu": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "image_size": [args.image_height, args.image_width],
        "video_keys": list(VIDEO_KEYS),
        "train_frames_per_episode": args.train_frames_per_episode,
        "val_frames_per_episode": args.val_frames_per_episode,
        "max_episodes_per_split": args.max_episodes_per_split,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "codec_config": codec_config(args),
        "steps_per_epoch": list(map(int, steps_per_epoch)),
        "total_steps": int(sum(steps_per_epoch)),
        "teacher": "frozen Qwen3-VL post-merger features extracted online",
        "feature_cache": "none",
        "posterior": "sample_for_training_mean_for_validation_and_export",
    }


def _weights_payload(
    codec: nn.Module,
    config: dict,
    run: dict,
    manifest_summary: dict,
    epoch: int,
    global_step: int,
    validation: dict,
) -> dict:
    return {
        "version": CHECKPOINT_VERSION,
        "codec_config": config,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in codec.state_dict().items()
        },
        "training": {
            **run,
            "epoch": epoch,
            "global_step": global_step,
        },
        "data_summary": manifest_summary,
        "validation": validation,
    }


def _save_resume(
    path: Path,
    codec: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run: dict,
    manifest_summary: dict,
    next_epoch: int,
    next_step_in_epoch: int,
    global_step: int,
    best_nmse: float,
    best_epoch: int | None,
    history: list[dict],
) -> None:
    payload = _weights_payload(
        codec,
        run["codec_config"],
        run,
        manifest_summary,
        max(next_epoch - (1 if next_step_in_epoch == 0 else 0), 0),
        global_step,
        history[-1] if history else {},
    )
    payload.update(
        {
            "resume": {
                "run_config_sha256": _json_sha256(run),
                "next_epoch": next_epoch,
                "next_step_in_epoch": next_step_in_epoch,
                "best_nmse": best_nmse,
                "best_epoch": best_epoch,
                "history": history,
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
    )
    atomic_torch_save(payload, path)


def _load_resume(
    path: Path,
    codec: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    run: dict,
) -> tuple[int, int, int, float, int | None, list[dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    resume = payload.get("resume") or {}
    if resume.get("run_config_sha256") != _json_sha256(run):
        raise ValueError(
            f"Resume configuration mismatch in {path}; use a new output directory"
        )
    codec.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    return (
        int(resume["next_epoch"]),
        int(resume["next_step_in_epoch"]),
        int(payload["training"]["global_step"]),
        float(resume["best_nmse"]),
        int(resume["best_epoch"]) if resume.get("best_epoch") is not None else None,
        list(resume.get("history", [])),
    )


@torch.inference_mode()
def evaluate(
    codec: nn.Module,
    teacher: FrozenQwenVision,
    records: Sequence[dict],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    codec.eval()
    metrics = FidelityMetrics(INPUT_DIM, args.token_dim, device)
    natural_views = sum(_record_views(record) for record in records)
    for images in iter_image_batches(
        args.data_root,
        records,
        args.eval_batch_size,
        (args.image_height, args.image_width),
        0,
        natural_views,
    ):
        features = teacher(images)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, mean, logvar = codec.encode_with_posterior(
                list(features.unbind(0)), sample_posterior=False
            )
            reconstruction = torch.stack(
                codec.decode(mean, [NATIVE_TOKENS] * len(features)), dim=0
            )
        metrics.update(
            features,
            reconstruction,
            tokens,
            posterior_mean=mean,
            posterior_logvar=logvar,
        )
    result = metrics.compute()
    codec.train()
    return result


def _replace_with_hardlink_or_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    global VIDEO_KEYS
    args = parse_args()
    VIDEO_KEYS = CAMERA_PRESETS[args.views]
    args.model = args.model.resolve()
    args.data_root = args.data_root.resolve()
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    manifest_summary = validate_manifest(
        args.data_root,
        args.manifest,
        args.expected_tasks,
        args.require_all_episodes_in_split,
    )
    records = build_episode_records(
        args.data_root,
        args.manifest,
        args.train_frames_per_episode,
        args.val_frames_per_episode,
        args.max_episodes_per_split,
    )
    train_records = [record for record in records if record["split"] == "train"]
    val_records = [record for record in records if record["split"] == "val"]
    rank, world_size, local_rank = distributed_info()
    if args.expected_world_size and world_size != args.expected_world_size:
        raise ValueError(
            f"Expected WORLD_SIZE={args.expected_world_size}, got {world_size}"
        )
    epoch_plans = []
    steps_per_epoch = []
    plan_summaries = []
    for epoch in range(1, args.epochs + 1):
        bins, loads = balanced_assignments(
            train_records, world_size, args.seed + epoch * 1000003
        )
        target_views = int(math.ceil(max(loads) / args.batch_size) * args.batch_size)
        steps = target_views // args.batch_size
        epoch_plans.append(bins)
        steps_per_epoch.append(steps)
        plan_summaries.append(
            {
                "epoch": epoch,
                "unique_views_per_rank": loads,
                "target_views_per_rank": target_views,
                "padding_views_per_rank": [target_views - value for value in loads],
                "steps_per_rank": steps,
            }
        )
    run = run_config(args, world_size, manifest_summary, steps_per_epoch)
    plan = {
        "manifest_summary": manifest_summary,
        "train_records": len(train_records),
        "validation_records": len(val_records),
        "train_selected_frames": sum(len(record["frame_indices"]) for record in train_records),
        "validation_selected_frames": sum(len(record["frame_indices"]) for record in val_records),
        "epochs": plan_summaries,
        "run": run,
    }
    if rank == 0:
        print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Streaming S-VAE training requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(plan, args.output / "training_plan.json")
    if dist.is_initialized():
        dist.barrier()

    codec = instantiate_codec(run["codec_config"]).to(device)
    objective = CodecObjective(codec).to(device)
    distributed_model: nn.Module = objective
    if world_size > 1:
        distributed_model = DistributedDataParallel(
            objective,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
    optimizer = torch.optim.AdamW(
        objective.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(run["total_steps"], 1)
    )
    resume_path = args.resume
    default_resume = args.output / "resume.pt"
    if resume_path is None and not args.no_auto_resume and default_resume.is_file():
        resume_path = default_resume
    start_epoch = 1
    start_step_in_epoch = 0
    global_step = 0
    best_nmse = float("inf")
    best_epoch = None
    history: list[dict] = []
    if resume_path is not None:
        (
            start_epoch,
            start_step_in_epoch,
            global_step,
            best_nmse,
            best_epoch,
            history,
        ) = _load_resume(resume_path, codec, optimizer, scheduler, run)
        if rank == 0:
            print(
                f"Resuming {resume_path}: epoch={start_epoch} "
                f"step_in_epoch={start_step_in_epoch} global_step={global_step}",
                flush=True,
            )

    teacher = FrozenQwenVision(args.model, device)
    if dist.is_initialized():
        dist.barrier()
    started = time.perf_counter()
    last_validation: dict = history[-1] if history else {}
    for epoch in range(start_epoch, args.epochs + 1):
        objective.train()
        local_records = epoch_plans[epoch - 1][rank]
        steps_this_epoch = steps_per_epoch[epoch - 1]
        first_step = start_step_in_epoch if epoch == start_epoch else 0
        start_view = first_step * args.batch_size
        target_views = steps_this_epoch * args.batch_size
        batches = iter_image_batches(
            args.data_root,
            local_records,
            args.batch_size,
            (args.image_height, args.image_width),
            start_view,
            target_views,
        )
        for step_in_epoch in range(first_step, steps_this_epoch):
            images = next(batches)
            if len(images) != args.batch_size:
                raise RuntimeError(
                    f"Training batch has {len(images)} images, expected {args.batch_size}"
                )
            features = teacher(images)
            optimizer.zero_grad(set_to_none=True)
            # Step-addressed posterior noise makes an interrupted run exactly resumable.
            torch.cuda.manual_seed(
                args.seed + epoch * 1_000_000_007 + global_step * world_size + rank
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses = distributed_model(features)
                loss = losses["vision_codec_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(objective.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1
            next_step = step_in_epoch + 1
            if global_step == 1 or global_step % args.log_every == 0:
                values = torch.stack(
                    [
                        loss.detach().float(),
                        losses["vision_semantic_mse_loss"].detach().float(),
                        losses["vision_semantic_cosine_loss"].detach().float(),
                        losses["vision_kl_loss"].detach().float(),
                    ]
                )
                if dist.is_initialized():
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    values /= world_size
                if rank == 0:
                    elapsed = time.perf_counter() - started
                    rate = global_step / max(elapsed, 1.0e-9)
                    remaining = (run["total_steps"] - global_step) / max(rate, 1.0e-9)
                    print(
                        f"step={global_step}/{run['total_steps']} epoch={epoch} "
                        f"step_in_epoch={next_step}/{steps_this_epoch} "
                        f"loss={float(values[0]):.6f} mse={float(values[1]):.6f} "
                        f"cosine={float(values[2]):.6f} kl={float(values[3]):.3f} "
                        f"lr={scheduler.get_last_lr()[0]:.3e} "
                        f"steps_per_second={rate:.4f} eta_hours={remaining / 3600:.2f}",
                        flush=True,
                    )
            should_save = (
                args.save_every_steps > 0
                and global_step % args.save_every_steps == 0
                and next_step < steps_this_epoch
            )
            if should_save:
                if dist.is_initialized():
                    dist.barrier()
                if rank == 0:
                    print(f"Saving resumable state at step {global_step}", flush=True)
                    _save_resume(
                        default_resume,
                        codec,
                        optimizer,
                        scheduler,
                        run,
                        manifest_summary,
                        epoch,
                        next_step,
                        global_step,
                        best_nmse,
                        best_epoch,
                        history,
                    )
                if dist.is_initialized():
                    dist.barrier()
        del batches

        val_bins, _ = balanced_assignments(
            val_records, world_size, args.seed + 900_000_001
        )
        validation = evaluate(codec, teacher, val_bins[rank], args, device)
        validation.update(
            {
                "epoch": epoch,
                "global_step": global_step,
                "selected_frames": sum(
                    len(record["frame_indices"]) for record in val_records
                ),
                "selected_views": sum(_record_views(record) for record in val_records),
            }
        )
        history.append(validation)
        last_validation = validation
        improved = validation["feature_nmse"] < best_nmse
        if improved:
            best_nmse = float(validation["feature_nmse"])
            best_epoch = epoch
        if rank == 0:
            print("validation=" + json.dumps(validation, sort_keys=True), flush=True)
            if improved:
                atomic_torch_save(
                    _weights_payload(
                        codec,
                        run["codec_config"],
                        run,
                        manifest_summary,
                        epoch,
                        global_step,
                        validation,
                    ),
                    args.output / "best.pt",
                )
            if args.save_every_epoch:
                atomic_torch_save(
                    _weights_payload(
                        codec,
                        run["codec_config"],
                        run,
                        manifest_summary,
                        epoch,
                        global_step,
                        validation,
                    ),
                    args.output / f"epoch_{epoch:04d}.pt",
                )
        if dist.is_initialized():
            dist.barrier()
        if epoch < args.epochs:
            if rank == 0:
                _save_resume(
                    default_resume,
                    codec,
                    optimizer,
                    scheduler,
                    run,
                    manifest_summary,
                    epoch + 1,
                    0,
                    global_step,
                    best_nmse,
                    best_epoch,
                    history,
                )
            if dist.is_initialized():
                dist.barrier()

    if rank == 0:
        best_path = args.output / "best.pt"
        final_path = args.output / "final.pt"
        if best_epoch == args.epochs:
            _replace_with_hardlink_or_copy(best_path, final_path)
        else:
            atomic_torch_save(
                _weights_payload(
                    codec,
                    run["codec_config"],
                    run,
                    manifest_summary,
                    args.epochs,
                    global_step,
                    last_validation,
                ),
                final_path,
            )
        _replace_with_hardlink_or_copy(final_path, args.output / "last.pt")
        default_resume.unlink(missing_ok=True)
        metrics = {
            "run": run,
            "data_summary": manifest_summary,
            "history": history,
            "best_epoch": best_epoch,
            "best_validation": next(
                value for value in history if value["epoch"] == best_epoch
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoints": {
                "best": str(best_path),
                "best_sha256": sha256_file(best_path),
                "final": str(final_path),
                "final_sha256": sha256_file(final_path),
            },
        }
        atomic_json_dump(metrics, args.output / "metrics.json")
        print(json.dumps(metrics, indent=2), flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
