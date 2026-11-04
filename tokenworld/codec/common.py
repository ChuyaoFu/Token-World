# Adapted for the TokenWorld RoboTwin release: local imports and portable entrypoints.
# Source attribution and original licenses are listed in THIRD_PARTY_NOTICES.md.
"""Shared utilities for the TokenWorld Qwen-codec fidelity experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from tokenworld.utils.io import atomic_json_dump, atomic_torch_save, distributed_info, sha256_file


CACHE_VERSION = 1
VIDEO_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
LEARNED_CODEC_METHODS = (
    "symmetric_qwen",
    "legacy_mlp",
    "deterministic_transformer",
    "psvae_no_pixel",
    "psvae",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def evenly_spaced_indices(length: int, requested: int) -> list[int]:
    """Return deterministic trajectory-wide samples including both endpoints."""
    if length <= 0:
        raise ValueError(f"Episode length must be positive, got {length}")
    if requested <= 0 or requested >= length:
        return list(range(length))
    raw = np.linspace(0, length - 1, num=requested)
    selected = sorted({int(round(value)) for value in raw})
    if len(selected) < requested:
        remaining = [index for index in range(length) if index not in set(selected)]
        selected.extend(remaining[: requested - len(selected)])
        selected.sort()
    return selected


def _episode_lengths(dataset_dir: Path) -> dict[int, int]:
    path = dataset_dir / "meta/episodes.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"LeRobot episode metadata is missing: {path}")
    lengths = {}
    with open(path, "r") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def build_episode_records(
    data_root: Path,
    manifest_path: Path,
    train_frames_per_episode: int,
    val_frames_per_episode: int,
    max_episodes_per_split: int = 0,
) -> list[dict]:
    """Resolve the immutable StarVLA split into deterministic cache records."""
    with open(manifest_path, "r") as handle:
        manifest = json.load(handle)
    records = []
    for dataset_name, dataset_entry in sorted(manifest["datasets"].items()):
        dataset_dir = data_root / dataset_name
        lengths = _episode_lengths(dataset_dir)
        seen: set[tuple[str, int]] = set()
        for task_name, task_entry in sorted(dataset_entry["tasks"].items()):
            for split in ("train", "val"):
                episode_ids = [int(value) for value in task_entry[split]]
                if max_episodes_per_split > 0:
                    episode_ids = episode_ids[:max_episodes_per_split]
                requested = train_frames_per_episode if split == "train" else val_frames_per_episode
                for episode_index in episode_ids:
                    key = (split, episode_index)
                    if key in seen:
                        raise ValueError(
                            f"Episode {episode_index} appears in multiple {split} tasks for {dataset_name}"
                        )
                    seen.add(key)
                    if episode_index not in lengths:
                        raise KeyError(f"Episode {episode_index} is absent from {dataset_dir}/meta/episodes.jsonl")
                    frame_indices = evenly_spaced_indices(lengths[episode_index], requested)
                    records.append(
                        {
                            "dataset": dataset_name,
                            "outcome": str(dataset_entry.get("outcome", "unknown")),
                            "task": task_name,
                            "split": split,
                            "episode_index": episode_index,
                            "episode_length": lengths[episode_index],
                            "frame_indices": frame_indices,
                            "num_frames": len(frame_indices),
                        }
                    )
    return records


def episode_cache_path(cache_root: Path, record: dict) -> Path:
    return (
        cache_root
        / record["split"]
        / record["dataset"]
        / f"episode_{int(record['episode_index']):06d}.pt"
    )


def resolve_video_paths(
    data_root: Path, record: dict, video_keys: Sequence[str] | None = None
) -> list[Path]:
    dataset_dir = data_root / record["dataset"]
    with open(dataset_dir / "meta/info.json", "r") as handle:
        info = json.load(handle)
    episode_index = int(record["episode_index"])
    fields = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // int(info["chunks_size"]),
    }
    paths = [
        dataset_dir / str(info["video_path"]).format(video_key=key, **fields)
        for key in (VIDEO_KEYS if video_keys is None else video_keys)
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Episode {episode_index} is missing videos: {missing}")
    return paths


def normalize_qwen_image_features(output, native_counts: Sequence[int]) -> list[torch.Tensor]:
    """Normalize Qwen3-VL image-feature returns across transformers releases."""
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (tuple, list)):
        features = list(output[0])
    elif hasattr(output, "pooler_output"):
        pooled = output.pooler_output
        features = list(pooled) if isinstance(pooled, (tuple, list)) else list(torch.split(pooled, native_counts))
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        if len(output) == len(native_counts) and all(torch.is_tensor(item) for item in output):
            features = list(output)
        else:
            features = list(torch.split(output[0], native_counts))
    elif torch.is_tensor(output):
        features = list(torch.split(output, native_counts))
    else:
        raise TypeError(f"Unsupported Qwen image-feature return type: {type(output)!r}")
    if len(features) != len(native_counts):
        raise ValueError(f"Expected {len(native_counts)} images, got {len(features)} feature tensors")
    for feature, expected in zip(features, native_counts):
        if feature.ndim != 2 or int(feature.shape[0]) != int(expected):
            raise ValueError(f"Expected image feature [{expected},D], got {tuple(feature.shape)}")
    return features


def load_cache_index(cache_root: Path) -> dict:
    path = cache_root / "index.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Teacher cache index is missing: {path}. Run extract_qwen_teacher_cache.py --finalize-only."
        )
    with open(path, "r") as handle:
        index = json.load(handle)
    if int(index.get("version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported teacher-cache version in {path}: {index.get('version')}")
    return index


def records_for_split(cache_root: Path, split: str) -> list[dict]:
    index = load_cache_index(cache_root)
    records = [record for record in index["records"] if record["split"] == split]
    if not records:
        raise ValueError(f"Teacher cache contains no {split!r} records")
    for record in records:
        record["absolute_path"] = str(cache_root / record["path"])
    return records


def build_codec(
    method: str,
    input_dim: int,
    native_tokens: int,
    tokens_per_view: int,
    token_dim: int,
    model_dim: int,
    depth: int,
    num_heads: int,
    pixel_output_size: Sequence[int],
    grid_size: Sequence[int] | None = None,
    latent_grid_size: Sequence[int] | None = None,
    projection_hidden_dim: int = 1024,
    kl_weight: float = 1.0e-6,
):
    from tokenworld.codec.shared_vision_token_codec import (
        PSVAESharedVisionTokenCodec,
        SymmetricQwenVisionTokenCodec,
        SharedVisionTokenCodec,
    )

    if method not in LEARNED_CODEC_METHODS:
        raise ValueError(f"Unknown codec method {method!r}; choose from {LEARNED_CODEC_METHODS}")
    if method == "legacy_mlp":
        return SharedVisionTokenCodec(
            input_dim=input_dim,
            token_dim=token_dim,
            tokens_per_view=tokens_per_view,
        )
    if method == "symmetric_qwen":
        if grid_size is None:
            raise ValueError("symmetric_qwen requires the fixed post-merger grid_size")
        if math.prod(map(int, grid_size)) != int(native_tokens):
            raise ValueError(
                f"grid_size={grid_size} does not contain native_tokens={native_tokens}"
            )
        if latent_grid_size is None:
            if int(tokens_per_view) != int(native_tokens):
                raise ValueError(
                    "A reduced symmetric_qwen bottleneck requires latent_grid_size; "
                    f"got tokens_per_view={tokens_per_view} and native_tokens={native_tokens}"
                )
            latent_grid_size = grid_size
        return SymmetricQwenVisionTokenCodec(
            input_dim=input_dim,
            token_dim=token_dim,
            tokens_per_view=tokens_per_view,
            grid_size=grid_size,
            latent_grid_size=latent_grid_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            projection_hidden_dim=projection_hidden_dim,
            sample_posterior=True,
            semantic_mse_weight=1.0,
            semantic_cosine_weight=1.0,
            kl_weight=kl_weight,
        )
    return PSVAESharedVisionTokenCodec(
        input_dim=input_dim,
        token_dim=token_dim,
        tokens_per_view=tokens_per_view,
        native_tokens_per_view=native_tokens,
        model_dim=model_dim,
        depth=depth,
        num_heads=num_heads,
        variational=method in {"psvae_no_pixel", "psvae"},
        sample_posterior=True,
        pixel_decoder=method == "psvae",
        pixel_output_size=tuple(map(int, pixel_output_size)),
        semantic_mse_weight=1.0,
        semantic_cosine_weight=1.0,
        kl_weight=1.0e-6,
        pixel_weight=0.1,
    )


def _post_merger_grid(cache_spec: dict, native_tokens: int) -> list[int]:
    grid_thw = list(map(int, cache_spec["image_grid_thw"]))
    if len(grid_thw) != 3:
        raise ValueError(f"Expected image_grid_thw=[T,H,W], got {grid_thw}")
    time, height, width = grid_thw
    pre_merger_tokens = time * height * width
    if pre_merger_tokens % native_tokens != 0:
        raise ValueError(
            f"Pre-merger grid {grid_thw} has {pre_merger_tokens} tokens, "
            f"not an integer multiple of native_tokens={native_tokens}"
        )
    merge_area = pre_merger_tokens // native_tokens
    merge_size = math.isqrt(merge_area)
    if merge_size**2 != merge_area or height % merge_size or width % merge_size:
        raise ValueError(
            f"Cannot infer a square spatial merge from grid={grid_thw}, native_tokens={native_tokens}"
        )
    merged_grid = [time * (height // merge_size), width // merge_size]
    if math.prod(merged_grid) != native_tokens:
        raise ValueError(
            f"Inferred post-merger grid {merged_grid} does not match native_tokens={native_tokens}"
        )
    return merged_grid


def codec_config_from_args(args, cache_spec: dict) -> dict:
    native_shape = tuple(map(int, cache_spec["native_feature_shape"]))
    native_grid = _post_merger_grid(cache_spec, native_shape[0])
    latent_grid = None
    if args.method == "symmetric_qwen":
        latent_grid = list(
            map(
                int,
                args.latent_grid_size
                if getattr(args, "latent_grid_size", None) is not None
                else native_grid,
            )
        )
        default_tokens = math.prod(latent_grid)
    else:
        default_tokens = 16
    tokens_per_view = (
        int(args.tokens_per_view)
        if args.tokens_per_view is not None
        else default_tokens
    )
    if args.method == "symmetric_qwen" and tokens_per_view != math.prod(latent_grid):
        raise ValueError(
            "For symmetric_qwen, --tokens-per-view must equal the latent-grid area "
            f"{latent_grid[0]}x{latent_grid[1]}={math.prod(latent_grid)}, "
            f"got {tokens_per_view}"
        )
    return {
        "method": args.method,
        "input_dim": native_shape[1],
        "native_tokens": native_shape[0],
        "tokens_per_view": tokens_per_view,
        "token_dim": int(args.token_dim),
        "model_dim": int(args.model_dim),
        "depth": int(args.depth),
        "num_heads": int(args.num_heads),
        "grid_size": native_grid,
        "latent_grid_size": latent_grid,
        "projection_hidden_dim": int(args.projection_hidden_dim),
        "kl_weight": float(args.kl_weight),
        "pixel_output_size": list(map(int, cache_spec["pixel_target_shape"][-2:])),
    }


def instantiate_codec(config: dict):
    return build_codec(
        method=config["method"],
        input_dim=int(config["input_dim"]),
        native_tokens=int(config["native_tokens"]),
        tokens_per_view=int(config["tokens_per_view"]),
        token_dim=int(config["token_dim"]),
        model_dim=int(config["model_dim"]),
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        pixel_output_size=config["pixel_output_size"],
        grid_size=config.get("grid_size"),
        latent_grid_size=config.get("latent_grid_size"),
        projection_hidden_dim=int(config.get("projection_hidden_dim", 1024)),
        kl_weight=float(config.get("kl_weight", 1.0e-6)),
    )


def load_codec_checkpoint(path: Path, device: torch.device | str = "cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if int(payload.get("version", -1)) != 1:
        raise ValueError(f"Unsupported codec checkpoint version in {path}")
    codec = instantiate_codec(payload["codec_config"])
    codec.load_state_dict(payload["state_dict"], strict=True)
    return codec.to(device), payload


def padded_rank_records(records: list[dict], rank: int, world_size: int, seed: int) -> list[dict]:
    """Shuffle and pad episode shards so every DDP rank executes equal steps."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if world_size > 1:
        target = int(math.ceil(len(shuffled) / world_size) * world_size)
        shuffled.extend(shuffled[: target - len(shuffled)])
    return shuffled[rank::world_size]


def chunked_indices(length: int, batch_size: int, seed: int | None = None) -> Iterable[list[int]]:
    indices = list(range(length))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start : start + batch_size]
