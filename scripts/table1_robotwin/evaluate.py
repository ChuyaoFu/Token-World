"""IRASim-compatible long-horizon recorded-action replay for TokenWorld.

The evaluator deliberately mirrors IRASim's
``evaluate_robotwin_action_replay_rollout.py`` protocol:

* use the exact episodes recorded in an IRASim reference report;
* initialize from ground-truth frame zero;
* generate 15 frames from recorded destination-aligned actions;
* feed only the final predicted visual/proprio state into the next chunk;
* compare predicted Qwen merger features against raw cam_high RGB features;
* compare the frozen QwenOFT expert's 50x14 outputs on those feature pairs.

This remains an offline replay diagnostic.  Recorded actions drive the world
model, and the action expert never closes the control loop.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image


_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tokenworld.data.dataset import (  # noqa: E402
    RobotwinSVAETokenDataset,
    resolve_latent_normalization,
)
from tokenworld.evaluation.rollout import WorldModelRollout  # noqa: E402


DEFAULT_IRASIM_ROOT = Path(__file__).resolve().parents[3] / "IRASim"
DEFAULT_CACHE = (
    DEFAULT_IRASIM_ROOT / "robotdata/robotwin_svae_d16_e4_epoch1_cache_50hz"
)
DEFAULT_DATA_ROOT = DEFAULT_IRASIM_ROOT / "robotdata/robotwin-success-fail-lerobot"
DEFAULT_CODEC = (
    DEFAULT_IRASIM_ROOT
    / "results/svae_epoch_search_16ppu/n108_d16_e4/epoch_0001.pt"
)
DEFAULT_REFERENCE_REPORT = (
    DEFAULT_IRASIM_ROOT
    / "evaluations/robotwin_action_replay_rollout/"
    "svae16_100k/balanced100_ema.json"
)
DEFAULT_ACTION_CONFIG = (
    DEFAULT_IRASIM_ROOT
    / "results/action_expert/tokenworld_qwenoft_success_cam_high_teacher/"
    "config.full.yaml"
)
DEFAULT_ACTION_CHECKPOINT = (
    DEFAULT_IRASIM_ROOT
    / "results/action_expert/tokenworld_qwenoft_success_cam_high_teacher/"
    "checkpoints/steps_50000_pytorch_model.pt"
)
DEFAULT_QWEN_MODEL = DEFAULT_IRASIM_ROOT / "pretrained_models/Qwen3-VL-4B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--irasim-root", type=Path, default=DEFAULT_IRASIM_ROOT)
    parser.add_argument("--reference-report", type=Path, default=DEFAULT_REFERENCE_REPORT)
    parser.add_argument("--codec-checkpoint", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--action-config", type=Path, default=DEFAULT_ACTION_CONFIG)
    parser.add_argument(
        "--action-checkpoint", type=Path, default=DEFAULT_ACTION_CHECKPOINT
    )
    parser.add_argument("--qwen-model", type=Path, default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--latent-method", default="symmetric_qwen")
    parser.add_argument("--view", default="cam_high")
    parser.add_argument("--stats-scope", choices=("checkpoint", "global", "per_view", "auto"), default="checkpoint")
    parser.add_argument("--execution-horizon", type=int, default=15)
    parser.add_argument("--num-denoise-steps", type=int, default=16)
    parser.add_argument("--policy-outcome", default="success")
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames-per-episode", type=int, default=0)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--action-pair-batch-size", type=int, default=2)
    parser.add_argument("--action-interface", choices=("adapter", "merger_only"), required=True)
    parser.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--cache-mode", choices=("gru", "reset"), required=True)
    return parser.parse_args()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_dump(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_text_dump(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


class FidelityAccumulator:
    """IRASim-compatible streaming element errors and token-wise cosine."""

    def __init__(self) -> None:
        self.sse = 0.0
        self.reference_energy = 0.0
        self.reference_sum = 0.0
        self.reference_square_sum = 0.0
        self.absolute_error = 0.0
        self.elements = 0
        self.cosine_sum = 0.0
        self.cosine_rows = 0
        self.maximum = 0.0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(
                f"Metric shape mismatch: {reference.shape} vs {candidate.shape}"
            )
        reference = reference.detach().double()
        candidate = candidate.detach().double()
        difference = candidate - reference
        if not torch.isfinite(difference).all():
            raise FloatingPointError("Non-finite prediction encountered")
        self.sse += float(difference.square().sum().cpu())
        self.reference_energy += float(reference.square().sum().cpu())
        self.reference_sum += float(reference.sum().cpu())
        self.reference_square_sum += float(reference.square().sum().cpu())
        self.absolute_error += float(difference.abs().sum().cpu())
        self.elements += difference.numel()
        self.maximum = max(self.maximum, float(difference.abs().max().cpu()))
        flat_reference = reference.reshape(-1, reference.shape[-1])
        flat_candidate = candidate.reshape(-1, candidate.shape[-1])
        cosine = F.cosine_similarity(flat_candidate, flat_reference, dim=-1)
        self.cosine_sum += float(cosine.sum().cpu())
        self.cosine_rows += cosine.numel()

    def report(self) -> dict:
        if self.elements == 0:
            return {
                name: float("nan")
                for name in (
                    "mse", "rmse", "mae", "max_absolute_error", "nmse",
                    "relative_rmse", "r2", "row_cosine_similarity",
                )
            } | {"elements": 0, "rows": 0}
        mse = self.sse / self.elements
        nmse = self.sse / max(self.reference_energy, 1.0e-12)
        centered_energy = self.reference_square_sum - (
            self.reference_sum**2 / self.elements
        )
        return {
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error / self.elements,
            "max_absolute_error": self.maximum,
            "nmse": nmse,
            "relative_rmse": math.sqrt(nmse),
            "r2": 1.0 - self.sse / max(centered_energy, 1.0e-12),
            "row_cosine_similarity": self.cosine_sum / max(self.cosine_rows, 1),
            "elements": self.elements,
            "rows": self.cosine_rows,
        }


_ACCUMULATOR_FIELDS = (
    "sse", "reference_energy", "reference_sum", "reference_square_sum",
    "absolute_error", "elements", "cosine_sum", "cosine_rows",
)


def accumulator_state(accumulator: FidelityAccumulator) -> dict:
    return {
        **{name: getattr(accumulator, name) for name in _ACCUMULATOR_FIELDS},
        "maximum": accumulator.maximum,
    }


def merge_accumulator_states(states: list[dict]) -> FidelityAccumulator:
    merged = FidelityAccumulator()
    for name in _ACCUMULATOR_FIELDS:
        setattr(merged, name, sum(state[name] for state in states))
    merged.maximum = max((state["maximum"] for state in states), default=0.0)
    return merged


def merge_grouped_accumulators(payloads: list[dict], key: str) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for payload in payloads:
        for index, state in payload[key].items():
            grouped[int(index)].append(state)
    return {
        index: merge_accumulator_states(states)
        for index, states in sorted(grouped.items())
    }


def merge_counts(payloads: list[dict], key: str) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for payload in payloads:
        for index, value in payload[key].items():
            result[int(index)] += int(value)
    return dict(sorted(result.items()))


def pair_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference = reference.detach().float()
    candidate = candidate.detach().float()
    difference = candidate - reference
    flat_reference = reference.reshape(-1, reference.shape[-1])
    flat_candidate = candidate.reshape(-1, candidate.shape[-1])
    return {
        "mse": float(difference.square().mean().cpu()),
        "mae": float(difference.abs().mean().cpu()),
        "cosine": float(
            F.cosine_similarity(flat_reference, flat_candidate, dim=-1).mean().cpu()
        ),
    }


def load_fidelity_module(starvla_root: Path):
    import importlib.util

    path = (
        starvla_root
        / "examples/simBenchmarks/Robotwin/codec_fidelity/"
        "evaluate_svae_action_fidelity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tokenworld_svae_action_fidelity", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_action_expert(
    fidelity,
    config_path: Path,
    checkpoint_path: Path,
    qwen_model: Path,
    device: torch.device,
):
    config = fidelity.apply_config_compat(OmegaConf.load(config_path))
    config.framework.qwenvl.base_vlm = str(qwen_model)
    policy = fidelity.build_framework(config)
    incompatible = policy.load_state_dict(
        fidelity._load_state_dict(checkpoint_path), strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Action checkpoint/model mismatch: "
            f"missing={incompatible.missing_keys[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )
    policy = policy.to(device).eval().requires_grad_(False)
    return policy, config


def action_instruction(policy, task: str, outcome: str) -> str:
    action_tokens = policy.action_token * policy.chunk_len
    suffix = (
        f" Please predict the next {policy.chunk_len} robot actions: "
        f"<action>{action_tokens}<action>."
    )
    return f"{task}\nTrajectory outcome: {outcome}." + suffix


def build_policy_template(
    policy,
    batch_size: int,
    num_views: int,
    image_size: tuple[int, int],
    task: str,
    outcome: str,
    native_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    height, width = image_size
    dummy = Image.new("RGB", (width, height))
    inputs = policy.qwen_vl_interface.build_qwenvl_inputs(
        images=[[dummy] * num_views for _ in range(batch_size)],
        instructions=[action_instruction(policy, task, outcome)] * batch_size,
    )
    template = {
        key: inputs[key].to(device)
        for key in ("input_ids", "attention_mask", "image_grid_thw")
    }
    image_token_id = int(policy.qwen_vl_interface.model.config.image_token_id)
    counts = template["input_ids"].eq(image_token_id).sum(dim=1)
    expected = num_views * native_tokens
    if counts.tolist() != [expected] * batch_size:
        raise ValueError(f"Expected {expected} image tokens/sample, got {counts.tolist()}")
    return template


@torch.inference_mode()
def policy_actions(fidelity, policy, template: dict, features: torch.Tensor) -> torch.Tensor:
    forward_inputs, input_ids, _ = fidelity._injected_inputs(
        policy, template, features
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = policy.qwen_vl_interface.model.model(**forward_inputs)
    action_queries = policy._gather_action_token_embeddings(
        output.last_hidden_state,
        input_ids,
        action_token_id=policy.action_token_id,
    )
    with torch.autocast("cuda", dtype=torch.float32):
        actions = policy.action_model.predict_action(action_queries)
    return actions.float()


def action_outputs_for_feature_pairs(
    fidelity,
    policy,
    template_cache: dict,
    reference_features: torch.Tensor,
    candidate_features: torch.Tensor,
    task: str,
    policy_outcome: str,
    image_size: tuple[int, int],
    pair_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    references, candidates = [], []
    native_tokens = int(reference_features.shape[-2])
    for start in range(0, len(reference_features), pair_batch_size):
        stop = min(start + pair_batch_size, len(reference_features))
        count = stop - start
        paired = torch.cat(
            (reference_features[start:stop], candidate_features[start:stop]), dim=0
        )
        key = (task, policy_outcome, 2 * count, native_tokens)
        if key not in template_cache:
            template_cache[key] = build_policy_template(
                policy, 2 * count, 1, image_size, task, policy_outcome,
                native_tokens, device,
            )
        paired_actions = policy_actions(
            fidelity, policy, template_cache[key], paired
        )
        references.append(paired_actions[:count])
        candidates.append(paired_actions[count:])
    return torch.cat(references), torch.cat(candidates)


@torch.no_grad()
def adapter_action_pairs(policy, codec, adapter, template, tokens, images, task, outcome, check_parity):
    from tokenworld.codec.deepstack_adapter import adapter_policy_actions, inject_actions

    references, candidates = [], []
    instruction = f"{task}\nTrajectory outcome: {outcome}."
    device = next(policy.parameters()).device
    for i, (raw, image) in enumerate(zip(tokens, images, strict=True)):
        out = policy.predict_action(examples=[dict(image=[image], lang=instruction)])
        native = torch.as_tensor(out["normalized_actions"], device=device, dtype=torch.float32)
        if check_parity and i == 0:
            interface = policy.qwen_vl_interface
            inputs = interface.build_qwenvl_inputs(
                images=[[image]], instructions=[action_instruction(policy, task, outcome)]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                merger, deep = interface.model.model.get_image_features(
                    inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device)
                )
            injected = inject_actions(policy, template, torch.cat(list(merger)), deep, readout="native_pre_norm")
            err = (injected - native).abs().max().item()
            if not err <= 1e-5:
                raise RuntimeError(f"RGB native/injected parity failed for {task}: {err}")
        candidate = adapter_policy_actions(policy, codec, adapter, template, raw)
        if native.shape != (1, 50, 14) or candidate.shape != native.shape:
            raise ValueError(f"Unexpected action shapes {native.shape}, {candidate.shape}")
        references.append(native)
        candidates.append(candidate)
    return torch.cat(references), torch.cat(candidates)


def to_pil(values: torch.Tensor) -> list[Image.Image]:
    arrays = (
        ((values.detach().float().cpu() + 1.0) * 127.5)
        .round().clamp(0, 255).to(torch.uint8)
        .permute(0, 2, 3, 1).numpy()
    )
    return [Image.fromarray(array, mode="RGB") for array in arrays]


def normalize_qwen_image_features(output, native_counts: list[int]) -> list[torch.Tensor]:
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], (tuple, list)):
        features = list(output[0])
    elif hasattr(output, "pooler_output"):
        pooled = output.pooler_output
        features = list(pooled) if isinstance(pooled, (tuple, list)) else list(torch.split(pooled, native_counts))
    elif isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        features = list(output) if len(output) == len(native_counts) else list(torch.split(output[0], native_counts))
    elif torch.is_tensor(output):
        features = list(torch.split(output, native_counts))
    else:
        raise TypeError(f"Unsupported Qwen image-feature return type: {type(output)!r}")
    if len(features) != len(native_counts):
        raise ValueError("Qwen returned the wrong number of image feature tensors")
    for feature, expected in zip(features, native_counts):
        if feature.ndim != 2 or int(feature.shape[0]) != int(expected):
            raise ValueError(f"Expected image feature [{expected},D], got {feature.shape}")
    return features


@torch.inference_mode()
def rgb_qwen_merger_features(
    policy, images: list[Image.Image], batch_size: int
) -> torch.Tensor:
    device = next(policy.parameters()).device
    causal_model = policy.qwen_vl_interface.model
    merge = int(causal_model.config.vision_config.spatial_merge_size)
    features = []
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size]
        inputs = policy.qwen_vl_interface.build_qwenvl_inputs(
            images=[[image] for image in chunk], instructions=[""] * len(chunk)
        )
        grid = inputs["image_grid_thw"].to(device)
        native_counts = [int(row.prod().item()) // (merge**2) for row in grid]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = causal_model.get_image_features(
                inputs["pixel_values"].to(device=device, dtype=torch.bfloat16), grid
            )
        features.extend(normalize_qwen_image_features(output, native_counts))
    return torch.stack(features).unsqueeze(1)


def qwen_features_from_rgb(policy, rgb: torch.Tensor, batch_size: int) -> torch.Tensor:
    batch, frames = rgb.shape[:2]
    features = rgb_qwen_merger_features(
        policy, to_pil(rgb.reshape(-1, *rgb.shape[-3:])), batch_size
    )
    return features.reshape(batch, frames, *features.shape[1:])


def decode_svae_tokens(codec, tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 4:
        tokens = tokens.unsqueeze(2)
    if tokens.ndim != 5:
        raise ValueError(f"Expected [B,F,V,T,C] S-VAE tokens, got {tokens.shape}")
    batch, frames, views, latent_tokens, _ = tokens.shape
    flat = tokens.reshape(batch * frames * views, latent_tokens, -1)
    native_tokens = int(getattr(codec, "native_tokens_per_view", latent_tokens))
    decoded = codec.decode(flat, [native_tokens] * flat.shape[0])
    return torch.stack(decoded).reshape(
        batch, frames, views, native_tokens, -1
    )


def raw_cam_high_rgb(
    dataset,
    record: dict,
    payload: dict,
    start: int,
    stop: int,
) -> torch.Tensor:
    from decord import VideoReader, cpu

    video_key = "observation.images.cam_high"
    episode_index = int(record["episode_index"])
    path = (
        Path(dataset.spec["data_root"])
        / str(record["dataset"])
        / "videos"
        / f"chunk-{episode_index // 1000:03d}"
        / video_key
        / f"episode_{episode_index:06d}.mp4"
    )
    source_ids = torch.as_tensor(payload["frame_indices"], dtype=torch.long)[start:stop]
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    frames = torch.from_numpy(reader.get_batch(source_ids.tolist()).asnumpy())
    frames = frames.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
    frames = F.interpolate(
        frames,
        size=tuple(map(int, dataset.spec["image_size"])),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return frames


class RawCacheDatasetAdapter:
    """Minimal interface required by IRASim's exact raw RGB loader."""

    def __init__(self, cache_spec: dict, data_root: Path) -> None:
        self.spec = dict(cache_spec)
        self.spec["data_root"] = str(data_root.expanduser().resolve())
        self.video_keys = list(cache_spec["video_keys"])


def resolve_view(view: str, video_keys: list[str]) -> tuple[int, str]:
    aliases = RobotwinSVAETokenDataset.VIEW_ALIASES
    if str(view).isdigit():
        index = int(view)
    else:
        key = aliases.get(str(view), str(view))
        if key not in video_keys:
            raise ValueError(f"Unknown view {view!r}; available={video_keys}")
        index = video_keys.index(key)
    if index < 0 or index >= len(video_keys):
        raise ValueError(f"View index {index} is out of range")
    return index, video_keys[index]


def checkpoint_stats_scope(args: argparse.Namespace, config: dict) -> tuple[str, str]:
    if args.stats_scope != "checkpoint":
        return args.stats_scope, "command_line"
    configured = config.get("data", {}).get("latent_stats_scope")
    if configured is not None:
        return str(configured), "checkpoint_config"
    return "global", "legacy_checkpoint_without_scope"


def selected_reference_records(
    reference_report: dict,
    cache_records: list[dict],
    maximum: int,
) -> list[tuple[int, dict]]:
    lookup = {
        (str(record["dataset"]), int(record["episode_index"])): (index, record)
        for index, record in enumerate(cache_records)
        if str(record.get("split")) == "val"
    }
    selected = []
    for episode in reference_report["episodes"]:
        key = (str(episode["dataset"]), int(episode["episode_index"]))
        if key not in lookup:
            raise KeyError(f"Reference episode is absent from cache index: {key}")
        record_index, record = lookup[key]
        if str(record["task"]) != str(episode["task"]):
            raise ValueError(f"Task mismatch for reference episode {key}")
        if str(record["outcome"]) != str(episode["outcome"]):
            raise ValueError(f"Outcome mismatch for reference episode {key}")
        selected.append((record_index, record))
    if maximum > 0:
        selected = selected[:maximum]
    if not selected:
        raise ValueError("No reference episodes selected")
    return selected


def recorded_action_chunk(
    actions: torch.Tensor,
    frame_index: int,
    execution_horizon: int,
    total_frames: int,
) -> tuple[torch.Tensor, int, int]:
    valid_steps = min(execution_horizon, total_frames - 1 - frame_index)
    if valid_steps <= 0:
        raise ValueError("A replay chunk requires at least one destination frame")
    chunk = actions[frame_index + 1 : frame_index + valid_steps + 1].float()
    if len(chunk) != valid_steps:
        raise ValueError(f"Expected {valid_steps} recorded actions, got {len(chunk)}")
    padding = execution_horizon - valid_steps
    if padding:
        chunk = torch.cat((chunk, chunk[-1:].expand(padding, -1)), dim=0)
    return chunk, valid_steps, padding


def report_markdown(report: dict) -> str:
    lines = [
        "# TokenWorld IRASim-compatible recorded-action replay",
        "",
        f"- Checkpoint: `{report['world_model']['checkpoint']}`",
        f"- Reference episodes: `{report['provenance']['reference_report']}`",
        f"- Episodes: {report['evaluation']['evaluated_episodes']}; "
        f"generated frames: {report['evaluation']['generated_frames']}",
        f"- Action interface: {report['protocol']['action_interface']}",
        f"- Action reference: {report['protocol']['reference_action_source']}",
        f"- Denoise steps: {report['world_model']['num_inference_steps']}",
        "",
        "| replay chunk | time range | Qwen feature cosine | action cosine | frames |",
        "|---:|---:|---:|---:|---:|",
    ]
    chunks = {row["replay_chunk"]: row for row in report["by_replay_chunk"]}
    horizon = int(report["protocol"]["execution_horizon"])
    fps = float(report["protocol"]["frame_rate_hz"])
    for index in (0, 5, 10, 15):
        if index not in chunks:
            continue
        row = chunks[index]
        start = (index * horizon + 1) / fps
        stop = ((index + 1) * horizon) / fps
        lines.append(
            f"| {index} | {start:.2f}-{stop:.2f} s | "
            f"{row['qwen_feature']['row_cosine_similarity']:.6f} | "
            f"{row['action_expert_output_50x14']['row_cosine_similarity']:.6f} | "
            f"{row['frame_count']} |"
        )
    lines.extend(
        [
            "",
            "The frozen action expert is diagnostic-only. Recorded dataset actions "
            "drive TokenWorld; no simulator success rate is measured.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.execution_horizon != 15:
        raise ValueError("IRASim comparison protocol requires execution_horizon=15")
    if min(args.num_denoise_steps, args.feature_batch_size, args.action_pair_batch_size) <= 0:
        raise ValueError("Step and batch-size arguments must be positive")
    if args.max_episodes < 0 or args.max_frames_per_episode < 0:
        raise ValueError("Episode/frame limits cannot be negative")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    paths = (
        args.checkpoint,
        args.cache / "cache_spec.json",
        args.cache / "index.json",
        args.cache / "train_stats.pt",
        args.reference_report,
        args.codec_checkpoint,
        args.action_config,
        args.action_checkpoint,
        args.qwen_model,
    )
    for path in paths:
        if not path.expanduser().resolve().exists():
            raise FileNotFoundError(path)

    with (args.cache / "cache_spec.json").open(encoding="utf-8") as handle:
        cache_spec = json.load(handle)
    with (args.cache / "index.json").open(encoding="utf-8") as handle:
        cache_index = json.load(handle)
    with args.reference_report.open(encoding="utf-8") as handle:
        reference_report = json.load(handle)
    reference_records = selected_reference_records(
        reference_report, cache_index["records"], args.max_episodes
    )
    local_records = reference_records[rank::world_size]
    if not local_records:
        raise ValueError(f"No episodes assigned to rank {rank}/{world_size}")

    video_keys = list(cache_spec["video_keys"])
    view_index, video_key = resolve_view(args.view, video_keys)
    if video_key != "observation.images.cam_high":
        raise ValueError("IRASim comparison protocol requires cam_high")
    frame_rate_hz = float(cache_spec.get("export_frame_rate_hz", 50.0))
    if frame_rate_hz != 50.0:
        raise ValueError(f"Expected 50 Hz cache, got {frame_rate_hz}")
    raw_dataset = RawCacheDatasetAdapter(cache_spec, args.data_root)

    print(
        f"[rank {rank}/{world_size}] loading TokenWorld {args.checkpoint}",
        flush=True,
    )
    rollout = WorldModelRollout(
        str(args.checkpoint),
        device=str(device),
        num_steps=args.num_denoise_steps,
        no_shortcut=False,
    )
    stats = _torch_load(args.cache / "train_stats.pt")
    stats_scope, stats_source = checkpoint_stats_scope(args, rollout.config)
    latent_mean, latent_std, resolved_scope, stat_keys = resolve_latent_normalization(
        stats,
        latent_method=args.latent_method,
        video_keys=video_keys,
        view_indices=[view_index],
        scope=stats_scope,
    )
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    action_mean = stats["action"]["mean"].float().to(device)
    action_std = stats["action"]["std"].float().clamp_min(1.0e-6).to(device)
    continuous_indices = list(RobotwinSVAETokenDataset.CONTINUOUS_INDICES)
    gripper_indices = list(RobotwinSVAETokenDataset.GRIPPER_INDICES)

    starvla = args.action_config.resolve().parents[1] / "code/starvla"
    sys.path.insert(0, str(starvla))
    fidelity = load_fidelity_module(
        starvla
    )
    policy, action_config = load_action_expert(
        fidelity,
        args.action_config.expanduser().resolve(),
        args.action_checkpoint.expanduser().resolve(),
        args.qwen_model.expanduser().resolve(),
        device,
    )
    if int(policy.action_horizon) != 50:
        raise ValueError(f"Expected action horizon 50, got {policy.action_horizon}")
    from tokenworld.codec.common import load_codec_checkpoint
    codec, codec_payload = load_codec_checkpoint(
        args.codec_checkpoint.expanduser().resolve(), device=device
    )
    codec.eval().requires_grad_(False)
    if str(codec_payload["codec_config"]["method"]) != args.latent_method:
        raise ValueError("Codec/checkpoint latent methods do not match")

    adapter = None
    if args.action_interface == "adapter":
        from tokenworld.codec.deepstack_adapter import DeepStackAdapter
        if args.adapter_checkpoint is None:
            raise ValueError("--adapter-checkpoint is required for the adapter interface")
        adapter = DeepStackAdapter().to(device)
        adapter.load_state_dict(torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)["model"])
        adapter.eval().requires_grad_(False)

    feature_metrics = FidelityAccumulator()
    action_metrics = FidelityAccumulator()
    feature_by_replan = defaultdict(FidelityAccumulator)
    action_by_replan = defaultdict(FidelityAccumulator)
    feature_by_offset = defaultdict(FidelityAccumulator)
    action_by_offset = defaultdict(FidelityAccumulator)
    replan_counts: dict[int, int] = defaultdict(int)
    offset_counts: dict[int, int] = defaultdict(int)
    frame_records: list[dict] = []
    episode_summaries: list[dict] = []
    template_cache = {}
    generated_frames = 0
    replay_chunks = 0
    padded_action_steps = 0
    tasks, outcomes = set(), set()
    started = time.perf_counter()

    for local_episode, (record_index, record) in enumerate(local_records, start=1):
        payload = _torch_load(args.cache / record["path"])
        raw_latents = payload["latents"][args.latent_method].float()
        raw_actions = payload["actions"].float()
        raw_states = payload["states"].float()
        if not torch.equal(raw_actions, raw_states):
            raise ValueError(f"action/state mismatch in {record['path']}")
        total_frames = min(
            int(record["num_frames"]),
            len(raw_latents),
            len(raw_actions),
            len(raw_states),
            len(payload.get("frame_indices", raw_states)),
        )
        if args.max_frames_per_episode > 0:
            total_frames = min(total_frames, args.max_frames_per_episode)
        if total_frames < 2:
            continue

        task = str(record["task"])
        outcome = str(record["outcome"])
        tasks.add(task)
        outcomes.add(outcome)
        initial_raw = raw_latents[0:1, view_index : view_index + 1].to(
            device, non_blocking=True
        )
        current_latent = ((initial_raw - latent_mean) / latent_std).unsqueeze(0)
        current_proprio = (
            (raw_states[0:1].to(device, non_blocking=True) - action_mean) / action_std
        ).unsqueeze(0)

        frame_index = 0
        replan_index = 0
        history = None
        episode_feature = FidelityAccumulator()
        episode_action = FidelityAccumulator()
        episode_frames = 0

        while frame_index < total_frames - 1:
            action_chunk, valid_steps, padding = recorded_action_chunk(
                raw_actions,
                frame_index,
                args.execution_horizon,
                total_frames,
            )
            # Release WM takes one boundary action plus destination-frame actions.
            action_chunk = torch.cat((raw_actions[frame_index:frame_index+1], action_chunk))
            if args.cache_mode == "gru":
                action_chunk = action_chunk[:valid_steps + 1]
            action_chunk = action_chunk.to(device, non_blocking=True)
            continuous = (
                action_chunk[..., continuous_indices]
                - action_mean[continuous_indices]
            ) / action_std[continuous_indices]
            discrete = (action_chunk[..., gripper_indices] > 0.5).long()
            torch.manual_seed(args.seed + record_index * 1000 + replan_index)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                result = rollout.rollout(
                    current_latent,
                    discrete_actions=discrete.unsqueeze(0),
                    continuous_actions=continuous.unsqueeze(0),
                    init_proprio=current_proprio,
                    time_cache=history, return_time_cache=args.cache_mode == "gru",
                    gru_only_boundary=args.cache_mode == "gru",
                )
            history = result.time_cache
            expected_frames = len(action_chunk)
            if result.latents.shape[1] != expected_frames:
                raise RuntimeError(
                    f"TokenWorld returned {result.latents.shape[1]} frames, "
                    f"expected {expected_frames}"
                )
            predicted_normalized = result.latents[:, 1 : valid_steps + 1].float()
            predicted_tokens = predicted_normalized * latent_std + latent_mean
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                candidate_features = decode_svae_tokens(
                    codec, predicted_tokens
                )

            raw_rgb = raw_cam_high_rgb(
                raw_dataset,
                record,
                payload,
                frame_index + 1,
                frame_index + valid_steps + 1,
            )
            target_features = qwen_features_from_rgb(
                policy,
                raw_rgb.unsqueeze(0),
                args.feature_batch_size,
            )
            if target_features.shape != candidate_features.shape:
                raise ValueError(
                    f"GT/candidate feature mismatch: {target_features.shape} vs "
                    f"{candidate_features.shape}"
                )
            if adapter is not None:
                from tokenworld.evaluation.closed_loop import build_policy_template as closed_loop_template
                if task not in template_cache:
                    template_cache[task] = closed_loop_template(policy, task, args.policy_outcome)
                reference_actions, candidate_actions = adapter_action_pairs(
                    policy, codec, adapter, template_cache[task], predicted_tokens[0], to_pil(raw_rgb),
                    task, args.policy_outcome, check_parity=(replan_index == 0),
                )
            else:
                reference_actions, candidate_actions = (
                    action_outputs_for_feature_pairs(
                        fidelity,
                        policy,
                        template_cache,
                        target_features[0],
                        candidate_features[0],
                        task,
                        args.policy_outcome,
                        tuple(map(int, cache_spec["image_size"])),
                        args.action_pair_batch_size,
                        device,
                    )
                )

            feature_metrics.update(target_features, candidate_features)
            action_metrics.update(reference_actions, candidate_actions)
            feature_by_replan[replan_index].update(
                target_features, candidate_features
            )
            action_by_replan[replan_index].update(
                reference_actions, candidate_actions
            )
            episode_feature.update(target_features, candidate_features)
            episode_action.update(reference_actions, candidate_actions)
            for offset in range(valid_steps):
                feature_by_offset[offset].update(
                    target_features[:, offset], candidate_features[:, offset]
                )
                action_by_offset[offset].update(
                    reference_actions[offset], candidate_actions[offset]
                )
                replan_counts[replan_index] += 1
                offset_counts[offset] += 1
                frame_records.append(
                    {
                        "dataset": str(record["dataset"]),
                        "episode_index": int(record["episode_index"]),
                        "task": task,
                        "outcome": outcome,
                        "nominal_frame": frame_index + offset + 1,
                        "nominal_seconds": (frame_index + offset + 1) / frame_rate_hz,
                        "replay_chunk": replan_index,
                        "chunk_offset": offset + 1,
                        "qwen_feature": pair_summary(
                            target_features[:, offset], candidate_features[:, offset]
                        ),
                        "action_expert_output_50x14": pair_summary(
                            reference_actions[offset], candidate_actions[offset]
                        ),
                    }
                )

            generated_frames += valid_steps
            episode_frames += valid_steps
            replay_chunks += 1
            if args.cache_mode == "reset":
                padded_action_steps += padding
            frame_index += valid_steps
            replan_index += 1
            if frame_index < total_frames - 1:
                current_latent = result.latents[
                    :, valid_steps : valid_steps + 1
                ].detach()
                if result.proprio is None:
                    raise RuntimeError("TokenWorld did not return predicted proprio")
                current_proprio = result.proprio[
                    :, valid_steps : valid_steps + 1
                ].detach()

            del (
                action_chunk,
                continuous,
                discrete,
                result,
                predicted_normalized,
                predicted_tokens,
                candidate_features,
                raw_rgb,
                target_features,
                reference_actions,
                candidate_actions,
            )

        episode_summaries.append(
            {
                "dataset": str(record["dataset"]),
                "episode_index": int(record["episode_index"]),
                "task": task,
                "outcome": outcome,
                "frames": total_frames,
                "generated_frames": episode_frames,
                "replay_chunks": replan_index,
                "qwen_feature": episode_feature.report(),
                "action_expert_output_50x14": episode_action.report(),
            }
        )
        print(
            f"[rank {rank}] [{local_episode}/{len(local_records)}] "
            f"task={task} outcome={outcome} frames={total_frames} "
            f"feature_cos={feature_metrics.report()['row_cosine_similarity']:.6g} "
            f"action_cos={action_metrics.report()['row_cosine_similarity']:.6g} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    local_payload = {
        "feature": accumulator_state(feature_metrics),
        "action": accumulator_state(action_metrics),
        "feature_by_replan": {
            index: accumulator_state(metric)
            for index, metric in feature_by_replan.items()
        },
        "action_by_replan": {
            index: accumulator_state(metric)
            for index, metric in action_by_replan.items()
        },
        "feature_by_offset": {
            index: accumulator_state(metric)
            for index, metric in feature_by_offset.items()
        },
        "action_by_offset": {
            index: accumulator_state(metric)
            for index, metric in action_by_offset.items()
        },
        "replan_counts": dict(replan_counts),
        "offset_counts": dict(offset_counts),
        "frame_records": frame_records,
        "episodes": episode_summaries,
        "generated_frames": generated_frames,
        "replay_chunks": replay_chunks,
        "padded_action_steps": padded_action_steps,
        "tasks": sorted(tasks),
        "outcomes": sorted(outcomes),
        "elapsed": time.perf_counter() - started,
    }
    if distributed:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
        dist.destroy_process_group()
        if rank != 0:
            return
        payloads = gathered
    else:
        payloads = [local_payload]

    feature_metrics = merge_accumulator_states(
        [payload["feature"] for payload in payloads]
    )
    action_metrics = merge_accumulator_states(
        [payload["action"] for payload in payloads]
    )
    feature_by_replan = merge_grouped_accumulators(
        payloads, "feature_by_replan"
    )
    action_by_replan = merge_grouped_accumulators(
        payloads, "action_by_replan"
    )
    feature_by_offset = merge_grouped_accumulators(
        payloads, "feature_by_offset"
    )
    action_by_offset = merge_grouped_accumulators(
        payloads, "action_by_offset"
    )
    replan_counts = merge_counts(payloads, "replan_counts")
    offset_counts = merge_counts(payloads, "offset_counts")
    frame_records = sorted(
        [row for payload in payloads for row in payload["frame_records"]],
        key=lambda row: (row["dataset"], row["episode_index"], row["nominal_frame"]),
    )
    episode_summaries = sorted(
        [row for payload in payloads for row in payload["episodes"]],
        key=lambda row: (row["dataset"], row["episode_index"]),
    )
    generated_frames = sum(payload["generated_frames"] for payload in payloads)
    replay_chunks = sum(payload["replay_chunks"] for payload in payloads)
    padded_action_steps = sum(payload["padded_action_steps"] for payload in payloads)

    report = {
        "version": 1,
        "protocol": {
            "cross_segment_cache": args.cache_mode,
            "action_interface": args.action_interface,
            "reference_action_source": ("native_rgb_predict_action" if adapter is not None else "gt_merger_only_post_norm"),
            "action_space": "normalized",
            "native_injection_parity": ("first predicted frame of every episode, max abs <= 1e-5" if adapter is not None else None),
            "world_model_action_source": "recorded_dataset_trajectory",
            "action_alignment": "destination frame actions predict destination frames",
            "recursive_feedback": "predicted frame 15 and predicted proprio condition the next chunk",
            "initial_ground_truth_context_frames": 1,
            "execution_horizon": args.execution_horizon,
            "frame_rate_hz": frame_rate_hz,
            "action_expert_role": "diagnostic_only_never_drives_world_model",
            "action_expert_evaluation": "every_generated_frame",
            "action_expert_output_shape_per_frame": [50, 14],
            "ground_truth_qwen_feature_source": "raw_cam_high_rgb",
            "policy_outcome": args.policy_outcome,
            "tail_action_padding": ("repeat final action and skip padded frames" if args.cache_mode == "reset"
                                    else "generate only the valid tail"),
        },
        "world_model": {
            "representation": "tokenworld_qwen_svae",
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "weights": "model",
            "num_inference_steps": args.num_denoise_steps,
        },
        "normalization": {
            "scope": resolved_scope,
            "source": stats_source,
            "keys": stat_keys,
            "action_scope": "train_split_global",
        },
        "provenance": {
            "reference_report": str(args.reference_report.expanduser().resolve()),
            "reference_report_sha256": sha256_file(args.reference_report),
            "adapter_checkpoint": str(args.adapter_checkpoint.resolve()) if adapter is not None else None,
            "adapter_checkpoint_sha256": sha256_file(args.adapter_checkpoint) if adapter is not None else None,
            "checkpoint": str(args.checkpoint.expanduser().resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "cache": str(args.cache.expanduser().resolve()),
            "cache_spec_sha256": sha256_file(args.cache / "cache_spec.json"),
            "train_stats_sha256": sha256_file(args.cache / "train_stats.pt"),
        },
        "codec": {
            "checkpoint": str(args.codec_checkpoint.expanduser().resolve()),
            "method": str(codec_payload["codec_config"]["method"]),
            "token_dim": int(codec_payload["codec_config"]["token_dim"]),
            "tokens_per_view": 108,
        },
        "action_expert": {
            "config": str(args.action_config.expanduser().resolve()),
            "checkpoint": str(args.action_checkpoint.expanduser().resolve()),
            "framework": str(action_config.framework.name),
            "view": "cam_high",
            "include_state": False,
            "output_shape": [50, 14],
        },
        "evaluation": {
            "requested_episodes": len(reference_records),
            "evaluated_episodes": len(episode_summaries),
            "generated_frames": generated_frames,
            "replay_chunks": replay_chunks,
            "tail_padded_action_steps": padded_action_steps,
            "tasks": sorted({task for payload in payloads for task in payload["tasks"]}),
            "outcomes": sorted(
                {outcome for payload in payloads for outcome in payload["outcomes"]}
            ),
            "world_size": world_size,
            "elapsed_seconds": max(payload["elapsed"] for payload in payloads),
        },
        "fidelity": {
            "qwen_features": feature_metrics.report(),
            "action_expert_outputs_50x14": action_metrics.report(),
        },
        "by_replay_chunk": [
            {
                "replay_chunk": index,
                "nominal_frame_start": index * args.execution_horizon + 1,
                "nominal_seconds_start": (
                    index * args.execution_horizon + 1
                ) / frame_rate_hz,
                "frame_count": replan_counts[index],
                "qwen_feature": feature_by_replan[index].report(),
                "action_expert_output_50x14": action_by_replan[index].report(),
            }
            for index in sorted(feature_by_replan)
        ],
        "by_chunk_offset": [
            {
                "chunk_offset": index + 1,
                "frame_count": offset_counts[index],
                "qwen_feature": feature_by_offset[index].report(),
                "action_expert_output_50x14": action_by_offset[index].report(),
            }
            for index in sorted(feature_by_offset)
        ],
        "frames": frame_records,
        "episodes": episode_summaries,
    }
    output = args.output.expanduser().resolve()
    atomic_json_dump(report, output / "report.json")
    atomic_text_dump(report_markdown(report), output / "summary.md")
    print(json.dumps({
        "evaluated_episodes": len(episode_summaries),
        "generated_frames": generated_frames,
        "qwen_feature_cosine": feature_metrics.report()["row_cosine_similarity"],
        "action_cosine": action_metrics.report()["row_cosine_similarity"],
        "report": str(output / "report.json"),
    }, indent=2), flush=True)
    gc.collect()


if __name__ == "__main__":
    main()
