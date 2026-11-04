"""Canonical TokenWorld open-loop fidelity evaluation for RoboTwin.

The evaluator uses recorded actions only.  The world model predicts future
native latents autoregressively from one GT frame by default (four-frame
context is available explicitly). Subsequent model inputs use predictions.  Predicted
latents are first denormalized with the cache's train statistics.  For S-VAE
latents, both branches are decoded with the same frozen codec; for qwen_raw
latents, the native Qwen features are compared directly.  The same frozen
QwenOFT action expert is then applied to both feature branches.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from collections import defaultdict
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class FidelityAccumulator:
    """Canonical streaming element errors and row-wise cosine."""

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

    @torch.no_grad()
    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        reference = reference.detach().double().cpu()
        candidate = candidate.detach().double().cpu()
        if reference.shape != candidate.shape:
            raise ValueError(f"Metric shape mismatch: {tuple(reference.shape)} vs {tuple(candidate.shape)}")
        difference = candidate - reference
        if not torch.isfinite(difference).all():
            raise FloatingPointError("Non-finite fidelity value")
        self.sse += float(difference.square().sum())
        self.reference_energy += float(reference.square().sum())
        self.reference_sum += float(reference.sum())
        self.reference_square_sum += float(reference.square().sum())
        self.absolute_error += float(difference.abs().sum())
        self.elements += int(difference.numel())
        if difference.numel():
            self.maximum = max(self.maximum, float(difference.abs().max()))
        flat_reference = reference.reshape(-1, reference.shape[-1])
        flat_candidate = candidate.reshape(-1, candidate.shape[-1])
        cosine = F.cosine_similarity(flat_candidate, flat_reference, dim=-1)
        self.cosine_sum += float(cosine.sum())
        self.cosine_rows += int(cosine.numel())

    def merge(self, other: "FidelityAccumulator") -> None:
        for name in ("sse", "reference_energy", "reference_sum", "reference_square_sum", "absolute_error", "elements", "cosine_sum", "cosine_rows"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.maximum = max(self.maximum, other.maximum)

    def report(self) -> dict:
        if not self.elements:
            return {"count": 0, "elements": 0, "rows": 0}
        mse = self.sse / self.elements
        nmse = self.sse / max(self.reference_energy, 1.0e-12)
        centered = self.reference_square_sum - self.reference_sum ** 2 / self.elements
        cosine = self.cosine_sum / max(self.cosine_rows, 1)
        return {
            "count": int(self.elements),
            "elements": int(self.elements),
            "rows": int(self.cosine_rows),
            "mse": mse,
            "rmse": math.sqrt(mse),
            "mae": self.absolute_error / self.elements,
            "max_absolute_error": self.maximum,
            "nmse": nmse,
            "relative_rmse": math.sqrt(nmse),
            "r2": 1.0 - self.sse / max(centered, 1.0e-12),
            "row_cosine_similarity": cosine,
            "cosine": cosine,
            "cosine_rows": int(self.cosine_rows),
        }


def _json_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_float(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_float(item) for item in value]
    return value


def _action_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    acc = FidelityAccumulator()
    acc.update(reference, candidate)
    report = acc.report()
    report["action_vector_cosine_similarity"] = report.get("row_cosine_similarity", float("nan"))
    return report


def _physical_action_metrics(reference: torch.Tensor, candidate: torch.Tensor, statistics_path: Path, profile: str) -> dict:
    stats = json.loads(statistics_path.read_text())
    if len(stats) != 1:
        raise ValueError(f"Expected one action-statistics entry, got {list(stats)}")
    action_stats = next(iter(stats.values()))["action"]
    minimum = torch.as_tensor(action_stats["min"], dtype=torch.float64)
    maximum = torch.as_tensor(action_stats["max"], dtype=torch.float64)
    reference = reference.detach().cpu().double()
    candidate = candidate.detach().cpu().double()
    if minimum.numel() != reference.shape[-1] or maximum.shape != minimum.shape:
        raise ValueError("Action statistics do not match action dimension")
    if profile == "muka":
        ref_physical = (reference + 1.0) * 0.5 * (maximum - minimum) + minimum
        cand_physical = (candidate + 1.0) * 0.5 * (maximum - minimum) + minimum
        difference = cand_physical - ref_physical
        return {
            "mae": float(difference.abs().mean()),
            "rmse": float(difference.square().mean().sqrt()),
            "max_absolute_error": float(difference.abs().max()),
            "per_dimension_mae": difference.abs().reshape(-1, difference.shape[-1]).mean(0).tolist(),
            "units": ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper_closedness"],
        }
    ref_joint = (reference[..., :12] + 1.0) * 0.5 * (maximum[:12] - minimum[:12]) + minimum[:12]
    cand_joint = (candidate[..., :12] + 1.0) * 0.5 * (maximum[:12] - minimum[:12]) + minimum[:12]
    difference = cand_joint - ref_joint
    ref_gripper = reference[..., 12:] > 0.49
    cand_gripper = candidate[..., 12:] > 0.49
    return {
        "joint_mae": float(difference.abs().mean()),
        "joint_rmse": float(difference.square().mean().sqrt()),
        "joint_max_absolute_error": float(difference.abs().max()),
        "gripper_binary_agreement": float((ref_gripper == cand_gripper).double().mean()),
    }


def _cache_action_to_model(action: torch.Tensor, dataset, device: torch.device):
    normalized = ((action.to(device) - dataset.action_mean.to(device)) / dataset.action_std.to(device)).float()
    if dataset.action_schema == "muka7":
        return (action.to(device)[6:7] > 0.5).long().unsqueeze(0), normalized[:6].unsqueeze(0)
    return (
        (action.to(device)[[6, 13]] > 0.5).long().unsqueeze(0),
        normalized[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]].unsqueeze(0),
    )


def _decode_qwen(codec, native_latent: torch.Tensor) -> torch.Tensor:
    if native_latent.ndim == 3:
        native_latent = native_latent.unsqueeze(0)
    if native_latent.ndim != 4:
        raise ValueError(f"Expected native latent [B,V,N,D], got {tuple(native_latent.shape)}")
    batch, views, tokens, _ = native_latent.shape
    expected = int(getattr(codec, "tokens_per_view", 0))
    joint = views > 1 and expected == views * tokens
    if joint:
        flat = native_latent.reshape(batch, views * tokens, -1)
        decoded = codec.decode(flat, [views * tokens] * batch)
    else:
        flat = native_latent.reshape(batch * views, tokens, -1)
        try:
            decoded = codec.decode(flat, [tokens] * (batch * views))
        except ValueError as error:
            if views <= 1 or "Expected shared_tokens" not in str(error):
                raise
            flat = native_latent.reshape(batch, views * tokens, -1)
            decoded = codec.decode(flat, [views * tokens] * batch)
            joint = True
    if isinstance(decoded, (tuple, list)):
        decoded = decoded[0] if len(decoded) == 1 and torch.is_tensor(decoded[0]) else torch.stack(list(decoded), dim=0)
    if decoded.ndim == 2 and batch == 1:
        decoded = decoded.unsqueeze(0)
    if decoded.ndim != 3:
        raise ValueError(f"Codec returned unexpected shape {tuple(decoded.shape)}")
    if joint:
        return decoded.reshape(batch, views, decoded.shape[1] // views, decoded.shape[2])
    return decoded.reshape(batch, views, decoded.shape[1], decoded.shape[2])


@torch.inference_mode()
def _extract_teacher_qwen(teacher, frames: torch.Tensor, normalize_features) -> torch.Tensor:
    from PIL import Image
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    images = [Image.fromarray(frame.cpu().numpy()).convert("RGB").resize((384, 288), Image.BILINEAR) for frame in frames]
    interface = teacher.qwen_vl_interface
    encoded = interface.processor.image_processor(images=images, return_tensors="pt")
    grid = encoded["image_grid_thw"]
    merge = int(interface.model.config.vision_config.spatial_merge_size)
    counts = [int(row.prod().item()) // (merge ** 2) for row in grid]
    parameter = next(interface.model.parameters())
    output = interface.model.get_image_features(
        encoded["pixel_values"].to(parameter.device, dtype=torch.bfloat16),
        grid.to(parameter.device),
    )
    features = normalize_features(output, counts)
    return torch.stack(features, dim=0).float().unsqueeze(0)


def _load_qwen_teacher(model_path: Path, device: torch.device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    class Interface: pass
    class Wrapper: pass
    interface = Interface()
    interface.processor = AutoProcessor.from_pretrained(model_path)
    interface.model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to(device).eval()
    wrapper = Wrapper()
    wrapper.qwen_vl_interface = interface
    return wrapper


class ActionExpert:
    def __init__(self, bundle: Path, profile: str, device: torch.device):
        self.bundle = bundle
        self.profile = profile
        self.device = device
        from tokenworld.policies import qwenoft as helper
        self.helper = helper
        self.policy, self.config_path, self.checkpoint_path = helper._load_policy(bundle, device)
        self.state_stats = None
        self.action_stats_path = bundle / "model/dataset_statistics.json"

    @torch.inference_mode()
    def actions(self, task: str, outcome: str, state: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        template = self.helper._build_policy_template(self.policy, task, outcome, (224, 224) if self.profile == "muka" else (288, 384))
        return self.helper._policy_actions(self.policy, template, features)


def _infer_profile(dataset, requested: str) -> str:
    if requested != "auto":
        return requested
    if dataset.action_schema == "muka7":
        return "muka"
    return "robotwin"


def select_episode_indices(records, indices, maximum, selection, seed):
    """Use the historical balanced_episode_indices task/outcome allocation.

    With 50 tasks, both outcomes and maximum=100, select the first validation
    episode in each group, matching the historical evenly-spaced selector.
    """
    if maximum <= 0 or maximum >= len(indices):
        return list(indices)
    if selection == "sequential":
        return list(indices[:maximum])
    grouped = defaultdict(list)
    for index in indices:
        record = records[index]
        grouped[(str(record["task"]), str(record["outcome"]))].append(index)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    def spaced(values, count):
        if count <= 0:
            return []
        if count >= len(values):
            return list(values)
        if count == 1:
            return [values[0]]
        return [values[round(i * (len(values) - 1) / (count - 1))] for i in range(count)]
    if maximum < len(keys):
        return sorted(grouped[keys[i]][0] for i in spaced(list(range(len(keys))), maximum))
    quotient, remainder = divmod(maximum, len(keys))
    selected = []
    for position, key in enumerate(keys):
        selected.extend(spaced(grouped[key], quotient + int(position < remainder)))
    # Redistribute only if some groups are too small to fill their quota.
    chosen = set(selected)
    while len(selected) < maximum:
        for key in keys:
            candidate = next((i for i in grouped[key] if i not in chosen), None)
            if candidate is not None:
                selected.append(candidate); chosen.add(candidate)
                if len(selected) == maximum:
                    break
    return sorted(selected)


def step_time_report(seconds, latent_method, warmup_steps=0, initial_context_frames=1):
    skip = min(warmup_steps, len(seconds))
    measured = seconds[skip:]
    report = {
        "scope": "world_model_forward + latent_denormalization + predicted_feature_decode",
        "decoder_applied": latent_method != "qwen_raw",
        "excludes": ["data_loading", "input_preparation", "history_update", "ground_truth_branch", "action_expert", "fidelity_metrics", "logging"],
        "synchronization": "CUDA synchronize immediately before and after measured region",
        "batch_size": 1, "initial_context_frames": initial_context_frames,
        "warmup_steps_excluded": skip, "measured_steps": len(measured),
        "total_seconds": sum(measured),
    }
    if measured:
        values = torch.tensor(measured, dtype=torch.float64)
        report.update(mean_ms=float(values.mean()) * 1000,
                      p50_ms=float(values.quantile(0.5)) * 1000,
                      p90_ms=float(values.quantile(0.9)) * 1000,
                      p95_ms=float(values.quantile(0.95)) * 1000)
    return report


@torch.inference_mode()
def _run_episode(*, dataset, record_index: int, world_model, codec, teacher, action_expert,
                 normalize_features, device: torch.device, max_steps: int, query_every: int,
                 reference_mode: str, initial_context_frames: int = 1, chunk_size: int = 16):
    record = dataset.records[record_index]
    payload = dataset._load_episode(record_index)
    native = payload["latents"][dataset.latent_method].float()[:, dataset.view_indices]
    if native.ndim == 3:
        native = native.unsqueeze(1)
    total = int(native.shape[0])
    last = total - 1 if max_steps <= 0 else min(total - 1, int(max_steps))
    if initial_context_frames not in (1, 4):
        raise ValueError("initial_context_frames must be 1 or 4")
    if last < initial_context_frames:
        raise ValueError(f"Episode {record} needs at least {initial_context_frames + 1} frames")
    mean = dataset.latent_mean
    std = dataset.latent_std
    normalized = ((native - mean) / std).to(device)
    states = ((payload["states"].float() - dataset.action_mean) / dataset.action_std).to(device)
    history = normalized[:initial_context_frames].unsqueeze(0)
    history_state = states[:initial_context_frames].unsqueeze(0)
    # Match the destination-frame action alignment used by the training dataset.
    replay_actions = payload["states"] if dataset.action_schema == "muka7" else payload["actions"]
    model_actions = [_cache_action_to_model(item, dataset, device) for item in replay_actions.float()]
    action_d = torch.cat([model_actions[i][0] for i in range(initial_context_frames)], dim=0).unsqueeze(0)
    action_c = torch.cat([model_actions[i][1] for i in range(initial_context_frames)], dim=0).unsqueeze(0)
    raw_frames = None
    if reference_mode == "raw_teacher":
        raw_frames = dataset._load_raw_images(record_index, record, torch.arange(initial_context_frames, last + 1, dtype=torch.long))
    if chunk_size < 1: raise ValueError("chunk_size must be positive")
    chunk_accumulators = {}
    feature_acc = FidelityAccumulator()
    action_acc = FidelityAccumulator()
    predicted_actions, reference_actions = [], []
    started = time.perf_counter()
    step_seconds = []
    mean_device, std_device = mean.to(device), std.to(device)
    for step in range(initial_context_frames, last + 1):
        discrete, continuous = model_actions[step]
        discrete_sequence = torch.cat([action_d, discrete.unsqueeze(1)], 1)
        continuous_sequence = torch.cat([action_c, continuous.unsqueeze(1)], 1)
        if device.type == "cuda": torch.cuda.synchronize(device)
        tic = time.perf_counter()
        if initial_context_frames == 1:
            # Only frame 0 is GT. All subsequent visual/proprio inputs are
            # predictions, through the existing single-frame rollout API.
            next_latent, next_state = world_model.step(
                history[:, -1], discrete, continuous, history_state[:, -1],
            )
        else:
            next_latent, next_state = world_model.step_with_history(
                history, discrete_sequence, continuous_sequence, history_state,
            )
        if next_latent.ndim == 5:
            next_latent = next_latent[:, 0]
        if history.ndim == 5 and next_latent.ndim == 3:
            next_latent = next_latent.unsqueeze(1)
        if next_latent.shape != history[:, -1].shape:
            raise ValueError(f"Predicted latent shape {tuple(next_latent.shape)} does not match history {tuple(history[:, -1].shape)}")
        predicted_native = next_latent * std_device + mean_device
        predicted_features = predicted_native if dataset.latent_method == "qwen_raw" else _decode_qwen(codec, predicted_native)
        if device.type == "cuda": torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - tic)
        # Everything below is outside the timed model -> feature path.
        history = torch.cat([history[:, 1:], next_latent.unsqueeze(1)], 1)
        if next_state is not None:
            if next_state.ndim == 3: next_state = next_state[:, 0]
            history_state = torch.cat([history_state[:, 1:], next_state.unsqueeze(1)], 1)
        action_d = discrete_sequence[:, 1:]
        action_c = continuous_sequence[:, 1:]
        gt_native = native[step].to(device).unsqueeze(0)
        if reference_mode == "raw_teacher":
            gt_features = _extract_teacher_qwen(teacher, raw_frames[step - initial_context_frames], normalize_features)
        else:
            gt_features = gt_native if dataset.latent_method == "qwen_raw" else _decode_qwen(codec, gt_native)
        feature_acc.update(gt_features, predicted_features)
        chunk_index = (step - initial_context_frames) // chunk_size
        if chunk_index not in chunk_accumulators:
            chunk_accumulators[chunk_index] = (FidelityAccumulator(), FidelityAccumulator(), [])
        chunk_features, chunk_actions, chunk_steps = chunk_accumulators[chunk_index]
        chunk_features.update(gt_features, predicted_features)
        chunk_steps.append(step)
        if action_expert is not None and ((step - initial_context_frames) % max(query_every, 1) == 0):
            pred_action = action_expert.actions(str(record["task"]), str(record.get("outcome", "success")), payload["states"][step].float(), predicted_features)
            gt_action = action_expert.actions(str(record["task"]), str(record.get("outcome", "success")), payload["states"][step].float(), gt_features)
            action_acc.update(gt_action, pred_action)
            chunk_actions.update(gt_action, pred_action)
            predicted_actions.append(pred_action.detach().cpu())
            reference_actions.append(gt_action.detach().cpu())
    result = {
        "episode_id": f"{record['dataset']}_{int(record['episode_index']):06d}",
        "dataset": record["dataset"], "episode_index": int(record["episode_index"]),
        "task": record["task"], "task_id": int(record.get("task_id", -1)),
        "source_outcome": record.get("outcome"), "initial_context_frames": initial_context_frames,
        "predicted_frames": last - initial_context_frames + 1, "action_query_frames": len(predicted_actions),
        "qwen_feature_metrics": feature_acc.report(), "action_metrics": action_acc.report(),
        "step_time": step_time_report(step_seconds, dataset.latent_method, initial_context_frames=initial_context_frames),
        "step_seconds": step_seconds,
        "chunk_size": chunk_size,
        "chunks": [{"chunk_index": key, "first_predicted_frame": indices[0],
                    "last_predicted_frame": indices[-1], "predicted_frames": len(indices),
                    "complete_chunk": len(indices) == chunk_size,
                    "qwen_feature_metrics": features.report(), "action_metrics": actions.report()}
                   for key, (features, actions, indices) in sorted(chunk_accumulators.items())],
        "elapsed_seconds": time.perf_counter() - started,
    }
    ref = torch.cat(reference_actions) if reference_actions else torch.empty(0)
    pred = torch.cat(predicted_actions) if predicted_actions else torch.empty(0)
    return result, feature_acc, action_acc, ref, pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, default=None)
    parser.add_argument("--policy-bundle", type=Path, default=None)
    parser.add_argument("--raw-data-root", type=Path, default=None)
    parser.add_argument("--qwen-model", type=Path, default=None)
    parser.add_argument("--latent-method", default="auto")
    parser.add_argument("--profile", choices=("auto", "robotwin"), default="auto")
    parser.add_argument("--action-schema", choices=("robotwin14",), default="robotwin14")
    parser.add_argument("--reference", choices=("auto", "cached", "raw_teacher", "codec_gt"), default="auto")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--outcomes", default="all")
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--views", default=None)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--selection", choices=("sequential", "balanced_task_outcome"), default="balanced_task_outcome")
    parser.add_argument("--timing-warmup-steps", type=int, default=5)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--episode-ids", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--initial-context-frames", type=int, choices=(1, 4), default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--query-every", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.chunk_size < 1: parser.error("--chunk-size must be positive")
    if args.timing_warmup_steps < 0: parser.error("--timing-warmup-steps must be nonnegative")
    torch.manual_seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Separate partial results from the final metrics file.
    progress_path = args.output.with_suffix(".episodes.jsonl")
    if progress_path.exists():
        progress_path.replace(progress_path.with_name(progress_path.name + f".{time.time_ns()}.bak"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    from tokenworld.data.dataset import RobotwinSVAETokenDataset
    from tokenworld.evaluation.rollout import WorldModelRollout
    spec = json.loads((args.cache / "cache_spec.json").read_text())
    latent_method = spec["latent_method"] if args.latent_method == "auto" else args.latent_method
    if latent_method not in spec["latent_specs"]: raise ValueError(f"Latent method {latent_method!r} is not present in cache")
    views = None if args.views is None else [int(v) for v in args.views.split(",") if v.strip()]
    outcomes = None if args.outcomes == "all" else args.outcomes
    dataset = RobotwinSVAETokenDataset(
        args.cache, raw_data_path=args.raw_data_root, split=args.split,
        num_hist=args.initial_context_frames, num_pred=1, latent_method=latent_method,
        tasks=args.tasks, outcomes=outcomes, views=views,
        normalize_latents=True, latent_stats_scope="per_view", normalize_action=True,
        verify_action_equals_state=True, sequence_interval=1, window_stride=1, tail_align=False,
        action_schema=args.action_schema,
    )
    profile = _infer_profile(dataset, args.profile)
    if latent_method == "qwen_raw":
        reference = "cached" if args.reference == "auto" else args.reference
        codec = None
    else:
        reference = "raw_teacher" if args.reference == "auto" else args.reference
        if args.codec_checkpoint is None: raise ValueError("--codec-checkpoint is required for S-VAE latent evaluation")
        from tokenworld.codec import common as codec_common
        codec, codec_payload = codec_common.load_codec_checkpoint(args.codec_checkpoint, device=device)
        codec.eval()
    teacher = None
    if reference == "raw_teacher":
        if args.qwen_model is None: raise ValueError("--qwen-model is required for raw_teacher")
        if args.raw_data_root is None: raise ValueError("--raw-data-root is required for raw_teacher")
        teacher = _load_qwen_teacher(args.qwen_model, device)
    if reference == "codec_gt" and codec is None: raise ValueError("codec_gt requires an S-VAE codec")
    action_expert = ActionExpert(args.policy_bundle, profile, device) if args.policy_bundle else None
    world_model = WorldModelRollout(str(args.world_model_checkpoint), external_decoder_path=None, device=str(device), num_steps=args.sampling_steps)
    indices = list(range(len(dataset.records)))[args.start_episode:]
    if args.episode_ids:
        wanted = {int(x) for x in args.episode_ids.split(",") if x.strip()}
        indices = [i for i in indices if int(dataset.records[i]["episode_index"]) in wanted]
    indices = select_episode_indices(dataset.records, indices, args.max_episodes, args.selection, args.seed)
    selection_records = [{key: dataset.records[i][key] for key in ("dataset", "task", "outcome", "episode_index", "num_frames")} for i in indices]
    args.output.with_suffix(".selection.json").write_text(json.dumps({"selection": args.selection, "seed": args.seed, "episodes": selection_records}, indent=2) + "\n")
    normalize_features = None
    if teacher is not None or reference == "raw_teacher":
        from tokenworld.codec.common import normalize_qwen_image_features
        normalize_features = normalize_qwen_image_features
    feature_total, action_total = FidelityAccumulator(), FidelityAccumulator()
    episodes, refs, preds, all_step_seconds = [], [], [], []
    print(f"Evaluating {len(indices)} {args.split} episodes profile={profile} latent_method={latent_method} reference={reference} initial_context_frames={args.initial_context_frames} sampling_steps={args.sampling_steps}", flush=True)
    for number, index in enumerate(indices, 1):
        result, feature_acc, action_acc, ref, pred = _run_episode(
            dataset=dataset, record_index=index, world_model=world_model, codec=codec,
            teacher=teacher, action_expert=action_expert, normalize_features=normalize_features,
            device=device, max_steps=args.max_steps, query_every=args.query_every, reference_mode=reference, initial_context_frames=args.initial_context_frames, chunk_size=args.chunk_size,
        )
        feature_total.merge(feature_acc); action_total.merge(action_acc); episodes.append(result)
        all_step_seconds.extend(result["step_seconds"])
        if ref.numel(): refs.append(ref); preds.append(pred)
        with progress_path.open("a", encoding="utf-8") as progress:
            progress.write(json.dumps(_json_float(result), allow_nan=False) + "\n")
        partial = {"status": "running", "completed_episodes": len(episodes), "total_episodes": len(indices),
                   "qwen_features": feature_total.report(), "normalized_actions": action_total.report(),
                   "sampling_steps": args.sampling_steps, "initial_context_frames": args.initial_context_frames, "selection": args.selection,
                   "step_time": step_time_report(all_step_seconds, latent_method, args.timing_warmup_steps, args.initial_context_frames)}
        partial_path = args.output.with_suffix(".progress.json")
        partial_tmp = partial_path.with_suffix(".json.tmp")
        partial_tmp.write_text(json.dumps(partial, indent=2) + "\n")
        partial_tmp.replace(partial_path)
        print(f"[{number}/{len(indices)}] {result['episode_id']} feature_cos={result['qwen_feature_metrics'].get('row_cosine_similarity', float('nan')):.6f} action_cos={result['action_metrics'].get('row_cosine_similarity', float('nan')):.6f}", flush=True)
    if refs:
        all_ref, all_pred = torch.cat(refs), torch.cat(preds)
        action_report = _action_metrics(all_ref, all_pred)
        physical_report = _physical_action_metrics(all_ref, all_pred, action_expert.action_stats_path, profile)
    else:
        action_report, physical_report = {"count": 0}, None
    from tokenworld.evaluation.aggregate import macro_average
    summary = {
        "aggregate_macro": macro_average(episodes),
        "chunk_size": args.chunk_size,
        "version": 5, "initial_context_frames": args.initial_context_frames, "status": "complete", "seed": args.seed, "protocol": "canonical_tokenworld_open_loop_fidelity",
        "selection": args.selection, "selected_episodes": selection_records,
        "step_time": step_time_report(all_step_seconds, latent_method, args.timing_warmup_steps, args.initial_context_frames),
        "step_time_including_warmup": step_time_report(all_step_seconds, latent_method, initial_context_frames=args.initial_context_frames),
        "profile": profile, "latent_method": latent_method, "reference": reference,
        "cache": str(args.cache.resolve()), "world_model_checkpoint": str(args.world_model_checkpoint.resolve()),
        "codec_checkpoint": str(args.codec_checkpoint.resolve()) if args.codec_checkpoint else None,
        "policy_bundle": str(args.policy_bundle.resolve()) if args.policy_bundle else None,
        "action_feature_interface": "qwen3_merger_only_no_deepstack" if args.policy_bundle else None,
        "action_reference": "same_merger_only_policy_on_reference_features" if args.policy_bundle else None,
        "qwen_model": str(args.qwen_model.resolve()) if args.qwen_model else None,
        "raw_data_root": str(args.raw_data_root.resolve()) if args.raw_data_root else None,
        "views": dataset.view_indices, "max_steps": args.max_steps, "query_every": args.query_every,
        "split": args.split, "num_episodes": len(episodes), "sampling_steps": args.sampling_steps,
        "fidelity": {
            "native_latents_or_qwen_features": feature_total.report(),
            "qwen_features": feature_total.report(),
            "normalized_actions": action_report,
            "physical_actions": physical_report,
        },
        "aggregate_micro": {"qwen_features": feature_total.report(), "actions": action_report, "action_physical": physical_report},
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_float(summary), indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    args.output.with_suffix(".progress.json").unlink(missing_ok=True)
    print(json.dumps(summary["fidelity"], indent=2), flush=True)


if __name__ == "__main__":
    main()
