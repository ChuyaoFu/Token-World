#!/usr/bin/env python3
"""Closed-loop Robotwin replay with QwenOFT actions and RGB visualization.

This is an adapter around the already validated Robotwin replay contract.  It
reads the replay manifest/cache, queries the frozen action expert on the
current model-predicted frame, feeds the resulting actions back to
``WorldModelRollout`` and writes one closed-loop video plus a seven-state
thumbnail.

The open-loop replay entrypoints are intentionally not imported or modified.
Their manifest, cache, latent normalization and 15-step action alignment are
treated as read-only input contracts here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np


CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES = (6, 13)
QWEN_TO_CACHE = (0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--world-model-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path, required=True)
    parser.add_argument("--policy-statistics", type=Path)
    parser.add_argument("--adapter-readout", choices=("native_pre_norm", "legacy_post_norm"),
                        default="native_pre_norm", help="Hidden state consumed by the adapter policy action head")
    parser.add_argument("--adapter-checkpoint", type=Path, help="Optional latent-to-DeepStack adapter; default remains merger-only")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default=f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
    parser.add_argument("--num-denoise-steps", type=int, default=64)
    parser.add_argument("--cache-mode", choices=("gru", "full", "reset"), default="gru",
                        help="Temporal state retained across WM segments")
    parser.add_argument("--execution-horizon", type=int, default=15)
    parser.add_argument("--query-every", type=int, default=50)
    parser.add_argument("--gripper-mode", choices=("binary", "legacy_minmax"), default="binary",
                        help="Policy grippers are binary [0,1]; legacy_minmax reproduces the historical conversion bug")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--decode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("RANK", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def torch_load(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_entries(path: Path, max_episodes: int, shard_index: int, num_shards: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = list(payload["episodes"])
    if max_episodes > 0:
        entries = entries[:max_episodes]
    return entries[shard_index::num_shards]


def episode_name(entry: dict) -> str:
    return (
        f"{int(entry['manifest_index']):02d}_{entry['outcome']}_"
        f"{entry['task']}_{entry['dataset']}_{int(entry['episode_index']):06d}"
    )


def load_latent_stats(cache_dir: Path, video_key: str, device):
    import torch

    stats = torch_load(cache_dir / "train_stats.pt")
    by_view = stats.get("latent_by_view", {}).get("symmetric_qwen", {})
    if video_key in by_view:
        mean = by_view[video_key]["mean"].float()
        std = by_view[video_key]["std"].float().clamp_min(1.0e-6)
    else:
        global_stats = stats["symmetric_qwen"]
        mean = global_stats["mean"].float()
        std = global_stats["std"].float().clamp_min(1.0e-6)
    action_mean = stats["action"]["mean"].float()
    action_std = stats["action"]["std"].float().clamp_min(1.0e-6)
    return mean.to(device), std.to(device), action_mean.to(device), action_std.to(device)


def _load_state_dict(path: Path) -> dict:
    import torch

    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected policy checkpoint payload: {type(state)!r}")
    return {key[7:] if key.startswith("module.") else key: value for key, value in state.items()}


def load_policy(bundle: Path, device):
    from omegaconf import OmegaConf

    starvla_root = bundle / "code" / "starvla"
    if str(starvla_root) not in sys.path:
        sys.path.insert(0, str(starvla_root))
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.framework.share_tools import apply_config_compat

    config_path = bundle / "model" / "config.full.yaml"
    checkpoints = sorted((bundle / "model" / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No policy checkpoint under {bundle / 'model' / 'checkpoints'}")
    config = apply_config_compat(OmegaConf.load(config_path))
    if os.environ.get("QWEN_MODEL"):
        config.framework.qwenvl.base_vlm = os.environ["QWEN_MODEL"]
    policy = build_framework(config)
    incompatible = policy.load_state_dict(_load_state_dict(checkpoints[-1]), strict=False)
    missing = [key for key in incompatible.missing_keys if "num_batches_tracked" not in key]
    unexpected = [key for key in incompatible.unexpected_keys if "num_batches_tracked" not in key]
    if missing or unexpected:
        raise RuntimeError(f"Action checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}")
    policy.to(device).eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy, config_path, checkpoints[-1]


def action_instruction(policy, task: str, outcome: str) -> str:
    tokens = policy.action_token * policy.chunk_len
    return (
        f"{task}\nTrajectory outcome: {outcome}."
        f" Please predict the next {policy.chunk_len} robot actions: "
        f"<action>{tokens}<action>."
    )


def build_policy_template(policy, task: str, outcome: str) -> dict:
    from PIL import Image

    dummy = Image.new("RGB", (384, 288))
    inputs = policy.qwen_vl_interface.build_qwenvl_inputs(
        images=[[dummy]],
        instructions=[action_instruction(policy, task, outcome)],
    )
    return {key: inputs[key] for key in ("input_ids", "attention_mask", "image_grid_thw")}


def policy_actions(policy, template: dict, features):
    import torch

    causal_model = policy.qwen_vl_interface.model
    backbone = causal_model.model
    input_ids = template["input_ids"].to(next(policy.parameters()).device)
    attention_mask = template["attention_mask"].to(input_ids.device)
    image_grid_thw = template["image_grid_thw"].to(input_ids.device)
    embeds = causal_model.get_input_embeddings()(input_ids)
    flat_features = features.reshape(-1, features.shape[-1]).to(
        device=embeds.device, dtype=embeds.dtype
    )
    image_mask, _ = backbone.get_placeholder_mask(
        input_ids, inputs_embeds=embeds, image_features=flat_features
    )
    embeds = embeds.masked_scatter(image_mask, flat_features)
    try:
        positions, _ = backbone.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
    except TypeError:
        positions = None
    kwargs = {
        "inputs_embeds": embeds,
        "attention_mask": attention_mask,
        "use_cache": False,
        "output_hidden_states": False,
        "return_dict": True,
    }
    if positions is not None:
        kwargs["position_ids"] = positions
    with torch.autocast(
        device_type=embeds.device.type,
        dtype=torch.bfloat16,
        enabled=embeds.device.type == "cuda",
    ):
        output = causal_model.model(**kwargs)
    queries = policy._gather_action_token_embeddings(
        output.last_hidden_state, input_ids, action_token_id=policy.action_token_id
    )
    with torch.autocast(
        device_type=embeds.device.type,
        dtype=torch.float32,
        enabled=embeds.device.type == "cuda",
    ):
        return policy.action_model.predict_action(queries).float()


def policy_to_cache_order(normalized_qwen, statistics: dict, *, gripper_mode="binary"):
    """Invert policy training normalization: min/max joints and binary grippers."""
    import torch

    action_stats = next(iter(statistics.values()))["action"]
    minimum = torch.as_tensor(action_stats["min"], dtype=torch.float32, device=normalized_qwen.device)
    maximum = torch.as_tensor(action_stats["max"], dtype=torch.float32, device=normalized_qwen.device)
    if minimum.numel() != 14 or maximum.shape != minimum.shape:
        raise ValueError("Robotwin policy statistics must contain 14 action dimensions")
    physical_qwen = (normalized_qwen + 1.0) * 0.5 * (maximum - minimum) + minimum
    if gripper_mode == "binary":
        physical_qwen[..., 12:] = (normalized_qwen[..., 12:] > 0.49).to(physical_qwen.dtype)
    elif gripper_mode != "legacy_minmax":
        raise ValueError(f"Unknown gripper mode: {gripper_mode}")
    return physical_qwen[..., list(QWEN_TO_CACHE)]


def action_to_model_inputs(actions, action_mean, action_std):
    normalized = (actions - action_mean.reshape(1, -1)) / action_std.reshape(1, -1)
    continuous = normalized[..., list(CONTINUOUS_INDICES)]
    discrete = (actions[..., list(GRIPPER_INDICES)] > 0.5).long()
    return discrete, continuous


def decode_rgb_frames(codec, rgb_decoder, raw_latents, device, batch_size: int) -> np.ndarray:
    import torch

    frames = []
    raw_latents = raw_latents.float()
    for start in range(0, len(raw_latents), max(1, batch_size)):
        chunk = raw_latents[start : start + max(1, batch_size)].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            decoded = codec.decode(chunk, [int(chunk.shape[1])] * len(chunk))
            vlm_tokens = torch.stack(list(decoded), dim=0)
            output = rgb_decoder(vlm_tokens)
            pixels = output[0] if isinstance(output, tuple) else output
        pixels = pixels.detach().float().cpu()
        if float(pixels.min()) < 0.0:
            pixels = (pixels + 1.0) / 2.0
        pixels = pixels.clamp(0, 1).permute(0, 2, 3, 1).mul(255).round().to(torch.uint8).numpy()
        frames.append(pixels)
    if not frames:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
    return np.concatenate(frames, axis=0)


def write_video(frames: np.ndarray, path: Path, fps: int) -> None:
    import cv2

    if len(frames) == 0:
        raise ValueError(f"No frames to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_thumbnail(frames: np.ndarray, path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    count = len(frames)
    if count == 0:
        return
    indices = np.linspace(0, count - 1, 7).round().astype(int).tolist()
    width, height, gap, label = 320, 256, 6, 120
    canvas = Image.new("RGB", (label + 7 * width + 6 * gap, height + 40), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    y = 20
    draw.text((8, y + height // 2), "Closed-loop", fill=(0, 0, 0), font=font)
    for col, index in enumerate(indices):
        frame = Image.fromarray(frames[index]).resize((width, height))
        canvas.paste(frame, (label + col * (width + gap), y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def run_episode(entry, args, *, policy, codec, world_model, rgb_decoder, latent_mean, latent_std, action_mean, action_std, policy_statistics, device, adapter=None) -> dict:
    import torch

    cache_path = args.cache / entry["cache_path"]
    payload = torch_load(cache_path)
    raw_latents = payload["latents"]["symmetric_qwen"].float()
    if raw_latents.ndim != 4 or raw_latents.shape[1] < 1 or tuple(raw_latents.shape[2:]) != (108, 16):
        raise ValueError(f"Expected [T,V,108,16] latents, got {tuple(raw_latents.shape)}")
    raw_latents = raw_latents[:, :1]
    raw_actions = payload["actions"].float()
    raw_states = payload["states"].float()
    total_frames = min(int(entry["num_frames"]), len(raw_latents), len(raw_actions), len(raw_states))
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    total_steps = total_frames - 1
    if total_steps < 1:
        raise ValueError(f"Episode {cache_path} has no transition")

    out = args.output_root / "episodes" / episode_name(entry)
    if (
        args.skip_existing
        and (out / "metadata.json").is_file()
        and (out / "videos" / "closed_loop.mp4").is_file()
        and (out / "thumbnail.png").is_file()
    ):
        saved = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        if saved.get("cache_mode", "reset") != args.cache_mode:
            raise ValueError(f"Cached episode used a different temporal cache mode: {out}")
        saved_mode = saved.get("gripper_mode", "legacy_minmax")
        if saved_mode != args.gripper_mode:
            raise ValueError(f"Existing output uses gripper_mode={saved_mode}, requested {args.gripper_mode}; use a new output root")
        requested_adapter = getattr(args, "adapter_checkpoint", None)
        requested_adapter = str(requested_adapter.resolve()) if requested_adapter else None
        if saved.get("adapter_checkpoint") != requested_adapter:
            raise ValueError("Existing output uses a different adapter; use a new output root")
        if requested_adapter:
            if saved.get("adapter_readout", "legacy_post_norm") != args.adapter_readout:
                raise ValueError("Existing output uses a different adapter readout; use a new output root")
            from tokenworld.utils.io import sha256_file
            if saved.get("adapter_sha256") != sha256_file(Path(requested_adapter)):
                raise ValueError("Adapter checkpoint content changed; use a new output root")
        return saved
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed + int(entry["cache_record_index"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    current_raw = raw_latents[0].to(device)  # [V,108,16]
    current_latent = ((current_raw - latent_mean.reshape(1, 1, -1)) / latent_std.reshape(1, 1, -1)).unsqueeze(0)
    current_proprio = ((raw_states[0].to(device) - action_mean) / action_std).unsqueeze(0)
    template = build_policy_template(policy, str(entry["task"]), "success")
    policy_chunk = None
    chunk_offset = 0
    query_count = 0
    frame_index = 0
    # Keep the view axis on the initial frame so it concatenates with the
    # [steps, view, tokens, channels] chunks returned by WorldModelRollout.
    predicted = [raw_latents[0:1].cpu()]
    used_actions = []
    query_trace = []
    time_cache = None
    while frame_index < total_steps:
        if policy_chunk is None or chunk_offset >= args.query_every:
            raw_for_codec = current_latent[0] * latent_std.reshape(1, 1, -1) + latent_mean.reshape(1, 1, -1)
            decoded = codec.decode(raw_for_codec, [108])
            features = torch.stack(list(decoded), dim=0).unsqueeze(0)
            if adapter is None:
                normalized_qwen = policy_actions(policy, template, features)
            else:
                from tokenworld.codec.deepstack_adapter import inject_actions
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # Match the pilot's bf16 feature cache; preserve raw latent coordinates.
                    merger = features.reshape(1, 108, 2560).to(torch.bfloat16).float()
                    deep = adapter.features(raw_for_codec.to(torch.bfloat16).float(), merger)[0]
                    normalized_qwen = inject_actions(policy, template, merger, deep, readout=args.adapter_readout)
            policy_chunk = policy_to_cache_order(normalized_qwen, policy_statistics, gripper_mode=args.gripper_mode)[0].detach()
            if policy_chunk.ndim != 2 or policy_chunk.shape[1] != 14:
                raise ValueError(f"Malformed action expert output: {tuple(policy_chunk.shape)}")
            if args.query_every > policy_chunk.shape[0]:
                raise ValueError(f"query-every={args.query_every} exceeds action horizon={policy_chunk.shape[0]}")
            chunk_offset = 0
            query_count += 1
            query_trace.append({"query_index": query_count - 1, "frame": frame_index})

        valid = min(args.execution_horizon, total_steps - frame_index, args.query_every - chunk_offset)
        action_count = valid + 1
        action_slice = policy_chunk[chunk_offset : chunk_offset + action_count]
        if len(action_slice) < action_count:
            action_slice = torch.cat((action_slice, action_slice[-1:].expand(action_count - len(action_slice), -1)), dim=0)
        if args.cache_mode == "reset" and len(action_slice) < args.execution_horizon + 1:
            action_slice = torch.cat((action_slice, action_slice[-1:].expand(args.execution_horizon + 1 - len(action_slice), -1)), dim=0)
        discrete, continuous = action_to_model_inputs(action_slice, action_mean, action_std)
        torch.manual_seed(seed + frame_index)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            result = world_model.rollout(
                current_latent,
                discrete_actions=discrete.unsqueeze(0),
                continuous_actions=continuous.unsqueeze(0),
                init_proprio=current_proprio,
                time_cache=time_cache,
                return_time_cache=args.cache_mode != "reset",
                gru_only_boundary=args.cache_mode == "gru",
            )
        time_cache = result.time_cache
        if result.latents.shape[1] != len(action_slice):
            raise RuntimeError(f"Expected {len(action_slice)} latent frames, got {tuple(result.latents.shape)}")
        if result.proprio is None:
            raise RuntimeError("TokenWorld returned no predicted proprio in closed-loop replay")
        next_norm = result.latents[:, 1 : valid + 1].float()
        if next_norm.ndim == 4:
            next_norm = next_norm.unsqueeze(2)
        next_raw = (next_norm * latent_std.reshape(1, 1, 1, -1) + latent_mean.reshape(1, 1, 1, -1))[0].cpu()
        predicted.append(next_raw)
        used_actions.append(action_slice[:valid].cpu())
        frame_index += valid
        chunk_offset += valid
        current_latent = result.latents[:, valid].detach()
        current_proprio = result.proprio[:, valid].detach()

    predicted_raw = torch.cat(predicted, dim=0)[:total_frames].float()
    candidate_actions = torch.cat(used_actions, dim=0).numpy().astype("float32")
    # The replay contract aligns action ``a_t`` with the transition from
    # frame ``t`` to frame ``t+1``; keep the same alignment for diagnostics.
    recorded_actions = raw_actions[: len(candidate_actions)].numpy().astype("float32")
    model_frames = decode_rgb_frames(codec, rgb_decoder, predicted_raw[:, 0], device, args.decode_batch_size)
    write_video(model_frames, out / "videos" / "closed_loop.mp4", args.fps)
    write_thumbnail(model_frames, out / "thumbnail.png")
    np.save(out / "actions_closed_loop.npy", candidate_actions)
    np.save(out / "actions_recorded.npy", recorded_actions)
    torch.save(predicted_raw, out / "predicted_svae_latents.pt")
    metadata = {
        "protocol": "robotwin_tokenworld_svae_qwenoft_closed_loop_replay",
        "manifest_entry": entry,
        "frames": int(len(predicted_raw)),
        "predicted_steps": int(len(candidate_actions)),
        "policy_queries": int(query_count),
        "query_trace": query_trace,
        "cache_mode": args.cache_mode,
        "execution_horizon": int(args.execution_horizon),
        "query_every": int(args.query_every),
        "gripper_mode": args.gripper_mode,
        "action_feature_interface": "adapter_deepstack" if adapter is not None else "merger_only",
        "num_denoise_steps": int(args.num_denoise_steps),
        "world_model_checkpoint": str(args.world_model_checkpoint.resolve()),
        "codec_checkpoint": str(args.codec_checkpoint.resolve()),
        "decoder_checkpoint": str(args.decoder_checkpoint.resolve()),
        "policy_bundle": str(args.policy_bundle.resolve()),
        "cache": str(cache_path.resolve()),
        "video": str((out / "videos" / "closed_loop.mp4").resolve()),
        "thumbnail": str((out / "thumbnail.png").resolve()),
        "action_order": "cache=[left6,left_gripper,right6,right_gripper]; policy=[left6,right6,left_gripper,right_gripper]",
        "initial_frame_is_ground_truth": True,
    }
    if adapter is not None:
        from tokenworld.utils.io import sha256_file
        metadata["adapter_readout"] = args.adapter_readout
        metadata["adapter_checkpoint"] = str(args.adapter_checkpoint.resolve())
        metadata["adapter_sha256"] = sha256_file(args.adapter_checkpoint)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard configuration")
    if args.execution_horizon < 1 or args.query_every < 1 or args.decode_batch_size < 1:
        raise ValueError("execution/query/decode batch sizes must be positive")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for closed-loop Robotwin replay")
    import torch

    device = torch.device(args.device)
    for path in (args.manifest, args.cache / "cache_spec.json", args.cache / "train_stats.pt", args.world_model_checkpoint, args.codec_checkpoint, args.decoder_checkpoint, args.policy_bundle):
        if not path.exists():
            raise FileNotFoundError(path)
    spec = json.loads((args.cache / "cache_spec.json").read_text(encoding="utf-8"))
    video_key = str(spec["video_keys"][0])
    latent_mean, latent_std, action_mean, action_std = load_latent_stats(args.cache, video_key, device)
    policy_statistics_path = args.policy_statistics or (args.policy_bundle / "model" / "dataset_statistics.json")
    policy_statistics = json.loads(policy_statistics_path.read_text(encoding="utf-8"))

    from tokenworld.decoder.checkpoint import load_external_decoder
    from tokenworld.evaluation.rollout import WorldModelRollout

    from tokenworld.codec.common import load_codec_checkpoint

    policy, policy_config, policy_checkpoint = load_policy(args.policy_bundle.resolve(), device)
    codec, codec_payload = load_codec_checkpoint(args.codec_checkpoint.resolve(), device=device)
    codec.eval()
    rgb_decoder = load_external_decoder(str(args.decoder_checkpoint.resolve()), device=str(device))
    rgb_decoder.eval().requires_grad_(False)
    world_model = WorldModelRollout(
        str(args.world_model_checkpoint.resolve()),
        device=str(device),
        num_steps=args.num_denoise_steps,
    )
    adapter = None
    if args.adapter_checkpoint is not None:
        from tokenworld.codec.deepstack_adapter import DeepStackAdapter
        adapter = DeepStackAdapter().to(device)
        checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
        adapter.load_state_dict(checkpoint["model"], strict=True)
        adapter.eval().requires_grad_(False)
    entries = load_entries(args.manifest, args.max_episodes, args.shard_index, args.num_shards)
    if not entries:
        raise ValueError("No manifest entries selected for this shard")
    reports = []
    for index, entry in enumerate(entries, start=1):
        print(f"[{args.shard_index}] [{index}/{len(entries)}] {episode_name(entry)}", flush=True)
        reports.append(run_episode(entry, args, policy=policy, codec=codec, world_model=world_model, rgb_decoder=rgb_decoder, latent_mean=latent_mean, latent_std=latent_std, action_mean=action_mean, action_std=action_std, policy_statistics=policy_statistics, device=device, adapter=adapter))
    summary = {"cache_mode": args.cache_mode, "protocol": "robotwin_tokenworld_svae_qwenoft_closed_loop_replay", "policy_config": str(policy_config), "policy_checkpoint": str(policy_checkpoint), "codec_payload": codec_payload.get("codec_config", {}), "episodes": reports}
    summary_path = args.output_root / f"closed_loop_summary.rank{args.shard_index:02d}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
