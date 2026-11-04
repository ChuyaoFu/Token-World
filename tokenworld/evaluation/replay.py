#!/usr/bin/env python3
"""RobotWin TokenWorld recorded-action replay with raw S-VAE latents saved.

Saved tensors are *denormalized* symmetric-Qwen S-VAE tokens, shape
``[frames, 1, 108, 16]`` for the current cam_high view.  Frame zero is the
ground-truth cache token; all following frames are autoregressive predictions.
The codec checkpoint and normalization metadata are recorded beside each
tensor so it can be decoded later without rerunning the replay.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


LOG = logging.getLogger("tokenworld-replay")
CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_INDICES = (6, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cache-mode", choices=("gru", "full", "reset"), default="gru",
                        help="Temporal state retained across WM segments")
    parser.add_argument("--execution-horizon", type=int, default=15)
    parser.add_argument("--num-denoise-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument(
        "--device",
        default=f"cuda:{os.environ.get('LOCAL_RANK', '0')}",
        help="Device for this worker; defaults to cuda:$LOCAL_RANK under torchrun.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("RANK", "0")),
        help="Worker shard index; defaults to RANK under torchrun.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="Total number of episode shards; defaults to WORLD_SIZE under torchrun.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip episodes with existing latent and metadata outputs.",
    )
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_manifest(
    path: Path,
    max_episodes: int,
    shard_index: int,
    num_shards: int,
) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        entries = json.load(handle)["episodes"]
    if max_episodes > 0:
        entries = entries[:max_episodes]
    return entries[shard_index::num_shards]


def recorded_action_chunk(
    actions: torch.Tensor,
    frame_index: int,
    horizon: int,
    total_frames: int,
) -> tuple[torch.Tensor, int]:
    valid = min(horizon, total_frames - 1 - frame_index)
    # WorldModelRollout.rollout expects one action aligned with the current
    # conditioning frame plus one action for every generated destination
    # frame: [a_t, a_{t+1}, ..., a_{t+horizon}].
    chunk = actions[frame_index : frame_index + valid + 1].float()
    if valid <= 0 or len(chunk) != valid + 1:
        raise ValueError(f"Invalid recorded-action chunk: valid={valid}, shape={chunk.shape}")
    if valid < horizon:
        chunk = torch.cat((chunk, chunk[-1:].expand(horizon - valid, -1)), dim=0)
    # Add the current-frame action to the padded horizon as well.
    if len(chunk) != horizon + 1:
        chunk = torch.cat((chunk, chunk[-1:].expand(horizon + 1 - len(chunk), -1)), dim=0)
    return chunk, valid


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(
            f"invalid sharding: shard_index={args.shard_index}, "
            f"num_shards={args.num_shards}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                args.output_dir / f"replay.rank{args.shard_index:02d}.log",
                mode="a",
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for path in (args.checkpoint, args.cache / "cache_spec.json", args.cache / "index.json", args.cache / "train_stats.pt"):
        if not path.exists():
            raise FileNotFoundError(path)
    from tokenworld.data.dataset import resolve_latent_normalization
    from tokenworld.evaluation.rollout import WorldModelRollout

    device = torch.device(args.device)
    with (args.cache / "cache_spec.json").open(encoding="utf-8") as handle:
        cache_spec = json.load(handle)
    with (args.cache / "train_stats.pt").open("rb") as handle:
        stats = torch.load(handle, map_location="cpu", weights_only=True)
    video_keys = list(cache_spec["video_keys"])
    view_index = video_keys.index("observation.images.cam_high")
    # Current TokenWorld checkpoint is configured with per-view statistics.
    latent_mean, latent_std, resolved_scope, stat_keys = resolve_latent_normalization(
        stats,
        latent_method="symmetric_qwen",
        video_keys=video_keys,
        view_indices=[view_index],
        scope="per_view",
    )
    latent_mean = latent_mean.to(device)
    latent_std = latent_std.to(device)
    action_mean = stats["action"]["mean"].float().to(device)
    action_std = stats["action"]["std"].float().clamp_min(1.0e-6).to(device)

    LOG.info("loading TokenWorld checkpoint: %s", args.checkpoint)
    rollout = WorldModelRollout(
        str(args.checkpoint),
        device=str(device),
        num_steps=args.num_denoise_steps,
        no_shortcut=False,
    )
    entries = load_manifest(
        args.manifest,
        args.max_episodes,
        args.shard_index,
        args.num_shards,
    )
    LOG.info(
        "replaying %d episodes on %s (shard %d/%d)",
        len(entries),
        device,
        args.shard_index,
        args.num_shards,
    )
    for entry_index, entry in enumerate(entries, start=1):
        payload = torch_load(args.cache / entry["cache_path"])
        raw_latents = payload["latents"]["symmetric_qwen"].float()
        raw_actions = payload["actions"].float()
        raw_states = payload["states"].float()
        if not torch.equal(raw_actions, raw_states):
            raise ValueError(f"cache action/state mismatch: {entry['cache_path']}")
        total_frames = min(
            int(entry["num_frames"]),
            len(raw_latents),
            len(raw_actions),
            len(raw_states),
        )
        if args.max_frames > 0:
            total_frames = min(total_frames, args.max_frames)
        if total_frames < 2:
            LOG.warning("skip %s: only %d frame(s)", entry["cache_path"], total_frames)
            continue

        episode_name = (
            f"{entry['manifest_index']:02d}_{entry['outcome']}_"
            f"{entry['task']}_{entry['dataset']}_{entry['episode_index']:06d}"
        )
        episode_dir = args.output_dir / "tokenworld" / episode_name
        episode_dir.mkdir(parents=True, exist_ok=True)
        if (
            args.skip_existing
            and (episode_dir / "predicted_svae_latents.pt").is_file()
            and (episode_dir / "metadata.json").is_file()
        ):
            saved = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
            if saved.get("cache_mode", "reset") != args.cache_mode:
                raise ValueError(f"Cached episode used a different temporal cache mode: {episode_dir}")
            LOG.info("skip existing %s", episode_dir)
            continue
        seed = args.seed + int(entry["cache_record_index"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        initial_raw = raw_latents[0:1, view_index : view_index + 1].to(device)
        current_latent = ((initial_raw - latent_mean) / latent_std).unsqueeze(0)
        current_proprio = (
            (raw_states[0:1].to(device) - action_mean) / action_std
        ).unsqueeze(0)

        predicted_chunks: list[torch.Tensor] = [initial_raw.detach().cpu()]
        frame_index = 0
        replay_chunks = 0
        time_cache = None
        while frame_index < total_frames - 1:
            action_chunk, valid_steps = recorded_action_chunk(
                raw_actions, frame_index, args.execution_horizon, total_frames
            )
            if args.cache_mode != "reset":
                action_chunk = action_chunk[:valid_steps + 1]
            action_chunk = action_chunk.to(device)
            continuous = (
                action_chunk[..., list(CONTINUOUS_INDICES)] - action_mean[list(CONTINUOUS_INDICES)]
            ) / action_std[list(CONTINUOUS_INDICES)]
            discrete = (action_chunk[..., list(GRIPPER_INDICES)] > 0.5).long()
            torch.manual_seed(seed + replay_chunks)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                result = rollout.rollout(
                    current_latent,
                    discrete_actions=discrete.unsqueeze(0),
                    continuous_actions=continuous.unsqueeze(0),
                    init_proprio=current_proprio,
                    time_cache=time_cache,
                    return_time_cache=args.cache_mode != "reset",
                    gru_only_boundary=args.cache_mode == "gru",
                )
            time_cache = result.time_cache
            if result.latents.shape[1] != len(action_chunk):
                raise RuntimeError(
                    f"Expected {len(action_chunk)} latent frames, got {result.latents.shape}"
                )
            predicted_normalized = result.latents[:, 1 : valid_steps + 1].float()
            if predicted_normalized.ndim == 4:
                predicted_normalized = predicted_normalized.unsqueeze(2)
            predicted_raw = (predicted_normalized * latent_std + latent_mean)[0].detach().cpu()
            predicted_chunks.append(predicted_raw)

            frame_index += valid_steps
            replay_chunks += 1
            if frame_index < total_frames - 1:
                current_latent = result.latents[:, valid_steps : valid_steps + 1].detach()
                if result.proprio is None:
                    raise RuntimeError("TokenWorld returned no predicted proprio")
                current_proprio = result.proprio[:, valid_steps : valid_steps + 1].detach()
            del action_chunk, continuous, discrete, result, predicted_normalized, predicted_raw
            torch.cuda.empty_cache()

        sequence = torch.cat(predicted_chunks, dim=0)[:total_frames].float().contiguous()
        torch.save(sequence, episode_dir / "predicted_svae_latents.pt")
        torch.save(sequence[:, 0], episode_dir / "predicted_svae_latents_view0.pt")
        np.save(episode_dir / "predicted_svae_latents.npy", sequence.numpy())
        metadata = {
            "model": "TokenWorld",
            "checkpoint": str(args.checkpoint),
            "codec_checkpoint": str(args.codec_checkpoint),
            "codec_method": "symmetric_qwen",
            "manifest_entry": entry,
            "num_frames": int(sequence.shape[0]),
            "latent_shape_with_view": list(sequence.shape),
            "latent_shape_view0": list(sequence[:, 0].shape),
            "conditioning_frame_0_is_ground_truth": True,
            "latent_normalization": {
                "scope_requested": "per_view",
                "scope_resolved": resolved_scope,
                "stat_keys": stat_keys,
            },
            "num_denoise_steps": int(args.num_denoise_steps),
            "cache_mode": args.cache_mode,
            "execution_horizon": int(args.execution_horizon),
            "frame_rate_hz": float(cache_spec.get("export_frame_rate_hz", 50.0)),
            "view": "cam_high",
            "view_index": int(view_index),
            "cache": str(args.cache),
            "replay_chunks": int(replay_chunks),
        }
        with (episode_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        LOG.info(
            "[%d/%d] saved %s (%d frames, shape=%s)",
            entry_index,
            len(entries),
            episode_dir,
            sequence.shape[0],
            tuple(sequence.shape),
        )


if __name__ == "__main__":
    main()
