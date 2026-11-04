"""Train a standalone RobotWin decoder from S-VAE latents.

This version is aligned with the TokenWorld S-VAE pipeline:
1. Loads RobotWin S-VAE latents lazily
2. Caches the decoder-facing latent tensors to disk per episode
3. Trains a single RGB decoder on those latents and raw images
4. Deletes the generated feature cache after successful training unless retained
"""

from __future__ import annotations

import argparse
import gc
import importlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tokenworld.utils.checkpoint_compat import load_checkpoint_payload
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm
from accelerate import Accelerator


from tokenworld.data.dataset import load_robotwin_svae_train_val
from tokenworld.codec.decoder import load_symmetric_qwen_decoder

try:
    from tokenworld.decoder.metrics.image import psnr, ssim
except Exception:  # pragma: no cover
    psnr = None
    ssim = None

try:
    from tokenworld.decoder.metrics.lpips import lpips
except Exception:  # pragma: no cover
    lpips = None


from tokenworld.utils.config import DictConfig


def load_yaml_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return DictConfig(yaml.safe_load(handle))


def to_dict(obj):
    if isinstance(obj, DictConfig):
        return {key: to_dict(value) for key, value in obj.__dict__.items()}
    return obj


def save_yaml_config(config, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False)


def create_decoder_from_config(decoder_config):
    target = decoder_config._target_
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    decoder_class = getattr(module, class_name)
    params = {k: v for k, v in decoder_config.__dict__.items() if k != "_target_"}
    return decoder_class(**params)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _loader_kwargs(cfg, split: str, batch_size: int) -> dict:
    default_workers = 0 if split == "val" else 4
    num_workers = int(
        cfg.data.get("num_workers", cfg.training.get("num_workers", default_workers))
    )
    if split == "val":
        num_workers = int(
            cfg.data.get("val_num_workers", cfg.training.get("val_num_workers", num_workers))
        )
    kwargs = {
        "batch_size": batch_size,
        "shuffle": split == "train",
        "num_workers": num_workers,
        "drop_last": bool(cfg.training.get("drop_last", True)) if split == "train" else bool(cfg.training.get("val_drop_last", False)),
    }
    if num_workers > 0:
        kwargs["pin_memory"] = bool(cfg.data.get("pin_memory", True))
        kwargs["persistent_workers"] = bool(cfg.data.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 2))
    return kwargs


def create_datasets(cfg, latent_cache_root):
    svae_cache_dir = cfg.data.get("svae_cache_dir", cfg.data.get("data_path"))
    raw_data_path = cfg.data.get("raw_data_path")
    if svae_cache_dir is None:
        raise ValueError("data.svae_cache_dir (or data.data_path) is required")
    if raw_data_path is None:
        raise ValueError("data.raw_data_path is required for decoder training")

    datasets, _ = load_robotwin_svae_train_val(
        cache_dir=svae_cache_dir,
        raw_data_path=raw_data_path,
        latent_cache_dir=latent_cache_root,
        num_hist=cfg.data.num_hist,
        num_pred=cfg.data.num_pred,
        latent_method=cfg.data.get("latent_method", "symmetric_qwen"),
        tasks=cfg.data.get("tasks"),
        outcomes=cfg.data.get("outcomes"),
        views=cfg.data.get("views"),
        normalize_latents=cfg.data.get("normalize_latents", True),
        normalize_action=cfg.data.get("normalize_action", True),
        verify_action_equals_state=cfg.data.get("verify_action_equals_state", True),
        load_raw_images=True,
        random_window_per_episode=cfg.data.get("random_window_per_episode", True),
        random_stride_min=cfg.data.get("random_stride_min", 1),
        random_stride_max=cfg.data.get("random_stride_max", 8),
        train_sequence_interval=cfg.data.get("train_sequence_interval", 1),
        train_window_stride=cfg.data.get("train_window_stride", 1),
        val_sequence_interval=cfg.data.get("val_sequence_interval", 1),
        val_window_stride=cfg.data.get("val_window_stride"),
        tail_align=cfg.data.get("tail_align", True),
        lru_episodes=cfg.data.get("lru_episodes", 8),
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["valid"]

    expected_emb_dim = int(cfg.decoder.emb_dim)
    expected_grid = tuple(map(int, cfg.decoder.grid_size))
    actual_grid = (int(train_dataset.grid_height), int(train_dataset.grid_width))
    if expected_emb_dim != int(train_dataset.decoder_feature_dim):
        raise ValueError(
            f"decoder.emb_dim={expected_emb_dim} does not match decoded VLM "
            f"feature dimension {train_dataset.decoder_feature_dim}"
        )
    if expected_grid != actual_grid:
        raise ValueError(
            f"decoder.grid_size={expected_grid} does not match latent grid {actual_grid}"
        )

    return train_dataset, val_dataset


def materialize_decoder_caches(cfg, datasets, accelerator):
    """Decode compressed S-VAE tokens to disk before constructing RGB loaders."""

    train_dataset, val_dataset = datasets
    checkpoint = cfg.data.get("codec_checkpoint") or train_dataset.spec.get("codec_checkpoint")
    if checkpoint is None:
        raise ValueError(
            "Missing data.codec_checkpoint and cache_spec.codec_checkpoint; "
            "the decoded VLM feature cache cannot be generated"
        )
    checkpoint = Path(checkpoint).expanduser().resolve()
    if accelerator.is_main_process:
        total_frames = sum(
            int(record["num_frames"])
            for dataset in (train_dataset, val_dataset)
            for record in dataset.records
        )
        estimated_gib = (
            total_frames
            * train_dataset.num_views
            * train_dataset.tokens_per_view
            * train_dataset.decoder_feature_dim
            * 2
            / (1024**3)
        )
        print(
            f"Decoded VLM cache estimate: {estimated_gib:.2f} GiB "
            f"({total_frames} frames, views={train_dataset.view_indices})"
        )
        print(f"S-VAE codec checkpoint: {checkpoint}")

    codec_decoder, codec_metadata = load_symmetric_qwen_decoder(checkpoint, accelerator.device)
    codec_feature_dim = int(codec_metadata["codec_config"]["input_dim"])
    if codec_feature_dim != int(train_dataset.decoder_feature_dim):
        raise ValueError(
            f"Codec output dim={codec_feature_dim} does not match cache spec "
            f"input_dim={train_dataset.decoder_feature_dim}"
        )
    train_dataset.decoder_cache_metadata = codec_metadata
    val_dataset.decoder_cache_metadata = codec_metadata
    decode_batch_size = int(cfg.data.get("codec_decode_batch_size", 32))
    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        rank_records = range(accelerator.process_index, len(dataset.records), accelerator.num_processes)
        rank_records = list(rank_records)
        progress = tqdm(
            rank_records,
            desc=f"Decode {split_name} VLM cache rank {accelerator.process_index}",
            disable=not accelerator.is_main_process,
        )
        for record_index in progress:
            dataset.materialize_decoder_episode(
                record_index,
                codec_decoder,
                accelerator.device,
                decode_batch_size=decode_batch_size,
                cache_dtype=torch.bfloat16,
                cache_metadata=codec_metadata,
                autocast_context=accelerator.autocast,
            )
        accelerator.wait_for_everyone()

    del codec_decoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Decoded VLM cache materialization complete; S-VAE decoder released")


def create_dataloaders(cfg, datasets, gpu_batch_size):
    train_dataset, val_dataset = datasets
    train_loader = DataLoader(train_dataset, **_loader_kwargs(cfg, "train", gpu_batch_size))
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, **_loader_kwargs(cfg, "val", gpu_batch_size))
    return train_dataset, val_dataset, train_loader, val_loader


def create_optimizer(decoder, cfg):
    return torch.optim.Adam(
        decoder.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.get("weight_decay", 0.0),
    )


def create_scheduler(optimizer, cfg, steps_per_epoch):
    scheduler_name = cfg.training.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.num_epochs * steps_per_epoch,
            eta_min=cfg.training.get("lr_min", 1.0e-6),
        )
    if scheduler_name == "step":
        return StepLR(
            optimizer,
            step_size=cfg.training.get("lr_step_size", 30) * steps_per_epoch,
            gamma=cfg.training.get("lr_gamma", 0.1),
        )
    return None


def compute_reconstruction_loss(pred_images, target_images):
    return F.mse_loss(pred_images.float(), target_images.float())


def compute_metrics(pred_images, target_images, lpips_fn=None):
    pred = torch.clamp((pred_images.float() + 1) / 2, 0, 1)
    target = torch.clamp((target_images.float() + 1) / 2, 0, 1)
    metrics = {
        "mse": F.mse_loss(pred, target).item(),
        "l1": F.l1_loss(pred, target).item(),
    }
    if psnr is not None:
        metrics["psnr"] = psnr(pred, target).mean().item()
    if ssim is not None:
        metrics["ssim"] = ssim(pred, target).mean().item()
    if lpips_fn is not None:
        with torch.no_grad():
            metrics["lpips"] = lpips_fn(pred_images.float(), target_images.float()).mean().item()
    return metrics


def visualize_reconstruction(target_images, pred_images, epoch, save_dir, num_samples=16):
    num_samples = min(num_samples, target_images.size(0))
    target = torch.clamp((target_images[:num_samples] + 1) / 2, 0, 1)
    pred = torch.clamp((pred_images[:num_samples] + 1) / 2, 0, 1)
    comparison = torch.cat([target, pred], dim=0)
    grid = vutils.make_grid(comparison, nrow=num_samples, padding=2, normalize=False)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"reconstruction_epoch_{epoch:04d}.png"
    vutils.save_image(grid, save_path)
    return grid, save_path


def save_checkpoint(decoder, optimizer, scheduler, epoch, cfg, metrics, accelerator, best_val_loss=float("inf"), is_best=False):
    Path(cfg.logging.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "decoder_state_dict": accelerator.unwrap_model(decoder).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
        "config": to_dict(cfg),
        "rng_state": {
            "torch": torch.get_rng_state().to(torch.uint8),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    filename = (
        f"decoder_best_epoch_{epoch:04d}.pt"
        if is_best
        else f"decoder_epoch_{epoch:04d}.pt"
    )
    path = Path(cfg.logging.checkpoint_dir) / filename
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(checkpoint_path, decoder, optimizer=None, scheduler=None, device="cuda", accelerator=None):
    checkpoint = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model = accelerator.unwrap_model(decoder) if accelerator is not None else decoder
    model.load_state_dict(checkpoint["decoder_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        try:
            torch_state = rng_state.get("torch")
            if torch_state is not None:
                if isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.to(torch.uint8)
                torch.set_rng_state(torch_state)
            if rng_state.get("torch_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            if rng_state.get("numpy") is not None:
                np.random.set_state(rng_state["numpy"])
        except Exception as exc:  # pragma: no cover
            print(f"Warning: could not restore RNG state: {exc}")

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    print(f"Resumed from epoch {checkpoint['epoch']}")
    return start_epoch, best_val_loss


def _flatten_decoder_batch(visual, target_images):
    if visual.ndim == 5:
        batch, time, views = visual.shape[:3]
        visual = visual.reshape(batch * time * views, *visual.shape[3:])
        target_images = target_images.reshape(batch * time * views, *target_images.shape[3:])
    elif visual.ndim == 4:
        batch, time = visual.shape[:2]
        visual = visual.reshape(batch * time, *visual.shape[2:])
        target_images = target_images.reshape(batch * time, *target_images.shape[2:])
    else:
        raise ValueError(f"Unexpected latent batch shape: {tuple(visual.shape)}")
    return visual, target_images


def _match_target_resolution(target_images, reference_images):
    target_size = reference_images.shape[-2:]
    if target_images.shape[-2:] == target_size:
        return target_images
    return F.interpolate(target_images.float(), size=target_size, mode="bilinear", align_corners=False)


def _move_decoder_batch_to_device(batch, device):
    obs, act, state, rewards, image = batch
    visual = obs["visual"].to(device, non_blocking=True)
    target_images = image.to(device, non_blocking=True)
    return visual, target_images


def train_epoch(decoder, dataloader, optimizer, scheduler, lpips_fn, cfg, epoch, accelerator):
    decoder.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_batches = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{cfg.training.num_epochs}", disable=not accelerator.is_main_process)
    for batch in pbar:
        visual, target_images = _move_decoder_batch_to_device(batch, accelerator.device)
        target_images = target_images * 2.0 - 1.0
        visual, target_images = _flatten_decoder_batch(visual, target_images)

        with accelerator.autocast():
            pred_images, diff = decoder(visual)
            target_images = _match_target_resolution(target_images, pred_images)
            recon_loss = compute_reconstruction_loss(pred_images, target_images)
            if bool(cfg.decoder.get("quantize", False)):
                loss = recon_loss + cfg.training.get("decoder_latent_loss_weight", 0.0) * diff
            else:
                loss = recon_loss

        optimizer.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        loss_value = accelerator.gather_for_metrics(loss.detach()).mean()
        recon_value = accelerator.gather_for_metrics(recon_loss.detach()).mean()
        total_loss += float(loss_value.item())
        total_recon_loss += float(recon_value.item())
        total_batches += 1
        pbar.set_postfix(
            loss=f"{loss_value.item():.4f}",
            recon=f"{recon_value.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

    return {
        "train/loss": total_loss / max(total_batches, 1),
        "train/recon_loss": total_recon_loss / max(total_batches, 1),
        "train/lr": optimizer.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(decoder, dataloader, lpips_fn, cfg, epoch, accelerator, visualize=True):
    decoder.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    metrics_sum = {}
    total_batches = 0
    vis_target = None
    vis_pred = None
    num_vis_samples = int(cfg.logging.get("num_vis_samples", 16))
    vis_interval = max(1, len(dataloader) // max(num_vis_samples, 1))

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation", disable=not accelerator.is_main_process)):
        visual, target_images = _move_decoder_batch_to_device(batch, accelerator.device)
        target_images = target_images * 2.0 - 1.0
        visual, target_images = _flatten_decoder_batch(visual, target_images)

        with accelerator.autocast():
            pred_images, diff = decoder(visual)
            target_images = _match_target_resolution(target_images, pred_images)
            recon_loss = compute_reconstruction_loss(pred_images, target_images)
            if bool(cfg.decoder.get("quantize", False)):
                loss = recon_loss + cfg.training.get("decoder_latent_loss_weight", 0.0) * diff
            else:
                loss = recon_loss

        total_loss += float(accelerator.gather_for_metrics(loss.detach()).mean().item())
        total_recon_loss += float(
            accelerator.gather_for_metrics(recon_loss.detach()).mean().item()
        )
        total_batches += 1

        batch_metrics = compute_metrics(pred_images, target_images, lpips_fn)
        for key, value in batch_metrics.items():
            value_tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
            value_global = accelerator.gather_for_metrics(value_tensor).mean().item()
            metrics_sum[key] = metrics_sum.get(key, 0.0) + float(value_global)

        if visualize and batch_idx % vis_interval == 0 and vis_target is None:
            vis_target = target_images[:num_vis_samples].cpu()
            vis_pred = pred_images[:num_vis_samples].cpu()

    avg_metrics = {
        "val/loss": total_loss / max(total_batches, 1),
        "val/recon_loss": total_recon_loss / max(total_batches, 1),
    }
    for key, value in metrics_sum.items():
        avg_metrics[f"val/{key}"] = value / max(total_batches, 1)

    grid = None
    if visualize and vis_target is not None and accelerator.is_main_process:
        vis_dir = Path(cfg.logging.checkpoint_dir) / "visualizations"
        grid, save_path = visualize_reconstruction(
            vis_target, vis_pred, epoch, vis_dir, num_vis_samples
        )
        print(f"Saved visualization to {save_path}")

    return avg_metrics, grid


def print_config(cfg):
    def recurse(obj, indent=0):
        for key, value in obj.__dict__.items():
            if isinstance(value, DictConfig):
                print(" " * indent + f"{key}:")
                recurse(value, indent + 2)
            else:
                print(" " * indent + f"{key}: {value}")

    print("=" * 80)
    print("Configuration:")
    recurse(cfg)
    print("=" * 80)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RobotWin decoder from S-VAE latents")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, default=None)
    parser.add_argument("--svae_cache_dir", type=str, default=None)
    parser.add_argument("--latent_cache_dir", type=str, default=None)
    parser.add_argument("--codec_checkpoint", type=str, default=None)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--prepare_metrics_only", action="store_true")
    parser.add_argument("--expected_processes", type=int, default=1)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--keep_feature_cache", action="store_true")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--outcomes", type=str, default=None)
    parser.add_argument("--views", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_num_workers", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def override_config_with_args(cfg, args):
    if args.codec_checkpoint is not None:
        cfg.data.codec_checkpoint = args.codec_checkpoint
    if args.raw_data_path is not None:
        cfg.data.raw_data_path = args.raw_data_path
    if args.svae_cache_dir is not None:
        cfg.data.svae_cache_dir = args.svae_cache_dir
    if args.latent_cache_dir is not None:
        cfg.data.latent_cache_dir = args.latent_cache_dir
    if args.tasks is not None:
        cfg.data.tasks = [value.strip() for value in args.tasks.split(",") if value.strip()]
    if args.outcomes is not None:
        cfg.data.outcomes = [value.strip() for value in args.outcomes.split(",") if value.strip()]
    if args.views is not None:
        cfg.data.views = [int(value.strip()) for value in args.views.split(",") if value.strip()]
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.val_num_workers is not None:
        cfg.data.val_num_workers = args.val_num_workers
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        cfg.training.num_epochs = args.num_epochs
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.checkpoint_dir is not None:
        cfg.logging.checkpoint_dir = args.checkpoint_dir
    if args.exp_name is not None:
        cfg.logging.exp_name = args.exp_name
    if args.no_wandb:
        cfg.logging.no_wandb = True
    if args.wandb_project is not None:
        cfg.logging.wandb_project = args.wandb_project
    if args.resume is not None:
        cfg.resume = args.resume
    return cfg


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    cfg = override_config_with_args(cfg, args)
    if args.prepare_metrics_only:
        if lpips is None:
            raise RuntimeError("LPIPS metric dependency is unavailable")
        with torch.inference_mode():
            value = lpips(torch.zeros(1, 3, 64, 64), torch.ones(1, 3, 64, 64))
        if not torch.isfinite(value).all():
            raise RuntimeError("LPIPS warmup returned non-finite values")
        print(f"LPIPS/AlexNet weights ready in {torch.hub.get_dir()}")
        return
    set_seed(cfg.seed)

    mixed_precision = cfg.training.get("mixed_precision", "no")
    log_with = None if cfg.logging.get("no_wandb", False) else "wandb"
    accelerator = Accelerator(mixed_precision=mixed_precision, log_with=log_with)
    device = accelerator.device

    if accelerator.is_main_process:
        print_config(cfg)

    process_count = args.expected_processes if args.check_only else accelerator.num_processes
    if args.smoke_test:
        if args.batch_size is None:
            cfg.training.batch_size = 4 * process_count
        cfg.training.num_epochs = 1
        cfg.data.num_workers = 0
        cfg.data.val_num_workers = 0
        cfg.logging.checkpoint_every = 1
        cfg.logging.visualize_every = 1
        cfg.logging.exp_name = "rgb_decoder_smoke_test"
        if cfg.get("resume", None):
            raise ValueError("Use a fresh output for --smoke_test; resume uses the full recipe")
    if process_count < 1 or cfg.training.batch_size < 1 or cfg.training.num_epochs < 1:
        raise ValueError("Process count, global batch size and epochs must be positive")
    assert cfg.training.batch_size % process_count == 0, (
        "Batch size must be divisible by number of processes"
    )
    gpu_batch_size = cfg.training.batch_size // process_count

    if not hasattr(cfg.logging, "exp_name") or cfg.logging.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg.logging.exp_name = f"robotwin_svae_decoder_{timestamp}"

    latent_cache_root = Path(
        cfg.data.get("latent_cache_dir")
        or (Path(cfg.logging.checkpoint_dir) / "_latent_cache")
    ).expanduser().resolve()
    cfg.data.latent_cache_dir = str(latent_cache_root)

    output_root = Path(cfg.logging.checkpoint_dir).expanduser().resolve()
    if output_root == latent_cache_root or output_root not in latent_cache_root.parents:
        raise ValueError("Decoded feature cache must be a subdirectory of checkpoint_dir")
    for name in ("svae_cache_dir", "raw_data_path", "codec_checkpoint"):
        source = Path(cfg.data.get(name, "")).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Missing data.{name}: {source}")
        if (source == latent_cache_root or latent_cache_root in source.parents
                or source in latent_cache_root.parents):
            raise ValueError(f"Generated cache must not overlap input data.{name}: {source}")
    if accelerator.is_main_process:
        print("Creating S-VAE datasets...")
    datasets = create_datasets(cfg, latent_cache_root)
    if args.smoke_test:
        for dataset in datasets:
            count = min(int(cfg.training.batch_size), len(dataset.records))
            dataset.records = dataset.records[:count]
            dataset.samples = [x for x in dataset.samples if x["record_index"] < count][:cfg.training.batch_size]
            if len(dataset) < cfg.training.batch_size:
                raise ValueError("Not enough samples for the requested smoke-test batch")
    if accelerator.is_main_process:
        frames = sum(int(r["num_frames"]) for d in datasets for r in d.records)
        gib = frames * datasets[0].num_views * datasets[0].tokens_per_view * datasets[0].decoder_feature_dim * 2 / 1024**3
        print(f"Global batch={cfg.training.batch_size}; sequences/rank={gpu_batch_size}; "
              f"RGB images/rank={gpu_batch_size * datasets[0].num_frames * datasets[0].num_views}")
        print(f"Train samples={len(datasets[0])}; validation windows={len(datasets[1])}; "
              f"decoded feature cache approximately {gib:.2f} GiB")
        for dataset in datasets:
            for view in dataset.view_indices:
                print(f"{dataset.split} sample video: {dataset._resolve_raw_video_path(dataset.records[0], view)}")
    if args.check_only:
        if accelerator.is_main_process:
            print("Input/config check passed; no feature cache or training output was written.")
        accelerator.end_training()
        return
    latent_cache_root.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("Materializing decoded VLM features to the temporary disk cache...")
    materialize_decoder_caches(cfg, datasets, accelerator)
    train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(
        cfg, datasets, gpu_batch_size
    )

    if accelerator.is_main_process:
        print("Creating decoder...")
    decoder = create_decoder_from_config(cfg.decoder).to(device)
    if accelerator.is_main_process:
        print(f"Decoder type: {type(decoder).__name__}")
        print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()) / 1e6:.2f}M")

    optimizer = create_optimizer(decoder, cfg)
    prepare_items = [decoder, optimizer, train_loader]
    if val_loader is not None:
        prepare_items.append(val_loader)
    prepared = accelerator.prepare(*prepare_items)

    idx = 0
    decoder = prepared[idx]
    idx += 1
    optimizer = prepared[idx]
    idx += 1
    train_loader = prepared[idx]
    idx += 1
    if val_loader is not None:
        val_loader = prepared[idx]

    # Create the scheduler after Accelerate shards the loader so its horizon
    # matches the number of optimizer updates seen by each rank.
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))

    lpips_fn = lpips
    if lpips_fn is None and accelerator.is_main_process:
        print("LPIPS metric unavailable; continuing without it")

    start_epoch = 1
    best_val_loss = float("inf")
    if getattr(cfg, "resume", None):
        start_epoch, best_val_loss = load_checkpoint(
            cfg.resume, decoder, optimizer, scheduler, device, accelerator
        )
        if accelerator.is_main_process:
            print(f"Resuming training from epoch {start_epoch}")

    if accelerator.is_main_process:
        config_save_path = Path(cfg.logging.checkpoint_dir) / "config.yaml"
        save_yaml_config(cfg, config_save_path)
        print(f"Saved config to {config_save_path}")
        print(f"Latent cache root: {latent_cache_root}")

    training_completed = False
    try:
        if accelerator.is_main_process:
            print(f"\nStarting training from epoch {start_epoch} to {cfg.training.num_epochs}...")

        for epoch in range(start_epoch, cfg.training.num_epochs + 1):
            accelerator.wait_for_everyone()
            train_metrics = train_epoch(
                decoder, train_loader, optimizer, scheduler, lpips_fn, cfg, epoch, accelerator
            )

            if not cfg.logging.get("no_wandb", False):
                accelerator.log(train_metrics, step=epoch)

            if accelerator.is_main_process:
                print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
                print(f"  Train Loss: {train_metrics['train/loss']:.4f}")
                print(f"  Train Recon Loss: {train_metrics['train/recon_loss']:.4f}")
                print(f"  Learning Rate: {train_metrics['train/lr']:.6f}")

            validation_every = max(1, int(cfg.training.get("validation_every", 1)))
            should_validate = (
                epoch % validation_every == 0 or epoch == cfg.training.num_epochs
            )
            val_metrics = {}
            if should_validate:
                eval_loader = val_loader if val_loader is not None else train_loader
                eval_name = "Val" if val_loader is not None else "Train"
                accelerator.wait_for_everyone()
                do_viz = (
                    epoch % cfg.logging.get("visualize_every", 1) == 0
                    or epoch == cfg.training.num_epochs
                )
                val_metrics, _ = validate(
                    decoder, eval_loader, lpips_fn, cfg, epoch, accelerator, visualize=do_viz
                )

            is_best = False
            if accelerator.is_main_process and should_validate:
                if not cfg.logging.get("no_wandb", False):
                    accelerator.log(dict(val_metrics), step=epoch)
                print(f"  {eval_name} Loss: {val_metrics['val/loss']:.4f}")
                print(f"  {eval_name} Recon Loss: {val_metrics['val/recon_loss']:.4f}")
                if "val/psnr" in val_metrics:
                    print(f"  {eval_name} PSNR: {val_metrics['val/psnr']:.2f} dB")
                if "val/ssim" in val_metrics:
                    print(f"  {eval_name} SSIM: {val_metrics['val/ssim']:.4f}")
                if "val/lpips" in val_metrics:
                    print(f"  {eval_name} LPIPS: {val_metrics['val/lpips']:.4f}")

                is_best = val_metrics["val/loss"] < best_val_loss
                if is_best:
                    best_val_loss = val_metrics["val/loss"]
                    print(f"  New best model! {eval_name} Loss: {best_val_loss:.4f}")

            accelerator.wait_for_everyone()
            if epoch % cfg.logging.get("checkpoint_every", 1) == 0 or epoch == cfg.training.num_epochs:
                if accelerator.is_main_process:
                    save_checkpoint(
                        decoder,
                        optimizer,
                        scheduler,
                        epoch,
                        cfg,
                        {**train_metrics, **val_metrics},
                        accelerator,
                        best_val_loss=best_val_loss,
                        is_best=is_best,
                    )
                accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            print("\nTraining completed!")
        training_completed = True
    finally:
        if training_completed:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and not args.keep_feature_cache:
                shutil.rmtree(latent_cache_root)
                print(f"Removed temporary latent cache: {latent_cache_root}")
            accelerator.wait_for_everyone()
        elif accelerator.is_main_process:
            print(f"Training interrupted; retaining decoded feature cache: {latent_cache_root}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
