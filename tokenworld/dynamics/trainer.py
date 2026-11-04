from __future__ import annotations

import torch
from tokenworld.utils.checkpoint_compat import load_checkpoint_payload
import torch.nn as nn
from torch import is_tensor
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import Dataset, TensorDataset, DataLoader

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from tqdm import tqdm

import numpy as np
import yaml
from einops import rearrange
from PIL import Image
import math
import os
from datetime import datetime

from adam_atan2_pytorch import MuonAdamAtan2

from tokenworld.dynamics.model import (
    VideoTokenizer,
    DynamicsWorldModel,
    Experience,
    combine_experiences
)

from ema_pytorch import EMA
from tokenworld.decoder.model import VQGAN

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def cycle(dl):
    while True:
        for batch in dl:
            yield batch


def _try_resample(dataset, epoch: int):
    """Recursively walk the dataset chain and call resample(epoch) if found."""
    if hasattr(dataset, 'resample'):
        dataset.resample(epoch)
    elif hasattr(dataset, 'dataset'):
        _try_resample(dataset.dataset, epoch)


def visualize_reconstruction(
    target_video,      # (b*v*t, 3, h, w) - ground truth images (multi-view)
    recon_video,       # (b*v*t, 3, h, w) - decoded from clean VLM tokens
    pred_recon_video,  # (b*v*t, 3, h, w) - decoded from predicted VLM tokens (optional)
    num_samples=None,  # None means all frames
    step=0,
    save_dir='./visualization',
    signal_levels=None,  # (b, t) tensor of signal levels for each frame
    gt_proprio=None,     # (b, t, d) ground truth proprio (optional)
    pred_proprio=None    # (b, t, d) predicted proprio (optional)
):
    """
    Create visualization with rows:
    - Row 1: all target_video frames (ground truth)
    - Row 2: all recon_video frames (reconstructed from clean VLM tokens)
    - Row 3: all pred_recon_video frames with signal levels displayed at bottom
    
    For multi-view: frames are interleaved as [main_t0, wrist_t0, main_t1, wrist_t1, ...]
    to make it easier to compare views at the same timestep.
    
    Saves to local folder instead of wandb.
    """
    if target_video is None or recon_video is None:
        return None
    
    # Create save directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    target_video = target_video.float()  # bfloat16 -> float32
    recon_video = recon_video.float()

    # from [-1,1] to [0,1]
    target_video = (target_video + 1) / 2
    recon_video = (recon_video + 1) / 2
    
    # Clamp values to [0, 1] for visualization
    target_video = target_video.clamp(0, 1)
    recon_video = recon_video.clamp(0, 1)
    
    # Get total frames and detect multi-view
    total_frames = target_video.shape[0]
    
    # Detect number of views from signal_levels
    # signal_levels is (b, t) or (b*t,), total_frames is b*v*t
    if exists(signal_levels):
        signal_levels_flat = signal_levels.flatten() if signal_levels.ndim == 2 else signal_levels
        signal_levels_len = len(signal_levels_flat)
        num_views = total_frames // signal_levels_len
    else:
        num_views = 1
        signal_levels_len = total_frames
    
    # Create interleaved indices for multi-view visualization
    # Original order: (b, v, t) -> [v0_t0, v0_t1, ..., v1_t0, v1_t1, ...]
    # Interleaved order: [v0_t0, v1_t0, v0_t1, v1_t1, ...] (alternating views at each timestep)
    if num_views > 1:
        t_per_view = signal_levels_len  # b * t
        interleaved_indices = []
        for t_idx in range(t_per_view):
            for v_idx in range(num_views):
                # Original index for (v_idx, t_idx) is: v_idx * t_per_view + t_idx
                orig_idx = v_idx * t_per_view + t_idx
                interleaved_indices.append(orig_idx)
        indices = interleaved_indices
    else:
        indices = list(range(total_frames))
    
    # Sample if needed
    if num_samples is not None:
        num_samples = min(num_samples, len(indices))
        sample_positions = np.linspace(0, len(indices) - 1, num_samples, dtype=int)
        indices = [indices[i] for i in sample_positions]
    
    # Build each row: concatenate all frames horizontally
    # Row 1: target_video
    target_frames = []
    for idx in indices:
        img = target_video[idx].cpu().numpy()  # (3, h, w)
        img = np.transpose(img, (1, 2, 0))     # (h, w, 3)
        target_frames.append(img)
    target_row = np.concatenate(target_frames, axis=1)  # (h, w*num_frames, 3)
    
    # Row 2: recon_video
    recon_frames = []
    for idx in indices:
        img = recon_video[idx].cpu().numpy()  # (3, h, w)
        img = np.transpose(img, (1, 2, 0))    # (h, w, 3)
        recon_frames.append(img)
    recon_row = np.concatenate(recon_frames, axis=1)  # (h, w*num_frames, 3)
    
    # Row 3: pred_recon_video (optional)
    if exists(pred_recon_video):
        pred_recon_video = pred_recon_video.float() # bfloat16 -> float32
        pred_recon_video = (pred_recon_video + 1) / 2
        pred_recon_video = pred_recon_video.clamp(0, 1)
        pred_frames = []
        for idx in indices:
            img = pred_recon_video[idx].cpu().numpy()  # (3, h, w)
            img = np.transpose(img, (1, 2, 0))         # (h, w, 3)
            pred_frames.append(img)
        pred_row = np.concatenate(pred_frames, axis=1)  # (h, w*num_frames, 3)
        # Stack rows vertically: [target_row, recon_row, pred_row]
        grid = np.concatenate([target_row, recon_row, pred_row], axis=0)
    else:
        # Stack rows vertically: [target_row, recon_row]
        grid = np.concatenate([target_row, recon_row], axis=0)
    
    # Convert to uint8 for saving
    grid_uint8 = (grid * 255).astype(np.uint8)
    
    if exists(pred_recon_video) and exists(signal_levels):
        from PIL import ImageDraw, ImageFont
        
        img = Image.fromarray(grid_uint8)
        draw = ImageDraw.Draw(img)
        
        # Try to use a decent font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
        
        # Get dimensions
        img_height, img_width = grid_uint8.shape[:2]
        single_row_height = img_height // 3  # Height of each row
        frame_width = img_width // len(indices)
        
        # Use already-flattened signal_levels from above
        # For multi-view: each view at the same timestep shares the same signal level
        # signal_idx = idx % signal_levels_len maps original frame index to signal level
        
        # Draw signal level for each frame at the bottom of third row
        for i, idx in enumerate(indices):
            # Map frame index to signal level (handles multi-view via modulo)
            signal_idx = idx % signal_levels_len
            signal_level = signal_levels_flat[signal_idx].item()
            text = f"SL:{signal_level}"
            
            # Position: bottom of third row, centered in each frame
            x_pos = i * frame_width + frame_width // 2 - 25  # Rough centering
            y_pos = img_height - 25  # Bottom of third row
            
            # Draw text with black outline for visibility
            for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                draw.text((x_pos + offset[0], y_pos + offset[1]), text, font=font, fill=(0, 0, 0))
            draw.text((x_pos, y_pos), text, font=font, fill=(255, 255, 0))  # Yellow text
        
        # Convert back to array
        grid_uint8 = np.array(img)
    else:
        img = Image.fromarray(grid_uint8)
    
    # Save image
    save_path = os.path.join(save_dir, f'recon_step_{step:06d}.png')
    img.save(save_path)
    
    print(f"Saved visualization to {save_path}")
    
    # Proprio visualization (optional)
    # X-axis: timestep, Y-axis: value, each subplot shows one proprio dimension
    if exists(gt_proprio) and exists(pred_proprio):
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            
            # Take first sample from batch for visualization
            gt_np = gt_proprio[0].detach().cpu().numpy()   # (T, D)
            pred_np = pred_proprio[0].detach().cpu().numpy()  # (T, D)
            
            T, D = gt_np.shape
            timesteps = np.arange(T)
            
            # Create figure: one subplot per proprio dimension
            # Arrange in a grid: max 4 columns
            n_cols = min(4, D)
            n_rows = (D + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
            
            for d in range(D):
                row = d // n_cols
                col = d % n_cols
                ax = axes[row, col]
                
                # Plot GT and Pred proprio over time for this dimension
                ax.plot(timesteps, gt_np[:, d], 'b-', label='GT', linewidth=1.5, alpha=0.8)
                ax.plot(timesteps, pred_np[:, d], 'r--', label='Pred', linewidth=1.5, alpha=0.8)
                
                ax.set_xlabel('Timestep', fontsize=8)
                ax.set_ylabel('Value', fontsize=8)
                ax.set_title(f'Dim {d}', fontsize=9)
                ax.legend(fontsize=7, loc='upper right')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', labelsize=7)
            
            # Hide empty subplots
            for d in range(D, n_rows * n_cols):
                row = d // n_cols
                col = d % n_cols
                axes[row, col].set_visible(False)
            
            plt.suptitle(f'Proprio Comparison (Step {step})', fontsize=11, y=1.02)
            plt.tight_layout()
            
            proprio_save_path = os.path.join(save_dir, f'proprio_step_{step:06d}.png')
            plt.savefig(proprio_save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved proprio visualization to {proprio_save_path}")
        except Exception as e:
            print(f"Warning: Failed to save proprio visualization: {e}")
    
    return save_path

def print_cuda_memory(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[{prefix}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

def add_memory_hooks(module):
    def forward_hook(module, input, output):
        print_cuda_memory(prefix=f"Forward {module.__class__.__name__}")
    def backward_hook(module, grad_input, grad_output):
        print_cuda_memory(prefix=f"Backward {module.__class__.__name__}")
    module.register_forward_hook(forward_hook)
    module.register_backward_hook(backward_hook)

# trainers

class VideoTokenizerTrainer(Module):
    def __init__(
        self,
        model: VideoTokenizer,
        dataset: Dataset,
        optim_klass = MuonAdamAtan2,
        batch_size = 16,
        learning_rate = 3e-4,
        max_grad_norm = None,
        num_train_steps = 10_000,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        checkpoint_every = 5000,
        checkpoint_folder = './checkpoints'
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        self.model = model
        self.dataset = dataset
        self.train_dataloader = DataLoader(dataset, batch_size = batch_size, drop_last = True, shuffle = True)

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        if optim_klass is MuonAdamAtan2:
            optim = MuonAdamAtan2(
                model.muon_parameters(),
                model.parameters(),
                **optim_kwargs
            )
        else:
            optim = optim_klass(
                model.parameters(),
                **optim_kwargs
            )

        self.optim = optim

        self.max_grad_norm = max_grad_norm

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = checkpoint_folder
        self.val_every = val_every
        self.viz_every = viz_every

        (
            self.model,
            self.train_dataloader,
            self.optim
        ) = self.accelerator.prepare(
            self.model,
            self.train_dataloader,
            self.optim
        )

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def forward(
        self
    ):
        iter_train_dl = cycle(self.train_dataloader)

        for _ in range(self.num_train_steps):
            video = next(iter_train_dl)

            loss = self.model(video)
            self.accelerator.backward(loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optim.step()
            self.optim.zero_grad()

        self.print('training complete')

# dynamics world model

class BehaviorCloneTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        dataset: Dataset,
        val_dataset: Dataset = None,  # 验证集
        # optim_klass = MuonAdamAtan2,
        optim_klass = torch.optim.Adam,
        batch_size = 16,
        num_workers: int = 0,
        val_num_workers: int = None,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: int = 2,
        drop_last: bool = True,
        find_unused_parameters: bool = True,
        learning_rate = 3e-4,
        decoder_learning_rate = 1e-4,  # Separate learning rate for decoder
        max_grad_norm = None,
        num_epochs = 100,  # 改为 epoch 数量
        max_train_steps: int | None = None,
        weight_decay = 0., # TODO: Should weight decay to be zero?
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
        checkpoint_every = 5,  # 每多少个 epoch 保存一次
        checkpoint_every_steps: int | None = None,
        checkpoint_folder = f'./checkpoints/{datetime.now().strftime("%Y%m%d_%H%M")}',
        train_decoder = True,  # Whether to train decoder
        val_every = 1,  # 每多少个 epoch 验证一次
        viz_every = 100,  # 每多少个 step 可视化一次
        external_decoder_path: str = None,
        external_decoder_paths: dict = None,  # New: dual decoder support {'view0': path, 'view1_2': path}
        use_idm: bool = False,
        use_reward: bool = False,
        training_mode: str = 'world_model',  # NEW: 'world_model' | 'policy'
    ):
        super().__init__()
        batch_size = min(batch_size, len(dataset))

        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=find_unused_parameters
        )
        self.accelerator = Accelerator(
            cpu = cpu,
            kwargs_handlers=[ddp_kwargs],
            **accelerate_kwargs
        )

        self.model = model
        self.use_idm = use_idm
        self.use_reward = use_reward
        self.training_mode = training_mode  # NEW: store training mode
        # for name, module in self.model.named_modules():
        #     if not isinstance(module, nn.Sequential) and module != self.model:
        #         add_memory_hooks(module)
        self.external_decoder = None
        self.external_decoders = None  # New: for dual decoder mode
        
        # Handle dual decoder mode (for RobotWin: view0 vs view1+view2)
        if external_decoder_paths is not None:
            self.external_decoders = {}
            for key, path in external_decoder_paths.items():
                if path is not None:
                    self.external_decoders[key] = self._load_external_decoder(path)
                    self.print(f"Loaded external decoder '{key}' from {path}")
            train_decoder = False
            self.print("train_decoder is forced to False when using external decoders.")
        elif external_decoder_path is not None:
            self.external_decoder = self._load_external_decoder(external_decoder_path)
            train_decoder = False  # 使用外部 decoder 时强制不训练 decoder
            self.print(f"Loaded external decoder from {external_decoder_path}")
            self.print("train_decoder is forced to False when using external decoder.")

        # Frozen RL-Related Parameters
        # In policy mode, we need to train action prediction heads, so don't freeze them
        if training_mode == 'policy':
            rl_modules = [
                'value_head', 'to_reward_pred', 
                'task_embed', 'latent_genes', 'reward_learned_embed',
                # Keep action_unembed trainable for policy mode
            ]
            self.print("Policy mode: action prediction heads (policy_head, action_unembed) will be trained.")
        else:
            rl_modules = [
                'value_head', 'policy_head', 'to_reward_pred', 
                'task_embed', 'latent_genes', 'reward_learned_embed',
                'discrete_action_unembed', 'continuous_action_unembed'
            ]
        for name, param in self.model.named_parameters():
            if any(module_name in name for module_name in rl_modules):
                param.requires_grad = False

        # The generic action embeddings are bypassed by the per-view action
        # embedding path. Freezing them makes every remaining trainable
        # parameter participate in the world-model graph, allowing faster DDP
        # operation with find_unused_parameters=False.
        if (
            training_mode == 'world_model'
            and getattr(self.model, 'use_multi_view_action_embed', False)
        ):
            self.model.action_learned_embed.requires_grad = False
            for param in self.model.action_embedder.discrete_action_embed.parameters():
                param.requires_grad = False
            for param in self.model.action_embedder.continuous_action_embed.parameters():
                param.requires_grad = False
                
        self.dataset = dataset
        self.val_dataset = val_dataset
        val_num_workers = num_workers if val_num_workers is None else val_num_workers

        def dataloader_runtime_kwargs(worker_count):
            kwargs = dict(
                num_workers=worker_count,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers and worker_count > 0,
            )
            if worker_count > 0:
                kwargs['prefetch_factor'] = prefetch_factor
            return kwargs

        self.train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=True,
            **dataloader_runtime_kwargs(num_workers),
        )
        self.val_dataloader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=False,
                **dataloader_runtime_kwargs(val_num_workers),
            )
            if val_dataset is not None
            else None
        )

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        # Use non_decoder_parameters only when training decoder separately
        if train_decoder and exists(model.decoder):
            main_params = model.non_decoder_parameters()
        else:
            main_params = model.parameters()

        # if optim_klass is MuonAdamAtan2:
        #     optim = MuonAdamAtan2(
        #         model.muon_parameters(),
        #         main_params,
        #         **optim_kwargs
        #     )
        # else:
        #     optim = optim_klass(
        #         main_params,
        #         **optim_kwargs
        #     )
        optim = optim_klass(
            main_params,
            **optim_kwargs
        )

        self.optim = optim
        
        # Separate decoder optimizer
        self.train_decoder = train_decoder
        self.decoder_optim = None
        if train_decoder and exists(model.decoder):
            self.decoder_optim = torch.optim.Adam(
                model.decoder_parameters(),
                lr = decoder_learning_rate,
            )

        self.max_grad_norm = max_grad_norm

        self.num_epochs = num_epochs
        self.max_train_steps = (
            None if max_train_steps is None else int(max_train_steps)
        )
        if exists(self.max_train_steps) and self.max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive")
        self.batch_size = batch_size
        self.checkpoint_every = checkpoint_every
        self.checkpoint_every_steps = (
            None
            if checkpoint_every_steps is None
            else int(checkpoint_every_steps)
        )
        if (
            exists(self.checkpoint_every_steps)
            and self.checkpoint_every_steps <= 0
        ):
            raise ValueError("checkpoint_every_steps must be positive")
        self.checkpoint_folder = checkpoint_folder
        self.val_every = val_every
        self.viz_every = viz_every
        self.learning_rate = learning_rate
        self.decoder_learning_rate = decoder_learning_rate

        # Prepare with accelerator
        prepare_list = [self.model, self.train_dataloader, self.optim]
        if exists(self.decoder_optim):
            prepare_list.append(self.decoder_optim)
        if exists(self.val_dataloader):
            prepare_list.append(self.val_dataloader)
        
        prepared = self.accelerator.prepare(*prepare_list)
        
        self.model = prepared[0]
        self.train_dataloader = prepared[1]
        self.optim = prepared[2]
        idx = 3
        if exists(self.decoder_optim):
            self.decoder_optim = prepared[idx]
            idx += 1
        if exists(self.val_dataloader):
            self.val_dataloader = prepared[idx]

    @property
    def device(self):
        return self.accelerator.device

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)
    
    def _load_external_decoder(self, decoder_path: str):
        from tokenworld.decoder.checkpoint import load_external_decoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return load_external_decoder(decoder_path, device=device, freeze=True)

    def forward(self, resume_from: str = None):
        """
        Training loop with optional resume support.
        
        Args:
            resume_from: Path to checkpoint file to resume from (optional)
        """
        global_step = 0
        best_val_loss = float('inf')
        start_epoch = 0
        
        # Resume from checkpoint if provided
        if exists(resume_from):
            start_epoch, global_step, best_val_loss = self.load_checkpoint(resume_from)
        
        num_train_epochs = self.num_epochs
        if exists(self.max_train_steps):
            steps_per_epoch = len(self.train_dataloader)
            num_train_epochs = max(
                num_train_epochs,
                math.ceil(self.max_train_steps / steps_per_epoch),
            )
            self.print(
                f"Training to {self.max_train_steps} optimizer steps "
                f"({steps_per_epoch} steps/epoch, up to {num_train_epochs} epochs)."
            )

        last_epoch = start_epoch
        for epoch in range(start_epoch, num_train_epochs):
            if exists(self.max_train_steps) and global_step >= self.max_train_steps:
                break

            last_epoch = epoch + 1
            # Resample slices for EpochSlicerDataset (no-op for other datasets)
            _try_resample(self.dataset, epoch)

            # ============ Training ============
            self.model.train()
            train_losses = []
            
            pbar = tqdm(self.train_dataloader, 
                       desc=f"Epoch {epoch+1}/{num_train_epochs} [Train]",
                       disable=not self.accelerator.is_local_main_process)
            
            for batch_data in pbar:
                can_visualize = (
                    exists(self.viz_every)
                    and (
                        self.train_decoder
                        or exists(self.external_decoder)
                        or exists(self.external_decoders)
                    )
                )
                need_viz = (
                    can_visualize
                    and global_step % self.viz_every == 0
                    and self.accelerator.is_main_process
                )
                
                # Handle decoder selection for visualization
                viz_decoder = None
                viz_decoders = None
                if need_viz:
                    if exists(self.external_decoders):
                        viz_decoders = self.external_decoders
                    elif exists(self.external_decoder):
                        viz_decoder = self.external_decoder
                
                add_reward_loss = False
                if self.use_reward:
                    # Check if rewards exist
                    rewards = batch_data.get('rewards')
                    if not exists(rewards):
                        self.print("WARNING: Batch missing rewards, skipping reward loss")
                    else:
                        add_reward_loss = True
                if is_tensor(batch_data):
                    loss, losses, recon_video_out, target_video_out, pred_recon_video_out, signal_levels, proprio_viz_data = self.model(
                        batch_data,
                        add_reward_loss=add_reward_loss,
                        train_decoder=self.train_decoder,
                        external_decoder=viz_decoder,
                        external_decoders=viz_decoders,
                        training_mode=self.training_mode,  # NEW: pass training mode
                    )
                else:
                    loss, losses, recon_video_out, target_video_out, pred_recon_video_out, signal_levels, proprio_viz_data = self.model(
                        **batch_data,
                        add_reward_loss=add_reward_loss,
                        train_decoder=self.train_decoder,
                        external_decoder=viz_decoder,
                        external_decoders=viz_decoders,
                        training_mode=self.training_mode,  # NEW: pass training mode
                    )

                self.accelerator.backward(loss)

                if exists(self.max_grad_norm):
                    if self.train_decoder and exists(self.accelerator.unwrap_model(self.model).decoder):
                        clip_params = list(self.accelerator.unwrap_model(self.model).non_decoder_parameters())
                    else:
                        clip_params = list(self.accelerator.unwrap_model(self.model).parameters())
                    self.accelerator.clip_grad_norm_(clip_params, self.max_grad_norm)

                self.optim.step()
                self.optim.zero_grad()
                
                if self.train_decoder and exists(self.decoder_optim):
                    self.decoder_optim.step()
                    self.decoder_optim.zero_grad()
                
                train_losses.append(loss.item())
                flow_loss = losses.flow.item()
                latent_flow_loss = losses.latent_flow.item()
                proprio_flow_loss = losses.proprio_flow.item()
                reward_loss = losses.rewards.sum().item() if exists(losses.rewards) else self.model.zero
                if self.train_decoder:
                    recon_loss_val = losses.reconstruction.item()
                    pbar.set_postfix(flow=f"{losses.flow.item():.4f}", recon=f"{losses.reconstruction.item():.4f}")
                else:
                    pbar.set_postfix(flow=f"{losses.flow.item():.4f}")
                
                # Logging
                self.accelerator.log({"train/flow_loss": flow_loss}, step=global_step)
                self.accelerator.log({"train/noisy_flow_loss": losses.noisy_flow}, step=global_step)
                self.accelerator.log({"train/latent_flow_loss": latent_flow_loss}, step=global_step)
                self.accelerator.log({"train/proprio_flow_loss": proprio_flow_loss}, step=global_step)
                if self.use_idm:
                    self.accelerator.log({"train/idm_loss": losses.idm}, step=global_step)
                if self.train_decoder:
                    self.accelerator.log({"train/recon_loss": losses.reconstruction.item()}, step=global_step)
                if add_reward_loss and exists(rewards):
                    self.accelerator.log({"train/reward_loss": reward_loss}, step=global_step)
                # NEW: Log action losses in policy mode
                if self.training_mode == 'policy':
                    if exists(losses.discrete_actions) and not (isinstance(losses.discrete_actions, (int, float)) and losses.discrete_actions == 0):
                        discrete_action_loss = losses.discrete_actions.sum().item() if hasattr(losses.discrete_actions, 'sum') else float(losses.discrete_actions)
                        self.accelerator.log({"train/discrete_action_loss": discrete_action_loss}, step=global_step)
                    if exists(losses.continuous_actions) and not (isinstance(losses.continuous_actions, (int, float)) and losses.continuous_actions == 0):
                        continuous_action_loss = losses.continuous_actions.sum().item() if hasattr(losses.continuous_actions, 'sum') else float(losses.continuous_actions)
                        self.accelerator.log({"train/continuous_action_loss": continuous_action_loss}, step=global_step)
                
                # Visualization
                if need_viz:
                    visualize_reconstruction(
                        target_video=target_video_out,
                        recon_video=recon_video_out,
                        pred_recon_video=pred_recon_video_out,
                        num_samples=None,
                        step=global_step,
                        save_dir=os.path.join(self.checkpoint_folder, 'visualizations'),
                        signal_levels=signal_levels,
                        gt_proprio=proprio_viz_data.get('gt_proprio') if proprio_viz_data else None,
                        pred_proprio=proprio_viz_data.get('pred_proprio') if proprio_viz_data else None
                    )
                
                global_step += 1

                if (
                    exists(self.checkpoint_every_steps)
                    and global_step % self.checkpoint_every_steps == 0
                ):
                    ckpt_path = (
                        f"{self.checkpoint_folder}/"
                        f"checkpoint_step_{global_step:07d}.pt"
                    )
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoint(
                        ckpt_path,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                    )
                    self.accelerator.wait_for_everyone()

                if exists(self.max_train_steps) and global_step >= self.max_train_steps:
                    break
            
            avg_train_loss = sum(train_losses) / len(train_losses)
            # Keep every metric on the optimizer-step axis so resumed W&B runs
            # append to the existing curve without stepping backwards to epoch 0.
            self.accelerator.log(
                {"train/epoch_loss": avg_train_loss},
                step=max(global_step - 1, 0),
            )
            self.print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
            
            # ============ Validation ============
            # if exists(self.val_dataloader) and (epoch + 1) % self.val_every == 0:
            #     val_loss = self.validate(epoch)
                
            #     # Save best model
            #     if val_loss < best_val_loss:
            #         best_val_loss = val_loss
            #         best_ckpt_dir = f"{self.checkpoint_folder}/best_epoch_{epoch+1}"
            #         self.accelerator.save_state(output_dir=best_ckpt_dir, safe_serialization=False)
            #         self.print(f"New best model saved with val_loss: {val_loss:.4f} to {best_ckpt_dir}")

            # ============ Checkpoint ============
            if (
                not exists(self.checkpoint_every_steps)
                and exists(self.checkpoint_every)
                and (epoch + 1) % self.checkpoint_every == 0
            ):
                ckpt_path = f"{self.checkpoint_folder}/checkpoint_epoch_{epoch+1:04d}.pt"
                self.save_checkpoint(ckpt_path, epoch=epoch+1, global_step=global_step, best_val_loss=best_val_loss)

        self.print('training complete')
        final_step_was_checkpointed = (
            exists(self.checkpoint_every_steps)
            and global_step > 0
            and global_step % self.checkpoint_every_steps == 0
        )
        if not final_step_was_checkpointed:
            final_ckpt_path = f"{self.checkpoint_folder}/checkpoint_step_final.pt"
            self.accelerator.wait_for_everyone()
            self.save_checkpoint(
                final_ckpt_path,
                epoch=last_epoch,
                global_step=global_step,
                best_val_loss=best_val_loss,
            )
            self.accelerator.wait_for_everyone()

    def save_checkpoint(self, save_path: str, epoch: int = None, global_step: int = None, best_val_loss: float = None):
        """
        Save checkpoint as a single .pt file (without DTensor).
        Only saves on main process.
        """
        if not self.accelerator.is_main_process:
            return
        
        # Get unwrapped model to avoid DDP/FSDP wrapper issues
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Use accelerator.get_state_dict to properly handle distributed tensors
        # This ensures we get the full tensor even in FSDP/distributed scenarios
        model_state_dict = self.accelerator.get_state_dict(self.model, unwrap=True)
        
        # Convert any remaining DTensor to regular Tensor (safety check)
        clean_state_dict = {}
        for key, value in model_state_dict.items():
            if hasattr(value, 'full_tensor'):
                # DTensor: get full tensor
                clean_state_dict[key] = value.full_tensor().clone().cpu()
            elif hasattr(value, '_local_tensor'):
                # DTensor with _local_tensor: this is a shard, try to get full tensor
                if hasattr(value, 'full_tensor'):
                    clean_state_dict[key] = value.full_tensor().clone().cpu()
                else:
                    # Fallback: use local tensor (may be incomplete in FSDP)
                    clean_state_dict[key] = value._local_tensor.clone().cpu()
                    self.print(f"Warning: {key} may be incomplete (DTensor without full_tensor)")
            else:
                clean_state_dict[key] = value.cpu() if hasattr(value, 'cpu') else value
        
        checkpoint = {
            'model_state_dict': clean_state_dict,
            'optimizer_state_dict': self.optim.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'best_val_loss': best_val_loss,
            'learning_rate': self.learning_rate,
            # Save random states for reproducibility
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        
        # Save decoder optimizer if exists
        if exists(self.decoder_optim):
            checkpoint['decoder_optimizer_state_dict'] = self.decoder_optim.state_dict()
            checkpoint['decoder_learning_rate'] = self.decoder_learning_rate
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torch.save(checkpoint, save_path)
        self.print(f"Checkpoint saved to {save_path}")

    def load_checkpoint(self, load_path: str):
        """
        Load checkpoint from a .pt file for resuming training.
        Returns: (epoch, global_step, best_val_loss) to resume from.
        """
        self.print(f"Loading checkpoint from {load_path}")
        checkpoint = load_checkpoint_payload(load_path, map_location='cpu')
        
        # Load model state
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load decoder optimizer if exists
        if exists(self.decoder_optim) and 'decoder_optimizer_state_dict' in checkpoint:
            self.decoder_optim.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
        
        # Restore random states if available
        if 'rng_state' in checkpoint and checkpoint['rng_state'] is not None:
            torch.set_rng_state(checkpoint['rng_state'])
        if 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        
        epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.print(f"Resumed from epoch {epoch}, global_step {global_step}")
        return epoch, global_step, best_val_loss

    @torch.no_grad()
    def validate(self, epoch):
        """Validation loop"""
        self.model.eval()
        val_losses = []
        val_flow_losses = []
        val_recon_losses = []
        
        pbar = tqdm(self.val_dataloader,
                   desc=f"Epoch {epoch+1} [Val]",
                   disable=not self.accelerator.is_local_main_process)
        visualized = False
        
        for batch_idx, batch_data in enumerate(pbar):
            need_viz = (batch_idx == 0 and 
                   self.accelerator.is_main_process and 
                   (exists(self.external_decoder) or exists(self.external_decoders)))
            
            # Handle decoder selection for visualization
            viz_decoder = None
            viz_decoders = None
            if need_viz:
                if exists(self.external_decoders):
                    viz_decoders = self.external_decoders
                elif exists(self.external_decoder):
                    viz_decoder = self.external_decoder
            
            if is_tensor(batch_data):
                loss, losses, recon_video_out, target_video_out, pred_recon_video_out, signal_levels, proprio_viz_data = self.model(
                    batch_data,
                    train_decoder=self.train_decoder,
                    external_decoder=viz_decoder,
                    external_decoders=viz_decoders,
                    training_mode=self.training_mode,  # NEW: pass training mode
                )
            else:
                loss, losses, recon_video_out, target_video_out, pred_recon_video_out, signal_levels, proprio_viz_data = self.model(
                    **batch_data,
                    train_decoder=self.train_decoder,
                    external_decoder=viz_decoder,
                    external_decoders=viz_decoders,
                    training_mode=self.training_mode,  # NEW: pass training mode
                )
            
            val_losses.append(loss.item())
            val_flow_losses.append(losses.flow.item())
            if self.train_decoder:
                val_recon_losses.append(losses.reconstruction.item())
                pbar.set_postfix(flow=f"{losses.flow.item():.4f}", recon=f"{losses.reconstruction.item():.4f}")
            else:
                pbar.set_postfix(flow=f"{losses.flow.item():.4f}")
        
        if need_viz and not visualized:
            visualize_reconstruction(
                target_video=target_video_out,
                recon_video=recon_video_out,
                pred_recon_video=pred_recon_video_out,
                num_samples=None,
                step=epoch,  # 使用 epoch 作为 step
                save_dir=os.path.join(self.checkpoint_folder, 'visualizations', 'val'),
                gt_proprio=proprio_viz_data.get('gt_proprio') if proprio_viz_data else None,
                pred_proprio=proprio_viz_data.get('pred_proprio') if proprio_viz_data else None
            )
            visualized = True
            
        avg_val_loss = sum(val_losses) / len(val_losses)
        avg_flow_loss = sum(val_flow_losses) / len(val_flow_losses)
        self.accelerator.log({"val/epoch_loss": avg_val_loss}, step=epoch)
        self.accelerator.log({"val/flow_loss": avg_flow_loss}, step=epoch)

        if self.train_decoder:
            avg_recon_loss = sum(val_recon_losses) / len(val_recon_losses)
            self.accelerator.log({"val/recon_loss": avg_recon_loss}, step=epoch)
            self.print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} (flow: {avg_flow_loss:.4f}, recon: {avg_recon_loss:.4f})")
        else:
            self.print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f} (flow: {avg_flow_loss:.4f}")
        
        self.model.train()
        return avg_val_loss

# training from dreams

class DreamTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        optim_klass = AdamW,
        batch_size = 16,
        generate_timesteps = 16,
        learning_rate = 3e-4,
        max_grad_norm = None,
        num_train_steps = 10_000,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        self.model = model

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.policy_head_optim = AdamW(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = AdamW(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size
        self.generate_timesteps = generate_timesteps

        self.unwrapped_model = self.model

        (
            self.model,
            self.policy_head_optim,
            self.value_head_optim,
        ) = self.accelerator.prepare(
            self.model,
            self.policy_head_optim,
            self.value_head_optim
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def forward(
        self
    ):

        for _ in range(self.num_train_steps):

            dreams = self.unwrapped_model.generate(
                self.generate_timesteps + 1, # plus one for bootstrap value
                batch_size = self.batch_size,
                return_rewards_per_frame = True,
                return_agent_actions = True,
                return_log_probs_and_values = True
            )

            policy_head_loss, value_head_loss = self.model.learn_from_experience(dreams)

            self.print(f'policy head loss: {policy_head_loss.item():.3f} | value head loss: {value_head_loss.item():.3f}')

            # update policy head

            self.accelerator.backward(policy_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.policy_head_parameters()(), self.max_grad_norm)

            self.policy_head_optim.step()
            self.policy_head_optim.zero_grad()

            # update value head

            self.accelerator.backward(value_head_loss)

            if exists(self.max_grad_norm):
                self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

            self.value_head_optim.step()
            self.value_head_optim.zero_grad()

        self.print('training complete')

# training from sim

class SimTrainer(Module):
    def __init__(
        self,
        model: DynamicsWorldModel,
        optim_klass = AdamW,
        batch_size = 16,
        generate_timesteps = 16,
        learning_rate = 3e-4,
        max_grad_norm = None,
        epochs = 2,
        weight_decay = 0.,
        accelerate_kwargs: dict = dict(),
        optim_kwargs: dict = dict(),
        cpu = False,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            cpu = cpu,
            **accelerate_kwargs
        )

        self.model = model

        optim_kwargs = dict(
            lr = learning_rate,
            weight_decay = weight_decay
        )

        self.policy_head_optim = AdamW(model.policy_head_parameters(), **optim_kwargs)
        self.value_head_optim = AdamW(model.value_head_parameters(), **optim_kwargs)

        self.max_grad_norm = max_grad_norm

        self.epochs = epochs
        self.batch_size = batch_size

        self.generate_timesteps = generate_timesteps

        self.unwrapped_model = self.model

        (
            self.model,
            self.policy_head_optim,
            self.value_head_optim,
        ) = self.accelerator.prepare(
            self.model,
            self.policy_head_optim,
            self.value_head_optim
        )

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    def print(self, *args, **kwargs):
        return self.accelerator.print(*args, **kwargs)

    def learn(
        self,
        experience: Experience
    ):

        step_size = experience.step_size
        agent_index = experience.agent_index

        latents = experience.latents
        old_values = experience.values
        rewards = experience.rewards

        has_agent_embed = exists(experience.agent_embed)
        agent_embed = experience.agent_embed

        discrete_actions, continuous_actions = experience.actions
        discrete_log_probs, continuous_log_probs = experience.log_probs

        discrete_old_action_unembeds, continuous_old_action_unembeds = default(experience.old_action_unembeds, (None, None))

        # handle empties

        empty_tensor = torch.empty_like(rewards)

        agent_embed = default(agent_embed, empty_tensor)

        has_discrete = exists(discrete_actions)
        has_continuous = exists(continuous_actions)

        discrete_actions = default(discrete_actions, empty_tensor)
        continuous_actions = default(continuous_actions, empty_tensor)

        discrete_log_probs = default(discrete_log_probs, empty_tensor)
        continuous_log_probs = default(continuous_log_probs, empty_tensor)

        discrete_old_action_unembeds = default(discrete_old_action_unembeds, empty_tensor)
        continuous_old_action_unembeds = default(discrete_old_action_unembeds, empty_tensor)

        # create the dataset and dataloader

        dataset = TensorDataset(
            latents,
            discrete_actions,
            continuous_actions,
            discrete_log_probs,
            continuous_log_probs,
            agent_embed,
            discrete_old_action_unembeds,
            continuous_old_action_unembeds,
            old_values,
            rewards
        )

        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = True)

        for epoch in range(self.epochs):

            for (
                latents,
                discrete_actions,
                continuous_actions,
                discrete_log_probs,
                continuous_log_probs,
                agent_embed,
                discrete_old_action_unembeds,
                continuous_old_action_unembeds,
                old_values,
                rewards
            ) in dataloader:

                actions = (
                    discrete_actions if has_discrete else None,
                    continuous_actions if has_continuous else None
                )

                log_probs = (
                    discrete_log_probs if has_discrete else None,
                    continuous_log_probs if has_continuous else None
                )

                old_action_unembeds = (
                    discrete_old_action_unembeds if has_discrete else None,
                    continuous_old_action_unembeds if has_continuous else None
                )

                batch_experience = Experience(
                    latents = latents,
                    actions = actions,
                    log_probs = log_probs,
                    agent_embed = agent_embed if has_agent_embed else None,
                    old_action_unembeds = old_action_unembeds,
                    values = old_values,
                    rewards = rewards,
                    step_size = step_size,
                    agent_index = agent_index
                )

                policy_head_loss, value_head_loss = self.model.learn_from_experience(batch_experience)

                self.print(f'policy head loss: {policy_head_loss.item():.3f} | value head loss: {value_head_loss.item():.3f}')

                # update policy head

                self.accelerator.backward(policy_head_loss)

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.model.policy_head_parameters()(), self.max_grad_norm)

                self.policy_head_optim.step()
                self.policy_head_optim.zero_grad()

                # update value head

                self.accelerator.backward(value_head_loss)

                if exists(self.max_grad_norm):
                    self.accelerator.clip_grad_norm_(self.model.value_head_parameters(), self.max_grad_norm)

                self.value_head_optim.step()
                self.value_head_optim.zero_grad()

        self.print('training complete')

    def forward(
        self,
        env,
        num_episodes = 50000,
        max_experiences_before_learn = 8,
        env_is_vectorized = False
    ):

        for _ in range(num_episodes):

            total_experience = 0
            experiences = []

            while total_experience < max_experiences_before_learn:

                experience = self.unwrapped_model.interact_with_env(env, env_is_vectorized = env_is_vectorized)

                num_experience = experience.video.shape[0]

                total_experience += num_experience

                experiences.append(experience.cpu())

            combined_experiences = combine_experiences(experiences)

            self.learn(combined_experiences)

            experiences.clear()

        self.print('training complete')
