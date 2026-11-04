'''
World Model Rollout Interface for RL Training and Evaluation.

Provides clean interfaces for:
- Single-step prediction (step): for RL step-by-step interaction
- Multi-step rollout (rollout): given full action sequence
- Autoregressive rollout (rollout_autoregressive): long-horizon with sliding window
- Latent-to-pixel decoding (decode_latents): for visualization/evaluation

All interfaces work at the VLM latent level:
- Input: VLM latents + action sequences
- Output: predicted VLM latents (and optionally decoded video frames)

Usage:
    from tokenworld.evaluation.rollout import WorldModelRollout

    # Initialize
    wm = WorldModelRollout(
        ckpt_path='path/to/checkpoint.pt',
        decoder_view0_path='path/to/decoder_view0.pt',
        decoder_view1_2_path='path/to/decoder_view12.pt',
    )

    # RL single-step interface
    next_latent, next_proprio = wm.step(
        current_latent, discrete_action, continuous_action, current_proprio
    )

    # Multi-step rollout (replay evaluation)
    result = wm.rollout(init_latents, discrete_actions, continuous_actions, init_proprio)

    # Autoregressive rollout (long-horizon)
    result = wm.rollout_autoregressive(
        init_latents, discrete_actions, continuous_actions, init_proprio,
        pred_step=12, interact_num=3, num_history=4,
    )

    # Decode latents to video (optional, for visualization)
    video = wm.decode_latents(result.latents)
'''

import sys
import os
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor
from einops import rearrange

# Add parent directory to path

from tokenworld.utils.checkpoint import (
    load_checkpoint,
    load_fsdp_checkpoint,
)
from tokenworld.decoder.checkpoint import load_external_decoder
from tokenworld.dynamics.model import Experience, TransformerIntermediates


# ================================================================
# Data structures
# ================================================================

@dataclass
class RolloutResult:
    """Result of a world model rollout."""
    latents: Tensor                       # (B, T, V, N, D) or (B, T, N, D)
    proprio: Optional[Tensor] = None      # (B, T, D)
    generation_time: float = 0.0
    time_cache: Optional[TransformerIntermediates] = None


# ================================================================
# Checkpoint loading
# ================================================================

def load_world_model_checkpoint(ckpt_path: str, device: str = 'cpu'):
    """
    Load world model checkpoint with support for multiple formats:
    1. .pt/.pth file (direct state_dict)
    2. FSDP checkpoint (directory with pytorch_model_fsdp_0)
    3. Standard accelerate checkpoint (directory)

    Args:
        ckpt_path: Path to checkpoint file (.pt) or directory
        device: Device to load to

    Returns:
        Tuple of (model, config)
    """
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith('.pt') or ckpt_path.endswith('.pth'):
        print(f"Loading .pt checkpoint from: {ckpt_path}")
        return load_checkpoint(ckpt_path, device=device)
    else:
        return load_fsdp_checkpoint(ckpt_path, device=device)


# ================================================================
# WorldModelRollout
# ================================================================

class WorldModelRollout:
    """
    World model rollout interface for RL training and evaluation.

    Works at the VLM latent level:
    - Input: initial VLM latents + action sequences
    - Output: predicted VLM latents (and optionally decoded video frames)

    Supports:
    - Single-step rollout (for RL step-by-step interaction)
    - Multi-step rollout (given full action sequence)
    - Autoregressive rollout (long-horizon with sliding window)
    - Latent-to-pixel decoding (with single or dual decoders)
    """

    def __init__(
        self,
        ckpt_path: str,
        external_decoder_path: Optional[str] = None,
        decoder_view0_path: Optional[str] = None,
        decoder_view1_2_path: Optional[str] = None,
        device: str = 'cuda',
        num_steps: int = 4,
        no_shortcut: bool = False,
        decode_chunk_size: int = 8,
    ):
        """
        Initialize WorldModelRollout.

        Args:
            ckpt_path: Path to world model checkpoint (.pt or FSDP directory)
            external_decoder_path: Path to single decoder (for single-view or shared decoder)
            decoder_view0_path: Path to view0 decoder (dual decoder mode, RobotWin)
            decoder_view1_2_path: Path to view1+2 decoder (dual decoder mode, RobotWin)
            device: Device to use
            num_steps: Number of denoising steps
            no_shortcut: Disable shortcut connection (use max denoising steps)
            decode_chunk_size: Frames to decode at once (reduce if OOM)
        """
        self.device = device
        self.num_steps = num_steps
        self.no_shortcut = no_shortcut
        self.decode_chunk_size = decode_chunk_size

        # Load world model
        print(f"Loading world model from: {ckpt_path}")
        self.world_model, self.config = load_world_model_checkpoint(ckpt_path, device=device)
        self.world_model.eval()
        print(f"  ✓ World model loaded")

        # Auto-detect settings from config
        self._frameskip = self.config.get('data', {}).get('frameskip', 1)
        self._addskip   = self.config.get('data', {}).get('addskip', 0)
        self._views = self.config.get('data', {}).get('views', None)
        self._num_video_views = self.config.get('model', {}).get('num_video_views', 1)
        if self._views is None and self._num_video_views == 1:
            self._views = [0]
        print(f"  ✓ Config: frameskip={self._frameskip}, addskip={self._addskip}, views={self._views}, num_video_views={self._num_video_views}")

        # Load decoder(s)
        self.decoder = None
        self.decoders = None
        self.use_dual_decoders = decoder_view0_path is not None or decoder_view1_2_path is not None

        if self.use_dual_decoders:
            self.decoders = {}
            if decoder_view0_path is not None:
                self.decoders['view0'] = load_external_decoder(decoder_view0_path, device=device)
                print(f"  ✓ Loaded view0 decoder from {decoder_view0_path}")
            if decoder_view1_2_path is not None:
                self.decoders['view1_2'] = load_external_decoder(decoder_view1_2_path, device=device)
                print(f"  ✓ Loaded view1+2 decoder from {decoder_view1_2_path}")
        elif external_decoder_path is not None:
            self.decoder = load_external_decoder(external_decoder_path, device=device)
            print(f"  ✓ Loaded external decoder")
        elif hasattr(self.world_model, 'active_decoder') and self.world_model.active_decoder is not None:
            self.decoder = self.world_model.active_decoder.eval()
            print(f"  ✓ Using built-in decoder")
        else:
            print("  WARNING: No decoder available. decode_latents() will not work.")

    # ================================================================
    # Properties
    # ================================================================

    @property
    def frameskip(self) -> int:
        return self._frameskip

    @property
    def addskip(self) -> int:
        return self._addskip

    @property
    def views(self):
        return self._views

    @property
    def num_video_views(self) -> int:
        return self._num_video_views

    @property
    def has_decoder(self) -> bool:
        return self.decoder is not None or (self.decoders is not None and len(self.decoders) > 0)

    # ================================================================
    # Core RL Interface
    # ================================================================

    def step(
        self,
        current_latent: Tensor,
        discrete_action: Optional[Tensor] = None,
        continuous_action: Optional[Tensor] = None,
        current_proprio: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Single-step world model prediction: predict next frame from current frame + action.

        This is the core RL interface for step-by-step interaction.

        Args:
            current_latent: (B, V, N, D) or (B, N, D) - current frame VLM latent
            discrete_action: (B, num_discrete) - discrete action for this transition
            continuous_action: (B, num_continuous) - continuous action for this transition
            current_proprio: (B, D) - current proprioception

        Returns:
            next_latent: (B, V, N, D) or (B, N, D) - predicted next frame latent
            next_proprio: (B, D) or None - predicted next proprioception
        """
        B = current_latent.shape[0]

        # Add time dimension: (B, V, N, D) -> (B, 1, V, N, D) or (B, N, D) -> (B, 1, N, D)
        init_latents = current_latent.unsqueeze(1)

        # Build action sequences: need 2 actions for time_steps=1
        # world_model.generate expects time_steps+1 actions
        discrete_actions = None
        if discrete_action is not None:
            discrete_actions = discrete_action.unsqueeze(1).repeat(1, 2, 1)  # (B, 2, na)

        continuous_actions = None
        if continuous_action is not None:
            continuous_actions = continuous_action.unsqueeze(1).repeat(1, 2, 1)  # (B, 2, na)

        # Build proprio
        init_proprio = None
        if current_proprio is not None:
            init_proprio = current_proprio.unsqueeze(1)  # (B, 1, D)

        # Generate 1 step
        with torch.no_grad():
            gens = self.world_model.generate(
                time_steps=1,
                batch_size=B,
                num_steps=self.num_steps,
                tasks=None,
                return_decoded_video=False,
                return_rewards_per_frame=False,
                init_latents=init_latents,
                init_proprio=init_proprio,
                discrete_actions=discrete_actions,
                continuous_actions=continuous_actions,
                no_shortcut=self.no_shortcut,
            )

        # Extract results
        if isinstance(gens, Experience):
            latents = gens.latents   # (B, 2, ...) - [init, predicted]
            proprio = gens.proprio   # (B, 2, D) or None
        else:
            latents = gens
            proprio = None

        # Return only the predicted frame (last one)
        next_latent = latents[:, -1]      # (B, V, N, D) or (B, N, D)
        next_proprio = proprio[:, -1] if proprio is not None else None  # (B, D)

        return next_latent, next_proprio

    def step_with_history(
        self,
        history_latents: Tensor,
        discrete_actions: Optional[Tensor],
        continuous_actions: Optional[Tensor],
        history_proprio: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Predict one frame from a training-style four-frame context.

        ``history_latents`` and ``history_proprio`` contain the four context
        frames.  The action tensors contain five entries: the first four are
        aligned with the context frames and the fifth drives the newly
        generated frame, matching ``DynamicsWorldModel.generate``'s
        ``num_init_frames`` convention.
        """
        if history_latents.ndim == 3:
            history_latents = history_latents.unsqueeze(0)
        if history_latents.ndim not in (4, 5):
            raise ValueError(f"history_latents must be [B,4,N,D] or [B,4,V,N,D], got {tuple(history_latents.shape)}")
        if history_latents.shape[1] != 4:
            raise ValueError(f"history_latents must contain exactly 4 frames, got {history_latents.shape[1]}")
        B = history_latents.shape[0]
        if discrete_actions is not None and discrete_actions.shape[:2] != (B, 5):
            raise ValueError(f"discrete_actions must be [B,5,A], got {tuple(discrete_actions.shape)}")
        if continuous_actions is not None and continuous_actions.shape[:2] != (B, 5):
            raise ValueError(f"continuous_actions must be [B,5,A], got {tuple(continuous_actions.shape)}")
        if history_proprio is not None:
            if history_proprio.ndim == 2:
                history_proprio = history_proprio.unsqueeze(0)
            if history_proprio.shape[:2] != (B, 4):
                raise ValueError(f"history_proprio must be [B,4,D], got {tuple(history_proprio.shape)}")
        with torch.no_grad():
            gens = self.world_model.generate(
                time_steps=4,
                batch_size=B,
                num_init_frames=4,
                num_steps=self.num_steps,
                tasks=None,
                return_decoded_video=False,
                return_rewards_per_frame=False,
                init_latents=history_latents,
                init_proprio=history_proprio,
                discrete_actions=discrete_actions,
                continuous_actions=continuous_actions,
                no_shortcut=self.no_shortcut,
            )
        if not isinstance(gens, Experience):
            raise RuntimeError("Expected Experience from history-aware world-model generation")
        next_latent = gens.latents[:, -1]
        next_proprio = gens.proprio[:, -1] if gens.proprio is not None else None
        return next_latent, next_proprio

    def rollout(
        self,
        init_latents: Tensor,
        discrete_actions: Optional[Tensor] = None,
        continuous_actions: Optional[Tensor] = None,
        init_proprio: Optional[Tensor] = None,
        *,
        time_cache: Optional[TransformerIntermediates] = None,
        return_time_cache: bool = False,
        gru_only_boundary: bool = False,
    ) -> RolloutResult:
        """
        Multi-step rollout: given initial latents and full action sequence,
        generate all future frames in one call.

        Args:
            init_latents: (B, T_init, V, N, D) or (B, T_init, N, D) - initial frame latents
                          Can also be (B, V, N, D) or (B, N, D) for single init frame
            discrete_actions: (B, T_actions, num_discrete) - full action sequence
            continuous_actions: (B, T_actions, num_continuous) - full action sequence
            init_proprio: (B, T_init, D) or (B, D) - initial proprioception

            time_cache: full temporal KV and GRU state through the boundary frame.
                Resume with that frame and actions [boundary, next, ...].
            return_time_cache: include the generator cache in the result.
            gru_only_boundary: return only GRU state preceding the overlap frame.
                Supply that state on the next call to rebuild attention locally.

        Returns:
            RolloutResult with:
                latents: (B, T_total, ...) - init + generated latents
                proprio: (B, T_total, D) or None
                generation_time: float

        Note:
            T_actions must equal time_steps + 1 where time_steps is passed to
            world_model.generate(). The model generates T_actions - T_init new frames.
        """
        assert discrete_actions is not None or continuous_actions is not None, \
            'Must provide discrete_actions or continuous_actions'

        # Compute time_steps from action sequence length
        if discrete_actions is not None:
            time_steps = discrete_actions.shape[1] - 1
        else:
            time_steps = continuous_actions.shape[1] - 1

        B = init_latents.shape[0]

        start_time = time.time()

        with torch.no_grad():
            gens = self.world_model.generate(
                time_steps=time_steps,
                batch_size=B,
                num_steps=self.num_steps,
                tasks=None,
                return_decoded_video=False,
                return_rewards_per_frame=False,
                init_latents=init_latents,
                init_proprio=init_proprio,
                discrete_actions=discrete_actions,
                continuous_actions=continuous_actions,
                no_shortcut=self.no_shortcut,
                time_cache=time_cache,
                return_time_cache=return_time_cache,
                gru_only_boundary=gru_only_boundary,
            )

        generation_time = time.time() - start_time

        next_cache = None
        if return_time_cache:
            gens, next_cache = gens

        if isinstance(gens, Experience):
            latents = gens.latents
            proprio = gens.proprio
        else:
            latents = gens
            proprio = None

        return RolloutResult(
            latents=latents,
            proprio=proprio,
            generation_time=generation_time,
            time_cache=next_cache,
        )

    def rollout_autoregressive(
        self,
        init_latents: Tensor,
        discrete_actions: Optional[Tensor] = None,
        continuous_actions: Optional[Tensor] = None,
        init_proprio: Optional[Tensor] = None,
        full_proprio: Optional[Tensor] = None,
        pred_step: int = 12,
        interact_num: int = 3,
        num_history: int = 4,
    ) -> RolloutResult:
        """
        Autoregressive rollout with sliding window for long-horizon generation.

        Each interaction generates pred_step frames using up to num_history context frames.
        Successive interactions overlap by 1 frame (the last predicted frame becomes
        context for the next interaction).

        Args:
            init_latents: (B, 1, V, N, D) or (B, 1, N, D) - initial frame latent
            discrete_actions: (B, T_total, num_discrete) - full action sequence
            continuous_actions: (B, T_total, num_continuous) - full action sequence
            init_proprio: (B, 1, D) or (B, D) - initial proprioception
            full_proprio: (B, T_total, D) - GT proprio for all timesteps (optional fallback
                          when model doesn't predict proprio but needs it for conditioning)
            pred_step: frames per interaction (including context overlap)
            interact_num: number of autoregressive interactions
            num_history: max context frames from previous interactions

        Returns:
            RolloutResult with concatenated latents and proprio
        """
        # Normalize shapes
        if init_latents.ndim == 4:
            if init_latents.shape[1] == self._num_video_views and self._num_video_views > 1:
                init_latents = init_latents.unsqueeze(1)  # (B, 1, V, N, D)
        elif init_latents.ndim == 3:
            init_latents = init_latents.unsqueeze(1)  # (B, 1, N, D)

        if init_proprio is not None and init_proprio.ndim == 2:
            init_proprio = init_proprio.unsqueeze(1)  # (B, 1, D)

        B = init_latents.shape[0]
        T_total = discrete_actions.shape[1] if discrete_actions is not None else continuous_actions.shape[1]

        print(f"Autoregressive generation: {interact_num} interactions x {pred_step} frames")
        print(f"  Total action timesteps: {T_total}, num_history: {num_history}")

        start_time = time.time()

        # Initialize accumulated context with first frame
        accumulated_latents = init_latents  # (B, 1, ...)
        accumulated_proprio = init_proprio  # (B, 1, D) or None

        all_generated_latents = []
        all_generated_proprio = []

        with torch.no_grad():
            for interact_idx in range(interact_num):
                start_id = interact_idx * (pred_step - 1)
                end_id = start_id + pred_step

                if end_id > T_total:
                    end_id = T_total
                    actual_pred_step = end_id - start_id
                else:
                    actual_pred_step = pred_step

                if actual_pred_step <= 1:
                    print(f"    Warning: Not enough frames remaining, stopping early")
                    break

                # Context frames from accumulated latents
                n_accumulated = accumulated_latents.shape[1]
                n_context = min(num_history, n_accumulated)

                context_latents = accumulated_latents[:, -n_context:]
                context_proprio = accumulated_proprio[:, -n_context:] if accumulated_proprio is not None else None

                # New frames to generate
                gen_new_frames = actual_pred_step - 1
                gen_time_steps = gen_new_frames + n_context - 1

                # Action slicing
                action_context_start = max(0, start_id - n_context + 1)
                action_end = min(start_id + gen_new_frames + 1, T_total)

                step_discrete = discrete_actions[:, action_context_start:action_end] if discrete_actions is not None else None
                step_continuous = continuous_actions[:, action_context_start:action_end] if continuous_actions is not None else None

                # Verify action count
                expected_actions = gen_time_steps + 1
                actual_actions = step_discrete.shape[1] if step_discrete is not None else (
                    step_continuous.shape[1] if step_continuous is not None else 0)
                if actual_actions != expected_actions:
                    gen_time_steps = actual_actions - 1
                    gen_new_frames = gen_time_steps - n_context + 1
                    if gen_new_frames <= 0:
                        print(f"    Warning: Not enough actions, stopping early")
                        break

                print(f"  Interaction {interact_idx + 1}/{interact_num}: "
                      f"context={n_context}, new={gen_new_frames}, "
                      f"actions=[{action_context_start}:{action_end}]")

                gens = self.world_model.generate(
                    time_steps=gen_time_steps,
                    batch_size=B,
                    num_steps=self.num_steps,
                    tasks=None,
                    return_decoded_video=False,
                    return_rewards_per_frame=False,
                    init_latents=context_latents,
                    init_proprio=context_proprio,
                    discrete_actions=step_discrete,
                    continuous_actions=step_continuous,
                    no_shortcut=self.no_shortcut,
                )

                # Extract latents and proprio
                step_proprio = None
                if isinstance(gens, Experience):
                    step_latents = gens.latents
                    step_proprio = gens.proprio
                else:
                    step_latents = gens

                # Fix view dimension mismatch (single-view models remove V dim)
                if step_latents.ndim == 4 and accumulated_latents.ndim == 5:
                    step_latents = step_latents.unsqueeze(2)

                # Extract only NEW frames (skip context)
                new_latents = step_latents[:, n_context:]
                new_proprio = step_proprio[:, n_context:] if step_proprio is not None else None

                # Collect for output
                if interact_idx == 0:
                    all_generated_latents.append(step_latents)  # Include init + generated
                    if step_proprio is not None:
                        all_generated_proprio.append(step_proprio)
                else:
                    all_generated_latents.append(new_latents)
                    if new_proprio is not None:
                        all_generated_proprio.append(new_proprio)

                # Update accumulated context
                accumulated_latents = torch.cat([accumulated_latents, new_latents], dim=1)
                if new_proprio is not None:
                    accumulated_proprio = torch.cat([accumulated_proprio, new_proprio], dim=1) if accumulated_proprio is not None else new_proprio
                elif full_proprio is not None:
                    # Fallback: use GT proprio for accumulation
                    gt_proprio_start = start_id + 1
                    gt_proprio_end = min(start_id + actual_pred_step, T_total)
                    if gt_proprio_end > gt_proprio_start:
                        accumulated_proprio = torch.cat([
                            accumulated_proprio,
                            full_proprio[:, gt_proprio_start:gt_proprio_end]
                        ], dim=1)
                elif accumulated_proprio is not None:
                    # Last resort: repeat last known proprio
                    last_proprio = accumulated_proprio[:, -1:]
                    accumulated_proprio = torch.cat([
                        accumulated_proprio,
                        last_proprio.repeat(1, new_latents.shape[1], 1)
                    ], dim=1)

        generation_time = time.time() - start_time

        all_latents = torch.cat(all_generated_latents, dim=1)
        generated_proprio = torch.cat(all_generated_proprio, dim=1) if all_generated_proprio else None

        return RolloutResult(
            latents=all_latents,
            proprio=generated_proprio,
            generation_time=generation_time,
        )

    # ================================================================
    # Decoding (latent -> pixel)
    # ================================================================

    def _decode_latents_chunked(self, latents_flat: Tensor, decoder) -> Tensor:
        """
        Decode latents in chunks to avoid OOM.

        Args:
            latents_flat: (N, num_tokens, D) flattened latents
            decoder: decoder model

        Returns:
            video_flat: (N, C, H, W) decoded video frames
        """
        chunk_size = self.decode_chunk_size
        decoded_chunks = []

        for i in range(0, latents_flat.shape[0], chunk_size):
            chunk = latents_flat[i:i + chunk_size]
            with torch.no_grad():
                chunk_recon, _ = decoder(chunk)
            decoded_chunks.append(chunk_recon)
            # Clear cache to help with memory fragmentation
            if i > 0 and i % (chunk_size * 4) == 0:
                torch.cuda.empty_cache()

        return torch.cat(decoded_chunks, dim=0)

    def decode_latents(self, latents: Tensor) -> Tensor:
        """
        Decode VLM latents to video frames with chunked decoding to avoid OOM.
        Supports single decoder or dual decoder (view0 vs view1+view2).

        Args:
            latents: (B, T, N, D) for single-view or (B, T, V, N, D) for multi-view

        Returns:
            video: (B, T, V, C, H, W) - always includes view dimension (V=1 for single-view)
                   in range [0, 1]
        """
        B, T = latents.shape[:2]

        # Flatten batch and time (and views if multi-view)
        if latents.ndim == 5:  # (B, T, V, N, D) - multi-view
            V = latents.shape[2]

            if self.use_dual_decoders and self.decoders:
                # Dual decoder mode: decode each view with appropriate decoder (chunked)
                recon_videos = []
                for view_idx in range(V):
                    # Select decoder based on view index
                    if view_idx == 0:
                        decoder = self.decoders.get('view0')
                    else:
                        decoder = self.decoders.get('view1_2')

                    if decoder is None:
                        # Fallback to any available decoder
                        decoder = list(self.decoders.values())[0]

                    # Decode this view in chunks: (B, T, N, D) -> (B*T, C, H, W)
                    view_latents = latents[:, :, view_idx, :, :]  # (B, T, N, D)
                    view_decoder_input = rearrange(view_latents, 'b t n d -> (b t) n d')

                    # Chunked decoding to avoid OOM
                    view_recon = self._decode_latents_chunked(view_decoder_input, decoder)
                    recon_videos.append(view_recon)  # (B*T, C, H, W)

                    # Clear cache between views
                    torch.cuda.empty_cache()

                # Stack views: list of (B*T, C, H, W) -> (B*T, V, C, H, W) -> (B, T, V, C, H, W)
                recon_stacked = torch.stack(recon_videos, dim=1)  # (B*T, V, C, H, W)
                _, _, C, H, W = recon_stacked.shape
                video = recon_stacked.view(B, T, V, C, H, W)
            else:
                # Single decoder mode (chunked)
                latents_flat = rearrange(latents, 'b t v n d -> (b t v) n d')
                decoder = self.decoder if self.decoder is not None else list(self.decoders.values())[0]
                video_flat = self._decode_latents_chunked(latents_flat, decoder)
                _, C, H, W = video_flat.shape
                video = video_flat.view(B, T, V, C, H, W)
        else:  # (B, T, N, D) - single-view
            latents_flat = rearrange(latents, 'b t n d -> (b t) n d')

            # Decode in chunks (use single decoder or first available dual decoder)
            if self.decoder is not None:
                video_flat = self._decode_latents_chunked(latents_flat, self.decoder)
            else:
                decoder = list(self.decoders.values())[0]
                video_flat = self._decode_latents_chunked(latents_flat, decoder)

            # Reshape back to (B, T, 1, C, H, W) - always include view dimension for consistency
            _, C, H, W = video_flat.shape
            video = video_flat.view(B, T, 1, C, H, W)

        # Normalize to [0, 1]
        video = torch.clamp((video + 1) / 2 if video.min() < 0 else video, 0, 1)

        return video
