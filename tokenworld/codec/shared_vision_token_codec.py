"""A small, policy-agnostic bottleneck for StarVLA vision tokens.

The codec deliberately sits *after* the Qwen vision merger and *before* the
language model.  This makes its serialized tokens independent of the StarVLA
action head (OFT, PI, GR00T, ...), while retaining a decoder that maps the
shared token space back to the VLM embedding size.

The legacy codec keeps deterministic adaptive average pooling for the original
baseline.  The symmetric Qwen codec preserves every merged token and compresses
only its channel dimension with three freshly initialized Qwen-style vision
blocks on either side of the bottleneck.  The older PS-VAE-inspired query
codec remains available for reproducing prior baselines.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SharedVisionTokenSpec:
    encoder_model: str
    vision_hidden_dim: int
    merged_hidden_dim: int
    patch_size: int
    spatial_merge_size: int
    tokens_per_view: int
    token_dim: int
    image_size: tuple[int, int] | None = None
    native_tokens_per_view: int | None = None
    codec_type: str = "legacy_mlp"
    variational: bool = False
    pixel_decoder: bool = False
    codec_version: int = 1
    codec_model_dim: int | None = None
    codec_depth: int | None = None
    exported_posterior: str = "deterministic"

    def to_dict(self) -> dict:
        return asdict(self)


class SharedVisionTokenCodec(nn.Module):
    """Compress per-view merged vision embeddings and reconstruct them.

    Input and output are kept as lists because Qwen supports dynamic image
    resolutions.  Shared tokens require a fixed ``tokens_per_view`` but do not
    require every source view to contain the same number of native tokens.
    """

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 256,
        tokens_per_view: int = 16,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or token_dim <= 0 or tokens_per_view <= 0:
            raise ValueError("input_dim, token_dim and tokens_per_view must all be positive")

        hidden_dim = max(token_dim, min(input_dim, int(token_dim * mlp_ratio)))
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.tokens_per_view = int(tokens_per_view)

        self.encoder = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.token_dim),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.input_dim),
        )

    @staticmethod
    def _resize_tokens(tokens: torch.Tensor, target_length: int) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected [N, D] tokens, got shape={tuple(tokens.shape)}")
        if target_length <= 0:
            raise ValueError(f"target_length must be positive, got {target_length}")
        if tokens.shape[0] == target_length:
            return tokens
        # adaptive_avg_pool1d is stable for reduction; linear interpolation is
        # smoother than repeated tokens when reconstructing the native grid.
        channels_first = tokens.transpose(0, 1).unsqueeze(0)
        if target_length < tokens.shape[0]:
            resized = F.adaptive_avg_pool1d(channels_first, target_length)
        else:
            resized = F.interpolate(channels_first, size=target_length, mode="linear", align_corners=False)
        return resized.squeeze(0).transpose(0, 1)

    def encode(self, per_view_features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Return ``[num_images, tokens_per_view, token_dim]`` shared tokens."""
        if not per_view_features:
            raise ValueError("At least one image/view feature tensor is required")
        encoded = []
        encoder_parameter = next(self.encoder.parameters())
        for features in per_view_features:
            if features.ndim != 2 or features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected per-view [N, {self.input_dim}] features, got {tuple(features.shape)}"
                )
            reduced = self._resize_tokens(features, self.tokens_per_view)
            reduced = reduced.to(device=encoder_parameter.device, dtype=encoder_parameter.dtype)
            encoded.append(self.encoder(reduced))
        return torch.stack(encoded, dim=0)

    def decode(
        self,
        shared_tokens: torch.Tensor,
        native_token_counts: Sequence[int],
    ) -> list[torch.Tensor]:
        """Decode shared tokens to one native-length tensor for every image."""
        if shared_tokens.ndim != 3:
            raise ValueError(
                "shared_tokens must have shape [num_images, tokens_per_view, token_dim], "
                f"got {tuple(shared_tokens.shape)}"
            )
        expected = (self.tokens_per_view, self.token_dim)
        if tuple(shared_tokens.shape[1:]) != expected:
            raise ValueError(f"Expected trailing shared-token shape {expected}, got {tuple(shared_tokens.shape[1:])}")
        if len(native_token_counts) != shared_tokens.shape[0]:
            raise ValueError(
                f"Got {shared_tokens.shape[0]} encoded views but {len(native_token_counts)} native token counts"
            )

        decoder_parameter = next(self.decoder.parameters())
        shared_tokens = shared_tokens.to(device=decoder_parameter.device, dtype=decoder_parameter.dtype)
        decoded = self.decoder(shared_tokens)
        return [
            self._resize_tokens(view_tokens, int(native_count))
            for view_tokens, native_count in zip(decoded, native_token_counts)
        ]

    def forward(
        self,
        per_view_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        native_counts = [int(features.shape[0]) for features in per_view_features]
        shared_tokens = self.encode(per_view_features)
        return shared_tokens, self.decode(shared_tokens, native_counts)


class QwenVisionSelfAttention(nn.Module):
    """Qwen3-VL vision-attention topology adapted to a batched fixed grid."""

    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")
        head_dim = hidden_dim // num_heads
        if head_dim % 4 != 0:
            raise ValueError(
                "Qwen 2D rotary attention requires head_dim to be divisible by 4; "
                f"got hidden_dim={hidden_dim}, num_heads={num_heads}, head_dim={head_dim}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.qkv = nn.Linear(self.hidden_dim, 3 * self.hidden_dim, bias=True)
        self.proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

    @staticmethod
    def _rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )

        # Qwen applies RoPE in float32 and casts back to the activation dtype.
        cos = rope_cos[None, None].float()
        sin = rope_sin[None, None].float()
        query_dtype = query.dtype
        key_dtype = key.dtype
        query_float = query.float()
        key_float = key.float()
        query = (
            query_float * cos + self._rotate_half(query_float) * sin
        ).to(query_dtype)
        key = (
            key_float * cos + self._rotate_half(key_float) * sin
        ).to(key_dtype)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.hidden_dim)
        return self.proj(attended)


class QwenVisionMLP(nn.Module):
    """The bias-bearing GELU MLP used by Qwen3-VL vision blocks."""

    def __init__(self, hidden_dim: int, intermediate_dim: int) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(hidden_dim, intermediate_dim, bias=True)
        self.linear_fc2 = nn.Linear(intermediate_dim, hidden_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(F.gelu(self.linear_fc1(hidden_states), approximate="tanh"))


class QwenVisionTransformerBlock(nn.Module):
    """A from-scratch block structurally matching a Qwen3-VL vision block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1.0e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1.0e-6)
        self.attn = QwenVisionSelfAttention(hidden_dim, num_heads)
        self.mlp = QwenVisionMLP(hidden_dim, intermediate_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class SymmetricQwenVisionTokenCodec(nn.Module):
    """Spatial-and-channel semantic VAE for post-merger Qwen embeddings.

    This is the paper-aligned comparison codec requested for the fidelity
    experiment.  The native ``9x12`` path is exactly the original channel-only
    codec.  Smaller latent grids use deterministic 2-D average pooling before
    the posterior and bilinear expansion before the decoder, which changes
    only bottleneck capacity and introduces no learned resampler parameters:

    ``Qwen blocks -> optional 2-D pool -> latent MLP -> optional 2-D expand -> Qwen blocks``.

    The blocks reproduce the Qwen3-VL vision-block topology but are freshly
    initialized and trained as part of the codec.  The compression MLP
    parameterizes a diagonal Gaussian posterior over every 96-channel token.
    """

    codec_type = "symmetric_qwen"
    variational = True
    pixel_decoder_enabled = False

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 96,
        tokens_per_view: int = 108,
        grid_size: Sequence[int] = (9, 12),
        latent_grid_size: Sequence[int] | None = None,
        depth: int = 3,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        projection_hidden_dim: int = 1024,
        rope_theta: float = 10000.0,
        initializer_range: float = 0.02,
        sample_posterior: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        semantic_mse_weight: float = 1.0,
        semantic_cosine_weight: float = 1.0,
        kl_weight: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if min(input_dim, token_dim, tokens_per_view, depth, num_heads, projection_hidden_dim) <= 0:
            raise ValueError("All codec dimensions, depth and head count must be positive")
        if len(grid_size) != 2:
            raise ValueError(f"grid_size must be [height,width], got {grid_size}")
        self.grid_size = tuple(map(int, grid_size))
        self.native_grid_size = self.grid_size
        self.native_tokens_per_view = math.prod(self.native_grid_size)
        self.latent_grid_size = tuple(
            map(int, latent_grid_size if latent_grid_size is not None else grid_size)
        )
        if len(self.latent_grid_size) != 2 or min(self.latent_grid_size) <= 0:
            raise ValueError(
                f"latent_grid_size must be two positive integers, got {self.latent_grid_size}"
            )
        if any(
            latent > native
            for latent, native in zip(self.latent_grid_size, self.native_grid_size)
        ):
            raise ValueError(
                f"latent_grid_size={self.latent_grid_size} cannot exceed native "
                f"grid_size={self.native_grid_size}"
            )
        if math.prod(self.latent_grid_size) != int(tokens_per_view):
            raise ValueError(
                f"latent_grid_size={self.latent_grid_size} contains "
                f"{math.prod(self.latent_grid_size)} tokens, "
                f"but tokens_per_view={tokens_per_view}; token resampling is not allowed"
            )

        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.tokens_per_view = int(tokens_per_view)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.intermediate_dim = int(round(self.input_dim * mlp_ratio))
        self.projection_hidden_dim = int(projection_hidden_dim)
        self.rope_theta = float(rope_theta)
        self.initializer_range = float(initializer_range)
        self.sample_posterior = bool(sample_posterior)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.semantic_mse_weight = float(semantic_mse_weight)
        self.semantic_cosine_weight = float(semantic_cosine_weight)
        self.kl_weight = float(kl_weight)
        if self.logvar_min >= self.logvar_max:
            raise ValueError(
                f"logvar_min={self.logvar_min} must be smaller than logvar_max={self.logvar_max}"
            )
        if min(self.semantic_mse_weight, self.semantic_cosine_weight, self.kl_weight) < 0:
            raise ValueError("Loss weights must be non-negative")

        self.encoder_blocks = nn.ModuleList(
            [
                QwenVisionTransformerBlock(
                    hidden_dim=self.input_dim,
                    num_heads=self.num_heads,
                    intermediate_dim=self.intermediate_dim,
                )
                for _ in range(self.depth)
            ]
        )
        self.posterior_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.projection_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
        )
        self.to_mean = nn.Linear(self.projection_hidden_dim, self.token_dim, bias=True)
        self.to_logvar = nn.Linear(self.projection_hidden_dim, self.token_dim, bias=True)
        self.from_latent = nn.Sequential(
            nn.Linear(self.token_dim, self.projection_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.projection_hidden_dim, self.input_dim, bias=True),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                QwenVisionTransformerBlock(
                    hidden_dim=self.input_dim,
                    num_heads=self.num_heads,
                    intermediate_dim=self.intermediate_dim,
                )
                for _ in range(self.depth)
            ]
        )

        rope_cos, rope_sin = self._make_2d_rope(
            self.native_grid_size,
            head_dim=self.input_dim // self.num_heads,
            theta=self.rope_theta,
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)
        self.apply(self._initialize_qwen_weights)

    def _initialize_qwen_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @staticmethod
    def _make_2d_rope(
        grid_size: tuple[int, int],
        head_dim: int,
        theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )
        rows = torch.arange(grid_size[0], dtype=torch.float32)
        columns = torch.arange(grid_size[1], dtype=torch.float32)
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        row_freqs = torch.outer(row_grid.reshape(-1), inv_freq)
        column_freqs = torch.outer(column_grid.reshape(-1), inv_freq)
        frequencies = torch.cat((row_freqs, column_freqs), dim=-1)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        if embeddings.shape[-1] != head_dim:
            raise ValueError(
                f"2D RoPE produced dim={embeddings.shape[-1]}, expected head_dim={head_dim}"
            )
        return embeddings.cos(), embeddings.sin()

    def _stack_features(self, per_view_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not per_view_features:
            raise ValueError("At least one image/view feature tensor is required")
        parameter = self.posterior_projection[0].weight
        stacked = []
        for features in per_view_features:
            expected = (self.native_tokens_per_view, self.input_dim)
            if features.ndim != 2 or tuple(features.shape) != expected:
                raise ValueError(
                    f"Expected every view to have fixed shape {expected}, got {tuple(features.shape)}; "
                    "the symmetric codec requires the fixed native Qwen grid"
                )
            stacked.append(features.to(device=parameter.device, dtype=parameter.dtype))
        return torch.stack(stacked, dim=0)

    def _pool_to_latent_grid(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.latent_grid_size == self.native_grid_size:
            return hidden_states
        batch, _, channels = hidden_states.shape
        native_height, native_width = self.native_grid_size
        grid = hidden_states.reshape(batch, native_height, native_width, channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        pooled = F.adaptive_avg_pool2d(grid, self.latent_grid_size)
        return pooled.permute(0, 2, 3, 1).reshape(batch, self.tokens_per_view, channels)

    def _expand_to_native_grid(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.latent_grid_size == self.native_grid_size:
            return hidden_states
        batch, _, channels = hidden_states.shape
        latent_height, latent_width = self.latent_grid_size
        grid = hidden_states.reshape(batch, latent_height, latent_width, channels)
        grid = grid.permute(0, 3, 1, 2).contiguous()
        expanded = F.interpolate(
            grid,
            size=self.native_grid_size,
            mode="bilinear",
            align_corners=False,
        )
        return expanded.permute(0, 2, 3, 1).reshape(
            batch, self.native_tokens_per_view, channels
        )

    def encode_with_posterior(
        self,
        per_view_features: Sequence[torch.Tensor],
        sample_posterior: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = self._stack_features(per_view_features)
        for block in self.encoder_blocks:
            hidden_states = block(hidden_states, self.rope_cos, self.rope_sin)
        hidden_states = self._pool_to_latent_grid(hidden_states)
        posterior_hidden = self.posterior_projection(hidden_states)
        mean = self.to_mean(posterior_hidden)
        logvar = self.to_logvar(posterior_hidden).clamp(self.logvar_min, self.logvar_max)
        should_sample = self.training and self.sample_posterior
        if sample_posterior is not None:
            should_sample = bool(sample_posterior)
        tokens = (
            mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
            if should_sample
            else mean
        )
        return tokens, mean, logvar

    def encode(self, per_view_features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sample during training; evaluation/export returns posterior means."""
        tokens, _, _ = self.encode_with_posterior(per_view_features)
        return tokens

    def decode(
        self,
        shared_tokens: torch.Tensor,
        native_token_counts: Sequence[int],
    ) -> list[torch.Tensor]:
        expected = (self.tokens_per_view, self.token_dim)
        if shared_tokens.ndim != 3 or tuple(shared_tokens.shape[1:]) != expected:
            raise ValueError(
                f"Expected shared_tokens [B,{expected[0]},{expected[1]}], "
                f"got {tuple(shared_tokens.shape)}"
            )
        if len(native_token_counts) != shared_tokens.shape[0]:
            raise ValueError("native_token_counts must contain one entry per encoded view")
        if any(int(count) != self.native_tokens_per_view for count in native_token_counts):
            raise ValueError(
                f"All native token counts must remain {self.native_tokens_per_view}; "
                f"got {list(map(int, native_token_counts))}"
            )

        parameter = self.from_latent[0].weight
        hidden_states = self.from_latent(
            shared_tokens.to(device=parameter.device, dtype=parameter.dtype)
        )
        hidden_states = self._expand_to_native_grid(hidden_states)
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, self.rope_cos, self.rope_sin)
        return list(hidden_states.unbind(0))

    def forward_with_auxiliary_losses(
        self,
        per_view_features: Sequence[torch.Tensor],
        pixel_targets: torch.Tensor | None = None,
        detach_pixel_latent: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        del pixel_targets, detach_pixel_latent
        tokens, mean, logvar = self.encode_with_posterior(per_view_features)
        decoded = self.decode(tokens, [self.native_tokens_per_view] * tokens.shape[0])
        teacher = torch.stack(
            [features.detach().float() for features in per_view_features],
            dim=0,
        )
        reconstruction = torch.stack([features.float() for features in decoded], dim=0)
        semantic_mse = F.mse_loss(reconstruction, teacher)
        semantic_cosine = (
            1.0
            - F.cosine_similarity(reconstruction, teacher, dim=-1).mean()
        )

        # Match the LDM AutoencoderKL convention: sum latent dimensions for
        # every sample/view, then average over the batch.  This is materially
        # different from taking one mean over [B,N,D] when lambda=1e-6.
        mean_float = mean.float()
        logvar_float = logvar.float()
        kl_elements = -0.5 * (
            1.0
            + logvar_float
            - mean_float.square()
            - logvar_float.exp()
        )
        kl = kl_elements.flatten(1).sum(dim=1).mean()
        pixel_loss = semantic_mse.new_zeros(())
        losses = {
            "vision_semantic_mse_loss": semantic_mse,
            "vision_semantic_cosine_loss": semantic_cosine,
            "vision_kl_loss": kl,
            "vision_kl_per_dimension": kl_elements.mean().detach(),
            "vision_pixel_loss": pixel_loss,
            "vision_latent_mean": mean_float.mean().detach(),
            "vision_latent_std": mean_float.std(unbiased=False).detach(),
            "vision_posterior_std": torch.exp(0.5 * logvar_float).mean().detach(),
        }
        losses["vision_codec_loss"] = (
            self.semantic_mse_weight * semantic_mse
            + self.semantic_cosine_weight * semantic_cosine
            + self.kl_weight * kl
        )
        return tokens, decoded, losses

    def forward(
        self,
        per_view_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens, decoded, _ = self.forward_with_auxiliary_losses(per_view_features)
        return tokens, decoded


class PSVAESharedVisionTokenCodec(nn.Module):
    """PS-VAE-inspired semantic and pixel-preserving token bottleneck.

    Unlike :class:`SharedVisionTokenCodec`, this codec mixes spatial tokens
    with learned latent queries before applying the channel bottleneck.  A
    symmetric Transformer decoder reconstructs the frozen Qwen merger
    features.  During training, a variational posterior regularizes the shared
    space and an optional lightweight RGB decoder prevents the bottleneck from
    discarding manipulation-relevant geometry.

    The RGB decoder is an auxiliary pretraining head.  Export and policy
    inference use the deterministic posterior mean and require only the
    semantic encoder/decoder.
    """

    codec_type = "ps_vae"

    def __init__(
        self,
        input_dim: int,
        token_dim: int = 96,
        tokens_per_view: int = 16,
        native_tokens_per_view: int = 49,
        model_dim: int = 384,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        variational: bool = True,
        sample_posterior: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        pixel_decoder: bool = True,
        pixel_output_size: int | Sequence[int] = 112,
        pixel_decoder_channels: int = 128,
        semantic_mse_weight: float = 1.0,
        semantic_cosine_weight: float = 1.0,
        kl_weight: float = 1.0e-6,
        pixel_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, token_dim, tokens_per_view, native_tokens_per_view, model_dim, depth) <= 0:
            raise ValueError("All codec dimensions and depth must be positive")
        if model_dim % num_heads != 0:
            raise ValueError(f"model_dim={model_dim} must be divisible by num_heads={num_heads}")
        if pixel_decoder and math.isqrt(tokens_per_view) ** 2 != tokens_per_view:
            raise ValueError("pixel_decoder requires tokens_per_view to form a square latent grid")

        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.tokens_per_view = int(tokens_per_view)
        self.native_tokens_per_view = int(native_tokens_per_view)
        self.model_dim = int(model_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.variational = bool(variational)
        self.sample_posterior = bool(sample_posterior)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.pixel_decoder_enabled = bool(pixel_decoder)
        if isinstance(pixel_output_size, int):
            self.pixel_output_size = (int(pixel_output_size), int(pixel_output_size))
        else:
            if len(pixel_output_size) != 2:
                raise ValueError("pixel_output_size must be an int or [height, width]")
            self.pixel_output_size = tuple(map(int, pixel_output_size))
        if min(self.pixel_output_size) <= 0:
            raise ValueError("pixel_output_size dimensions must be positive")

        self.semantic_mse_weight = float(semantic_mse_weight)
        self.semantic_cosine_weight = float(semantic_cosine_weight)
        self.kl_weight = float(kl_weight)
        self.pixel_weight = float(pixel_weight)

        self.feature_norm = nn.LayerNorm(self.input_dim)
        self.feature_projection = nn.Linear(self.input_dim, self.model_dim)
        self.encoder_native_pos = nn.Parameter(
            torch.randn(1, self.native_tokens_per_view, self.model_dim) * 0.02
        )
        self.latent_queries = nn.Parameter(torch.randn(1, self.tokens_per_view, self.model_dim) * 0.02)

        encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=int(self.model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_encoder = nn.TransformerDecoder(
            encoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(self.model_dim),
        )
        self.to_mean = nn.Linear(self.model_dim, self.token_dim)
        self.to_logvar = nn.Linear(self.model_dim, self.token_dim) if self.variational else None

        self.latent_projection = nn.Linear(self.token_dim, self.model_dim)
        self.decoder_native_queries = nn.Parameter(
            torch.randn(1, self.native_tokens_per_view, self.model_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=int(self.model_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.semantic_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=depth,
            norm=nn.LayerNorm(self.model_dim),
        )
        self.feature_reconstruction = nn.Linear(self.model_dim, self.input_dim)

        if self.pixel_decoder_enabled:
            channels = int(pixel_decoder_channels)
            self.latent_to_pixel = nn.Linear(self.token_dim, channels)
            pixel_blocks = []
            for block_index in range(4):
                out_channels = max(32, channels // (2 if block_index < 2 else 1))
                groups = self._group_count(out_channels)
                pixel_blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
                        nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
                        nn.GroupNorm(groups, out_channels),
                        nn.SiLU(),
                    )
                )
                channels = out_channels
            self.pixel_blocks = nn.ModuleList(pixel_blocks)
            self.to_rgb = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        else:
            self.latent_to_pixel = None
            self.pixel_blocks = nn.ModuleList()
            self.to_rgb = None

    @staticmethod
    def _group_count(channels: int) -> int:
        for groups in (32, 16, 8, 4, 2, 1):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def _resize_sequence_embedding(embedding: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resize learned positions in 2D when possible, otherwise in 1D."""
        if embedding.shape[1] == target_length:
            return embedding
        source_side = math.isqrt(embedding.shape[1])
        target_side = math.isqrt(target_length)
        channels_first = embedding.transpose(1, 2)
        if source_side**2 == embedding.shape[1] and target_side**2 == target_length:
            grid = channels_first.reshape(1, embedding.shape[2], source_side, source_side)
            resized = F.interpolate(grid, size=(target_side, target_side), mode="bicubic", align_corners=False)
            return resized.flatten(2).transpose(1, 2)
        return F.interpolate(channels_first, size=target_length, mode="linear", align_corners=False).transpose(1, 2)

    def _stack_features(self, per_view_features: Sequence[torch.Tensor]) -> torch.Tensor:
        if not per_view_features:
            raise ValueError("At least one image/view feature tensor is required")
        counts = {int(features.shape[0]) for features in per_view_features}
        if len(counts) != 1:
            raise ValueError("PS-VAE batches currently require one fixed native token count per view")
        parameter = self.feature_projection.weight
        normalized = []
        for features in per_view_features:
            if features.ndim != 2 or features.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected per-view [N, {self.input_dim}] features, got {tuple(features.shape)}"
                )
            normalized.append(features.to(device=parameter.device, dtype=parameter.dtype))
        return torch.stack(normalized, dim=0)

    def encode_with_posterior(
        self,
        per_view_features: Sequence[torch.Tensor],
        sample_posterior: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        features = self._stack_features(per_view_features)
        native_pos = self._resize_sequence_embedding(self.encoder_native_pos, features.shape[1])
        memory = self.feature_projection(self.feature_norm(features)) + native_pos
        queries = self.latent_queries.expand(features.shape[0], -1, -1)
        encoded = self.semantic_encoder(tgt=queries, memory=memory)
        mean = self.to_mean(encoded)
        if not self.variational:
            return mean, mean, None

        assert self.to_logvar is not None
        logvar = self.to_logvar(encoded).clamp(self.logvar_min, self.logvar_max)
        should_sample = self.training and self.sample_posterior
        if sample_posterior is not None:
            should_sample = bool(sample_posterior)
        tokens = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if should_sample else mean
        return tokens, mean, logvar

    def encode(self, per_view_features: Sequence[torch.Tensor]) -> torch.Tensor:
        """Encode views; evaluation/export deterministically returns posterior means."""
        tokens, _, _ = self.encode_with_posterior(per_view_features)
        return tokens

    def decode(
        self,
        shared_tokens: torch.Tensor,
        native_token_counts: Sequence[int],
    ) -> list[torch.Tensor]:
        if shared_tokens.ndim != 3:
            raise ValueError(
                "shared_tokens must have shape [num_images, tokens_per_view, token_dim], "
                f"got {tuple(shared_tokens.shape)}"
            )
        if tuple(shared_tokens.shape[1:]) != (self.tokens_per_view, self.token_dim):
            raise ValueError(
                f"Expected trailing token shape {(self.tokens_per_view, self.token_dim)}, "
                f"got {tuple(shared_tokens.shape[1:])}"
            )
        if len(native_token_counts) != shared_tokens.shape[0]:
            raise ValueError("native_token_counts must contain one entry per encoded view")

        parameter = self.latent_projection.weight
        shared_tokens = shared_tokens.to(device=parameter.device, dtype=parameter.dtype)
        memory = self.latent_projection(shared_tokens)
        decoded = []
        for view_index, native_count in enumerate(native_token_counts):
            queries = self._resize_sequence_embedding(self.decoder_native_queries, int(native_count))
            hidden = self.semantic_decoder(tgt=queries, memory=memory[view_index : view_index + 1])
            decoded.append(self.feature_reconstruction(hidden).squeeze(0))
        return decoded

    def reconstruct_pixels(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        if not self.pixel_decoder_enabled or self.latent_to_pixel is None or self.to_rgb is None:
            raise RuntimeError("Pixel decoder is disabled")
        side = math.isqrt(self.tokens_per_view)
        hidden = self.latent_to_pixel(shared_tokens)
        hidden = hidden.transpose(1, 2).reshape(shared_tokens.shape[0], -1, side, side)
        scale = 2 ** len(self.pixel_blocks)
        initial_size = (
            max(side, math.ceil(self.pixel_output_size[0] / scale)),
            max(side, math.ceil(self.pixel_output_size[1] / scale)),
        )
        if hidden.shape[-2:] != initial_size:
            hidden = F.interpolate(hidden, size=initial_size, mode="bilinear", align_corners=False)
        for block in self.pixel_blocks:
            hidden = block(hidden)
        pixels = torch.sigmoid(self.to_rgb(hidden))
        if pixels.shape[-2:] != self.pixel_output_size:
            pixels = F.interpolate(
                pixels,
                size=self.pixel_output_size,
                mode="bilinear",
                align_corners=False,
            )
        return pixels

    def forward_with_auxiliary_losses(
        self,
        per_view_features: Sequence[torch.Tensor],
        pixel_targets: torch.Tensor | None = None,
        detach_pixel_latent: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
        native_counts = [int(features.shape[0]) for features in per_view_features]
        shared_tokens, mean, logvar = self.encode_with_posterior(per_view_features)
        decoded = self.decode(shared_tokens, native_counts)

        teacher = torch.cat([features.detach().float() for features in per_view_features], dim=0)
        reconstruction = torch.cat([features.float() for features in decoded], dim=0)
        semantic_mse = F.mse_loss(reconstruction, teacher)
        semantic_cosine = 1.0 - F.cosine_similarity(reconstruction, teacher, dim=-1).mean()
        if logvar is None:
            kl = mean.float().new_zeros(())
        else:
            mean_float = mean.float()
            logvar_float = logvar.float()
            kl = -0.5 * (1.0 + logvar_float - mean_float.square() - logvar_float.exp()).mean()

        losses = {
            "vision_semantic_mse_loss": semantic_mse,
            "vision_semantic_cosine_loss": semantic_cosine,
            "vision_kl_loss": kl,
            "vision_latent_mean": mean.float().mean().detach(),
            "vision_latent_std": mean.float().std(unbiased=False).detach(),
        }
        pixel_loss = mean.float().new_zeros(())
        if self.pixel_decoder_enabled and pixel_targets is not None:
            pixel_latent = shared_tokens.detach() if detach_pixel_latent else shared_tokens
            prediction = self.reconstruct_pixels(pixel_latent).float()
            target = pixel_targets.to(device=prediction.device, dtype=prediction.dtype)
            if target.shape[-2:] != prediction.shape[-2:]:
                target = F.interpolate(target, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
            pixel_loss = F.l1_loss(prediction, target)
        losses["vision_pixel_loss"] = pixel_loss
        losses["vision_codec_loss"] = (
            self.semantic_mse_weight * semantic_mse
            + self.semantic_cosine_weight * semantic_cosine
            + self.kl_weight * kl
            + self.pixel_weight * pixel_loss
        )
        return shared_tokens, decoded, losses

    def forward(
        self,
        per_view_features: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        tokens, decoded, _ = self.forward_with_auxiliary_losses(per_view_features)
        return tokens, decoded
