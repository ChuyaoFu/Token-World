"""Decoder-only loader for symmetric Qwen S-VAE checkpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenworld.codec.shared_vision_token_codec import QwenVisionTransformerBlock


class SymmetricQwenVisionTokenDecoder(nn.Module):
    """The frozen decode half of ``SymmetricQwenVisionTokenCodec``."""

    def __init__(
        self,
        *,
        input_dim: int,
        token_dim: int,
        tokens_per_view: int,
        grid_size: Sequence[int],
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        projection_hidden_dim: int = 1024,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.tokens_per_view = int(tokens_per_view)
        self.grid_size = tuple(map(int, grid_size))
        if math.prod(self.grid_size) != self.tokens_per_view:
            raise ValueError(
                f"grid_size={self.grid_size} does not match "
                f"tokens_per_view={self.tokens_per_view}"
            )
        intermediate_dim = int(round(self.input_dim * float(mlp_ratio)))
        self.from_latent = nn.Sequential(
            nn.Linear(self.token_dim, int(projection_hidden_dim), bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(projection_hidden_dim), self.input_dim, bias=True),
        )
        self.decoder_blocks = nn.ModuleList(
            [
                QwenVisionTransformerBlock(
                    hidden_dim=self.input_dim,
                    num_heads=int(num_heads),
                    intermediate_dim=intermediate_dim,
                )
                for _ in range(int(depth))
            ]
        )
        rope_cos, rope_sin = self._make_2d_rope(
            self.grid_size,
            head_dim=self.input_dim // int(num_heads),
            theta=float(rope_theta),
        )
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    @staticmethod
    def _make_2d_rope(
        grid_size: tuple[int, int],
        head_dim: int,
        theta: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary_dim = head_dim // 2
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
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
                f"2D RoPE produced dim={embeddings.shape[-1]}, expected {head_dim}"
            )
        return embeddings.cos(), embeddings.sin()

    def forward(self, shared_tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.tokens_per_view, self.token_dim)
        if shared_tokens.ndim != 3 or tuple(shared_tokens.shape[1:]) != expected:
            raise ValueError(
                f"Expected shared_tokens [B,{expected[0]},{expected[1]}], "
                f"got {tuple(shared_tokens.shape)}"
            )
        parameter = self.from_latent[0].weight
        hidden_states = self.from_latent(
            shared_tokens.to(device=parameter.device, dtype=parameter.dtype)
        )
        for block in self.decoder_blocks:
            hidden_states = block(hidden_states, self.rope_cos, self.rope_sin)
        return hidden_states


def _load_checkpoint_payload(path: Path) -> dict:
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:  # Older PPU PyTorch builds do not expose mmap/weights_only.
        kwargs.pop("weights_only", None)
        return torch.load(path, map_location="cpu")


def load_symmetric_qwen_decoder(
    checkpoint_path: str | Path,
    device: torch.device | str,
) -> tuple[SymmetricQwenVisionTokenDecoder, dict]:
    """Load only the S-VAE decoder weights needed for cache materialization."""

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing S-VAE codec checkpoint: {checkpoint_path}")
    payload = _load_checkpoint_payload(checkpoint_path)
    if int(payload.get("version", -1)) != 1:
        raise ValueError(f"Unsupported codec checkpoint version in {checkpoint_path}")
    config = dict(payload["codec_config"])
    if config.get("method") != "symmetric_qwen":
        raise ValueError(
            "Decoder cache generation currently requires a symmetric_qwen codec, "
            f"got {config.get('method')!r}"
        )

    decoder = SymmetricQwenVisionTokenDecoder(
        input_dim=int(config["input_dim"]),
        token_dim=int(config["token_dim"]),
        tokens_per_view=int(config["tokens_per_view"]),
        grid_size=config["grid_size"],
        depth=int(config["depth"]),
        num_heads=int(config["num_heads"]),
        mlp_ratio=float(config.get("mlp_ratio", 4.0)),
        projection_hidden_dim=int(config.get("projection_hidden_dim", 1024)),
        rope_theta=float(config.get("rope_theta", 10000.0)),
    )
    prefixes = ("from_latent.", "decoder_blocks.")
    decoder_state = {
        key: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefixes)
    }
    decoder.load_state_dict(decoder_state, strict=True)
    stat = checkpoint_path.stat()
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "codec_config": config,
    }
    del payload, decoder_state
    decoder.requires_grad_(False).eval()
    return decoder.to(device), metadata
