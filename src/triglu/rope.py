"""Rotary position embeddings used by attention and TriGLU."""

from __future__ import annotations

import torch
from torch import nn


class RopeModule(nn.Module):
    """Apply RoPE to the last axis, with sequence on the penultimate axis.

    This shape convention supports both attention tensors ``[B, H, T, D]`` and
    full-width TriGLU tensors ``[B, T, C]``.  Rotary pairs are adjacent
    elements: ``(0, 1), (2, 3), ...``.
    """

    def __init__(self, dim: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be a positive even integer, got {dim}")
        if theta <= 0.0:
            raise ValueError(f"RoPE theta must be positive, got {theta}")

        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"RoPE input must have at least two dimensions, got {x.shape}")
        if x.size(-1) != self.dim:
            raise ValueError(
                f"RoPE expected last dimension {self.dim}, got {x.size(-1)}"
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"RoPE offset must be a non-negative integer, got {offset!r}")

        seq_len = x.size(-2)
        positions = torch.arange(
            offset,
            offset + seq_len,
            device=x.device,
            dtype=torch.float32,
        )
        # The angle math must stay float32 even when the module or its input
        # runs in bfloat16/half: at large offsets a low-precision frequency
        # table corrupts the rotation silently.  The frequencies are therefore
        # derived here rather than kept as a registered buffer, which
        # module-wide dtype conversion (`.to(torch.bfloat16)`) would downcast.
        inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, device=x.device, dtype=torch.float32)
                / self.dim
            )
        )
        frequencies = torch.outer(positions, inv_freq)
        cos = frequencies.cos().to(dtype=x.dtype)
        sin = frequencies.sin().to(dtype=x.dtype)

        # Broadcast over every leading dimension and repeat each angle over its
        # adjacent even/odd rotary pair.
        broadcast_shape = [1] * (x.ndim - 2) + [seq_len, self.dim]
        cos = cos.repeat_interleave(2, dim=-1).view(broadcast_shape)
        sin = sin.repeat_interleave(2, dim=-1).view(broadcast_shape)

        x_pairs = x.reshape(*x.shape[:-1], self.dim // 2, 2)
        x_rotated = torch.stack((-x_pairs[..., 1], x_pairs[..., 0]), dim=-1)
        x_rotated = x_rotated.flatten(-2)
        return x * cos + x_rotated * sin


# A descriptive alias for callers that prefer the longer name.
RotaryEmbedding = RopeModule

