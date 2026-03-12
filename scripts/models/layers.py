"""Reusable neural network layer building blocks."""

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalPositionalEncoding2D(nn.Module):
    """Absolute 2D sinusoidal positional encoding for patch grids.

    First d_model//2 dimensions encode the row (y) coordinate;
    the remaining d_model//2 dimensions encode the column (x) coordinate.
    Each half uses the standard 1D sinusoidal scheme over d_half//2 frequency bands.

    Args:
        d_model: Embedding dimension. Must be even.
    """

    def __init__(self, d_model: int):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for 2D sinusoidal PE"
        self.d_model = d_model
        d_half = d_model // 2
        # div_term shape: (d_half // 2,)
        self.register_buffer(
            "div_term",
            torch.exp(torch.arange(0, d_half, 2).float() * -(math.log(10000.0) / d_half)),
        )

    def forward(self, coords: Tensor) -> Tensor:
        """Compute 2D sinusoidal positional encodings.

        Args:
            coords: (B, N, 2) long tensor of (row, col) patch grid indices.
        Returns:
            pe: (B, N, d_model) float32 PE tensor on the same device as coords.
        """
        B, N, _ = coords.shape
        d_half = self.d_model // 2
        rows = coords[..., 0].float()  # (B, N)
        cols = coords[..., 1].float()  # (B, N)

        pe = torch.zeros(B, N, self.d_model, device=coords.device, dtype=torch.float32)
        div = self.div_term  # (d_half // 2,)

        pe[..., 0:d_half:2]    = torch.sin(rows.unsqueeze(-1) * div)
        pe[..., 1:d_half:2]    = torch.cos(rows.unsqueeze(-1) * div)
        pe[..., d_half::2]     = torch.sin(cols.unsqueeze(-1) * div)
        pe[..., d_half + 1::2] = torch.cos(cols.unsqueeze(-1) * div)

        return pe


class MLPRelativePositionBias(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # Single linear projection: (B, N, N, 2) → (B, N, N, num_heads).
        # A 2-layer MLP would expand to (B, N, N, hidden_dim), which OOMs on the
        # T4 at ~6k patches (hidden_dim=64 → 9.5 GB intermediate). The linear
        # projection peaks at ~600 MB for the same N and is still learnable.
        self.mlp = nn.Linear(2, num_heads)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, N, 2) — (row, col) pixel-space patch coords
        # returns: (B*num_heads, N, N) additive attention bias
        B, N, _ = coords.shape
        rel = coords.unsqueeze(2) - coords.unsqueeze(1)   # (B, N, N, 2)
        scale = rel.abs().amax(dim=(1,2,3), keepdim=True).clamp(min=1e-6)
        rel = rel / scale                                  # normalize to [-1, 1] per slide
        bias = self.mlp(rel)                               # (B, N, N, num_heads)
        bias = bias.permute(0, 3, 1, 2).contiguous()      # (B, num_heads, N, N)
        return bias.view(B * self.num_heads, N, N)         # (B*H, N, N)
