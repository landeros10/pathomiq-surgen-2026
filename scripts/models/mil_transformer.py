"""Transformer-based MIL classifier — SurGen paper baseline.

Architecture (per the paper):
    patch embeddings  (N × input_dim)
    → Linear(input_dim, hidden_dim) + ReLU
    → TransformerEncoder(num_layers, num_heads, ffn_dim, dropout)
    → mean pooling over N patches  →  (hidden_dim,)
    → Linear(hidden_dim, 1)        →  logit scalar

Pair the logit output with nn.BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn


class MILTransformer(nn.Module):
    """Multiple-Instance Learning transformer for WSI patch embeddings.

    Args:
        input_dim:   Patch embedding dimension (1024 for UNI).
        hidden_dim:  Projected dimension fed into the transformer (512).
        num_layers:  Number of TransformerEncoder layers (2).
        num_heads:   Attention heads per layer (2).
        ffn_dim:     Feedforward network dimension inside each layer (2048).
        dropout:     Dropout probability applied inside the transformer (0.15).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.15,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,  # expects (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Patch embeddings of shape (batch, N_patches, input_dim).
               During training batch=1 because each slide has a variable
               number of patches and no padding is applied.

        Returns:
            Logit tensor of shape (batch,).
        """
        x = self.input_proj(x)          # (B, N, hidden_dim)
        x = self.transformer(x)         # (B, N, hidden_dim)
        x = x.mean(dim=1)              # (B, hidden_dim) — mean pool over patches
        return self.classifier(x).squeeze(-1)  # (B,)
