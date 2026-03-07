"""Transformer-based MIL classifier — SurGen paper baseline.

Architecture (per the paper):
    patch embeddings  (N × input_dim)
    → Linear(input_dim, hidden_dim) + ReLU
    → TransformerEncoder(num_layers, num_heads, ffn_dim, dropout)
    → aggregation over N patches  →  (hidden_dim,)
    → Linear(hidden_dim, 1)       →  logit scalar

Aggregation modes (config key: ``aggregation``):
    "mean"      — arithmetic mean pool (Phase 5 baseline, default)
    "attention" — ABMIL with task-specific 2-layer GELU attention scorer (Option B):
                  Shared:  LayerNorm(x) → Linear(hidden_dim, attn_hidden_dim) → GELU
                  Per-task: Linear(attn_hidden_dim, T) → softmax (over patches)
                  → T independent weighted sums → T independent classifiers.
                  Scores are computed on normalized tokens; pooling uses raw tokens.

Pair the logit output with nn.BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn


class MILTransformer(nn.Module):
    """Multiple-Instance Learning transformer for WSI patch embeddings.

    Unified modular pattern (T=1 special case of MultiMILTransformer):
        - attn_l1: shared Linear(hidden_dim, attn_hidden_dim) + GELU
        - attn_l2: Linear(attn_hidden_dim, 1)  — single attention distribution
        - classifier: ModuleList([Linear(hidden_dim, 1)])

    Args:
        input_dim:       Patch embedding dimension (1024 for UNI).
        hidden_dim:      Projected dimension fed into the transformer (512).
        num_layers:      Number of TransformerEncoder layers (2).
        num_heads:       Attention heads per layer (2).
        ffn_dim:         Feedforward network dimension inside each layer (2048).
        dropout:         Dropout probability applied inside the transformer (0.15).
        layer_norm_eps:  Epsilon for LayerNorm (1e-5, per Table 5 of the paper).
        aggregation:     "mean" (default) or "attention" (ABMIL task-specific L2).
        attn_hidden_dim: Hidden dim of the attention MLP L1 (128). Only used
                         when aggregation="attention".
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.15,
        layer_norm_eps: float = 1e-5,
        aggregation: str = "mean",
        attn_hidden_dim: int = 128,
        positional_encoding: str = "none",
    ):
        super().__init__()
        self.aggregation = aggregation
        self.output_classes = 1

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,  # expects (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if positional_encoding == "sinusoidal":
            from scripts.models.layers import SinusoidalPositionalEncoding2D
            self.pos_enc = SinusoidalPositionalEncoding2D(hidden_dim)
        else:
            self.pos_enc = None

        if aggregation == "attention":
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_l1   = nn.Sequential(nn.Linear(hidden_dim, attn_hidden_dim), nn.GELU())
            self.attn_l2   = nn.Linear(attn_hidden_dim, 1)

        self.classifier = nn.ModuleList([nn.Linear(hidden_dim, 1)])

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None, return_weights: bool = False):
        """Forward pass.

        Args:
            x:              Patch embeddings of shape (batch, N_patches, input_dim).
                            During training batch=1 because each slide has a variable
                            number of patches and no padding is applied.
            coords:         Optional (batch, N_patches, 2) long tensor of (row, col)
                            patch indices. Used by sinusoidal PE when enabled.
            return_weights: If True, return (logits, attn_weights) instead of logits.
                            Only meaningful when aggregation="attention".

        Returns:
            Logit tensor of shape (batch,), or tuple (logits, weights) when
            return_weights=True. weights shape: (batch, N_patches, 1).
        """
        x = self.input_proj(x)   # (B, N, hidden_dim)
        if self.pos_enc is not None and coords is not None:
            x = x + self.pos_enc(coords)
        x = self.transformer(x)  # (B, N, hidden_dim)

        if self.aggregation == "attention":
            h       = self.attn_l1(self.attn_norm(x))  # (B, N, attn_hidden_dim)
            scores  = self.attn_l2(h)                   # (B, N, 1)
            weights = torch.softmax(scores, dim=1)       # (B, N, 1)
            pooled  = (weights * x).sum(dim=1)           # (B, hidden_dim)
        else:
            pooled  = x.mean(dim=1)                      # (B, hidden_dim)
            weights = None

        logits = self.classifier[0](pooled).squeeze(-1)  # (B,)
        if return_weights:
            return logits, weights
        return logits


class MultiMILTransformer(nn.Module):
    """Multi-task MIL transformer with task-specific attention (Option B).

    Shared backbone through the transformer encoder, then task-specific attention:
        - attn_l1: shared Linear(hidden_dim, attn_hidden_dim) + GELU
          learns a general patch-importance representation
        - attn_l2: Linear(attn_hidden_dim, T) produces T independent attention
          distributions at a cost of only +T*attn_hidden_dim parameters
        - classifier: ModuleList of T independent Linear(hidden_dim, 1) heads

    This allows each task to attend to different patches, rather than forcing
    all tasks to share a single attention distribution.

    Args:
        output_classes:  Number of independent binary tasks (e.g. 3 for MMR+RAS+BRAF).
        aggregation:     "mean" (default) or "attention" (ABMIL).
        attn_variant:    "split" (default) — task-specific L2, each task attends to
                         different patches (Linear(attn_hidden_dim, T) → T distributions);
                         "joined" — shared L2, all tasks attend to the same patches
                         (Linear(attn_hidden_dim, 1) → 1 distribution broadcast to all tasks).
                         Only meaningful when aggregation="attention".
        attn_hidden_dim: Hidden dim of the attention MLP L1 (128). Only used
                         when aggregation="attention".
        All other args:  same as MILTransformer.

    Returns:
        Logit tensor of shape (batch, output_classes).
        With return_weights=True: tuple (logits, weights) where
        weights shape: (batch, N_patches, output_classes) for "split",
        or (batch, N_patches, 1) for "joined".
    """

    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, num_heads=2,
                 ffn_dim=2048, dropout=0.15, layer_norm_eps=1e-5, output_classes=3,
                 aggregation="mean", attn_variant="split", attn_hidden_dim=128,
                 positional_encoding="none"):
        super().__init__()
        self.aggregation = aggregation
        self.attn_variant = attn_variant
        self.output_classes = output_classes
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, layer_norm_eps=layer_norm_eps, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        if positional_encoding == "sinusoidal":
            from scripts.models.layers import SinusoidalPositionalEncoding2D
            self.pos_enc = SinusoidalPositionalEncoding2D(hidden_dim)
        else:
            self.pos_enc = None

        if aggregation == "attention":
            self.attn_norm = nn.LayerNorm(hidden_dim)
            self.attn_l1   = nn.Sequential(nn.Linear(hidden_dim, attn_hidden_dim), nn.GELU())
            # "split": one score per task → T independent attention distributions
            # "joined": one shared score → single distribution broadcast to all tasks
            attn_out_dim = output_classes if attn_variant == "split" else 1
            self.attn_l2 = nn.Linear(attn_hidden_dim, attn_out_dim)

        self.classifier = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(output_classes)])

    def forward(self, x, coords: torch.Tensor = None, return_weights: bool = False):
        """Forward pass.

        Args:
            x:              Patch embeddings of shape (batch, N_patches, input_dim).
            coords:         Optional (batch, N_patches, 2) long tensor of (row, col)
                            patch indices. Used by sinusoidal PE when enabled.
            return_weights: If True, return (logits, attn_weights) instead of logits.
                            Only meaningful when aggregation="attention".

        Returns:
            Logit tensor of shape (batch, output_classes), or tuple (logits, weights)
            when return_weights=True. weights shape: (batch, N_patches, output_classes).
        """
        x = self.input_proj(x)   # (B, N, hidden_dim)
        if self.pos_enc is not None and coords is not None:
            x = x + self.pos_enc(coords)
        x = self.transformer(x)  # (B, N, hidden_dim)

        T = len(self.classifier)

        if self.aggregation == "attention":
            h      = self.attn_l1(self.attn_norm(x))   # (B, N, attn_hidden_dim)
            scores = self.attn_l2(h)                    # (B, N, T) or (B, N, 1)
            weights = torch.softmax(scores, dim=1)      # (B, N, T) or (B, N, 1)
            if self.attn_variant == "joined":
                # broadcast single distribution: all tasks pool identical patches
                pooled = (weights * x).sum(dim=1)               # (B, hidden_dim)
                pooled = pooled.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            else:
                # split: each task has its own distribution
                pooled = (weights.unsqueeze(-1) * x.unsqueeze(2)).sum(dim=1)  # (B, T, hidden_dim)
        else:
            pooled  = x.mean(dim=1).unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)
            weights = None

        logits = torch.stack(
            [self.classifier[t](pooled[:, t, :]).squeeze(-1) for t in range(T)],
            dim=1,
        )  # (B, T)

        if return_weights:
            return logits, weights
        return logits
