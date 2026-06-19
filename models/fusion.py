"""
models/fusion.py
=================
Late-fusion model combining a frozen 1D backbone and a frozen 2D backbone.
Both backbones output a 256-d embedding; embeddings are concatenated and
passed through a lightweight fusion head.

Usage:
    from models.fusion import LateFusionModel
    model = LateFusionModel(model_1d, model_2d, num_classes=5, head_type="mlp")
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionHead(nn.Module):
    """Lightweight classification head on top of concatenated embeddings."""

    def __init__(self, in_dim: int = 512, num_classes: int = 5,
                 head_type: str = "mlp", dropout: float = 0.1):
        super().__init__()
        self.head_type = head_type

        if head_type == "linear":
            self.net = nn.Linear(in_dim, num_classes)

        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )

        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LateFusionModel(nn.Module):
    """
    Combines a frozen 1D backbone and a frozen 2D backbone via late fusion.

    Both backbones must expose an `embed(x) -> (B, 256)` method, or we fall
    back to calling the model and assuming its penultimate output is the
    embedding (see `_get_embedding`).
    """

    def __init__(self, model_1d: nn.Module, model_2d: nn.Module,
                 num_classes: int = 5, head_type: str = "mlp",
                 dim_1d: int | None = None, dim_2d: int | None = None,
                 freeze_backbones: bool = True):
        super().__init__()
        self.model_1d = model_1d
        self.model_2d = model_2d

        # Auto-detect embedding dims with a dummy forward pass if not given.
        if dim_1d is None or dim_2d is None:
            dim_1d, dim_2d = self._infer_dims()

        self.head = FusionHead(
            in_dim=dim_1d + dim_2d,
            num_classes=num_classes,
            head_type=head_type,
        )

        if freeze_backbones:
            for p in self.model_1d.parameters():
                p.requires_grad = False
            for p in self.model_2d.parameters():
                p.requires_grad = False
            self.model_1d.eval()
            self.model_2d.eval()

    def _infer_dims(self) -> tuple[int, int]:
        """Probe both backbones with a dummy input to get embedding sizes."""
        device = next(self.model_1d.parameters()).device
        with torch.no_grad():
            dummy_1d = torch.zeros(1, 12, 1000, device=device)   # (B, leads, T)
            dummy_2d = torch.zeros(1, 12, 224, 224, device=device)  # (B, leads, H, W)
            e1 = self._get_embedding(self.model_1d, dummy_1d)
            e2 = self._get_embedding(self.model_2d, dummy_2d)
        return e1.shape[1], e2.shape[1]

    def _get_embedding(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 256-d embedding from a backbone.
        Assumes the model has an `.embed()` method; if not, assumes the
        model's forward pass returns (embedding, logits) or just logits
        with a separate `.embedding` buffer set during forward.
        """
        if hasattr(model, "embed"):
            return model.embed(x)
        # Fallback: assume model.forward sets self.last_embedding
        _ = model(x)
        if hasattr(model, "last_embedding"):
            return model.last_embedding
        raise AttributeError(
            "Backbone has no `.embed()` method and no `.last_embedding` "
            "attribute. Add one of these to extract embeddings."
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            e1 = self._get_embedding(self.model_1d, x_1d)
            e2 = self._get_embedding(self.model_2d, x_2d)
        e = torch.cat([e1, e2], dim=1)  # (B, 512)
        return self.head(e)
