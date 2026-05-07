"""
models/heads.py
===============
Frozen / partial fine-tune senaryolarında backbone üstüne takılan kafalar.
Hepsi (B, emb_dim) → (B, num_classes) logit üretir.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class LinearHead(nn.Module):
    def __init__(self, emb_dim: int = 256, num_classes: int = 5) -> None:
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(
        self,
        emb_dim: int = 256,
        hidden: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── KAN head (piecewise-linear spline) ───────────────────────────────────────

class _SplineLinear(nn.Module):
    """
    Triangular-basis (piecewise-linear) spline genişletme + lineer birleştirme.
    KAN'ın basitleştirilmiş 1-katman versiyonu.
    """

    def __init__(self, in_f: int, out_f: int, grid_size: int = 16) -> None:
        super().__init__()
        knots = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("knots", knots)
        self.register_buffer("delta", torch.tensor(2.0 / (grid_size - 1)))
        # (out_f, in_f, grid_size)
        self.weight = nn.Parameter(torch.randn(out_f, in_f, grid_size) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # phi_k(x_i) = max(1 - |x_i - knot_k| / delta, 0)
        phi = torch.clamp(
            1.0 - torch.abs(x.unsqueeze(-1) - self.knots) / self.delta,
            min=0.0,
        )  # (B, in_f, grid_size)
        return torch.einsum("big,oig->bo", phi, self.weight) + self.bias


class KANHead(nn.Module):
    """
    Girişi tanh ile [-1, 1]'e normalize eder, ardından SplineLinear uygular.

    Parameters
    ----------
    emb_dim   : embedding boyutu (FeatureExtractor çıkışı)
    num_classes
    grid_size : spline düğüm sayısı
    scale     : tanh(x / scale) — büyük değer → girişi daha dar sıkıştırır
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 5,
        grid_size: int = 16,
        scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.scale  = scale
        self.spline = _SplineLinear(emb_dim, num_classes, grid_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spline(torch.tanh(x / self.scale))
