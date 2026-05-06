"""
models/heads.py
===============
Frozen / fine-tune senaryolarında backbone üstüne eklenen sınıflandırma kafaları.
"""

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Basit lineer projeksiyon."""

    def __init__(self, emb_dim: int = 256, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class MLPHead(nn.Module):
    """Tek gizli katmanlı MLP + dropout."""

    def __init__(
        self,
        emb_dim: int = 256,
        hidden: int = 256,
        num_classes: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


# ── KAN (piecewise-linear spline) ─────────────────────────────────────────────

class SplineLinear(nn.Module):
    """
    Piecewise-linear (triangular basis) genişletme + lineer birleştirme.
    KAN'ın öğrenilebilir aktivasyon fonksiyonu yerine kullanılan basit versiyon.
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 16):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size

        knots = torch.linspace(-1.0, 1.0, grid_size)
        self.register_buffer("knots", knots)
        self.register_buffer("delta", torch.tensor(2.0 / (grid_size - 1)))

        # (out_features, in_features, grid_size)
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        # triangular basis: phi_k(x_i) = max(1 - |x_i - knot_k| / delta, 0)
        phi = torch.clamp(
            1.0 - torch.abs(x.unsqueeze(-1) - self.knots) / self.delta,
            min=0.0,
        )  # (B, in_features, grid_size)
        return torch.einsum("bik,oik->bo", phi, self.weight) + self.bias


class KANHead(nn.Module):
    """
    Tek katmanlı KAN kafası.

    Girişi tanh ile [-1,1] aralığına normalize ettikten sonra
    SplineLinear uygular.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        num_classes: int = 5,
        grid_size: int = 16,
        scale: float = 2.0,
    ):
        super().__init__()
        self.scale  = scale
        self.spline = SplineLinear(emb_dim, num_classes, grid_size)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(emb / self.scale)
        return self.spline(z)
