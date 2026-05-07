"""
models/backbones.py
===================
1D EKG backbone modelleri:
  ResNet1D      — standart residual bloklar
  SEResNet1D    — squeeze-and-excitation ekli versiyon
  InceptionTime1D — çok ölçekli inception blokları

Her backbone (B, 12, T) → (B, num_classes) logit üretir.
Frozen/partial-ft senaryolarında registry.py fc'yi Identity() ile değiştirir.
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ═════════════════════════════════════════════════════════════════════════════
# ResNet1D
# ═════════════════════════════════════════════════════════════════════════════

class _BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


class ResNet1D(nn.Module):
    """
    Parameters
    ----------
    in_ch       : giriş kanal sayısı (12)
    num_classes : sınıf sayısı
    layers      : her stage blok sayıları, örn. (3,4,6,3) = ResNet-34
    base_ch     : ilk stage kanal sayısı
    """

    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        layers: tuple[int, ...] = (3, 4, 6, 3),
        base_ch: int = 64,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self._ch = base_ch
        self.layer1 = self._stage(base_ch * 1, layers[0], stride=1)
        self.layer2 = self._stage(base_ch * 2, layers[1], stride=2)
        self.layer3 = self._stage(base_ch * 4, layers[2], stride=2)
        self.layer4 = self._stage(base_ch * 8, layers[3], stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(base_ch * 8, num_classes)

    def _stage(self, out_ch: int, n: int, stride: int) -> nn.Sequential:
        blocks = [_BasicBlock(self._ch, out_ch, stride)]
        self._ch = out_ch
        for _ in range(1, n):
            blocks.append(_BasicBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))


# ═════════════════════════════════════════════════════════════════════════════
# SEResNet1D
# ═════════════════════════════════════════════════════════════════════════════

class _SEBlock(nn.Module):
    def __init__(self, ch: int, r: int = 16) -> None:
        super().__init__()
        h = max(1, ch // r)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(ch, h), nn.ReLU(inplace=True),
            nn.Linear(h, ch), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x).squeeze(-1)
        return x * self.fc(s).unsqueeze(-1)


class _SEBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, r: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.se    = _SEBlock(out_ch, r)
        self.skip  = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + self.skip(x))


class SEResNet1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        layers: tuple[int, ...] = (3, 4, 6, 3),
        base_ch: int = 64,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self._ch = base_ch
        self.layer1 = self._stage(base_ch * 1, layers[0], 1, se_reduction)
        self.layer2 = self._stage(base_ch * 2, layers[1], 2, se_reduction)
        self.layer3 = self._stage(base_ch * 4, layers[2], 2, se_reduction)
        self.layer4 = self._stage(base_ch * 8, layers[3], 2, se_reduction)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(base_ch * 8, num_classes)

    def _stage(self, out_ch: int, n: int, stride: int, r: int) -> nn.Sequential:
        blocks = [_SEBasicBlock(self._ch, out_ch, stride, r)]
        self._ch = out_ch
        for _ in range(1, n):
            blocks.append(_SEBasicBlock(out_ch, out_ch, r=r))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))


# ═════════════════════════════════════════════════════════════════════════════
# InceptionTime1D
# ═════════════════════════════════════════════════════════════════════════════

class _InceptionModule(nn.Module):
    """Tek inception modülü: 3 multi-scale branch + maxpool branch → concat."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernels: tuple[int, ...] = (9, 19, 39),
        bottleneck: int = 32,
    ) -> None:
        super().__init__()
        bn_ch = bottleneck if in_ch > 1 else in_ch
        self.bn = nn.Conv1d(in_ch, bn_ch, 1, bias=False) if in_ch > 1 else nn.Identity()
        self.convs = nn.ModuleList([
            nn.Conv1d(bn_ch, out_ch, k, padding=k // 2, bias=False)
            for k in kernels
        ])
        self.pool_conv = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.mp        = nn.MaxPool1d(3, stride=1, padding=1)
        self.norm      = nn.BatchNorm1d(out_ch * (len(kernels) + 1))
        self.act       = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z      = self.bn(x)
        outs   = [c(z) for c in self.convs] + [self.pool_conv(self.mp(x))]
        return self.act(self.norm(torch.cat(outs, dim=1)))


class _InceptionBlock(nn.Module):
    """3 inception modülü + residual bağlantısı."""

    def __init__(self, in_ch: int, out_ch: int, n_modules: int = 3) -> None:
        super().__init__()
        chs = [in_ch] + [out_ch * 4] * (n_modules - 1)
        self.mods = nn.ModuleList([
            _InceptionModule(chs[i], out_ch) for i in range(n_modules)
        ])
        out_total = out_ch * 4
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_ch, out_total, 1, bias=False),
                nn.BatchNorm1d(out_total),
            )
            if in_ch != out_total
            else nn.Identity()
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for m in self.mods:
            out = m(out)
        return self.act(out + self.skip(x))


class InceptionTime1D(nn.Module):
    """
    Parameters
    ----------
    in_ch    : 12
    num_classes : sınıf sayısı
    n_blocks : InceptionBlock sayısı
    out_ch   : her modülün per-branch çıkış kanalı
               → toplam kanal = out_ch * 4 per block
    """

    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        n_blocks: int = 3,
        out_ch: int = 32,
    ) -> None:
        super().__init__()
        blocks, ch = [], in_ch
        for _ in range(n_blocks):
            blocks.append(_InceptionBlock(ch, out_ch))
            ch = out_ch * 4
        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return self.fc(self.pool(x).squeeze(-1))
