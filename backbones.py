"""
models/backbones.py
===================
1D EKG backbone modelleri.
  - ResNet1D      : standart residual network
  - SEResNet1D    : squeeze-and-excitation eklenmiş versiyon
  - InceptionTime1D : çok ölçekli inception blokları
"""

import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# ResNet1D
# ══════════════════════════════════════════════════════════════════════════════

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(nn.Module):
    """
    12-lead EKG için 1D ResNet.

    Args:
        in_ch       : giriş kanal sayısı (12)
        num_classes : çıkış boyutu (örn. 5)
        layers      : her stage'deki blok sayıları (ResNet-34 için (3,4,6,3))
        base_ch     : ilk stage kanal sayısı
    """

    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        layers: tuple = (3, 4, 6, 3),
        base_ch: int = 64,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self._inplanes = base_ch
        self.layer1 = self._make_layer(base_ch,     layers[0], stride=1)
        self.layer2 = self._make_layer(base_ch * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_ch * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_ch * 8, layers[3], stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(base_ch * 8, num_classes)

    def _make_layer(self, out_ch, n_blocks, stride):
        blocks = [BasicBlock1D(self._inplanes, out_ch, stride=stride)]
        self._inplanes = out_ch
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
# SEResNet1D
# ══════════════════════════════════════════════════════════════════════════════

class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.pool(x).squeeze(-1)
        return x * self.fc(s).unsqueeze(-1)


class SEBasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, reduction: int = 16):
        super().__init__()
        self.conv1    = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU(inplace=True)
        self.conv2    = nn.Conv1d(out_ch, out_ch, 7, padding=3, bias=False)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.se       = SEBlock1D(out_ch, reduction)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return self.relu(out + self.shortcut(x))


class SEResNet1D(nn.Module):
    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        layers: tuple = (3, 4, 6, 3),
        base_ch: int = 64,
        se_reduction: int = 16,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self._inplanes = base_ch
        self.layer1 = self._make_layer(base_ch,     layers[0], 1, se_reduction)
        self.layer2 = self._make_layer(base_ch * 2, layers[1], 2, se_reduction)
        self.layer3 = self._make_layer(base_ch * 4, layers[2], 2, se_reduction)
        self.layer4 = self._make_layer(base_ch * 8, layers[3], 2, se_reduction)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(base_ch * 8, num_classes)

    def _make_layer(self, out_ch, n_blocks, stride, reduction):
        blocks = [SEBasicBlock1D(self._inplanes, out_ch, stride, reduction)]
        self._inplanes = out_ch
        for _ in range(1, n_blocks):
            blocks.append(SEBasicBlock1D(out_ch, out_ch, reduction=reduction))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(self.pool(x).squeeze(-1))


# ══════════════════════════════════════════════════════════════════════════════
# InceptionTime1D
# ══════════════════════════════════════════════════════════════════════════════

class InceptionModule1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_sizes=(9, 19, 39), bottleneck=32):
        super().__init__()
        bn_ch = bottleneck if in_ch > 1 else in_ch
        self.bottleneck = nn.Conv1d(in_ch, bn_ch, 1, bias=False) if in_ch > 1 else nn.Identity()
        self.conv_k = nn.ModuleList([
            nn.Conv1d(bn_ch, out_ch, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.pool     = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_p   = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn       = nn.BatchNorm1d(out_ch * (len(kernel_sizes) + 1))
        self.relu     = nn.ReLU(inplace=True)

    def forward(self, x):
        z  = self.bottleneck(x)
        bs = [c(z) for c in self.conv_k] + [self.conv_p(self.pool(x))]
        return self.relu(self.bn(torch.cat(bs, dim=1)))


class InceptionBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_modules: int = 3):
        super().__init__()
        channels = [in_ch] + [out_ch * 4] * (n_modules - 1)
        self.modules_list = nn.ModuleList([
            InceptionModule1D(channels[i], out_ch)
            for i in range(n_modules)
        ])
        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch * 4, 1, bias=False),
            nn.BatchNorm1d(out_ch * 4),
        ) if in_ch != out_ch * 4 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = x
        for m in self.modules_list:
            out = m(out)
        return self.relu(out + self.residual(x))


class InceptionTime1D(nn.Module):
    """
    Args:
        in_ch        : 12 (EKG lead sayısı)
        num_classes  : çıkış boyutu
        n_blocks     : InceptionBlock sayısı
        out_ch       : her inception module'ün tek branch çıkış kanalı
                       (toplam kanal = out_ch * 4 per block)
    """

    def __init__(
        self,
        in_ch: int = 12,
        num_classes: int = 5,
        n_blocks: int = 3,
        out_ch: int = 32,
    ):
        super().__init__()
        blocks, ch = [], in_ch
        for _ in range(n_blocks):
            blocks.append(InceptionBlock1D(ch, out_ch))
            ch = out_ch * 4
        self.blocks = nn.Sequential(*blocks)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.blocks(x)
        return self.fc(self.pool(x).squeeze(-1))
