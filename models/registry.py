"""
models/registry.py
==================
Model fabrikası — config'e göre doğru modeli kurar, checkpoint'i yükler,
backbone'u dondurur / kısmen çözer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from config import ModelConfig
from models.backbones import ResNet1D, SEResNet1D, InceptionTime1D
from models.heads import LinearHead, MLPHead, KANHead


# ── Feature extractor wrapper ─────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Herhangi bir backbone'dan embedding çıkaran wrapper.
    backbone.fc katmanını nn.Identity() ile değiştirir,
    ardından isteğe bağlı bir projeksiyon uygular.
    """

    def __init__(self, backbone: nn.Module, emb_dim: int = 256):
        super().__init__()
        in_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj     = nn.Linear(in_dim, emb_dim)

    def forward(self, x: torch.Tensor):
        raw = self.backbone(x)   # (B, in_dim)
        emb = self.proj(raw)     # (B, emb_dim)
        return raw, emb


class EmbeddingClassifier(nn.Module):
    """FeatureExtractor + herhangi bir head."""

    def __init__(self, feat: FeatureExtractor, head: nn.Module):
        super().__init__()
        self.feat = feat
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, emb = self.feat(x)
        return self.head(emb)


# ── Backbone fabrikası ────────────────────────────────────────────────────────

def _build_backbone(cfg: ModelConfig) -> nn.Module:
    name = cfg.name.lower()
    common = dict(in_ch=cfg.in_channels, num_classes=cfg.num_classes)

    if name == "resnet1d":
        return ResNet1D(**common, layers=cfg.layers, base_ch=cfg.base_channels)
    if name == "seresnet1d":
        return SEResNet1D(
            **common,
            layers=cfg.layers,
            base_ch=cfg.base_channels,
            se_reduction=cfg.se_reduction,
        )
    if name == "inceptiontime":
        return InceptionTime1D(
            **common,
            n_blocks=cfg.n_inception_blocks,
            out_ch=cfg.inception_out_ch,
        )
    raise ValueError(f"Bilinmeyen backbone: '{name}'")


def _build_head(cfg: ModelConfig) -> nn.Module:
    ht = cfg.head_type.lower()
    if ht == "linear":
        return LinearHead(cfg.emb_dim, cfg.num_classes)
    if ht == "mlp":
        return MLPHead(cfg.emb_dim, cfg.mlp_hidden, cfg.num_classes, cfg.mlp_dropout)
    if ht == "kan":
        return KANHead(cfg.emb_dim, cfg.num_classes, cfg.kan_grid_size, cfg.kan_scale)
    raise ValueError(f"Bilinmeyen head_type: '{ht}'")


# ── Freeze / unfreeze yardımcıları ───────────────────────────────────────────

def _freeze_backbone(feat: FeatureExtractor) -> None:
    for p in feat.backbone.parameters():
        p.requires_grad = False


def _unfreeze_last_n_blocks(feat: FeatureExtractor, n: int) -> None:
    """InceptionTime için son n bloğu çözer. Diğer backbone'lar için layer4 vb."""
    backbone = feat.backbone

    if hasattr(backbone, "blocks"):           # InceptionTime
        blocks = list(backbone.blocks.children())
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    else:                                     # ResNet / SEResNet
        all_layers = [backbone.layer1, backbone.layer2,
                      backbone.layer3, backbone.layer4]
        for layer in all_layers[-n:]:
            for p in layer.parameters():
                p.requires_grad = True


# ── Ana fabrika fonksiyonu ────────────────────────────────────────────────────

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    config.ModelConfig'e göre model oluşturur.

    - head_type == "none"  → saf backbone (end-to-end eğitim)
    - head_type != "none"  → FeatureExtractor + Head
                             freeze_backbone=True ise backbone dondurulur
                             unfreeze_last_n_blocks > 0 ise kısmi açılır
    """
    backbone = _build_backbone(cfg)

    if cfg.head_type == "none":
        return backbone

    feat = FeatureExtractor(backbone, cfg.emb_dim)
    head = _build_head(cfg)

    if cfg.freeze_backbone:
        _freeze_backbone(feat)
        if cfg.unfreeze_last_n_blocks > 0:
            _unfreeze_last_n_blocks(feat, cfg.unfreeze_last_n_blocks)

    return EmbeddingClassifier(feat, head)


# ── Checkpoint yardımcıları ───────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {path}")
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return model


def get_param_groups(model: nn.Module, lr: float, lr_backbone: float):
    """
    Kısmi fine-tune için iki öğrenme hızlı param grupları.
    Backbone parametrelerine lr_backbone, diğerlerine lr uygulanır.
    """
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "feat.backbone" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return groups
