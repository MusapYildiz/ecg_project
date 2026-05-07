"""
models/registry.py
==================
Model fabrikası, checkpoint yönetimi, freeze/unfreeze yardımcıları.

Dışarıya açılan ana fonksiyon:  build_model(cfg: ModelConfig) -> nn.Module
"""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

from config import ModelConfig
from models.backbones import ResNet1D, SEResNet1D, InceptionTime1D
from models.heads import LinearHead, MLPHead, KANHead


# ─────────────────────────────────────────────────────────────────────────────
# FeatureExtractor  —  backbone'un fc'sini kaldırıp projeksiyon ekler
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    Backbone'u feature extractor'a dönüştürür:
      1. backbone.fc → nn.Identity()   (fc'yi devre dışı bırak)
      2. proj : Linear(in_dim, emb_dim) ekle

    forward: (B, 12, T) → emb (B, emb_dim)
    """

    def __init__(self, backbone: nn.Module, emb_dim: int = 256) -> None:
        super().__init__()
        in_dim       = backbone.fc.in_features
        backbone.fc  = nn.Identity()
        self.backbone = backbone
        self.proj     = nn.Linear(in_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# EmbeddingClassifier  —  FeatureExtractor + herhangi bir head
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingClassifier(nn.Module):
    """
    forward: (B, 12, T) → logits (B, num_classes)
    """

    def __init__(self, feat: FeatureExtractor, head: nn.Module) -> None:
        super().__init__()
        self.feat = feat
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.feat(x))


# ─────────────────────────────────────────────────────────────────────────────
# İç yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _build_backbone(cfg: ModelConfig) -> nn.Module:
    common = dict(in_ch=cfg.in_channels, num_classes=cfg.num_classes)
    name   = cfg.name.lower()

    if name == "resnet1d":
        return ResNet1D(**common, layers=cfg.layers, base_ch=cfg.base_channels)
    if name == "seresnet1d":
        return SEResNet1D(**common, layers=cfg.layers,
                          base_ch=cfg.base_channels, se_reduction=cfg.se_reduction)
    if name == "inceptiontime":
        return InceptionTime1D(**common,
                               n_blocks=cfg.n_inception_blocks,
                               out_ch=cfg.inception_out_ch)
    raise ValueError(f"Bilinmeyen backbone: '{cfg.name}'")


def _build_head(cfg: ModelConfig) -> nn.Module:
    ht = cfg.head_type.lower()
    if ht == "linear":
        return LinearHead(cfg.emb_dim, cfg.num_classes)
    if ht == "mlp":
        return MLPHead(cfg.emb_dim, cfg.mlp_hidden, cfg.num_classes, cfg.mlp_dropout)
    if ht == "kan":
        return KANHead(cfg.emb_dim, cfg.num_classes, cfg.kan_grid_size, cfg.kan_scale)
    raise ValueError(f"Bilinmeyen head_type: '{cfg.head_type}'")


def _freeze(feat: FeatureExtractor) -> None:
    for p in feat.backbone.parameters():
        p.requires_grad = False


def _unfreeze_last_n(feat: FeatureExtractor, n: int) -> None:
    """Son n bloğu / layer'ı eğitilebilir yapar."""
    if n <= 0:
        return
    bb = feat.backbone
    if hasattr(bb, "blocks"):                          # InceptionTime
        targets = list(bb.blocks.children())[-n:]
    else:                                              # ResNet / SEResNet
        targets = [bb.layer1, bb.layer2, bb.layer3, bb.layer4][-n:]
    for t in targets:
        for p in t.parameters():
            p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Ana fabrika
# ─────────────────────────────────────────────────────────────────────────────

def build_model(cfg: ModelConfig) -> nn.Module:
    """
    ModelConfig'e göre model oluşturur:

    head_type == "none"
        → saf backbone, end-to-end eğitim
    head_type != "none"
        → FeatureExtractor + Head
        freeze_backbone=True  → backbone tamamen dondurulur
        unfreeze_last_n_blocks > 0 → son N blok yeniden açılır (partial fine-tune)
    """
    backbone = _build_backbone(cfg)

    if cfg.head_type == "none":
        return backbone

    feat = FeatureExtractor(backbone, cfg.emb_dim)
    head = _build_head(cfg)

    if cfg.freeze_backbone:
        _freeze(feat)
        _unfreeze_last_n(feat, cfg.unfreeze_last_n_blocks)

    return EmbeddingClassifier(feat, head)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(model: nn.Module, path: Path, **meta) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **meta}, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    device: Optional[torch.device] = None,
    strict: bool = True,
) -> nn.Module:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint bulunamadı: {path}")
    ckpt = torch.load(path, map_location=device or "cpu")
    model.load_state_dict(ckpt["model_state"], strict=strict)
    return model


def load_backbone_into_embedding_classifier(
    model: EmbeddingClassifier,
    ckpt_path: Path,
    device: Optional[torch.device] = None,
) -> None:
    """
    End-to-end eğitilmiş bir backbone checkpointini
    EmbeddingClassifier.feat.backbone'a yükler.
    fc boyut uyuşmazlığını görmezden gelir (strict=False).
    """
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Backbone checkpoint bulunamadı: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device or "cpu")
    model.feat.backbone.load_state_dict(ckpt["model_state"], strict=False)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizasyon yardımcıları
# ─────────────────────────────────────────────────────────────────────────────

def get_param_groups(
    model: nn.Module,
    lr: float,
    lr_backbone: float,
) -> list[dict]:
    """
    Partial fine-tune için iki farklı lr'li param grupları.
    feat.backbone parametrelerine lr_backbone, kalanına lr uygulanır.
    """
    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if "feat.backbone" in name else head_params).append(p)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        groups.append({"params": head_params, "lr": lr})
    return groups
