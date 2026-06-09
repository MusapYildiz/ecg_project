"""
models/backbones_2d.py
======================
2D backbone'lar — CWT görüntüleri için.

ResNet2D  : ImageNet pretrained ResNet-50, 12 kanallı girişe uyarlandı
ViTBase2D : ImageNet pretrained ViT-Base/16, 12 kanallı girişe uyarlandı

İlk conv katmanı 3 → 12 kanala genişletilir:
    w_new = w_old.repeat(1, 4, 1, 1)[:, :12, :, :] * (3/12)
Bu yöntem medikal görüntülemede yaygın kullanılan pretrained ağırlık
uyarlama tekniğidir.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı
# ─────────────────────────────────────────────────────────────────────────────

def _expand_first_conv(conv: nn.Conv2d, in_ch: int) -> nn.Conv2d:
    """
    Pretrained 3-kanallı Conv2d'yi in_ch kanallı hale getirir.
    Yeni ağırlık = orijinal ağırlığın kanallar boyunca tekrarı,
    ölçeklenmiş (3 / in_ch) ile.
    """
    w_old  = conv.weight.data                              # (out, 3, H, W)
    repeat = in_ch // 3 + (1 if in_ch % 3 != 0 else 0)
    w_new  = w_old.repeat(1, repeat, 1, 1)[:, :in_ch, :, :]
    w_new  = w_new * (3.0 / in_ch)

    new_conv = nn.Conv2d(
        in_ch, conv.out_channels,
        kernel_size = conv.kernel_size,
        stride      = conv.stride,
        padding     = conv.padding,
        bias        = (conv.bias is not None),
    )
    new_conv.weight = nn.Parameter(w_new)
    if conv.bias is not None:
        new_conv.bias = nn.Parameter(conv.bias.data.clone())
    return new_conv


# ─────────────────────────────────────────────────────────────────────────────
# ResNet2D
# ─────────────────────────────────────────────────────────────────────────────

class ResNet2D(nn.Module):
    """
    ImageNet pretrained ResNet-50, 12 kanallı CWT girişi için uyarlandı.

    Parameters
    ----------
    in_ch       : 12 (CWT kanalları = EKG lead sayısı)
    num_classes : sınıf sayısı
    pretrained  : True → ImageNet ağırlıkları
    """

    def __init__(
        self,
        in_ch      : int  = 12,
        num_classes: int  = 5,
        pretrained : bool = True,
    ) -> None:
        super().__init__()
        weights    = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        base       = tv_models.resnet50(weights=weights)
        base.conv1 = _expand_first_conv(base.conv1, in_ch)
        base.fc    = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        

class ScratchResNet18_2D(nn.Module):
    """
    ImageNet pretraining olmadan sıfırdan eğitilen ResNet-18.
    12 kanallı CWT girişi için uyarlandı.

    Parameters
    ----------
    in_ch       : 12 (CWT kanalları = EKG lead sayısı)
    num_classes : sınıf sayısı
    """

    def __init__(
        self,
        in_ch      : int = 12,
        num_classes: int = 5,
    ) -> None:
        super().__init__()
        # Pretrained=False — sıfırdan başla
        base       = tv_models.resnet18(weights=None)
        # İlk conv: 3 → 12 kanal (sıfırdan init, ImageNet ağırlığı yok)
        base.conv1 = nn.Conv2d(
            in_ch, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─────────────────────────────────────────────────────────────────────────────
# ViTBase2D
# ─────────────────────────────────────────────────────────────────────────────

class ViTBase2D(nn.Module):
    """
    ImageNet pretrained ViT-Base/16, 12 kanallı CWT girişi için uyarlandı.
    timm kütüphanesi gerektirir: pip install timm

    Parameters
    ----------
    in_ch       : 12
    num_classes : sınıf sayısı
    pretrained  : True → ImageNet ağırlıkları
    img_size    : CWT görüntü boyutu (default: 128)
    """

    def __init__(
        self,
        in_ch      : int  = 12,
        num_classes: int  = 5,
        pretrained : bool = True,
        img_size   : int  = 128,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError(
                "timm kütüphanesi gerekli: pip install timm"
            )
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained  = pretrained,
            num_classes = num_classes,
            in_chans    = in_ch,
            img_size    = img_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        
        

def get_resnet2d_param_groups(
    model: ResNet2D,
    lr_head: float = 1e-3,
    lr_layer4: float = 1e-4,
    lr_layer3: float = 1e-5,
) -> list[dict]:
    """
    ResNet2D için katman bazlı param grupları.
    stem, layer1, layer2 frozen kalır.
    """
    # Önce tümünü dondur
    for p in model.parameters():
        p.requires_grad = False

    # layer3 aç
    for p in model.model.layer3.parameters():
        p.requires_grad = True

    # layer4 aç
    for p in model.model.layer4.parameters():
        p.requires_grad = True

    # FC head aç
    for p in model.model.fc.parameters():
        p.requires_grad = True

    return [
        {"params": model.model.layer3.parameters(), "lr": lr_layer3},
        {"params": model.model.layer4.parameters(), "lr": lr_layer4},
        {"params": model.model.fc.parameters(),     "lr": lr_head},
    ]


def get_resnet2d_frozen_param_groups(
    model: ResNet2D,
    lr_head: float = 1e-3,
) -> list[dict]:
    """
    ResNet2D için sadece head eğitimi.
    Tüm backbone frozen.
    """
    for p in model.parameters():
        p.requires_grad = False

    for p in model.model.fc.parameters():
        p.requires_grad = True

    return [
        {"params": model.model.fc.parameters(), "lr": lr_head},
    ]
