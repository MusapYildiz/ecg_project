"""
config.py
=========
Tüm path, sabit ve hyperparameter tanımları.

Colab'da sadece bu dosyadaki DRIVE_OUTPUT_DIR değiştirilmesi yeterli.
Preset'ler get_experiment() fonksiyonu aracılığıyla kullanılır.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Colab path'leri
# ─────────────────────────────────────────────────────────────────────────────

CONTENT             = Path("/content")

# Ham veri (session'a özel, Drive'dan extract edilir)
NPY_100HZ_DIR       = CONTENT / "PTBXL_records100_restore/npy_100"
NPY_500HZ_DIR       = CONTENT / "PTBXL_records500_restore/npy_500"
ARRHYTHMIA_NPY_DIR  = CONTENT / "arrhythmia/npy"

# CV split dosyaları (session'a özel)
CV_100HZ_DIR        = CONTENT / "cv_npy100_patientwise_superclass_k5"
CV_500HZ_DIR        = CONTENT / "cv_npy500_patientwise_superclass_k5"
ARRHYTHMIA_CV_DIR   = CONTENT / "arrhythmia/cv_k5"

# Çıktılar — Drive varsa Drive'a, yoksa /content'e yazar
def _output_root() -> Path:
    """
    Drive mount edilmişse Drive'a, edilmemişse /content'e yazar.
    Her çağrıda kontrol edilir (lazy) — mount sırasına göre doğru path döner.
    """
    drive = Path("/content/drive/MyDrive/ecg_outputs")
    local = Path("/content/ecg_outputs")
    return drive if Path("/content/drive/MyDrive").exists() else local


# Dataset → (npy_dir, cv_dir) eşlemesi
_DATASET_PATHS: dict[str, tuple[Path, Path]] = {
    "ptbxl_100hz" : (NPY_100HZ_DIR,      CV_100HZ_DIR),
    "ptbxl_500hz" : (NPY_500HZ_DIR,      CV_500HZ_DIR),
    "arrhythmia"  : (ARRHYTHMIA_NPY_DIR, ARRHYTHMIA_CV_DIR),
}


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass'ları
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DataConfig:
    sampling_rate : int  = 100    # 100 veya 500
    n_leads       : int  = 12
    n_folds       : int  = 5
    batch_size    : int  = 64
    num_workers   : int  = 4
    normalize     : bool = True   # per-lead z-score normalizasyonu


@dataclass
class ModelConfig:
    # Backbone seçimi
    name         : str   = "resnet1d"   # resnet1d | seresnet1d | inceptiontime
    num_classes  : int   = 5
    in_channels  : int   = 12

    # ResNet / SEResNet
    layers       : Tuple[int, ...] = (3, 4, 6, 3)
    base_channels: int   = 64
    se_reduction : int   = 16

    # InceptionTime
    n_inception_blocks : int = 3
    inception_out_ch   : int = 32

    # Head (frozen/partial-ft senaryoları için)
    head_type    : str   = "none"   # none | linear | mlp | kan
    emb_dim      : int   = 256
    mlp_hidden   : int   = 256
    mlp_dropout  : float = 0.1
    kan_grid_size: int   = 16
    kan_scale    : float = 2.0

    # Freeze stratejisi
    freeze_backbone        : bool = False
    unfreeze_last_n_blocks : int  = 0   # 0 = tam frozen, 1+ = kısmi fine-tune


@dataclass
class TrainConfig:
    epochs          : int   = 20
    lr              : float = 1e-3
    lr_backbone     : float = 1e-4   # kısmi fine-tune backbone LR'ı
    weight_decay    : float = 1e-4
    use_amp         : bool  = True


@dataclass
class EvalConfig:
    fixed_threshold       : float = 0.5
    tune_thresholds_on_val: bool  = True
    thr_grid_steps        : int   = 91   # linspace(0.05, 0.95, 91)


@dataclass
class ExperimentConfig:
    name   : str   = "experiment"
    dataset: str   = "ptbxl_100hz"
    data   : DataConfig  = field(default_factory=DataConfig)
    model  : ModelConfig = field(default_factory=ModelConfig)
    train  : TrainConfig = field(default_factory=TrainConfig)
    eval   : EvalConfig  = field(default_factory=EvalConfig)

    # ── path helpers ─────────────────────────────────────────────────────────
    def npy_dir(self) -> Path:
        return _DATASET_PATHS[self.dataset][0]

    def cv_dir(self) -> Path:
        return _DATASET_PATHS[self.dataset][1]

    def ckpt_dir(self) -> Path:
        d = _output_root() / "checkpoints" / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def report_dir(self) -> Path:
        d = _output_root() / "reports" / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fold_ckpt(self, fold: int) -> Path:
        return self.ckpt_dir() / f"fold_{fold}_best.pt"

    def sr(self) -> int:
        return self.data.sampling_rate


# ─────────────────────────────────────────────────────────────────────────────
# Deney presetleri
# ─────────────────────────────────────────────────────────────────────────────

def get_experiment(preset: str) -> ExperimentConfig:
    """
    Kullanım
    --------
    cfg = get_experiment("resnet1d_100hz")
    cfg = get_experiment("inception_frozen_kan_100hz")
    cfg = get_experiment("inception_partial_ft_mlp_100hz")
    """

    _P = {

        # ── PTB-XL 100 Hz — end-to-end backbone'lar ──────────────────────────
        "resnet1d_100hz": ExperimentConfig(
            name="resnet1d_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(name="resnet1d"),
        ),
        "seresnet1d_100hz": ExperimentConfig(
            name="seresnet1d_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(name="seresnet1d"),
        ),
        "inceptiontime_100hz": ExperimentConfig(
            name="inceptiontime_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(name="inceptiontime"),
        ),

        # ── PTB-XL 100 Hz — frozen backbone + head ───────────────────────────
        "inception_frozen_linear_100hz": ExperimentConfig(
            name="inception_frozen_linear_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime", head_type="linear",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_mlp_100hz": ExperimentConfig(
            name="inception_frozen_mlp_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime", head_type="mlp",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_kan_100hz": ExperimentConfig(
            name="inception_frozen_kan_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime", head_type="kan",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),

        # ── PTB-XL 100 Hz — kısmi fine-tune (son 1 blok açık) ────────────────
        "inception_partial_ft_mlp_100hz": ExperimentConfig(
            name="inception_partial_ft_mlp_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime", head_type="mlp",
                freeze_backbone=True, unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr=1e-3, lr_backbone=1e-4),
        ),
        "inception_partial_ft_kan_100hz": ExperimentConfig(
            name="inception_partial_ft_kan_100hz", dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime", head_type="kan",
                freeze_backbone=True, unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr=5e-4, lr_backbone=1e-4),
        ),

        # ── PTB-XL 500 Hz — end-to-end (ileride tamamlanacak) ────────────────
        "resnet1d_500hz": ExperimentConfig(
            name="resnet1d_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(name="resnet1d"),
        ),
        "seresnet1d_500hz": ExperimentConfig(
            name="seresnet1d_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(name="seresnet1d"),
        ),
        "inceptiontime_500hz": ExperimentConfig(
            name="inceptiontime_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(name="inceptiontime"),
        ),

        # ── PTB-XL 500 Hz — frozen backbone + head ───────────────────────────
        "inception_frozen_linear_500hz": ExperimentConfig(
            name="inception_frozen_linear_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(
                name="inceptiontime", head_type="linear",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_mlp_500hz": ExperimentConfig(
            name="inception_frozen_mlp_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(
                name="inceptiontime", head_type="mlp",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_kan_500hz": ExperimentConfig(
            name="inception_frozen_kan_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(
                name="inceptiontime", head_type="kan",
                freeze_backbone=True, unfreeze_last_n_blocks=0,
            ),
        ),

        # ── PTB-XL 500 Hz — kısmi fine-tune ──────────────────────────────────
        "inception_partial_ft_mlp_500hz": ExperimentConfig(
            name="inception_partial_ft_mlp_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(
                name="inceptiontime", head_type="mlp",
                freeze_backbone=True, unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr=1e-3, lr_backbone=1e-4),
        ),
        "inception_partial_ft_kan_500hz": ExperimentConfig(
            name="inception_partial_ft_kan_500hz", dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(
                name="inceptiontime", head_type="kan",
                freeze_backbone=True, unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr=5e-4, lr_backbone=1e-4),
        ),

        # ── ECG-Arrhythmia (ileride tamamlanacak) ────────────────────────────
        "resnet1d_arrhythmia": ExperimentConfig(
            name="resnet1d_arrhythmia", dataset="arrhythmia",
            model=ModelConfig(name="resnet1d"),
        ),
        "inceptiontime_arrhythmia": ExperimentConfig(
            name="inceptiontime_arrhythmia", dataset="arrhythmia",
            model=ModelConfig(name="inceptiontime"),
        ),
    }

    if preset not in _P:
        raise ValueError(
            f"Bilinmeyen preset: '{preset}'\n"
            f"Mevcut presetler:\n  " + "\n  ".join(sorted(_P.keys()))
        )
    return _P[preset]


def list_presets() -> list[str]:
    """Mevcut tüm preset isimlerini döndürür."""
    cfg = get_experiment  # dummy call ile dict'i yeniden oluşturmamak için
    # doğrudan liste döndür
    return [
        "resnet1d_100hz", "seresnet1d_100hz", "inceptiontime_100hz",
        "inception_frozen_linear_100hz", "inception_frozen_mlp_100hz",
        "inception_frozen_kan_100hz",
        "inception_partial_ft_mlp_100hz", "inception_partial_ft_kan_100hz",
        "resnet1d_500hz", "seresnet1d_500hz", "inceptiontime_500hz",
        "inception_frozen_linear_500hz", "inception_frozen_mlp_500hz",
        "inception_frozen_kan_500hz",
        "inception_partial_ft_mlp_500hz", "inception_partial_ft_kan_500hz",
        "resnet1d_arrhythmia", "inceptiontime_arrhythmia",
    ]
