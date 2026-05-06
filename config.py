"""
config.py
=========
Tüm path, sabit ve hyperparameter tanımları tek bir yerde.
Colab'da çalıştırırken sadece bu dosyayı düzenlemeniz yeterli.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List


# ── Colab path'leri ──────────────────────────────────────────────────────────
COLAB_ROOT        = Path("/content")
NPY_100HZ_DIR     = COLAB_ROOT / "PTBXL_records100_restore/npy_100"
NPY_500HZ_DIR     = COLAB_ROOT / "PTBXL_records500_restore/npy_500"   # 500 Hz eklenince
LABELS_DIR        = COLAB_ROOT / "labels_superclass"
CV_DIR_100HZ      = COLAB_ROOT / "cv_npy100_patientwise_superclass_k5"
CV_DIR_500HZ      = COLAB_ROOT / "cv_npy500_patientwise_superclass_k5"
CKPT_DIR          = COLAB_ROOT / "checkpoints"
REPORT_DIR        = COLAB_ROOT / "reports"

# İkinci veri seti (ECG-Arrhythmia) — ileride doldurulacak
ARRHYTHMIA_NPY_DIR = COLAB_ROOT / "arrhythmia/npy"
ARRHYTHMIA_CV_DIR  = COLAB_ROOT / "arrhythmia/cv_k5"


# ── Veri ────────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    sampling_rate: int = 100          # 100 veya 500
    n_leads: int = 12
    n_folds: int = 5
    batch_size: int = 64
    num_workers: int = 4
    normalize: bool = True            # per-lead z-score
    drop_last_train: bool = True


# ── Model ────────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    name: str = "resnet1d"            # resnet1d | seresnet1d | inceptiontime
    num_classes: int = 5
    in_channels: int = 12
    # ResNet / SEResNet
    layers: Tuple[int, ...] = (3, 4, 6, 3)
    base_channels: int = 64
    se_reduction: int = 16
    # InceptionTime
    n_inception_blocks: int = 3
    inception_out_ch: int = 32
    # Head (frozen backbone senaryosu için)
    head_type: str = "none"           # none | linear | mlp | kan
    emb_dim: int = 256
    mlp_hidden: int = 256
    mlp_dropout: float = 0.1
    kan_grid_size: int = 16
    kan_scale: float = 2.0
    # Fine-tune
    freeze_backbone: bool = False
    unfreeze_last_n_blocks: int = 1   # kısmi fine-tune için


# ── Eğitim ───────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    lr_backbone: float = 1e-4        # kısmi fine-tune backbone LR
    weight_decay: float = 1e-4
    use_amp: bool = True              # Automatic Mixed Precision
    seed: int = 42
    early_stopping_patience: int = 0  # 0 = kapalı


# ── Değerlendirme ─────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    fixed_threshold: float = 0.5
    tune_thresholds_on_val: bool = True
    thr_grid_steps: int = 91          # linspace(0.05, 0.95, steps)


# ── Üst düzey deney config'i ─────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name: str = "exp_resnet1d_100hz"
    dataset: str = "ptbxl_100hz"      # ptbxl_100hz | ptbxl_500hz | arrhythmia
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def npy_dir(self) -> Path:
        mapping = {
            "ptbxl_100hz":  NPY_100HZ_DIR,
            "ptbxl_500hz":  NPY_500HZ_DIR,
            "arrhythmia":   ARRHYTHMIA_NPY_DIR,
        }
        return mapping[self.dataset]

    def cv_dir(self) -> Path:
        mapping = {
            "ptbxl_100hz":  CV_DIR_100HZ,
            "ptbxl_500hz":  CV_DIR_500HZ,
            "arrhythmia":   ARRHYTHMIA_CV_DIR,
        }
        return mapping[self.dataset]

    def ckpt_dir(self) -> Path:
        d = CKPT_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def report_dir(self) -> Path:
        d = REPORT_DIR / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d


# ── Hazır deney presetleri ───────────────────────────────────────────────────
def get_experiment(preset: str) -> ExperimentConfig:
    """
    Kullanım:
        cfg = get_experiment("resnet1d_100hz")
        cfg = get_experiment("inception_frozen_mlp_100hz")
        cfg = get_experiment("inception_partial_ft_kan_100hz")
    """
    presets = {
        # ── End-to-end backbone'lar ──────────────────────────────────────────
        "resnet1d_100hz": ExperimentConfig(
            name="resnet1d_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(name="resnet1d"),
        ),
        "seresnet1d_100hz": ExperimentConfig(
            name="seresnet1d_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(name="seresnet1d"),
        ),
        "inceptiontime_100hz": ExperimentConfig(
            name="inceptiontime_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(name="inceptiontime"),
        ),

        # ── Frozen backbone + head ───────────────────────────────────────────
        "inception_frozen_linear_100hz": ExperimentConfig(
            name="inception_frozen_linear_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime",
                head_type="linear",
                freeze_backbone=True,
                unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_mlp_100hz": ExperimentConfig(
            name="inception_frozen_mlp_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime",
                head_type="mlp",
                freeze_backbone=True,
                unfreeze_last_n_blocks=0,
            ),
        ),
        "inception_frozen_kan_100hz": ExperimentConfig(
            name="inception_frozen_kan_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime",
                head_type="kan",
                freeze_backbone=True,
                unfreeze_last_n_blocks=0,
            ),
        ),

        # ── Kısmi fine-tune ──────────────────────────────────────────────────
        "inception_partial_ft_mlp_100hz": ExperimentConfig(
            name="inception_partial_ft_mlp_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime",
                head_type="mlp",
                freeze_backbone=True,
                unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr_backbone=1e-4, lr=1e-3),
        ),
        "inception_partial_ft_kan_100hz": ExperimentConfig(
            name="inception_partial_ft_kan_100hz",
            dataset="ptbxl_100hz",
            model=ModelConfig(
                name="inceptiontime",
                head_type="kan",
                freeze_backbone=True,
                unfreeze_last_n_blocks=1,
            ),
            train=TrainConfig(lr_backbone=1e-4, lr=5e-4),
        ),

        # ── 500 Hz (ileride eklenecek) ───────────────────────────────────────
        "resnet1d_500hz": ExperimentConfig(
            name="resnet1d_500hz",
            dataset="ptbxl_500hz",
            data=DataConfig(sampling_rate=500),
            model=ModelConfig(name="resnet1d"),
        ),
    }

    if preset not in presets:
        raise ValueError(
            f"Bilinmeyen preset: '{preset}'\n"
            f"Mevcut presetler: {list(presets.keys())}"
        )
    return presets[preset]
