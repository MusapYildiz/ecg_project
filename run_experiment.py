"""
run_experiment.py
=================
Herhangi bir deney presetini 5-fold cross-validation ile çalıştırır.

Kullanım (Colab cell):
    from run_experiment import run
    run("resnet1d_100hz", epochs=20)
    run("inception_partial_ft_kan_100hz", epochs=20, backbone_ckpt_dir="/content/checkpoints/inceptiontime_100hz")
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from config import get_experiment, ExperimentConfig
from data.loaders import make_loaders
from models.registry import (
    build_model,
    load_checkpoint,
    get_param_groups,
)
from training.trainer import train_fold


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def _mean_std(values: list) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  {summary['experiment']}  — {summary['epochs']} epoch, 5 fold")
    print("=" * 60)
    for key in ["test_macro_auc", "test_macro_f1", "test_subset_acc", "test_hamming_acc"]:
        m = summary[f"{key}_mean"]
        s = summary[f"{key}_std"]
        label = key.replace("test_", "").replace("_", " ").upper()
        print(f"  {label:<20}  {m:.4f} ± {s:.4f}")
    print("=" * 60)


# ── Backbone checkpoint yükleyici (frozen/partial-ft için) ────────────────────

def _maybe_load_backbone(model, cfg: ExperimentConfig, fold: int, backbone_ckpt_dir: Optional[str]) -> None:
    """
    Eğer head_type != 'none' (yani frozen/partial-ft senaryosu) ise
    backbone_ckpt_dir'den fold'a ait checkpoint'i yükler.
    """
    if cfg.model.head_type == "none":
        return
    if backbone_ckpt_dir is None:
        raise ValueError(
            "head_type != 'none' için backbone_ckpt_dir belirtilmeli.\n"
            "Örn: run(..., backbone_ckpt_dir='/content/checkpoints/inceptiontime_100hz')"
        )
    ckpt_path = Path(backbone_ckpt_dir) / f"fold_{fold}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint bulunamadı: {ckpt_path}")

    # model.feat.backbone'u yükle (EmbeddingClassifier.feat.backbone)
    device = next(model.parameters()).device
    ckpt   = torch.load(ckpt_path, map_location=device)
    model.feat.backbone.load_state_dict(ckpt["model_state"], strict=False)
    print(f"  ✅ backbone yüklendi: {ckpt_path.name}")


# ── Ana fonksiyon ─────────────────────────────────────────────────────────────

def run(
    preset: str,
    epochs: Optional[int] = None,
    folds: tuple = (1, 2, 3, 4, 5),
    backbone_ckpt_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Parametreler
    ------------
    preset            : config.py'deki preset adı
    epochs            : None ise config'deki değer kullanılır
    folds             : çalıştırılacak fold'lar
    backbone_ckpt_dir : frozen/partial-ft için backbone ckpt klasörü
    device            : None ise otomatik (cuda > cpu)
    """
    cfg = get_experiment(preset)
    if epochs is not None:
        cfg.train.epochs = epochs

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Deney: {cfg.name}  |  device: {device}  |  epochs: {cfg.train.epochs}")

    fold_results = []

    for fold in folds:
        print(f"\n{'─'*55}\n  Fold {fold} / {len(folds)}\n{'─'*55}")

        # Loader'lar
        train_loader, val_loader, test_loader, classes = make_loaders(
            cv_dir=cfg.cv_dir(),
            npy_dir=cfg.npy_dir(),
            fold=fold,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            sr=cfg.data.sampling_rate,
            normalize=cfg.data.normalize,
        )
        cfg.model.num_classes = len(classes)

        # Model
        model = build_model(cfg.model).to(device)
        _maybe_load_backbone(model, cfg, fold, backbone_ckpt_dir)

        # Param grupları (kısmi fine-tune → iki lr)
        is_partial_ft = (
            cfg.model.head_type != "none"
            and cfg.model.freeze_backbone
            and cfg.model.unfreeze_last_n_blocks > 0
        )
        param_groups = (
            get_param_groups(model, cfg.train.lr, cfg.train.lr_backbone)
            if is_partial_ft else None
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  params: {trainable/1e6:.2f}M trainable / {total/1e6:.2f}M total")

        # Checkpoint yolu
        ckpt_path = cfg.ckpt_dir() / f"fold_{fold}_best.pt"

        # Eğitim
        result = train_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            ckpt_path=ckpt_path,
            epochs=cfg.train.epochs,
            lr=cfg.train.lr,
            lr_backbone=cfg.train.lr_backbone,
            weight_decay=cfg.train.weight_decay,
            use_amp=cfg.train.use_amp,
            tune_thr=cfg.eval.tune_thresholds_on_val,
            thr_steps=cfg.eval.thr_grid_steps,
            param_groups=param_groups,
        )
        result["fold"]  = fold
        result["model"] = cfg.model.name
        fold_results.append(result)

        # GPU belleğini temizle
        del model
        torch.cuda.empty_cache()

    # ── 5-fold özet ───────────────────────────────────────────────────────────
    df = pd.DataFrame(fold_results)

    metric_keys = ["test_macro_auc", "test_macro_f1", "test_subset_acc", "test_hamming_acc"]
    summary = {"experiment": cfg.name, "epochs": cfg.train.epochs}
    for k in metric_keys:
        if k in df.columns:
            m, s = _mean_std(df[k].tolist())
            summary[f"{k}_mean"] = m
            summary[f"{k}_std"]  = s

    _print_summary(summary)

    # Kaydet
    df.to_csv(cfg.report_dir() / "fold_results.csv", index=False)
    with open(cfg.report_dir() / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ Sonuçlar kaydedildi: {cfg.report_dir()}")
    return df


# ── Tüm presetleri sırayla çalıştır ──────────────────────────────────────────

def run_all_end2end(epochs: int = 20) -> pd.DataFrame:
    """3 backbone × 1 dataset = 3 deney."""
    presets = ["resnet1d_100hz", "seresnet1d_100hz", "inceptiontime_100hz"]
    frames  = [run(p, epochs=epochs) for p in presets]
    return pd.concat(frames, ignore_index=True)


def run_all_heads(backbone_ckpt_dir: str, epochs: int = 10) -> pd.DataFrame:
    """Frozen backbone + 3 head tipi = 3 deney."""
    presets = [
        "inception_frozen_linear_100hz",
        "inception_frozen_mlp_100hz",
        "inception_frozen_kan_100hz",
    ]
    frames = [run(p, epochs=epochs, backbone_ckpt_dir=backbone_ckpt_dir) for p in presets]
    return pd.concat(frames, ignore_index=True)


def run_all_partial_ft(backbone_ckpt_dir: str, epochs: int = 20) -> pd.DataFrame:
    """Kısmi fine-tune = 2 deney."""
    presets = [
        "inception_partial_ft_mlp_100hz",
        "inception_partial_ft_kan_100hz",
    ]
    frames = [run(p, epochs=epochs, backbone_ckpt_dir=backbone_ckpt_dir) for p in presets]
    return pd.concat(frames, ignore_index=True)
