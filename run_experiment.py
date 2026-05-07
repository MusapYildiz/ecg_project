"""
run_experiment.py
=================
Herhangi bir deney presetini 5-fold cross-validation ile çalıştırır.

Colab kullanımı
---------------
from run_experiment import run

# End-to-end backbone:
run("resnet1d_100hz", epochs=20)

# Frozen head (backbone önceden eğitilmiş olmalı):
run("inception_frozen_kan_100hz",
    backbone_ckpt_dir="/content/drive/MyDrive/ecg_outputs/checkpoints/inceptiontime_100hz")

# Partial fine-tune:
run("inception_partial_ft_mlp_100hz",
    backbone_ckpt_dir="/content/drive/MyDrive/ecg_outputs/checkpoints/inceptiontime_100hz")
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

from config          import get_experiment, ExperimentConfig
from data.loaders    import make_loaders
from models.registry import (
    build_model,
    load_backbone_into_embedding_classifier,
    get_param_groups,
    EmbeddingClassifier,
)
from training.trainer        import train_fold
from evaluation.reporter     import save_experiment_summary


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1))


def _print_summary(summary: dict) -> None:
    w = 60
    print("\n" + "═" * w)
    print(f"  {summary['experiment']}   ({summary['epochs']} epoch · 5 fold)")
    print("═" * w)
    for key in ["test_macro_auc", "test_macro_f1", "test_subset_acc", "test_hamming_acc"]:
        label = key.replace("test_", "").replace("_", " ").upper()
        m = summary.get(f"{key}_mean", float("nan"))
        s = summary.get(f"{key}_std",  float("nan"))
        print(f"  {label:<22} {m:.4f} ± {s:.4f}")
    print("═" * w + "\n")


def _is_partial_ft(cfg: ExperimentConfig) -> bool:
    m = cfg.model
    return (
        m.head_type != "none"
        and m.freeze_backbone
        and m.unfreeze_last_n_blocks > 0
    )


def _is_frozen_head(cfg: ExperimentConfig) -> bool:
    m = cfg.model
    return (
        m.head_type != "none"
        and m.freeze_backbone
        and m.unfreeze_last_n_blocks == 0
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def run(
    preset           : str,
    epochs           : Optional[int] = None,
    folds            : tuple[int, ...] = (1, 2, 3, 4, 5),
    backbone_ckpt_dir: Optional[str] = None,
    device           : Optional[torch.device] = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    preset            : config.py'deki preset adı
    epochs            : None → config'deki değer
    folds             : hangi fold'lar çalıştırılsın
    backbone_ckpt_dir : frozen/partial-ft için backbone checkpoint klasörü
                        (içinde fold_1_best.pt, fold_2_best.pt ... olmalı)
    device            : None → otomatik (cuda > cpu)
    """
    cfg    = get_experiment(preset)
    epochs = epochs or cfg.train.epochs
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    needs_backbone = _is_frozen_head(cfg) or _is_partial_ft(cfg)
    if needs_backbone and backbone_ckpt_dir is None:
        raise ValueError(
            f"'{preset}' için backbone_ckpt_dir gereklidir.\n"
            "Örn: run(..., backbone_ckpt_dir='/content/drive/MyDrive/ecg_outputs/checkpoints/inceptiontime_100hz')"
        )

    print(f"\n{'━'*60}")
    print(f"  Deney : {cfg.name}")
    print(f"  Device: {device}   Epochs: {epochs}   Folds: {list(folds)}")
    print(f"{'━'*60}")

    fold_results: list[dict] = []

    for fold in folds:
        print(f"\n┌─ Fold {fold} / {len(folds)} {'─'*40}")

        # ── Loader'lar ────────────────────────────────────────────────────────
        train_loader, val_loader, test_loader, classes = make_loaders(
            cv_dir     = cfg.cv_dir(),
            npy_dir    = cfg.npy_dir(),
            fold       = fold,
            batch_size = cfg.data.batch_size,
            num_workers= cfg.data.num_workers,
            sr         = cfg.data.sampling_rate,
            normalize  = cfg.data.normalize,
        )
        cfg.model.num_classes = len(classes)

        # ── Model ─────────────────────────────────────────────────────────────
        model = build_model(cfg.model).to(device)

        # Backbone checkpoint yükle (frozen/partial-ft)
        if needs_backbone:
            ckpt = Path(backbone_ckpt_dir) / f"fold_{fold}_best.pt"
            load_backbone_into_embedding_classifier(model, ckpt, device=device)
            print(f"│  backbone yüklendi: {ckpt.name}")

        tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in model.parameters())
        print(f"│  params: {tr/1e6:.2f}M trainable / {tot/1e6:.2f}M total")

        # ── Optimizer param grupları ──────────────────────────────────────────
        pg = (
            get_param_groups(model, cfg.train.lr, cfg.train.lr_backbone)
            if _is_partial_ft(cfg) else None
        )

        # ── Eğitim ───────────────────────────────────────────────────────────
        result = train_fold(
            model        = model,
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            ckpt_path    = cfg.fold_ckpt(fold),
            report_dir   = cfg.report_dir(),
            fold         = fold,
            class_names  = classes,
            epochs       = epochs,
            lr           = cfg.train.lr,
            weight_decay = cfg.train.weight_decay,
            use_amp      = cfg.train.use_amp,
            tune_thr     = cfg.eval.tune_thresholds_on_val,
            thr_steps    = cfg.eval.thr_grid_steps,
            param_groups = pg,
        )
        result["fold"] = fold
        fold_results.append(result)

        print(f"└─ Fold {fold} tamamlandı.")
        del model
        torch.cuda.empty_cache()

    # ── 5-fold özet ───────────────────────────────────────────────────────────
    df = pd.DataFrame(fold_results)

    save_experiment_summary(
        fold_summaries = fold_results,
        class_names    = classes,
        report_dir     = cfg.report_dir(),
        experiment     = cfg.name,
        epochs         = epochs,
    )

    _print_summary({"experiment": cfg.name, "epochs": epochs, **{
        f"{k}_mean": round(float(np.mean([r[k] for r in fold_results if k in r])), 4)
        for k in ["macro_auc", "macro_f1", "subset_acc", "hamming_acc"]
    }, **{
        f"{k}_std": round(float(np.std([r[k] for r in fold_results if k in r], ddof=1)), 4)
        for k in ["macro_auc", "macro_f1", "subset_acc", "hamming_acc"]
    }})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Toplu çalıştırıcılar
# ─────────────────────────────────────────────────────────────────────────────

def run_all_end2end(
    hz: int = 100,
    epochs: int = 20,
    **kw,
) -> pd.DataFrame:
    """3 backbone × 1 dataset."""
    tag     = f"_{hz}hz"
    presets = [f"resnet1d{tag}", f"seresnet1d{tag}", f"inceptiontime{tag}"]
    return pd.concat([run(p, epochs=epochs, **kw) for p in presets], ignore_index=True)


def run_all_frozen_heads(
    backbone_ckpt_dir: str,
    hz: int = 100,
    epochs: int = 10,
    **kw,
) -> pd.DataFrame:
    """Frozen backbone + 3 head."""
    tag     = f"_frozen_{{head}}_{hz}hz"
    presets = [f"inception_frozen_linear_{hz}hz",
               f"inception_frozen_mlp_{hz}hz",
               f"inception_frozen_kan_{hz}hz"]
    return pd.concat(
        [run(p, epochs=epochs, backbone_ckpt_dir=backbone_ckpt_dir, **kw) for p in presets],
        ignore_index=True,
    )


def run_all_partial_ft(
    backbone_ckpt_dir: str,
    hz: int = 100,
    epochs: int = 20,
    **kw,
) -> pd.DataFrame:
    """Partial fine-tune × 2 head."""
    presets = [f"inception_partial_ft_mlp_{hz}hz",
               f"inception_partial_ft_kan_{hz}hz"]
    return pd.concat(
        [run(p, epochs=epochs, backbone_ckpt_dir=backbone_ckpt_dir, **kw) for p in presets],
        ignore_index=True,
    )
