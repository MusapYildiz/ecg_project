"""
evaluation/reporter.py
======================
Her fold için detaylı metrik, raw data ve confusion matrix kaydeder.

Kaydedilen dosyalar (fold bazlı):
    fold_N/
    ├── y_true.npy               — test seti gerçek etiketler  (N, C) int32
    ├── y_prob.npy               — test seti tahmin olasılıkları (N, C) float32
    ├── thresholds.npy           — per-class optimal eşikler (C,) float32
    ├── per_class_metrics.csv    — sınıf bazlı TP/FP/FN/TN/precision/recall/f1/threshold
    ├── summary_metrics.csv      — fold özet metrikleri (macro + diğer)
    └── confusion_matrix_{cls}.png — her sınıf için 2x2 confusion matrix

Proje bazlı:
    fold_results.csv             — 5 fold özet tablosu
    summary.json                 — mean ± std
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # GUI gerektirmez (Colab uyumlu)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from evaluation.metrics import (
    binary_counts,
    safe_prf,
    macro_auroc,
    macro_f1,
    subset_accuracy,
    hamming_accuracy,
    apply_threshold,
)
from evaluation.threshold import tune_thresholds


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix çizici
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    class_name: str,
    fold: int,
    save_path: Path,
) -> None:
    mat = np.array([[tn, fp],
                    [fn, tp]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(mat, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: 0", "Pred: 1"], fontsize=11)
    ax.set_yticklabels(["True: 0", "True: 1"], fontsize=11)
    ax.set_title(f"{class_name}  |  Fold {fold}", fontsize=12, fontweight="bold")

    # Hücre değerleri
    thresh = mat.max() / 2.0
    for (i, j), v in np.ndenumerate(mat):
        color = "white" if v > thresh else "black"
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                fontsize=13, fontweight="bold", color=color)

    # Etiketler
    ax.set_xlabel("Tahmin", fontsize=11)
    ax.set_ylabel("Gerçek",  fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Tek fold raporu
# ─────────────────────────────────────────────────────────────────────────────

def save_fold_report(
    fold       : int,
    y_true     : np.ndarray,          # (N, C) int32
    y_prob     : np.ndarray,          # (N, C) float32
    thresholds : np.ndarray,          # (C,)   float32
    class_names: list[str],
    report_dir : Path,
) -> dict:
    """
    Bir fold için tüm çıktıları kaydeder.

    Returns
    -------
    dict — fold özet metrikleri (fold_results.csv'ye yazılır)
    """
    fold_dir = Path(report_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Raw data kaydet ────────────────────────────────────────────────────
    np.save(fold_dir / "y_true.npy",      y_true.astype(np.int32))
    np.save(fold_dir / "y_prob.npy",      y_prob.astype(np.float32))
    np.save(fold_dir / "thresholds.npy",  thresholds.astype(np.float32))

    # ── 2. Per-class metrikler ────────────────────────────────────────────────
    y_pred = apply_threshold(y_prob, thresholds)
    C      = len(class_names)

    per_class_rows = []
    for c, cname in enumerate(class_names):
        tp, fp, fn, tn = binary_counts(y_true[:, c], y_pred[:, c])
        prec, rec, f1  = safe_prf(tp, fp, fn)

        per_class_rows.append({
            "class"    : cname,
            "threshold": float(thresholds[c]),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": round(prec, 6),
            "recall"   : round(rec,  6),
            "f1"       : round(f1,   6),
            "support"  : int(y_true[:, c].sum()),   # pozitif örnek sayısı
        })

        # Confusion matrix görseli
        _plot_confusion_matrix(
            tp, fp, fn, tn,
            class_name = cname,
            fold       = fold,
            save_path  = fold_dir / f"confusion_matrix_{cname}.png",
        )

    df_pc = pd.DataFrame(per_class_rows)
    df_pc.to_csv(fold_dir / "per_class_metrics.csv", index=False)

    # ── 3. Özet metrikler ─────────────────────────────────────────────────────
    macro_prec = float(df_pc["precision"].mean())
    macro_rec  = float(df_pc["recall"].mean())
    macro_f1_  = float(df_pc["f1"].mean())
    macro_auc  = macro_auroc(y_true, y_prob)
    sub_acc    = subset_accuracy(y_true, y_pred)
    ham_acc    = hamming_accuracy(y_true, y_pred)

    # Per-class AUC
    from sklearn.metrics import roc_auc_score
    per_class_auc = {}
    for c, cname in enumerate(class_names):
        if len(np.unique(y_true[:, c])) < 2:
            per_class_auc[f"auc_{cname}"] = float("nan")
        else:
            per_class_auc[f"auc_{cname}"] = round(
                float(roc_auc_score(y_true[:, c], y_prob[:, c])), 6
            )

    summary_row = {
        "fold"           : fold,
        "macro_precision": round(macro_prec, 6),
        "macro_recall"   : round(macro_rec,  6),
        "macro_f1"       : round(macro_f1_,  6),
        "mean_f1"        : round(macro_f1_,  6),   # alias (macro_f1 ile aynı)
        "macro_auc"      : round(macro_auc,  6),
        "subset_acc"     : round(sub_acc,    6),
        "hamming_acc"    : round(ham_acc,    6),
        **per_class_auc,
        **{f"f1_{r['class']}":        round(r["f1"],        6) for r in per_class_rows},
        **{f"precision_{r['class']}": round(r["precision"], 6) for r in per_class_rows},
        **{f"recall_{r['class']}":    round(r["recall"],    6) for r in per_class_rows},
        **{f"threshold_{r['class']}": round(r["threshold"], 6) for r in per_class_rows},
    }

    pd.DataFrame([summary_row]).to_csv(
        fold_dir / "summary_metrics.csv", index=False
    )

    print(
        f"  [Fold {fold} raporu kaydedildi → {fold_dir.name}]  "
        f"AUC={macro_auc:.4f}  F1={macro_f1_:.4f}  "
        f"subsetAcc={sub_acc:.4f}  hammAcc={ham_acc:.4f}"
    )

    return summary_row


# ─────────────────────────────────────────────────────────────────────────────
# 5-fold özet
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment_summary(
    fold_summaries: list[dict],
    class_names   : list[str],
    report_dir    : Path,
    experiment    : str,
    epochs        : int,
) -> None:
    """
    5 fold sonuçlarını toplar, mean ± std hesaplar, Drive'a kaydeder.
    """
    report_dir = Path(report_dir)
    df = pd.DataFrame(fold_summaries)
    df.to_csv(report_dir / "fold_results.csv", index=False)

    # mean ± std
    metric_cols = [
        "macro_precision", "macro_recall", "macro_f1",
        "macro_auc", "subset_acc", "hamming_acc",
    ] + [f"auc_{c}" for c in class_names] \
      + [f"f1_{c}"  for c in class_names]

    summary: dict = {"experiment": experiment, "epochs": epochs}
    for col in metric_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna().values.astype(np.float64)
        summary[f"{col}_mean"] = round(float(vals.mean()), 6)
        summary[f"{col}_std"]  = round(float(vals.std(ddof=1)), 6)

    with open(report_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Okunabilir özet tablosu
    rows = []
    for col in ["macro_auc", "macro_f1", "macro_precision", "macro_recall",
                "subset_acc", "hamming_acc"]:
        if f"{col}_mean" in summary:
            rows.append({
                "metric": col,
                "mean"  : summary[f"{col}_mean"],
                "std"   : summary[f"{col}_std"],
            })
    pd.DataFrame(rows).to_csv(report_dir / "summary_table.csv", index=False)

    print(f"\n  Tüm raporlar kaydedildi: {report_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Sonradan yeniden analiz — model gerekmez
# ─────────────────────────────────────────────────────────────────────────────

def reload_fold_results(
    report_dir : str | Path,
    fold       : int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Kaydedilmiş y_true, y_prob, thresholds array'lerini yükler.
    Model yüklemeye gerek yoktur.

    Returns
    -------
    y_true, y_prob, thresholds
    """
    d = Path(report_dir) / f"fold_{fold}"
    y_true     = np.load(d / "y_true.npy")
    y_prob     = np.load(d / "y_prob.npy")
    thresholds = np.load(d / "thresholds.npy")
    return y_true, y_prob, thresholds


def reanalyze(
    report_dir : str | Path,
    class_names: list[str],
    new_threshold: Optional[float | np.ndarray] = None,
    folds      : tuple = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    """
    Kaydedilmiş raw data'dan metrikleri yeniden hesaplar.
    İsteğe bağlı olarak farklı bir threshold uygular.

    Kullanım
    --------
    # Orijinal threshold ile yeniden hesapla:
    df = reanalyze("/content/drive/.../reports/resnet1d_100hz", classes)

    # Sabit 0.4 threshold ile:
    df = reanalyze(..., new_threshold=0.4)

    # Per-class yeni threshold ile:
    df = reanalyze(..., new_threshold=np.array([0.3, 0.5, 0.4, 0.6, 0.45]))
    """
    rows = []
    for fold in folds:
        y_true, y_prob, saved_thr = reload_fold_results(report_dir, fold)

        thr = saved_thr if new_threshold is None else np.full(
            len(class_names),
            new_threshold if np.isscalar(new_threshold) else new_threshold,
            dtype=np.float32,
        )

        y_pred = apply_threshold(y_prob, thr)

        row = {"fold": fold}
        for c, cname in enumerate(class_names):
            tp, fp, fn, tn = binary_counts(y_true[:, c], y_pred[:, c])
            prec, rec, f1  = safe_prf(tp, fp, fn)
            row[f"f1_{cname}"]        = round(f1,   4)
            row[f"precision_{cname}"] = round(prec, 4)
            row[f"recall_{cname}"]    = round(rec,  4)

        row["macro_f1"]   = round(macro_f1(y_true, y_pred),        4)
        row["macro_auc"]  = round(macro_auroc(y_true, y_prob),      4)
        row["subset_acc"] = round(subset_accuracy(y_true, y_pred),  4)
        row["hamming_acc"]= round(hamming_accuracy(y_true, y_pred), 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df
