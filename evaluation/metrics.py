"""
evaluation/metrics.py
=====================
Tüm metrik fonksiyonları — tek tanım, sıfır duplicate.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# Düşük seviye sayaçlar
# ─────────────────────────────────────────────────────────────────────────────

def binary_counts(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
) -> tuple[int, int, int, int]:
    """Tek sınıf için (TP, FP, FN, TN)."""
    t = y_true_c.astype(np.int32)
    p = y_pred_c.astype(np.int32)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


def safe_prf(tp: int, fp: int, fn: int, eps: float = 1e-12) -> tuple[float, float, float]:
    """(precision, recall, f1) — sıfır bölmeye karşı güvenli."""
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


# ─────────────────────────────────────────────────────────────────────────────
# Eşik gerektirmeyen metrikler
# ─────────────────────────────────────────────────────────────────────────────

def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multi-label macro AUROC.
    Yalnız pozitif veya yalnız negatif olan sınıflar atlanır.
    """
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
    return float(np.mean(aucs)) if aucs else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Eşik gerektiren metrikler
# ─────────────────────────────────────────────────────────────────────────────

def apply_threshold(
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
) -> np.ndarray:
    """
    threshold: tek float (tüm sınıflara) ya da (C,) array (per-class).
    """
    thr = np.asarray(threshold, dtype=np.float32)
    if thr.ndim == 0:
        return (y_prob >= float(thr)).astype(np.int32)
    return (y_prob >= thr.reshape(1, -1)).astype(np.int32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1s = []
    for c in range(y_true.shape[1]):
        tp, fp, fn, _ = binary_counts(y_true[:, c], y_pred[:, c])
        _, _, f1 = safe_prf(tp, fp, fn)
        f1s.append(f1)
    return float(np.mean(f1s))


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tüm etiketlerin doğru olduğu örneklerin oranı (exact match)."""
    return float((y_true.astype(np.int32) == y_pred).all(axis=1).mean())


def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Label-wise doğruluk oranı."""
    return float((y_true.astype(np.int32) == y_pred).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Toplu hesaplama
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray = 0.5,
) -> dict[str, float]:
    """
    Tek çağrıda tüm metrikleri hesaplar.

    Returns
    -------
    dict anahtarları: macro_auc, macro_f1, subset_acc, hamming_acc
    """
    y_pred = apply_threshold(y_prob, threshold)
    return {
        "macro_auc"  : macro_auroc(y_true, y_prob),
        "macro_f1"   : macro_f1(y_true, y_pred),
        "subset_acc" : subset_accuracy(y_true, y_pred),
        "hamming_acc": hamming_accuracy(y_true, y_pred),
    }


def per_class_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray,
) -> list[dict]:
    """
    Her sınıf için ayrıntılı rapor döndürür.

    Returns
    -------
    list of dicts: class, TP, FP, FN, TN, precision, recall, f1, threshold
    """
    y_pred = apply_threshold(y_prob, thresholds)
    rows = []
    for c, name in enumerate(class_names):
        tp, fp, fn, tn = binary_counts(y_true[:, c], y_pred[:, c])
        prec, rec, f1  = safe_prf(tp, fp, fn)
        rows.append({
            "class": name, "threshold": float(thresholds[c]),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": f1,
        })
    return rows
