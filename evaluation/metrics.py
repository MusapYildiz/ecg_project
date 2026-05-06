"""
evaluation/metrics.py
=====================
Tüm metrik fonksiyonları — tek tanım, hiç duplicate yok.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


# ── Temel yardımcılar ─────────────────────────────────────────────────────────

def _safe_prf(tp: int, fp: int, fn: int, eps: float = 1e-12):
    """(precision, recall, f1) — sıfır bölme güvenli."""
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


def _binary_counts(y_true_c: np.ndarray, y_pred_c: np.ndarray):
    """(TP, FP, FN, TN) — tek sınıf için."""
    t, p = y_true_c.astype(np.int32), y_pred_c.astype(np.int32)
    tp = int(((t == 1) & (p == 1)).sum())
    fp = int(((t == 0) & (p == 1)).sum())
    fn = int(((t == 1) & (p == 0)).sum())
    tn = int(((t == 0) & (p == 0)).sum())
    return tp, fp, fn, tn


# ── Eşik gerektirmeyen metrikler ─────────────────────────────────────────────

def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Multi-label macro AUROC.
    Sadece 0 ve 1 içeren sınıflar (tanımsız AUROC) atlanır.
    """
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


# ── Eşik gerektiren metrikler ─────────────────────────────────────────────────

def apply_threshold(y_prob: np.ndarray, threshold) -> np.ndarray:
    """
    threshold: float (scalar) ya da np.ndarray (C,) — per-class.
    """
    thr = np.asarray(threshold, dtype=np.float32).reshape(1, -1) \
          if not np.isscalar(threshold) else threshold
    return (y_prob >= thr).astype(np.int32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    C = y_true.shape[1]
    f1s = []
    for c in range(C):
        tp, fp, fn, _ = _binary_counts(y_true[:, c], y_pred[:, c])
        _, _, f1 = _safe_prf(tp, fp, fn)
        f1s.append(f1)
    return float(np.mean(f1s))


def subset_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Exact match oranı (tüm etiketlerin doğru olduğu örnekler)."""
    return float((y_true.astype(np.int32) == y_pred).all(axis=1).mean())


def hamming_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Label-wise doğruluk oranı."""
    return float((y_true.astype(np.int32) == y_pred).mean())


# ── Per-class detaylı rapor ───────────────────────────────────────────────────

def per_class_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    thresholds: np.ndarray,
) -> list[dict]:
    """
    Her sınıf için {class, TP, FP, FN, TN, precision, recall, f1, threshold} döndürür.
    """
    y_pred = apply_threshold(y_prob, thresholds)
    rows = []
    for c, name in enumerate(class_names):
        tp, fp, fn, tn = _binary_counts(y_true[:, c], y_pred[:, c])
        prec, rec, f1  = _safe_prf(tp, fp, fn)
        rows.append({
            "class": name, "threshold": float(thresholds[c]),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": prec, "recall": rec, "f1": f1,
        })
    return rows


# ── Özet ─────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds,
) -> dict:
    """
    Tek çağrıda tüm metrikleri döndürür.

    Returns
    -------
    dict ile anahtarlar:
        macro_auc, macro_f1, subset_acc, hamming_acc
    """
    y_pred = apply_threshold(y_prob, thresholds)
    return {
        "macro_auc":    macro_auroc(y_true, y_prob),
        "macro_f1":     macro_f1(y_true, y_pred),
        "subset_acc":   subset_accuracy(y_true, y_pred),
        "hamming_acc":  hamming_accuracy(y_true, y_pred),
    }
