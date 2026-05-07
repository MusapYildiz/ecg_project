"""
evaluation/threshold.py
=======================
Validation seti üzerinde per-class optimal eşik araması.
"""

from __future__ import annotations
import numpy as np
from evaluation.metrics import binary_counts, safe_prf


def tune_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    n_steps: int = 91,
) -> np.ndarray:
    """
    Her sınıf için validation F1'ini maksimize eden eşiği bulur.

    Parameters
    ----------
    y_val   : (N, C) int — gerçek etiketler
    p_val   : (N, C) float — tahmin olasılıkları
    n_steps : grid yoğunluğu (default: 0.05..0.95 arası 91 nokta)

    Returns
    -------
    thresholds : (C,) float32
    """
    grid = np.linspace(0.05, 0.95, n_steps, dtype=np.float32)
    C    = y_val.shape[1]
    best = np.full(C, 0.5, dtype=np.float32)

    for c in range(C):
        yt       = y_val[:, c].astype(np.int32)
        pv       = p_val[:, c]
        best_f1  = -1.0
        for t in grid:
            yp = (pv >= t).astype(np.int32)
            tp, fp, fn, _ = binary_counts(yt, yp)
            _, _, f1 = safe_prf(tp, fp, fn)
            if f1 > best_f1:
                best_f1, best[c] = f1, float(t)

    return best
