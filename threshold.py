"""
evaluation/threshold.py
=======================
Validation seti üzerinde per-class optimal eşik araması.
"""

import numpy as np
from evaluation.metrics import _binary_counts, _safe_prf


def tune_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    n_steps: int = 91,
) -> np.ndarray:
    """
    Her sınıf için F1'i maksimize eden eşiği validation seti üzerinde bulur.

    Parameters
    ----------
    y_val   : (N, C) gerçek etiketler, {0,1}
    p_val   : (N, C) tahmin olasılıkları [0,1]
    n_steps : grid'in kaç noktadan oluşacağı (default: 0.05..0.95 arası 91 nokta)

    Returns
    -------
    thresholds : (C,) float32 array
    """
    grid = np.linspace(0.05, 0.95, n_steps, dtype=np.float32)
    C    = y_val.shape[1]
    best = np.full(C, 0.5, dtype=np.float32)

    for c in range(C):
        yt = y_val[:, c].astype(np.int32)
        pv = p_val[:, c]
        best_f1 = -1.0

        for t in grid:
            yp = (pv >= t).astype(np.int32)
            tp, fp, fn, _ = _binary_counts(yt, yp)
            _, _, f1 = _safe_prf(tp, fp, fn)
            if f1 > best_f1:
                best_f1, best[c] = f1, float(t)

    return best
