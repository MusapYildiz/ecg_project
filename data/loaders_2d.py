"""
data/loaders_2d.py
==================
CWT görüntüleri için fold bazlı DataLoader üretici.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.loaders    import load_fold_arrays
from data.dataset_2d import CWTNpyDataset


def make_cwt_loaders(
    cv_dir     : Path | str,
    cwt_dir    : Path | str,
    fold       : int,
    batch_size : int = 32,
    num_workers: int = 4,
    normalize  : bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    CWT görüntüleri için train / val / test DataLoader'ları döndürür.

    Returns
    -------
    train_loader, val_loader, test_loader, classes
    """
    idx_train, idx_val, idx_test, ecg_ids, labels, classes = load_fold_arrays(
        cv_dir, fold
    )

    def _ds(idx: np.ndarray) -> CWTNpyDataset:
        return CWTNpyDataset(
            cwt_dir   = cwt_dir,
            ecg_ids   = ecg_ids,
            labels    = labels,
            indices   = idx,
            normalize = normalize,
        )

    kw = dict(
        batch_size         = batch_size,
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
    )

    train_loader = DataLoader(_ds(idx_train), shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(_ds(idx_val),   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(_ds(idx_test),  shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, test_loader, classes
