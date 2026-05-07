"""
data/loaders.py
===============
Fold bazlı DataLoader üretici.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import PTBXLNpyDataset


def load_fold_arrays(
    cv_dir: Path | str,
    fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Bir fold için disk'ten numpy array'leri yükler.

    Returns
    -------
    idx_train, idx_val, idx_test : kayıt indexleri
    ecg_ids                      : shape (N,)
    labels                       : shape (N, C) float32
    classes                      : sınıf isimleri listesi
    """
    d = Path(cv_dir) / f"fold_{fold}"
    return (
        np.load(d / "idx_train.npy"),
        np.load(d / "idx_val.npy"),
        np.load(d / "idx_test.npy"),
        np.load(d / "ecg_id.npy").astype(int),
        np.load(d / "Y.npy").astype(np.float32),
        np.load(d / "classes.npy", allow_pickle=True).tolist(),
    )


def make_loaders(
    cv_dir: Path | str,
    npy_dir: Path | str,
    fold: int,
    batch_size: int = 64,
    num_workers: int = 4,
    sr: int = 100,
    normalize: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Bir fold için train / val / test DataLoader'larını ve sınıf listesini döndürür.
    """
    idx_train, idx_val, idx_test, ecg_ids, labels, classes = load_fold_arrays(
        cv_dir, fold
    )

    def _ds(idx: np.ndarray) -> PTBXLNpyDataset:
        return PTBXLNpyDataset(
            npy_dir=npy_dir,
            ecg_ids=ecg_ids,
            labels=labels,
            indices=idx,
            sr=sr,
            normalize=normalize,
        )

    kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(_ds(idx_train), shuffle=True,  drop_last=True,  **kw)
    val_loader   = DataLoader(_ds(idx_val),   shuffle=False, drop_last=False, **kw)
    test_loader  = DataLoader(_ds(idx_test),  shuffle=False, drop_last=False, **kw)

    return train_loader, val_loader, test_loader, classes
