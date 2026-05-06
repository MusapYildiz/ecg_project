"""
data/loaders.py
===============
Fold bazlı DataLoader üretici — tek, net bir imza.
"""

import os
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from data.dataset import PTBXLNpyDataset


def load_fold_arrays(cv_dir: Path, fold: int):
    """
    Bir fold için gerekli numpy array'lerini diskten yükler.

    Returns
    -------
    idx_train, idx_val, idx_test : np.ndarray  — kayıt indexleri
    ecg_ids                       : np.ndarray  — shape (N,)
    labels                        : np.ndarray  — shape (N, C) float32
    classes                       : list[str]   — sınıf isimleri
    """
    fold_dir = Path(cv_dir) / f"fold_{fold}"

    idx_train = np.load(fold_dir / "idx_train.npy")
    idx_val   = np.load(fold_dir / "idx_val.npy")
    idx_test  = np.load(fold_dir / "idx_test.npy")
    ecg_ids   = np.load(fold_dir / "ecg_id.npy").astype(int)
    labels    = np.load(fold_dir / "Y.npy").astype(np.float32)
    classes   = np.load(fold_dir / "classes.npy", allow_pickle=True).tolist()

    return idx_train, idx_val, idx_test, ecg_ids, labels, classes


def make_loaders(
    cv_dir: Path,
    npy_dir: Path,
    fold: int,
    batch_size: int = 64,
    num_workers: int = 4,
    sr: int = 100,
    normalize: bool = True,
):
    """
    Bir fold için train / val / test DataLoader'larını döndürür.

    Returns
    -------
    train_loader, val_loader, test_loader, classes
    """
    idx_train, idx_val, idx_test, ecg_ids, labels, classes = load_fold_arrays(
        cv_dir, fold
    )

    def _make_ds(idx, shuffle):
        return PTBXLNpyDataset(
            npy_dir=npy_dir,
            ecg_ids=ecg_ids,
            labels=labels,
            indices=idx,
            sr=sr,
            normalize=normalize,
        )

    common_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(
        _make_ds(idx_train, True),
        shuffle=True,
        drop_last=True,
        **common_kw,
    )
    val_loader = DataLoader(
        _make_ds(idx_val, False),
        shuffle=False,
        **common_kw,
    )
    test_loader = DataLoader(
        _make_ds(idx_test, False),
        shuffle=False,
        **common_kw,
    )

    return train_loader, val_loader, test_loader, classes
