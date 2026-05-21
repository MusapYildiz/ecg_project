"""
data/dataset_2d.py
==================
CWT görüntülerini okuyan Dataset sınıfı.

Dosya adı formatı: {ecg_id:05d}_cwt.npy   shape: (12, 128, 128)
"""

from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CWTNpyDataset(Dataset):
    """
    Parameters
    ----------
    cwt_dir   : _cwt.npy dosyalarının klasörü
    ecg_ids   : shape (N,)
    labels    : shape (N, C) float32
    indices   : bu split'e ait satır indexleri
    normalize : True → per-channel z-score
    """

    def __init__(
        self,
        cwt_dir  : str | os.PathLike,
        ecg_ids  : np.ndarray,
        labels   : np.ndarray,
        indices  : np.ndarray,
        normalize: bool = True,
    ) -> None:
        self.cwt_dir   = str(cwt_dir)
        self.ecg_ids   = ecg_ids
        self.labels    = labels.astype(np.float32)
        self.indices   = indices
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = self.indices[i]
        eid = int(self.ecg_ids[idx])
        x   = self._load(eid)
        y   = torch.from_numpy(self.labels[idx])
        return x, y, eid

    def _load(self, eid: int) -> torch.Tensor:
        path = os.path.join(self.cwt_dir, f"{eid:05d}_cwt.npy")
        x = np.load(path).astype(np.float32)  # (12, 128, 128)

        if x.ndim != 3 or x.shape[0] != 12:
            raise ValueError(
                f"Beklenmeyen CWT şekli {x.shape}, (12, H, W) bekleniyor. "
                f"Dosya: {path}"
            )

        if self.normalize:
            mean = x.mean(axis=(1, 2), keepdims=True)
            std  = x.std(axis=(1, 2),  keepdims=True) + 1e-6
            x    = (x - mean) / std

        return torch.from_numpy(x)
