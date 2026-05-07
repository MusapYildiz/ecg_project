"""
data/dataset.py
===============
PTB-XL .npy dosyalarını okuyan Dataset sınıfı.

Desteklenen dosya adı formatları (otomatik tespit edilir):
  100 Hz → {ecg_id:05d}_lr.npy   örn: 00001_lr.npy
  500 Hz → {ecg_id:05d}_hr.npy   örn: 00001_hr.npy
  500 Hz → {ecg_id:05d}.npy      örn: 00001.npy      (eski format)
"""

from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PTBXLNpyDataset(Dataset):
    """
    Parameters
    ----------
    npy_dir   : .npy dosyalarının klasörü
    ecg_ids   : shape (N,) — kayıt ID'leri
    labels    : shape (N, C) float32 — multi-hot etiket matrisi
    indices   : bu split'e ait satır indexleri
    sr        : örnekleme hızı (100 ya da 500)
    normalize : True → per-lead z-score
    """

    _DEFAULT_SUFFIX: dict[int, str] = {100: "_lr.npy", 500: "_hr.npy"}

    def __init__(
        self,
        npy_dir: str | os.PathLike,
        ecg_ids: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        sr: int = 100,
        normalize: bool = True,
    ) -> None:
        self.npy_dir   = str(npy_dir)
        self.ecg_ids   = ecg_ids
        self.labels    = labels.astype(np.float32)
        self.indices   = indices
        self.normalize = normalize
        self.suffix    = self._detect_suffix(
            self._DEFAULT_SUFFIX.get(sr, "_lr.npy")
        )

    def _detect_suffix(self, expected: str) -> str:
        """
        İlk birkaç dosyaya bakarak gerçek suffix'i tespit eder.
        Önce expected'ı dener, bulamazsa '.npy' (bare) dener.
        """
        for idx in self.indices[:10]:
            eid = int(self.ecg_ids[idx])
            # Önce beklenen suffix'i dene
            if os.path.exists(os.path.join(self.npy_dir, f"{eid:05d}{expected}")):
                return expected
            # Bare suffix dene: sadece {id}.npy
            if os.path.exists(os.path.join(self.npy_dir, f"{eid:05d}.npy")):
                return ".npy"
        # Bulamazsa default'u döndür (yükleme sırasında hata verir)
        return expected

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = self.indices[i]
        eid = int(self.ecg_ids[idx])
        x   = self._load(eid)
        y   = torch.from_numpy(self.labels[idx])
        return x, y, eid

    def _load(self, eid: int) -> torch.Tensor:
        path = os.path.join(self.npy_dir, f"{eid:05d}{self.suffix}")
        x = np.load(path).astype(np.float32)

        if x.ndim != 2 or x.shape[0] != 12:
            raise ValueError(
                f"Beklenmeyen sinyal şekli {x.shape} (12, T) bekleniyor. "
                f"Dosya: {path}"
            )

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std  = x.std(axis=1,  keepdims=True) + 1e-6
            x    = (x - mean) / std

        return torch.from_numpy(x)
