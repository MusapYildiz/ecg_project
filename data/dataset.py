"""
data/dataset.py
===============
PTB-XL .npy dosyalarını okuyan Dataset sınıfı.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PTBXLNpyDataset(Dataset):
    """
    .npy formatında kaydedilmiş 12-lead EKG sinyallerini yükler.

    Beklenen dosya adı formatı: {ecg_id:05d}_lr.npy  (100 Hz)
                                 {ecg_id:05d}_hr.npy  (500 Hz)

    Args:
        npy_dir   : .npy dosyalarının bulunduğu klasör
        ecg_ids   : kayıt ID'leri, shape (N,)
        labels    : multi-hot etiket matrisi, shape (N, C), float32
        indices   : bu split'te kullanılacak satır indexleri
        sr        : örnekleme hızı (100 ya da 500); dosya suffix'ini belirler
        normalize : True ise per-lead z-score uygulanır
    """

    _SUFFIX = {100: "_lr.npy", 500: "_hr.npy"}

    def __init__(
        self,
        npy_dir: str,
        ecg_ids: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        sr: int = 100,
        normalize: bool = True,
    ):
        self.npy_dir   = str(npy_dir)
        self.ecg_ids   = ecg_ids
        self.labels    = labels.astype(np.float32)
        self.indices   = indices
        self.normalize = normalize
        self.suffix    = self._SUFFIX.get(sr, "_lr.npy")

    # ── temel metodlar ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        eid = int(self.ecg_ids[idx])

        x = self._load_signal(eid)          # (12, T)
        y = torch.from_numpy(self.labels[idx])  # (C,)
        return x, y, eid

    # ── yardımcı ─────────────────────────────────────────────────────────────
    def _load_signal(self, eid: int) -> torch.Tensor:
        path = os.path.join(self.npy_dir, f"{eid:05d}{self.suffix}")
        x = np.load(path).astype(np.float32)  # (12, T)

        if x.ndim != 2 or x.shape[0] != 12:
            raise ValueError(
                f"Beklenmeyen sinyal şekli {x.shape} — "
                f"(12, T) bekleniyor. Dosya: {path}"
            )

        if self.normalize:
            mean = x.mean(axis=1, keepdims=True)
            std  = x.std(axis=1,  keepdims=True) + 1e-6
            x    = (x - mean) / std

        return torch.from_numpy(x)
