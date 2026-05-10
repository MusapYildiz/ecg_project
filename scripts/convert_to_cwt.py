"""
scripts/convert_to_cwt.py
=========================
PTB-XL .npy sinyallerini CWT görüntülerine dönüştürür.

Her kayıt: (12, T) sinyal → (12, 128, 128) CWT görüntüsü

Colab kullanımı — 100 Hz:
    !python /content/ecg_project/scripts/convert_to_cwt.py \
        --npy_dir  /content/PTBXL_records100_restore/npy_100 \
        --out_dir  /content/PTBXL_cwt_100hz \
        --sr       100

Colab kullanımı — 500 Hz:
    !python /content/ecg_project/scripts/convert_to_cwt.py \
        --npy_dir  /content/PTBXL_records500_restore/npy_500 \
        --out_dir  /content/PTBXL_cwt_500hz \
        --sr       500

Çıktı dosya adları: {ecg_id:05d}_cwt.npy  shape: (12, 128, 128)
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pywt
from pathlib import Path
from scipy.ndimage import zoom
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# CWT dönüşümü
# ─────────────────────────────────────────────────────────────────────────────

def signal_to_cwt(
    signal: np.ndarray,
    sr: int = 100,
    n_scales: int = 128,
    output_size: int = 128,
    wavelet: str = "morl",
) -> np.ndarray:
    """
    Tek bir lead sinyalini CWT görüntüsüne dönüştürür.

    Parameters
    ----------
    signal      : (T,) float32 — tek lead
    sr          : örnekleme hızı
    n_scales    : frekans bandı sayısı
    output_size : çıktı görüntü boyutu (kare)
    wavelet     : kullanılacak wavelet (default: Morlet)

    Returns
    -------
    (output_size, output_size) float32
    """
    # Frekans aralığı: 0.5 Hz - sr/2 Hz (Nyquist)
    freqs  = np.linspace(0.5, sr / 2, n_scales)
    scales = pywt.frequency2scale(wavelet, freqs / sr)

    # CWT uygula
    coeffs, _ = pywt.cwt(signal, scales, wavelet)  # (n_scales, T)
    power      = np.abs(coeffs).astype(np.float32)  # güç spektrumu

    # Log ölçeği — dinamik aralığı daralt
    power = np.log1p(power)

    # (n_scales, T) → (output_size, output_size) resize
    zoom_r = (output_size / power.shape[0], output_size / power.shape[1])
    power  = zoom(power, zoom_r, order=1).astype(np.float32)

    # Min-max normalizasyon [0, 1]
    mn, mx = power.min(), power.max()
    if mx > mn:
        power = (power - mn) / (mx - mn)

    return power


def record_to_cwt(
    x: np.ndarray,
    sr: int = 100,
    n_scales: int = 128,
    output_size: int = 128,
) -> np.ndarray:
    """
    12-lead sinyali CWT görüntüsüne dönüştürür.

    Parameters
    ----------
    x : (12, T) float32

    Returns
    -------
    (12, output_size, output_size) float32
    """
    cwt_img = np.zeros((12, output_size, output_size), dtype=np.float32)
    for i in range(12):
        cwt_img[i] = signal_to_cwt(x[i], sr, n_scales, output_size)
    return cwt_img


# ─────────────────────────────────────────────────────────────────────────────
# Ana dönüşüm
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    npy_dir   : str | Path,
    out_dir   : str | Path,
    sr        : int = 100,
    n_scales  : int = 128,
    output_size: int = 128,
    overwrite : bool = False,
) -> None:
    npy_dir = Path(npy_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tüm .npy dosyalarını bul
    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"npy_dir altında .npy bulunamadı: {npy_dir}")

    print(f"Toplam kayıt : {len(files)}")
    print(f"sr={sr} Hz  |  scales={n_scales}  |  output={output_size}×{output_size}")

    converted, skipped, errors = 0, 0, []

    for fpath in tqdm(files, desc="cwt"):
        # Çıktı dosya adı: 00001_cwt.npy
        stem     = fpath.stem.split("_")[0]   # "00001_lr" → "00001"
        out_path = out_dir / f"{stem}_cwt.npy"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            x = np.load(fpath).astype(np.float32)  # (12, T)

            if x.ndim != 2 or x.shape[0] != 12:
                raise ValueError(f"Beklenmeyen şekil: {x.shape}")

            cwt_img = record_to_cwt(x, sr, n_scales, output_size)
            np.save(out_path, cwt_img)
            converted += 1

        except Exception as e:
            errors.append((fpath.name, str(e)))

    print(f"\n✅ Tamamlandı:")
    print(f"   Dönüştürülen : {converted}")
    print(f"   Atlanan      : {skipped}")
    print(f"   Hata         : {len(errors)}")

    if errors:
        print("\n⚠️  İlk 5 hata:")
        for name, msg in errors[:5]:
            print(f"   {name}: {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EKG sinyal → CWT görüntüsü dönüştürücü")
    parser.add_argument("--npy_dir",    required=True, help=".npy sinyal dosyaları klasörü")
    parser.add_argument("--out_dir",    required=True, help="CWT çıktı klasörü")
    parser.add_argument("--sr",         type=int, default=100, choices=[100, 500])
    parser.add_argument("--n_scales",   type=int, default=128, help="Frekans bandı sayısı")
    parser.add_argument("--output_size",type=int, default=128, help="Çıktı görüntü boyutu")
    parser.add_argument("--overwrite",  action="store_true")
    args = parser.parse_args()

    convert(
        npy_dir    = args.npy_dir,
        out_dir    = args.out_dir,
        sr         = args.sr,
        n_scales   = args.n_scales,
        output_size= args.output_size,
        overwrite  = args.overwrite,
    )


if __name__ == "__main__":
    main()
