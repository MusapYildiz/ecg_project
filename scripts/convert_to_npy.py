"""
scripts/convert_to_npy.py
=========================
PTB-XL wfdb kayıtlarını (.hea/.dat) numpy array'e dönüştürür.
100 Hz için de, 500 Hz için de kullanılabilir.

Colab kullanımı — 500 Hz:
    !python /content/ecg_project/scripts/convert_to_npy.py \\
        --records_dir /content/PTBXL_records500/records500 \\
        --out_dir     /content/PTBXL_records500/npy_500 \\
        --sr          500

Colab kullanımı — 100 Hz:
    !python /content/ecg_project/scripts/convert_to_npy.py \\
        --records_dir /content/PTBXL_records100/records100 \\
        --out_dir     /content/PTBXL_records100/npy_100 \\
        --sr          100

Çıktı dosya adları:
    100 Hz → {ecg_id:05d}_lr.npy   shape: (12, 1000)
    500 Hz → {ecg_id:05d}_hr.npy   shape: (12, 5000)
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import wfdb
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def _suffix(sr: int) -> str:
    return "_lr.npy" if sr == 100 else "_hr.npy"


def _collect_records(records_dir: Path) -> list[Path]:
    """
    records_dir altındaki tüm .hea dosyalarını bulur,
    extension'sız path listesi döndürür (wfdb.rdsamp uzantısız ister).
    """
    paths = sorted(records_dir.rglob("*.hea"))
    return [p.with_suffix("") for p in paths]


# ─────────────────────────────────────────────────────────────────────────────
# Ana dönüşüm
# ─────────────────────────────────────────────────────────────────────────────

def convert(
    records_dir: str | Path,
    out_dir: str | Path,
    sr: int = 500,
    overwrite: bool = False,
) -> None:
    """
    Parameters
    ----------
    records_dir : records500/ (veya records100/) klasörü
    out_dir     : .npy dosyalarının yazılacağı klasör
    sr          : beklenen örnekleme hızı (doğrulama için)
    overwrite   : True → var olan .npy dosyalarının üstüne yaz
    """
    records_dir = Path(records_dir)
    out_dir     = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix  = _suffix(sr)
    records = _collect_records(records_dir)

    if not records:
        raise FileNotFoundError(
            f"records_dir altında .hea dosyası bulunamadı: {records_dir}\n"
            "Klasör yolunu kontrol edin."
        )

    print(f"Toplam kayıt: {len(records)}  |  sr={sr} Hz  |  suffix='{suffix}'")

    converted, skipped, errors = 0, 0, []

    for rec_path in tqdm(records, desc="convert"):
        # Dosya adından ecg_id çıkar: "00001_hr" → 1
        stem  = rec_path.stem                  # örn. "00001_hr" veya "00001_lr"
        eid   = int(stem.split("_")[0])
        out_f = out_dir / f"{eid:05d}{suffix}"

        if out_f.exists() and not overwrite:
            skipped += 1
            continue

        try:
            sig, fields = wfdb.rdsamp(str(rec_path))   # (T, 12)

            # Örnekleme hızı kontrolü
            actual_sr = fields.get("fs", None)
            if actual_sr is not None and int(actual_sr) != sr:
                raise ValueError(
                    f"Beklenen sr={sr}, kayıtta fs={actual_sr}. "
                    "Doğru records klasörünü kullandığınızdan emin olun."
                )

            if sig.ndim != 2 or sig.shape[1] != 12:
                raise ValueError(f"Beklenmeyen sinyal şekli: {sig.shape}, (T,12) bekleniyor.")

            x = sig.T.astype(np.float32)               # (12, T)
            np.save(out_f, x)
            converted += 1

        except Exception as e:
            errors.append((str(rec_path), str(e)))

    # ── Özet ─────────────────────────────────────────────────────────────────
    print(f"\n✅ Tamamlandı:")
    print(f"   Dönüştürülen : {converted}")
    print(f"   Atlanan      : {skipped}  (zaten mevcut)")
    print(f"   Hata         : {len(errors)}")

    if errors:
        print("\n⚠️  İlk 10 hata:")
        for path, msg in errors[:10]:
            print(f"   {Path(path).name}: {msg}")

    if converted + skipped == 0:
        raise RuntimeError("Hiç dosya işlenmedi. records_dir yolunu kontrol edin.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PTB-XL wfdb → .npy dönüştürücü (100 Hz ve 500 Hz)"
    )
    parser.add_argument(
        "--records_dir", required=True,
        help="records100/ veya records500/ klasörü (alt klasörler dahil taranır)",
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Çıktı .npy dosyalarının yazılacağı klasör",
    )
    parser.add_argument(
        "--sr", type=int, default=500, choices=[100, 500],
        help="Örnekleme hızı: 100 veya 500 (varsayılan: 500)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Var olan .npy dosyalarının üstüne yaz",
    )
    args = parser.parse_args()

    convert(
        records_dir = args.records_dir,
        out_dir     = args.out_dir,
        sr          = args.sr,
        overwrite   = args.overwrite,
    )


if __name__ == "__main__":
    main()
