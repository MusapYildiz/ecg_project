"""
scripts/prepare_arrhythmia.py
==============================
ECG-Arrhythmia veri setini hazırlar:
1. .hea/.mat → .npy dönüşümü
2. Etiket çıkarımı (≥200 örnek, 27 sınıf)
3. 5-fold stratified CV split

Colab kullanımı:
    !python /content/ecg_project/scripts/prepare_arrhythmia.py \
        --data_dir  "/content/drive/MyDrive/ecg-auto-dx/ecg_arrhythmia_dataset" \
        --out_npy   /content/arrhythmia_npy \
        --out_cv    /content/arrhythmia_cv_k5 \
        --min_count 200
"""

from __future__ import annotations

import argparse
import os
import re
import numpy as np
import pandas as pd
import wfdb
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcılar
# ─────────────────────────────────────────────────────────────────────────────

def collect_records(data_dir: str) -> list[str]:
    """Tüm .hea dosyalarını bulup extension'sız path listesi döndürür."""
    paths = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".hea"):
                paths.append(os.path.join(root, f[:-4]))
    return sorted(paths)


def parse_dx(hea_path: str) -> list[str]:
    """Bir .hea dosyasından Dx kodlarını çıkarır."""
    try:
        with open(hea_path + ".hea") as f:
            content = f.read()
        match = re.search(r'Dx:\s*([^\n]+)', content)
        if match:
            return [c.strip() for c in match.group(1).split(",")]
    except:
        pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Adım 1 — Etiketleri çıkar ve geçerli sınıfları belirle
# ─────────────────────────────────────────────────────────────────────────────

def extract_labels(
    records   : list[str],
    cond_csv  : str,
    min_count : int = 200,
) -> tuple[np.ndarray, list[str], list[list[str]]]:
    """
    Returns
    -------
    Y          : (N, C) int8 multi-hot
    classes    : C sınıf ismi (Acronym)
    rec_labels : her kayıt için ham kod listesi
    """
    cond = pd.read_csv(cond_csv)
    code_to_name = dict(zip(
        cond['Snomed_CT'].astype(str),
        cond['Acronym Name'].str.strip()
    ))

    # Tüm kodları topla
    rec_codes = [parse_dx(r) for r in tqdm(records, desc="parsing labels")]

    # Kod frekansı
    all_codes = [c for codes in rec_codes for c in codes]
    code_counts = Counter(all_codes)

    # Geçerli sınıflar: CSV'de var + min_count üzeri
    valid_codes = sorted([
        c for c, n in code_counts.items()
        if n >= min_count and c in code_to_name
    ])
    classes = [code_to_name[c] for c in valid_codes]
    code_to_idx = {c: i for i, c in enumerate(valid_codes)}

    print(f"\nGeçerli sınıf sayısı: {len(classes)}")
    for c, name in zip(valid_codes, classes):
        print(f"  {name:12s}: {code_counts[c]}")

    # Multi-hot matrix
    N, C = len(records), len(classes)
    Y = np.zeros((N, C), dtype=np.int8)
    for i, codes in enumerate(rec_codes):
        for c in codes:
            if c in code_to_idx:
                Y[i, code_to_idx[c]] = 1

    # Etiketsiz kayıtları say
    no_label = int((Y.sum(axis=1) == 0).sum())
    print(f"\nEtiketsiz kayıt: {no_label}")

    return Y, classes, rec_codes


# ─────────────────────────────────────────────────────────────────────────────
# Adım 2 — .hea/.mat → .npy
# ─────────────────────────────────────────────────────────────────────────────

def convert_to_npy(
    records : list[str],
    out_dir : str,
    overwrite: bool = False,
) -> list[int]:
    """
    Her kaydı (5000, 12) → (12, 5000) float32 .npy olarak kaydeder.
    Dosya adı: {idx:05d}.npy

    Returns
    -------
    valid_indices : başarıyla dönüştürülen kayıt indexleri
    """
    os.makedirs(out_dir, exist_ok=True)
    valid = []
    errors = []

    for i, rec in enumerate(tqdm(records, desc="convert npy")):
        out_path = os.path.join(out_dir, f"{i:05d}.npy")

        if os.path.exists(out_path) and not overwrite:
            valid.append(i)
            continue

        try:
            sig, fields = wfdb.rdsamp(rec)   # (5000, 12)

            if sig.shape != (5000, 12):
                raise ValueError(f"Beklenmeyen şekil: {sig.shape}")

            x = sig.T.astype(np.float32)     # (12, 5000)
            np.save(out_path, x)
            valid.append(i)

        except Exception as e:
            errors.append((i, str(e)))

    print(f"\n✅ Dönüştürülen: {len(valid)}")
    print(f"❌ Hata: {len(errors)}")
    if errors:
        for i, msg in errors[:5]:
            print(f"   {i}: {msg}")

    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Adım 3 — CV split
# ─────────────────────────────────────────────────────────────────────────────

def make_cv_split(
    Y          : np.ndarray,
    valid_idx  : list[int],
    classes    : list[str],
    out_dir    : str,
    k          : int = 5,
    seed       : int = 42,
) -> None:
    """
    Kayıt bazlı (hasta ID yok) stratified k-fold split.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Sadece geçerli ve en az bir etiketi olan kayıtlar
    valid_arr  = np.array(valid_idx)
    Y_valid    = Y[valid_arr]
    has_label  = Y_valid.sum(axis=1) > 0
    valid_arr  = valid_arr[has_label]
    Y_valid    = Y_valid[has_label]

    print(f"\nSplit için kayıt sayısı: {len(valid_arr)}")

    # record_id olarak index kullan
    record_ids = valid_arr.copy()

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    rows = []

    for fold, (tr_idx, te_idx) in enumerate(
        mskf.split(np.zeros(len(valid_arr)), Y_valid), start=1
    ):
        tr_ids = valid_arr[tr_idx]
        te_ids = valid_arr[te_idx]
        Y_tr   = Y_valid[tr_idx]

        # Val split
        inner = MultilabelStratifiedKFold(n_splits=5, shuffle=True,
                                          random_state=seed + fold)
        inner_tr, inner_va = next(inner.split(np.zeros(len(tr_ids)), Y_tr))
        tr_final = tr_ids[inner_tr]
        va_ids   = tr_ids[inner_va]

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        np.save(os.path.join(fold_dir, "idx_train.npy"),  tr_final)
        np.save(os.path.join(fold_dir, "idx_val.npy"),    va_ids)
        np.save(os.path.join(fold_dir, "idx_test.npy"),   te_ids)
        np.save(os.path.join(fold_dir, "ecg_id.npy"),     record_ids)
        np.save(os.path.join(fold_dir, "Y.npy"),          Y_valid.astype(np.int8))
        np.save(os.path.join(fold_dir, "classes.npy"),
                np.array(classes, dtype=object))

        rows.append({
            "fold": fold,
            "train": len(tr_final),
            "val":   len(va_ids),
            "test":  len(te_ids),
        })
        print(f"Fold {fold}: train={len(tr_final)} val={len(va_ids)} test={len(te_ids)}")

    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "fold_summary.csv"), index=False
    )
    print(f"\n✅ CV split kaydedildi: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  required=True)
    parser.add_argument("--out_npy",   required=True)
    parser.add_argument("--out_cv",    required=True)
    parser.add_argument("--min_count", type=int, default=200)
    parser.add_argument("--k",         type=int, default=5)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir
    cond_csv = os.path.join(data_dir, "ConditionNames_SNOMED-CT.csv")

    print("1) Kayıtlar toplanıyor...")
    records = collect_records(data_dir)
    print(f"   Toplam kayıt: {len(records)}")

    print("\n2) Etiketler çıkarılıyor...")
    Y, classes, _ = extract_labels(records, cond_csv, args.min_count)

    print("\n3) .npy dönüşümü başlıyor...")
    valid_idx = convert_to_npy(records, args.out_npy, args.overwrite)

    print("\n4) CV split oluşturuluyor...")
    make_cv_split(Y, valid_idx, classes, args.out_cv, args.k, args.seed)

    print("\n🎉 Tamamlandı!")


if __name__ == "__main__":
    main()
