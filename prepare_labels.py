"""
scripts/prepare_labels.py
=========================
PTB-XL'den diagnostic superclass etiketlerini çıkarır ve
patient-wise 5-fold CV split'i oluşturur.

Colab'da bir kez çalıştırın:
    !python /content/ecg_project/scripts/prepare_labels.py \\
        --root /content \\
        --npy_dir /content/PTBXL_records100_restore/npy_100 \\
        --out_dir /content/cv_npy100_patientwise_superclass_k5 \\
        --sr 100

Çıktılar out_dir/fold_{1..5}/ altına kaydedilir.
Drive'a taşımayı unutmayın:
    !cp -r /content/cv_npy100_patientwise_superclass_k5 /content/drive/MyDrive/ecg_outputs/
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


# ─────────────────────────────────────────────────────────────────────────────
# Label extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_superclass_labels(root: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Returns
    -------
    Y        : (N, C) int8   multi-hot
    ecg_id   : (N,)   int
    classes  : list[str]     sınıf isimleri
    df       : DataFrame     (ecg_id, patient_id, superclasses)
    """
    db_csv  = os.path.join(root, "ptbxl_database.csv")
    scp_csv = os.path.join(root, "scp_statements.csv")

    assert os.path.exists(db_csv),  f"Bulunamadı: {db_csv}"
    assert os.path.exists(scp_csv), f"Bulunamadı: {scp_csv}"

    df  = pd.read_csv(db_csv)
    scp = pd.read_csv(scp_csv, index_col=0)

    # superclass kolon adını bul
    super_col = next(
        (c for c in ["diagnostic_class", "diagnostic_superclass"] if c in scp.columns),
        None,
    )
    if super_col is None:
        raise ValueError("scp_statements.csv'de 'diagnostic_class' veya 'diagnostic_superclass' yok.")

    code_to_super: dict[str, str] = (
        scp[super_col].dropna().astype(str).to_dict()
    )

    def _parse(x):
        if pd.isna(x): return {}
        try:    return ast.literal_eval(x)
        except: return {}

    def _supers(d: dict) -> list[str]:
        return sorted({code_to_super[k] for k in d if k in code_to_super})

    df["scp_dict"]    = df["scp_codes"].apply(_parse)
    df["superclasses"] = df["scp_dict"].apply(_supers)

    all_classes = sorted({s for lst in df["superclasses"] for s in lst})
    c2i         = {c: i for i, c in enumerate(all_classes)}

    N, C = len(df), len(all_classes)
    Y    = np.zeros((N, C), dtype=np.int8)
    for i, lst in enumerate(df["superclasses"]):
        for c in lst:
            Y[i, c2i[c]] = 1

    # etiketi olmayan kayıtları çıkar
    mask   = Y.sum(axis=1) > 0
    Y      = Y[mask]
    ecg_id = df["ecg_id"].to_numpy()[mask].astype(int)

    # patient_id
    pat_col = next((c for c in ["patient_id","patient","patientid","subject_id","subject"]
                    if c in df.columns), None)
    if pat_col is None:
        raise ValueError(f"Hasta ID kolonu bulunamadı. Kolonlar: {list(df.columns)}")

    df_out = df[["ecg_id", pat_col]].iloc[mask].reset_index(drop=True)
    df_out = df_out.rename(columns={pat_col: "patient_id"})

    print(f"Superclasses ({C}): {all_classes}")
    print(f"Etiketli kayıt sayısı: {len(ecg_id)}")
    for c, cnt in zip(all_classes, Y.sum(axis=0)):
        print(f"  {c}: {int(cnt)}")

    return Y, ecg_id, all_classes, df_out


# ─────────────────────────────────────────────────────────────────────────────
# CV split
# ─────────────────────────────────────────────────────────────────────────────

def make_cv_split(
    Y: np.ndarray,
    ecg_id: np.ndarray,
    df_meta: pd.DataFrame,
    classes: list[str],
    out_dir: str,
    k: int = 5,
    seed: int = 42,
) -> None:
    patient_id = df_meta["patient_id"].astype(int).to_numpy()
    unique_pts, inv = np.unique(patient_id, return_inverse=True)
    P = len(unique_pts)

    # Patient-level OR label
    Y_pat = np.zeros((P, Y.shape[1]), dtype=np.int8)
    for i in range(len(ecg_id)):
        Y_pat[inv[i]] |= Y[i]

    mskf = MultilabelStratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    rows = []
    for fold, (p_tr_idx, p_te_idx) in enumerate(mskf.split(np.zeros(P), Y_pat), start=1):
        tr_pts  = unique_pts[p_tr_idx]
        te_pts  = unique_pts[p_te_idx]
        Y_tr    = Y_pat[p_tr_idx]

        # inner val split
        inner = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=seed + fold)
        inner_tr, inner_va = next(inner.split(np.zeros(len(tr_pts)), Y_tr))
        tr_pts_final = tr_pts[inner_tr]
        va_pts       = tr_pts[inner_va]

        idx_train = np.where(np.isin(patient_id, tr_pts_final))[0]
        idx_val   = np.where(np.isin(patient_id, va_pts))[0]
        idx_test  = np.where(np.isin(patient_id, te_pts))[0]

        # overlap kontrolü
        s_tr = set(patient_id[idx_train].tolist())
        s_va = set(patient_id[idx_val].tolist())
        s_te = set(patient_id[idx_test].tolist())
        assert s_tr.isdisjoint(s_va) and s_tr.isdisjoint(s_te) and s_va.isdisjoint(s_te), \
            "Hasta leak tespit edildi!"

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        np.save(os.path.join(fold_dir, "idx_train.npy"),  idx_train)
        np.save(os.path.join(fold_dir, "idx_val.npy"),    idx_val)
        np.save(os.path.join(fold_dir, "idx_test.npy"),   idx_test)
        np.save(os.path.join(fold_dir, "ecg_id.npy"),     ecg_id)
        np.save(os.path.join(fold_dir, "patient_id.npy"), patient_id)
        np.save(os.path.join(fold_dir, "Y.npy"),          Y.astype(np.int8))
        np.save(os.path.join(fold_dir, "classes.npy"),    np.array(classes, dtype=object))

        rows.append({
            "fold": fold,
            "train_records": len(idx_train), "val_records": len(idx_val),
            "test_records":  len(idx_test),
            "train_patients": len(s_tr), "val_patients": len(s_va),
            "test_patients":  len(s_te),
        })
        print(f"Fold {fold}: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "fold_summary.csv"), index=False)
    print(f"\n✅ CV split kaydedildi: {out_dir}")
    print(summary.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PTB-XL label extraction + CV split")
    parser.add_argument("--root",    required=True, help="ptbxl_database.csv'nin bulunduğu klasör")
    parser.add_argument("--npy_dir", required=True, help=".npy sinyal dosyalarının klasörü")
    parser.add_argument("--out_dir", required=True, help="CV split çıktı klasörü")
    parser.add_argument("--sr",      type=int, default=100, help="Örnekleme hızı (100 veya 500)")
    parser.add_argument("--k",       type=int, default=5,   help="Fold sayısı")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    Y, ecg_id, classes, df_meta = extract_superclass_labels(args.root)
    make_cv_split(Y, ecg_id, df_meta, classes, args.out_dir, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
