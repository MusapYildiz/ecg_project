"""
training/trainer.py
===================
Eğitim döngüsü.

Kural: criterion / optimizer / scaler hiçbir zaman global scope'ta değil.
       Her fold kendi nesnelerini oluşturur.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from evaluation.metrics    import compute_all_metrics
from evaluation.threshold  import tune_thresholds
from evaluation.reporter   import save_fold_report
from models.registry       import save_checkpoint, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Tek epoch — eğitim
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler   : torch.cuda.amp.GradScaler,
    device   : torch.device,
    desc     : str = "train",
) -> float:
    model.train()
    total = 0.0

    for xb, yb, _ in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = criterion(model(xb), yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * xb.size(0)

    return total / len(loader.dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Tek epoch — değerlendirme
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model    : nn.Module,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    threshold: float | np.ndarray = 0.5,
    desc     : str = "eval",
) -> dict[str, float]:
    """
    Returns
    -------
    dict: loss, macro_auc, macro_f1, subset_acc, hamming_acc
    """
    model.eval()
    total, logits_all, true_all = 0.0, [], []

    for xb, yb, _ in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        total += criterion(logits, yb).item() * xb.size(0)
        logits_all.append(logits.cpu().numpy())
        true_all.append(yb.cpu().numpy())

    logits_np = np.concatenate(logits_all)
    y_true    = np.concatenate(true_all).astype(np.int32)
    y_prob    = 1.0 / (1.0 + np.exp(-logits_np))

    metrics         = compute_all_metrics(y_true, y_prob, threshold)
    metrics["loss"] = total / len(loader.dataset)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Olasılık toplayıcı  (threshold tuning için)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_probs(
    model : nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """(y_prob, y_true) döndürür."""
    model.eval()
    logits_all, true_all = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        logits_all.append(model(xb).cpu().numpy())
        true_all.append(yb.numpy())
    y_prob = 1.0 / (1.0 + np.exp(-np.concatenate(logits_all)))
    y_true = np.concatenate(true_all).astype(np.int32)
    return y_prob, y_true


# ─────────────────────────────────────────────────────────────────────────────
# Tam fold eğitimi
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    model       : nn.Module,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    test_loader : DataLoader,
    device      : torch.device,
    ckpt_path   : Path,
    report_dir  : Path,
    fold        : int,
    class_names : list[str],
    epochs      : int = 20,
    lr          : float = 1e-3,
    weight_decay: float = 1e-4,
    use_amp     : bool  = True,
    tune_thr    : bool  = True,
    thr_steps   : int   = 91,
    param_groups: Optional[list[dict]] = None,
) -> dict[str, float]:
    """
    Bir fold için tam eğitim + val izleme + test değerlendirmesi + rapor kaydı.

    Parameters
    ----------
    param_groups : None → tüm trainable parametreler tek lr ile;
                   list → AdamW'ye doğrudan (partial fine-tune için).

    Returns
    -------
    dict — fold özet metrikleri
    """
    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    if param_groups is None:
        optim = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=weight_decay,
        )
    else:
        optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    best_auc = -1.0

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optim, criterion, scaler, device,
            desc=f"ep{epoch:02d}/train",
        )
        val_m = eval_epoch(
            model, val_loader, criterion, device,
            threshold=0.5, desc=f"ep{epoch:02d}/val",
        )

        print(
            f"  Ep{epoch:02d} | "
            f"train_loss={tr_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_AUC={val_m['macro_auc']:.4f} | "
            f"val_F1={val_m['macro_f1']:.4f}"
        )

        if not np.isnan(val_m["macro_auc"]) and val_m["macro_auc"] > best_auc:
            best_auc = val_m["macro_auc"]
            save_checkpoint(model, ckpt_path, epoch=epoch, val_auc=best_auc)

    # ── Test: best checkpoint + threshold tuning ──────────────────────────────
    model = load_checkpoint(model, ckpt_path, device=device)

    if tune_thr:
        p_val, y_val = collect_probs(model, val_loader, device)
        thresholds   = tune_thresholds(y_val, p_val, n_steps=thr_steps)
    else:
        thresholds = np.full(len(class_names), 0.5, dtype=np.float32)

    p_test, y_test = collect_probs(model, test_loader, device)

    # ── Detaylı rapor kaydet ──────────────────────────────────────────────────
    summary_row = save_fold_report(
        fold        = fold,
        y_true      = y_test,
        y_prob      = p_test,
        thresholds  = thresholds,
        class_names = class_names,
        report_dir  = report_dir,
    )

    return {"best_val_auc": float(best_auc), **summary_row}
