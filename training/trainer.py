"""
training/trainer.py
===================
Tek, temiz eğitim döngüsü.
Global criterion / optimizer / scaler YOK — hepsi lokal scope'ta.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from evaluation.metrics import compute_all_metrics, macro_auroc
from evaluation.threshold import tune_thresholds
from models.registry import save_checkpoint


# ── Tek epoch — eğitim ───────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    desc: str = "train",
) -> float:
    model.train()
    total_loss = 0.0

    for xb, yb, _ in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            loss = criterion(model(xb), yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * xb.size(0)

    return total_loss / len(loader.dataset)


# ── Tek epoch — değerlendirme ─────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold=0.5,
    desc: str = "eval",
) -> dict:
    """
    Returns
    -------
    dict:  loss, macro_auc, macro_f1, subset_acc, hamming_acc
    """
    model.eval()
    total_loss = 0.0
    all_logits, all_true = [], []

    for xb, yb, _ in tqdm(loader, desc=desc, leave=False):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)

        all_logits.append(logits.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    y_true    = np.concatenate(all_true).astype(np.int32)
    y_prob    = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid

    metrics = compute_all_metrics(y_true, y_prob, threshold)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def collect_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """(y_prob, y_true) toplayıcı — threshold tuning için."""
    model.eval()
    all_logits, all_true = [], []
    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        all_logits.append(model(xb).cpu().numpy())
        all_true.append(yb.numpy())
    y_prob = 1.0 / (1.0 + np.exp(-np.concatenate(all_logits)))
    y_true = np.concatenate(all_true).astype(np.int32)
    return y_prob, y_true


# ── Tam fold eğitimi ─────────────────────────────────────────────────────────

def train_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    ckpt_path: Path,
    epochs: int = 20,
    lr: float = 1e-3,
    lr_backbone: float = 1e-4,
    weight_decay: float = 1e-4,
    use_amp: bool = True,
    tune_thr: bool = True,
    thr_steps: int = 91,
    param_groups: Optional[list] = None,
) -> dict:
    """
    Bir fold için tam eğitim + val izleme + test değerlendirmesi.

    Parameters
    ----------
    param_groups : None → tüm parametreler aynı lr ile;
                   list → AdamW'ye doğrudan verilir (kısmi fine-tune için).

    Returns
    -------
    dict: fold sonuçları (test metrikleri + best_val_auc)
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    if param_groups is None:
        optim = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay,
        )
    else:
        optim = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    best_auc  = -1.0
    fixed_thr = 0.5

    for epoch in range(1, epochs + 1):
        tr_loss = train_epoch(
            model, train_loader, optim, criterion, scaler, device,
            desc=f"ep{epoch:02d} train",
        )
        val_m = eval_epoch(
            model, val_loader, criterion, device,
            threshold=fixed_thr,
            desc=f"ep{epoch:02d} val",
        )

        print(
            f"Ep{epoch:02d} | train_loss={tr_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | "
            f"val_AUC={val_m['macro_auc']:.4f} | "
            f"val_F1={val_m['macro_f1']:.4f}"
        )

        if not np.isnan(val_m["macro_auc"]) and val_m["macro_auc"] > best_auc:
            best_auc = val_m["macro_auc"]
            save_checkpoint(model, ckpt_path, epoch=epoch, val_auc=best_auc)

    # ── Test: best checkpoint + threshold tuning ──────────────────────────────
    from models.registry import load_checkpoint
    model = load_checkpoint(model, ckpt_path, device=device)

    # threshold tuning on val
    if tune_thr:
        p_val, y_val = collect_probs(model, val_loader, device)
        thresholds   = tune_thresholds(y_val, p_val, n_steps=thr_steps)
    else:
        thresholds = fixed_thr

    p_test, y_test = collect_probs(model, test_loader, device)
    test_m = compute_all_metrics(y_test, p_test, thresholds)

    print(
        f"\n[TEST] AUC={test_m['macro_auc']:.4f} | "
        f"F1={test_m['macro_f1']:.4f} | "
        f"subsetAcc={test_m['subset_acc']:.4f} | "
        f"hammAcc={test_m['hamming_acc']:.4f}"
    )

    return {"best_val_auc": best_auc, **{f"test_{k}": v for k, v in test_m.items()}}
