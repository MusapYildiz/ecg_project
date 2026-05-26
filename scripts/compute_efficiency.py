"""
scripts/compute_efficiency.py
==============================
Kaydedilmiş checkpoint'lerden model verimlilik metriklerini hesaplar.

Ölçülen metrikler:
  - Parametre sayısı (toplam ve eğitilebilir)
  - Model boyutu (MB)
  - FLOPs (GFLOPs) — dummy input ile
  - Inference hızı — batch=1 (ms/örnek) ve batch=32 (örnek/saniye)
  - GPU bellek kullanımı (inference sırasında, MB)

Colab kullanımı:
    !pip -q install thop
    !python /content/ecg_project/scripts/compute_efficiency.py \
        --ckpt_dir  /content/drive/MyDrive/ecg_outputs/checkpoints \
        --out_dir   /content/drive/MyDrive/ecg_outputs/reports \
        --device    cuda
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# thop: FLOPs hesabı için
try:
    from thop import profile, clever_format
    _THOP = True
except ImportError:
    _THOP = False
    print("⚠️  thop bulunamadı. FLOPs hesaplanmayacak. pip install thop")


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — preset adından modeli kur
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_from_preset(preset: str, num_classes: int = 5) -> torch.nn.Module:
    """Preset adına göre doğru modeli ve dummy input'u döndürür."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from config import ModelConfig
    from models.registry import build_model

    name = preset.lower()

    # 2D modeller
    if "resnet2d" in name:
        img_size = 224 if "224" in name else 128
        cfg = ModelConfig(name="resnet2d", num_classes=num_classes,
                         pretrained=False, cwt_img_size=img_size)
        dummy = torch.randn(1, 12, img_size, img_size)
        return build_model(cfg), dummy

    if "vitbase2d" in name:
        img_size = 224 if "224" in name else 128
        cfg = ModelConfig(name="vitbase2d", num_classes=num_classes,
                         pretrained=False, cwt_img_size=img_size)
        dummy = torch.randn(1, 12, img_size, img_size)
        return build_model(cfg), dummy

    # 1D modeller
    if "resnet1d" in name or "seresnet1d" in name:
        t = 5000 if "500hz" in name else 1000
        model_name = "seresnet1d" if "seresnet" in name else "resnet1d"
        cfg = ModelConfig(name=model_name, num_classes=num_classes)
        dummy = torch.randn(1, 12, t)
        return build_model(cfg), dummy

    if "inceptiontime" in name:
        t = 5000 if "500hz" in name else 1000
        cfg = ModelConfig(name="inceptiontime", num_classes=num_classes)
        dummy = torch.randn(1, 12, t)
        return build_model(cfg), dummy

    if "tcn" in name:
        t = 5000 if "500hz" in name else 1000
        cfg = ModelConfig(name="tcn", num_classes=num_classes)
        dummy = torch.randn(1, 12, t)
        return build_model(cfg), dummy

    # Head modeller (inception + head)
    if "frozen_linear" in name or "frozen_mlp" in name or \
       "frozen_kan" in name or "partial_ft" in name:
        t = 5000 if "500hz" in name else 1000
        if "linear" in name:
            head = "linear"
        elif "kan" in name:
            head = "kan"
        else:
            head = "mlp"
        cfg = ModelConfig(name="inceptiontime", num_classes=num_classes,
                         head_type=head, freeze_backbone=True,
                         unfreeze_last_n_blocks=0)
        dummy = torch.randn(1, 12, t)
        return build_model(cfg), dummy

    raise ValueError(f"Preset tanınamadı: {preset}")


# ─────────────────────────────────────────────────────────────────────────────
# Metrik hesaplayıcılar
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: torch.nn.Module) -> tuple[int, int]:
    """(total_params, trainable_params)"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(ckpt_path: Path) -> float:
    """Checkpoint dosya boyutu (MB)."""
    return os.path.getsize(ckpt_path) / (1024 ** 2)


def compute_flops(model: torch.nn.Module, dummy: torch.Tensor) -> tuple[float, float]:
    """
    (flops_G, params_M) — GFLOPs ve milyon parametre.
    thop kütüphanesi gereklidir.
    """
    if not _THOP:
        return float("nan"), float("nan")
    model.eval()
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy,), verbose=False)
    return flops / 1e9, params / 1e6


def measure_inference_speed(
    model     : torch.nn.Module,
    dummy_b1  : torch.Tensor,
    dummy_b32 : torch.Tensor,
    device    : torch.device,
    n_warmup  : int = 10,
    n_runs    : int = 100,
) -> dict[str, float]:
    """
    Inference hızını ölçer.

    Returns
    -------
    dict:
        latency_ms    — batch=1, ms/örnek
        throughput    — batch=32, örnek/saniye
        gpu_mem_mb    — peak GPU bellek (MB), sadece CUDA
    """
    model = model.to(device).eval()
    dummy_b1  = dummy_b1.to(device)
    dummy_b32 = dummy_b32.to(device)

    # GPU bellek sıfırla
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_b1)

    # Batch=1 latency
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_b1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) / n_runs * 1000

    # Batch=32 throughput
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy_b32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed    = (time.perf_counter() - t0) / n_runs
    throughput = 32 / elapsed  # örnek/saniye

    # GPU bellek
    gpu_mem_mb = float("nan")
    if device.type == "cuda":
        gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "latency_ms"  : round(latency_ms, 3),
        "throughput"  : round(throughput, 1),
        "gpu_mem_mb"  : round(gpu_mem_mb, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Ana fonksiyon
# ─────────────────────────────────────────────────────────────────────────────

def compute_all(
    ckpt_dir  : Path,
    out_dir   : Path,
    device    : torch.device,
    num_classes: int = 5,
    fold      : int  = 1,
) -> pd.DataFrame:
    """
    ckpt_dir altındaki tüm deney klasörlerini tarar,
    fold_1_best.pt checkpoint'ini yükler, metrikleri hesaplar.
    """
    ckpt_dir = Path(ckpt_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Her deney klasörü = bir preset
    experiment_dirs = sorted([d for d in ckpt_dir.iterdir() if d.is_dir()])
    if not experiment_dirs:
        raise FileNotFoundError(f"Checkpoint klasörü bulunamadı: {ckpt_dir}")

    print(f"Toplam deney: {len(experiment_dirs)}")
    print(f"Device: {device}\n")

    rows = []

    for exp_dir in experiment_dirs:
        preset    = exp_dir.name
        ckpt_path = exp_dir / f"fold_{fold}_best.pt"

        if not ckpt_path.exists():
            print(f"⏭️  Atlandı (checkpoint yok): {preset}")
            continue

        print(f"📊 {preset} ...", end=" ", flush=True)

        try:
            # Model kur
            model, dummy_b1 = _build_model_from_preset(preset, num_classes)

            # Checkpoint yükle
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state"], strict=False)
            model.eval()

            # dummy batch=32
            dummy_b32 = dummy_b1.repeat(32, 1, *([1] * (dummy_b1.ndim - 2)))

            # Metrikler
            total_p, train_p = count_params(model)
            size_mb           = model_size_mb(ckpt_path)
            flops_g, _        = compute_flops(model, dummy_b1)
            speed             = measure_inference_speed(
                model, dummy_b1, dummy_b32, device
            )

            row = {
                "experiment"       : preset,
                "total_params_M"   : round(total_p / 1e6, 2),
                "trainable_params_M": round(train_p / 1e6, 2),
                "model_size_MB"    : round(size_mb, 1),
                "flops_G"          : round(flops_g, 2) if not np.isnan(flops_g) else "N/A",
                "latency_ms"       : speed["latency_ms"],
                "throughput_per_s" : speed["throughput"],
                "gpu_mem_MB"       : speed["gpu_mem_mb"],
            }
            rows.append(row)
            print(f"✅  params={row['total_params_M']}M | "
                  f"flops={row['flops_G']}G | "
                  f"latency={row['latency_ms']}ms")

        except Exception as e:
            print(f"❌  Hata: {e}")

        finally:
            # Belleği temizle
            try:
                del model
            except:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # Kaydet
    csv_path  = out_dir / "efficiency_metrics.csv"
    json_path = out_dir / "efficiency_metrics.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"\n✅ Kaydedildi:\n   {csv_path}\n   {json_path}")
    print("\n" + df.to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint'lerden model verimlilik metriklerini hesaplar"
    )
    parser.add_argument(
        "--ckpt_dir", required=True,
        help="Checkpoint'lerin bulunduğu ana klasör "
             "(içinde resnet1d_100hz/, inceptiontime_100hz/ gibi alt klasörler olmalı)"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="efficiency_metrics.csv ve .json'ın kaydedileceği klasör"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Inference cihazı (default: cuda)"
    )
    parser.add_argument(
        "--num_classes", type=int, default=5,
        help="Sınıf sayısı (default: 5)"
    )
    parser.add_argument(
        "--fold", type=int, default=1,
        help="Hangi fold checkpoint'i kullanılacak (default: 1)"
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available())
        else "cpu"
    )

    compute_all(
        ckpt_dir    = Path(args.ckpt_dir),
        out_dir     = Path(args.out_dir),
        device      = device,
        num_classes = args.num_classes,
        fold        = args.fold,
    )


if __name__ == "__main__":
    main()
