# ECG Classification — PTB-XL

12-lead EKG sinyallerinde multi-label diagnostic superclass sınıflandırması.  
PTB-XL veri seti üzerinde 5-fold cross-validation ile model karşılaştırması.

---

## Proje yapısı

```
ecg_project/
├── config.py                   # Tüm path, sabit ve hyperparameter tanımları
├── run_experiment.py           # Ana giriş noktası
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PTBXLNpyDataset (100 Hz ve 500 Hz)
│   └── loaders.py              # make_loaders() — fold bazlı DataLoader
│
├── models/
│   ├── __init__.py
│   ├── backbones.py            # ResNet1D, SEResNet1D, InceptionTime1D
│   ├── heads.py                # LinearHead, MLPHead, KANHead
│   └── registry.py            # build_model(), freeze/unfreeze, checkpoint helpers
│
├── training/
│   ├── __init__.py
│   └── trainer.py             # train_fold(), train_epoch(), eval_epoch()
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # macro_auroc, macro_f1, compute_all_metrics
│   ├── threshold.py           # tune_thresholds() — per-class val F1 optimizasyonu
│   └── reporter.py            # Detaylı fold raporu, confusion matrix, raw data kaydı
│
├── scripts/
│   ├── prepare_labels.py      # PTB-XL label extraction + CV split (bir kez çalıştırılır)
│   └── convert_to_npy.py     # wfdb → .npy dönüşümü (100 Hz ve 500 Hz)
│
├── notebooks/
│   └── 00_colab_runner.ipynb  # Colab çalıştırma notebook'u
│
├── requirements.txt
└── .gitignore
```

---

## Desteklenen modeller

### End-to-end backbone'lar
| Preset | Açıklama |
|--------|----------|
| `resnet1d_100hz` | ResNet1D — 100 Hz |
| `seresnet1d_100hz` | SE-ResNet1D — 100 Hz |
| `inceptiontime_100hz` | InceptionTime1D — 100 Hz |
| `resnet1d_500hz` | ResNet1D — 500 Hz |
| `seresnet1d_500hz` | SE-ResNet1D — 500 Hz |
| `inceptiontime_500hz` | InceptionTime1D — 500 Hz |

### Frozen backbone + head
| Preset | Açıklama |
|--------|----------|
| `inception_frozen_linear_100hz` | InceptionTime backbone + Linear head |
| `inception_frozen_mlp_100hz` | InceptionTime backbone + MLP head |
| `inception_frozen_kan_100hz` | InceptionTime backbone + KAN head |

### Partial fine-tune
| Preset | Açıklama |
|--------|----------|
| `inception_partial_ft_mlp_100hz` | Son 1 blok açık + MLP head |
| `inception_partial_ft_kan_100hz` | Son 1 blok açık + KAN head |

> Frozen / partial fine-tune deneyleri için `inceptiontime_100hz` backbone'unun
> önceden eğitilmiş olması gerekir.

---

## Colab'da kullanım

### Her session başında

```python
!pip -q install wfdb iterative-stratification tqdm

import sys, os

if not os.path.exists("/content/ecg_project"):
    !git clone https://github.com/MusapYildiz/ecg_project.git /content/ecg_project
else:
    !git -C /content/ecg_project pull

if "/content/ecg_project" not in sys.path:
    sys.path.insert(0, "/content/ecg_project")

from google.colab import drive
drive.mount("/content/drive")
```

### Veriyi hazırla (her session)

```python
# npy_100 extract
if not os.path.exists("/content/PTBXL_records100_restore/npy_100"):
    !mkdir -p /content/PTBXL_records100_restore
    !tar -xf "/content/drive/MyDrive/PTBXL_exports/npy_100.tar" \
             -C /content/PTBXL_records100_restore/

# CV split
if not os.path.exists("/content/cv_npy100_patientwise_superclass_k5"):
    !cp -r "/content/drive/MyDrive/ecg-auto-dx/cv_npy100_patientwise_superclass_k5" \
            /content/
```

### Deney çalıştır

```python
from run_experiment import run

# Tek model:
run("resnet1d_100hz", epochs=20)

# 3 backbone birden:
from run_experiment import run_all_end2end
run_all_end2end(hz=100, epochs=20)

# Frozen head (InceptionTime önce eğitilmiş olmalı):
BACKBONE = "/content/drive/MyDrive/ecg_outputs/checkpoints/inceptiontime_100hz"
run("inception_frozen_kan_100hz", backbone_ckpt_dir=BACKBONE, epochs=10)

# Partial fine-tune:
run("inception_partial_ft_mlp_100hz", backbone_ckpt_dir=BACKBONE, epochs=20)
```

### Sonuçları karşılaştır

```python
import json, glob, pandas as pd

summaries = []
for p in glob.glob("/content/drive/MyDrive/ecg_outputs/reports/*/summary.json"):
    with open(p) as f:
        summaries.append(json.load(f))

df = pd.DataFrame(summaries)
cols = ["experiment", "macro_auc_mean", "macro_auc_std",
        "macro_f1_mean", "macro_f1_std"]
df[cols].sort_values("macro_auc_mean", ascending=False).round(4)
```

### Sonradan yeniden analiz (model gerekmez)

```python
from evaluation.reporter import reanalyze

# Orijinal threshold ile:
df = reanalyze(
    "/content/drive/MyDrive/ecg_outputs/reports/resnet1d_100hz",
    class_names=["CD", "HYP", "MI", "NORM", "STTC"],
)

# Farklı threshold ile:
df = reanalyze(..., new_threshold=0.4)
```

---

## Drive çıktı yapısı

Her deney tamamlandığında Drive'a şu yapı kaydedilir:

```
ecg_outputs/
├── checkpoints/
│   └── resnet1d_100hz/
│       ├── fold_1_best.pt
│       └── ...
└── reports/
    └── resnet1d_100hz/
        ├── fold_1/
        │   ├── y_true.npy                  # Test seti gerçek etiketler
        │   ├── y_prob.npy                  # Test seti tahmin olasılıkları
        │   ├── thresholds.npy              # Per-class optimal eşikler
        │   ├── per_class_metrics.csv       # TP/FP/FN/TN/precision/recall/f1
        │   ├── summary_metrics.csv         # Macro metrikler + per-class AUC
        │   └── confusion_matrix_{cls}.png  # Her sınıf için 2x2 CM
        ├── fold_2/ ... fold_5/
        ├── fold_results.csv                # 5 fold özet tablosu
        ├── summary.json                    # Mean ± std
        └── summary_table.csv              # Okunabilir özet
```

---

## İlk kurulum — label ve CV split oluşturma

`ptbxl_database.csv` ve `scp_statements.csv` hazırsa:

```bash
python scripts/prepare_labels.py \
    --root    "/content/drive/MyDrive/ecg-auto-dx/ptb-xl-.../" \
    --npy_dir "/content/PTBXL_records100_restore/npy_100" \
    --out_dir "/content/cv_npy100_patientwise_superclass_k5" \
    --sr      100
```

Ham `.hea/.dat` kayıtlarını `.npy`'ye dönüştürmek için:

```bash
python scripts/convert_to_npy.py \
    --records_dir /content/PTBXL_records500/records500 \
    --out_dir     /content/PTBXL_records500/npy_500 \
    --sr          500
```

---

## Gereksinimler

```
torch>=2.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
wfdb>=4.1
tqdm>=4.65
iterative-stratification>=0.1.7
matplotlib>=3.7
```
