# ECG Classification — PTB-XL

12-lead EKG sinyallerinde multi-label diagnostic superclass sınıflandırması.

## Proje yapısı

```
ecg_project/
├── config.py               # Tüm path, sabit ve hyperparameter tanımları
├── run_experiment.py        # Ana giriş noktası
│
├── data/
│   ├── dataset.py          # PTBXLNpyDataset
│   └── loaders.py          # make_loaders()
│
├── models/
│   ├── backbones.py        # ResNet1D, SEResNet1D, InceptionTime1D
│   ├── heads.py            # LinearHead, MLPHead, KANHead
│   └── registry.py         # build_model(), checkpoint helpers
│
├── training/
│   └── trainer.py          # train_fold(), train_epoch(), eval_epoch()
│
├── evaluation/
│   ├── metrics.py          # macro_auroc, macro_f1, compute_all_metrics
│   └── threshold.py        # tune_thresholds()
│
├── notebooks/
│   └── 00_setup.ipynb      # Colab başlangıç notebook'u
│
├── scripts/
│   └── prepare_labels.py   # PTB-XL label extraction (bir kez çalıştırılır)
│
├── requirements.txt
└── .gitignore
```

## Colab'da kullanım

```python
# Her session başında:
!git clone https://github.com/KULLANICI/ecg_project.git
import sys; sys.path.insert(0, "/content/ecg_project")

# Deney çalıştır:
from run_experiment import run
run("resnet1d_100hz", epochs=20)
```

## Veri akışı

```
PTB-XL .hea/.dat  →  npy_100/  (records100, 100 Hz)
                  →  npy_500/  (records500, 500 Hz)
ptbxl_database.csv + scp_statements.csv  →  labels_superclass/
labels + CV split  →  cv_k5/fold_{1..5}/
```

Checkpointler ve raporlar `/content/drive/MyDrive/ecg_outputs/` altına kaydedilir.
