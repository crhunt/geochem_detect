# geochem

Scalable Python package for **anomaly detection** and **multi-class classification** of geochemical (major-oxide) data.

## Methods
| Task | Method | Library |
|------|--------|---------|
| Anomaly detection | Isolation Forest | scikit-learn |
| Anomaly detection | Spatial Autoencoder | Keras / TensorFlow |
| Classification | Imbalance-aware MLP | Keras / TensorFlow |

Experiments are tracked with **MLFlow**. Spatial data is handled via **GeoPandas**.

## Setup
```bash
uv sync
```

## Train models

```bash
# Isolation Forest (multiclass_clean.csv)
uv run python scripts/train_isolation_forest.py --contamination 0.05

# Autoencoder (Data1.csv, optionally with spatial features)
uv run python scripts/train_autoencoder.py --epochs 50 --spatial

# Multi-class classifier (multiclass_clean.csv)
uv run python scripts/train_classifier.py --epochs 100
```

## View MLFlow UI
```bash
uv run mlflow ui
# open http://localhost:5000
```

## Package structure
```
src/geochem/
├── data/
│   ├── loader.py        # CSV → DataFrame / GeoDataFrame
│   └── preprocessor.py  # scaling, encoding, splitting
├── models/
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   └── classifier.py
├── training/
│   └── trainer.py       # MLFlow wrappers
└── visualization/
    └── plots.py         # PR curves, confusion matrix, spatial map
```

## Performance metric
Primary metric: **PR-AUC** (macro-averaged for the classifier; binary for anomaly detectors).
