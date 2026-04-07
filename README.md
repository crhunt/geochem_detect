# geochem_detect

Scalable Python package for **anomaly detection** and **multi-class classification** of geochemical (major-oxide) data.

## Methods
| Task | Method | Library |
|------|--------|---------|
| Anomaly detection | Isolation Forest | scikit-learn |
| Anomaly detection | Spatial Autoencoder | Keras / TensorFlow |
| Classification | Imbalance-aware MLP | Keras / TensorFlow |

Experiments are tracked with **MLFlow**. Spatial data is handled via **GeoPandas**.

---

## Getting started

### 1. Create the environment

```bash
make venv       # or: uv sync
```

### 2. Preprocess data

Raw data lives under `data/gvirm/`. Processed output mirrors the source layout under `data/processed/gvirm/`.

```bash
make preprocess          # multiclass_clean.csv  (non-spatial)
make preprocess-spatial  # Data1.csv             (lat/lon retained)
make preprocess-all      # both datasets
```

### 3. Train models

Each training script reads all hyperparameters from a YAML config file.  When
`--config` is omitted the bundled default config is used automatically.

```bash
# Isolation Forest — anomaly detection on multiclass_clean.csv
make train-iforest

# Spatial autoencoder — anomaly detection on Data1.csv (with lat/lon)
make train-autoencoder

# Multi-class MLP classifier — classification on multiclass_clean.csv
make train-classifier

# Train all three sequentially
make train-all
```

Supply a custom config to override any subset of hyperparameters:

```bash
make train-classifier CONFIG=my_configs/deep_net.yml
make train-iforest    CONFIG=my_configs/high_contamination.yml
```

Or call scripts directly:

```bash
uv run python scripts/train_isolation_forest.py --config my_configs/iforest.yml
uv run python scripts/train_autoencoder.py --spatial --config my_configs/ae.yml
uv run python scripts/train_classifier.py --config my_configs/clf.yml
```

### 4. Run predictions

After training, use the printed `run_id` (or find it in the MLFlow UI) to run
a trained model against any data split or a new dataset.

```bash
# Against individual splits used during training
make predict-train RUN_ID=<run_id> MODEL_TYPE=classifier
make predict-val   RUN_ID=<run_id> MODEL_TYPE=classifier
make predict-test  RUN_ID=<run_id> MODEL_TYPE=classifier

# Against all three splits at once
make predict-all   RUN_ID=<run_id> MODEL_TYPE=classifier

# Against a full dataset (original training data or any new file)
make predict-full  RUN_ID=<run_id> MODEL_TYPE=autoencoder
make predict-full  RUN_ID=<run_id> MODEL_TYPE=autoencoder DATA_PATH=data/gvirm/Data1.csv
```

Predictions are written to `outputs/<run_id>/predictions_<split>.csv`.

### 5. View results in MLFlow

```bash
make mlflow-ui
# open http://localhost:5000
```

---

## Model configuration

Each model has a bundled default YAML config under `src/geochem_detect/config/`:

| Model | Default config file |
|-------|---------------------|
| Isolation Forest | `default_config_isolation_forest.yml` |
| Spatial Autoencoder | `default_config_autoencoder.yml` |
| MLP Classifier | `default_config_classifier.yml` |

### Isolation Forest defaults

```yaml
model:
  n_estimators: 200
  contamination: 0.05
  max_features: 1.0
  random_state: 42
  n_jobs: -1

training:
  contamination_threshold: 0.05  # fraction of full dataset that defines "rare" classes
```

### Autoencoder defaults

```yaml
model:
  encoding_dim: 4
  hidden_dims: [32, 16]
  dropout_rate: 0.2
  learning_rate: 0.001
  epochs: 50
  batch_size: 64
  patience: 10

training:
  spatial: false              # set true to include scaled lat/lon as auxiliary inputs
  contamination_threshold: 0.05
```

### Classifier defaults

```yaml
model:
  hidden_dims: [64, 32]
  dropout_rate: 0.3
  learning_rate: 0.001
  epochs: 100
  batch_size: 64
  patience: 15
```

Custom configs only need to include the keys you want to override — all other
values fall back to the defaults shown above.

---

## Package structure

```
src/geochem_detect/
├── config/
│   ├── __init__.py                          # load_config(), model_params(), training_params()
│   ├── default_config_isolation_forest.yml
│   ├── default_config_autoencoder.yml
│   └── default_config_classifier.yml
├── data/
│   ├── loader.py        # CSV → DataFrame / GeoDataFrame
│   └── preprocessor.py  # scaling, encoding, 70/15/15 splits with index tracking
├── models/
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   └── classifier.py
├── training/
│   └── trainer.py       # MLFlow wrappers; saves artefacts per run
└── visualization/
    └── plots.py         # PR curves, confusion matrix, spatial map

scripts/
├── preprocess_data.py   # env-var-driven preprocessing (spatial + non-spatial)
├── train_isolation_forest.py
├── train_autoencoder.py
├── train_classifier.py
└── predict.py           # run any trained model against any split or dataset
```

### Run artefacts

Each training run saves the following under `outputs/<run_id>/artefacts/`:

| File | Contents |
|------|----------|
| `scaler.pkl` | Fitted `RobustScaler` |
| `label_encoder.pkl` | Fitted `LabelEncoder` |
| `splits.npz` | `train_idx`, `val_idx`, `test_idx` |
| `dataset_info.json` | Dataset path, feature columns, label column |
| `model.pkl` or `keras_model.keras` | Serialised model |

Plots are written to `outputs/<run_id>/`.

---

## Makefile reference

Run `make help` to list all targets.  Key targets:

| Target | Description |
|--------|-------------|
| `venv` | Create / sync the virtual environment |
| `preprocess[-spatial\|-all]` | Preprocess datasets |
| `train-iforest` | Train Isolation Forest |
| `train-autoencoder` | Train spatial autoencoder |
| `train-classifier` | Train MLP classifier |
| `train-all` | Train all three models |
| `predict-[train\|val\|test\|all\|full]` | Run a trained model (requires `RUN_ID=` `MODEL_TYPE=`) |
| `mlflow-ui` | Launch MLFlow UI at `http://localhost:5000` |
| `lint` / `format` | ruff check / format |
| `clean` | Remove outputs and caches |
| `clean-processed` | Remove processed data |

---

## Performance metric

Primary metric: **PR-AUC** (macro-averaged for the classifier; binary for anomaly detectors).
