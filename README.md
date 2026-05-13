# geochem_detect

Scalable Python package for **anomaly detection** and **multi-class classification** of geochemical data.

## Methods
| Task | Method | Library |
|------|--------|---------|
| Anomaly detection | Isolation Forest | scikit-learn |
| Anomaly detection | Autoencoder | Keras / TensorFlow |
| Anomaly detection | CNN + Sparse Autoencoder (CNN-SAE) | Keras / TensorFlow |
| Classification | Imbalance-aware MLP | Keras / TensorFlow |

Experiments are tracked with **MLflow**. Spatial data is handled with **GeoPandas**.

## Getting started

### 1. Create the environment

```bash
make venv      # base environment
make install   # base + dev dependencies
```

Use `make install` if you plan to run tests or notebooks.

### 2. Preprocess data

Raw data lives under `data/<dataset>/`. Processed output mirrors that layout under `data/processed/<dataset>/`.

The preprocessing script can handle labeled or unlabeled CSVs. Negative feature values are remapped with `x -> -0.5x` before writing output.

Preprocessing is driven by YAML files under `src/geochem_detect/config/`, for example:

- `default_prep_multiclass.yml`
- `default_prep_spatial.yml`
- `bn_prep.yml`

```bash
make preprocess          # multiclass_clean.csv  (non-spatial)
make preprocess-spatial  # Data1.csv             (lat/lon retained)

# Use a custom preprocessing config
make preprocess CONFIG=src/geochem_detect/config/bn_prep.yml
```

The preprocess config controls the source path, processed output path, feature columns, optional normalization column, coordinate columns, identifier column, and optional label source column.

Processed files keep the configured coordinate column names. Training and prediction use config values or saved run metadata to resolve those columns.

### 3. Train models

Each training script reads its settings from a YAML config file. If `--config` is omitted, the bundled default is used.

Training configs should include:

- `data.data_path`
- `data.feature_columns`
- any schema columns needed at training or evaluation time, such as `data.label`, `data.latitude`, and `data.longitude`

The training entry points validate required config fields up front and fail early with a clear error if a required field is missing.

```bash
# Isolation Forest — anomaly detection on multiclass_clean.csv
make train-iforest

# Autoencoder — anomaly detection on tabular data, optionally with spatial inputs
make train-autoencoder

# CNN + Sparse Autoencoder — spatial anomaly detection on Data1.csv
make train-cnn-sae

# Multi-class MLP classifier — classification on multiclass_clean.csv
make train-classifier

# Train all four sequentially
make train-all
```

Supply a custom config to override any subset of hyperparameters:

```bash
make train-classifier CONFIG=my_configs/deep_net.yml
make train-iforest    CONFIG=my_configs/high_contamination.yml
```

If anomaly evaluation uses class labels, `evaluation.label` can override `data.label`. In the common case both should remain `label` for processed datasets.

You can also call the scripts directly:

```bash
uv run python scripts/train_isolation_forest.py --config my_configs/iforest.yml
uv run python scripts/train_autoencoder.py --config my_configs/ae.yml
uv run python scripts/train_classifier.py --config my_configs/clf.yml
```

### 4. View results in MLFlow

```bash
make mlflow-ui
# open http://localhost:5000
```

### 5. Run predictions

After training, use the printed `run_id` or find it in the MLflow UI.

```bash
# Run a saved model on individual splits
make predict-train RUN_ID=<run_id> MODEL_TYPE=classifier
make predict-val   RUN_ID=<run_id> MODEL_TYPE=classifier
make predict-test  RUN_ID=<run_id> MODEL_TYPE=classifier

# Run all saved splits
make predict-all   RUN_ID=<run_id> MODEL_TYPE=classifier

# Run on a full dataset
make predict-full  RUN_ID=<run_id> MODEL_TYPE=autoencoder
make predict-full  RUN_ID=<run_id> MODEL_TYPE=autoencoder DATA_PATH=data/gvirm/Data1.csv
```

Predictions are written to:

- `outputs/<model_type>/<run_id>/predictions/predictions_<split>.csv`

CNN-SAE uses a separate prediction flow because it works on sampled spatial windows rather than tabular rows:

```bash
make predict-cnn-sae-train RUN_ID=<run_id>
make predict-cnn-sae-val   RUN_ID=<run_id>
make predict-cnn-sae-test  RUN_ID=<run_id>
make predict-cnn-sae-all   RUN_ID=<run_id>
make predict-cnn-sae-full  RUN_ID=<run_id>           # fresh windows from source data
make predict-cnn-sae-full  RUN_ID=<run_id> DATA_PATH=data/gvirm/Data1.csv
```

CNN-SAE predictions are written to:

- `outputs/cnn_sae/<run_id>/predictions/predictions_cnn_sae_<split>.csv`

### 6. Compute Shapley attributions

After training an anomaly detector (Isolation Forest, Autoencoder, or CNN-SAE), use `compute_attribution.py` to explain which features drive each sample's anomaly score.

The script uses **SHAP KernelExplainer**, which is model-agnostic and works identically for all three methods. For CNN-SAE, window tensors are aggregated to a tabular representation before explanation.

```bash
# Run attribution for any anomaly-detection run
make attribution RUN_DIR=outputs/isolation_forest/<run_id>
make attribution RUN_DIR=outputs/autoencoder/<run_id>
make attribution RUN_DIR=outputs/cnn_sae/<run_id>

# Tune the number of background samples and cap the explained set
make attribution RUN_DIR=outputs/autoencoder/<run_id> MAX_BACKGROUND=200 MAX_SAMPLES=500

# Call the script directly for full control
uv run python scripts/compute_attribution.py outputs/isolation_forest/<run_id>
uv run python scripts/compute_attribution.py outputs/autoencoder/<run_id> \
    --max-background 200 --max-samples 500
```

Outputs are written to `outputs/<method>/<run_id>/attribution/`:

| File | Contents |
|------|----------|
| `shap_values.csv` | Per-sample SHAP values, anomaly score, and metadata |
| `shap_summary_bar.png` | Bar chart of mean absolute SHAP values per feature |
| `shap_summary_beeswarm.png` | Beeswarm plot showing feature impact distribution |

The `training_config.yml` saved in `artefacts/` is used to reconstruct the exact dataset that was seen during training, so the attribution is always consistent with the model.

### 7. Run unit tests

```bash
make test-unit
```

This runs the pytest unit suite and prints terminal coverage output.

---

## Model configuration

Each model has a bundled default YAML config under `src/geochem_detect/config/`:

| Model | Default config file |
|-------|---------------------|
| Isolation Forest | `default_config_isolation_forest.yml` |
| Autoencoder | `default_config_autoencoder.yml` |
| CNN-SAE | `default_config_cnn_sae.yml` |
| MLP Classifier | `default_config_classifier.yml` |

### Isolation Forest defaults

```yaml
data:
  data_path: gvirm/multiclass_clean.csv
  feature_columns: [SIO2(WT%), TIO2(WT%), AL2O3(WT%), FEOT(WT%), CAO(WT%), MGO(WT%), MNO(WT%), K2O(WT%), NA2O(WT%), P2O5(WT%)]
  label: label

model:
  n_estimators: 200
  contamination: 0.05
  max_features: 1.0
  random_state: 42
  n_jobs: -1

evaluation:
  label: label
  contamination_threshold: 0.05  # fraction of full dataset that defines "rare" classes
```

### Autoencoder defaults

```yaml
data:
  data_path: gvirm/Data1.csv
  feature_columns: [SiO2n, TiO2n, Al2O3n, FeO*n, MnOn, MgOn, CaOn, Na2On, K2On, P2O5n]
  longitude: long
  latitude: lat
  label: label

model:
  encoding_dim: 4
  hidden_dims: [32, 16]
  dropout_rate: 0.2
  learning_rate: 0.001
  epochs: 50
  batch_size: 64
  patience: 10

training:
  spatial: false

evaluation:
  label: label
  contamination_threshold: 0.01
```

Set `training.spatial: true` to include the configured coordinate columns as auxiliary inputs.

### CNN-SAE defaults

CNN-SAE tiles the survey area into sparse 2-D grids and learns to reconstruct typical geochemical patterns. Windows with high reconstruction error are flagged as anomalous.

```yaml
data:
  data_path: gvirm/Data1.csv
  feature_columns: [SiO2n, TiO2n, Al2O3n, FeO*n, MnOn, MgOn, CaOn, Na2On, K2On, P2O5n]
  longitude: long
  latitude: lat
  label: label

sampling:
  window_deg: 0.5    # 0.5° × 0.5° bounding box (~55 km × 39 km at 45 °N)
  grid_size: 16      # 16 × 16 cells; each cell ≈ 3.5 km × 3.5 km
  n_samples: 1000    # windows sampled (centres drawn from existing data points)
  min_points: 2      # discard windows with fewer than 2 occupied cells

model:
  cnn_filters: [32, 64]
  encoding_dim: 64
  dense_hidden_dims: [256, 128]
  dropout_rate: 0.2
  learning_rate: 0.001
  sparsity_weight: 0.0001   # L1 activity regulariser on the bottleneck
  epochs: 50
  batch_size: 32
  patience: 10

training:
  val_size: 0.15
  test_size: 0.15

evaluation:
  label: label
  contamination_threshold: 0.001
  anomaly_sigma_cutoff: 2.0
```

**Why these spatial defaults?**
Data1.csv covers 6.7 ° lat × 2.7 ° lon at about 231 points per square degree. A 1.0 ° window packs most cells in a 16 × 16 grid and weakens the sparsity signal. A 0.5 ° window keeps enough empty cells for the anomaly signal to remain useful while still giving the CNN spatial context.

### Classifier defaults

```yaml
data:
  data_path: gvirm/multiclass_clean.csv

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
│   ├── __init__.py
│   ├── default_config_isolation_forest.yml
│   ├── default_config_autoencoder.yml
│   ├── default_config_cnn_sae.yml
│   └── default_config_classifier.yml
├── data/
│   ├── loader.py
│   ├── preprocessor.py
│   └── spatial_sampler.py
├── models/
│   ├── isolation_forest.py
│   ├── autoencoder.py
│   ├── cnn_sae.py
│   └── classifier.py
├── training/
│   └── trainer.py
└── visualization/
  └── plots.py

scripts/
├── preprocess_data.py
├── train_isolation_forest.py
├── train_autoencoder.py
├── train_cnn_sae.py
├── train_classifier.py
├── predict.py
├── predict_cnn_sae.py
└── compute_attribution.py
```

### Run artefacts

Each training run saves artefacts under `outputs/<model_type>/<run_id>/artefacts/`.

| File | Contents |
|------|----------|
| `scaler.pkl` | Fitted scaler |
| `label_encoder.pkl` | Fitted `LabelEncoder` |
| `splits.npz` | `train_idx`, `val_idx`, `test_idx` |
| `dataset_info.json` | Dataset path, feature columns, label column |
| `model.pkl` or `keras_model.keras` | Saved model |
| `anomaly_threshold.json` | Saved anomaly threshold for anomaly detectors |
| `training_config.yml` | Exact merged config used during training |

CNN-SAE runs additionally save:

| File | Contents |
|------|----------|
| `sampling_params.json` | `window_deg`, `grid_size`, `n_samples`, etc. |
| `window_splits.npz` | Window-level `train_idx`, `val_idx`, `test_idx` |
| `window_metadata.json` | Centre lat/lon and point indices for every window |

Plots and prediction outputs are written under `outputs/<model_type>/<run_id>/`.

---

## Makefile reference

Run `make help` to list all targets.  Key targets:

| Target | Description |
|--------|-------------|
| `venv` | Create / sync the virtual environment |
| `install` | Install runtime and dev dependencies |
| `preprocess` | Preprocess the default non-spatial dataset |
| `preprocess-spatial` | Preprocess the default spatial dataset |
| `train-iforest` | Train Isolation Forest |
| `train-autoencoder` | Train autoencoder |
| `train-cnn-sae` | Train CNN-SAE spatial anomaly detector |
| `train-classifier` | Train MLP classifier |
| `train-all` | Train all four models |
| `predict-[train\|val\|test\|all\|full]` | Run a trained model (requires `RUN_ID=` `MODEL_TYPE=`) |
| `predict-cnn-sae-[train\|val\|test\|all\|full]` | Run CNN-SAE predictions (requires `RUN_ID=`) |
| `attribution` | Compute SHAP attributions (requires `RUN_DIR=`; optional `MAX_BACKGROUND=` `MAX_SAMPLES=`) |
| `mlflow-ui` | Launch MLFlow UI at `http://localhost:5000` |
| `lint` / `format` | Run Ruff checks or formatting |
| `test-unit` | Run unit tests with coverage output |
| `clean` | Remove outputs and caches |
| `clean-processed` | Remove processed data |

---

## Performance metric

Primary metric: **PR-AUC** (macro-averaged for the classifier; binary for anomaly detectors).
