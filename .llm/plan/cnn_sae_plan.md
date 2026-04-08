# Plan: CNN + Sparse Autoencoder (SAE) for Spatial Anomaly Detection

## TL;DR
Add a `cnn_sae` model that ingests spatially-windowed samples from Data1.csv, represents each window as a sparse 2D grid of geochemical features, and trains a CNN encoder + sparse bottleneck + CNN decoder to detect anomalous regions via reconstruction error. Ground truth uses the same label-frequency contamination approach as the existing autoencoder. Follows all existing repo design patterns.

---

## User Decisions
- Train on **all windows** (not just normal ones) — consistent with existing autoencoder
- Window centers sampled from **existing data points** (guarantees non-empty windows)
- Inference covered by a **new `scripts/predict_cnn_sae.py`**

---

## Phase 1: Data Layer — Spatial Sampler

**1.1** Create `src/geochem_detect/data/spatial_sampler.py` — new `SpatialSampler` class:
- Constructor params: `gdf`, `feature_cols`, `anomaly_labels` (binary per point), `window_deg`, `grid_size`, `n_samples`, `min_points`, `random_state`
- `_center_to_points(lat, lon)` → filters GeoDataFrame for points within `[lat±window_deg/2, lon±window_deg/2]`
- `_points_to_grid(points_gdf, lat_c, lon_c)` → maps each point to its `(row, col)` cell in a `grid_size×grid_size` grid; multiple points per cell are averaged; returns `(grid_size, grid_size, n_features+1)` where the last channel is an **occupancy mask** (1=data present, 0=empty)
- `generate()` → returns `(X, y, metadata)`:
  - `X`: `(n_samples, grid_size, grid_size, n_features+1)` float32
  - `y`: `(n_samples,)` int (1 if ANY in-window point has `anomaly_label=1`, else 0)
  - `metadata`: list of dicts `{center_lat, center_lon, point_indices, n_points}`
- Rejects windows with fewer than `min_points` occupied cells; samples centers (with replacement) from existing data-point locations

**1.2** Anomaly ground truth for individual points:
- Computed once before calling `SpatialSampler`: same contamination-threshold logic as existing `train_autoencoder` in `trainer.py`
  - `rare = classes[counts < contamination_threshold * len(y)]`
  - `point_anomaly_labels = np.isin(y_encoded, rare).astype(int)`
- Window label = `max(point_anomaly_labels[in_window_indices])`

---

## Phase 2: Model — CNN-SAE

**2.1** Create `src/geochem_detect/models/cnn_sae.py`:

`build_cnn_sae(grid_size, n_features, cnn_filters, cnn_kernel_size, encoding_dim, dense_hidden_dims, dropout_rate, learning_rate, sparsity_weight)`:
- Input: `(grid_size, grid_size, n_features+1)` — last channel is occupancy mask
- **Encoder**:
  - `Conv2D(cnn_filters[0], kernel, padding='same', activation='relu')` × 2 strides
  - `Conv2D(cnn_filters[1], kernel, stride=2, padding='same', activation='relu')`
  - Flatten → `Dense(dense_hidden_dims[0], relu)` → `Dense(dense_hidden_dims[1], relu)`
  - Bottleneck: `Dense(encoding_dim, activation='relu', activity_regularizer=L1(sparsity_weight))` — this is the **sparse** constraint
- **Decoder**:
  - `Dense(dense_hidden_dims[1], relu)` → `Dense(dense_hidden_dims[0], relu)`
  - `Dense(H_reduced * W_reduced * cnn_filters[-1], relu)` → Reshape
  - `Conv2DTranspose(cnn_filters[0], kernel, stride=2, padding='same', relu)`
  - `Conv2D(n_features, 1, padding='same', activation='linear')` — only reconstructs feature channels (no mask channel)
- Output: `(grid_size, grid_size, n_features)` — reconstruction of features only
- **Custom masked MSE loss**: loss computed only over occupied cells (`occupancy_mask == 1`); avoids penalizing empty-cell reconstruction
- Compile with `Adam(learning_rate)` + custom loss

`CnnSaeDetector` class (parallel to `AutoencoderDetector`):
- Constructor captures all hyperparams + calls `build_cnn_sae()`
- `fit(X, X_val=None)` → EarlyStopping on `val_loss`
- `reconstruction_errors(X)` → per-sample masked MSE (mean over occupied cells)
- `anomaly_scores(X)` → normalised [0,1]
- `pr_auc(X, y_true)` → `average_precision_score`
- `is_anomaly(X, sigma_cutoff)` → same sigma-threshold logic as `AutoencoderDetector`
- `self.params` dict for `mlflow.log_param()`

---

## Phase 3: Config

**3.1** Create `src/geochem_detect/config/default_config_cnn_sae.yml`:
```yaml
sampling:
  window_deg: 1.0
  grid_size: 16
  n_samples: 2000
  min_points: 2
  random_state: 42

model:
  cnn_filters: [32, 64]
  cnn_kernel_size: 3
  encoding_dim: 64
  dense_hidden_dims: [256, 128]
  dropout_rate: 0.2
  learning_rate: 0.001
  sparsity_weight: 0.0001
  epochs: 50
  batch_size: 32
  patience: 10

training:
  contamination_threshold: 0.05
  anomaly_sigma_cutoff: 2.0
  val_size: 0.15
  test_size: 0.15
```

**3.2** Modify `src/geochem_detect/config/__init__.py`:
- Add `"cnn_sae": _CONFIG_DIR / "default_config_cnn_sae.yml"` to `_DEFAULT_CONFIGS`
- Add `"sampling"` to the list of deep-merged sections (alongside `"model"` and `"training"`)
- Export `sampling_params(cfg)` helper (mirrors `model_params`, `training_params`)

---

## Phase 4: Trainer

**4.1** Add `train_cnn_sae()` to `src/geochem_detect/training/trainer.py`:

```
train_cnn_sae(gdf, y_encoded, label_encoder, dataset_info, sampling_params, params, experiment_name, run_name)
```
Steps inside:
1. Compute per-point anomaly labels (contamination_threshold)
2. Instantiate `SpatialSampler`, call `.generate()` → `X, y_windows, metadata`
3. Log class balance of windows (# anomalous / total)
4. Random train/val/test split of windows (sklearn `train_test_split`, stratified on `y_windows`)
5. MLFlow run: log all params, train `CnnSaeDetector`, log `pr_auc` + `epochs_run` + split sizes
6. `_save_run_artefacts()` — save scaler (None, features already normalised in grid), label_encoder, sample metadata (as JSON), anomaly threshold
7. Save Keras model as `keras_model.keras` + log to MLFlow via `mlflow.tensorflow.log_model`

Note: Features in the grid are **pre-scaled with RobustScaler** before gridding (fit on all in-window points from train windows); scaler saved to artefacts.

---

## Phase 5: Training Script

**5.1** Create `scripts/train_cnn_sae.py` (follows `scripts/train_autoencoder.py` pattern exactly):
- CLI args: `--config`, `--run-name`, `--data-path`, `--experiment`
- Env var: `MLFLOW_TRACKING_URI` (default `./mlruns`)
- Loads config via `load_config("cnn_sae", config_path)`
- Loads `load_spatial()` GeoDataFrame
- Extracts features, encodes labels: `split_features_labels(gdf, FEATURE_COLS_SPATIAL)`
- Scales features with `scale_features()` and stores scaler
- Calls `train_cnn_sae()`
- Generates plots: PR curve, anomaly score histogram, anomaly spatial map (using existing `plot_pr_curve`, `plot_anomaly_score_histogram`, `plot_anomaly_map` from `visualization/plots.py`)
- Saves plots to `outputs/<run_id>/artefacts/`

---

## Phase 6: Prediction Script

**6.1** Create `scripts/predict_cnn_sae.py`:
- CLI: `--run-id`, `--data-path`, `--output-dir`, `--split {train,val,test,all}`
- Loads `keras_model.keras` + scaler + label_encoder + sample metadata JSON from `outputs/<run_id>/artefacts/`
- Re-creates the relevant window samples using saved metadata (center coords + window_deg from artefacts)
- Runs `CnnSaeDetector.reconstruction_errors()` + `anomaly_scores()` + `is_anomaly()`
- Outputs CSV: `sample_idx, center_lat, center_lon, n_points, anomaly_score, is_anomaly, true_label`
- Saves to `outputs/<run_id>/predictions/`

---

## Phase 7: Makefile

**7.1** Add `train-cnn-sae` target to `Makefile` (mirrors `train-autoencoder` target):
```makefile
train-cnn-sae:
    python scripts/train_cnn_sae.py
```

---

## Relevant Files

### New
- `src/geochem_detect/data/spatial_sampler.py` — SpatialSampler class
- `src/geochem_detect/models/cnn_sae.py` — CnnSaeDetector + build_cnn_sae()
- `src/geochem_detect/config/default_config_cnn_sae.yml` — YAML defaults
- `scripts/train_cnn_sae.py` — CLI training script
- `scripts/predict_cnn_sae.py` — CLI inference script

### Modified
- `src/geochem_detect/config/__init__.py` — register "cnn_sae", add sampling_params()
- `src/geochem_detect/training/trainer.py` — add train_cnn_sae()
- `Makefile` — add train-cnn-sae target

### Reference (do not modify)
- `src/geochem_detect/models/autoencoder.py` — template for CnnSaeDetector interface
- `src/geochem_detect/training/trainer.py` (train_autoencoder) — template for trainer function
- `scripts/train_autoencoder.py` — template for training script
- `src/geochem_detect/data/loader.py` — FEATURE_COLS_SPATIAL, LON_COL, LAT_COL

---

## Verification

1. Run `python scripts/preprocess_data.py` (ensure Data1.csv processed data is present)
2. Run `python scripts/train_cnn_sae.py` with default config — confirm MLFlow run created, PR-AUC metric logged
3. Inspect `outputs/<run_id>/artefacts/` — verify keras_model.keras, scaler.pkl, metadata JSON exist
4. Run `python scripts/predict_cnn_sae.py --run-id <run_id> --split test` — confirm CSV output
5. Check MLFlow UI (`make mlflow-ui`) to verify params/metrics appear for the `cnn_sae` experiment
6. Confirm PR-AUC > 0 (model training completes without error)

---

## Decisions
- **Masked reconstruction loss**: MSE computed only over occupied grid cells (occupancy_mask channel)
- **Feature scaling**: RobustScaler fit on all points from training-set windows before gridding
- **Window label**: window is anomalous if ANY contained point has a rare-class label (contamination_threshold × total_count)
- **Center selection**: random sample from existing data-point locations (with replacement)
- **No geographic train/val/test split**: windows are split randomly (consistent with existing approach)
- **Occupancy mask**: appended as last channel of input tensor; only feature channels reconstructed in output
- **Sparsity mechanism**: L1 activity regularization on bottleneck Dense layer (standard SAE approach)

## Further Considerations
1. **Grid projection**: Window bins are in degrees (not equal-area). For moderate-lat data this is acceptable. A future refinement could project to UTM before gridding.
2. **Multiple points per cell**: Averaged if multiple source points fall in the same cell. Alternative: take max (to preserve anomalous signal). Worth noting in implementation.
3. **Window overlap leakage**: Since centers are sampled from data points and windows are large, the same source point can appear in both train and test windows. This is accepted for now (consistent with spirit of the existing autoencoder approach).
