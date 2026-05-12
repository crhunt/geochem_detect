# Plan: Shapley Attribution for Anomaly Detection

## TL;DR
Add model-agnostic Shapley value computation to all 3 anomaly detectors (IsolationForest, Autoencoder, CNN-SAE) using `shap.KernelExplainer`. A single CLI script accepts any run directory, auto-detects the method, constructs a unified tabular predict function, and saves SHAP values + visualizations to `attribution/` under the run dir.

## Key Design Decisions
- **Library**: `shap` (KernelExplainer) — works with any callable predict function, no model-specific assumptions
- **Data split**: "all" (train + val + test, all labeled samples)
- **CNN-SAE aggregation**: Max of occupied cells per feature → tabular (n_windows, n_features)
- **Background dataset**: k-means(100) summary of training split tabular features via `shap.kmeans()`
- **Score function**: `model.anomaly_scores()` for autoencoder and CNN-SAE; negated `decision_function()` for IsolationForest (so higher = more anomalous consistently)
- **CNN-SAE predict wrapper**: Expands tabular (n_samples, n_features) to uniform grids (all cells filled with feature value, mask=1), passes through `model.reconstruction_errors()`

## Relevant Files
- `src/geochem_detect/models/isolation_forest.py` — `IsolationForestDetector.anomaly_scores()` + `decision_function`
- `src/geochem_detect/models/autoencoder.py` — `AutoencoderDetector.anomaly_scores()`, spatial handling
- `src/geochem_detect/models/cnn_sae.py` — `CnnSaeDetector.reconstruction_errors()`, grid shape (n_windows, 16, 16, 7)
- `src/geochem_detect/data/preprocessor.py` — scaler/encoder loading reference
- `src/geochem_detect/data/spatial_sampler.py` — `SpatialSampler` for CNN-SAE window rebuilding
- `scripts/predict.py` — reference for loading artefacts + data (tabular models)
- `scripts/predict_cnn_sae.py` — reference for loading CNN-SAE artefacts + rebuilding windows
- `scripts/compute_attribution.py` — NEW
- `src/geochem_detect/attribution/__init__.py` — NEW (empty)
- `src/geochem_detect/attribution/explainer.py` — NEW (core logic)
- `pyproject.toml` — add `shap` dependency

## Steps

### Phase 1: Dependency (1 step)
1. Add `shap` to pyproject.toml dependencies

### Phase 2: Attribution Module (parallel-ready after step 1)
2. Create `src/geochem_detect/attribution/__init__.py` (empty)
3. Create `src/geochem_detect/attribution/explainer.py` with these functions:
   - `detect_method(run_dir)` — checks artefacts: `model.pkl` → isolation_forest; `keras_model.keras` + `sampling_params.json` → cnn_sae; `keras_model.keras` alone → autoencoder
   - `load_artefacts(run_dir, method)` — loads model, scaler, label_encoder, splits.npz, dataset_info.json, anomaly_threshold.json; for CNN-SAE also loads sampling_params.json + window_splits.npz
   - `build_tabular_data(run_dir, method, artefacts)` → (X_tab_all, y_all, feature_names, sample_meta_df)
     - IsolationForest/Autoencoder: loads raw dataset CSV, applies scaler, concatenates all split indices; if spatial=True appends coord features
     - CNN-SAE: rebuilds windows for all splits using SpatialSampler; aggregates to (n_windows, n_features) via max of occupied cells; stores window centroids in sample_meta_df
   - `build_predict_fn(method, model, artefacts)` → callable (X_tab: ndarray) → 1D scores
     - IsolationForest: `lambda X: -model.model.decision_function(X)` (negate; IsolationForestDetector wraps sklearn model)
     - Autoencoder: `lambda X: model.reconstruction_errors(X)` (pass scaled X directly)
     - CNN-SAE: closure that expands each row to a uniform filled grid (mask=1 everywhere, cell values = feature values), then calls `model.reconstruction_errors()`
   - `compute_shap(predict_fn, X_explain, X_train, feature_names, max_background=100)` → shap_values ndarray (n_samples, n_features)
     - Background: `shap.kmeans(X_train, max_background)`
     - Explainer: `shap.KernelExplainer(predict_fn, background)`
     - Compute: `explainer.shap_values(X_explain)`
   - `save_attribution(run_dir, shap_values, X_explain, feature_names, sample_meta_df, predict_fn)` → saves all outputs

### Phase 3: Script
4. Create `scripts/compute_attribution.py`:
   - CLI args: `run_dir` (positional), `--max-background` (default 100), `--max-samples` (default: all, for downsampling explained set)
   - Orchestrates phases: detect → load → build_tabular_data → build_predict_fn → compute_shap → save_attribution
   - Logs progress to stdout

## Output Files (under `<run_dir>/attribution/`)
- `shap_values.csv` — columns: [index, anomaly_score, is_anomaly, shap_<feature1>, ..., shap_<featureN>]; plus lat/lon for CNN-SAE windows
- `shap_summary_plot.png` — `shap.summary_plot` beeswarm (feature values on color axis, SHAP value on x-axis)
- `shap_bar_plot.png` — mean |SHAP| per feature bar chart using matplotlib (consistent style with existing plots.py)

## Verification
1. Run `python scripts/compute_attribution.py outputs/isolation_forest/0771596ef96f444583936eb2b3d7aa2b` — check `attribution/` created with 3 files
2. Run against autoencoder run `outputs/autoencoder/a689b6bb5903416fa723c48742ea25a3` — check spatial features (NORTHING, EASTING) appear in shap_values.csv
3. Check shap_values.csv column sum ≈ (anomaly_score - expected_value) for a few rows (KernelExplainer property)
4. Run existing tests: `make test` or `pytest tests/` to ensure no regressions
5. CNN-SAE: no current output run exists to test against, but can add a mock unit test for the grid-expansion wrapper
