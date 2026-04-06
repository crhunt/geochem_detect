# Plan: geochem_detect — Geochemical Anomaly Detection & Classification Package

## Status: Draft (pending user approval)

## Data Summary
- `multiclass_clean.csv`: 2,198 rows, 10 geochemical features, 7 clean classes (BASALT, KOMATIITE, THOLEIITE, RHYOLITE, ANDESITE, GABBRO, DACITE), no coordinates
- `Data1.csv`: ~4,100 rows, 10 geochemical features + long/lat, 100+ noisy label variants, has blank separator rows, `<0.05` literal, trailing whitespace in labels

## User Decisions
- Anomaly detection: unsupervised, score-based thresholding (no ground-truth anomaly labels)
- Classifier labels: configurable taxonomy (base_class + emplacement_type)
- Spatial autoencoder: Data1.csv only (has coords); Isolation Forest only for multiclass_clean
- Package name: geochem_detect

---

## Project Structure
```
geochem_detect/              # repo root
├── .llm/
│   └── plan.md
├── data/
│   └── gvirm/
├── pyproject.toml           # uv project config
├── uv.lock
├── README.md
├── src/
│   └── geochem_detect/
│       ├── __init__.py
│       ├── config.py              # Pydantic settings (paths, hyperparams)
│       ├── data/
│       │   ├── loaders.py         # load_tabular(), load_spatial()
│       │   ├── preprocessors.py   # impute, scale, clean labels
│       │   ├── taxonomy.py        # RockLabel dataclass + TAXONOMY_MAP
│       │   └── samplers.py        # spatial neighborhood + grid samplers
│       ├── models/
│       │   ├── anomaly/
│       │   │   ├── isolation_forest.py    # sklearn IF + MLFlow logging
│       │   │   └── spatial_autoencoder.py # Keras encoder-decoder
│       │   └── classifier/
│       │       └── rare_class_net.py      # Keras focal-loss multi-class net
│       ├── training/
│       │   ├── trainer.py          # MLFlowTrainer base class
│       │   └── callbacks.py        # Keras/MLFlow sync callbacks
│       ├── evaluation/
│       │   ├── metrics.py          # PR-AUC, ROC-AUC, F1, soft ground truth
│       │   └── plots.py            # PR curves, spatial maps, confusion matrix, latent space
│       └── pipelines/
│           ├── anomaly_pipeline.py        # CLI: geochem-train-anomaly
│           └── classification_pipeline.py # CLI: geochem-train-classifier
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_anomaly_detection.ipynb
│   └── 03_classification.ipynb
├── scripts/
│   ├── train_anomaly.py
│   └── train_classifier.py
└── tests/
    ├── conftest.py
    ├── test_loaders.py
    ├── test_preprocessors.py
    ├── test_taxonomy.py
    ├── test_isolation_forest.py
    ├── test_spatial_autoencoder.py
    └── test_classifier.py
```

## Dependencies (pyproject.toml)
Core: geopandas>=0.14, pandas>=2.0, numpy>=1.26, scikit-learn>=1.4, tensorflow>=2.16, keras>=3.0, mlflow>=2.12, matplotlib>=3.8, seaborn>=0.13, imbalanced-learn>=0.12, pydantic>=2.6, pydantic-settings>=2.2, contextily>=1.6, scipy>=1.12, shapely>=2.0
Dev: pytest>=8.0, pytest-cov, jupyter, ipykernel, umap-learn

## Implementation Phases

### Phase 1: Project Scaffold & Data Pipeline
1. Init uv project: `uv init geochem_detect --lib`, configure pyproject.toml
2. Implement `loaders.py`: load_tabular() (multiclass_clean), load_spatial() (Data1, returns GeoDataFrame)
3. Implement `preprocessors.py`: strip whitespace, handle `<0.05` → 0.025, drop blank rows, median imputation per class, StandardScaler
4. Implement `taxonomy.py`: RockLabel(base_class, emplacement, uncertain), TAXONOMY_MAP dict covering all 100+ variants, YAML override support
5. Implement `samplers.py`: SpatialGridSampler (H3 or equal-area grid tiles), KNNNeighborhoodSampler (BallTree, K=8)
6. Write tests for loaders, preprocessors, taxonomy

### Phase 2: Anomaly Detection
7. `isolation_forest.py`: IsolationForestDetector wrapping sklearn IF; fit(), score(), predict(); MLFlow param/metric logging; contamination configurable
8. `spatial_autoencoder.py`: SpatialAutoencoder Keras model; input = K-neighbor feature matrix (10*K features); Encoder Dense[128→64→32→latent_dim]; Decoder Dense[32→64→10]; reconstruction MSE = anomaly score; KNN aggregation in samplers.py
9. `trainer.py`: MLFlowTrainer.fit() handles run context, logs params, logs model artifact; subclasses for IF and AE
10. `callbacks.py`: MLFlowMetricsCallback logs epoch loss to MLFlow
11. Soft ground truth for anomaly evaluation: `metrics.py#geochemical_soft_labels()` — flags samples >2 SD from class-conditional median as "anomalous" (used only for PR-AUC benchmarking, not training)
12. Write tests for IF, AE, trainer

### Phase 3: Rare-Class Classifier
13. `rare_class_net.py`: Keras multi-class net; Dense[128] → BatchNorm → Dropout(0.3) → Dense[64] → softmax(n_classes); focal loss (gamma=2, configurable) or class_weight via sklearn; input = normalized 10-feature vector
14. Label encoding: LabelEncoder on base_class; stratified train/val/test 70/15/15 split respecting rare classes
15. Update `trainer.py` with ClassifierTrainer subclass; log per-epoch val PR-AUC via custom Keras callback
16. Write tests for classifier

### Phase 4: Evaluation & Visualization
17. `metrics.py` functions: compute_prauc(y_true, y_scores, average), compute_rocauc(), compute_f1(), precision_at_k(), anomaly_pr_curve()
18. `plots.py` functions: pr_curve_plot() — per-class + macro curve; roc_curve_plot(); confusion_matrix_plot() — normalized; anomaly_score_histogram(); spatial_anomaly_map() — GeoPandas choropleth + contextily basemap; reconstruction_error_map(); latent_space_plot() — optional UMAP
19. All plots save as MLFlow artifacts via trainer

### Phase 5: Pipelines & Notebooks
20. `anomaly_pipeline.py`: CLI main() parses args (dataset path, model type, config YAML), runs full pipeline (load → preprocess → train → evaluate → log)
21. `classification_pipeline.py`: same pattern for classifier
22. Notebooks: 01_eda (GeoPandas maps, class distributions, missingness heatmap), 02_anomaly (IF + AE results, spatial maps), 03_classification (PR curves, confusion matrix, rare class analysis)

## Key Architecture Details

### Spatial Autoencoder Neighborhood Sampling
- For each sample in Data1, find K=8 spatial neighbors using BallTree on (long, lat)
- Concatenate neighbor feature vectors → input shape (K * n_features,) = (80,)
- Autoencoder learns typical spatial co-occurrence; high MSE = spatial anomaly
- Scalability: precompute neighbor indices once; use tf.data.Dataset for batching

### Soft Ground Truth for Unsupervised PR-AUC
- Compute per-class, per-feature mean and std from training data
- Label a sample as anomalous if any feature > mean ± 2*std for its class
- Used ONLY as a proxy evaluation label; never seen by models during training
- Threshold configurable via config.py

### Taxonomy Mapping
```python
@dataclass
class RockLabel:
    base_class: str           # e.g. "Basalt"
    emplacement: str | None   # e.g. "sill", "dike", None for plain
    uncertain: bool           # True if label ends with '?'
```
TAXONOMY_MAP covers: all whitespace variants, sill/dike/plug/intrusion/stock qualifiers, tuff/breccia/pyroclastic, altered types, compound "A/B" types (map to primary), "?" variants

### Scalability Approach
- load_tabular() supports chunksize parameter for chunked reading
- score_samples() in IsolationForestDetector processes in configurable batch sizes
- Keras models use tf.data.Dataset pipeline with prefetch(AUTOTUNE)
- GeoPandas sindex (R-tree) for efficient spatial queries on large datasets

## Verification Steps
1. `pytest tests/` — all unit tests pass
2. `uv run geochem-train-anomaly --dataset multiclass_clean --model isolation_forest` — IF trains, MLFlow run logged
3. `uv run geochem-train-anomaly --dataset data1 --model spatial_autoencoder` — AE trains, spatial anomaly map artifact in MLFlow
4. `uv run geochem-train-classifier --dataset multiclass_clean` — classifier trains, PR-AUC logged per class
5. `mlflow ui` — inspect all experiments, compare runs, view artifacts
6. Notebook 01_eda runs end-to-end, produces GeoPandas spatial map of Data1 samples
7. PR-AUC for classifier baseline (majority class) is ~class_frequency; trained model should significantly exceed baseline
8. Reconstruction error map for spatial AE shows higher scores in geochemically extreme sample locations

## Scope Exclusions
- No real-time/streaming inference
- No REST API / serving layer
- No GPU-specific optimization (TF auto-detects)
- UMAP latent visualization is optional dev dependency only
- No Dask in initial version (chunked pandas is sufficient for ~5K rows)
