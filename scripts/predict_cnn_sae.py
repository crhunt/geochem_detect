"""Run a trained CNN-SAE model against data splits or a full dataset.

Artefacts (scaler, label encoder, window metadata, sampling params) must have
been saved by ``scripts/train_cnn_sae.py``.  They live under:
    outputs/<run_id>/artefacts/

Usage examples
--------------
# Run against the test split used during training:
uv run python scripts/predict_cnn_sae.py --run-id <run_id> --split test

# Run against all splits individually:
uv run python scripts/predict_cnn_sae.py --run-id <run_id> --split all

# Run against a different CSV (generates new windows using saved sampling params):
uv run python scripts/predict_cnn_sae.py --run-id <run_id> --split full \\
    --data-path /path/to/other.csv

Splits:
  train | val | test   — windows used during the corresponding training phase
  all                  — train, val, and test individually
  full                 — generate a fresh set of windows from the source data
                         (uses saved sampling_params.json and contamination config)

Output CSV columns:
  window_idx, center_lat, center_lon, n_points,
  anomaly_score, is_anomaly, true_label
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_artefacts(run_id: str) -> dict:
    art = OUTPUT_ROOT / "cnn_sae" / run_id / "artefacts"
    if not art.exists():
        raise FileNotFoundError(
            f"Artefacts not found for run_id={run_id} at {art}.\n"
            "Make sure the model was trained with scripts/train_cnn_sae.py."
        )
    scaler = pickle.loads((art / "scaler.pkl").read_bytes())
    le     = pickle.loads((art / "label_encoder.pkl").read_bytes())
    with open(art / "dataset_info.json") as f:
        info = json.load(f)
    with open(art / "anomaly_threshold.json") as f:
        threshold_cfg = json.load(f)
    with open(art / "sampling_params.json") as f:
        sp = json.load(f)
    with open(art / "window_metadata.json") as f:
        metadata = json.load(f)
    splits = np.load(art / "window_splits.npz")
    return dict(
        scaler=scaler, le=le, info=info,
        threshold_cfg=threshold_cfg, sampling_params=sp,
        metadata=metadata, splits=splits, art_dir=art,
    )


def _load_keras_model(run_id: str):
    keras_path = OUTPUT_ROOT / "cnn_sae" / run_id / "artefacts" / "keras_model.keras"
    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found at {keras_path}")
    import tensorflow as tf
    return tf.keras.models.load_model(str(keras_path), compile=False)


def _load_source_data(info: dict, data_path: str | None):
    """Load and clean the source GeoDataFrame, returning scaled features."""
    from geochem_detect.data.loader import load_spatial
    from geochem_detect.data.preprocessor import split_features_labels

    gdf = load_spatial(data_path)
    feat_cols: list[str] = info["feature_cols"]
    X_raw, y_raw, class_names, _ = split_features_labels(gdf, feat_cols)
    gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)
    return gdf_clean, X_raw, y_raw, class_names, feat_cols


def _rebuild_windows(
    gdf_clean,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler,
    feat_cols: list[str],
    contamination: float,
    sampling_params: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Re-generate windows using the saved sampling params."""
    from geochem_detect.data.spatial_sampler import SpatialSampler

    X_scaled = scaler.transform(X_raw).astype("float32")
    gdf_scaled = gdf_clean.copy()
    gdf_scaled[feat_cols] = X_scaled

    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]
    point_anomaly = np.isin(y_raw, rare).astype(np.int32)

    sampler = SpatialSampler(
        gdf=gdf_scaled,
        feature_cols=feat_cols,
        anomaly_labels=point_anomaly,
        **sampling_params,
    )
    return sampler.generate()


def _reconstruct_windows_from_metadata(
    metadata: list[dict],
    idx: np.ndarray,
    gdf_clean,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler,
    feat_cols: list[str],
    contamination: float,
    sampling_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-build window tensors for specific indices using saved metadata."""
    from geochem_detect.data.spatial_sampler import SpatialSampler

    X_scaled = scaler.transform(X_raw).astype("float32")
    gdf_scaled = gdf_clean.copy()
    gdf_scaled[feat_cols] = X_scaled

    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]
    point_anomaly = np.isin(y_raw, rare).astype(np.int32)

    # Instantiate sampler just to access _points_to_grid helper
    grid_size = sampling_params.get("grid_size", 16)
    window_deg = sampling_params.get("window_deg", 1.0)
    sampler = SpatialSampler(
        gdf=gdf_scaled,
        feature_cols=feat_cols,
        anomaly_labels=point_anomaly,
        window_deg=window_deg,
        grid_size=grid_size,
        n_samples=1,  # not used directly
    )

    grids = []
    labels = []
    for i in idx:
        m = metadata[i]
        pt_idx = np.array(m["point_indices"], dtype=int)
        grid = sampler._points_to_grid(pt_idx, m["center_lat"], m["center_lon"])
        label = int(point_anomaly[pt_idx].max()) if len(pt_idx) > 0 else 0
        grids.append(grid)
        labels.append(label)

    X_windows = np.stack(grids, axis=0)
    y_windows = np.array(labels, dtype=np.int32)
    return X_windows, y_windows


def _predict_and_save(
    model,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    metadata: list[dict],
    n_features: int,
    sigma_cutoff: float,
    split_name: str,
    run_id: str,
    output_dir: Path,
) -> None:
    """Score windows, build result DataFrame, and write CSV."""
    # Wrap bare Keras model in a lightweight scorer rather than re-instantiating
    # the full detector (avoids needing all constructor params at predict time)
    class _Scorer:
        def __init__(self, keras_model, n_feat: int) -> None:
            self.model = keras_model
            self.n_features = n_feat

        def reconstruction_errors(self, X_in: np.ndarray) -> np.ndarray:
            X_feat = X_in[:, :, :, : self.n_features]
            occ    = X_in[:, :, :, self.n_features]
            preds  = self.model.predict(X_in, verbose=0)
            per_cell = np.mean((X_feat - preds) ** 2, axis=-1)
            masked   = per_cell * occ
            n_occ    = np.maximum(occ.sum(axis=(1, 2)), 1.0)
            return (masked.sum(axis=(1, 2)) / n_occ).astype(np.float32)

        def anomaly_scores(self, X_in: np.ndarray) -> np.ndarray:
            errors = self.reconstruction_errors(X_in)
            mn, mx = errors.min(), errors.max()
            return (errors - mn) / (mx - mn) if mx > mn else np.zeros_like(errors)

    scorer = _Scorer(model, n_features)
    scores = scorer.anomaly_scores(X)
    threshold = float(np.mean(scores) + sigma_cutoff * np.std(scores))
    is_anom = (scores >= threshold).astype(int)

    rows = []
    for rank, (win_idx, score, flag, label) in enumerate(
        zip(idx, scores, is_anom, y)
    ):
        m = metadata[int(win_idx)]
        rows.append({
            "window_idx":    int(win_idx),
            "center_lat":    m["center_lat"],
            "center_lon":    m["center_lon"],
            "n_points":      m["n_points"],
            "anomaly_score": float(score),
            "is_anomaly":    int(flag),
            "true_label":    int(label),
        })

    df = pd.DataFrame(rows)
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"predictions_cnn_sae_{split_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  [{split_name}] {len(df)} windows → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained CNN-SAE model."
    )
    parser.add_argument("--run-id",    required=True, metavar="ID",
                        help="MLFlow run ID of the trained CNN-SAE.")
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test", "all", "full"],
                        help="Which data split to evaluate.")
    parser.add_argument("--data-path", default=None, metavar="PATH",
                        help="Override the source CSV path.")
    parser.add_argument("--output-dir", default=None, metavar="DIR",
                        help="Root directory for output files "
                             "(default: outputs/<run_id>/).")
    args = parser.parse_args()

    art = _load_artefacts(args.run_id)
    model = _load_keras_model(args.run_id)
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_ROOT / "cnn_sae" / args.run_id

    info          = art["info"]
    scaler        = art["scaler"]
    sp            = art["sampling_params"]
    metadata      = art["metadata"]
    splits        = art["splits"]
    threshold_cfg = art["threshold_cfg"]
    sigma_cutoff  = threshold_cfg.get("sigma_cutoff", 2.0)
    contamination = threshold_cfg.get("contamination_threshold", 0.05)
    feat_cols      = info["feature_cols"]
    n_features     = len(feat_cols)

    gdf_clean, X_raw, y_raw, class_names, _ = _load_source_data(info, args.data_path)

    if args.split == "full":
        print("Generating new windows from source data…")
        X_all, y_all, new_meta = _rebuild_windows(
            gdf_clean, X_raw, y_raw, scaler, feat_cols, contamination, sp
        )
        all_idx = np.arange(len(X_all))
        # Use new metadata for full split
        _predict_and_save(
            model, X_all, y_all, all_idx, new_meta,
            n_features, sigma_cutoff, "full", args.run_id, out_dir,
        )
        return

    all_split_indices = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
    }
    if args.split == "all":
        targets = all_split_indices
    else:
        targets = {args.split: all_split_indices[args.split]}

    for split_name, idx in targets.items():
        print(f"Reconstructing {len(idx)} windows for '{split_name}' split…")
        X_s, y_s = _reconstruct_windows_from_metadata(
            metadata, idx, gdf_clean, X_raw, y_raw,
            scaler, feat_cols, contamination, sp,
        )
        _predict_and_save(
            model, X_s, y_s, idx, metadata,
            n_features, sigma_cutoff, split_name, args.run_id, out_dir,
        )


if __name__ == "__main__":
    main()
