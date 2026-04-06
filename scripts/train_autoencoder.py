"""Train spatial autoencoder on Data1.csv.

Usage:
    uv run python scripts/train_autoencoder.py [--epochs 50] [--spatial]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

from geochem_detect.data.loader import feature_columns, load_spatial
from geochem_detect.data.preprocessor import make_splits, scale_features, split_features_labels
from geochem_detect.training.trainer import train_autoencoder
from geochem_detect.visualization.plots import (
    plot_anomaly_scores_histogram,
    plot_pr_curve_binary,
    plot_spatial_anomalies,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--encoding_dim", type=int, default=4)
    parser.add_argument("--spatial", action="store_true", help="Include lat/lon features")
    parser.add_argument("--contamination", type=float, default=0.05)
    args = parser.parse_args()

    gdf = load_spatial()
    feat_cols = feature_columns("spatial")
    X_raw, y_raw, class_names, orig_idx = split_features_labels(gdf, feat_cols)

    splits = make_splits(X_raw, y_raw, orig_idx)

    # Fit scaler on training rows only; apply to full array
    scaler = RobustScaler().fit(X_raw[splits["train_idx"]])
    X_all_s = scaler.transform(X_raw).astype("float32")

    le = LabelEncoder()
    le.classes_ = class_names

    # Spatial features (fit scaler on train rows)
    X_spatial = None
    if args.spatial:
        gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)
        coords = gdf_clean[["lat", "long"]].values.astype(np.float32)
        sp_scaler = RobustScaler().fit(coords[splits["train_idx"]])
        X_spatial = sp_scaler.transform(coords).astype(np.float32)

    dataset_info = {
        "dataset": "Data1.csv",
        "feature_cols": feat_cols,
        "label_col": "label",
        "n_samples": len(X_raw),
        "spatial": args.spatial,
    }

    params = dict(
        epochs=args.epochs,
        encoding_dim=args.encoding_dim,
        contamination_threshold=args.contamination,
    )
    det, pr_auc, run_id = train_autoencoder(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        X_spatial=X_spatial,
        params=params,
        run_name="data1_spatial" if args.spatial else "data1_chem_only",
    )

    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    te = splits["test_idx"]
    X_te = X_all_s[te]
    X_sp_te = X_spatial[te] if X_spatial is not None else None
    y_test = y_raw[te]
    # Mirror the rare-class definition used in trainer: based on full y distribution
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(args.contamination * len(y_raw))]
    y_anomaly = np.isin(y_raw[splits["test_idx"]], rare).astype(int)

    scores = det.anomaly_scores(X_te, X_sp_te)
    plot_pr_curve_binary(y_anomaly, scores, save_path=out_dir / "pr_curve_autoencoder.png")
    plot_anomaly_scores_histogram(scores, save_path=out_dir / "scores_autoencoder.png")

    gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)
    plot_spatial_anomalies(gdf_clean.iloc[te], scores, save_path=out_dir / "spatial_anomaly_map.png")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
