"""Train spatial autoencoder on Data1.csv.

Usage:
    uv run python scripts/train_autoencoder.py [--config path/to/config.yml]

All hyperparameters are read from the config file (or from the bundled default
when --config is omitted).  See src/geochem_detect/config/default_config_autoencoder.yml
for the full list of tuneable settings.  The ``training.spatial`` flag can also
be overridden on the command line with ``--spatial`` / ``--no-spatial``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

from geochem_detect.config import load_config, model_params, training_params
from geochem_detect.data.loader import feature_columns, load_spatial
from geochem_detect.data.preprocessor import make_splits, split_features_labels
from geochem_detect.training.trainer import train_autoencoder
from geochem_detect.visualization.plots import (
    plot_anomaly_scores_histogram,
    plot_pr_curve_binary,
    plot_spatial_anomalies,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train spatial autoencoder anomaly detector."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file.  Overrides the bundled default.",
    )
    # Allow CLI to flip the spatial flag without needing a full custom config
    spatial_group = parser.add_mutually_exclusive_group()
    spatial_group.add_argument(
        "--spatial", dest="spatial", action="store_true", default=None,
        help="Include lat/lon features (overrides config).",
    )
    spatial_group.add_argument(
        "--no-spatial", dest="spatial", action="store_false",
        help="Disable lat/lon features (overrides config).",
    )
    args = parser.parse_args()

    cfg = load_config("autoencoder", args.config)
    mp = model_params(cfg)
    tp = training_params(cfg)

    # CLI spatial flag takes precedence over config
    use_spatial = tp.get("spatial", False) if args.spatial is None else args.spatial
    tp["spatial"] = use_spatial

    gdf = load_spatial()
    feat_cols = feature_columns("spatial")
    X_raw, y_raw, class_names, orig_idx = split_features_labels(gdf, feat_cols)

    splits = make_splits(X_raw, y_raw, orig_idx)

    scaler = RobustScaler().fit(X_raw[splits["train_idx"]])
    X_all_s = scaler.transform(X_raw).astype("float32")

    le = LabelEncoder()
    le.classes_ = class_names

    X_spatial = None
    if use_spatial:
        gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)
        coords = gdf_clean[["lat", "long"]].values.astype(np.float32)
        sp_scaler = RobustScaler().fit(coords[splits["train_idx"]])
        X_spatial = sp_scaler.transform(coords).astype(np.float32)

    dataset_info = {
        "dataset": "Data1.csv",
        "feature_cols": feat_cols,
        "label_col": "label",
        "n_samples": len(X_raw),
        "spatial": use_spatial,
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_autoencoder(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        X_spatial=X_spatial,
        params=params,
        run_name="data1_spatial" if use_spatial else "data1_chem_only",
    )

    out_dir = OUTPUT_ROOT / "autoencoder" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_cutoff = tp.get("anomaly_sigma_cutoff", 2.0)
    contamination = tp.get("contamination_threshold", 0.05)
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]

    gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)

    named_splits = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
        "all":   np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]]),
    }

    for split_name, idx in named_splits.items():
        X_s   = X_all_s[idx]
        X_sp_s = X_spatial[idx] if X_spatial is not None else None
        y_anom = np.isin(y_raw[idx], rare).astype(int)

        scores    = det.anomaly_scores(X_s, X_sp_s)
        threshold = float(np.mean(scores) + sigma_cutoff * np.std(scores))
        title_sfx = f"({split_name})"

        plot_pr_curve_binary(
            y_anom, scores,
            title=f"Precision-Recall Curve {title_sfx}",
            save_path=out_dir / f"pr_curve_autoencoder_{split_name}.png",
        )
        plot_anomaly_scores_histogram(
            scores, sigma_cutoff=sigma_cutoff,
            title=f"Anomaly Score Distribution {title_sfx}",
            save_path=out_dir / f"scores_autoencoder_{split_name}.png",
        )
        gdf_split = gdf_clean.iloc[idx]
        plot_spatial_anomalies(
            gdf_split, scores,
            threshold=threshold,
            y_true=y_anom,
            title=f"Spatial Anomaly Map {title_sfx}",
            save_path=out_dir / f"spatial_anomaly_map_{split_name}.png",
            raw_gdf=gdf_split,
            raw_y=y_anom,
        )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
