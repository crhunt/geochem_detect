"""Train CNN-SAE spatial anomaly detector on Data1.csv.

Usage
-----
    uv run python scripts/train_cnn_sae.py [--config path/to/config.yml]

All hyperparameters are read from the config file (or the bundled default when
``--config`` is omitted).  See
``src/geochem_detect/config/default_config_cnn_sae.yml`` for the full list of
tuneable settings.

The model ingests spatially-windowed samples from Data1.csv, where each window
is a ``grid_size × grid_size`` grid of geochemical features.  Anomaly ground
truth is derived from label frequency: rock classes whose global count falls
below ``contamination_threshold × N`` are considered anomalous, and a window is
anomalous if it contains at least one anomalous source point.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler

from geochem_detect.config import load_config, model_params, sampling_params, training_params
from geochem_detect.data.loader import FEATURE_COLS_SPATIAL, load_spatial
from geochem_detect.data.preprocessor import split_features_labels
from geochem_detect.training.trainer import train_cnn_sae
from geochem_detect.visualization.plots import (
    plot_anomaly_scores_histogram,
    plot_pr_curve_binary,
    plot_spatial_anomalies,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-SAE spatial anomaly detector."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file.  Overrides the bundled default.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Display name for the MLFlow run.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        metavar="PATH",
        help="Override the default Data1.csv path.",
    )
    parser.add_argument(
        "--experiment",
        default="cnn_sae",
        metavar="NAME",
        help="MLFlow experiment name (default: cnn_sae).",
    )
    args = parser.parse_args()

    cfg = load_config("cnn_sae", args.config)
    mp = model_params(cfg)
    tp = training_params(cfg)
    sp = sampling_params(cfg)

    # ── Load and clean data ──────────────────────────────────────────────────
    gdf = load_spatial(args.data_path)
    feat_cols = FEATURE_COLS_SPATIAL

    # split_features_labels drops NaN rows and returns positional indices
    X_raw, y_raw, class_names, orig_idx = split_features_labels(gdf, feat_cols)

    # Align GeoDataFrame to the cleaned rows
    gdf_clean = gdf.dropna(subset=feat_cols).reset_index(drop=True)

    # Scale features — fit on all data (consistent with spatial windowing where
    # we cannot do a geographic train split before sampling)
    scaler = RobustScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw).astype("float32")

    # Write scaled features back into the GeoDataFrame columns so that
    # SpatialSampler reads the already-normalised values
    gdf_clean = gdf_clean.copy()
    gdf_clean[feat_cols] = X_scaled

    le = LabelEncoder()
    le.classes_ = class_names

    dataset_info = {
        "dataset": "Data1.csv",
        "feature_cols": feat_cols,
        "label_col": "label",
        "n_samples": len(X_raw),
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_cnn_sae(
        gdf_clean,
        y_raw,
        le,
        scaler,
        dataset_info,
        sampling_params=sp,
        params=params,
        experiment_name=args.experiment,
        run_name=args.run_name or "data1_cnn_sae",
    )

    # ── Post-training plots ──────────────────────────────────────────────────
    import json

    out_dir = OUTPUT_ROOT / "cnn_sae" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_cutoff   = tp.get("anomaly_sigma_cutoff", 2.0)
    contamination  = tp.get("contamination_threshold", 0.05)
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]
    # Per-point ground-truth anomaly labels (used for raw-data-point overlay)
    raw_anom_labels = np.isin(y_raw, rare).astype(np.int32)

    # Reload window splits and metadata from saved artefacts
    art_dir = out_dir / "artefacts"
    splits_npz = np.load(art_dir / "window_splits.npz")
    with open(art_dir / "window_metadata.json") as f:
        all_metadata = json.load(f)

    # Re-generate all windows from saved metadata to compute plots
    # (we need the same X tensor as used for training)
    from geochem_detect.data.spatial_sampler import SpatialSampler

    sampler = SpatialSampler(
        gdf=gdf_clean,
        feature_cols=feat_cols,
        anomaly_labels=raw_anom_labels,
        **sp,
    )
    X_all, y_all, _ = sampler.generate()

    # ── Score ALL windows at once for globally consistent normalization ───────
    # anomaly_scores() uses per-batch min-max normalization, so scoring each
    # split separately produces incomparable values (e.g. "all" can appear to
    # have fewer detections than "train" alone).  Scoring everything together
    # keeps the scale consistent across all per-split slices.
    all_scores = det.anomaly_scores(X_all)

    # ── Calibrate threshold in global score-space on the validation windows ──
    val_idx_saved = splits_npz["val_idx"]
    val_scores = all_scores[val_idx_saved]
    threshold = float(np.mean(val_scores) + sigma_cutoff * np.std(val_scores))

    # Update the saved anomaly_threshold.json with the calibrated value
    with open(art_dir / "anomaly_threshold.json") as f:
        threshold_data = json.load(f)
    threshold_data["threshold"] = threshold
    with open(art_dir / "anomaly_threshold.json", "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"  [CNN-SAE] val threshold = {threshold:.4f} "
          f"(mean={np.mean(val_scores):.4f}, sigma_cutoff={sigma_cutoff})")

    named_splits = {
        "train": splits_npz["train_idx"],
        "val":   splits_npz["val_idx"],
        "test":  splits_npz["test_idx"],
        "all":   np.concatenate([
            splits_npz["train_idx"],
            splits_npz["val_idx"],
            splits_npz["test_idx"],
        ]),
    }

    for split_name, idx in named_splits.items():
        y_anom = y_all[idx]

        # Slice pre-computed global scores so normalization is consistent
        scores = all_scores[idx]
        title_sfx = f"({split_name})"

        plot_pr_curve_binary(
            y_anom, scores,
            title=f"Precision-Recall Curve {title_sfx}",
            save_path=out_dir / f"pr_curve_cnn_sae_{split_name}.png",
        )
        plot_anomaly_scores_histogram(
            scores, sigma_cutoff=sigma_cutoff,
            title=f"Anomaly Score Distribution {title_sfx}",
            save_path=out_dir / f"scores_cnn_sae_{split_name}.png",
            threshold=threshold,
        )

        # Build a per-window GeoDataFrame for the spatial plot
        # Use the center coordinates of each window as the plot location
        import geopandas as gpd

        meta_s = [all_metadata[i] for i in idx]
        window_gdf = gpd.GeoDataFrame(
            {
                "lat":  [m["center_lat"] for m in meta_s],
                "long": [m["center_lon"] for m in meta_s],
            },
            geometry=gpd.points_from_xy(
                [m["center_lon"] for m in meta_s],
                [m["center_lat"] for m in meta_s],
            ),
            crs="EPSG:4326",
        )
        # Raw points that contributed to any window in this split,
        # colored by their ground-truth anomaly label (rare class).
        split_pt_idx = np.unique(
            np.concatenate([all_metadata[i]["point_indices"] for i in idx])
        ).astype(int)

        plot_spatial_anomalies(
            window_gdf, scores,
            threshold=threshold,
            y_true=y_anom,
            title=f"Spatial Anomaly Map {title_sfx}",
            save_path=out_dir / f"spatial_anomaly_map_cnn_sae_{split_name}.png",
            window_deg=sp["window_deg"],
            raw_gdf=gdf_clean.iloc[split_pt_idx],
            raw_y=raw_anom_labels[split_pt_idx],
        )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
