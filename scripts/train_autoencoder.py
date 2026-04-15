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

from geochem_detect.config import data_params, evaluation_params, load_config, model_params, training_params
from geochem_detect.data.loader import DEFAULT_SPATIAL_DATA, load_spatial_frame
from geochem_detect.data.preprocessor import make_splits, prepare_labeled_frame, scale_features
from geochem_detect.training.trainer import resolve_anomaly_ground_truth, train_autoencoder
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
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)
    ep = evaluation_params(cfg)

    # CLI spatial flag takes precedence over config
    use_spatial = tp.get("spatial", False) if args.spatial is None else args.spatial
    tp["spatial"] = use_spatial

    gdf, data_options = load_spatial_frame(data_cfg, DEFAULT_SPATIAL_DATA)
    gdf_clean, feat_cols = prepare_labeled_frame(
        gdf,
        data_options["feature_columns"],
        data_options["label_col"],
        normalize_by=data_options["normalize_by"],
    )

    X_raw = gdf_clean[feat_cols].to_numpy(dtype=np.float32)
    le = LabelEncoder()
    y_raw = le.fit_transform(gdf_clean[data_options["label_col"]].values)

    splits = make_splits(X_raw, y_raw)

    ( _, X_all_s), scaler = scale_features(
        X_raw[splits["train_idx"]],
        X_raw,
        enabled=data_options["scale_features"],
    )

    X_spatial = None
    if use_spatial:
        coords = gdf_clean[[data_options["latitude"], data_options["longitude"]]].values.astype(np.float32)
        sp_scaler = RobustScaler().fit(coords[splits["train_idx"]])
        X_spatial = sp_scaler.transform(coords).astype(np.float32)

    dataset_info = {
        "dataset": data_options["data_path"],
        "feature_cols": feat_cols,
        "label_col": data_options["label_col"],
        "n_samples": len(X_raw),
        "spatial": use_spatial,
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": data_options["normalize_by"],
        "scale_features": data_options["scale_features"],
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_autoencoder(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        X_spatial=X_spatial,
        params=params,
        evaluation=ep,
        run_name="data1_spatial" if use_spatial else "data1_chem_only",
    )

    out_dir = OUTPUT_ROOT / "autoencoder" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    y_anom_all, _, _ = resolve_anomaly_ground_truth(y_raw, le, ep)
    sigma_cutoff = ep.get("anomaly_sigma_cutoff", 2.0)
    if y_anom_all is None:
        print("  Evaluation plots skipped where ground-truth anomalies are required")

    # ── Calibrate threshold on the validation set ────────────────────────────
    # ── Score ALL points at once for globally consistent normalization ────────
    # anomaly_scores() uses per-batch min-max normalization; scoring each split
    # separately produces incomparable values.  Score everything together, then
    # calibrate the threshold from the val slice of those global scores.
    all_scores = det.anomaly_scores(X_all_s, X_spatial)
    val_idx = splits["val_idx"]
    val_scores = all_scores[val_idx]
    threshold = float(np.mean(val_scores) + sigma_cutoff * np.std(val_scores))

    # Persist the calibrated threshold so predict.py can apply the same value
    import json
    art_dir = out_dir / "artefacts"
    with open(art_dir / "anomaly_threshold.json", "w") as f:
        json.dump({"sigma_cutoff": sigma_cutoff, "threshold": threshold}, f, indent=2)
    print(f"  [Autoencoder] val threshold = {threshold:.4f} "
          f"(mean={np.mean(val_scores):.4f}, sigma_cutoff={sigma_cutoff})")

    named_splits = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
        "all":   np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]]),
    }

    for split_name, idx in named_splits.items():
        # Slice pre-computed global scores so normalization is consistent
        scores    = all_scores[idx]
        title_sfx = f"({split_name})"

        if y_anom_all is not None:
            y_anom = y_anom_all[idx]
            plot_pr_curve_binary(
                y_anom, scores,
                title=f"Precision-Recall Curve {title_sfx}",
                save_path=out_dir / f"pr_curve_autoencoder_{split_name}.png",
            )
        plot_anomaly_scores_histogram(
            scores, sigma_cutoff=sigma_cutoff,
            title=f"Anomaly Score Distribution {title_sfx}",
            save_path=out_dir / f"scores_autoencoder_{split_name}.png",
            threshold=threshold,
        )
        gdf_split = gdf_clean.iloc[idx]
        plot_spatial_anomalies(
            gdf_split, scores,
            threshold=threshold,
            y_true=y_anom_all[idx] if y_anom_all is not None else None,
            title=f"Spatial Anomaly Map {title_sfx}",
            save_path=out_dir / f"spatial_anomaly_map_{split_name}.png",
            raw_gdf=gdf_split,
            raw_y=y_anom_all[idx] if y_anom_all is not None else None,
        )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
