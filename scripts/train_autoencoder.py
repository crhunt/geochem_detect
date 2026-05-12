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

import geopandas as gpd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from geochem_detect.config import (
    data_params,
    evaluation_params,
    load_config,
    model_params,
    training_params,
    validate_training_config,
)
from geochem_detect.data.loader import DEFAULT_SPATIAL_DATA, load_dataset_frame
from geochem_detect.data.preprocessor import (
    IdentityScaler,
    make_splits,
)
from geochem_detect.training.trainer import resolve_anomaly_ground_truth, train_autoencoder
from geochem_detect.visualization.plots import (
    plot_anomaly_scores_histogram,
    plot_pr_curve_binary,
    plot_spatial_anomalies,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def _evaluation_requires_labels(evaluation: dict | None) -> bool:
    evaluation = dict(evaluation or {})
    return any(
        evaluation.get(key) is not None
        for key in ("anomaly_labels", "contamination_threshold")
    )


def _evaluation_label_col(evaluation: dict | None, data_cfg: dict | None) -> str:
    evaluation = dict(evaluation or {})
    data_cfg = dict(data_cfg or {})
    return str(evaluation.get("label") or data_cfg.get("label") or "label")


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
    validate_training_config("autoencoder", cfg)
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)
    ep = evaluation_params(cfg)
    labels_required = _evaluation_requires_labels(ep)
    label_col = _evaluation_label_col(ep, data_cfg)

    # CLI spatial flag takes precedence over config
    use_spatial = tp.get("spatial", False) if args.spatial is None else args.spatial
    tp["spatial"] = use_spatial

    df, data_options = load_dataset_frame(
        data_cfg,
        DEFAULT_SPATIAL_DATA,
        require_label=labels_required,
        label_col=label_col,
    )
    feat_cols = data_options["feature_columns"]

    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    label_available = data_options["label_col"] in df.columns
    if labels_required and not label_available:
        raise ValueError(
            f"Evaluation requires label column '{data_options['label_col']}' in the processed dataset. "
            "Re-run preprocessing with a label source column or remove evaluation settings."
        )

    if label_available:
        le = LabelEncoder()
        y_raw = le.fit_transform(df[data_options["label_col"]].values)
    else:
        le = LabelEncoder()
        y_raw = np.zeros(len(df), dtype=np.int32)
        le.fit(np.array(["data"], dtype=object))

    splits = make_splits(X_all, y_raw)
    scaler = IdentityScaler().fit(X_all[splits["train_idx"]])

    X_spatial = None
    coord_cols_available = (
        data_options["longitude"] is not None
        and data_options["latitude"] is not None
        and data_options["longitude"] in df.columns
        and data_options["latitude"] in df.columns
    )
    if use_spatial:
        if not coord_cols_available:
            raise SystemExit(
                "ERROR: training.spatial requires coordinate columns. "
                "Set data.longitude and data.latitude in the config or disable spatial features."
            )
        coords = df[
            [data_options["latitude"], data_options["longitude"]]
        ].values.astype(np.float32)
        X_spatial = coords

    points_gdf = None
    if coord_cols_available:
        points_gdf = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(
                df[data_options["longitude"]],
                df[data_options["latitude"]],
            ),
            crs="EPSG:4326",
        )

    dataset_info = {
        "dataset": data_options["data_path"],
        "feature_cols": feat_cols,
        "label_col": data_options["label_col"],
        "n_samples": len(X_all),
        "spatial": use_spatial,
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": None,
        "scale_features": False,
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_autoencoder(
        X_all, y_raw, splits, le, scaler, dataset_info,
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
    if points_gdf is None:
        print("  Spatial anomaly maps skipped: no coordinate columns configured")

    # ── Calibrate threshold on the validation set ────────────────────────────
    # ── Score ALL points at once for globally consistent normalization ────────
    # anomaly_scores() uses per-batch min-max normalization; scoring each split
    # separately produces incomparable values.  Score everything together, then
    # calibrate the threshold from the val slice of those global scores.
    all_scores = det.anomaly_scores(X_all, X_spatial)
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
        if points_gdf is not None:
            gdf_split = points_gdf.iloc[idx]
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
