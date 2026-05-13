"""Train Isolation Forest on multiclass_clean.csv.

Usage:
    uv run python scripts/train_isolation_forest.py [--config path/to/config.yml]

All hyperparameters are read from the config file (or from the bundled default
when --config is omitted).  See src/geochem_detect/config/default_config_isolation_forest.yml
for the full list of tuneable settings.
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
from geochem_detect.data.loader import DEFAULT_MULTICLASS_DATA, load_dataset_frame
from geochem_detect.data.preprocessor import (
    IdentityScaler,
    make_splits,
)
from geochem_detect.training.trainer import resolve_anomaly_ground_truth, train_isolation_forest
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
        description="Train Isolation Forest anomaly detector."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file.  Overrides the bundled default.",
    )
    args = parser.parse_args()

    cfg = load_config("isolation_forest", args.config)
    validate_training_config("isolation_forest", cfg)
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)
    ep = evaluation_params(cfg)
    labels_required = _evaluation_requires_labels(ep)
    label_col = _evaluation_label_col(ep, data_cfg)

    df, data_options = load_dataset_frame(
        data_cfg,
        DEFAULT_MULTICLASS_DATA,
        require_label=labels_required,
        label_col=label_col,
    )
    feat_cols = data_options["feature_columns"]
    X_all = df[feat_cols].to_numpy(dtype=np.float32)

    # Stamp the resolved absolute path and actual feature columns back into cfg
    # so the saved training_config.yml is fully self-contained.
    cfg["data"]["data_path"] = data_options["data_path"]
    cfg["data"]["feature_columns"] = feat_cols
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

    dataset_info = {
        "dataset": data_options["data_path"],
        "feature_cols": feat_cols,
        "label_col": data_options["label_col"],
        "n_samples": len(X_all),
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": None,
        "scale_features": False,
    }

    params = {**mp, **tp}
    det, _pr_auc, run_id = train_isolation_forest(
        X_all, y_raw, splits, le, scaler, dataset_info,
        params=params,
        evaluation=ep,
        run_name="multiclass_clean",
        cfg=cfg,
    )

    out_dir = OUTPUT_ROOT / "isolation_forest" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    y_anom_all, _, _ = resolve_anomaly_ground_truth(y_raw, le, ep)
    cutoff = float(ep.get("cutoff", 0.0))
    if y_anom_all is None:
        print(
            "  Evaluation plots skipped: no evaluation.anomaly_labels "
            "or evaluation.contamination_threshold configured"
        )

    coord_cols_available = (
        data_options["longitude"] is not None
        and data_options["latitude"] is not None
        and data_options["longitude"] in df.columns
        and data_options["latitude"] in df.columns
    )
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
    else:
        print("  Spatial anomaly maps skipped: no coordinate columns configured")

    named_splits = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
        "all":   np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]]),
    }

    for split_name, idx in named_splits.items():
        X_s    = X_all[idx]
        scores = det.anomaly_scores(X_s)
        title_sfx = f"({split_name})"
        if y_anom_all is not None:
            y_anom = y_anom_all[idx]
            plot_pr_curve_binary(
                y_anom, -scores,
                title=f"Precision-Recall Curve {title_sfx}",
                save_path=out_dir / f"pr_curve_iforest_{split_name}.png",
            )
        plot_anomaly_scores_histogram(
            scores,
            title=f"Decision Function Distribution {title_sfx}",
            save_path=out_dir / f"scores_iforest_{split_name}.png",
            threshold=cutoff,
            threshold_label=f"Anomaly cutoff = {cutoff:.3f}",
            x_label="decision_function",
            show_sigma_guides=False,
        )
        if points_gdf is not None:
            gdf_split = points_gdf.iloc[idx]
            plot_spatial_anomalies(
                gdf_split,
                -scores,
                threshold=-cutoff,
                y_true=y_anom_all[idx] if y_anom_all is not None else None,
                title=f"Spatial Anomaly Map {title_sfx}",
                save_path=out_dir / f"spatial_anomaly_map_iforest_{split_name}.png",
                raw_gdf=gdf_split,
                raw_y=y_anom_all[idx] if y_anom_all is not None else None,
            )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
