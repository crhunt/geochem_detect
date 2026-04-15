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

import numpy as np
from sklearn.preprocessing import LabelEncoder

from geochem_detect.config import data_params, evaluation_params, load_config, model_params, training_params
from geochem_detect.data.loader import DEFAULT_MULTICLASS_DATA, load_dataset_frame
from geochem_detect.data.preprocessor import make_splits, scale_features, split_features_labels
from geochem_detect.training.trainer import resolve_anomaly_ground_truth, train_isolation_forest
from geochem_detect.visualization.plots import plot_anomaly_scores_histogram, plot_pr_curve_binary

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


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
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)
    ep = evaluation_params(cfg)

    df, data_options = load_dataset_frame(data_cfg, DEFAULT_MULTICLASS_DATA)
    feat_cols = data_options["feature_columns"]
    X_raw, y_raw, class_names, _ = split_features_labels(
        df,
        feat_cols,
        data_options["label_col"],
        normalize_by=data_options["normalize_by"],
    )

    splits = make_splits(X_raw, y_raw)

    ( _, X_all_s), scaler = scale_features(
        X_raw[splits["train_idx"]],
        X_raw,
        enabled=data_options["scale_features"],
    )

    le = LabelEncoder()
    le.classes_ = class_names

    dataset_info = {
        "dataset": data_options["data_path"],
        "feature_cols": feat_cols,
        "label_col": data_options["label_col"],
        "n_samples": len(X_raw),
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": data_options["normalize_by"],
        "scale_features": data_options["scale_features"],
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_isolation_forest(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        params=params,
        evaluation=ep,
        run_name="multiclass_clean",
    )

    out_dir = OUTPUT_ROOT / "isolation_forest" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    y_anom_all, _, _ = resolve_anomaly_ground_truth(y_raw, le, ep)
    cutoff = float(ep.get("cutoff", 0.0))
    if y_anom_all is None:
        print("  Evaluation plots skipped: no evaluation.anomaly_labels or evaluation.contamination_threshold configured")

    named_splits = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
        "all":   np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]]),
    }

    for split_name, idx in named_splits.items():
        X_s    = X_all_s[idx]
        scores = det.anomaly_scores(X_s)
        title_sfx = f"({split_name})"
        if y_anom_all is not None:
            y_anom = y_anom_all[idx]
            plot_pr_curve_binary(
                y_anom, scores,
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

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
