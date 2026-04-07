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
from sklearn.preprocessing import LabelEncoder, RobustScaler

from geochem_detect.config import load_config, model_params, training_params
from geochem_detect.data.loader import feature_columns, load_multiclass
from geochem_detect.data.preprocessor import make_splits, split_features_labels
from geochem_detect.training.trainer import train_isolation_forest
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
    mp = model_params(cfg)
    tp = training_params(cfg)

    df = load_multiclass()
    feat_cols = feature_columns("multiclass")
    X_raw, y_raw, class_names, orig_idx = split_features_labels(df, feat_cols)

    splits = make_splits(X_raw, y_raw, orig_idx)

    scaler = RobustScaler().fit(X_raw[splits["train_idx"]])
    X_all_s = scaler.transform(X_raw).astype("float32")

    le = LabelEncoder()
    le.classes_ = class_names

    dataset_info = {
        "dataset": "multiclass_clean.csv",
        "feature_cols": feat_cols,
        "label_col": "label",
        "n_samples": len(X_raw),
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_isolation_forest(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        params=params,
        run_name="multiclass_clean",
    )

    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    contamination = tp.get("contamination_threshold", mp.get("contamination", 0.05))
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]
    y_anomaly = np.isin(y_raw[splits["test_idx"]], rare).astype(int)

    X_test_s = X_all_s[splits["test_idx"]]
    scores = det.anomaly_scores(X_test_s)
    plot_pr_curve_binary(y_anomaly, scores, save_path=out_dir / "pr_curve_iforest.png")
    plot_anomaly_scores_histogram(scores, save_path=out_dir / "scores_iforest.png")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
