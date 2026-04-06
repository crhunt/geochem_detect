"""Train Isolation Forest on multiclass_clean.csv.

Usage:
    uv run python scripts/train_isolation_forest.py [--contamination 0.05]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from geochem_detect.data.loader import feature_columns, load_multiclass
from geochem_detect.data.preprocessor import make_splits, scale_features, split_features_labels
from geochem_detect.training.trainer import train_isolation_forest
from geochem_detect.visualization.plots import plot_anomaly_scores_histogram, plot_pr_curve_binary

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = load_multiclass()
    feat_cols = feature_columns("multiclass")
    X_raw, y_raw, class_names, orig_idx = split_features_labels(df, feat_cols)

    splits = make_splits(X_raw, y_raw, orig_idx)
    (X_all_s,), scaler = scale_features(X_raw)

    import numpy as np
    # Re-scale consistently: fit on train, apply to full array
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler().fit(X_raw[splits["train_idx"]])
    X_all_s = scaler.transform(X_raw).astype("float32")

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = class_names

    dataset_info = {
        "dataset": "multiclass_clean.csv",
        "feature_cols": feat_cols,
        "label_col": "label",
        "n_samples": len(X_raw),
    }

    params = dict(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        contamination_threshold=args.contamination,
    )
    det, pr_auc, run_id = train_isolation_forest(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        params=params,
        run_name="multiclass_clean",
    )

    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mirror the rare-class definition used in trainer: based on full y distribution
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(args.contamination * len(y_raw))]
    y_anomaly = np.isin(y_raw[splits["test_idx"]], rare).astype(int)

    X_test_s = X_all_s[splits["test_idx"]]
    scores = det.anomaly_scores(X_test_s)
    plot_pr_curve_binary(y_anomaly, scores, save_path=out_dir / "pr_curve_iforest.png")
    plot_anomaly_scores_histogram(scores, save_path=out_dir / "scores_iforest.png")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
