"""Train multi-class classifier on multiclass_clean.csv.

Usage:
    uv run python scripts/train_classifier.py [--epochs 100]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, RobustScaler

from geochem_detect.data.loader import feature_columns, load_multiclass
from geochem_detect.data.preprocessor import make_splits, split_features_labels
from geochem_detect.training.trainer import train_classifier
from geochem_detect.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_curves_multiclass,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[64, 32])
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args = parser.parse_args()

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

    params = dict(
        epochs=args.epochs,
        hidden_dims=tuple(args.hidden_dims),
        dropout_rate=args.dropout_rate,
        learning_rate=args.learning_rate,
    )
    clf, pr_auc, run_id = train_classifier(
        X_all_s, y_raw, splits, le, scaler, dataset_info,
        params=params,
        run_name="multiclass_clean",
    )

    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    X_test_s = X_all_s[splits["test_idx"]]
    y_test = y_raw[splits["test_idx"]]
    y_pred = clf.predict(X_test_s)
    proba = clf.predict_proba(X_test_s)

    plot_class_distribution(y_raw, class_names=list(class_names), save_path=out_dir / "class_distribution.png")
    plot_confusion_matrix(y_test, y_pred, class_names=list(class_names), save_path=out_dir / "confusion_matrix.png")
    plot_pr_curves_multiclass(y_test, proba, class_names=list(class_names), save_path=out_dir / "pr_curves_multiclass.png")
    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
