"""Train multi-class classifier on multiclass_clean.csv.

Usage:
    uv run python scripts/train_classifier.py [--config path/to/config.yml]

All hyperparameters are read from the config file (or from the bundled default
when --config is omitted).  See src/geochem_detect/config/default_config_classifier.yml
for the full list of tuneable settings.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from geochem_detect.config import (
    data_params,
    load_config,
    model_params,
    training_params,
    validate_training_config,
)
from geochem_detect.data.loader import DEFAULT_MULTICLASS_DATA, load_dataset_frame
from geochem_detect.data.preprocessor import IdentityScaler, make_splits
from geochem_detect.training.trainer import train_classifier
from geochem_detect.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_pr_curves_multiclass,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train multi-class geochemical classifier."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file.  Overrides the bundled default.",
    )
    args = parser.parse_args()

    cfg = load_config("classifier", args.config)
    validate_training_config("classifier", cfg)
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)

    # hidden_dims must be a tuple for the model constructor
    if "hidden_dims" in mp:
        mp["hidden_dims"] = tuple(mp["hidden_dims"])

    df, data_options = load_dataset_frame(data_cfg, DEFAULT_MULTICLASS_DATA)
    feat_cols = data_options["feature_columns"]
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    le = LabelEncoder()
    y_raw = le.fit_transform(df[data_options["label_col"]].values)
    class_names = le.classes_

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
    clf, pr_auc, run_id = train_classifier(
        X_all, y_raw, splits, le, scaler, dataset_info,
        params=params,
        run_name="multiclass_clean",
    )

    out_dir = OUTPUT_ROOT / "classifier" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Class distribution is a property of the full dataset, plotted once
    plot_class_distribution(
        y_raw, class_names=list(class_names),
        save_path=out_dir / "class_distribution.png",
    )

    named_splits = {
        "train": splits["train_idx"],
        "val":   splits["val_idx"],
        "test":  splits["test_idx"],
        "all":   np.concatenate([splits["train_idx"], splits["val_idx"], splits["test_idx"]]),
    }

    for split_name, idx in named_splits.items():
        X_s    = X_all[idx]
        y_s    = y_raw[idx]
        y_pred = clf.predict(X_s)
        proba  = clf.predict_proba(X_s)
        title_sfx = f"({split_name})"
        plot_confusion_matrix(
            y_s, y_pred, class_names=list(class_names),
            title=f"Confusion Matrix {title_sfx}",
            save_path=out_dir / f"confusion_matrix_{split_name}.png",
        )
        plot_pr_curves_multiclass(
            y_s, proba, class_names=list(class_names),
            title=f"Per-class Precision-Recall Curves {title_sfx}",
            save_path=out_dir / f"pr_curves_multiclass_{split_name}.png",
        )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
