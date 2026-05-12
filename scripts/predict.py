"""Run a trained model against data splits or a full dataset.

The model artefacts (scaler, label encoder, split indices, dataset info) must
have been saved by a training script.  They live under:
    outputs/<run_id>/artefacts/

Usage examples
--------------
# Run against the test split used during training:
uv run python scripts/predict.py --run-id <run_id> --model-type classifier

# Run against all splits individually:
uv run python scripts/predict.py --run-id <run_id> --model-type classifier --split all

# Run against a completely different CSV (full dataset, no split filtering):
uv run python scripts/predict.py --run-id <run_id> --model-type classifier \\
    --split full --data-path /path/to/other.csv

Model types: classifier | autoencoder | isolation_forest
Splits:      train | val | test | full | all
             'full' uses every row of the source dataset (or --data-path if given).
             'all'  runs train, val, and test splits separately and reports each.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"
_LEGACY_COORDINATE_CANDIDATES = [
    ("latitude", "longitude"),
    ("lat", "long"),
    ("EASTING", "NORTHING"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_artefacts(run_id: str, model_type: str) -> dict:
    art = OUTPUT_ROOT / model_type / run_id / "artefacts"
    if not art.exists():
        raise FileNotFoundError(
            f"Artefacts not found for run_id={run_id} at {art}.\n"
            "Make sure the model was trained with a current training script."
        )
    scaler = pickle.loads((art / "scaler.pkl").read_bytes())
    le     = pickle.loads((art / "label_encoder.pkl").read_bytes())
    splits = np.load(art / "splits.npz")
    with open(art / "dataset_info.json") as f:
        info = json.load(f)
    return dict(scaler=scaler, le=le, splits=splits, info=info, art_dir=art)


def _load_model(run_id: str, model_type: str):
    art = OUTPUT_ROOT / model_type / run_id / "artefacts"
    if model_type == "isolation_forest":
        return pickle.loads((art / "model.pkl").read_bytes())
    keras_path = art / "keras_model.keras"
    if not keras_path.exists():
        raise FileNotFoundError(f"Keras model not found at {keras_path}")
    import tensorflow as tf
    return tf.keras.models.load_model(str(keras_path))


def _load_data(
    info: dict,
    data_path: str | None,
    *,
    require_label: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load the prediction dataset and rebuild only the model input tensors."""
    from geochem_detect.data.loader import load_dataset_frame

    data_cfg = {
        "data_path": data_path or info["dataset"],
        "feature_columns": info["feature_cols"],
        "longitude": info.get("longitude"),
        "latitude": info.get("latitude"),
        "normalize_by": info.get("normalize_by"),
        "scale_features": info.get("scale_features", True),
    }
    df, data_options = load_dataset_frame(
        data_cfg,
        info["dataset"],
        require_label=require_label,
        label_col=info.get("label_col", "label"),
    )
    label_col = data_options["label_col"]
    if label_col in df.columns:
        df = df.copy()
        df[label_col] = df[label_col].astype("string").str.strip()
    feat_cols = data_options["feature_columns"]
    missing = [column_name for column_name in feat_cols if column_name not in df.columns]
    if missing:
        raise ValueError(f"Configured feature columns not found in prediction data: {missing}")
    if df[feat_cols].isna().any().any():
        raise ValueError(
            "Prediction data contains missing feature values. Run preprocessing "
            "first instead of relying on prediction-time cleaning."
        )

    df_clean = df.reset_index(drop=True)
    X_chem = df_clean[feat_cols].to_numpy(dtype=np.float32)
    X_spatial = None
    if info.get("spatial", False):
        latitude, longitude = _resolve_spatial_columns(
            df_clean,
            data_options.get("latitude"),
            data_options.get("longitude"),
        )
        X_spatial = df_clean[[latitude, longitude]].to_numpy(dtype=np.float32)

    y_true = None
    if label_col in df_clean.columns:
        y_true = df_clean[label_col].fillna("").astype(str).to_numpy(dtype=object)

    return df_clean.reset_index(drop=True), X_chem, X_spatial, y_true


def _resolve_spatial_columns(
    df_input: pd.DataFrame,
    latitude: str | None,
    longitude: str | None,
) -> tuple[str, str]:
    """Resolve usable spatial columns from saved metadata or common fallbacks."""
    if (
        latitude is not None
        and longitude is not None
        and latitude in df_input.columns
        and longitude in df_input.columns
    ):
        return latitude, longitude

    for candidate_lat, candidate_lon in _LEGACY_COORDINATE_CANDIDATES:
        if candidate_lat in df_input.columns and candidate_lon in df_input.columns:
            return candidate_lat, candidate_lon

    raise ValueError(
        "Saved run expects spatial inputs, but prediction data has no usable "
        "latitude/longitude columns."
    )


def _predict_classifier(model, X_s: np.ndarray, le) -> pd.DataFrame:
    proba = model(X_s, training=False).numpy()
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.classes_[pred_idx]
    df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in le.classes_])
    df.insert(0, "predicted_label", pred_labels)
    df.insert(1, "predicted_idx", pred_idx)
    return df


def _predict_anomaly(
    model,
    X_chem: np.ndarray,
    model_type: str,
    art_dir: Path,
    X_spatial: np.ndarray | None = None,
) -> pd.DataFrame:
    if model_type == "isolation_forest":
        from geochem_detect.models.isolation_forest import IsolationForestDetector
        det = IsolationForestDetector.__new__(IsolationForestDetector)
        det._model = model
        scores = det.anomaly_scores(X_chem)
        threshold_file = art_dir / "anomaly_threshold.json"
        cutoff = 0.0
        if threshold_file.exists():
            cfg = json.loads(threshold_file.read_text())
            cutoff = float(cfg.get("cutoff", 0.0))
        flags = (scores < cutoff).astype(int)
    else:
        # Compute normalised anomaly scores
        inputs = [X_chem, X_spatial] if X_spatial is not None else X_chem
        preds = model(inputs, training=False).numpy()
        errors = np.mean((X_chem - preds) ** 2, axis=1)
        mn, mx = errors.min(), errors.max()
        scores = (errors - mn) / (mx - mn) if mx > mn else np.zeros_like(errors)

        # Load the calibrated threshold persisted at training time
        threshold_file = art_dir / "anomaly_threshold.json"
        if threshold_file.exists():
            cfg = json.loads(threshold_file.read_text())
            if "threshold" in cfg:
                threshold = float(cfg["threshold"])
            else:
                # Legacy run: fall back to sigma-based computation
                sigma_cutoff = cfg.get("sigma_cutoff", 2.0)
                threshold = float(np.mean(scores) + sigma_cutoff * np.std(scores))
                print(
                    "  [warn] anomaly_threshold.json has no 'threshold' key; "
                    "recomputing from sigma_cutoff.  Re-train to persist the calibrated value."
                )
        else:
            threshold = float(np.mean(scores) + 2.0 * np.std(scores))
            print(
                "  [warn] anomaly_threshold.json not found; using sigma_cutoff=2.0."
            )
        flags = (scores >= threshold).astype(int)

    return pd.DataFrame({"anomaly_score": scores, "is_anomaly": flags})


def _run_split(
    split_name: str,
    idx: np.ndarray,
    df_input: pd.DataFrame,
    X_chem: np.ndarray,
    X_spatial: np.ndarray | None,
    y_true: np.ndarray | None,
    scaler,
    model,
    le,
    model_type: str,
    out_dir: Path,
    art_dir: Path,
) -> None:
    if len(idx) and int(np.max(idx)) >= len(df_input):
        raise ValueError(
            "Saved split indices are incompatible with the current prediction "
            "dataset length. Re-run training or use --split full."
        )

    chem_slice = scaler.transform(X_chem[idx]).astype("float32")
    spatial_slice = X_spatial[idx].astype("float32") if X_spatial is not None else None

    if model_type == "classifier":
        results = _predict_classifier(model, chem_slice, le)
    else:
        results = _predict_anomaly(model, chem_slice, model_type, art_dir, X_spatial=spatial_slice)

    output_df = df_input.iloc[idx].reset_index(drop=True).copy()
    output_df.insert(0, "prediction_row_idx", idx)
    if (
        y_true is not None
        and "true_label" not in output_df.columns
        and "label" not in output_df.columns
    ):
        output_df.insert(1, "true_label", y_true[idx])
    output_df = pd.concat([output_df, results.reset_index(drop=True)], axis=1)

    out_path = out_dir / f"predictions_{split_name}.csv"
    output_df.to_csv(out_path, index=False)
    print(f"  [{split_name:5s}]  {len(idx):5d} samples → {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id",     required=True, help="MLFlow run ID")
    parser.add_argument("--model-type", required=True,
                        choices=["classifier", "autoencoder", "isolation_forest"])
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test", "full", "all"],
                        help="Which data split to run inference on")
    parser.add_argument("--data-path",  default=None,
                        help="Override: path to a different CSV (only with --split full)")
    args = parser.parse_args()

    if args.data_path and args.split != "full":
        parser.error("--data-path can only be used with --split full")

    art = _load_artefacts(args.run_id, args.model_type)
    model   = _load_model(args.run_id, args.model_type)
    scaler  = art["scaler"]
    le      = art["le"]
    splits  = art["splits"]
    info    = art["info"]
    art_dir = art["art_dir"]

    sentinel_unlabeled = len(le.classes_) == 1 and str(le.classes_[0]) == "data"
    require_label = args.split != "full" and (
        args.model_type == "classifier" or not sentinel_unlabeled
    )

    df_input, X_chem, X_spatial, y_true = _load_data(
        info,
        args.data_path,
        require_label=require_label,
    )

    out_dir = OUTPUT_ROOT / args.model_type / args.run_id / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun ID   : {args.run_id}")
    print(f"Model    : {args.model_type}")
    print(f"Dataset  : {args.data_path or info['dataset']}  ({len(X_chem)} rows)")
    print(f"Output   : {out_dir}\n")

    if args.split == "full":
        _run_split("full", np.arange(len(X_chem)), df_input, X_chem, X_spatial, y_true,
                   scaler, model, le, args.model_type, out_dir, art_dir)

    elif args.split == "all":
        for name in ("train", "val", "test"):
            _run_split(name, splits[f"{name}_idx"], df_input, X_chem, X_spatial, y_true,
                       scaler, model, le, args.model_type, out_dir, art_dir)

    else:
        key = f"{args.split}_idx"
        if key not in splits:
            raise KeyError(
                f"Split '{args.split}' not found in saved splits "
                f"({list(splits.keys())})"
            )
        _run_split(args.split, splits[key], df_input, X_chem, X_spatial, y_true,
                   scaler, model, le, args.model_type, out_dir, art_dir)


if __name__ == "__main__":
    main()
