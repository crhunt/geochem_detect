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


def _load_data(info: dict, data_path: str | None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load the dataset referenced in dataset_info (or --data-path override),
    returning (X_raw, y_raw, feat_cols) without scaling."""
    from geochem_detect.data.preprocessor import split_features_labels

    feat_cols: list[str] = info["feature_cols"]
    label_col: str = info["label_col"]

    if data_path:
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found in {data_path}")
    else:
        dataset = info["dataset"]
        if dataset == "multiclass_clean.csv":
            from geochem_detect.data.loader import load_multiclass
            df = load_multiclass()
        else:
            from geochem_detect.data.loader import load_spatial
            df = load_spatial()

    X, y, class_names, orig_idx = split_features_labels(df, feat_cols, label_col)
    return X, y, list(class_names), orig_idx


def _predict_classifier(model, X_s: np.ndarray, le, label_name: str) -> pd.DataFrame:
    proba = model.predict(X_s, verbose=0)
    pred_idx = np.argmax(proba, axis=1)
    pred_labels = le.classes_[pred_idx]
    df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in le.classes_])
    df.insert(0, "predicted_label", pred_labels)
    df.insert(1, "predicted_idx", pred_idx)
    return df


def _predict_anomaly(model, X_s: np.ndarray, model_type: str, art_dir: Path) -> pd.DataFrame:
    if model_type == "isolation_forest":
        from geochem_detect.models.isolation_forest import IsolationForestDetector
        det = IsolationForestDetector.__new__(IsolationForestDetector)
        det._model = model
        scores = det.anomaly_scores(X_s)
        flags  = det.predict(X_s)
    else:
        # Compute normalised anomaly scores
        preds = model.predict(X_s, verbose=0)
        errors = np.mean((X_s - preds) ** 2, axis=1)
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
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler,
    model,
    le,
    model_type: str,
    out_dir: Path,
    art_dir: Path,
) -> None:
    X_s = scaler.transform(X_raw[idx]).astype("float32")
    true_labels = le.classes_[y_raw[idx]]

    if model_type == "classifier":
        results = _predict_classifier(model, X_s, le, split_name)
    else:
        results = _predict_anomaly(model, X_s, model_type, art_dir)

    results.insert(0, "true_label", true_labels)
    results.insert(0, "sample_idx", idx)

    out_path = out_dir / f"predictions_{split_name}.csv"
    results.to_csv(out_path, index=False)
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

    X_raw, y_raw, class_names, orig_idx = _load_data(info, args.data_path)

    out_dir = OUTPUT_ROOT / args.model_type / args.run_id / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun ID   : {args.run_id}")
    print(f"Model    : {args.model_type}")
    print(f"Dataset  : {args.data_path or info['dataset']}  ({len(X_raw)} rows)")
    print(f"Output   : {out_dir}\n")

    if args.split == "full":
        _run_split("full", np.arange(len(X_raw)), X_raw, y_raw,
                   scaler, model, le, args.model_type, out_dir, art_dir)

    elif args.split == "all":
        for name in ("train", "val", "test"):
            _run_split(name, splits[f"{name}_idx"], X_raw, y_raw,
                       scaler, model, le, args.model_type, out_dir, art_dir)

    else:
        key = f"{args.split}_idx"
        if key not in splits:
            raise KeyError(f"Split '{args.split}' not found in saved splits ({list(splits.keys())})")
        _run_split(args.split, splits[key], X_raw, y_raw,
                   scaler, model, le, args.model_type, out_dir, art_dir)


if __name__ == "__main__":
    main()
