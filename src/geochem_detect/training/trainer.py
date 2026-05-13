"""MLFlow-integrated training runners for all three models."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import yaml

import mlflow
import numpy as np

OUTPUT_ROOT = Path(__file__).parents[3] / "outputs"


def _save_keras_model(model, path: str | Path) -> None:
    """Save a Keras model to disk."""
    model.save(str(path))


def _mlflow_log_keras_model(model, artifact_path: str) -> None:
    """Log a Keras model to MLflow."""
    mlflow.tensorflow.log_model(model, artifact_path=artifact_path)


def _log_params(params: dict[str, Any]) -> None:
    for k, v in params.items():
        mlflow.log_param(k, v)


def resolve_anomaly_ground_truth(
    y_encoded: np.ndarray,
    label_encoder,
    evaluation: dict | None = None,
) -> tuple[np.ndarray | None, list[str], dict[str, Any]]:
    """Build binary anomaly labels for evaluation from config.

    Evaluation uses one of two mutually-exclusive sources:
    explicit label values via ``evaluation.anomaly_labels`` or label-frequency
    thresholding via ``evaluation.contamination_threshold``.
    """
    evaluation = dict(evaluation or {})
    anomaly_labels = evaluation.get("anomaly_labels")
    contamination = evaluation.get("contamination_threshold")

    if anomaly_labels is not None and contamination is not None:
        raise ValueError(
            "Configure either evaluation.anomaly_labels or "
            "evaluation.contamination_threshold, not both."
        )

    if anomaly_labels is not None:
        configured = [str(label).strip() for label in anomaly_labels if str(label).strip()]
        if not configured:
            return None, [], {"method": None}
        class_names = np.asarray(label_encoder.classes_, dtype=object)
        missing = [label for label in configured if label not in class_names.tolist()]
        if missing:
            raise ValueError(
                f"Unknown evaluation.anomaly_labels: {missing}. "
                f"Available labels: {class_names.tolist()}"
            )
        rare_mask = np.isin(class_names, configured)
        rare_indices = np.flatnonzero(rare_mask)
        y_anomaly = np.isin(y_encoded, rare_indices).astype(np.int32)
        return y_anomaly, configured, {"method": "anomaly_labels", "anomaly_labels": configured}

    if contamination is not None:
        classes_all, counts_all = np.unique(y_encoded, return_counts=True)
        threshold_count = int(float(contamination) * len(y_encoded))
        rare = classes_all[counts_all < threshold_count]
        rare_labels = label_encoder.classes_[rare].tolist()
        y_anomaly = np.isin(y_encoded, rare).astype(np.int32)
        return y_anomaly, rare_labels, {
            "method": "contamination_threshold",
            "contamination_threshold": float(contamination),
        }

    return None, [], {"method": None}


def _save_run_artefacts(
    run_id: str,
    *,
    scaler,
    label_encoder,
    splits: dict,
    dataset_info: dict,
    model_obj=None,
    model_type: str = "",
    cfg: dict | None = None,
) -> Path:
    """Persist scaler, label encoder, split indices, and dataset metadata to disk.

    Saved under outputs/<model_type>/<run_id>/artefacts/ so predict.py can load
    them without touching the MLFlow artifact store.
    """
    out = OUTPUT_ROOT / model_type / run_id / "artefacts"
    out.mkdir(parents=True, exist_ok=True)

    (out / "scaler.pkl").write_bytes(pickle.dumps(scaler))
    (out / "label_encoder.pkl").write_bytes(pickle.dumps(label_encoder))

    np.savez(
        out / "splits.npz",
        train_idx=splits["train_idx"],
        val_idx=splits["val_idx"],
        test_idx=splits["test_idx"],
    )

    with open(out / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    if model_obj is not None:
        (out / "model.pkl").write_bytes(pickle.dumps(model_obj))

    if cfg is not None:
        with open(out / "training_config.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # Log artefacts folder into MLFlow as well
    mlflow.log_artifacts(str(out), artifact_path="run_artefacts")
    return out


def _validation_threshold(scores: np.ndarray, sigma_cutoff: float) -> float:
    """Calibrate an anomaly cutoff as mean + sigma * std on validation scores."""
    return float(np.mean(scores) + sigma_cutoff * np.std(scores))


def train_isolation_forest(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    label_encoder,
    scaler,
    dataset_info: dict,
    params: dict | None = None,
    evaluation: dict | None = None,
    experiment_name: str = "isolation_forest",
    run_name: str | None = None,
    cfg: dict | None = None,
) -> tuple[Any, float | None, str]:
    """Train an Isolation Forest and log to MLFlow.

    Parameters
    ----------
    X : full scaled feature array (all rows, in original order)
    y : encoded labels (all rows)
    splits : dict from make_splits() with *_idx arrays
    label_encoder : fitted LabelEncoder
    scaler : fitted RobustScaler
    dataset_info : dict with 'dataset', 'feature_cols', etc.

    Returns
    -------
    (detector, pr_auc, run_id)
    """
    from geochem_detect.models.isolation_forest import IsolationForestDetector

    params = params or {}
    X_train = X[splits["train_idx"]]
    X_test  = X[splits["test_idx"]]

    y_anomaly_all, anomaly_labels, evaluation_info = resolve_anomaly_ground_truth(
        y, label_encoder, evaluation
    )
    y_anomaly = y_anomaly_all[splits["test_idx"]] if y_anomaly_all is not None else None
    cutoff = 0.0 if evaluation is None else float(evaluation.get("cutoff", 0.0))
    if y_anomaly is not None:
        print(
            f"  Evaluation anomaly labels: {anomaly_labels}, "
            f"positives in test={int(y_anomaly.sum())}"
        )
    else:
        print("  Evaluation disabled: no anomaly labels configured")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        det = IsolationForestDetector(**params)
        _log_params(det.params)
        det.fit(X_train)

        pr_auc = det.pr_auc(X_test, y_anomaly) if y_anomaly is not None else None
        if pr_auc is not None:
            mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("train_size", len(splits["train_idx"]))
        mlflow.log_metric("val_size",   len(splits["val_idx"]))
        mlflow.log_metric("test_size",  len(splits["test_idx"]))
        if evaluation_info.get("method") is not None:
            _log_params({f"evaluation_{k}": v for k, v in evaluation_info.items() if v is not None})
        mlflow.log_param("evaluation_cutoff", cutoff)

        art_dir = _save_run_artefacts(
            run_id,
            scaler=scaler,
            label_encoder=label_encoder,
            splits=splits,
            dataset_info=dataset_info,
            model_obj=det._model,
            model_type="isolation_forest",
            cfg=cfg,
        )
        with open(art_dir / "anomaly_threshold.json", "w") as f:
            json.dump({"cutoff": cutoff}, f, indent=2)
        if pr_auc is None:
            print(f"[IsolationForest] Evaluation skipped  run_id={run_id}")
        else:
            print(f"[IsolationForest] PR-AUC: {pr_auc:.4f}  run_id={run_id}")
    return det, pr_auc, run_id


def train_autoencoder(
    X_chem: np.ndarray,
    y: np.ndarray,
    splits: dict,
    label_encoder,
    scaler,
    dataset_info: dict,
    X_spatial: np.ndarray | None = None,
    params: dict | None = None,
    evaluation: dict | None = None,
    experiment_name: str = "autoencoder",
    run_name: str | None = None,
    cfg: dict | None = None,
) -> tuple[Any, float | None, str]:
    """Train the autoencoder anomaly detector and log to MLFlow.

    Returns
    -------
    (detector, pr_auc, run_id)
    """
    from geochem_detect.models.autoencoder import AutoencoderDetector

    params = params or {}
    n_features = X_chem.shape[1]
    n_spatial = X_spatial.shape[1] if X_spatial is not None else 0

    tr, va, te = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    X_chem_tr, X_chem_va, X_chem_te = X_chem[tr], X_chem[va], X_chem[te]
    X_sp_tr = X_spatial[tr] if X_spatial is not None else None
    X_sp_va = X_spatial[va] if X_spatial is not None else None
    X_sp_te = X_spatial[te] if X_spatial is not None else None

    # Remove training-level keys that are not AutoencoderDetector constructor params
    params.pop("spatial", None)          # handled by the caller via X_spatial
    y_anomaly_all, anomaly_labels, evaluation_info = resolve_anomaly_ground_truth(
        y, label_encoder, evaluation
    )
    y_anomaly = y_anomaly_all[te] if y_anomaly_all is not None else None
    sigma_cutoff = None if evaluation is None else evaluation.get("anomaly_sigma_cutoff")
    if y_anomaly is not None:
        print(
            f"  Evaluation anomaly labels: {anomaly_labels}, "
            f"positives in test={int(y_anomaly.sum())}"
        )
    else:
        print("  Evaluation disabled: no anomaly labels configured")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        det = AutoencoderDetector(n_features=n_features, n_spatial=n_spatial, **params)
        _log_params(det.params)
        det.fit(X_chem_tr, X_sp_tr, X_chem_val=X_chem_va, X_spatial_val=X_sp_va)

        epochs_run = len(det.history_.history["loss"])
        pr_auc = det.pr_auc(X_chem_te, y_anomaly, X_sp_te) if y_anomaly is not None else None

        if sigma_cutoff is not None:
            mlflow.log_param("evaluation_anomaly_sigma_cutoff", sigma_cutoff)
        if evaluation_info.get("method") is not None:
            _log_params({f"evaluation_{k}": v for k, v in evaluation_info.items() if v is not None})
        if pr_auc is not None:
            mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("epochs_run", epochs_run)
        mlflow.log_metric("train_size", len(tr))
        mlflow.log_metric("val_size",   len(va))
        mlflow.log_metric("test_size",  len(te))

        _mlflow_log_keras_model(det.model, artifact_path="autoencoder_model")
        art_dir = _save_run_artefacts(
            run_id,
            scaler=scaler,
            label_encoder=label_encoder,
            splits=splits,
            dataset_info=dataset_info,
            model_type="autoencoder",
            cfg=cfg,
        )
        if sigma_cutoff is not None:
            # Persist sigma_cutoff so predict.py can apply the same threshold rule
            with open(art_dir / "anomaly_threshold.json", "w") as f:
                json.dump({"sigma_cutoff": sigma_cutoff}, f, indent=2)
        keras_path = art_dir / "keras_model.keras"
        _save_keras_model(det.model, keras_path)

        if pr_auc is None:
            print(f"[Autoencoder] Evaluation skipped  (epochs={epochs_run})  run_id={run_id}")
        else:
            print(f"[Autoencoder] PR-AUC: {pr_auc:.4f}  (epochs={epochs_run})  run_id={run_id}")
    return det, pr_auc, run_id


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    label_encoder,
    scaler,
    dataset_info: dict,
    params: dict | None = None,
    experiment_name: str = "classifier",
    run_name: str | None = None,
    cfg: dict | None = None,
) -> tuple[Any, float, str]:
    """Train the multi-class classifier and log to MLFlow.

    Returns
    -------
    (classifier, pr_auc_macro, run_id)
    """
    from sklearn.metrics import f1_score

    from geochem_detect.models.classifier import GeochemClassifier

    params = params or {}
    tr, va, te = splits["train_idx"], splits["val_idx"], splits["test_idx"]
    X_train, X_val, X_test = X[tr], X[va], X[te]
    y_train, y_val, y_test = y[tr], y[va], y[te]
    n_classes = len(np.unique(y_train))

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        clf = GeochemClassifier(
            n_features=X.shape[1],
            n_classes=n_classes,
            class_names=label_encoder.classes_,
            **params,
        )
        _log_params(clf.params)
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        epochs_run = len(clf.history_.history["loss"])
        pr_auc = clf.pr_auc_macro(X_test, y_test)
        y_pred = clf.predict(X_test)
        f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

        mlflow.log_metric("pr_auc_macro", pr_auc)
        mlflow.log_metric("f1_macro", f1)
        mlflow.log_metric("epochs_run", epochs_run)
        mlflow.log_metric("train_size", len(tr))
        mlflow.log_metric("val_size",   len(va))
        mlflow.log_metric("test_size",  len(te))

        _mlflow_log_keras_model(clf.model, artifact_path="classifier_model")
        _save_run_artefacts(
            run_id,
            scaler=scaler,
            label_encoder=label_encoder,
            splits=splits,
            dataset_info=dataset_info,
            model_type="classifier",
            cfg=cfg,
        )
        keras_path = OUTPUT_ROOT / "classifier" / run_id / "artefacts" / "keras_model.keras"
        _save_keras_model(clf.model, keras_path)

        print(
            f"[Classifier] PR-AUC macro: {pr_auc:.4f}  F1 macro: {f1:.4f}"
            f"  (epochs={epochs_run})  run_id={run_id}"
        )
    return clf, pr_auc, run_id


def train_cnn_sae(
    gdf,
    y_encoded: np.ndarray,
    label_encoder,
    scaler,
    dataset_info: dict,
    sampling_params: dict | None = None,
    params: dict | None = None,
    evaluation: dict | None = None,
    experiment_name: str = "cnn_sae",
    run_name: str | None = None,
    cfg: dict | None = None,
) -> tuple[Any, float | None, str]:
    """Train the CNN-SAE spatial anomaly detector and log to MLFlow.

    The model learns to reconstruct spatially-windowed geochemical grids from
    Data1.csv.  Anomaly ground truth follows the same label-frequency approach
    as the existing autoencoder: any class whose global count falls below
    ``contamination_threshold × N`` is considered anomalous, and a window is
    anomalous if it contains at least one anomalous source point.

    Parameters
    ----------
    gdf:
        GeoDataFrame loaded by :func:`geochem_detect.data.loader.load_spatial`,
        pre-cleaned and with features already scaled by *scaler*.
    y_encoded:
        Integer-encoded label array aligned with the valid rows in *gdf*.
    label_encoder:
        Fitted :class:`sklearn.preprocessing.LabelEncoder`.
    scaler:
        Fitted :class:`sklearn.preprocessing.RobustScaler` (used for artefact
        persistence; features in *gdf* should already be transformed).
    dataset_info:
        Dict with ``dataset``, ``feature_cols``, ``label_col``, ``n_samples``.
    sampling_params:
        Dict with window-sampling settings; forwarded to
        :class:`geochem_detect.data.spatial_sampler.SpatialSampler`.
    params:
        Combined model + training hyperparameters. Training-level keys
        (``val_size``, ``test_size``) are extracted and removed; remainder forwarded to
        :class:`geochem_detect.models.cnn_sae.CnnSaeDetector`.
    experiment_name:
        MLFlow experiment name.
    run_name:
        Optional display name for this MLFlow run.

    Returns
    -------
    (detector, pr_auc, run_id)
    """
    from sklearn.model_selection import train_test_split

    from geochem_detect.data.spatial_sampler import SpatialSampler
    from geochem_detect.models.cnn_sae import CnnSaeDetector

    sampling_params = dict(sampling_params or {})
    params = dict(params or {})

    # ── Extract training-level keys ─────────────────────────────────────────
    val_size       = params.pop("val_size", 0.15)
    test_size      = params.pop("test_size", 0.15)

    y_anomaly_all, anomaly_labels, evaluation_info = resolve_anomaly_ground_truth(
        y_encoded, label_encoder, evaluation
    )
    sigma_cutoff = None if evaluation is None else evaluation.get("anomaly_sigma_cutoff")
    point_anomaly = (
        y_anomaly_all.astype(np.int32)
        if y_anomaly_all is not None
        else np.zeros(len(y_encoded), dtype=np.int32)
    )
    if y_anomaly_all is not None:
        print(
            f"  Evaluation anomaly labels: {anomaly_labels}, "
            f"anomalous points={int(point_anomaly.sum())}/{len(point_anomaly)}"
        )
    else:
        print("  Evaluation disabled: no anomaly labels configured")

    # ── Generate spatially-windowed samples ─────────────────────────────────
    feat_cols: list[str] = dataset_info["feature_cols"]
    sampler = SpatialSampler(
        gdf=gdf,
        feature_cols=feat_cols,
        anomaly_labels=point_anomaly,
        **sampling_params,
    )
    X, y_windows, metadata = sampler.generate()
    n_features = len(feat_cols)
    n_anomalous = int(y_windows.sum())
    print(
        f"  Generated {len(X)} windows: {n_anomalous} anomalous "
        f"({100 * n_anomalous / max(len(X), 1):.1f}%)"
    )

    # ── Train / val / test split of windows ─────────────────────────────────
    all_idx = np.arange(len(X))
    # Stratify only if both classes are present
    strat = y_windows if len(np.unique(y_windows)) > 1 else None
    idx_tv, idx_test = train_test_split(
        all_idx, test_size=test_size, random_state=42, stratify=strat
    )
    strat_tv = y_windows[idx_tv] if strat is not None else None
    relative_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=relative_val, random_state=42, stratify=strat_tv
    )

    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_test = y_windows[idx_test]

    # ── MLFlow run ──────────────────────────────────────────────────────────
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Derive grid_size and n_features from the actual data
        grid_size = X.shape[1]
        det_params = {
            "grid_size": grid_size,
            "n_features": n_features,
            **params,
        }
        det = CnnSaeDetector(**det_params)
        _log_params(det.params)
        _log_params({f"sampling_{k}": v for k, v in sampling_params.items()})
        if sigma_cutoff is not None:
            mlflow.log_param("evaluation_anomaly_sigma_cutoff", sigma_cutoff)
        if evaluation_info.get("method") is not None:
            _log_params({f"evaluation_{k}": v for k, v in evaluation_info.items() if v is not None})

        det.fit(X_train, X_val=X_val)

        epochs_run = len(det.history_.history["loss"])
        all_scores = det.anomaly_scores(X)
        threshold = None
        if sigma_cutoff is not None:
            val_scores = all_scores[idx_val]
            threshold = _validation_threshold(val_scores, float(sigma_cutoff))
            mlflow.log_param("evaluation_anomaly_threshold", threshold)
        pr_auc = det.pr_auc(X_test, y_test) if y_anomaly_all is not None else None

        if pr_auc is not None:
            mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("epochs_run", epochs_run)
        mlflow.log_metric("train_size", len(idx_train))
        mlflow.log_metric("val_size",   len(idx_val))
        mlflow.log_metric("test_size",  len(idx_test))
        mlflow.log_metric("window_anomaly_rate", n_anomalous / max(len(X), 1))

        # ── Persist artefacts ────────────────────────────────────────────────
        art_dir = OUTPUT_ROOT / "cnn_sae" / run_id / "artefacts"
        art_dir.mkdir(parents=True, exist_ok=True)

        (art_dir / "scaler.pkl").write_bytes(pickle.dumps(scaler))
        (art_dir / "label_encoder.pkl").write_bytes(pickle.dumps(label_encoder))

        with open(art_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        if cfg is not None:
            with open(art_dir / "training_config.yml", "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

        if sigma_cutoff is not None:
            with open(art_dir / "anomaly_threshold.json", "w") as f:
                json.dump(
                    {"sigma_cutoff": sigma_cutoff, "threshold": threshold},
                    f,
                    indent=2,
                )

        with open(art_dir / "sampling_params.json", "w") as f:
            json.dump(sampling_params, f, indent=2)

        # Save window split indices
        np.savez(
            art_dir / "window_splits.npz",
            train_idx=idx_train,
            val_idx=idx_val,
            test_idx=idx_test,
        )

        # Save metadata for prediction reconstruction
        with open(art_dir / "window_metadata.json", "w") as f:
            # Convert numpy int types to plain Python ints for JSON serialisation
            serialisable = [
                {
                    "center_lat": m["center_lat"],
                    "center_lon": m["center_lon"],
                    "point_indices": [int(i) for i in m["point_indices"]],
                    "n_points": m["n_points"],
                }
                for m in metadata
            ]
            json.dump(serialisable, f, indent=2)

        keras_path = art_dir / "keras_model.keras"
        _save_keras_model(det.model, keras_path)
        _mlflow_log_keras_model(det.model, artifact_path="cnn_sae_model")
        mlflow.log_artifacts(str(art_dir), artifact_path="run_artefacts")

        if pr_auc is None:
            print(f"[CNN-SAE] Evaluation skipped  (epochs={epochs_run})  run_id={run_id}")
        else:
            print(f"[CNN-SAE] PR-AUC: {pr_auc:.4f}  (epochs={epochs_run})  run_id={run_id}")
        if threshold is not None:
            print(
                f"  [CNN-SAE] val threshold = {threshold:.4f} "
                f"(mean={np.mean(all_scores[idx_val]):.4f}, sigma_cutoff={sigma_cutoff})"
            )
    return det, pr_auc, run_id
