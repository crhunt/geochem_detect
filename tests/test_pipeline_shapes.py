from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from geochem_detect.data.loader import DEFAULT_MULTICLASS_DATA, DEFAULT_SPATIAL_DATA, load_dataset_frame
from geochem_detect.data.preprocessor import IdentityScaler, make_splits
from geochem_detect.training import trainer as trainer_module


MULTICLASS_FEATURE_COLUMNS = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
]

SPATIAL_FEATURE_COLUMNS = [
    "SiO2n",
    "TiO2n",
    "Al2O3n",
    "FeO*n",
    "MnOn",
    "MgOn",
    "CaOn",
    "Na2On",
    "K2On",
    "P2O5n",
]


def _csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _configure_preprocess_module(
    module,
    *,
    data_path: Path,
    label_col: str,
    spatial: bool,
    feature_columns: list[str],
    scale_features: bool,
) -> None:
    module.DATA_PATH = data_path
    module.LABEL_COL = label_col
    module.IDENTIFIER_COL = None
    module.SPATIAL = spatial
    module.LAT_COL = "lat"
    module.LON_COL = "long"
    module.SCALE_FEATURES = scale_features
    module.NORMALIZE_BY = None
    module.COLS_FILE = None
    module.FEATURE_COLUMNS = feature_columns


def _configure_runtime_roots(tmp_path, monkeypatch, predict_script) -> Path:
    outputs_root = tmp_path / "outputs"
    tracking_db = tmp_path / "mlflow.db"
    monkeypatch.setattr(trainer_module, "OUTPUT_ROOT", outputs_root)
    monkeypatch.setattr(predict_script, "OUTPUT_ROOT", outputs_root)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_db}")
    return outputs_root


def test_isolation_forest_pipeline_preserves_expected_shapes(
    preprocess_script,
    predict_script,
    tmp_path,
    monkeypatch,
) -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    raw_path = fixture_dir / "multiclass_raw_sample.csv"
    expected_path = fixture_dir / "multiclass_expected_processed.csv"
    processed_path = tmp_path / "multiclass_processed.csv"
    outputs_root = _configure_runtime_roots(tmp_path, monkeypatch, predict_script)

    _configure_preprocess_module(
        preprocess_script,
        data_path=raw_path,
        label_col="ROCK1",
        spatial=False,
        feature_columns=MULTICLASS_FEATURE_COLUMNS,
        scale_features=True,
    )
    preprocess_script.preprocess(raw_path, processed_path)

    assert _csv_text(processed_path) == _csv_text(expected_path)

    df, data_options = load_dataset_frame(
        {
            "data_path": str(processed_path),
            "feature_columns": MULTICLASS_FEATURE_COLUMNS,
            "label": "label",
            "scale_features": False,
        },
        DEFAULT_MULTICLASS_DATA,
    )
    X_all = df[data_options["feature_columns"]].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[data_options["label_col"]].values)
    splits = make_splits(X_all, y_all, random_state=0)
    scaler = IdentityScaler().fit(X_all[splits["train_idx"]])
    dataset_info = {
        "dataset": str(processed_path),
        "feature_cols": data_options["feature_columns"],
        "label_col": data_options["label_col"],
        "n_samples": len(X_all),
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": None,
        "scale_features": False,
    }

    detector, pr_auc, run_id = trainer_module.train_isolation_forest(
        X_all,
        y_all,
        splits,
        label_encoder,
        scaler,
        dataset_info,
        params={
            "n_estimators": 16,
            "contamination": 0.2,
            "max_features": 1.0,
            "random_state": 0,
            "n_jobs": 1,
        },
        evaluation=None,
        experiment_name="pytest_isolation_forest",
        run_name="pytest_isolation_forest",
        cfg={
            "data": {
                "data_path": str(processed_path),
                "feature_columns": list(data_options["feature_columns"]),
                "label": data_options["label_col"],
                "longitude": data_options["longitude"],
                "latitude": data_options["latitude"],
                "normalize_by": None,
                "scale_features": False,
            },
            "model": {},
            "training": {"spatial": False},
            "evaluation": {},
        },
    )

    assert pr_auc is None
    assert detector.anomaly_scores(X_all).shape == (len(df),)

    art = predict_script._load_artefacts(run_id, "isolation_forest")
    model = predict_script._load_model(run_id, "isolation_forest")
    df_input, X_chem, X_spatial, y_true = predict_script._load_data(
        art["cfg"],
        None,
        require_label=True,
    )
    prediction_dir = outputs_root / "isolation_forest" / run_id / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    predict_script._run_split(
        "test",
        art["splits"]["test_idx"],
        df_input,
        X_chem,
        X_spatial,
        y_true,
        art["scaler"],
        model,
        art["le"],
        "isolation_forest",
        prediction_dir,
        art["art_dir"],
    )

    predictions = pd.read_csv(prediction_dir / "predictions_test.csv")

    assert predictions.shape[0] == len(art["splits"]["test_idx"])
    assert {"anomaly_score", "is_anomaly"}.issubset(predictions.columns)


def test_autoencoder_pipeline_preserves_expected_shapes(
    preprocess_script,
    predict_script,
    tmp_path,
    monkeypatch,
) -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    raw_path = fixture_dir / "spatial_raw_sample.csv"
    expected_path = fixture_dir / "spatial_expected_processed.csv"
    processed_path = tmp_path / "spatial_processed.csv"
    outputs_root = _configure_runtime_roots(tmp_path, monkeypatch, predict_script)

    _configure_preprocess_module(
        preprocess_script,
        data_path=raw_path,
        label_col="rock_name",
        spatial=True,
        feature_columns=SPATIAL_FEATURE_COLUMNS,
        scale_features=False,
    )
    preprocess_script.preprocess(raw_path, processed_path)

    assert _csv_text(processed_path) == _csv_text(expected_path)

    monkeypatch.setattr(trainer_module, "_mlflow_log_keras_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "_save_keras_model", lambda *args, **kwargs: None)

    df, data_options = load_dataset_frame(
        {
            "data_path": str(processed_path),
            "feature_columns": SPATIAL_FEATURE_COLUMNS,
            "label": "label",
            "latitude": "lat",
            "longitude": "long",
            "scale_features": False,
        },
        DEFAULT_SPATIAL_DATA,
    )
    X_all = df[data_options["feature_columns"]].to_numpy(dtype=np.float32)
    X_spatial = df[[data_options["latitude"], data_options["longitude"]]].to_numpy(dtype=np.float32)
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(df[data_options["label_col"]].values)
    splits = make_splits(X_all, y_all, random_state=0)
    scaler = IdentityScaler().fit(X_all[splits["train_idx"]])
    dataset_info = {
        "dataset": str(processed_path),
        "feature_cols": data_options["feature_columns"],
        "label_col": data_options["label_col"],
        "n_samples": len(X_all),
        "spatial": True,
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": None,
        "scale_features": False,
    }

    detector, pr_auc, run_id = trainer_module.train_autoencoder(
        X_all,
        y_all,
        splits,
        label_encoder,
        scaler,
        dataset_info,
        X_spatial=X_spatial,
        params={
            "encoding_dim": 2,
            "hidden_dims": [8],
            "dropout_rate": 0.0,
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 2,
            "validation_split": 0.1,
            "patience": 1,
            "spatial": True,
        },
        evaluation=None,
        experiment_name="pytest_autoencoder",
        run_name="pytest_autoencoder",
        cfg={
            "data": {
                "data_path": str(processed_path),
                "feature_columns": list(data_options["feature_columns"]),
                "label": data_options["label_col"],
                "longitude": data_options["longitude"],
                "latitude": data_options["latitude"],
                "normalize_by": None,
                "scale_features": False,
            },
            "model": {},
            "training": {"spatial": True},
            "evaluation": {},
        },
    )

    assert pr_auc is None
    assert detector.anomaly_scores(X_all, X_spatial).shape == (len(df),)

    art = predict_script._load_artefacts(run_id, "autoencoder")
    model = detector.model
    df_input, X_chem, X_spatial_loaded, y_true = predict_script._load_data(
        art["cfg"],
        None,
        require_label=True,
    )
    prediction_dir = outputs_root / "autoencoder" / run_id / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    predict_script._run_split(
        "test",
        art["splits"]["test_idx"],
        df_input,
        X_chem,
        X_spatial_loaded,
        y_true,
        art["scaler"],
        model,
        art["le"],
        "autoencoder",
        prediction_dir,
        art["art_dir"],
    )

    predictions = pd.read_csv(prediction_dir / "predictions_test.csv")

    assert predictions.shape[0] == len(art["splits"]["test_idx"])
    assert {"anomaly_score", "is_anomaly"}.issubset(predictions.columns)