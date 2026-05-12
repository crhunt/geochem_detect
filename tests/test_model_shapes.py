from __future__ import annotations

import numpy as np

from geochem_detect.models.autoencoder import AutoencoderDetector
from geochem_detect.models.classifier import GeochemClassifier
from geochem_detect.models.cnn_sae import CnnSaeDetector
from geochem_detect.models.isolation_forest import IsolationForestDetector


def test_isolation_forest_detector_accepts_expected_feature_shape() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 4)).astype(np.float32)
    detector = IsolationForestDetector(n_estimators=16, random_state=0)

    detector.fit(X)

    assert detector.anomaly_scores(X).shape == (8,)


def test_autoencoder_detector_accepts_expected_spatial_shape() -> None:
    X_chem = np.zeros((3, 4), dtype=np.float32)
    X_spatial = np.zeros((3, 2), dtype=np.float32)
    detector = AutoencoderDetector(
        n_features=4,
        n_spatial=2,
        encoding_dim=2,
        hidden_dims=(8,),
        dropout_rate=0.0,
        epochs=1,
        batch_size=2,
        patience=1,
    )

    output = detector.model([X_chem, X_spatial], training=False).numpy()

    assert output.shape == X_chem.shape


def test_autoencoder_detector_accepts_expected_non_spatial_shape() -> None:
    X_chem = np.zeros((3, 4), dtype=np.float32)
    detector = AutoencoderDetector(
        n_features=4,
        n_spatial=0,
        encoding_dim=2,
        hidden_dims=(8,),
        dropout_rate=0.0,
        epochs=1,
        batch_size=2,
        patience=1,
    )

    output = detector.model(X_chem, training=False).numpy()

    assert output.shape == X_chem.shape


def test_classifier_accepts_expected_feature_shape() -> None:
    X = np.zeros((4, 5), dtype=np.float32)
    classifier = GeochemClassifier(
        n_features=5,
        n_classes=3,
        hidden_dims=(8,),
        dropout_rate=0.0,
        epochs=1,
        batch_size=2,
        patience=1,
    )

    output = classifier.predict_proba(X)

    assert output.shape == (4, 3)


def test_cnn_sae_detector_accepts_expected_grid_shape() -> None:
    X = np.zeros((2, 4, 4, 4), dtype=np.float32)
    detector = CnnSaeDetector(
        grid_size=4,
        n_features=3,
        cnn_filters=(4, 8),
        dense_hidden_dims=(16, 8),
        encoding_dim=4,
        dropout_rate=0.0,
        epochs=1,
        batch_size=1,
        patience=1,
    )

    output = detector.model(X, training=False).numpy()

    assert output.shape == (2, 4, 4, 3)