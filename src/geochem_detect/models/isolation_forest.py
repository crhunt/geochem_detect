"""Isolation Forest anomaly detector with MLFlow-friendly interface."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score


class IsolationForestDetector:
    """Thin wrapper around sklearn IsolationForest.

    Anomaly score: higher value → more anomalous.
    Binary label convention: 1 = anomaly, 0 = normal.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        max_features: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._model = IsolationForest(**self.params)

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        self._model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        raw = self._model.predict(X)  # sklearn: -1 anomaly, +1 normal
        return ((raw == -1)).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1] (higher → more anomalous)."""
        # decision_function: more negative → more anomalous
        raw = self._model.decision_function(X)
        # Normalise to [0, 1] with higher = more anomalous
        shifted = -raw  # flip sign
        mn, mx = shifted.min(), shifted.max()
        if mx > mn:
            return (shifted - mn) / (mx - mn)
        return np.zeros_like(shifted)

    def pr_auc(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Compute PR-AUC treating anomaly label=1 as the positive class."""
        scores = self.anomaly_scores(X)
        return float(average_precision_score(y_true, scores))
