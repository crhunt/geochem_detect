"""Isolation Forest anomaly detector with MLFlow-friendly interface."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score


class IsolationForestDetector:
    """Thin wrapper around sklearn IsolationForest.

    Anomaly score is the raw ``decision_function`` output.
    Lower value → more anomalous, with the standard cutoff at 0.
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
        """Return raw decision_function scores (lower → more anomalous)."""
        return self._model.decision_function(X)

    def pr_auc(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Compute PR-AUC treating anomaly label=1 as the positive class."""
        scores = -self.anomaly_scores(X)
        return float(average_precision_score(y_true, scores))
