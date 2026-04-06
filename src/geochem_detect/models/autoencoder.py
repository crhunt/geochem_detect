"""Spatial-aware autoencoder for anomaly detection."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def build_autoencoder(
    n_features: int,
    n_spatial: int = 0,
    encoding_dim: int = 4,
    hidden_dims: tuple[int, ...] = (32, 16),
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
):
    """Build and compile a (spatial-aware) autoencoder.

    Parameters
    ----------
    n_features:
        Number of geochemical features.
    n_spatial:
        Number of spatial features (lat, lon encoded). Set 0 to disable.
    encoding_dim:
        Bottleneck dimension.
    """
    import tensorflow as tf
    from tensorflow import keras  # noqa: F401 – ensure keras registers

    # Geochemical encoder
    chem_input = tf.keras.Input(shape=(n_features,), name="chem_input")
    x = chem_input
    for dim in hidden_dims:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if n_spatial > 0:
        spatial_input = tf.keras.Input(shape=(n_spatial,), name="spatial_input")
        x = tf.keras.layers.Concatenate()([x, spatial_input])

    encoded = tf.keras.layers.Dense(encoding_dim, activation="relu", name="encoded")(x)

    # Decoder reconstructs geochemical features only
    y = encoded
    for dim in reversed(hidden_dims):
        y = tf.keras.layers.Dense(dim, activation="relu")(y)
        y = tf.keras.layers.Dropout(dropout_rate)(y)
    decoded = tf.keras.layers.Dense(n_features, activation="linear", name="decoded")(y)

    inputs = [chem_input] if n_spatial == 0 else [chem_input, spatial_input]
    model = tf.keras.Model(inputs=inputs, outputs=decoded, name="autoencoder")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="mse",
    )
    return model


class AutoencoderDetector:
    """Anomaly detector using reconstruction error from a Keras autoencoder."""

    def __init__(
        self,
        n_features: int,
        n_spatial: int = 0,
        encoding_dim: int = 4,
        hidden_dims: tuple[int, ...] = (32, 16),
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.1,
        patience: int = 10,
    ) -> None:
        self.n_features = n_features
        self.n_spatial = n_spatial
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.params = dict(
            n_features=n_features,
            n_spatial=n_spatial,
            encoding_dim=encoding_dim,
            hidden_dims=list(hidden_dims),
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        self.model = build_autoencoder(
            n_features=n_features,
            n_spatial=n_spatial,
            encoding_dim=encoding_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        self.history_ = None

    def _prepare_inputs(
        self, X_chem: np.ndarray, X_spatial: np.ndarray | None = None
    ):
        if self.n_spatial > 0 and X_spatial is not None:
            return [X_chem, X_spatial]
        return X_chem

    def fit(
        self,
        X_chem: np.ndarray,
        X_spatial: np.ndarray | None = None,
        X_chem_val: np.ndarray | None = None,
        X_spatial_val: np.ndarray | None = None,
    ) -> "AutoencoderDetector":
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]
        inputs = self._prepare_inputs(X_chem, X_spatial)
        val_data = None
        if X_chem_val is not None:
            val_inputs = self._prepare_inputs(X_chem_val, X_spatial_val)
            val_data = (val_inputs, X_chem_val)

        fit_kwargs: dict = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if val_data is not None:
            fit_kwargs["validation_data"] = val_data
        else:
            fit_kwargs["validation_split"] = self.validation_split

        self.history_ = self.model.fit(inputs, X_chem, **fit_kwargs)
        return self

    def reconstruction_errors(
        self, X_chem: np.ndarray, X_spatial: np.ndarray | None = None
    ) -> np.ndarray:
        """Mean squared reconstruction error per sample."""
        inputs = self._prepare_inputs(X_chem, X_spatial)
        preds = self.model.predict(inputs, verbose=0)
        return np.mean((X_chem - preds) ** 2, axis=1)

    def anomaly_scores(
        self, X_chem: np.ndarray, X_spatial: np.ndarray | None = None
    ) -> np.ndarray:
        """Normalised anomaly scores in [0, 1]."""
        errors = self.reconstruction_errors(X_chem, X_spatial)
        mn, mx = errors.min(), errors.max()
        if mx > mn:
            return (errors - mn) / (mx - mn)
        return np.zeros_like(errors)

    def pr_auc(
        self,
        X_chem: np.ndarray,
        y_true: np.ndarray,
        X_spatial: np.ndarray | None = None,
    ) -> float:
        """PR-AUC treating anomaly=1 as the positive class."""
        scores = self.anomaly_scores(X_chem, X_spatial)
        return float(average_precision_score(y_true, scores))
