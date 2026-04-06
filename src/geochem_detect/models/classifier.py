"""Imbalance-aware multi-class classifier optimised for PR-AUC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight


def build_classifier(
    n_features: int,
    n_classes: int,
    hidden_dims: tuple[int, ...] = (64, 32),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    """Build and compile a dense classifier."""
    import tensorflow as tf

    inp = tf.keras.Input(shape=(n_features,), name="features")
    x = inp
    for dim in hidden_dims:
        x = tf.keras.layers.Dense(dim, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class GeochemClassifier:
    """Multi-class classifier with class-weight balancing for rare classes."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        class_names: np.ndarray | None = None,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.15,
        patience: int = 15,
    ) -> None:
        self.n_classes = n_classes
        self.class_names = class_names
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.params = dict(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dims=list(hidden_dims),
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        self.model = build_classifier(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )
        self.history_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "GeochemClassifier":
        import tensorflow as tf

        weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
        class_weight = dict(enumerate(weights))

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ]

        fit_kwargs: dict = dict(
            epochs=self.epochs,
            batch_size=self.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )
        if X_val is not None and y_val is not None:
            fit_kwargs["validation_data"] = (X_val, y_val)
        else:
            fit_kwargs["validation_split"] = self.validation_split

        self.history_ = self.model.fit(X, y, **fit_kwargs)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities (n_samples, n_classes)."""
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    def pr_auc_macro(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Macro-averaged PR-AUC across all classes."""
        proba = self.predict_proba(X)
        classes = np.arange(self.n_classes)
        y_bin = label_binarize(y_true, classes=classes)
        if self.n_classes == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        return float(
            average_precision_score(y_bin, proba, average="macro")
        )
