"""CNN + Sparse Autoencoder (CNN-SAE) for spatial anomaly detection.

Architecture
------------
The model accepts a (grid_size × grid_size × (n_features + 1)) tensor where
the last channel is an **occupancy mask** (1 = cell contains data, 0 = empty).

Encoder:
  Conv2D(filters[0])  → Conv2D(filters[1], stride=2)  → Flatten
  → Dense(hidden[0])  → Dense(hidden[1])
  → Dense(encoding_dim, relu, L1-activity-regularizer)   ← SAE sparsity

Decoder:
  Dense(hidden[1]) → Dense(hidden[0])
  → Dense(H_reduced × W_reduced × filters[-1]) → Reshape
  → Conv2DTranspose(filters[0], stride=2) → Conv2D(n_features, kernel=1)

The **custom masked MSE loss** only penalises reconstruction error over grid
cells that contain data (i.e. occupancy_mask == 1), which avoids driving the
model toward reconstructing empty regions.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def build_cnn_sae(
    grid_size: int,
    n_features: int,
    cnn_filters: tuple[int, int] = (32, 64),
    cnn_kernel_size: int = 3,
    encoding_dim: int = 64,
    dense_hidden_dims: tuple[int, int] = (256, 128),
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
    sparsity_weight: float = 1e-4,
):
    """Build and compile the CNN-SAE model.

    Parameters
    ----------
    grid_size:
        Spatial resolution of each input window (H = W = grid_size).
    n_features:
        Number of geochemical feature channels (occupancy mask is channel n_features).
    cnn_filters:
        Number of filters for the two Conv2D encoder layers.
    cnn_kernel_size:
        Kernel size used for all Conv2D / Conv2DTranspose layers.
    encoding_dim:
        Dimension of the sparse bottleneck layer.
    dense_hidden_dims:
        Sizes of the two Dense layers bridging the CNN and the bottleneck.
    dropout_rate:
        Dropout applied after Dense hidden layers.
    learning_rate:
        Adam learning rate.
    sparsity_weight:
        L1 regularisation coefficient on the bottleneck activations (the SAE
        sparsity penalty).

    Returns
    -------
    tf.keras.Model compiled with the custom masked MSE loss.
    """
    import tensorflow as tf

    # ── Input ────────────────────────────────────────────────────────────────
    inp = tf.keras.Input(
        shape=(grid_size, grid_size, n_features + 1), name="grid_input"
    )
    # Encoder receives only the feature channels; occupancy mask is in y_true
    features_in = inp[:, :, :, :n_features]   # (B, H, W, C)

    # ── Encoder ──────────────────────────────────────────────────────────────
    x = tf.keras.layers.Conv2D(
        cnn_filters[0], cnn_kernel_size, padding="same", activation="relu",
        name="enc_conv1",
    )(features_in)
    x = tf.keras.layers.Conv2D(
        cnn_filters[1], cnn_kernel_size, strides=2, padding="same",
        activation="relu", name="enc_conv2",
    )(x)
    # Compute spatial dims after stride-2 conv (floor division)
    reduced = (grid_size + 1) // 2
    x = tf.keras.layers.Flatten(name="enc_flatten")(x)
    x = tf.keras.layers.Dense(dense_hidden_dims[0], activation="relu", name="enc_dense1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="enc_drop1")(x)
    x = tf.keras.layers.Dense(dense_hidden_dims[1], activation="relu", name="enc_dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="enc_drop2")(x)

    # Bottleneck with L1 sparsity (the "Sparse" in SAE)
    encoded = tf.keras.layers.Dense(
        encoding_dim,
        activation="relu",
        activity_regularizer=tf.keras.regularizers.L1(sparsity_weight),
        name="bottleneck",
    )(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    y = tf.keras.layers.Dense(dense_hidden_dims[1], activation="relu", name="dec_dense1")(encoded)
    y = tf.keras.layers.Dropout(dropout_rate, name="dec_drop1")(y)
    y = tf.keras.layers.Dense(dense_hidden_dims[0], activation="relu", name="dec_dense2")(y)
    y = tf.keras.layers.Dropout(dropout_rate, name="dec_drop2")(y)
    y = tf.keras.layers.Dense(
        reduced * reduced * cnn_filters[-1], activation="relu", name="dec_project"
    )(y)
    y = tf.keras.layers.Reshape(
        (reduced, reduced, cnn_filters[-1]), name="dec_reshape"
    )(y)
    y = tf.keras.layers.Conv2DTranspose(
        cnn_filters[0], cnn_kernel_size, strides=2, padding="same",
        activation="relu", name="dec_deconv1",
    )(y)
    # Crop back to exact grid_size if stride-2 upsampling overshoots
    reduced = (grid_size + 1) // 2
    crop = reduced * 2 - grid_size
    if crop > 0:
        y = tf.keras.layers.Cropping2D(
            cropping=((0, crop), (0, crop)), name="dec_crop"
        )(y)
    reconstruction = tf.keras.layers.Conv2D(
        n_features, 1, padding="same", activation="linear", name="reconstruction"
    )(y)

    # ── Model ────────────────────────────────────────────────────────────────
    model = tf.keras.Model(inputs=inp, outputs=reconstruction, name="cnn_sae")

    # ── Masked MSE loss ──────────────────────────────────────────────────────
    # The loss receives y_true = X (full tensor, n_features+1 channels) and
    # y_pred = reconstruction (n_features channels).  It extracts the
    # occupancy mask from y_true[:,:,:, n_features] so that only occupied cells
    # contribute to the error.  n_features is a Python int captured from the
    # enclosing scope — no KerasTensors involved in the closure.
    _n_feat = n_features  # plain Python int

    def masked_mse(y_true, y_pred):
        """MSE over occupied cells only.  y_true includes the occupancy mask."""
        feat_true = y_true[:, :, :, :_n_feat]
        mask = y_true[:, :, :, _n_feat : _n_feat + 1]  # (B, H, W, 1)
        per_cell = tf.reduce_mean(
            tf.square(feat_true - y_pred), axis=-1, keepdims=True
        )
        masked = per_cell * mask
        # Average only over cells that have data
        n_occ = tf.maximum(tf.reduce_sum(mask, axis=[1, 2, 3]), 1.0)
        loss = tf.reduce_sum(masked, axis=[1, 2, 3]) / n_occ
        return tf.reduce_mean(loss)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=masked_mse,
    )
    return model


class CnnSaeDetector:
    """Anomaly detector based on CNN-SAE reconstruction error.

    Interface mirrors ``AutoencoderDetector`` for consistency with the rest of
    the geochem_detect pipeline.

    Parameters
    ----------
    grid_size, n_features, cnn_filters, cnn_kernel_size, encoding_dim,
    dense_hidden_dims, dropout_rate, learning_rate, sparsity_weight:
        Forwarded to :func:`build_cnn_sae`.
    epochs, batch_size, patience:
        Training loop configuration.
    """

    def __init__(
        self,
        grid_size: int = 16,
        n_features: int = 10,
        cnn_filters: tuple[int, ...] | list[int] = (32, 64),
        cnn_kernel_size: int = 3,
        encoding_dim: int = 64,
        dense_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        dropout_rate: float = 0.2,
        learning_rate: float = 1e-3,
        sparsity_weight: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        self.grid_size = grid_size
        self.n_features = n_features
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.params = dict(
            grid_size=grid_size,
            n_features=n_features,
            cnn_filters=list(cnn_filters),
            cnn_kernel_size=cnn_kernel_size,
            encoding_dim=encoding_dim,
            dense_hidden_dims=list(dense_hidden_dims),
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            sparsity_weight=sparsity_weight,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
        )
        self.model = build_cnn_sae(
            grid_size=grid_size,
            n_features=n_features,
            cnn_filters=tuple(cnn_filters),
            cnn_kernel_size=cnn_kernel_size,
            encoding_dim=encoding_dim,
            dense_hidden_dims=tuple(dense_hidden_dims),
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            sparsity_weight=sparsity_weight,
        )
        self.history_ = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        X_val: np.ndarray | None = None,
    ) -> "CnnSaeDetector":
        """Train the model.

        Parameters
        ----------
        X:
            Training grids, shape (n, grid_size, grid_size, n_features+1).
            The full X is passed as both input *and* target: the loss function
            extracts the feature channels and occupancy mask from y_true.
        X_val:
            Optional validation grids with the same shape.
        """
        import tensorflow as tf

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
            )
        ]

        # Pass X as both input and target; loss extracts mask from y_true
        fit_kwargs: dict = dict(
            x=X,
            y=X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        if X_val is not None:
            fit_kwargs["validation_data"] = (X_val, X_val)
        else:
            fit_kwargs["validation_split"] = 0.1

        self.history_ = self.model.fit(**fit_kwargs)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Masked MSE reconstruction error per sample.

        Only occupied cells (occupancy_mask == 1) contribute to each sample's
        error, preventing empty regions from diluting the anomaly signal.
        """
        X_feat = X[:, :, :, : self.n_features]
        occ = X[:, :, :, self.n_features]  # (n, H, W)
        preds = self.model.predict(X, verbose=0)  # (n, H, W, C)
        per_cell = np.mean((X_feat - preds) ** 2, axis=-1)  # (n, H, W)
        masked = per_cell * occ
        n_occ = np.maximum(occ.sum(axis=(1, 2)), 1.0)
        return (masked.sum(axis=(1, 2)) / n_occ).astype(np.float32)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Normalised anomaly scores in [0, 1] — higher means more anomalous."""
        errors = self.reconstruction_errors(X)
        mn, mx = errors.min(), errors.max()
        if mx > mn:
            return (errors - mn) / (mx - mn)
        return np.zeros_like(errors)

    def pr_auc(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """PR-AUC treating anomaly=1 as the positive class."""
        scores = self.anomaly_scores(X)
        return float(average_precision_score(y_true, scores))

    def is_anomaly(
        self,
        X: np.ndarray,
        sigma_cutoff: float = 2.0,
    ) -> np.ndarray:
        """Binary anomaly flags using a sigma-based threshold.

        Threshold = mean(scores) + sigma_cutoff × std(scores).
        """
        scores = self.anomaly_scores(X)
        threshold = float(np.mean(scores) + sigma_cutoff * np.std(scores))
        return (scores >= threshold).astype(int)
