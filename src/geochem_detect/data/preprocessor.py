"""Preprocessing utilities: scaling, encoding, train/val/test splits."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


class IdentityScaler:
    """Pickle-friendly no-op scaler with a sklearn-like interface."""

    def fit(self, X: np.ndarray) -> "IdentityScaler":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


def prepare_labeled_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    normalize_by: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a cleaned labeled frame aligned to the model feature matrix."""
    cleaned = df.copy().reset_index(drop=True)
    numeric_cols = list(feature_cols)
    if normalize_by is not None:
        if normalize_by not in cleaned.columns:
            raise KeyError(f"Normalization column '{normalize_by}' not found in data")
        if normalize_by not in numeric_cols:
            numeric_cols.append(normalize_by)

    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    required = list(feature_cols)
    if normalize_by is not None:
        required.append(normalize_by)
    cleaned = cleaned.dropna(subset=required)

    effective_feature_cols = [col for col in feature_cols if col != normalize_by]
    if normalize_by is not None:
        non_zero_mask = cleaned[normalize_by] != 0
        dropped = int((~non_zero_mask).sum())
        if dropped:
            warnings.warn(
                f"Dropped {dropped} rows with zero '{normalize_by}' while normalizing features.",
                UserWarning,
                stacklevel=2,
            )
        cleaned = cleaned[non_zero_mask].copy()
        for col in effective_feature_cols:
            cleaned[col] = cleaned[col] / cleaned[normalize_by]

    if not effective_feature_cols:
        raise ValueError("No feature columns remain after applying normalization settings")

    return cleaned.reset_index(drop=True), effective_feature_cols


def split_features_labels(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
    normalize_by: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_encoded, class_names, cleaned_indices) from a DataFrame.

    Non-numeric sentinels and NaN feature rows are dropped. When ``normalize_by``
    is provided, feature columns are divided by that column value per row and the
    normalizing column is excluded from the returned feature matrix.
    """
    df, effective_feature_cols = prepare_labeled_frame(
        df,
        feature_cols,
        label_col=label_col,
        normalize_by=normalize_by,
    )

    orig_idx = np.arange(len(df), dtype=int)
    X = df[effective_feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    return X, y, le.classes_, orig_idx


def scale_features(
    X_train: np.ndarray,
    *extra: np.ndarray | None,
    enabled: bool = True,
) -> tuple[list[np.ndarray | None], RobustScaler | IdentityScaler]:
    """Fit an optional scaler on X_train and transform any additional arrays.

    Usage::

        (X_train_s, X_val_s, X_test_s), scaler = scale_features(X_train, X_val, X_test)
    """
    scaler = RobustScaler() if enabled else IdentityScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    extras_s = [scaler.transform(a).astype(np.float32) if a is not None else None for a in extra]
    return [X_train_s] + extras_s, scaler


def _can_stratify(y: np.ndarray) -> bool:
    _, counts = np.unique(y, return_counts=True)
    return bool(counts.min() >= 2)


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    orig_idx: np.ndarray | None = None,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Three-way stratified split: train / val / test.

    Returns a dict with keys:
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_idx, val_idx, test_idx   (positional indices into the cleaned array,
                                        or into orig_idx when provided)

    Falls back to random splits when any class has < 2 members.
    """
    idx = np.arange(len(X)) if orig_idx is None else orig_idx

    strat = y if _can_stratify(y) else None
    if strat is None:
        warnings.warn(
            "Falling back to non-stratified split: some classes have < 2 samples.",
            UserWarning,
            stacklevel=2,
        )

    # First carve out test set
    X_tv, X_test, y_tv, y_test, idx_tv, idx_test = train_test_split(
        X, y, idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    # Split the remainder into train + val
    strat_tv = y_tv if _can_stratify(y_tv) else None
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_tv, y_tv, idx_tv,
        test_size=relative_val,
        random_state=random_state,
        stratify=strat_tv,
    )

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        train_idx=idx_train, val_idx=idx_val, test_idx=idx_test,
    )


def rare_class_mask(y: np.ndarray, min_count: int = 30) -> np.ndarray:
    """Return a boolean mask marking samples whose class has < min_count samples."""
    classes, counts = np.unique(y, return_counts=True)
    rare = classes[counts < min_count]
    return np.isin(y, rare)

