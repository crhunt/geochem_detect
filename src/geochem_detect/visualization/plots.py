"""Visualisation utilities: PR curves, confusion matrices, spatial maps."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize


def plot_pr_curve_binary(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a binary PR curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, scores, ax=ax, name=title)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pr_curves_multiclass(
    y_true: np.ndarray,
    proba: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-class PR curves for a multi-class classifier."""
    n_classes = len(class_names)
    classes = np.arange(n_classes)
    y_bin = label_binarize(y_true, classes=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        col = y_bin[:, i] if n_classes > 2 else y_true
        sc = proba[:, i]
        ap = average_precision_score(col, sc)
        prec, rec, _ = precision_recall_curve(col, sc)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.2f})", linewidth=1.2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Per-class Precision-Recall Curves")
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(cm)), max(5, len(cm) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title("Normalised Confusion Matrix")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_anomaly_scores_histogram(
    scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Anomaly Score Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of anomaly scores with optional threshold line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=60, edgecolor="none", alpha=0.75, color="steelblue")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold={threshold:.3f}")
        ax.legend()
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_spatial_anomalies(
    gdf,
    scores: np.ndarray,
    threshold: float = 0.7,
    title: str = "Spatial Anomaly Map",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot anomaly scores on a map using a GeoDataFrame."""
    import geopandas as gpd  # noqa: F401

    fig, ax = plt.subplots(figsize=(10, 7))
    normal = gdf[scores < threshold]
    anomaly = gdf[scores >= threshold]

    normal.plot(ax=ax, color="steelblue", markersize=4, alpha=0.5, label="Normal")
    anomaly.plot(ax=ax, color="red", markersize=8, alpha=0.9, label="Anomaly")

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_class_distribution(
    y: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Class Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart of class counts."""
    classes, counts = np.unique(y, return_counts=True)
    labels = [class_names[c] for c in classes] if class_names is not None else classes.astype(str)
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), 4))
    bars = ax.bar(labels, counts, color=sns.color_palette("tab10", len(classes)))
    ax.bar_label(bars, padding=2, fontsize=8)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
