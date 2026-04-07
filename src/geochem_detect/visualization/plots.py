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
    sigma_cutoff: float = 2.0,
    title: str = "Anomaly Score Distribution",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of anomaly scores with per-sigma reference lines and cutoff.

    Vertical grey dotted lines are drawn at each whole-σ above and below the
    mean.  A red dashed line marks the anomaly cutoff at
    ``mean + sigma_cutoff * std``.  All statistics are derived from *scores*.

    Parameters
    ----------
    scores:
        1-D array of normalised anomaly scores.
    sigma_cutoff:
        Number of standard deviations above the mean used as the anomaly
        decision boundary.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=60, edgecolor="none", alpha=0.75, color="steelblue")

    mean = float(np.mean(scores))
    std  = float(np.std(scores))
    cutoff = mean + sigma_cutoff * std

    # Grey dotted lines at each whole-sigma within the visible range
    max_sigma = int(np.ceil(abs(sigma_cutoff))) + 1
    for s in range(1, max_sigma + 1):
        for sign in (1, -1):
            x = mean + sign * s * std
            if scores.min() <= x <= scores.max():
                ax.axvline(
                    x,
                    color="grey", linestyle=":", linewidth=0.9, alpha=0.7,
                    label=f"{'+' if sign > 0 else '-'}{s}σ",
                )

    # Mean and cutoff
    ax.axvline(mean, color="black", linestyle="-", linewidth=1.0, alpha=0.7,
               label=f"Mean = {mean:.3f}")
    ax.axvline(cutoff, color="red", linestyle="--", linewidth=1.5,
               label=f"Cutoff (mean + {sigma_cutoff:.1f}σ = {cutoff:.3f})")

    # Deduplicate legend entries (sigma labels can repeat for ±)
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique: list = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique.append((h, l))
    if unique:
        ax.legend(*zip(*unique), fontsize=8)

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
    y_true: np.ndarray | None = None,
    title: str = "Spatial Anomaly Map",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot anomaly scores on a map using a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame aligned with *scores* (same row order).
    scores:
        Continuous anomaly scores in [0, 1].
    threshold:
        Score cutoff above which a point is flagged as anomalous by the model.
        Should be ``mean(scores) + sigma_cutoff * std(scores)``.
    y_true:
        Optional binary ground-truth labels (1 = truly anomalous by training
        criteria, 0 = normal).  When provided, ground-truth anomalies are
        overlaid as large unfilled diamonds and prediction statistics
        (accuracy, TP, FP, FN) are shown on the plot.
    """
    import geopandas as gpd  # noqa: F401

    fig, ax = plt.subplots(figsize=(10, 7))
    mask_anomaly = scores >= threshold

    normal  = gdf[~mask_anomaly]
    anomaly = gdf[mask_anomaly]

    normal.plot(ax=ax, color="steelblue", markersize=4, alpha=0.5, label="Normal (model)")
    anomaly.plot(ax=ax, color="red", markersize=8, alpha=0.9, label="Anomaly (model)")

    # Overlay ground-truth anomalies as unfilled diamonds
    if y_true is not None:
        gt_mask = np.asarray(y_true, dtype=bool)
        gt_anom = gdf[gt_mask]
        if len(gt_anom):
            ax.scatter(
                gt_anom.geometry.x.values,
                gt_anom.geometry.y.values,
                marker="D", s=80,
                facecolors="none", edgecolors="black", linewidths=0.8,
                alpha=0.85, label="Anomaly (true label)", zorder=5,
            )

        # Compute and display prediction statistics
        y_pred = mask_anomaly.astype(int)
        y_t    = np.asarray(y_true, dtype=int)
        tp = int(((y_pred == 1) & (y_t == 1)).sum())
        fp = int(((y_pred == 1) & (y_t == 0)).sum())
        fn = int(((y_pred == 0) & (y_t == 1)).sum())
        tn = int(((y_pred == 0) & (y_t == 0)).sum())
        accuracy = (tp + tn) / len(y_t) if len(y_t) else 0.0
        stats_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}"
        )
        ax.text(
            0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )

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
