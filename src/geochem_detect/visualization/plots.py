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
    title: str = "Per-class Precision-Recall Curves",
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
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Normalised Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(max(6, len(cm)), max(5, len(cm) - 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title)
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
    window_deg: float | None = None,
    raw_gdf=None,
    raw_y: np.ndarray | None = None,
) -> plt.Figure:
    """Plot anomaly scores on a map using a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame aligned with *scores* (same row order).  Each row
        represents ONE window (its centre point), not an individual data point.
    scores:
        Continuous anomaly scores in [0, 1].
    threshold:
        Score cutoff above which a point is flagged as anomalous by the model.
        Should be ``mean(scores) + sigma_cutoff * std(scores)``.
    y_true:
        Optional binary ground-truth labels (1 = truly anomalous by training
        criteria, 0 = normal).  Labels are at the **window** level; a window
        is anomalous when the fraction of its constituent data points with rare
        rock-type labels meets the configured ``anomaly_fraction_threshold``.
        When provided, window-level prediction statistics (accuracy, TP, FP, FN)
        are shown on the plot.
    window_deg:
        Side length of the sampling window in decimal degrees.  When given,
        a single representative window boundary is drawn as a dashed rectangle
        to convey the spatial scale of each sample.
    raw_gdf:
        Optional GeoDataFrame of the original individual data points (not window
        centres).  When provided together with *raw_y*, each point is plotted as
        a small dot coloured by its ground-truth anomaly status.
    raw_y:
        Per-point binary ground-truth labels aligned with *raw_gdf* rows
        (1 = anomalous, 0 = normal).
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    mask_anomaly = scores >= threshold

    # Layer 1: original data points coloured by ground-truth anomaly status
    if raw_gdf is not None and raw_y is not None:
        raw_y_arr = np.asarray(raw_y, dtype=int)
        raw_normal  = raw_gdf[raw_y_arr == 0]
        raw_anomaly = raw_gdf[raw_y_arr == 1]
        if len(raw_normal) > 0:
            ax.scatter(
                raw_normal.geometry.x.values,
                raw_normal.geometry.y.values,
                s=8, color="steelblue", alpha=0.4,
                label="Normal (ground truth)", zorder=2,
            )
        if len(raw_anomaly) > 0:
            ax.scatter(
                raw_anomaly.geometry.x.values,
                raw_anomaly.geometry.y.values,
                s=8, color="firebrick", alpha=0.6,
                label="Anomaly (ground truth)", zorder=3,
            )

    # Layer 2: model-predicted anomalous window centres as transparent red diamonds
    det_anomaly = gdf[mask_anomaly]
    if len(det_anomaly) > 0:
        ax.scatter(
            det_anomaly.geometry.x.values,
            det_anomaly.geometry.y.values,
            marker="D", s=60, facecolors="none", edgecolors="red", linewidths=1.2,
            alpha=0.9, label="Anomaly detected (model)", zorder=5,
        )

    # Window-level prediction statistics (requires window ground-truth labels)
    if y_true is not None:
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

    # Draw a dashed bounding box for one representative window
    if window_deg is not None and len(gdf) > 0:
        import matplotlib.patches as mpatches
        half = window_deg / 2.0
        cx = float(gdf.geometry.x.iloc[0])
        cy = float(gdf.geometry.y.iloc[0])
        rect = mpatches.Rectangle(
            (cx - half, cy - half), window_deg, window_deg,
            linewidth=1.5, edgecolor="darkorange", facecolor="none",
            linestyle="--", label=f"Sample window ({window_deg}°)", zorder=6,
        )
        ax.add_patch(rect)

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
