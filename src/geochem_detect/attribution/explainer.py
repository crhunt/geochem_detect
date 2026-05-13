"""Model-agnostic Shapley value computation for geochem anomaly detectors.

Supports all three methods:
  - isolation_forest  (sklearn IsolationForest)
  - autoencoder       (Keras dense autoencoder, optionally spatial)
  - cnn_sae           (Keras CNN-SAE with spatial windows)

The public entry point is :func:`run_attribution`, which accepts the path to a
run directory (``outputs/<method>/<run_id>``) and writes results to
``<run_dir>/attribution/``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml


# ─── Method detection ────────────────────────────────────────────────────────


def detect_method(run_dir: Path) -> str:
    """Infer the anomaly-detection method from the artefacts directory.

    Returns one of ``"isolation_forest"``, ``"autoencoder"``, ``"cnn_sae"``.
    """
    art = run_dir / "artefacts"
    if (art / "model.pkl").exists():
        return "isolation_forest"
    if (art / "keras_model.keras").exists():
        if (art / "sampling_params.json").exists():
            return "cnn_sae"
        return "autoencoder"
    raise ValueError(
        f"Cannot detect method from artefacts at {art}. "
        "Expected 'model.pkl' (isolation_forest) or 'keras_model.keras' "
        "(autoencoder / cnn_sae)."
    )


# ─── Artefact loading ─────────────────────────────────────────────────────────


def load_artefacts(run_dir: Path, method: str) -> dict:
    """Load all serialised artefacts for *method* from *run_dir/artefacts*."""
    art = run_dir / "artefacts"

    scaler = pickle.loads((art / "scaler.pkl").read_bytes())
    le = pickle.loads((art / "label_encoder.pkl").read_bytes())

    with open(art / "dataset_info.json") as f:
        info = json.load(f)
    with open(art / "anomaly_threshold.json") as f:
        threshold_cfg = json.load(f)
    with open(art / "training_config.yml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    result: dict = dict(
        scaler=scaler,
        le=le,
        info=info,
        cfg=cfg,
        threshold_cfg=threshold_cfg,
        art_dir=art,
        method=method,
    )

    if method == "cnn_sae":
        with open(art / "sampling_params.json") as f:
            result["sampling_params"] = json.load(f)
        with open(art / "window_metadata.json") as f:
            result["window_metadata"] = json.load(f)
        result["splits"] = np.load(art / "window_splits.npz")

        import tensorflow as tf
        result["keras_model"] = tf.keras.models.load_model(
            str(art / "keras_model.keras"), compile=False
        )
    elif method == "autoencoder":
        result["splits"] = np.load(art / "splits.npz")

        import tensorflow as tf
        result["keras_model"] = tf.keras.models.load_model(
            str(art / "keras_model.keras"), compile=False
        )
    else:  # isolation_forest
        result["splits"] = np.load(art / "splits.npz")
        result["if_model"] = pickle.loads((art / "model.pkl").read_bytes())

    return result


# ─── Tabular data construction ────────────────────────────────────────────────


_LEGACY_COORD_CANDIDATES = [
    ("latitude", "longitude"),
    ("lat", "long"),
    ("EASTING", "NORTHING"),
]


def _resolve_coord_cols(df: pd.DataFrame, lat: str | None, lon: str | None) -> tuple[str, str]:
    if lat and lon and lat in df.columns and lon in df.columns:
        return lat, lon
    for cand_lat, cand_lon in _LEGACY_COORD_CANDIDATES:
        if cand_lat in df.columns and cand_lon in df.columns:
            return cand_lat, cand_lon
    raise ValueError("Spatial run requires lat/lon columns in dataset but none were found.")


def build_tabular_data(
    run_dir: Path,
    method: str,
    artefacts: dict,
) -> tuple[np.ndarray, np.ndarray | None, list[str], pd.DataFrame]:
    """Build a (n_samples, n_features) tabular matrix suitable for KernelExplainer.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        Scaled feature matrix (the exact values passed to the predict function).
    y : ndarray of shape (n_samples,) or None
        Integer anomaly labels (1=anomaly, 0=normal), if available.
    feature_names : list[str]
        Human-readable feature names aligned with columns of X_scaled.
    meta_df : pd.DataFrame
        Per-sample metadata (original row/window index, coordinates, etc.)
        suitable for enriching the output CSV.
    """
    info = artefacts["info"]
    scaler = artefacts["scaler"]
    cfg = artefacts["cfg"]
    feat_cols: list[str] = cfg["data"]["feature_columns"]

    if method == "cnn_sae":
        return _build_cnn_sae_tabular(run_dir, artefacts, feat_cols)

    # ── Tabular models (isolation_forest, autoencoder) ────────────────────────
    from geochem_detect.data.loader import load_dataset_frame

    data_cfg = dict(cfg["data"])
    label_col = data_cfg.get("label", "label")
    df, data_options = load_dataset_frame(
        data_cfg,
        data_cfg["data_path"],
        require_label=False,
        label_col=label_col,
    )
    df = df.reset_index(drop=True)
    actual_feat_cols: list[str] = data_options["feature_columns"]

    # Gather all split indices (train + val + test)
    splits = artefacts["splits"]
    idx_all = _concat_all_splits(splits)

    df_all = df.iloc[idx_all].reset_index(drop=True)
    X_chem = df_all[actual_feat_cols].to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_chem).astype(np.float32)

    feature_names = list(actual_feat_cols)

    if cfg.get("training", {}).get("spatial", False):
        lat_col, lon_col = _resolve_coord_cols(
            df_all, data_cfg.get("latitude"), data_cfg.get("longitude")
        )
        X_spatial = df_all[[lat_col, lon_col]].to_numpy(dtype=np.float32)
        X_scaled = np.concatenate([X_scaled, X_spatial], axis=1)
        feature_names = feature_names + [lat_col, lon_col]

    actual_label_col = data_options["label_col"]
    y: np.ndarray | None = None
    if actual_label_col in df_all.columns:
        le = artefacts["le"]
        try:
            y = le.transform(df_all[actual_label_col].fillna("").astype(str))
        except Exception:
            # label values may not match encoder classes — ignore
            y = None

    # Build metadata dataframe
    meta_df = df_all.copy()
    meta_df.insert(0, "original_row_idx", idx_all)

    return X_scaled, y, feature_names, meta_df
def _concat_all_splits(splits) -> np.ndarray:
    """Concatenate train/val/test indices from a splits.npz file."""
    parts = []
    for key in ("train_idx", "val_idx", "test_idx"):
        if key in splits:
            parts.append(splits[key])
    if not parts:
        raise ValueError("splits.npz contains none of train_idx/val_idx/test_idx.")
    return np.concatenate(parts).astype(int)


def _build_cnn_sae_tabular(
    run_dir: Path,
    artefacts: dict,
    feat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray | None, list[str], pd.DataFrame]:
    """Build tabular data for CNN-SAE by aggregating windows."""
    cfg = artefacts["cfg"]
    scaler = artefacts["scaler"]
    sp = artefacts["sampling_params"]
    metadata: list[dict] = artefacts["window_metadata"]
    splits = artefacts["splits"]

    from geochem_detect.data.loader import load_spatial_frame

    data_cfg = dict(cfg["data"])
    label_col = data_cfg.get("label", "label")
    gdf, data_options = load_spatial_frame(
        data_cfg,
        data_cfg["data_path"],
        label_col=label_col,
    )
    actual_feat_cols: list[str] = data_options["feature_columns"]
    gdf = gdf.reset_index(drop=True)
    X_raw = gdf[actual_feat_cols].to_numpy(dtype=np.float32)

    from sklearn.preprocessing import LabelEncoder as _LE
    _le = _LE()
    y_raw = _le.fit_transform(gdf[data_options["label_col"]].values)

    # Reconstruct windows for all splits
    idx_all = _concat_all_splits(splits)

    X_windows, y_windows, meta_rows = _reconstruct_windows(
        idx_all, metadata, gdf, X_raw, y_raw, scaler, actual_feat_cols, sp
    )

    # Aggregate each window to (n_features,) via max of occupied cells
    X_tab = _aggregate_windows_max(X_windows, n_features=len(actual_feat_cols))

    meta_df = pd.DataFrame(meta_rows)

    return X_tab.astype(np.float32), y_windows, list(actual_feat_cols), meta_df


def _reconstruct_windows(
    idx: np.ndarray,
    metadata: list[dict],
    gdf_clean,
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    scaler,
    feat_cols: list[str],
    sampling_params: dict,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Rebuild window tensors for the given window indices."""
    from geochem_detect.data.spatial_sampler import SpatialSampler

    contamination = sampling_params.get("contamination", 0.05)
    classes_all, counts_all = np.unique(y_raw, return_counts=True)
    rare = classes_all[counts_all < int(contamination * len(y_raw))]
    point_anomaly = np.isin(y_raw, rare).astype(np.int32)

    X_scaled = scaler.transform(X_raw).astype("float32")
    gdf_scaled = gdf_clean.copy()
    gdf_scaled[feat_cols] = X_scaled

    grid_size = sampling_params.get("grid_size", 16)
    window_deg = sampling_params.get("window_deg", 1.0)
    sampler = SpatialSampler(
        gdf=gdf_scaled,
        feature_cols=feat_cols,
        anomaly_labels=point_anomaly,
        window_deg=window_deg,
        grid_size=grid_size,
        n_samples=1,
    )

    grids, labels, meta_rows = [], [], []
    for i in idx:
        m = metadata[int(i)]
        pt_idx = np.array(m["point_indices"], dtype=int)
        grid = sampler._points_to_grid(pt_idx, m["center_lat"], m["center_lon"])
        label = int(point_anomaly[pt_idx].max()) if len(pt_idx) > 0 else 0
        grids.append(grid)
        labels.append(label)
        meta_rows.append({
            "window_idx": int(i),
            "center_lat": m["center_lat"],
            "center_lon": m["center_lon"],
            "n_points": m.get("n_points", len(pt_idx)),
        })

    return (
        np.stack(grids, axis=0),
        np.array(labels, dtype=np.int32),
        meta_rows,
    )


def _aggregate_windows_max(X_windows: np.ndarray, n_features: int) -> np.ndarray:
    """Aggregate (n, H, W, n_features+1) windows to (n, n_features) via max of occupied cells.

    Only grid cells where the occupancy mask == 1 contribute to the max.
    Unoccupied cells are treated as -inf so they never win the max.
    """
    X_feat = X_windows[:, :, :, :n_features]   # (n, H, W, F)
    occ = X_windows[:, :, :, n_features]        # (n, H, W)

    occ_expanded = occ[:, :, :, np.newaxis]     # (n, H, W, 1)
    masked = np.where(occ_expanded == 1, X_feat, -np.inf)  # (n, H, W, F)
    # Max over spatial dimensions; fall back to 0 when no occupied cell exists
    max_vals = masked.reshape(masked.shape[0], -1, n_features).max(axis=1)  # (n, F)
    return np.where(np.isfinite(max_vals), max_vals, 0.0)


# ─── Predict function construction ────────────────────────────────────────────


def build_predict_fn(
    method: str,
    artefacts: dict,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable (X_tab: ndarray) → 1-D anomaly score array.

    For all methods the score is oriented so that **higher = more anomalous**,
    making SHAP values directly interpretable: a positive SHAP value means the
    feature pushes the sample toward being anomalous.
    """
    info = artefacts["info"]

    if method == "isolation_forest":
        if_model = artefacts["if_model"]

        def _predict_if(X: np.ndarray) -> np.ndarray:
            # Negate so higher → more anomalous
            return -if_model.decision_function(X).astype(np.float64)

        return _predict_if

    if method == "autoencoder":
        keras_model = artefacts["keras_model"]
        n_features = len(info["feature_cols"])
        is_spatial = info.get("spatial", False)

        def _predict_ae(X: np.ndarray) -> np.ndarray:
            X32 = X.astype(np.float32)
            if is_spatial:
                X_chem = X32[:, :n_features]
                X_sp = X32[:, n_features:]
                inputs = [X_chem, X_sp]
            else:
                inputs = X32
            preds = keras_model(inputs, training=False).numpy()
            errors = np.mean((X32[:, :n_features] - preds) ** 2, axis=1)
            mn, mx = errors.min(), errors.max()
            if mx > mn:
                return ((errors - mn) / (mx - mn)).astype(np.float64)
            return errors.astype(np.float64)

        return _predict_ae

    if method == "cnn_sae":
        keras_model = artefacts["keras_model"]
        n_features = len(info["feature_cols"])
        sp = artefacts["sampling_params"]
        grid_size = sp.get("grid_size", 16)

        def _predict_cnn_sae(X_tab: np.ndarray) -> np.ndarray:
            """Expand tabular rows to uniform grids and score them."""
            n = X_tab.shape[0]
            # Build grid tensors: every cell filled with the feature value,
            # occupancy mask = 1 everywhere (we treat tabular aggregates as
            # a synthetic "full" window so the model sees consistent input)
            grids = np.zeros((n, grid_size, grid_size, n_features + 1), dtype=np.float32)
            grids[:, :, :, :n_features] = X_tab[:, np.newaxis, np.newaxis, :]
            grids[:, :, :, n_features] = 1.0  # mask = all occupied

            preds = keras_model(grids, training=False).numpy()  # (n, H, W, F)
            X_feat = grids[:, :, :, :n_features]
            occ = grids[:, :, :, n_features]
            per_cell = np.mean((X_feat - preds) ** 2, axis=-1)
            masked = per_cell * occ
            n_occ = np.maximum(occ.sum(axis=(1, 2)), 1.0)
            errors = (masked.sum(axis=(1, 2)) / n_occ).astype(np.float32)
            mn, mx = errors.min(), errors.max()
            if mx > mn:
                return ((errors - mn) / (mx - mn)).astype(np.float64)
            return errors.astype(np.float64)

        return _predict_cnn_sae

    raise ValueError(f"Unknown method: {method!r}")


# ─── SHAP computation ─────────────────────────────────────────────────────────


def compute_shap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_explain: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    max_background: int = 100,
) -> tuple[np.ndarray, float]:
    """Compute KernelSHAP values for every sample in *X_explain*.

    Parameters
    ----------
    predict_fn:
        Callable mapping (n, n_features) → (n,) anomaly scores.
    X_explain:
        Samples to explain, shape (n_explain, n_features).
    X_train:
        Training samples used to summarise the background distribution.
    feature_names:
        Feature names aligned with the columns of X_explain.
    max_background:
        Maximum number of k-means clusters for the background summary.

    Returns
    -------
    shap_values : ndarray of shape (n_explain, n_features)
    expected_value : float
        The model's expected output over the background dataset.
    """
    import shap

    k = min(max_background, len(X_train))
    background = shap.kmeans(X_train, k)

    explainer = shap.KernelExplainer(predict_fn, background)

    # Suppress shap's verbose progress bar to keep stdout clean
    shap_values = explainer.shap_values(X_explain, silent=True)

    return np.asarray(shap_values), float(explainer.expected_value)


# ─── Output persistence ───────────────────────────────────────────────────────


def save_attribution(
    run_dir: Path,
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    meta_df: pd.DataFrame,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    expected_value: float,
) -> Path:
    """Write SHAP values, summary CSV, and visualisations to *run_dir/attribution/*.

    Returns the path to the output directory.
    """
    out_dir = run_dir / "attribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = predict_fn(X_explain)
    threshold_cfg = {}
    art = run_dir / "artefacts" / "anomaly_threshold.json"
    if art.exists():
        threshold_cfg = json.loads(art.read_text())

    if "cutoff" in threshold_cfg:
        cutoff = float(threshold_cfg["cutoff"])
        # IsolationForest: raw score < cutoff → anomaly; we negated in predict_fn
        is_anomaly = (scores > -cutoff).astype(int)
    elif "threshold" in threshold_cfg:
        is_anomaly = (scores >= float(threshold_cfg["threshold"])).astype(int)
    else:
        sigma = float(threshold_cfg.get("sigma_cutoff", 2.0))
        is_anomaly = (scores >= float(np.mean(scores) + sigma * np.std(scores))).astype(int)

    # Build CSV
    shap_cols = {f"shap_{name}": shap_values[:, i] for i, name in enumerate(feature_names)}
    result_df = meta_df.reset_index(drop=True).copy()
    result_df["anomaly_score"] = scores
    result_df["is_anomaly"] = is_anomaly
    result_df["expected_value"] = expected_value
    for col, vals in shap_cols.items():
        result_df[col] = vals

    csv_path = out_dir / "shap_values.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── Anomaly-grouped bar chart (always) ───────────────────────────────────
    anomaly_group = np.where(is_anomaly == 1, "Anomalous", "Normal")
    _plot_grouped_bar(
        shap_values,
        feature_names,
        group_labels=anomaly_group,
        title="Mean |SHAP| by anomaly label",
        save_path=out_dir / "shap_bar_by_anomaly.png",
    )

    # ── Label-grouped bar chart (only when a label column is available) ──────
    cfg_path = run_dir / "artefacts" / "training_config.yml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as _f:
            _saved_cfg = yaml.safe_load(_f)
        label_col = _saved_cfg.get("data", {}).get("label", "label")
        if label_col in result_df.columns and result_df[label_col].notna().any():
            rock_labels = result_df[label_col].fillna("unknown").astype(str).to_numpy()
            _plot_grouped_bar(
                shap_values,
                feature_names,
                group_labels=rock_labels,
                title="Mean |SHAP| by sample label",
                save_path=out_dir / "shap_bar_by_label.png",
            )

    # ── Beeswarm summary ─────────────────────────────────────────────────────
    _plot_summary(shap_values, X_explain, feature_names, out_dir)

    return out_dir


def _plot_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> None:
    """Beeswarm summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.6 + 1)))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    save_path = out_dir / "shap_summary_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


def _plot_grouped_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    group_labels: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Horizontal grouped bar chart: mean |SHAP| per feature, one bar per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    unique_groups = sorted(set(group_labels), key=str)
    n_features = len(feature_names)
    n_groups = len(unique_groups)

    group_means: dict = {}
    for g in unique_groups:
        mask = group_labels == g
        group_means[g] = (
            np.abs(shap_values[mask]).mean(axis=0) if mask.any() else np.zeros(n_features)
        )

    # Sort features by overall mean |SHAP|; ascending so highest lands at the top
    order = np.argsort(np.abs(shap_values).mean(axis=0))

    bar_height = 0.8 / n_groups
    fig_height = max(4, n_features * max(0.45 * n_groups, 0.5) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    y_base = np.arange(n_features, dtype=float)

    for g_idx, g in enumerate(unique_groups):
        offsets = y_base + (g_idx - (n_groups - 1) / 2.0) * bar_height
        ax.barh(
            offsets,
            group_means[g][order],
            height=bar_height,
            label=str(g),
            color=colors[g_idx % len(colors)],
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_yticks(y_base)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {save_path}")


# ─── Public entry point ───────────────────────────────────────────────────────


def run_attribution(
    run_dir: Path,
    max_background: int = 100,
    max_samples: int | None = None,
) -> Path:
    """Compute and save SHAP attributions for a trained model run.

    Parameters
    ----------
    run_dir:
        Path to the run directory, e.g. ``outputs/isolation_forest/<run_id>``.
    max_background:
        Number of k-means background clusters for KernelExplainer (default 100).
    max_samples:
        If set, randomly subsample this many rows from ``X_all`` before
        computing SHAP values (useful for large datasets).

    Returns
    -------
    Path to the ``attribution/`` output directory.
    """
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    print(f"[attribution] Run directory : {run_dir}")

    method = detect_method(run_dir)
    print(f"[attribution] Detected method: {method}")

    print("[attribution] Loading artefacts …")
    artefacts = load_artefacts(run_dir, method)

    print("[attribution] Building tabular data …")
    X_all, y_all, feature_names, meta_df = build_tabular_data(run_dir, method, artefacts)
    print(f"[attribution] {X_all.shape[0]} samples × {X_all.shape[1]} features")

    if max_samples is not None and max_samples < len(X_all):
        rng = np.random.default_rng(42)
        idx_sub = rng.choice(len(X_all), size=max_samples, replace=False)
        idx_sub.sort()
        X_all = X_all[idx_sub]
        meta_df = meta_df.iloc[idx_sub].reset_index(drop=True)
        if y_all is not None:
            y_all = y_all[idx_sub]
        print(f"[attribution] Subsampled to {len(X_all)} samples (--max-samples {max_samples})")

    print("[attribution] Building predict function …")
    predict_fn = build_predict_fn(method, artefacts)

    # Use all available data as both background and explanation set
    splits = artefacts["splits"]
    train_idx = splits.get("train_idx", np.arange(len(X_all)))
    # X_all contains concatenated train/val/test; for CNN-SAE indices into X_all
    # aren't meaningful as split indices (windows were already filtered), so
    # fall back to using all rows as background
    n_train = len(train_idx) if len(train_idx) <= len(X_all) else len(X_all)
    X_train = X_all[:n_train]

    print(f"[attribution] Computing KernelSHAP (background={min(max_background, n_train)}, "
          f"explain={len(X_all)}) — this may take several minutes …")
    shap_values, expected_value = compute_shap(
        predict_fn,
        X_explain=X_all,
        X_train=X_train,
        feature_names=feature_names,
        max_background=max_background,
    )
    print(f"[attribution] expected_value = {expected_value:.4f}")

    print("[attribution] Saving outputs …")
    out_dir = save_attribution(
        run_dir=run_dir,
        shap_values=shap_values,
        X_explain=X_all,
        feature_names=feature_names,
        meta_df=meta_df,
        predict_fn=predict_fn,
        expected_value=expected_value,
    )
    print(f"[attribution] Done. Results in {out_dir}")
    return out_dir


def regenerate_plots(run_dir: Path) -> Path:
    """Regenerate attribution plots from a previously saved ``shap_values.csv``.

    Reads ``<run_dir>/attribution/shap_values.csv``, reconstructs the SHAP
    value matrix and group labels, then overwrites the plot files in-place.
    The CSV itself is not modified.

    Parameters
    ----------
    run_dir:
        Path to the run directory, e.g. ``outputs/isolation_forest/<run_id>``.

    Returns
    -------
    Path to the ``attribution/`` output directory.
    """
    run_dir = Path(run_dir).resolve()
    csv_path = run_dir / "attribution" / "shap_values.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No shap_values.csv found at {csv_path}.\n"
            "Run 'make attribution' first to compute SHAP values."
        )

    print(f"[attribution] Loading {csv_path} …")
    df = pd.read_csv(csv_path)

    # Recover feature names from 'shap_<feature>' columns
    shap_cols = [c for c in df.columns if c.startswith("shap_")]
    if not shap_cols:
        raise ValueError("shap_values.csv contains no 'shap_*' columns.")
    feature_names = [c[len("shap_"):] for c in shap_cols]
    shap_values = df[shap_cols].to_numpy(dtype=np.float32)
    X_explain = df[feature_names].to_numpy(dtype=np.float32) if all(
        f in df.columns for f in feature_names
    ) else shap_values  # fallback: use SHAP values as proxy for feature values

    out_dir = csv_path.parent

    # ── Anomaly-grouped bar chart ─────────────────────────────────────────────
    if "is_anomaly" in df.columns:
        anomaly_group = np.where(df["is_anomaly"].to_numpy() == 1, "Anomalous", "Normal")
    else:
        anomaly_score = df["anomaly_score"].to_numpy() if "anomaly_score" in df.columns else None
        if anomaly_score is not None:
            threshold = float(np.mean(anomaly_score) + 2.0 * np.std(anomaly_score))
            anomaly_group = np.where(anomaly_score >= threshold, "Anomalous", "Normal")
        else:
            anomaly_group = np.array(["Unknown"] * len(df))
    _plot_grouped_bar(
        shap_values,
        feature_names,
        group_labels=anomaly_group,
        title="Mean |SHAP| by anomaly label",
        save_path=out_dir / "shap_bar_by_anomaly.png",
    )

    # ── Label-grouped bar chart ───────────────────────────────────────────────
    cfg_path = run_dir / "artefacts" / "training_config.yml"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as _f:
            _saved_cfg = yaml.safe_load(_f)
        label_col = _saved_cfg.get("data", {}).get("label", "label")
        if label_col in df.columns and df[label_col].notna().any():
            rock_labels = df[label_col].fillna("unknown").astype(str).to_numpy()
            _plot_grouped_bar(
                shap_values,
                feature_names,
                group_labels=rock_labels,
                title="Mean |SHAP| by sample label",
                save_path=out_dir / "shap_bar_by_label.png",
            )

    # ── Beeswarm summary ─────────────────────────────────────────────────────
    _plot_summary(shap_values, X_explain, feature_names, out_dir)

    print(f"[attribution] Plots regenerated in {out_dir}")
    return out_dir
