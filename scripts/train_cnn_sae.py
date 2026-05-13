"""Train CNN-SAE spatial anomaly detector on Data1.csv.

Usage
-----
    uv run python scripts/train_cnn_sae.py [--config path/to/config.yml]

All hyperparameters are read from the config file (or the bundled default when
``--config`` is omitted).  See
``src/geochem_detect/config/default_config_cnn_sae.yml`` for the full list of
tuneable settings.

The model ingests spatially-windowed samples from Data1.csv, where each window
is a ``grid_size × grid_size`` grid of geochemical features. Evaluation can use
either explicit anomalous label values or frequency-derived anomalies when an
``evaluation`` config block is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from geochem_detect.config import (
    data_params,
    evaluation_params,
    load_config,
    model_params,
    sampling_params,
    training_params,
    validate_training_config,
)
from geochem_detect.data.loader import DEFAULT_SPATIAL_DATA, load_spatial_frame
from geochem_detect.data.preprocessor import IdentityScaler
from geochem_detect.training.trainer import resolve_anomaly_ground_truth, train_cnn_sae
from geochem_detect.visualization.plots import (
    plot_anomaly_scores_histogram,
    plot_pr_curve_binary,
    plot_spatial_anomalies,
)

OUTPUT_ROOT = Path(__file__).parents[1] / "outputs"


def _evaluation_requires_labels(evaluation: dict | None) -> bool:
    evaluation = dict(evaluation or {})
    return any(
        evaluation.get(key) is not None
        for key in ("anomaly_labels", "contamination_threshold")
    )


def _evaluation_label_col(evaluation: dict | None, data_cfg: dict | None) -> str:
    evaluation = dict(evaluation or {})
    data_cfg = dict(data_cfg or {})
    return str(evaluation.get("label") or data_cfg.get("label") or "label")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-SAE spatial anomaly detector."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to a YAML config file.  Overrides the bundled default.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        metavar="NAME",
        help="Display name for the MLFlow run.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        metavar="PATH",
        help="Override the default Data1.csv path.",
    )
    parser.add_argument(
        "--experiment",
        default="cnn_sae",
        metavar="NAME",
        help="MLFlow experiment name (default: cnn_sae).",
    )
    args = parser.parse_args()

    cfg = load_config("cnn_sae", args.config)
    validate_training_config("cnn_sae", cfg)
    data_cfg = data_params(cfg)
    mp = model_params(cfg)
    tp = training_params(cfg)
    ep = evaluation_params(cfg)
    sp = sampling_params(cfg)
    labels_required = _evaluation_requires_labels(ep)
    label_col = _evaluation_label_col(ep, data_cfg)

    # ── Load and clean data ──────────────────────────────────────────────────
    if args.data_path is not None:
        data_cfg = dict(data_cfg)
        data_cfg["data_path"] = args.data_path
    gdf, data_options = load_spatial_frame(
        data_cfg,
        DEFAULT_SPATIAL_DATA,
        require_label=labels_required,
        label_col=label_col,
    )
    feat_cols = data_options["feature_columns"]

    # Stamp the resolved absolute path and actual feature columns back into cfg
    # so the saved training_config.yml is fully self-contained.
    cfg["data"]["data_path"] = data_options["data_path"]
    cfg["data"]["feature_columns"] = feat_cols

    X_raw = gdf[feat_cols].to_numpy(dtype=np.float32)
    label_available = data_options["label_col"] in gdf.columns
    if labels_required and not label_available:
        raise ValueError(
            f"Evaluation requires label column '{data_options['label_col']}' in the processed dataset. "
            "Re-run preprocessing with a label source column or remove evaluation settings."
        )

    if label_available:
        le = LabelEncoder()
        y_raw = le.fit_transform(gdf[data_options["label_col"]].values)
    else:
        le = LabelEncoder()
        y_raw = np.zeros(len(gdf), dtype=np.int32)
        le.fit(np.array(["data"], dtype=object))
    scaler = IdentityScaler().fit(X_raw)

    dataset_info = {
        "dataset": data_options["data_path"],
        "feature_cols": feat_cols,
        "label_col": data_options["label_col"],
        "n_samples": len(X_raw),
        "longitude": data_options["longitude"],
        "latitude": data_options["latitude"],
        "normalize_by": None,
        "scale_features": False,
    }

    params = {**mp, **tp}
    det, pr_auc, run_id = train_cnn_sae(
        gdf,
        y_raw,
        le,
        scaler,
        dataset_info,
        sampling_params=sp,
        params=params,
        evaluation=ep,
        experiment_name=args.experiment,
        run_name=args.run_name or "data1_cnn_sae",
        cfg=cfg,
    )

    # ── Post-training plots ──────────────────────────────────────────────────
    import json

    out_dir = OUTPUT_ROOT / "cnn_sae" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    y_anom_all, _, _ = resolve_anomaly_ground_truth(y_raw, le, ep)
    sigma_cutoff = ep.get("anomaly_sigma_cutoff", 2.0)
    if y_anom_all is None:
        print("  Evaluation plots skipped where ground-truth anomalies are required")
    # Per-point ground-truth anomaly labels (used for raw-data-point overlay)
    raw_anom_labels = (
        y_anom_all.astype(np.int32)
        if y_anom_all is not None
        else np.zeros(len(y_raw), dtype=np.int32)
    )

    # Reload window splits and metadata from saved artefacts
    art_dir = out_dir / "artefacts"
    splits_npz = np.load(art_dir / "window_splits.npz")
    with open(art_dir / "window_metadata.json") as f:
        all_metadata = json.load(f)

    # Re-generate all windows from saved metadata to compute plots
    # (we need the same X tensor as used for training)
    from geochem_detect.data.spatial_sampler import SpatialSampler

    sampler = SpatialSampler(
        gdf=gdf,
        feature_cols=feat_cols,
        anomaly_labels=raw_anom_labels,
        **sp,
    )
    X_all, y_all, _ = sampler.generate()

    # ── Score ALL windows at once for globally consistent normalization ───────
    # anomaly_scores() uses per-batch min-max normalization, so scoring each
    # split separately produces incomparable values (e.g. "all" can appear to
    # have fewer detections than "train" alone).  Scoring everything together
    # keeps the scale consistent across all per-split slices.
    all_scores = det.anomaly_scores(X_all)

    threshold = None
    if sigma_cutoff is not None:
        threshold_file = art_dir / "anomaly_threshold.json"
        threshold_data = json.loads(threshold_file.read_text()) if threshold_file.exists() else {}
        threshold = threshold_data.get("threshold")
        if threshold is None:
            val_idx_saved = splits_npz["val_idx"]
            val_scores = all_scores[val_idx_saved]
            threshold = float(np.mean(val_scores) + sigma_cutoff * np.std(val_scores))
            threshold_data["sigma_cutoff"] = sigma_cutoff
            threshold_data["threshold"] = threshold
            threshold_file.write_text(json.dumps(threshold_data, indent=2))
            print(f"  [CNN-SAE] val threshold = {threshold:.4f} "
                  f"(mean={np.mean(val_scores):.4f}, sigma_cutoff={sigma_cutoff})")

    named_splits = {
        "train": splits_npz["train_idx"],
        "val":   splits_npz["val_idx"],
        "test":  splits_npz["test_idx"],
        "all":   np.concatenate([
            splits_npz["train_idx"],
            splits_npz["val_idx"],
            splits_npz["test_idx"],
        ]),
    }

    for split_name, idx in named_splits.items():
        # Slice pre-computed global scores so normalization is consistent
        scores = all_scores[idx]
        title_sfx = f"({split_name})"

        if y_anom_all is not None:
            y_anom = y_all[idx]
            plot_pr_curve_binary(
                y_anom, scores,
                title=f"Precision-Recall Curve {title_sfx}",
                save_path=out_dir / f"pr_curve_cnn_sae_{split_name}.png",
            )
        plot_anomaly_scores_histogram(
            scores, sigma_cutoff=sigma_cutoff,
            title=f"Anomaly Score Distribution {title_sfx}",
            save_path=out_dir / f"scores_cnn_sae_{split_name}.png",
            threshold=threshold,
        )

        # Build a per-window GeoDataFrame for the spatial plot
        # Use the center coordinates of each window as the plot location
        import geopandas as gpd

        meta_s = [all_metadata[i] for i in idx]
        window_gdf = gpd.GeoDataFrame(
            {
                "lat":  [m["center_lat"] for m in meta_s],
                "long": [m["center_lon"] for m in meta_s],
            },
            geometry=gpd.points_from_xy(
                [m["center_lon"] for m in meta_s],
                [m["center_lat"] for m in meta_s],
            ),
            crs="EPSG:4326",
        )
        # Raw points that contributed to any window in this split,
        # colored by their ground-truth anomaly label (rare class).
        split_pt_idx = np.unique(
            np.concatenate([all_metadata[i]["point_indices"] for i in idx])
        ).astype(int)

        plot_spatial_anomalies(
            window_gdf, scores,
            threshold=threshold,
            y_true=y_all[idx] if y_anom_all is not None else None,
            title=f"Spatial Anomaly Map {title_sfx}",
            save_path=out_dir / f"spatial_anomaly_map_cnn_sae_{split_name}.png",
            window_deg=sp["window_deg"],
            raw_gdf=gdf.iloc[split_pt_idx],
            raw_y=raw_anom_labels[split_pt_idx] if y_anom_all is not None else None,
        )

    print(f"Plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
