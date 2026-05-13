"""Compute model-agnostic Shapley values for a trained anomaly-detection run.

Supports all three geochem_detect anomaly-detection methods:
  isolation_forest | autoencoder | cnn_sae

The method is detected automatically from the artefacts saved under the run
directory.  Results are written to ``<run_dir>/attribution/``:

  shap_values.csv            — per-sample SHAP values + anomaly scores
  shap_summary_plot.png      — beeswarm plot (feature value vs. SHAP value)
  shap_bar_by_anomaly.png    — mean |SHAP| grouped by anomaly/normal label
  shap_bar_by_label.png      — mean |SHAP| grouped by rock/sample label

Usage
-----
# Auto-detect method and explain all samples:
uv run python scripts/compute_attribution.py outputs/isolation_forest/<run_id>

# Limit background clusters and subsample to speed things up:
uv run python scripts/compute_attribution.py outputs/autoencoder/<run_id> \\
    --max-background 50 --max-samples 500

# Regenerate plots from an existing shap_values.csv (no SHAP recomputation):
uv run python scripts/compute_attribution.py outputs/isolation_forest/<run_id> --plots-only

# CNN-SAE run:
uv run python scripts/compute_attribution.py outputs/cnn_sae/<run_id>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
_REPO_ROOT = Path(__file__).parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute KernelSHAP attributions for a trained anomaly-detection run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "run_dir",
        metavar="RUN_DIR",
        help="Path to the run directory (outputs/<method>/<run_id>).",
    )
    parser.add_argument(
        "--max-background",
        type=int,
        default=100,
        metavar="N",
        help="Number of k-means background clusters for KernelExplainer (default: 100).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Randomly subsample this many rows before computing SHAP values.  "
            "Useful for large datasets where full explanation would be slow.  "
            "Default: explain all samples."
        ),
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        default=False,
        help=(
            "Skip SHAP computation and regenerate plots from an existing "
            "shap_values.csv.  The CSV must already exist under "
            "<RUN_DIR>/attribution/shap_values.csv."
        ),
    )
    args = parser.parse_args()

    if args.plots_only:
        from geochem_detect.attribution.explainer import regenerate_plots
        regenerate_plots(run_dir=Path(args.run_dir))
    else:
        from geochem_detect.attribution.explainer import run_attribution
        run_attribution(
            run_dir=Path(args.run_dir),
            max_background=args.max_background,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()
