"""Spatial window sampler for CNN-SAE anomaly detection.

Given a GeoDataFrame of point geochemical measurements, this module generates
a set of spatially-windowed samples by:

1. Randomly choosing a center point from the existing data locations.
2. Collecting all source points that fall within a square bounding box of
   ``window_deg`` degrees around that center.
3. Binning the collected points into a ``grid_size × grid_size`` regular grid
   (in degree space), averaging geochemical features when multiple points land
   in the same cell, and recording an occupancy mask (1 = cell has data, 0 =
   empty).
4. Labelling the window as anomalous (1) when the **fraction** of its
   constituent source points with anomaly label 1 meets or exceeds
   ``anomaly_fraction_threshold`` (default 0.5).  The "any point" policy
   (threshold = 0) inflates the anomaly rate to near-100% when the
   per-point anomaly rate is high (e.g. 29% across all points → ~99% of
   windows contain at least one anomalous point).

The resulting sample tensor has shape
    ``(n_samples, grid_size, grid_size, n_features + 1)``
where the last channel is the occupancy mask.  The occupancy mask is used by
the CNN-SAE loss function so that reconstruction error is only computed over
cells that actually contain data.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class SpatialSampler:
    """Generate spatially-windowed grid samples from a GeoDataFrame.

    Parameters
    ----------
    gdf:
        GeoDataFrame with point geometries and pre-scaled geochemical feature
        columns. Must have been cleaned (no NaN in ``feature_cols``).
    feature_cols:
        Names of the geochemical feature columns inside *gdf*.
    anomaly_labels:
        Binary array (length == len(gdf)) where 1 = anomalous point.
    window_deg:
        Half-extent of the bounding box in decimal degrees.  A window is a
        square region of ``window_deg`` degrees on each side, centred on the
        sampled point.
    grid_size:
        Number of rows and columns in the output grid.
    n_samples:
        Number of windows to generate.  Centers are sampled **with
        replacement** from existing data-point locations, so the same center
        may appear more than once.
    min_points:
        Minimum number of distinct grid cells that must contain at least one
        data point for the window to be accepted.  Windows below this threshold
        are discarded and re-sampled.
    random_state:
        Seed for reproducibility.
    max_retries:
        Maximum number of re-sample attempts per accepted window when the
        current candidate has fewer than ``min_points`` occupied cells.
    """

    def __init__(
        self,
        gdf,
        feature_cols: list[str],
        anomaly_labels: np.ndarray,
        window_deg: float = 1.0,
        grid_size: int = 16,
        n_samples: int = 2000,
        min_points: int = 2,
        anomaly_fraction_threshold: float = 0.5,
        random_state: int = 42,
        max_retries: int = 10,
    ) -> None:
        import geopandas as gpd  # noqa: F401 – validated at runtime

        self.gdf = gdf.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.anomaly_labels = np.asarray(anomaly_labels, dtype=np.int32)
        self.half = window_deg / 2.0
        self.window_deg = window_deg
        self.grid_size = grid_size
        self.n_samples = n_samples
        self.min_points = min_points
        self.anomaly_fraction_threshold = anomaly_fraction_threshold
        self.rng = np.random.default_rng(random_state)
        self.max_retries = max_retries

        # Pre-extract arrays for fast filtering
        self._lats = self.gdf.geometry.y.values.astype(np.float64)
        self._lons = self.gdf.geometry.x.values.astype(np.float64)
        self._features = self.gdf[self.feature_cols].values.astype(np.float32)
        self._n_features = len(self.feature_cols)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _window_indices(self, lat_c: float, lon_c: float) -> np.ndarray:
        """Return row indices of source points within the window bounding box."""
        mask = (
            (self._lats >= lat_c - self.half) &
            (self._lats <= lat_c + self.half) &
            (self._lons >= lon_c - self.half) &
            (self._lons <= lon_c + self.half)
        )
        return np.where(mask)[0]

    def _points_to_grid(
        self, idx: np.ndarray, lat_c: float, lon_c: float
    ) -> np.ndarray:
        """Convert a set of source-point indices into a (H, W, C+1) grid tensor.

        The last channel is the occupancy mask (1.0 where data is present).
        When multiple points fall in the same cell their features are averaged.
        """
        H = W = self.grid_size
        C = self._n_features
        # Accumulators for feature sum and counts per cell
        feat_sum = np.zeros((H, W, C), dtype=np.float32)
        count = np.zeros((H, W), dtype=np.int32)

        lats_w = self._lats[idx]
        lons_w = self._lons[idx]

        # Map to [0, grid_size) bins
        lat_min = lat_c - self.half
        lon_min = lon_c - self.half
        rows = np.floor((lats_w - lat_min) / self.window_deg * H).astype(int)
        cols = np.floor((lons_w - lon_min) / self.window_deg * W).astype(int)
        # Clamp to valid range (handles edge cases where value == max)
        rows = np.clip(rows, 0, H - 1)
        cols = np.clip(cols, 0, W - 1)

        for r, c, feat in zip(rows, cols, self._features[idx]):
            feat_sum[r, c] += feat
            count[r, c] += 1

        # Average where count > 0
        occ = (count > 0).astype(np.float32)
        feat_avg = np.where(count[:, :, None] > 0, feat_sum / np.maximum(count[:, :, None], 1), 0.0)

        # Concatenate occupancy mask as last channel
        grid = np.concatenate([feat_avg, occ[:, :, None]], axis=-1)
        return grid.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Generate all window samples.

        Returns
        -------
        X : np.ndarray, shape (n_accepted, grid_size, grid_size, n_features+1)
            Grid tensors with occupancy mask as last channel.
        y : np.ndarray, shape (n_accepted,), dtype int32
            Window-level anomaly labels.  1 if the fraction of points within
            the window with ``anomaly_labels == 1`` is >= ``anomaly_fraction_threshold``,
            else 0.
        metadata : list[dict]
            One dict per accepted window with keys:
            ``center_lat``, ``center_lon``, ``point_indices`` (list[int]),
            ``n_points`` (int).
        """
        n_pts = len(self.gdf)
        grids: list[np.ndarray] = []
        labels: list[int] = []
        metadata: list[dict[str, Any]] = []

        accepted = 0
        # Pool of candidate center indices (sampled with replacement)
        center_pool = self.rng.integers(0, n_pts, size=self.n_samples * (self.max_retries + 1))
        pool_pos = 0

        while accepted < self.n_samples:
            if pool_pos >= len(center_pool):
                # Replenish pool
                center_pool = self.rng.integers(0, n_pts, size=self.n_samples * (self.max_retries + 1))
                pool_pos = 0

            ci = int(center_pool[pool_pos])
            pool_pos += 1
            lat_c = float(self._lats[ci])
            lon_c = float(self._lons[ci])

            idx = self._window_indices(lat_c, lon_c)
            # Count occupied cells before building the full grid
            n_occupied = len(np.unique(
                np.stack([
                    np.clip(np.floor((self._lats[idx] - (lat_c - self.half)) / self.window_deg * self.grid_size).astype(int), 0, self.grid_size - 1),
                    np.clip(np.floor((self._lons[idx] - (lon_c - self.half)) / self.window_deg * self.grid_size).astype(int), 0, self.grid_size - 1),
                ], axis=1),
                axis=0,
            ))
            if n_occupied < self.min_points:
                continue

            grid = self._points_to_grid(idx, lat_c, lon_c)
            anom_frac = float(self.anomaly_labels[idx].mean())
            window_label = int(anom_frac >= self.anomaly_fraction_threshold)

            grids.append(grid)
            labels.append(window_label)
            metadata.append({
                "center_lat": lat_c,
                "center_lon": lon_c,
                "point_indices": idx.tolist(),
                "n_points": len(idx),
            })
            accepted += 1

        X = np.stack(grids, axis=0)
        y = np.array(labels, dtype=np.int32)
        return X, y, metadata
