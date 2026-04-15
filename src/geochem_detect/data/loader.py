"""Data loading utilities for geochemical datasets."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_RAW_DIR = Path(__file__).parents[3] / "data" / "gvirm"
_PROCESSED_DIR = Path(__file__).parents[3] / "data" / "processed" / "gvirm"
_PROCESSED_ROOT = Path(__file__).parents[3] / "data" / "processed"

DEFAULT_MULTICLASS_DATA = Path("gvirm") / "multiclass_clean.csv"
DEFAULT_SPATIAL_DATA = Path("gvirm") / "Data1.csv"
DEFAULT_LABEL_COL = "label"


def resolve_data_path(default_relative: str | Path, override: str | Path | None = None) -> Path:
    """Resolve a dataset path, preferring absolute overrides or processed-data paths."""
    candidate_value = override if override is not None else os.environ.get("DATA_PATH")
    source = "DATA_PATH" if candidate_value is not None else "default"

    if candidate_value is None:
        candidate = Path(default_relative)
    else:
        candidate = Path(candidate_value)

    if candidate.is_absolute():
        resolved = candidate
    elif candidate.parts[:2] == ("data", "processed"):
        resolved = Path(__file__).parents[3] / candidate
    else:
        resolved = _PROCESSED_ROOT / candidate

    if resolved.exists():
        return resolved

    if source == "DATA_PATH" and candidate.is_absolute():
        raise FileNotFoundError(
            f"DATA_PATH points to '{resolved}', but that file does not exist."
        )

    expected_root = _PROCESSED_ROOT
    raise FileNotFoundError(
        f"Could not find dataset '{candidate}'. Expected a file under {expected_root} "
        f"(resolved to '{resolved}'). Set DATA_PATH to an absolute path or run `make preprocess` first."
    )


def resolve_data_options(
    data_cfg: dict | None,
    default_relative: str | Path,
) -> dict:
    """Resolve data loading options from config plus environment overrides."""
    data_cfg = dict(data_cfg or {})
    override_path = os.environ.get("DATA_PATH") or data_cfg.get("data_path")
    longitude = os.environ.get("LON_COL") or data_cfg.get("longitude")
    latitude = os.environ.get("LAT_COL") or data_cfg.get("latitude")
    feature_columns = data_cfg.get("feature_columns")
    if feature_columns is not None:
        feature_columns = list(feature_columns)

    return {
        "data_path": str(resolve_data_path(default_relative, override_path)),
        "label_col": DEFAULT_LABEL_COL,
        "feature_columns": feature_columns,
        "longitude": longitude,
        "latitude": latitude,
        "scale_features": bool(data_cfg.get("scale_features", True)),
        "normalize_by": data_cfg.get("normalize_by"),
    }


def resolve_feature_columns(df: pd.DataFrame, data_options: dict) -> list[str]:
    """Resolve feature columns from config or by auto-discovery."""
    reserved = {data_options["label_col"]}
    for column_name in (
        data_options.get("longitude"),
        data_options.get("latitude"),
        data_options.get("normalize_by"),
    ):
        if column_name:
            reserved.add(column_name)

    configured = data_options.get("feature_columns")
    if configured is None:
        cols = [column_name for column_name in df.columns if column_name not in reserved]
    else:
        missing = [column_name for column_name in configured if column_name not in df.columns]
        if missing:
            raise ValueError(f"Configured feature columns not found in data: {missing}")
        cols = [column_name for column_name in configured if column_name not in reserved]

    if not cols:
        raise ValueError("No usable feature columns were resolved from the dataset")
    return cols


def load_dataset_frame(
    data_cfg: dict | None,
    default_relative: str | Path,
    require_label: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Load a labeled dataset and resolve feature/coordinate options."""
    data_options = resolve_data_options(data_cfg, default_relative)
    df = pd.read_csv(data_options["data_path"], encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    label_col = data_options["label_col"]
    if require_label:
        if label_col not in df.columns:
            raise ValueError(
                f"Expected label column '{label_col}' in {data_options['data_path']}. "
                "Run preprocessing first or rename the label column to 'label'."
            )
        df[label_col] = df[label_col].astype("string").str.strip()
        df = df[df[label_col].notna() & (df[label_col] != "")].reset_index(drop=True)

    data_options["feature_columns"] = resolve_feature_columns(df, data_options)
    return df, data_options


def load_spatial_frame(
    data_cfg: dict | None,
    default_relative: str | Path,
    require_label: bool = True,
):
    """Load a labeled spatial dataset as a GeoDataFrame using configured coord columns."""
    import geopandas as gpd

    df, data_options = load_dataset_frame(data_cfg, default_relative, require_label=require_label)
    longitude = data_options["longitude"]
    latitude = data_options["latitude"]
    if longitude is None or latitude is None:
        raise ValueError(
            "Spatial data requires configured longitude and latitude columns. "
            "Set data.longitude and data.latitude in the config or override them via LON_COL/LAT_COL."
        )
    missing = [column_name for column_name in (longitude, latitude) if column_name not in df.columns]
    if missing:
        raise ValueError(f"Spatial columns not found in data: {missing}")

    df = df.dropna(subset=[longitude, latitude]).reset_index(drop=True)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[longitude], df[latitude]),
        crs="EPSG:4326",
    )
    return gdf, data_options


def load_multiclass(path: str | Path | None = None, data_cfg: dict | None = None) -> pd.DataFrame:
    """Load a processed labeled dataset as a plain DataFrame.

    Returns a DataFrame with a ``label`` column and numeric feature columns.
    """
    merged_cfg = dict(data_cfg or {})
    if path is not None:
        merged_cfg["data_path"] = str(path)
    df, _ = load_dataset_frame(merged_cfg, DEFAULT_MULTICLASS_DATA)
    return df


def load_spatial(path: str | Path | None = None, data_cfg: dict | None = None):
    """Load a processed labeled dataset as a spatial GeoDataFrame."""
    merged_cfg = dict(data_cfg or {})
    if path is not None:
        merged_cfg["data_path"] = str(path)
    gdf, _ = load_spatial_frame(merged_cfg, DEFAULT_SPATIAL_DATA)
    return gdf
