"""Cleanse raw geochemical CSVs and write processed copies to data/processed/gvirm/.

Usage:
    uv run python scripts/preprocess_data.py

Outputs
-------
data/processed/gvirm/multiclass_clean.csv
data/processed/gvirm/Data1.csv
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[1] / "data" / "gvirm"
OUT_DIR = Path(__file__).parents[1] / "data" / "processed" / "gvirm"

FEATURE_COLS_MULTICLASS = [
    "SIO2(WT%)", "TIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)",
    "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)", "P2O5(WT%)",
]

FEATURE_COLS_SPATIAL = [
    "SiO2n", "TiO2n", "Al2O3n", "FeO*n",
    "MnOn", "MgOn", "CaOn", "Na2On", "K2On", "P2O5n",
]


def _report(name: str, before: int, after: int) -> None:
    dropped = before - after
    print(f"  {name}: {before} → {after} rows  ({dropped} dropped)")


def preprocess_multiclass(raw_path: Path, out_path: Path) -> None:
    print("Processing multiclass_clean.csv …")
    df = pd.read_csv(raw_path)

    before = len(df)
    # Normalise label column
    df = df.rename(columns={"ROCK1": "label"})
    df["label"] = df["label"].str.strip()
    df = df[df["label"].notna() & (df["label"] != "")]

    # Coerce all feature columns to numeric; drop rows that become NaN
    for col in FEATURE_COLS_MULTICLASS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS_MULTICLASS).reset_index(drop=True)

    _report("multiclass_clean", before, len(df))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def preprocess_spatial(raw_path: Path, out_path: Path) -> None:
    print("Processing Data1.csv …")
    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    before = len(df)
    # Normalise label column
    df = df.rename(columns={"rock_name": "label"})
    df["label"] = df["label"].str.strip()
    df = df[df["label"].notna() & (df["label"] != "")]

    # Coerce feature columns
    for col in FEATURE_COLS_SPATIAL:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing coords, label, or any feature
    required = FEATURE_COLS_SPATIAL + ["long", "lat", "label"]
    df = df.dropna(subset=required).reset_index(drop=True)

    _report("Data1", before, len(df))

    # Validate as GeoDataFrame (ensures coords are numeric/valid)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["long"], df["lat"]),
        crs="EPSG:4326",
    )
    # Drop any rows where geometry is invalid
    gdf = gdf[gdf.is_valid].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save without geometry column — loader re-creates it from long/lat
    gdf.drop(columns="geometry").to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


def main() -> None:
    preprocess_multiclass(
        RAW_DIR / "multiclass_clean.csv",
        OUT_DIR / "multiclass_clean.csv",
    )
    preprocess_spatial(
        RAW_DIR / "Data1.csv",
        OUT_DIR / "Data1.csv",
    )
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
