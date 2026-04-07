"""Generalised geochemical data preprocessing script.

All configuration is driven by environment variables so the same script handles
any CSV dataset — spatial or non-spatial — without code changes.

Environment Variables
---------------------
DATA_PATH       Path to the raw input CSV.
                Default: ./data/gvirm/multiclass_clean.csv

OUT_PATH        Path to write the cleaned CSV.
                Default: auto-derived by inserting "processed/" after the first
                "data/" segment of DATA_PATH (e.g. data/gvirm/x.csv →
                data/processed/gvirm/x.csv).

LABEL_COL       Column name for the rock/class label.
                Default: rock_name  (matches Data1.csv)

SPATIAL         Set to "true" / "1" / "yes" to enable spatial processing.
                Default: false

LAT_COL         Column name for latitude (only used when SPATIAL=true).
                Default: lat

LON_COL         Column name for longitude (only used when SPATIAL=true).
                Default: long

COLS_FILE       Optional path to a plain-text file listing feature column names,
                one per line.  When omitted, feature columns are auto-detected as
                every column that is not the label column or a coordinate column.

Usage examples
--------------
# Non-spatial (multiclass_clean.csv)
DATA_PATH=data/gvirm/multiclass_clean.csv LABEL_COL=ROCK1 \\
    uv run python scripts/preprocess_data.py

# Spatial (Data1.csv) — coordinate & label columns already at defaults
SPATIAL=true DATA_PATH=data/gvirm/Data1.csv \\
    uv run python scripts/preprocess_data.py

# Override feature columns from a file
COLS_FILE=configs/feature_cols.txt DATA_PATH=data/gvirm/Data1.csv SPATIAL=true \\
    uv run python scripts/preprocess_data.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ─── Configuration from environment ─────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parents[1]

DATA_PATH: Path = Path(os.environ.get("DATA_PATH", "data/gvirm/multiclass_clean.csv"))
if not DATA_PATH.is_absolute():
    DATA_PATH = _PROJECT_ROOT / DATA_PATH

LABEL_COL: str = os.environ.get("LABEL_COL", "rock_name")
SPATIAL:   bool = os.environ.get("SPATIAL", "false").strip().lower() in ("true", "1", "yes")
LAT_COL:   str = os.environ.get("LAT_COL", "lat")
LON_COL:   str = os.environ.get("LON_COL", "long")
COLS_FILE: Path | None = Path(os.environ["COLS_FILE"]) if "COLS_FILE" in os.environ else None


def _derive_out_path(src: Path) -> Path:
    """Insert 'processed/' after the first 'data/' segment of the path."""
    parts = src.parts
    try:
        data_idx = next(i for i, p in enumerate(parts) if p == "data")
        new_parts = parts[: data_idx + 1] + ("processed",) + parts[data_idx + 1 :]
        return Path(*new_parts)
    except StopIteration:
        # Fallback: place alongside source under data/processed/
        return _PROJECT_ROOT / "data" / "processed" / src.name


_out_env = os.environ.get("OUT_PATH")
OUT_PATH: Path = Path(_out_env) if _out_env else _derive_out_path(DATA_PATH)
if not OUT_PATH.is_absolute():
    OUT_PATH = _PROJECT_ROOT / OUT_PATH

# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names, from COLS_FILE or auto-detected."""
    if COLS_FILE is not None:
        if not COLS_FILE.exists():
            sys.exit(f"ERROR: COLS_FILE not found: {COLS_FILE}")
        cols = [ln.strip() for ln in COLS_FILE.read_text().splitlines() if ln.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            sys.exit(f"ERROR: columns from COLS_FILE not found in data: {missing}")
        print(f"  Feature columns loaded from {COLS_FILE}  ({len(cols)} columns)")
        return cols

    # Auto-detect: all columns except the reserved ones
    reserved = {LABEL_COL, "label"}
    if SPATIAL:
        reserved |= {LAT_COL, LON_COL}
    cols = [c for c in df.columns if c not in reserved]
    print(f"  Feature columns auto-detected  ({len(cols)} columns): {cols}")
    return cols


def _report(name: str, before: int, after: int) -> None:
    print(f"  {name}: {before} → {after} rows  ({before - after} dropped)")


# ─── Core processing ────────────────────────────────────────────────────────


def preprocess(raw_path: Path, out_path: Path) -> None:
    name = raw_path.name
    print(f"Processing {name} …")
    print(f"  Source  : {raw_path}")
    print(f"  Output  : {out_path}")
    print(f"  Spatial : {SPATIAL}")

    df = pd.read_csv(raw_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    before = len(df)

    # Rename label column → "label"
    if LABEL_COL not in df.columns:
        sys.exit(
            f"ERROR: label column '{LABEL_COL}' not found in {raw_path}.\n"
            f"Available columns: {list(df.columns)}"
        )
    df = df.rename(columns={LABEL_COL: "label"})
    df["label"] = df["label"].str.strip()
    df = df[df["label"].notna() & (df["label"] != "")]

    # Validate coordinate columns when spatial
    if SPATIAL:
        for col in (LAT_COL, LON_COL):
            if col not in df.columns:
                sys.exit(
                    f"ERROR: coordinate column '{col}' not found in {raw_path}.\n"
                    f"Set LAT_COL / LON_COL env vars to match your dataset."
                )

    feat_cols = _load_feature_cols(df)

    # Coerce feature columns to numeric; drop rows that fail
    for col in feat_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    required = feat_cols + ["label"]
    if SPATIAL:
        required += [LAT_COL, LON_COL]
    df = df.dropna(subset=required).reset_index(drop=True)

    _report(name, before, len(df))

    if SPATIAL:
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
            crs="EPSG:4326",
        )
        invalid = (~gdf.is_valid).sum()
        if invalid:
            print(f"  Dropping {invalid} rows with invalid geometry")
        gdf = gdf[gdf.is_valid].reset_index(drop=True)
        out_df = gdf.drop(columns="geometry")
    else:
        out_df = df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}\n")


# ─── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: DATA_PATH not found: {DATA_PATH}\n"
            "Set the DATA_PATH environment variable to the raw CSV path."
        )
    preprocess(DATA_PATH, OUT_PATH)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
