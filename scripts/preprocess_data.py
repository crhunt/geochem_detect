"""Generalised geochemical data preprocessing script.

Configuration is resolved in this order: environment variables, then an
optional YAML config file's top-level data section, then built-in defaults.
That lets the same script handle any CSV dataset — spatial or non-spatial —
without code changes.

Environment Variables
---------------------
DATA_PATH       Path to the raw input CSV.
                Default: ./data/gvirm/multiclass_clean.csv
                Falls back to config data.data_path when --config is supplied.

OUT_PATH        Path to write the cleaned CSV.
                Default: auto-derived by inserting "processed/" after the first
                "data/" segment of DATA_PATH (e.g. data/gvirm/x.csv →
                data/processed/gvirm/x.csv).
                Falls back to config data.processed_data_path when --config is supplied.

LABEL_COL       Optional column name for the rock/class label.
                Default: rock_name  (matches Data1.csv)
                Set to an empty string or "none" to disable label handling.

SPATIAL         Set to "true" / "1" / "yes" to enable spatial processing.
                Default: false

LAT_COL         Column name for latitude (only used when SPATIAL=true).
                Default: lat
                Falls back to config data.latitude when --config is supplied.

LON_COL         Column name for longitude (only used when SPATIAL=true).
                Default: long
                Falls back to config data.longitude when --config is supplied.

COLS_FILE       Optional path to a plain-text file listing feature column names,
                one per line.  When omitted, feature columns are auto-detected as
                every column that is not the label column or a coordinate column.

FEATURE_COLUMNS Optional comma-separated list of feature column names.
                Takes precedence over COLS_FILE when set.
                Falls back to config data.feature_columns when --config is supplied.

IDENTIFIER_COL  Optional input column to preserve as canonical output column
                "identifier" without any processing. When omitted, the original
                row index is used instead.
                Falls back to config data.identifier when --config is supplied.

SCALE_FEATURES  Set to "true" / "1" / "yes" to robust-scale feature columns
                after negative-value remapping and optional normalization.
                Default: true
                Falls back to config data.scale_features when --config is supplied.

Negative feature values are remapped with x --> -0.5*x after numeric coercion.

Usage examples
--------------
# Non-spatial (multiclass_clean.csv)
DATA_PATH=data/gvirm/multiclass_clean.csv LABEL_COL=ROCK1 \\
    uv run python scripts/preprocess_data.py

# Spatial (Data1.csv) — coordinate & label columns already at defaults
SPATIAL=true DATA_PATH=data/gvirm/Data1.csv \\
    uv run python scripts/preprocess_data.py

# Override feature columns from a file
COLS_FILE=configs/feature_cols.txt DATA_PATH=data/gvirm/Data1.csv SPATIAL=true \
    uv run python scripts/preprocess_data.py

# Use config data parameters as fallback when env vars are unset
uv run python scripts/preprocess_data.py \
    --config src/geochem_detect/config/default_config_autoencoder.yml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from sklearn.preprocessing import RobustScaler

# ─── Configuration from environment ─────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parents[1]
_RAW_DATA_ROOT = _PROJECT_ROOT / "data"

DATA_PATH: Path
OUT_PATH: Path
LABEL_COL: str | None
IDENTIFIER_COL: str | None
SPATIAL: bool
LAT_COL: str
LON_COL: str
SCALE_FEATURES: bool
NORMALIZE_BY: str | None
COLS_FILE: Path | None
FEATURE_COLUMNS: list[str] | None


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


def _resolve_preprocess_data_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "data":
        return _PROJECT_ROOT / candidate
    return _RAW_DATA_ROOT / candidate


def _resolve_preprocess_output_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    if candidate.parts[:2] == ("data", "processed"):
        return _PROJECT_ROOT / candidate
    if candidate.parts and candidate.parts[0] == "data":
        return _PROJECT_ROOT / candidate
    return _PROJECT_ROOT / "data" / "processed" / candidate


def _config_value_as_env(value: object) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _load_config_data(config_path: str | None) -> dict:
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.exists():
        sys.exit(f"ERROR: config file not found: {path}")

    with open(path, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file) or {}

    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        sys.exit("ERROR: config file 'data' section must be a mapping")
    return data_cfg


def _resolve_runtime_settings(config_path: str | None) -> None:
    global DATA_PATH, OUT_PATH, LABEL_COL, IDENTIFIER_COL, SPATIAL
    global LAT_COL, LON_COL, SCALE_FEATURES, NORMALIZE_BY, COLS_FILE, FEATURE_COLUMNS

    data_cfg = _load_config_data(config_path)

    def _env_or_config(env_name: str, config_key: str | None = None) -> str | None:
        if env_name in os.environ:
            return os.environ[env_name]
        key = config_key if config_key is not None else env_name.lower()
        value = data_cfg.get(key)
        if value is None:
            return None
        return _config_value_as_env(value)

    data_path_value = _env_or_config("DATA_PATH", "data_path") or "data/gvirm/multiclass_clean.csv"
    DATA_PATH = _resolve_preprocess_data_path(data_path_value)

    label_col_env = (_env_or_config("LABEL_COL", "label") or "rock_name").strip()
    LABEL_COL = None if label_col_env.lower() in {"", "none"} else label_col_env
    identifier_col_env = (_env_or_config("IDENTIFIER_COL", "identifier") or "").strip()
    IDENTIFIER_COL = None if identifier_col_env.lower() in {"", "none"} else identifier_col_env

    spatial_value = _env_or_config("SPATIAL")
    if spatial_value is None:
        spatial_value = "true" if data_cfg.get("latitude") and data_cfg.get("longitude") else "false"
    SPATIAL = spatial_value.strip().lower() in ("true", "1", "yes")

    LAT_COL = (_env_or_config("LAT_COL", "latitude") or "lat").strip()
    LON_COL = (_env_or_config("LON_COL", "longitude") or "long").strip()
    SCALE_FEATURES = (_env_or_config("SCALE_FEATURES", "scale_features") or "true").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    normalize_by_env = (_env_or_config("NORMALIZE_BY", "normalize_by") or "").strip()
    NORMALIZE_BY = None if normalize_by_env.lower() in {"", "none"} else normalize_by_env

    cols_file_value = _env_or_config("COLS_FILE")
    COLS_FILE = Path(cols_file_value) if cols_file_value else None

    feature_columns_value = _env_or_config("FEATURE_COLUMNS", "feature_columns")
    FEATURE_COLUMNS = None
    if feature_columns_value:
        FEATURE_COLUMNS = [
            col.strip() for col in feature_columns_value.split(",") if col.strip()
        ]

    out_path_value = _env_or_config("OUT_PATH", "processed_data_path")
    OUT_PATH = (
        _resolve_preprocess_output_path(out_path_value)
        if out_path_value
        else _derive_out_path(DATA_PATH)
    )

# ─── Helpers ────────────────────────────────────────────────────────────────


def _load_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names, from COLS_FILE or auto-detected."""
    if FEATURE_COLUMNS is not None:
        missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing:
            sys.exit(f"ERROR: columns from FEATURE_COLUMNS not found in data: {missing}")
        print(f"  Feature columns loaded from FEATURE_COLUMNS  ({len(FEATURE_COLUMNS)} columns)")
        return FEATURE_COLUMNS

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
    reserved = {"label", "identifier"}
    if LABEL_COL is not None:
        reserved.add(LABEL_COL)
    if IDENTIFIER_COL is not None:
        reserved.add(IDENTIFIER_COL)
    reserved |= {LAT_COL, LON_COL}
    cols = [c for c in df.columns if c not in reserved]
    print(f"  Feature columns auto-detected  ({len(cols)} columns): {cols}")
    return cols


def _prepare_coordinate_columns(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Validate optional spatial columns and preserve their configured names."""
    if not SPATIAL:
        return df

    missing = [col for col in (LAT_COL, LON_COL) if col not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: coordinate columns {missing} not found in {raw_path}.\n"
            "Set LAT_COL / LON_COL env vars to match your dataset."
        )

    return df


def _prepare_identifier_column(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Normalize the optional identifier column to canonical 'identifier'."""
    if IDENTIFIER_COL is None:
        if "identifier" in df.columns:
            print("  Identifier: using existing 'identifier' column")
            return df
        df = df.copy()
        df.insert(0, "identifier", df.index.to_numpy())
        print("  Identifier: using original row index")
        return df

    if IDENTIFIER_COL not in df.columns:
        sys.exit(
            f"ERROR: identifier column '{IDENTIFIER_COL}' not found in {raw_path}.\n"
            "Set IDENTIFIER_COL to a valid source column name or omit it to use the original row index."
        )

    if IDENTIFIER_COL == "identifier":
        print("  Identifier: using existing 'identifier' column")
        return df

    df = df.rename(columns={IDENTIFIER_COL: "identifier"})
    print(f"  Identifier: preserving '{IDENTIFIER_COL}' as 'identifier'")
    return df


def _prepare_label_column(df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    """Normalize the optional label column to the canonical 'label' name."""
    if LABEL_COL is None:
        print("  Label   : disabled")
        return df

    if LABEL_COL not in df.columns:
        if "label" in df.columns:
            print("  Label   : using existing 'label' column")
            df["label"] = df["label"].astype("string").str.strip()
            return df[df["label"].notna() & (df["label"] != "")]
        print(f"  Label   : '{LABEL_COL}' not found, continuing without labels")
        return df

    df = df.rename(columns={LABEL_COL: "label"})
    df["label"] = df["label"].astype("string").str.strip()
    return df[df["label"].notna() & (df["label"] != "")]


def _fix_negative_feature_values(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Remap negative feature values with x --> -0.5*x."""
    df = df.astype({column_name: "float64" for column_name in feat_cols}, copy=False)
    feature_frame = df.loc[:, feat_cols]
    negative_mask = feature_frame < 0
    negative_count = int(negative_mask.sum().sum())
    if negative_count:
        remapped = feature_frame.where(~negative_mask, -0.5 * feature_frame)
        df.loc[:, feat_cols] = remapped
        print(f"  Negative feature values remapped: {negative_count}")
    else:
        print("  Negative feature values remapped: 0")
    return df


def _normalize_feature_values(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Normalize feature columns by NORMALIZE_BY when configured."""
    if NORMALIZE_BY is None:
        print("  Feature normalization: disabled")
        return df

    if NORMALIZE_BY not in df.columns:
        sys.exit(
            f"ERROR: normalization column '{NORMALIZE_BY}' not found in {DATA_PATH}.\n"
            "Set NORMALIZE_BY to a valid source column name or remove it from the config."
        )

    normalizer = pd.to_numeric(df[NORMALIZE_BY], errors="coerce").astype("float64")
    df = df.copy()
    df[NORMALIZE_BY] = normalizer
    zero_mask = normalizer == 0
    dropped = int(zero_mask.sum())
    if dropped:
        print(f"  Dropping {dropped} rows with zero '{NORMALIZE_BY}' before normalization")
        df = df.loc[~zero_mask].copy()
        normalizer = df[NORMALIZE_BY]

    for column_name in feat_cols:
        if column_name == NORMALIZE_BY:
            continue
        df[column_name] = df[column_name] / normalizer

    print(f"  Feature normalization: divided {len(feat_cols)} columns by '{NORMALIZE_BY}'")
    return df


def _scale_feature_values(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Robust-scale feature columns when configured."""
    if not SCALE_FEATURES:
        print("  Feature scaling: disabled")
        return df

    scaler = RobustScaler()
    df = df.copy()
    df.loc[:, feat_cols] = scaler.fit_transform(df.loc[:, feat_cols]).astype("float64")
    print(f"  Feature scaling: robust-scaled {len(feat_cols)} columns")
    return df


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

    df = _prepare_identifier_column(df, raw_path)
    df = _prepare_label_column(df, raw_path)
    df = _prepare_coordinate_columns(df, raw_path)

    feat_cols = _load_feature_cols(df)

    # Coerce feature columns and optional normalization column to numeric floats.
    numeric_cols = list(feat_cols)
    if NORMALIZE_BY is not None and NORMALIZE_BY not in numeric_cols:
        numeric_cols.append(NORMALIZE_BY)
    numeric_features = df.loc[:, numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.loc[:, numeric_cols] = numeric_features.astype("float64")

    df = _fix_negative_feature_values(df, feat_cols)
    df = _normalize_feature_values(df, feat_cols)

    required = list(feat_cols)
    if "label" in df.columns:
        required.append("label")
    if SPATIAL:
        required += [LAT_COL, LON_COL]
    df = df.dropna(subset=required).reset_index(drop=True)
    df = _scale_feature_values(df, feat_cols)

    keep_cols = list(feat_cols)
    if "identifier" in df.columns:
        keep_cols.insert(0, "identifier")
    if "label" in df.columns:
        insert_at = 1 if "identifier" in keep_cols else 0
        keep_cols.insert(insert_at, "label")
    for column_name in (LAT_COL, LON_COL):
        if column_name in df.columns and column_name not in keep_cols:
            keep_cols.append(column_name)
    df = df[keep_cols].copy()

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
    parser = argparse.ArgumentParser(
        description="Preprocess geochemical CSV data with env vars and optional config fallback."
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help=(
            "Optional YAML config file. Top-level data parameters are used as "
            "fallbacks when env vars are unset."
        ),
    )
    args = parser.parse_args()

    _resolve_runtime_settings(args.config)

    if not DATA_PATH.exists():
        sys.exit(
            f"ERROR: DATA_PATH not found: {DATA_PATH}\n"
            "Set the DATA_PATH environment variable to the raw CSV path."
        )
    preprocess(DATA_PATH, OUT_PATH)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
