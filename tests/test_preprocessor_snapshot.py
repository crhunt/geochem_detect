from __future__ import annotations

from pathlib import Path

import pytest


MULTICLASS_FEATURE_COLUMNS = [
    "SIO2(WT%)",
    "TIO2(WT%)",
    "AL2O3(WT%)",
    "FEOT(WT%)",
    "CAO(WT%)",
    "MGO(WT%)",
    "MNO(WT%)",
    "K2O(WT%)",
    "NA2O(WT%)",
    "P2O5(WT%)",
]

SPATIAL_FEATURE_COLUMNS = [
    "SiO2n",
    "TiO2n",
    "Al2O3n",
    "FeO*n",
    "MnOn",
    "MgOn",
    "CaOn",
    "Na2On",
    "K2On",
    "P2O5n",
]


def _csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip("\n")


def _configure_preprocess_module(
    module,
    *,
    data_path: Path,
    label_col: str | None,
    spatial: bool,
    feature_columns: list[str],
    scale_features: bool,
    latitude: str = "lat",
    longitude: str = "long",
) -> None:
    module.DATA_PATH = data_path
    module.LABEL_COL = label_col
    module.IDENTIFIER_COL = None
    module.SPATIAL = spatial
    module.LAT_COL = latitude
    module.LON_COL = longitude
    module.SCALE_FEATURES = scale_features
    module.NORMALIZE_BY = None
    module.COLS_FILE = None
    module.FEATURE_COLUMNS = feature_columns


@pytest.mark.parametrize(
    ("raw_name", "expected_name", "label_col", "spatial", "feature_columns", "scale_features"),
    [
        (
            "multiclass_raw_sample.csv",
            "multiclass_expected_processed.csv",
            "ROCK1",
            False,
            MULTICLASS_FEATURE_COLUMNS,
            True,
        ),
        (
            "spatial_raw_sample.csv",
            "spatial_expected_processed.csv",
            "rock_name",
            True,
            SPATIAL_FEATURE_COLUMNS,
            False,
        ),
    ],
)
def test_preprocess_matches_snapshot(
    preprocess_script,
    tmp_path,
    raw_name: str,
    expected_name: str,
    label_col: str,
    spatial: bool,
    feature_columns: list[str],
    scale_features: bool,
) -> None:
    fixture_dir = Path(__file__).parent / "fixtures"
    raw_path = fixture_dir / raw_name
    expected_path = fixture_dir / expected_name
    out_path = tmp_path / expected_name

    _configure_preprocess_module(
        preprocess_script,
        data_path=raw_path,
        label_col=label_col,
        spatial=spatial,
        feature_columns=feature_columns,
        scale_features=scale_features,
    )

    preprocess_script.preprocess(raw_path, out_path)

    assert _csv_text(out_path) == _csv_text(expected_path)