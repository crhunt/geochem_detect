from __future__ import annotations

import copy

import pytest

from geochem_detect.config import load_config, validate_training_config


@pytest.mark.parametrize(
    "model_type",
    ["isolation_forest", "autoencoder", "classifier", "cnn_sae"],
)
def test_default_training_configs_validate(model_type: str) -> None:
    cfg = load_config(model_type)

    assert validate_training_config(model_type, cfg) == cfg


@pytest.mark.parametrize(
    ("model_type", "section_name", "field_name"),
    [
        ("isolation_forest", "model", "contamination"),
        ("autoencoder", "training", "spatial"),
        ("classifier", "data", "feature_columns"),
        ("cnn_sae", "sampling", "grid_size"),
    ],
)
def test_validate_training_config_reports_missing_required_fields(
    model_type: str,
    section_name: str,
    field_name: str,
) -> None:
    cfg = copy.deepcopy(load_config(model_type))
    cfg[section_name][field_name] = None

    with pytest.raises(ValueError, match=rf"{section_name}\.{field_name}"):
        validate_training_config(model_type, cfg)


def test_load_config_raises_clean_error_for_invalid_override(tmp_path) -> None:
    override_path = tmp_path / "invalid_classifier.yml"
    override_path.write_text("data:\n  feature_columns: null\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"data\.feature_columns"):
        load_config("classifier", override_path)