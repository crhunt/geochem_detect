"""Config loading utilities for geochem_detect models."""
from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).parent

_DEFAULT_CONFIGS: dict[str, Path] = {
    "isolation_forest": _CONFIG_DIR / "default_config_isolation_forest.yml",
    "autoencoder":      _CONFIG_DIR / "default_config_autoencoder.yml",
    "classifier":       _CONFIG_DIR / "default_config_classifier.yml",
    "cnn_sae":          _CONFIG_DIR / "default_config_cnn_sae.yml",
}

_REQUIRED_CONFIG_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "isolation_forest": {
        "data": ("data_path", "feature_columns", "label"),
        "model": ("n_estimators", "contamination", "max_features", "random_state"),
    },
    "autoencoder": {
        "data": ("data_path", "feature_columns", "label", "longitude", "latitude"),
        "model": (
            "encoding_dim",
            "hidden_dims",
            "dropout_rate",
            "learning_rate",
            "epochs",
            "batch_size",
            "validation_split",
            "patience",
        ),
        "training": ("spatial",),
    },
    "classifier": {
        "data": ("data_path", "feature_columns", "label"),
        "model": (
            "hidden_dims",
            "dropout_rate",
            "learning_rate",
            "epochs",
            "batch_size",
            "validation_split",
            "patience",
        ),
    },
    "cnn_sae": {
        "data": ("data_path", "feature_columns", "label", "longitude", "latitude"),
        "sampling": (
            "window_deg",
            "grid_size",
            "n_samples",
            "min_points",
            "anomaly_fraction_threshold",
            "random_state",
        ),
        "model": (
            "cnn_filters",
            "cnn_kernel_size",
            "encoding_dim",
            "dense_hidden_dims",
            "dropout_rate",
            "learning_rate",
            "sparsity_weight",
            "epochs",
            "batch_size",
            "patience",
        ),
        "training": ("val_size", "test_size"),
    },
}


def load_config(model_type: str, config_path: str | Path | None = None) -> dict:
    """Load a YAML config for *model_type*, falling back to the bundled default.

    Parameters
    ----------
    model_type:
        One of ``"isolation_forest"``, ``"autoencoder"``, ``"classifier"``,
        ``"cnn_sae"``.
    config_path:
        Optional path to a custom YAML file.  When ``None`` the default config
        bundled with the package is used.

    Returns
    -------
    dict with keys ``"model"`` (hyperparams passed to the model class) and
    optionally ``"training"`` (trainer-level settings).
    """
    if model_type not in _DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Expected one of {list(_DEFAULT_CONFIGS)}"
        )

    # Start from the default so custom configs only need to override what changes
    default_path = _DEFAULT_CONFIGS[model_type]
    with open(default_path, encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, encoding="utf-8") as config_file:
            overrides = yaml.safe_load(config_file) or {}
        # Deep-merge: override section by section
        for section, values in overrides.items():
            if section == "evaluation":
                cfg[section] = {} if values in (None, {}) else dict(values)
                continue
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section].update(values)
            else:
                cfg[section] = values

    # Normalise list-valued fields (YAML loads them as lists already, but guard anyway)
    if "model" in cfg and "hidden_dims" in cfg["model"]:
        cfg["model"]["hidden_dims"] = list(cfg["model"]["hidden_dims"])
    if "model" in cfg and "cnn_filters" in cfg["model"]:
        cfg["model"]["cnn_filters"] = list(cfg["model"]["cnn_filters"])
    if "model" in cfg and "dense_hidden_dims" in cfg["model"]:
        cfg["model"]["dense_hidden_dims"] = list(cfg["model"]["dense_hidden_dims"])

    validate_training_config(model_type, cfg)

    return cfg


def validate_training_config(model_type: str, cfg: dict) -> dict:
    """Validate the minimum config contract required to train *model_type*."""
    if model_type not in _DEFAULT_CONFIGS:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Expected one of {list(_DEFAULT_CONFIGS)}"
        )

    missing: list[str] = []
    required_sections = _REQUIRED_CONFIG_FIELDS.get(model_type, {})
    for section_name, field_names in required_sections.items():
        section = cfg.get(section_name)
        if not isinstance(section, dict):
            missing.append(section_name)
            continue
        for field_name in field_names:
            value = section.get(field_name)
            if value is None or value == "" or value == []:
                missing.append(f"{section_name}.{field_name}")

    if missing:
        raise ValueError(
            f"Invalid {model_type} training config. Missing required fields: {missing}"
        )

    return cfg


def model_params(cfg: dict) -> dict:
    """Extract the ``model`` section of a loaded config dict."""
    return dict(cfg.get("model", {}))


def training_params(cfg: dict) -> dict:
    """Extract the ``training`` section of a loaded config dict."""
    return dict(cfg.get("training", {}))


def data_params(cfg: dict) -> dict:
    """Extract the top-level ``data`` section of a loaded config dict."""
    return dict(cfg.get("data", {}))


def evaluation_params(cfg: dict) -> dict:
    """Extract the ``evaluation`` section of a loaded config dict."""
    return dict(cfg.get("evaluation", {}))


def sampling_params(cfg: dict) -> dict:
    """Extract the ``sampling`` section of a loaded config dict (CNN-SAE only)."""
    return dict(cfg.get("sampling", {}))
