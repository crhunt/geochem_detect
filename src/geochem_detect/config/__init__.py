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
    with open(default_path) as f:
        cfg = yaml.safe_load(f)

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        # Deep-merge: override section by section
        for section, values in overrides.items():
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

    return cfg


def model_params(cfg: dict) -> dict:
    """Extract the ``model`` section of a loaded config dict."""
    return dict(cfg.get("model", {}))


def training_params(cfg: dict) -> dict:
    """Extract the ``training`` section of a loaded config dict."""
    return dict(cfg.get("training", {}))


def sampling_params(cfg: dict) -> dict:
    """Extract the ``sampling`` section of a loaded config dict (CNN-SAE only)."""
    return dict(cfg.get("sampling", {}))
