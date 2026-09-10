"""
Configuration loader for the agent determinacy test suite.

This is the single point of entry for config.yaml. Every other module
in the suite calls load_config() rather than opening the YAML directly.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load and return the full configuration dictionary from config.yaml.

    Validates that all required top-level keys are present. Does not
    validate nested values — individual modules are responsible for
    handling missing optional keys with safe defaults.

    Args:
        path: Path to the YAML config file. Defaults to config.yaml
              in the project root (i.e. wherever the suite is run from).

    Returns:
        Parsed config as a plain dict.

    Raises:
        FileNotFoundError: If the config file does not exist at the given path.
        KeyError: If a required top-level key is missing from the config.
        yaml.YAMLError: If the file is not valid YAML.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Config file not found at {resolved}. "
            "Make sure config.yaml exists in the project root."
        )

    with resolved.open("r", encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh)

    _validate_required_keys(cfg, resolved)

    logger.debug("Config loaded from %s", resolved)
    return cfg


def _validate_required_keys(cfg: dict[str, Any], path: Path) -> None:
    """
    Assert that all required top-level config sections are present.

    Args:
        cfg:  Parsed config dict.
        path: Path used in error messages.

    Raises:
        KeyError: If a required section is missing.
    """
    required = {"agent", "runs", "groups", "embeddings", "thresholds", "output"}
    missing = required - cfg.keys()
    if missing:
        raise KeyError(
            f"config.yaml at {path} is missing required sections: {sorted(missing)}"
        )