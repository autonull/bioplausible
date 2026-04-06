"""Configuration loading helpers."""

import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two dictionaries.

    Args:
        base: Base configuration.
        override: Override configuration.

    Returns:
        Merged configuration.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)

    return result


def load_config(
    config_path: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load configuration with layered overrides.

    Args:
        config_path: Explicit config file path.
        experiment_name: Optional experiment name used to resolve
            `experiments/<name>/config.yaml`.

    Returns:
        Resolved configuration dictionary.
    """
    project_root = Path(__file__).parent.parent
    default_config_path = project_root / "config.yaml"

    # 1. Start from repository defaults.
    if default_config_path.exists():
        with open(default_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # 2. Merge experiment-level overrides.
    if experiment_name:
        exp_config_path = project_root / "experiments" / experiment_name / "config.yaml"
        if exp_config_path.exists():
            with open(exp_config_path, "r", encoding="utf-8") as f:
                exp_config = yaml.safe_load(f) or {}
            config = deep_merge(config, exp_config)

    # 3. Apply the explicit config file last.
    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            custom_config = yaml.safe_load(f) or {}
        config = deep_merge(config, custom_config)

    # Resolve `${ENV_VAR}` placeholders.
    config = _resolve_env_vars(config)

    return config


def load_experiment_config(experiment_name: str) -> Dict[str, Any]:
    """
    Convenience wrapper for loading an experiment config.

    Args:
        experiment_name: Experiment name.

    Returns:
        Resolved configuration.
    """
    return load_config(experiment_name=experiment_name)


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve `${VAR_NAME}` placeholders from the environment."""
    if isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, "")
    return obj
