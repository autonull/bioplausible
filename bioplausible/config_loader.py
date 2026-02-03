"""
Configuration Loader

Handles loading and parsing of experiment configuration files (YAML).
"""

import os
import yaml
from typing import Any, Dict

def load_config(path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the configuration.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    return config
