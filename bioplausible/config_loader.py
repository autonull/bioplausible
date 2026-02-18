"""
Configuration Loader

Handles loading and parsing of experiment configuration files (YAML).
"""

import os
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError


class ExperimentSchema(BaseModel):
    """Schema for validating experiment configurations."""

    model: str = Field(..., description="Name of the model (e.g., LoopedMLP)")
    task: str = Field(default="mnist", description="Task name")
    hyperparams: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    training: Dict[str, Any] = Field(
        default_factory=dict, description="Training settings (lr, epochs)"
    )
    description: Optional[str] = None


def load_config(path: str) -> Dict[str, Any]:
    """
    Load and validate experiment configuration from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary containing the validated configuration.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML config: {e}")

    try:
        # Validate against schema
        validated_config = ExperimentSchema(**raw_config)
        return validated_config.model_dump()
    except ValidationError as e:
        raise ValueError(f"Invalid configuration format: {e}")
