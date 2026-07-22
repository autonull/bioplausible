"""
Zoo Models Package

All models registered with the unified registry system.
"""

from typing import List

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    LocalityLevel,
    Registry,
    register_model,
)

# Import and register existing models
# These will be registered when imported
from bioplausible.zoo.models.registered_models import (
    MLP,
    EqPropMLP,
    EquiTile,
    ForwardForwardNet,
)


def _register_legacy_models():
    """Register legacy models from models/ package into the zoo registry."""
    from bioplausible.models import MODEL_REGISTRY as LEGACY_MODEL_REGISTRY

    registry_entry = Registry._components.setdefault(ComponentCategory.MODEL, {})

    for name, model_cls in LEGACY_MODEL_REGISTRY.items():
        if model_cls is not None:
            meta = ComponentMetadata(
                name=name,
                category=ComponentCategory.MODEL,
                description=f"Legacy model: {name}",
                domains=[],
                locality_level=LocalityLevel.GLOBAL,
            )
            registry_entry[name] = {
                "class": model_cls,
                "metadata": meta,
            }


# Register legacy models
_register_legacy_models()

__all__: List[str] = [
    "register_model",
    "MLP",
    "EqPropMLP",
    "ForwardForwardNet",
    "EquiTile",
]
