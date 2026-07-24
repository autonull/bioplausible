"""
Zoo Package - Unified Component Registry

All models, propagators, optimizers, sparsity methods, and other components
are registered here with rich metadata for AutoScientist composition.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    register_callback,
    register_data_loader,
    register_domain,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
    register_task,
)

# Import submodules to trigger registration
from bioplausible.zoo import models, optimizers, propagators, sparsity

logger = logging.getLogger("bioplausible.zoo")


class _LegacyModelSpec:
    """Adapter providing legacy ModelSpec interface from Registry metadata."""

    __slots__ = (
        "credit_locality",
        "default_lr",
        "family",
        "model_type",
        "name",
        "requires_backward",
        "task_compat",
        "variant",
        "credit_assignment_type",
    )

    def __init__(self, meta: ComponentMetadata) -> None:
        self.name = meta.name
        # Infer family from tags or name
        tags = meta.tags or []
        self.family = next(
            (
                t
                for t in tags
                if t
                in {
                    "eqprop",
                    "fa",
                    "hebbian",
                    "forward_only",
                    "target_prop",
                    "spiking",
                    "predictive_coding",
                    "backprop",
                }
            ),
            "experimental",
        )
        # task_compat from domains
        self.task_compat = [d.value for d in meta.domains]
        # model_type from credit_assignment_type
        self.model_type = meta.credit_assignment_type
        # For backward compat with metamodel expecting credit_assignment_type
        self.credit_assignment_type = meta.credit_assignment_type
        # variant is not directly stored; could be in extra
        self.variant = meta.extra.get("variant")
        self.default_lr = meta.typical_lr_range[0] if meta.typical_lr_range else 1e-3
        self.credit_locality = meta.locality_level.value
        self.requires_backward = meta.requires_backward


def get_model_spec(name: str) -> _LegacyModelSpec:
    """Get a legacy-compatible ModelSpec from the Registry by model name."""
    meta = Registry.get_metadata(ComponentCategory.MODEL, name)
    if meta is None:
        raise ValueError(f"Model '{name}' not found in Registry")
    return _LegacyModelSpec(meta)


def load_weights(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    strict: bool = False,
    freeze_layers: bool = False,
) -> None:
    """Load weights from a checkpoint path into ``model``.

    Args:
        model: Target model whose state dict is updated in place.
        path: Path to a ``.pt``/``.pth`` checkpoint file.
        device: Device to map the loaded tensors onto.
        strict: If True, require an exact match of keys.
        freeze_layers: If True, freeze every parameter whose name appears in
            the loaded state dict (useful for transfer-learning probes).
    """
    if not path:
        return
    try:
        logger.info("Loading weights from %s", path)
        state_dict = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        if missing:
            logger.info("Missing keys: %d", len(missing))
        if unexpected:
            logger.info("Unexpected keys: %d", len(unexpected))
        if freeze_layers:
            logger.info("Freezing loaded layers for transfer learning")
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
                else:
                    logger.info("  -> %s remains trainable", name)
    except Exception:
        logger.exception("Failed to load weights from %s", path)


def get_models_for_task(
    domain: Domain, locality: LocalityLevel = None, requires_backward: bool = None
):
    return Registry.query(
        category="model",
        domain=domain,
        locality_level=locality,
        requires_backward=requires_backward,
    )


def get_propagators_for_model(model_name: str):
    model_meta = Registry.get_metadata("model", model_name)
    return Registry.query(
        category="propagator",
        locality_level=model_meta.locality_level,
        requires_backward=model_meta.requires_backward,
    )


def get_optimizers_for_propagator(propagator_name: str):
    prop_meta = Registry.get_metadata("propagator", propagator_name)
    return Registry.query(
        category="optimizer",
        requires_backward=prop_meta.requires_backward,
    )


__all__ = [
    # Registry
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    # Registration decorators
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_sparsity",
    "register_metric",
    "register_data_loader",
    "register_task",
    "register_callback",
    "register_domain",
    # Submodules
    "models",
    "propagators",
    "optimizers",
    "sparsity",
    # Discovery helpers
    "get_models_for_task",
    "get_propagators_for_model",
    "get_optimizers_for_propagator",
    # Weight utilities
    "load_weights",
]
