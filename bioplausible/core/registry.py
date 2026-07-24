"""
Unified Registry System for Bioplausible

Decorator-based + YAML-backed registry for all components:
- Models, Propagators, Optimizers, Sparsity, Metrics, DataLoaders, Tasks, etc.
Enables AutoScientist to query and compose intelligently.
"""

from __future__ import annotations

import builtins
import logging
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ComponentCategory(str, Enum):
    """Categories of components in the registry."""

    MODEL = "model"
    PROPAGATOR = "propagator"
    OPTIMIZER = "optimizer"
    SPARSITY = "sparsity"
    METRIC = "metric"
    DATA_LOADER = "data_loader"
    TASK = "task"
    CALLBACK = "callback"
    DOMAIN = "domain"


class Domain(str, Enum):
    """Supported domains."""

    VISION = "vision"
    LM = "lm"
    RL = "rl"
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    TABULAR = "tabular"
    SCIENTIFIC = "scientific"
    CONTINUAL = "continual"
    MULTITASK = "multitask"


class LocalityLevel(str, Enum):
    """Credit assignment locality level."""

    GLOBAL = "global"  # Full backprop
    LAYERWISE = "layerwise"  # Layer-local (Forward-Forward, Target Prop)
    LOCAL = "local"  # Neuron/synapse local (Hebbian, STDP, EquiTile)
    EQUILIBRIUM = "equilibrium"  # Energy-based (EqProp, CHL)
    FORWARD_ONLY = "forward-only"  # No backward pass (PEPITA, FF)


class ComputeProfile(str, Enum):
    """Compute profile for hardware affinity."""

    GPU = "gpu"
    CPU = "cpu"
    NEUROMORPHIC = "neuromorphic"
    ANALOG = "analog"
    OPTICAL = "optical"
    MEMRISTOR = "memristor"
    DISTRIBUTED = "distributed"


@dataclass
class ComponentMetadata:
    """Metadata for registered components enabling intelligent composition."""

    name: str
    category: ComponentCategory
    domains: list[Domain] = field(default_factory=lambda: [Domain.VISION])
    locality_level: LocalityLevel = LocalityLevel.GLOBAL
    compute_profile: ComputeProfile = ComputeProfile.GPU
    bio_plausibility_score: float = 0.5  # 0.0 = backprop, 1.0 = fully bio-plausible
    credit_assignment_type: str = (
        "gradient"  # gradient, equilibrium, hebbian, target, forward-only, spiking
    )
    requires_backward: bool = True
    memory_complexity: str = "O(N)"  # O(1) for MEP, O(N) standard
    min_params: int | None = None
    max_params: int | None = None
    typical_lr_range: tuple = (1e-5, 1e-1)
    typical_batch_size_range: tuple = (16, 512)
    supports_mixed_precision: bool = True
    supports_gradient_accumulation: bool = True
    supports_distributed: bool = False
    tags: list[str] = field(default_factory=list)
    citation: str | None = None
    description: str = ""
    version: str = "1.0.0"
    # Algorithm family tag (per REFACTOR2 §3.2): "eqprop", "fa", "hebbian",
    # "forward_only", "target_prop", "spiking", "predictive_coding", "backprop",
    # "mep", "equitile", etc. Directory layout mirrors this but `family` is the
    # canonical searchable attribute for grouping in the README/Registry queries.
    family: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Registry:
    """
    Central registry for all components.

    Supports:
    - Decorator-based registration: @register_component(...)
    - YAML-backed metadata (future)
    - Query by category, domain, locality, compute profile, etc.
    - Constraint satisfaction for AutoScientist
    """

    _components: dict[str, dict[str, Any]] = {}  # category -> {name: {cls, metadata}}
    _name_to_category: dict[str, str] = {}  # name -> category

    @classmethod
    def register(
        cls, category: ComponentCategory, name: str | None = None, **metadata_kwargs
    ) -> Callable[[type[T]], type[T]]:
        """
        Decorator to register a component.

        Usage:
            @register_component(
                category=ComponentCategory.MODEL,
                name="MyModel",
                domains=[Domain.VISION, Domain.LM],
                locality_level=LocalityLevel.EQUILIBRIUM,
                bio_plausibility_score=0.9,
                requires_backward=False
            )
            class MyModel(nn.Module):
                ...
        """
        if category not in cls._components:
            cls._components[category] = {}

        def decorator(component_cls: type[T]) -> type[T]:
            nonlocal name
            if name is None:
                name = component_cls.__name__

            if name in cls._components[category]:
                logger.warning(f"Overwriting component {category.value}/{name}")

            # Build metadata from kwargs, with sensible defaults from class
            metadata = ComponentMetadata(
                name=name, category=category, **metadata_kwargs
            )

            # Try to infer metadata from class attributes
            cls._infer_metadata(component_cls, metadata)

            cls._components[category][name] = {
                "class": component_cls,
                "metadata": metadata,
            }
            cls._name_to_category[name] = category.value

            # Attach metadata to class for easy access
            component_cls._registry_metadata = metadata
            component_cls._registry_name = name
            component_cls._registry_category = category

            logger.info(f"Registered {category.value}: {name}")
            return component_cls

        return decorator

    @classmethod
    def _infer_metadata(cls, component_cls: type, metadata: ComponentMetadata) -> None:
        """Infer metadata from class attributes if not explicitly provided."""
        # Check for class attributes that match metadata fields
        overrides = getattr(component_cls, "_registry_metadata_overrides", {})
        for fd in fields(ComponentMetadata):
            if fd.name in overrides:
                continue
            if (
                hasattr(component_cls, fd.name)
                and getattr(metadata, fd.name) == fd.default
            ):
                setattr(metadata, fd.name, getattr(component_cls, fd.name))

    @classmethod
    def _resolve_category(cls, category):
        """Resolve category from string or enum."""
        if isinstance(category, str):
            return ComponentCategory(category)
        return category

    @classmethod
    def get(cls, category: ComponentCategory, name: str) -> type:
        """Get a registered component class by category and name."""
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        if name not in cls._components[cat]:
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cls._components[cat][name]["class"]

    @classmethod
    def get_metadata(cls, category: ComponentCategory, name: str) -> ComponentMetadata:
        """Get metadata for a registered component."""
        cat = cls._resolve_category(category)
        if cat not in cls._components:
            raise ValueError(f"Unknown category: {cat}")
        if name not in cls._components[cat]:
            available = list(cls._components[cat].keys())
            raise ValueError(f"Unknown {cat.value}: {name}. Available: {available}")
        return cls._components[cat][name]["metadata"]

    @classmethod
    def list(
        cls, category: ComponentCategory | None = None
    ) -> dict[str, builtins.list[str]]:
        """List all registered components, optionally filtered by category."""
        if category is not None:
            # Accept both enum and string
            cat = ComponentCategory(category) if isinstance(category, str) else category
            if cat not in cls._components:
                return {cat.value: []}
            return {cat.value: list(cls._components[cat].keys())}
        return {cat.value: list(comps.keys()) for cat, comps in cls._components.items()}

    @classmethod
    def query(
        cls,
        category: ComponentCategory | None = None,
        domain: Domain | None = None,
        locality: LocalityLevel | None = None,
        compute: ComputeProfile | None = None,
        requires_backward: bool | None = None,
        min_bio_score: float | None = None,
        max_bio_score: float | None = None,
        tags: builtins.list[str] | None = None,
        credit_type: str | None = None,
    ) -> builtins.list[dict[str, Any]]:
        """
        Query registry with constraints - enables AutoScientist intelligent composition.

        Returns list of {name, category, class, metadata} matching all criteria.
        """
        results = []

        cats = [category] if category else list(cls._components.keys())
        categories = [
            cls._resolve_category(c) if isinstance(c, str) else c for c in cats
        ]

        for cat in categories:
            if cat not in cls._components:
                continue
            for name, info in cls._components[cat].items():
                meta = info["metadata"]

                # Apply filters
                if domain and domain not in meta.domains:
                    continue
                if locality and meta.locality_level != locality:
                    continue
                if compute and meta.compute_profile != compute:
                    continue
                if (
                    requires_backward is not None
                    and meta.requires_backward != requires_backward
                ):
                    continue
                if (
                    min_bio_score is not None
                    and meta.bio_plausibility_score < min_bio_score
                ):
                    continue
                if (
                    max_bio_score is not None
                    and meta.bio_plausibility_score > max_bio_score
                ):
                    continue
                if credit_type and meta.credit_assignment_type != credit_type:
                    continue
                if tags and not all(tag in meta.tags for tag in tags):
                    continue

                results.append({
                    "name": name,
                    "category": cat,
                    "class": info["class"],
                    "metadata": meta,
                })

        return results

    @classmethod
    def get_compatible(
        cls,
        model_name: str,
        model_category: ComponentCategory = ComponentCategory.MODEL,
    ) -> dict[ComponentCategory, builtins.list[dict[str, Any]]]:
        """
        Get components compatible with a given model.
        Used by AutoScientist to find valid optimizer/propagator combinations.
        """
        model_meta = cls.get_metadata(model_category, model_name)
        compat = {}

        for cat in ComponentCategory:
            if cat == model_category:
                continue
            compat[cat] = cls.query(
                category=cat,
                domain=model_meta.domains[0] if model_meta.domains else None,
                # Could add more sophisticated compatibility logic here
            )

        return compat

    @classmethod
    def clear(cls) -> None:
        """Clear registry (mainly for testing)."""
        cls._components.clear()
        cls._name_to_category.clear()

    @classmethod
    def export_yaml(cls, path: str) -> None:
        """
        Export all registered component metadata to a YAML file.

        This enables AutoScientist and external tools to inspect the full
        component catalog without importing Python modules.

        Args:
            path: Output YAML file path.
        """
        import yaml

        export_data = {}
        for category, comps in cls._components.items():
            cat_name = category.value if hasattr(category, "value") else str(category)
            export_data[cat_name] = {}
            for name, info in comps.items():
                meta = info["metadata"]
                entry = {
                    "name": meta.name,
                    "category": (
                        meta.category.value
                        if hasattr(meta.category, "value")
                        else str(meta.category)
                    ),
                    "domains": [
                        d.value if hasattr(d, "value") else str(d) for d in meta.domains
                    ],
                    "locality_level": (
                        meta.locality_level.value
                        if hasattr(meta.locality_level, "value")
                        else str(meta.locality_level)
                    ),
                    "compute_profile": (
                        meta.compute_profile.value
                        if hasattr(meta.compute_profile, "value")
                        else str(meta.compute_profile)
                    ),
                    "bio_plausibility_score": meta.bio_plausibility_score,
                    "credit_assignment_type": meta.credit_assignment_type,
                    "requires_backward": meta.requires_backward,
                    "memory_complexity": meta.memory_complexity,
                    "tags": meta.tags,
                    "description": meta.description,
                    "citation": meta.citation,
                    "version": meta.version,
                }
                export_data[cat_name][name] = entry

        with pathlib.Path(path).open("w") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        n_components = sum(len(v) for v in export_data.values())
        logger.info(f"Registry exported to {path}: {n_components} components")


# Convenience decorators
def register_model(name: str | None = None, **kwargs) -> Callable:
    """Register a model component."""
    return Registry.register(ComponentCategory.MODEL, name, **kwargs)


def register_propagator(name: str | None = None, **kwargs) -> Callable:
    """Register a propagator/learning rule component."""
    return Registry.register(ComponentCategory.PROPAGATOR, name, **kwargs)


def register_optimizer(name: str | None = None, **kwargs) -> Callable:
    """Register an optimizer component."""
    return Registry.register(ComponentCategory.OPTIMIZER, name, **kwargs)


def register_sparsity(name: str | None = None, **kwargs) -> Callable:
    """Register a sparsity component."""
    return Registry.register(ComponentCategory.SPARSITY, name, **kwargs)


def register_metric(name: str | None = None, **kwargs) -> Callable:
    """Register a metric component."""
    return Registry.register(ComponentCategory.METRIC, name, **kwargs)


def register_data_loader(name: str | None = None, **kwargs) -> Callable:
    """Register a data loader component."""
    return Registry.register(ComponentCategory.DATA_LOADER, name, **kwargs)


def register_task(name: str | None = None, **kwargs) -> Callable:
    """Register a task component."""
    return Registry.register(ComponentCategory.TASK, name, **kwargs)


def register_callback(name: str | None = None, **kwargs) -> Callable:
    """Register a callback component."""
    return Registry.register(ComponentCategory.CALLBACK, name, **kwargs)


def register_domain(name: str | None = None, **kwargs) -> Callable:
    """Register a domain component."""
    return Registry.register(ComponentCategory.DOMAIN, name, **kwargs)


# Legacy helpers (updated for new structure)
def get_model_registry() -> dict[str, type]:
    """Get model registry in legacy format."""
    models = {}
    for name, info in Registry._components.get(ComponentCategory.MODEL, {}).items():
        models[name] = info["class"]
    return models


def list_models() -> list[str]:
    """List available models."""
    return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())


# Export key types
__all__ = [
    "ComponentCategory",
    "ComponentMetadata",
    "ComputeProfile",
    "Domain",
    "LocalityLevel",
    "Registry",
    "get_model_registry",
    "list_models",
    "register_callback",
    "register_data_loader",
    "register_domain",
    "register_metric",
    "register_model",
    "register_optimizer",
    "register_propagator",
    "register_sparsity",
    "register_task",
]
