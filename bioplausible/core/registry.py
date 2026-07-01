"""
Unified Registry System for Bioplausible

Decorator-based + YAML-backed registry for all components:
- Models, Propagators, Optimizers, Sparsity, Metrics, DataLoaders, Tasks, etc.
Enables AutoScientist to query and compose intelligently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

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
    domains: List[Domain] = field(default_factory=lambda: [Domain.VISION])
    locality_level: LocalityLevel = LocalityLevel.GLOBAL
    compute_profile: ComputeProfile = ComputeProfile.GPU
    bio_plausibility_score: float = 0.5  # 0.0 = backprop, 1.0 = fully bio-plausible
    credit_assignment_type: str = (
        "gradient"  # gradient, equilibrium, hebbian, target, forward-only, spiking
    )
    requires_backward: bool = True
    memory_complexity: str = "O(N)"  # O(1) for MEP, O(N) standard
    min_params: Optional[int] = None
    max_params: Optional[int] = None
    typical_lr_range: tuple = (1e-5, 1e-1)
    typical_batch_size_range: tuple = (16, 512)
    supports_mixed_precision: bool = True
    supports_gradient_accumulation: bool = True
    supports_distributed: bool = False
    tags: List[str] = field(default_factory=list)
    citation: Optional[str] = None
    description: str = ""
    version: str = "1.0.0"
    extra: Dict[str, Any] = field(default_factory=dict)


class Registry:
    """
    Central registry for all components.

    Supports:
    - Decorator-based registration: @register_component(...)
    - YAML-backed metadata (future)
    - Query by category, domain, locality, compute profile, etc.
    - Constraint satisfaction for AutoScientist
    """

    _components: Dict[str, Dict[str, Any]] = {}  # category -> {name: {cls, metadata}}
    _name_to_category: Dict[str, str] = {}  # name -> category

    @classmethod
    def register(
        cls, category: ComponentCategory, name: Optional[str] = None, **metadata_kwargs
    ) -> Callable[[Type[T]], Type[T]]:
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

        def decorator(component_cls: Type[T]) -> Type[T]:
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
    def _infer_metadata(cls, component_cls: Type, metadata: ComponentMetadata) -> None:
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
    def get(cls, category: ComponentCategory, name: str) -> Type:
        """Get a registered component class by category and name."""
        if category not in cls._components:
            raise ValueError(f"Unknown category: {category}")
        if name not in cls._components[category]:
            available = list(cls._components[category].keys())
            raise ValueError(
                f"Unknown {category.value}: {name}. Available: {available}"
            )
        return cls._components[category][name]["class"]

    @classmethod
    def get_metadata(cls, category: ComponentCategory, name: str) -> ComponentMetadata:
        """Get metadata for a registered component."""
        if category not in cls._components:
            raise ValueError(f"Unknown category: {category}")
        if name not in cls._components[category]:
            available = list(cls._components[category].keys())
            raise ValueError(
                f"Unknown {category.value}: {name}. Available: {available}"
            )
        return cls._components[category][name]["metadata"]

    @classmethod
    def list(cls, category: Optional[ComponentCategory] = None) -> Dict[str, List[str]]:
        """List all registered components, optionally filtered by category."""
        if category:
            if category not in cls._components:
                return {category.value: []}
            return {category.value: list(cls._components[category].keys())}
        return {cat.value: list(comps.keys()) for cat, comps in cls._components.items()}

    @classmethod
    def query(
        cls,
        category: Optional[ComponentCategory] = None,
        domain: Optional[Domain] = None,
        locality: Optional[LocalityLevel] = None,
        compute: Optional[ComputeProfile] = None,
        requires_backward: Optional[bool] = None,
        min_bio_score: Optional[float] = None,
        max_bio_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        credit_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query registry with constraints - enables AutoScientist intelligent composition.

        Returns list of {name, category, class, metadata} matching all criteria.
        """
        results = []

        categories = [category] if category else list(cls._components.keys())

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

                results.append(
                    {
                        "name": name,
                        "category": cat,
                        "class": info["class"],
                        "metadata": meta,
                    }
                )

        return results

    @classmethod
    def get_compatible(
        cls,
        model_name: str,
        model_category: ComponentCategory = ComponentCategory.MODEL,
    ) -> Dict[ComponentCategory, List[Dict[str, Any]]]:
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

        with open(path, "w") as f:
            yaml.dump(export_data, f, default_flow_style=False, sort_keys=False)

        n_components = sum(len(v) for v in export_data.values())
        logger.info(f"Registry exported to {path}: {n_components} components")


# Convenience decorators
def register_model(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a model component."""
    return Registry.register(ComponentCategory.MODEL, name, **kwargs)


def register_propagator(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a propagator/learning rule component."""
    return Registry.register(ComponentCategory.PROPAGATOR, name, **kwargs)


def register_optimizer(name: Optional[str] = None, **kwargs) -> Callable:
    """Register an optimizer component."""
    return Registry.register(ComponentCategory.OPTIMIZER, name, **kwargs)


def register_sparsity(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a sparsity component."""
    return Registry.register(ComponentCategory.SPARSITY, name, **kwargs)


def register_metric(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a metric component."""
    return Registry.register(ComponentCategory.METRIC, name, **kwargs)


def register_data_loader(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a data loader component."""
    return Registry.register(ComponentCategory.DATA_LOADER, name, **kwargs)


def register_task(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a task component."""
    return Registry.register(ComponentCategory.TASK, name, **kwargs)


def register_callback(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a callback component."""
    return Registry.register(ComponentCategory.CALLBACK, name, **kwargs)


def register_domain(name: Optional[str] = None, **kwargs) -> Callable:
    """Register a domain component."""
    return Registry.register(ComponentCategory.DOMAIN, name, **kwargs)


# For backward compatibility with existing model registry
def get_model_registry() -> Dict[str, Type]:
    """Get model registry in legacy format."""
    models = {}
    for name, info in Registry._components.get(ComponentCategory.MODEL, {}).items():
        models[name] = info["class"]
    return models


def list_models() -> List[str]:
    """List available models."""
    return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())


# Export key types
__all__ = [
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_sparsity",
    "register_metric",
    "register_data_loader",
    "register_task",
    "register_callback",
    "register_domain",
    "get_model_registry",
    "list_models",
]
