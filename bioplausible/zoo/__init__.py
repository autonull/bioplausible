"""
Bioplausible Model and Optimizer Zoo

A unified registry for all biologically plausible models and optimizers,
combining Bioplausible's EqProp models with MEP's optimizer strategies.

Quick Start:
    from bioplausible.zoo import ModelZoo, OptimizerZoo
    
    # Get a model
    model = ModelZoo.get('looped_mlp', input_size=784, hidden_size=256, output_size=10)
    
    # Get an optimizer
    optimizer = OptimizerZoo.get('smep', model.parameters(), model=model)
    
    # List available options
    print(ModelZoo.list_models())
    print(OptimizerZoo.list_optimizers())
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch.nn as nn


@dataclass
class ModelSpec:
    """Specification for a model in the zoo."""
    name: str
    category: str  # 'eqprop', 'feedback_alignment', 'hebbian', 'hybrid'
    model_class: Type[nn.Module]
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # 'vision', 'lm', 'deep', 'fast'
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category}): {self.description}"


@dataclass
class OptimizerSpec:
    """Specification for an optimizer in the zoo."""
    name: str
    category: str  # 'ep', 'backprop', 'natural_gradient', 'low_rank'
    optimizer_class: Any  # Can be a class or factory function
    description: str
    default_params: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)  # 'fast', 'stable', 'continual'
    
    def __str__(self) -> str:
        return f"{self.name} ({self.category}): {self.description}"


class ModelZoo:
    """
    Unified registry for biologically plausible models.
    
    Models are organized by category:
    - eqprop: Equilibrium Propagation variants
    - feedback_alignment: Feedback Alignment family
    - hebbian: Hebbian learning variants
    - hybrid: Hybrid approaches (Predictive Coding, etc.)
    """
    
    _registry: Dict[str, ModelSpec] = {}
    
    @classmethod
    def register(cls, spec: ModelSpec) -> None:
        """Register a model specification."""
        cls._registry[spec.name] = spec
    
    @classmethod
    def get(cls, name: str, **kwargs: Any) -> nn.Module:
        """
        Get a model by name.
        
        Args:
            name: Model name (e.g., 'looped_mlp', 'conv_eqprop').
            **kwargs: Override default parameters.
        
        Returns:
            Instantiated model.
        
        Raises:
            ValueError: If model name not found.
        """
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Model '{name}' not found. Available: {available}"
            )
        
        spec = cls._registry[name]
        params = {**spec.default_params, **kwargs}
        return spec.model_class(**params)
    
    @classmethod
    def list_models(cls, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[str]:
        """
        List available models.
        
        Args:
            category: Filter by category (e.g., 'eqprop').
            tags: Filter by tags (e.g., ['vision', 'fast']).
        
        Returns:
            List of model names.
        """
        models = list(cls._registry.values())
        
        if category:
            models = [m for m in models if m.category == category]
        
        if tags:
            models = [m for m in models if all(tag in m.tags for tag in tags)]
        
        return sorted([m.name for m in models])
    
    @classmethod
    def describe(cls, name: str) -> str:
        """Get detailed description of a model."""
        if name not in cls._registry:
            return f"Model '{name}' not found"
        spec = cls._registry[name]
        return (
            f"Name: {spec.name}\n"
            f"Category: {spec.category}\n"
            f"Description: {spec.description}\n"
            f"Tags: {', '.join(spec.tags)}\n"
            f"Default params: {spec.default_params}"
        )
    
    @classmethod
    def get_spec(cls, name: str) -> Optional[ModelSpec]:
        """Get the full specification for a model."""
        return cls._registry.get(name)


class OptimizerZoo:
    """
    Unified registry for biologically plausible optimizers.
    
    Optimizers are organized by category:
    - ep: Equilibrium Propagation optimizers
    - backprop: Standard backpropagation with enhancements
    - natural_gradient: Fisher-based preconditioning
    - low_rank: Low-rank approximations (Dion)
    """
    
    _registry: Dict[str, OptimizerSpec] = {}
    
    @classmethod
    def register(cls, spec: OptimizerSpec) -> None:
        """Register an optimizer specification."""
        cls._registry[spec.name] = spec
    
    @classmethod
    def get(cls, name: str, params, **kwargs: Any) -> Any:
        """
        Get an optimizer by name.
        
        Args:
            name: Optimizer name (e.g., 'smep', 'muon_backprop').
            params: Model parameters to optimize.
            **kwargs: Override default parameters.
        
        Returns:
            Instantiated optimizer.
        
        Raises:
            ValueError: If optimizer name not found.
        """
        if name not in cls._registry:
            available = ', '.join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Optimizer '{name}' not found. Available: {available}"
            )
        
        spec = cls._registry[name]
        params_dict = {**spec.default_params, **kwargs}
        
        # Call the optimizer factory/class
        if callable(spec.optimizer_class):
            return spec.optimizer_class(params, **params_dict)
        else:
            raise ValueError(f"Optimizer '{name}' has non-callable optimizer_class")
    
    @classmethod
    def list_optimizers(cls, category: Optional[str] = None, tags: Optional[List[str]] = None) -> List[str]:
        """
        List available optimizers.
        
        Args:
            category: Filter by category (e.g., 'ep').
            tags: Filter by tags (e.g., ['fast', 'stable']).
        
        Returns:
            List of optimizer names.
        """
        optims = list(cls._registry.values())
        
        if category:
            optims = [o for o in optims if o.category == category]
        
        if tags:
            optims = [o for o in optims if all(tag in o.tags for tag in tags)]
        
        return sorted([o.name for o in optims])
    
    @classmethod
    def describe(cls, name: str) -> str:
        """Get detailed description of an optimizer."""
        if name not in cls._registry:
            return f"Optimizer '{name}' not found"
        spec = cls._registry[name]
        return (
            f"Name: {spec.name}\n"
            f"Category: {spec.category}\n"
            f"Description: {spec.description}\n"
            f"Tags: {', '.join(spec.tags)}\n"
            f"Default params: {spec.default_params}"
        )
    
    @classmethod
    def get_spec(cls, name: str) -> Optional[OptimizerSpec]:
        """Get the full specification for an optimizer."""
        return cls._registry.get(name)


# Convenience functions
def get_model(name: str, **kwargs: Any) -> nn.Module:
    """Get a model from the zoo."""
    return ModelZoo.get(name, **kwargs)


def get_optimizer(name: str, params, **kwargs: Any) -> Any:
    """Get an optimizer from the zoo."""
    return OptimizerZoo.get(name, params, **kwargs)


def list_models(category: Optional[str] = None) -> List[str]:
    """List available models."""
    return ModelZoo.list_models(category=category)


def list_optimizers(category: Optional[str] = None) -> List[str]:
    """List available optimizers."""
    return OptimizerZoo.list_optimizers(category=category)


__all__ = [
    # Classes
    "ModelZoo",
    "OptimizerZoo",
    "ModelSpec",
    "OptimizerSpec",
    # Convenience functions
    "get_model",
    "get_optimizer",
    "list_models",
    "list_optimizers",
]
