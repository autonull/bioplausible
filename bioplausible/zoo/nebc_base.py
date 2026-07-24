"""
NEBC Base Classes - Modular Foundation for Bio-Plausible Algorithms

This module provides:
1. NEBCBase - Abstract base class for all NEBC algorithms
2. Common utilities for spectral normalization, training, and evaluation
3. Extensible interface for adding new bio-plausible algorithms

All NEBC algorithms test spectral normalization as a "stability unlock".
"""

from abc import ABC
from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn.functional as F

from .base import BioModel


class NEBCBase(BioModel, ABC):
    """
    Abstract base class for NEBC (Nobody Ever Bothered Club) algorithms.

    Now inherits from BioModel for unified architecture.
    Kept for backward compatibility.
    """

    algorithm_name: str = "NEBCBase"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        lipschitz_mode: str = "power_iteration",
        **kwargs,
    ):
        self.num_layers = num_layers

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            max_steps=max_steps,
            lipschitz_mode=lipschitz_mode,
            **kwargs,
        )

    # _build_layers, forward, apply_spectral_norm, compute_lipschitz, etc.
    # are inherited from BioModel.
    # Subclasses must implement _build_layers and forward.

    def get_stats(self) -> Dict[str, float]:
        """Get algorithm-specific statistics for reporting."""
        stats = super().get_stats()
        stats["num_layers"] = self.num_layers
        return stats

    @classmethod
    def create_pair(
        cls, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
    ) -> Tuple["NEBCBase", "NEBCBase"]:
        """Create a pair of models: with and without spectral norm (for ablation)."""
        return super().create_pair(input_dim, hidden_dim, output_dim, **kwargs)


class NEBCRegistry:
    """
    Registry for NEBC algorithms.
    Wraps the core Registry for NEBC-specific registration.
    """

    @classmethod
    def register(cls, name: str):
        from bioplausible.core.registry import register_model

        return register_model(name=name)

    @classmethod
    def get(cls, name: str) -> type:
        from bioplausible.core.registry import ComponentCategory
        from bioplausible.core.registry import Registry

        return Registry.get(ComponentCategory.MODEL, name)

    @classmethod
    def list_all(cls) -> List[str]:
        from bioplausible.core.registry import ComponentCategory
        from bioplausible.core.registry import Registry

        return list(Registry._components.get(ComponentCategory.MODEL, {}).keys())

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> NEBCBase:
        algorithm_cls = cls.get(name)
        return algorithm_cls(*args, **kwargs)


# Convenience decorator
register_nebc = NEBCRegistry.register


def train_nebc_model(
    model: NEBCBase,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 50,
    lr: float = 0.01,
    verbose: bool = True,
) -> List[float]:
    """
    Standard training loop for NEBC models.

    Returns list of losses for analysis.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 5) == 0:
            acc = (out.argmax(dim=1) == y).float().mean().item() * 100
            L = model.compute_lipschitz()
            print(
                f"  [{model.algorithm_name}] Epoch {epoch+1}/{epochs}: "
                f"loss={loss.item():.3f}, acc={acc:.1f}%, L={L:.3f}"
            )

    return losses


def evaluate_nebc_model(
    model: NEBCBase,
    X: torch.Tensor,
    y: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate an NEBC model and return comprehensive metrics.
    """
    model.eval()
    with torch.no_grad():
        out = model(X)
        loss = F.cross_entropy(out, y).item()
        acc = (out.argmax(dim=1) == y).float().mean().item()
        L = model.compute_lipschitz()
    model.train()

    return {"accuracy": acc, "loss": loss, "lipschitz": L, **model.get_stats()}


def run_nebc_ablation(
    algorithm_name: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    epochs: int = 50,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Run ablation study comparing algorithm with/without spectral norm.

    Returns dict with 'with_sn' and 'without_sn' results.
    """
    algorithm_cls = NEBCRegistry.get(algorithm_name)

    results = {}
    for use_sn in [True, False]:
        label = "with_sn" if use_sn else "without_sn"
        print(f"\n  Training {algorithm_name} ({label})...")

        model = algorithm_cls(
            input_dim, hidden_dim, output_dim, use_spectral_norm=use_sn, **kwargs
        )

        train_nebc_model(model, X_train, y_train, epochs=epochs)
        metrics = evaluate_nebc_model(model, X_test, y_test)
        results[label] = metrics

    # Compute delta
    results["delta"] = {
        "accuracy": results["with_sn"]["accuracy"] - results["without_sn"]["accuracy"],
        "lipschitz": results["without_sn"]["lipschitz"]
        - results["with_sn"]["lipschitz"],
        "sn_stabilizes": results["with_sn"]["lipschitz"] <= 1.05,
    }

    return results
