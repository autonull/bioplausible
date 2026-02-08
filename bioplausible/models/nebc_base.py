"""
NEBC Base Classes - Modular Foundation for Bio-Plausible Algorithms

This module provides:
1. NEBCBase - Abstract base class for all NEBC algorithms
2. Common utilities for spectral normalization, training, and evaluation
3. Extensible interface for adding new bio-plausible algorithms

All NEBC algorithms test spectral normalization as a "stability unlock".
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class NEBCBase(nn.Module, ABC):
    """
    Abstract base class for NEBC (Nobody Ever Bothered Club) algorithms.

    All NEBC algorithms share:
    - Spectral normalization option
    - Lipschitz constant tracking
    - Common evaluation interface
    - Ablation study support (with/without SN)
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_spectral_norm = use_spectral_norm
        self.max_steps = max_steps
        self.lipschitz_mode = lipschitz_mode

        # Subclasses must initialize their layers
        self._build_layers()

    @abstractmethod
    def _build_layers(self):
        """Build the network layers. Called from __init__."""
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the network."""
        pass

    def apply_spectral_norm(self, layer: nn.Module) -> nn.Module:
        """Apply spectral normalization to a layer if enabled."""
        if self.use_spectral_norm and isinstance(layer, (nn.Linear, nn.Conv2d)):
            return spectral_norm(layer, n_power_iterations=5)
        return layer

    def compute_lipschitz(self) -> float:
        """Compute the maximum Lipschitz constant across all layers."""
        max_L = 0.0
        with torch.no_grad():
            for module in self.modules():
                # Access .weight property if available (handles spectral_norm)
                if hasattr(module, "weight") and isinstance(
                    module.weight, torch.Tensor
                ):
                    w = module.weight
                    if w.dim() >= 2:
                        if self.lipschitz_mode == "power_iteration":
                            # Optimization: Use Power Iteration (O(N^2))
                            L = self._approx_spectral_norm(w)
                        elif self.lipschitz_mode == "svd":
                            # Exact SVD (O(N^3))
                            w_mat = w.view(w.size(0), -1)
                            s = torch.linalg.svdvals(w_mat)
                            L = s[0].item() if s.numel() > 0 else 0.0
                        else:
                            # Fallback to SVD for safety
                            w_mat = w.view(w.size(0), -1)
                            s = torch.linalg.svdvals(w_mat)
                            L = s[0].item() if s.numel() > 0 else 0.0

                        max_L = max(max_L, L)
        return max_L

    def _approx_spectral_norm(self, weight: torch.Tensor, n_iter: int = 10) -> float:
        """Approximate spectral norm using power iteration (faster than SVD)."""
        if weight.dim() < 2:
            return 0.0

        w_mat = weight.view(weight.size(0), -1)
        out_dim, in_dim = w_mat.shape

        u = torch.randn(out_dim, device=weight.device)

        # Power iteration
        for _ in range(n_iter):
            # v = W^T u / ||W^T u||
            v = torch.mv(w_mat.t(), u)
            v = F.normalize(v, dim=0, eps=1e-12)

            # u = W v / ||W v||
            u = torch.mv(w_mat, v)
            u = F.normalize(u, dim=0, eps=1e-12)

        # sigma = u^T W v
        return torch.dot(u, torch.mv(w_mat, v)).item()

    def get_stats(self) -> Dict[str, float]:
        """Get algorithm-specific statistics for reporting."""
        return {
            "lipschitz": self.compute_lipschitz(),
            "num_params": sum(p.numel() for p in self.parameters()),
            "num_layers": self.num_layers,
            "spectral_norm": self.use_spectral_norm,
        }

    @classmethod
    def create_pair(
        cls, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
    ) -> Tuple["NEBCBase", "NEBCBase"]:
        """Create a pair of models: with and without spectral norm (for ablation)."""
        with_sn = cls(
            input_dim, hidden_dim, output_dim, use_spectral_norm=True, **kwargs
        )
        without_sn = cls(
            input_dim, hidden_dim, output_dim, use_spectral_norm=False, **kwargs
        )
        return with_sn, without_sn


class NEBCRegistry:
    """
    Registry for NEBC algorithms.

    Enables dynamic registration and discovery of new algorithms.
    Use @register_nebc decorator to add new algorithms.
    """

    _algorithms: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register an NEBC algorithm."""

        def decorator(algorithm_cls: type):
            cls._algorithms[name] = algorithm_cls
            algorithm_cls.algorithm_name = name
            return algorithm_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type:
        """Get an algorithm class by name."""
        if name not in cls._algorithms:
            available = list(cls._algorithms.keys())
            raise ValueError(f"Unknown algorithm: {name}. Available: {available}")
        return cls._algorithms[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered algorithms."""
        return list(cls._algorithms.keys())

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> NEBCBase:
        """Create an instance of an algorithm by name."""
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
