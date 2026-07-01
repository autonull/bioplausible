"""
Bio-Plausible Model Base Classes

Unified foundation for all biologically plausible learning algorithms and models.
Combines functionality for:
- Spectral Normalization (Stability)
- Lipschitz Constant Tracking
- Custom Training Steps (Heuristic/Contrastive updates)
- Configuration Management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

# Re-export for backward compatibility with models importing from .base
from bioplausible.models.registry import register_model  # noqa: F401


@dataclass
class ModelConfig:
    """Configuration for a bio-plausible model."""

    name: str
    input_dim: int
    output_dim: int
    hidden_dims: List[int] = field(default_factory=list)

    # Training hyperparameters
    learning_rate: float = 0.001
    beta: float = 0.2  # For EqProp
    # Equilibrium Steps (also known as max_steps)
    equilibrium_steps: int = 30
    max_steps: int = 30  # Alias for equilibrium_steps to match NEBCBase

    # Architecture
    use_spectral_norm: bool = True
    activation: str = "silu"
    lipschitz_mode: str = "power_iteration"  # "power_iteration" or "svd"

    # Additional kwargs
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # input_dim can be 0 for Conv models (placeholder)
        if isinstance(self.input_dim, tuple):
            import math

            self.input_dim = math.prod(self.input_dim)
        assert self.input_dim >= 0
        assert self.output_dim > 0

        # Sync steps if one is changed
        if self.equilibrium_steps != 30 and self.max_steps == 30:
            self.max_steps = self.equilibrium_steps
        elif self.max_steps != 30 and self.equilibrium_steps == 30:
            self.equilibrium_steps = self.max_steps


class BioModel(nn.Module, ABC):
    """
    Abstract base class for all bio-plausible models/algorithms.

    Unifies:
    - NEBCBase (Spectral Norm, Lipschitz)
    - BaseAlgorithm (train_step, config)
    """

    algorithm_name: str = "BioModel"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        # Legacy/Direct init support
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        lipschitz_mode: str = "power_iteration",
        **kwargs,
    ):
        super().__init__()

        # Handle config vs direct args
        if config is None:
            if input_dim is None or output_dim is None:
                # If inherited directly without config/dims (e.g. specialized subclass),
                # allow skipping, but warn/fail if methods need them.
                # However, for consistency with NEBCBase, we might need these.
                # Let's assume subclasses will call super().__init__ properly.
                pass

            # Legacy/Direct init
            self.config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim if input_dim is not None else 0,
                output_dim=output_dim if output_dim is not None else 0,
                hidden_dims=[hidden_dim] if hidden_dim else [],
                use_spectral_norm=use_spectral_norm,
                max_steps=max_steps,
                lipschitz_mode=lipschitz_mode,
                extra=kwargs,
            )
        else:
            self.config = config
            # Ensure max_steps override from kwargs if provided
            if "max_steps" in kwargs:
                self.config.max_steps = kwargs["max_steps"]
                self.config.equilibrium_steps = kwargs["max_steps"]

        # Shortcuts for convenience
        self.input_dim = self.config.input_dim
        self.output_dim = self.config.output_dim
        self.hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 0
        self.use_spectral_norm = self.config.use_spectral_norm
        self.max_steps = self.config.max_steps
        self.lipschitz_mode = self.config.lipschitz_mode

        # Helper for activation
        self.activation = self._get_activation(self.config.activation)

        # NEBCBase compatibility: Check for _build_layers hook
        if hasattr(self, "_build_layers"):
            self._build_layers()

    def _get_activation(self, name: str) -> nn.Module:
        if name == "silu":
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        return nn.ReLU()

    def apply_spectral_norm(self, layer: nn.Module) -> nn.Module:
        """Apply spectral normalization to a layer if enabled."""
        if self.use_spectral_norm and isinstance(layer, (nn.Linear, nn.Conv2d)):
            return spectral_norm(layer, n_power_iterations=5)
        return layer

    def _get_spectral_normalized_weight(self, layer: nn.Module) -> torch.Tensor:
        """Get spectral normalized weight, with caching in eval mode."""
        # Check for cached weight in eval mode
        if not self.training and hasattr(layer, "_cached_sn_weight"):
            return layer._cached_sn_weight

        # Compute normalized weight (.weight triggers spectral_norm if present)
        if hasattr(layer, "parametrizations") and hasattr(
            layer.parametrizations, "weight"
        ):
            weight = layer.weight
        else:
            weight = layer.weight

        # Cache in eval mode
        if not self.training:
            layer._cached_sn_weight = weight.detach()

        return weight

    def train(self, mode: bool = True):
        """Override train to clear caches."""
        super().train(mode)
        if mode:  # Entering training mode, clear cache
            for module in self.modules():
                if hasattr(module, "_cached_sn_weight"):
                    delattr(module, "_cached_sn_weight")
        return self

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
            "spectral_norm": self.use_spectral_norm,
        }

    @classmethod
    def create_pair(
        cls, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
    ) -> Tuple["BioModel", "BioModel"]:
        """Create a pair of models: with and without spectral norm (for ablation)."""
        # Note: Uses direct init assuming arguments match __init__
        with_sn = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=True,
            **kwargs,
        )
        without_sn = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=False,
            **kwargs,
        )
        return with_sn, without_sn

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Custom training step.
        Override this for algorithms that don't use standard autograd (e.g. EqProp, FA).
        If not overridden, EqPropTrainer will assume standard BPTT/Autograd can be used
        if this returns None or raises NotImplementedError, or EqPropTrainer handles BPTT.

        # For BaseAlgorithm compatibility, allow abstract or default to BPTT.
        """
        raise NotImplementedError(
            "Model does not implement custom train_step. Use BPTT."
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        """
        Generic build method for BioModels.
        Creates a ModelConfig from spec and args, then instantiates the model.
        """
        # Construct config
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim if input_dim is not None else 0,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            learning_rate=getattr(spec, "default_lr", 0.001),
            beta=0.1,  # Default, overridden by kwargs if needed
            equilibrium_steps=20,  # Default
            use_spectral_norm=True,
            extra=kwargs,
        )

        # Allow kwargs to override config defaults if they match config fields
        if "beta" in kwargs:
            config.beta = kwargs["beta"]
        if "equilibrium_steps" in kwargs:
            config.equilibrium_steps = kwargs["equilibrium_steps"]
            config.max_steps = kwargs["equilibrium_steps"]

        model = cls(config=config).to(device)
        return model
