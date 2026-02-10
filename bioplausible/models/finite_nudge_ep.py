"""
Finite-Nudge Equilibrium Propagation

Implementation of EqProp in the regime of finite nudging (large beta),
where the algorithm can be interpreted through Gibbs-Boltzmann statistics.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BioModel, ModelConfig, register_model
from .standard_eqprop import StandardEqProp


@register_model("finite_nudge_ep")
class FiniteNudgeEP(StandardEqProp):
    """
    Finite-Nudge EqProp.
    Operates with large beta values (e.g. beta=1.0) where the infinitesimal
    approximation of the gradient is replaced by a finite difference
    that optimizes a global energy bound.
    """

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        # Handle beta from kwargs if passed directly
        if "beta" in kwargs:
            self.beta = kwargs["beta"]
        elif self.config and self.config.extra and "beta" in self.config.extra:
            self.beta = self.config.extra["beta"]

        # Finite Nudge typically requires larger beta
        if self.beta < 0.5:
            self.beta = 1.0

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Training step handling finite beta.
        Uses the same contrastive rule, which is valid for finite nudges
        under the generalized objective function.
        """
        # We reuse the standard implementation but ensure we monitor the energy gap
        # The core update (h_nudged - h_free)/beta is the finite difference derivative.

        metrics = super().train_step(x, y)

        # Add energy metrics if possible?
        # Calculating energy would require access to states, but super().train_step
        # doesn't return them.
        # We trust the base implementation.

        return metrics

    @classmethod
    def build(
        cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs
    ):
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            extra=kwargs,
        )

        # Override defaults if provided in kwargs
        if "equilibrium_steps" in kwargs:
            config.equilibrium_steps = kwargs["equilibrium_steps"]
            config.max_steps = kwargs["equilibrium_steps"]
        if "beta" in kwargs:
            config.beta = kwargs["beta"]

        return cls(config=config).to(device)
