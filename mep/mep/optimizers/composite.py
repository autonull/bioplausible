"""
Composite Optimizer: Strategy pattern-based optimizer.

This module provides the main optimizer class that composes
various strategies for gradient computation, update transformation,
constraints, and error feedback.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import Optional, Callable, Any, Iterable, List, Dict, cast

from .strategies import (
    GradientStrategy,
    UpdateStrategy,
    ConstraintStrategy,
    FeedbackStrategy,
    NoConstraint,
    NoFeedback,
)
from .energy import EnergyFunction
from .inspector import ModelInspector
from .settling import Settler


class CompositeOptimizer(Optimizer):
    """
    Composable optimizer built from strategy components.
    
    Example usage:
        optimizer = CompositeOptimizer(
            model.parameters(),
            gradient=EPGradient(beta=0.5, settle_steps=20),
            update=MuonUpdate(ns_steps=5),
            constraint=SpectralConstraint(gamma=0.95),
            feedback=ErrorFeedback(beta=0.9),
            lr=0.02,
            model=model,
        )
    
    Attributes:
        model: The model being optimized (for EP).
        gradient: Strategy for computing gradients.
        update: Strategy for transforming gradients.
        constraint: Strategy for enforcing constraints.
        feedback: Strategy for error accumulation.
    """
    
    def __init__(
        self,
        params: Iterable[nn.Parameter],
        gradient: GradientStrategy,
        update: UpdateStrategy,
        constraint: Optional[ConstraintStrategy] = None,
        feedback: Optional[FeedbackStrategy] = None,
        lr: float = 0.02,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        model: Optional[nn.Module] = None,
        max_grad_norm: float = 10.0,
    ):
        """
        Initialize composite optimizer.
        
        Args:
            params: Iterable of parameters to optimize.
            gradient: Strategy for computing gradients.
            update: Strategy for transforming gradients to updates.
            constraint: Strategy for enforcing constraints (default: none).
            feedback: Strategy for error feedback (default: none).
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay coefficient.
            model: Model instance (required for EP gradient strategies).
            max_grad_norm: Maximum gradient norm for clipping.
        """
        # Validate
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        
        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)
        
        self.model = model
        self.gradient = gradient
        self.update = update
        self.constraint = constraint or NoConstraint()
        self.feedback = feedback or NoFeedback()

        # Utilities
        self._inspector = ModelInspector()

        # Get loss_type from gradient strategy if available
        # We access attributes that might not exist on the base protocol, so using getattr is safe
        loss_type = getattr(gradient, 'loss_type', 'mse')
        softmax_temperature = getattr(gradient, 'softmax_temperature', 1.0)
        self._energy_fn = EnergyFunction(
            loss_type=loss_type,
            softmax_temperature=softmax_temperature
        )

        # Cache for EP states (when using wrapped model)
        self._free_states: Optional[List[torch.Tensor]] = None
        self._nudged_states: Optional[List[torch.Tensor]] = None
        self._last_input: Optional[torch.Tensor] = None
        
        # Error feedback config (passed to update strategies)
        self._error_beta = getattr(feedback, 'beta', 0.9)
        self._use_error_feedback = not isinstance(feedback, NoFeedback)

    def step( # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        x: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Optional[float]:
        """
        Perform optimization step.

        Supports multiple calling conventions:

        1. Backprop mode:
            loss.backward()
            optimizer.step()

        2. EP mode with explicit arguments:
            optimizer.step(x=x, target=y)

        3. EP mode with wrapped model:
            output = model(x)  # Triggers free phase
            optimizer.step(target=y)  # Triggers nudged phase

        Args:
            closure: Optional closure for re-evaluating loss.
            x: Input tensor (required for EP mode).
            target: Target tensor (required for EP mode).
            **kwargs: Additional arguments passed to strategies.

        Returns:
            Loss value if closure provided, None otherwise.

        Raises:
            ValueError: If required arguments missing for EP mode.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Handle EP gradient computation (needs gradients enabled)
        if isinstance(self.gradient, (EPGradient, LocalEPGradient, NaturalGradient)):
            if x is None and self._last_input is None:
                raise ValueError(
                    "EP gradient strategies require x tensor. "
                    "Pass x to step() or call model(x) first."
                )

            if target is None:
                raise ValueError("EP gradient strategies require target tensor")

            # Since we verified self.gradient is one of the types that needs model/energy/structure,
            # we also need to ensure self.model is not None.
            if self.model is None:
                 raise ValueError("Model must be provided to CompositeOptimizer for EP strategies")

            # Get input
            x_input = x if x is not None else self._last_input

            # Since x_input was checked above (x or _last_input), it shouldn't be None unless logic is flawed.
            # But x is Optional, so x_input is Optional.
            if x_input is None:
                raise ValueError("Input tensor is None")

            # Compute gradients (this needs gradients enabled)
            # We call compute_gradients with the extra args.
            # Note: GradientStrategy protocol doesn't include energy_fn/structure_fn,
            # but EPGradient etc do. Since we did isinstance check, we know it's safeish,
            # but mypy might complain if we call it on 'self.gradient' which is GradientStrategy.
            # However, we pass them as kwargs, and self.gradient accepts **kwargs in protocol.
            self.gradient.compute_gradients(
                self.model,
                x_input,
                target,
                energy_fn=self._energy_fn,
                structure_fn=self._inspector.inspect,
                **kwargs
            )

        # Apply updates (no gradients needed here)
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    # Ensure state exists
                    state = self.state[param]

                    # Initialize momentum buffer
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)

                    # Error feedback handling:
                    # - DionUpdate handles error feedback internally (for low-rank SVD residuals)
                    # - MuonUpdate doesn't need EF (orthogonalization preserves info via rotation)
                    # - PlainUpdate could use EF but typically doesn't need it
                    # Pass error feedback config to update strategy via group_config
                    group["error_beta"] = self._error_beta
                    group["use_error_feedback"] = self._use_error_feedback

                    # Transform gradient (update strategy handles any needed error feedback internally)
                    update = self.update.transform_gradient(param, param.grad, state, group)

                    # Momentum
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(update)

                    # Weight decay + apply update
                    # In-place operations
                    param.data.mul_(1 - group["weight_decay"] * group["lr"])
                    param.data.add_(buf, alpha=-group["lr"])

                    # Constraint
                    self.constraint.enforce(param, state, group)

        return loss
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """
        Clear gradients.
        
        Args:
            set_to_none: If True, set grads to None (more memory efficient).
        """
        if self.param_groups:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            p.grad.zero_()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        state = super().state_dict()
        state["strategy_config"] = {
            "gradient": type(self.gradient).__name__,
            "update": type(self.update).__name__,
            "constraint": type(self.constraint).__name__,
            "feedback": type(self.feedback).__name__,
        }
        return cast(Dict[str, Any], state)


# Import after class definition to avoid circular imports
from .strategies.gradient import EPGradient, LocalEPGradient, NaturalGradient
from .strategies.feedback import ErrorFeedback
