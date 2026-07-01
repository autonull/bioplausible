"""
Backward Compatibility Wrappers

This module provides backward-compatible wrappers for models that have been
refactored into learning rule optimizers. Old code continues to work but
with deprecation warnings encouraging migration to the new pattern.

Old Pattern (deprecated):
    model = FeedbackAlignmentEqProp(input_dim=784, hidden_dim=256, output_dim=10)

New Pattern (recommended):
    model = ModelZoo.get('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = FeedbackAlignment(model.parameters(), model=model)
"""

import warnings
import torch
import torch.nn as nn


def _deprecated_warning(old_name: str, model_name: str, optimizer_name: str) -> None:
    """Show deprecation warning with migration guidance."""
    warnings.warn(
        f"{old_name} is deprecated. Use the new pattern:\n"
        f"  from bioplausible import ModelZoo, {optimizer_name}\n"
        f"  model = ModelZoo.get('{model_name}', ...)\n"
        f"  optimizer = {optimizer_name}(model.parameters(), model=model)\n",
        DeprecationWarning,
        stacklevel=3,
    )


# ============================================================================
# FEEDBACK ALIGNMENT WRAPPERS
# ============================================================================


class FeedbackAlignmentEqProp(nn.Module):
    """
    Deprecated: Use LoopedMLP + FeedbackAlignment optimizer instead.

    This wrapper maintains backward compatibility while encouraging
    migration to the new architecture/optimizer separation pattern.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning(
            "FeedbackAlignmentEqProp", "looped_mlp", "FeedbackAlignment"
        )

        super().__init__()
        from bioplausible import ModelZoo

        # Create the actual model
        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        # Store optimizer params for later creation
        self.optimizer_params = kwargs
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def create_optimizer(
        self, lr: float = 0.01, **kwargs
    ) -> "FeedbackAlignment":  # noqa: F821
        """Create the learning rule optimizer."""
        from bioplausible.optimizers import FeedbackAlignment

        params = {**self.optimizer_params, "lr": lr, **kwargs}
        self.optimizer = FeedbackAlignment(
            self.model.parameters(), self.model, **params
        )
        return self.optimizer


class DirectFeedbackAlignmentEqProp(nn.Module):
    """
    Deprecated: Use LoopedMLP + DirectFA optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning("DirectFeedbackAlignmentEqProp", "looped_mlp", "DirectFA")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = kwargs
        self.optimizer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class AdaptiveFeedbackAlignment(nn.Module):
    """
    Deprecated: Use LoopedMLP + AdaptiveFA optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning("AdaptiveFeedbackAlignment", "looped_mlp", "AdaptiveFA")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class StochasticFA(nn.Module):
    """
    Deprecated: Use LoopedMLP + StochasticFA optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        noise_std: float = 0.1,
        **kwargs,
    ):
        _deprecated_warning("StochasticFA", "looped_mlp", "StochasticFA")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = {"noise_std": noise_std, **kwargs}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ContrastiveFeedbackAlignment(nn.Module):
    """
    Deprecated: Use LoopedMLP + ContrastiveFA optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning(
            "ContrastiveFeedbackAlignment", "looped_mlp", "ContrastiveFA"
        )

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# EQPROP VARIANT WRAPPERS
# ============================================================================


class HolomorphicEP(nn.Module):
    """
    Deprecated: Use LoopedMLP + HolomorphicEqProp optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning("HolomorphicEP", "looped_mlp", "HolomorphicEqProp")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class FiniteNudgeEP(nn.Module):
    """
    Deprecated: Use LoopedMLP + FiniteNudgeEqProp optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        beta: float = 1.0,
        **kwargs,
    ):
        _deprecated_warning("FiniteNudgeEP", "looped_mlp", "FiniteNudgeEqProp")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = {"beta": beta, **kwargs}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class LazyEqProp(nn.Module):
    """
    Deprecated: Use LoopedMLP + LazyEqProp optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        threshold: float = 0.01,
        **kwargs,
    ):
        _deprecated_warning("LazyEqProp", "looped_mlp", "LazyEqProp")

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = {"threshold": threshold, **kwargs}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# HEBBIAN WRAPPERS
# ============================================================================


class ContrastiveHebbianLearning(nn.Module):
    """
    Deprecated: Use LoopedMLP + CHL optimizer instead.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        output_dim: int = 10,
        **kwargs,
    ):
        _deprecated_warning(
            "ContrastiveHebbianLearning", "looped_mlp", "ContrastiveHebbianLearning"
        )

        super().__init__()
        from bioplausible import ModelZoo

        self.model = ModelZoo.get(
            "looped_mlp",
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer_params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

MIGRATION_GUIDE = """
Migration Guide: Old Models → New Pattern
==========================================

OLD (deprecated):
    from bioplausible.models import FeedbackAlignmentEqProp
    model = FeedbackAlignmentEqProp(input_dim=784, hidden_dim=256, output_dim=10)

    NEW (recommended):
    from bioplausible import ModelZoo, FeedbackAlignment

    model = ModelZoo.get('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = FeedbackAlignment(model.parameters(), model=model)

MAPPING:
    FeedbackAlignmentEqProp       → LoopedMLP + FeedbackAlignment
    DirectFeedbackAlignmentEqProp → LoopedMLP + DirectFA
    AdaptiveFeedbackAlignment     → LoopedMLP + AdaptiveFA
    StochasticFA                  → LoopedMLP + StochasticFA
    ContrastiveFeedbackAlignment  → LoopedMLP + ContrastiveFA
    HolomorphicEP                 → LoopedMLP + HolomorphicEqProp
    FiniteNudgeEP                 → LoopedMLP + FiniteNudgeEqProp
    LazyEqProp                    → LoopedMLP + LazyEqProp
    ContrastiveHebbianLearning    → LoopedMLP + ContrastiveHebbianLearning

Benefits of new pattern:
    • Any architecture can use any learning rule
    • Easy experimentation with different learning rules
    • Clear separation of concerns
    • Reduced code duplication
"""


__all__ = [
    # FA wrappers
    "FeedbackAlignmentEqProp",
    "DirectFeedbackAlignmentEqProp",
    "AdaptiveFeedbackAlignment",
    "StochasticFA",
    "ContrastiveFeedbackAlignment",
    # EqProp wrappers
    "HolomorphicEP",
    "FiniteNudgeEP",
    "LazyEqProp",
    # Hebbian wrappers
    "ContrastiveHebbianLearning",
    # Migration
    "MIGRATION_GUIDE",
]
