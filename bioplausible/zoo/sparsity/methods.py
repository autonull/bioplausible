"""Sparsity / pruning methods registered with the unified Zoo.

Each class implements a ``step()`` method that updates parameter masks
based on a different sparsity heuristic.  Registration is done with the
``@register_sparsity`` decorator so the methods are discoverable through
``Registry.list(ComponentCategory.SPARSITY)``.
"""

import torch

from bioplausible.core.registry import (
    ComputeProfile,
    Domain,
    LocalityLevel,
    register_sparsity,
)


@register_sparsity(
    name="TopKPruning",
    domains=[Domain.VISION, Domain.LM, Domain.TIMESERIES],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.7,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(k)",
    typical_lr_range=(1e-4, 1e-2),
    typical_batch_size_range=(32, 256),
    tags=["sparsity", "pruning", "top-k"],
    description=(
        "Top-K Scheduling: selectively updates top-k most active"
        " connections per layer per step"
    ),
)
class TopKPruning:
    """Top-K activity sparsity.

    Args:
        model: nn.Module whose parameters will be pruned.
        k_ratio: fraction of weights (per parameter tensor) retained
            at each step (0, 1].
        activity_decay: EMA decay for the activity trace.
    """

    def __init__(self, model, k_ratio=0.5, activity_decay=0.99):
        self.model = model
        self.k_ratio = k_ratio
        self.activity_decay = activity_decay
        self.activity_trace = {}

    def step(self):
        """Apply top-k pruning based on accumulated activity traces."""
        for name, param in self.model.named_parameters():
            if param.ndim < 2 or param.grad is None:
                continue
            if name not in self.activity_trace:
                self.activity_trace[name] = torch.zeros_like(param)
            self.activity_trace[name].mul_(self.activity_decay).add_(
                param.grad.abs()
            )
            with torch.no_grad():
                trace = self.activity_trace[name]
                k = max(1, int(trace.numel() * self.k_ratio))
                if k >= trace.numel():
                    continue
                threshold = torch.topk(
                    trace.flatten(), k, largest=True
                ).values.min()
                mask = (trace >= threshold).to(param.dtype)
                param.data.mul_(mask)


@register_sparsity(
    name="ActivityDrivenPruning",
    domains=[Domain.VISION, Domain.LM, Domain.TIMESERIES],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.85,
    credit_assignment_type="local",
    requires_backward=False,
    memory_complexity="O(N)",
    typical_lr_range=(1e-4, 1e-2),
    typical_batch_size_range=(32, 256),
    tags=["sparsity", "pruning", "activity-driven", "hebbian"],
    description=(
        "Activity-driven sparsity: prune connections with low activity"
        " (Hebbian-inspired)."
    ),
)
class ActivityDrivenPruning:
    """Activity-driven sparsity: prune low-activity connections.

    Args:
        model: nn.Module whose parameters will be pruned.
        prune_fraction: fraction of weights pruned per step.
        activity_decay: EMA decay for the activity trace.
    """

    def __init__(self, model, prune_fraction=0.1, activity_decay=0.99):
        self.model = model
        self.prune_fraction = prune_fraction
        self.activity_decay = activity_decay
        self.activity_trace = {}

    def step(self):
        """Apply pruning based on activity traces."""
        for name, param in self.model.named_parameters():
            if param.ndim < 2 or param.grad is None:
                continue
            if name not in self.activity_trace:
                self.activity_trace[name] = torch.zeros_like(param)
            self.activity_trace[name].mul_(self.activity_decay).add_(
                param.grad.abs()
            )
            with torch.no_grad():
                mask = (
                    self.activity_trace[name]
                    > self.activity_trace[name].median()
                ).to(param.dtype)
                param.data.mul_(mask)


@register_sparsity(
    name="RandomPruning",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.2,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    tags=["sparsity", "pruning", "random", "baseline"],
    description=(
        "Random pruning: randomly drops connections as a sparsity baseline."
    ),
)
class RandomPruning:
    """Random pruning baseline.

    Args:
        model: nn.Module whose parameters will be pruned.
        prune_fraction: fraction of weights pruned per step.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, model, prune_fraction=0.1, seed=42):
        self.model = model
        self.prune_fraction = prune_fraction
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)

    def step(self):
        """Apply random pruning."""
        for param in self.model.parameters():
            if param.ndim < 2:
                continue
            with torch.no_grad():
                mask = torch.rand_like(param, generator=self._rng) > self.prune_fraction
                param.data.mul_(mask)


__all__ = ["TopKPruning", "ActivityDrivenPruning", "RandomPruning"]
