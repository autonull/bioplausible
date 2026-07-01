"""
Registered optimizers in the unified Zoo.

Standard PyTorch optimizers and bio-plausible variants registered
with @register_optimizer for AutoScientist discovery.
"""

import torch.optim as optim

from bioplausible.core.registry import (
    ComputeProfile,
    Domain,
    LocalityLevel,
    register_optimizer,
)


@register_optimizer(
    name="sgd",
    domains=[Domain.VISION, Domain.LM, Domain.RL, Domain.TABULAR, Domain.GRAPH],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    typical_lr_range=(1e-4, 1.0),
    typical_batch_size_range=(16, 512),
    tags=["baseline", "standard", "gradient-descent"],
    description="Stochastic Gradient Descent with optional momentum and weight decay",
)
class _RegisteredSGD(optim.SGD):
    pass


@register_optimizer(
    name="adam",
    domains=[
        Domain.VISION,
        Domain.LM,
        Domain.RL,
        Domain.TABULAR,
        Domain.GRAPH,
        Domain.TIMESERIES,
    ],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    typical_lr_range=(1e-5, 1e-3),
    typical_batch_size_range=(16, 512),
    tags=["baseline", "standard", "adaptive"],
    description="Adam: Adaptive Moment Estimation optimizer",
    citation="Kingma & Ba, 2015",
)
class _RegisteredAdam(optim.Adam):
    pass


@register_optimizer(
    name="adamw",
    domains=[
        Domain.VISION,
        Domain.LM,
        Domain.RL,
        Domain.TABULAR,
        Domain.GRAPH,
        Domain.TIMESERIES,
    ],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    typical_lr_range=(1e-5, 1e-3),
    typical_batch_size_range=(16, 512),
    tags=["baseline", "standard", "adaptive", "weight-decay"],
    description="AdamW: Adam with decoupled weight decay",
    citation="Loshchilov & Hutter, 2019",
)
class _RegisteredAdamW(optim.AdamW):
    pass
