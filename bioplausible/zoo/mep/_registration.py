"""Registration of MEP presets and strategies with the unified Registry."""

from bioplausible.core.registry import (
    ComponentCategory,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
)

from .optimizers import DionUpdate, FisherUpdate, MuonUpdate, PlainUpdate
from .presets import local_ep, muon_backprop, natural_ep, sdmep, smep, smep_fast

# Register MEP presets as propagators (credit assignment + update combined)
Registry.register(
    ComponentCategory.PROPAGATOR,
    name="smep",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="smep_fast",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(smep_fast)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="sdmep",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.93,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(sdmep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="local_ep",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.97,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    family="mep",
)(local_ep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="natural_ep",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.90,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(N^2)",
    family="mep",
)(natural_ep)

Registry.register(
    ComponentCategory.PROPAGATOR,
    name="muon_backprop",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM, Domain.RL],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.3,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(muon_backprop)

# Pure update strategies as optimizers (complement the propagator presets)
Registry.register(
    ComponentCategory.OPTIMIZER,
    name="muon",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM, Domain.RL],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(MuonUpdate)

Registry.register(
    ComponentCategory.OPTIMIZER,
    name="dion",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(DionUpdate)

Registry.register(
    ComponentCategory.OPTIMIZER,
    name="plain",
    domains=[
        Domain.VISION,
        Domain.TABULAR,
        Domain.LM,
        Domain.RL,
        Domain.GRAPH,
        Domain.TIMESERIES,
    ],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N)",
    family="mep",
)(PlainUpdate)

Registry.register(
    ComponentCategory.OPTIMIZER,
    name="fisher",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.0,
    credit_assignment_type="gradient",
    requires_backward=True,
    memory_complexity="O(N^2)",
    family="mep",
)(FisherUpdate)
