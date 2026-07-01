"""
Registered propagators (learning rules/credit assignment methods) in the unified Zoo.

Each propagator is wrapped with @register_propagator for AutoScientist discovery.
Wraps existing learning rules from bioplausible.optimizers.learning_rules.
"""

from bioplausible.core.registry import (ComputeProfile, Domain, LocalityLevel,
                                        register_propagator)
# Import all existing learning rules
from bioplausible.optimizers.learning_rules import (AdaptiveFA, ContrastiveFA,
                                                    ContrastiveHebbianLearning,
                                                    DirectFA, EqProp,
                                                    FeedbackAlignment,
                                                    FiniteNudgeEqProp,
                                                    HolomorphicEqProp,
                                                    LazyEqProp, StochasticFA)

# MEP optimizers are imported lazily inside the try block below

# ---------------------------------------------------------------------------
# Feedback Alignment Family
# ---------------------------------------------------------------------------


@register_propagator(
    name="FeedbackAlignment",
    domains=[Domain.VISION, Domain.TABULAR, Domain.LM],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.6,
    credit_assignment_type="hebbian",
    requires_backward=False,
    memory_complexity="O(N)",
    typical_lr_range=(1e-4, 1e-2),
    typical_batch_size_range=(32, 256),
    tags=["feedback-alignment", "bio-plausible", "forward-only"],
    description=(
        "Feedback Alignment: fixed random feedback replace"
        " transposed forward weights"
    ),
    citation="Lillicrap et al., 2016",
)
class _RegisteredFeedbackAlignment(FeedbackAlignment):
    pass


@register_propagator(
    name="DirectFA",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.5,
    credit_assignment_type="hebbian",
    requires_backward=False,
    memory_complexity="O(N)",
    typical_lr_range=(1e-4, 1e-2),
    tags=["direct-feedback-alignment", "bio-plausible"],
    description=(
        "Direct Feedback Alignment: output error is" " projected directly to each layer"
    ),
    citation="Nøkland, 2016",
)
class _RegisteredDirectFA(DirectFA):
    pass


@register_propagator(
    name="AdaptiveFA",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.4,
    credit_assignment_type="hebbian",
    requires_backward=False,
    memory_complexity="O(N)",
    tags=["adaptive-feedback-alignment", "bio-plausible"],
    description=(
        "Adaptive Feedback Alignment: feedback weights"
        " slowly adapt toward forward weights"
    ),
    citation="Akrout et al., 2019",
)
class _RegisteredAdaptiveFA(AdaptiveFA):
    pass


@register_propagator(
    name="StochasticFA",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.3,
    credit_assignment_type="hebbian",
    requires_backward=False,
    tags=["stochastic-feedback-alignment", "bio-plausible"],
    description=(
        "Stochastic Feedback Alignment: noise added"
        " to feedback weights for robustness"
    ),
)
class _RegisteredStochasticFA(StochasticFA):
    pass


@register_propagator(
    name="ContrastiveFA",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.GLOBAL,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.3,
    credit_assignment_type="hebbian",
    requires_backward=False,
    tags=["contrastive-feedback-alignment"],
    description=(
        "Contrastive Feedback Alignment: contrastive"
        " loss combined with feedback alignment"
    ),
)
class _RegisteredContrastiveFA(ContrastiveFA):
    pass


# ---------------------------------------------------------------------------
# Equilibrium Propagation Family
# ---------------------------------------------------------------------------


@register_propagator(
    name="EqProp",
    domains=[Domain.VISION, Domain.LM, Domain.RL, Domain.GRAPH],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.9,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    typical_lr_range=(1e-3, 1e-1),
    typical_batch_size_range=(16, 128),
    tags=["equilibrium-propagation", "energy-based", "bio-plausible"],
    description=(
        "Standard Equilibrium Propagation: gradient"
        " estimation via free and nudged phases"
    ),
    citation="Scellier & Bengio, 2017",
)
class _RegisteredEqProp(EqProp):
    pass


@register_propagator(
    name="HolomorphicEqProp",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.GPU,
    bio_plausibility_score=0.85,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    tags=["holomorphic", "eqprop", "complex-valued"],
    description=(
        "Holomorphic EqProp: complex-valued states" " for exact gradient estimation"
    ),
    citation="NeurIPS 2024",
)
class _RegisteredHolomorphicEqProp(HolomorphicEqProp):
    pass


@register_propagator(
    name="FiniteNudgeEqProp",
    domains=[Domain.VISION],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.8,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    tags=["finite-nudge", "eqprop", "noise-robust"],
    description=(
        "Finite Nudge EqProp: large beta for noise-robust" " gradient via finite diff"
    ),
)
class _RegisteredFiniteNudgeEqProp(FiniteNudgeEqProp):
    pass


@register_propagator(
    name="LazyEqProp",
    domains=[Domain.VISION, Domain.TIMESERIES],
    locality_level=LocalityLevel.EQUILIBRIUM,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.95,
    credit_assignment_type="equilibrium",
    requires_backward=False,
    memory_complexity="O(1)",
    tags=["lazy", "event-driven", "eqprop", "low-power"],
    description=(
        "Lazy EqProp: event-driven updates when inputs" " change (~97% reduction)"
    ),
)
class _RegisteredLazyEqProp(LazyEqProp):
    pass


# ---------------------------------------------------------------------------
# Hebbian Learning Family
# ---------------------------------------------------------------------------


@register_propagator(
    name="ContrastiveHebbian",
    domains=[Domain.VISION, Domain.TABULAR],
    locality_level=LocalityLevel.LOCAL,
    compute_profile=ComputeProfile.NEUROMORPHIC,
    bio_plausibility_score=0.85,
    credit_assignment_type="hebbian",
    requires_backward=False,
    memory_complexity="O(1)",
    typical_lr_range=(1e-3, 1e-1),
    tags=["hebbian", "contrastive", "local-learning", "bio-plausible"],
    description=(
        "Contrastive Hebbian: weight updates from" " free-vs-clamped Hebbian contrast"
    ),
    citation="Movellan, 1991",
)
class _RegisteredCHL(ContrastiveHebbianLearning):
    pass


# ---------------------------------------------------------------------------
# MEP (Memory-Efficient Propagation) Forward-Only Optimizers as Propagators
# ---------------------------------------------------------------------------


def _register_mep_direct(name, opt_cls, bio_score, description, domains, tags):
    """Register MEP optimizer directly as propagator (avoids metaclass conflicts)."""
    from bioplausible.core.registry import (ComponentCategory,
                                            ComponentMetadata, Registry)

    metadata = ComponentMetadata(
        name=name,
        category=ComponentCategory.PROPAGATOR,
        domains=domains,
        locality_level=LocalityLevel.FORWARD_ONLY,
        compute_profile=ComputeProfile.MEMRISTOR,
        bio_plausibility_score=bio_score,
        credit_assignment_type="forward-only",
        requires_backward=False,
        memory_complexity="O(1)",
        typical_lr_range=(1e-4, 1e-1),
        typical_batch_size_range=(16, 256),
        tags=tags,
        description=description,
        citation="Frenkel et al., 2024",
    )
    Registry._components.setdefault(ComponentCategory.PROPAGATOR, {})[name] = {
        "class": opt_cls,
        "metadata": metadata,
    }
    Registry._name_to_category[name] = ComponentCategory.PROPAGATOR.value


# Register MEP variants as propagators if available
try:
    from bioplausible.optimizers import (HAS_MEP, local_ep, muon_backprop,
                                         natural_ep, sdmep, smep, smep_fast)

    if HAS_MEP:
        _register_mep_direct(
            "smep",
            smep,
            0.95,
            "Synthetic MEP: O(1) memory forward-only learning via synthetic gradients",
            [Domain.VISION, Domain.TABULAR, Domain.LM],
            ["mep", "forward-only", "o1-memory", "bio-plausible"],
        )
        _register_mep_direct(
            "smep_fast",
            smep_fast,
            0.9,
            "Fast Synthetic MEP: optimized MEP variant with reduced compute overhead",
            [Domain.VISION, Domain.TABULAR],
            ["mep", "forward-only", "fast", "bio-plausible"],
        )
        _register_mep_direct(
            "sdmep",
            sdmep,
            0.95,
            "Synthetic Dual MEP: dual-pathway forward-only learning",
            [Domain.VISION, Domain.TABULAR],
            ["mep", "forward-only", "dual-pathway", "bio-plausible"],
        )
        _register_mep_direct(
            "local_ep",
            local_ep,
            0.9,
            "Local Equilibrium Propagation: layer-local EP with reduced communication",
            [Domain.VISION, Domain.TABULAR],
            ["mep", "local-ep", "equilibrium", "bio-plausible"],
        )
        _register_mep_direct(
            "natural_ep",
            natural_ep,
            0.85,
            "Natural Equilibrium Propagation: natural gradient in EP framework",
            [Domain.VISION, Domain.TABULAR],
            ["mep", "natural-ep", "equilibrium", "bio-plausible"],
        )
        _register_mep_direct(
            "muon_backprop",
            muon_backprop,
            0.0,
            "Muon Backprop: optimizer based on muon (orthogonal)"
            " updates with standard backprop",
            [Domain.VISION, Domain.LM],
            ["muon", "backprop", "optimizer"],
        )
except ImportError:
    pass
