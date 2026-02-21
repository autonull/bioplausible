"""
Bioplausible Models

Clean, minimal model exports.

Usage:
    from bioplausible.models import LoopedMLP, ConvEqProp
    from bioplausible.models import create_model, list_models
"""

# Core EqProp models
from .looped_mlp_simple import LoopedMLP

# Backprop baseline (for validation tracks)
try:
    from .looped_mlp import BackpropMLP
except ImportError:
    BackpropMLP = None

# Specialized variants (optional imports for validation tracks)
try:
    from .conv_eqprop import ConvEqProp
except ImportError:
    ConvEqProp = None

try:
    from .memory_efficient import MemoryEfficientLoopedMLP
except ImportError:
    MemoryEfficientLoopedMLP = None

try:
    from .transformer_eqprop import TransformerEqProp
except ImportError:
    TransformerEqProp = None

try:
    from .neural_cube import NeuralCube
except ImportError:
    NeuralCube = None

try:
    from .hebbian_chain import DeepHebbianChain
except ImportError:
    DeepHebbianChain = None

try:
    from .chl import ContrastiveHebbianLearning
except ImportError:
    ContrastiveHebbianLearning = None

try:
    from .lazy_eqprop import LazyEqProp
except ImportError:
    LazyEqProp = None

try:
    from .finite_nudge_ep import FiniteNudgeEP
except ImportError:
    FiniteNudgeEP = None

try:
    from .holomorphic_ep import HolomorphicEP
except ImportError:
    HolomorphicEP = None

try:
    from .deep_ep import DirectedEP
except ImportError:
    DirectedEP = None

try:
    from .feedback_alignment import (
        FeedbackAlignmentEqProp,
        AdaptiveFeedbackAlignment,
        DirectFeedbackAlignmentEqProp,
        StochasticFA,
        ContrastiveFeedbackAlignment,
    )
except ImportError:
    FeedbackAlignmentEqProp = None
    AdaptiveFeedbackAlignment = None
    DirectFeedbackAlignmentEqProp = None
    StochasticFA = None
    ContrastiveFeedbackAlignment = None

try:
    from .eq_align import EquilibriumAlignment
except ImportError:
    EquilibriumAlignment = None

try:
    from .causal_transformer_eqprop import CausalTransformerEqProp
except ImportError:
    CausalTransformerEqProp = None

try:
    from .eqprop_diffusion import EqPropDiffusion
except ImportError:
    EqPropDiffusion = None

try:
    from .modern_conv_eqprop import ModernConvEqProp
except ImportError:
    ModernConvEqProp = None

try:
    from .tile_eq import TileEQ
except ImportError:
    TileEQ = None

# =============================================================================
# EquiTile: Scalable Local-Learning Architecture
# =============================================================================
from .equitile import (
    # Core
    EquiTile,
    TileGraph,
    TileState,
    EdgeParams,
    
    # Config factories
    EquiTileConfig,
    create_production_config,
    create_research_config,
    create_fast_config,
)

# LM variants for validation tracks
try:
    from .eqprop_lm_variants import (
        EqPropAttentionOnlyLM,
        FullEqPropLM,
        HybridEqPropLM,
        LoopedMLPForLM,
        RecurrentEqPropLM,
        get_eqprop_lm,
    )
except ImportError:
    EqPropAttentionOnlyLM = None
    FullEqPropLM = None
    HybridEqPropLM = None
    LoopedMLPForLM = None
    RecurrentEqPropLM = None
    get_eqprop_lm = None

try:
    from .backprop_transformer_lm import BackpropTransformerLM
except ImportError:
    BackpropTransformerLM = None

# Additional models for validation tracks
try:
    from .homeostatic import HomeostaticEqProp
except ImportError:
    HomeostaticEqProp = None

try:
    from .temporal_resonance import TemporalResonanceEqProp
except ImportError:
    TemporalResonanceEqProp = None

try:
    from .ternary import TernaryEqProp
except ImportError:
    TernaryEqProp = None

try:
    from .simple_fa import StandardFA
except ImportError:
    StandardFA = None

try:
    from .standard_eqprop import StandardEqProp
except ImportError:
    StandardEqProp = None

try:
    from .mom_eq import MomentumEquilibrium
except ImportError:
    MomentumEquilibrium = None

try:
    from .sparse_eq import SparseEquilibrium
except ImportError:
    SparseEquilibrium = None

try:
    from .pc_hybrid import PredictiveCodingHybrid
except ImportError:
    PredictiveCodingHybrid = None

try:
    from .eg_fa import EnergyGuidedFA
except ImportError:
    EnergyGuidedFA = None

try:
    from .em_fa import EnergyMinimizingFA
except ImportError:
    EnergyMinimizingFA = None

try:
    from .leq_fa import LayerwiseEquilibriumFA
except ImportError:
    LayerwiseEquilibriumFA = None

try:
    from .sto_fa import StochasticFA as StochasticFAModel
except ImportError:
    StochasticFAModel = None

try:
    from .dfa_eqprop import DirectFeedbackAlignment
except ImportError:
    DirectFeedbackAlignment = None

# Aliases for validation track compatibility
AdaptiveFA = AdaptiveFeedbackAlignment


# Simple model registry
MODEL_REGISTRY = {
    'looped_mlp': LoopedMLP,
}

if ConvEqProp:
    MODEL_REGISTRY['conv_eqprop'] = ConvEqProp

if MemoryEfficientLoopedMLP:
    MODEL_REGISTRY['memory_efficient_mlp'] = MemoryEfficientLoopedMLP

if TransformerEqProp:
    MODEL_REGISTRY['transformer_eqprop'] = TransformerEqProp

if BackpropMLP:
    MODEL_REGISTRY['backprop_mlp'] = BackpropMLP

if TileEQ:
    MODEL_REGISTRY['tile_eq'] = TileEQ

if EquiTile:
    MODEL_REGISTRY['equitile'] = EquiTile


def create_model(name: str, **kwargs):
    """Create a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    # Core models
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'MemoryEfficientLoopedMLP',
    'TransformerEqProp',
    # Validation track models
    'NeuralCube',
    'DeepHebbianChain',
    'ContrastiveHebbianLearning',
    'LazyEqProp',
    'FiniteNudgeEP',
    'HolomorphicEP',
    'DirectedEP',
    'FeedbackAlignmentEqProp',
    'AdaptiveFeedbackAlignment',
    'DirectFeedbackAlignmentEqProp',
    'StochasticFA',
    'ContrastiveFeedbackAlignment',
    'EquilibriumAlignment',
    'CausalTransformerEqProp',
    'EqPropDiffusion',
    'ModernConvEqProp',
    'TileEQ',
    'EquiTile',
    # Additional validation models
    'HomeostaticEqProp',
    'TemporalResonanceEqProp',
    'TernaryEqProp',
    'StandardFA',
    'StandardEqProp',
    'MomentumEquilibrium',
    'SparseEquilibrium',
    'PredictiveCodingHybrid',
    'EnergyGuidedFA',
    'EnergyMinimizingFA',
    'LayerwiseEquilibriumFA',
    'DirectFeedbackAlignment',
    # LM variants
    'EqPropAttentionOnlyLM',
    'FullEqPropLM',
    'HybridEqPropLM',
    'LoopedMLPForLM',
    'RecurrentEqPropLM',
    'BackpropTransformerLM',
    'get_eqprop_lm',
    # Aliases
    'AdaptiveFA',
    # Factory
    'create_model',
    'list_models',
]
