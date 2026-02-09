from .ada_fa import AdaptiveFeedbackAlignment
from .adaptive_fa import AdaptiveFA
from .backprop_transformer_lm import BackpropTransformerLM
from .base import BioModel, ModelConfig
from .registry import ModelRegistry, register_model, ModelSpec, MODEL_REGISTRY
from .causal_transformer_eqprop import CausalTransformerEqProp
from .cf_align import ContrastiveFeedbackAlignment
from .chl import CHLAutoencoder, ContrastiveHebbianLearning
from .conv_eqprop import ConvEqProp
from .deep_ep import DirectedEP
from .dfa_eqprop import DirectFeedbackAlignmentEqProp
from .eg_fa import EnergyGuidedFA
from .em_fa import EnergyMinimizingFA
from .eq_align import EquilibriumAlignment
from .eqprop_base import EqPropModel, EquilibriumFunction
from .eqprop_diffusion import EqPropDiffusion
# Language Models
from .eqprop_lm_variants import (EqPropAttentionOnlyLM, FullEqPropLM,
                                 HybridEqPropLM, LoopedMLPForLM,
                                 RecurrentEqPropLM, create_eqprop_lm,
                                 get_eqprop_lm, list_eqprop_lm_variants)
# Feedback Alignment Variants
from .feedback_alignment import FeedbackAlignmentEqProp
from .finite_nudge_ep import FiniteNudgeEP
from .hebbian_chain import DeepHebbianChain
from .holomorphic_ep import HolomorphicEP
from .homeostatic import HomeostaticEqProp
from .lazy_eqprop import LazyEqProp
from .leq_fa import LayerwiseEquilibriumFA
# Core Models
from .looped_mlp import BackpropMLP, LoopedMLP
# Memory Efficient Models
from .memory_efficient import (MemoryEfficientEqPropModel,
                               MemoryEfficientLoopedMLP,
                               create_memory_efficient_model)
from .modern_conv_eqprop import ModernConvEqProp, SimpleConvEqProp
from .mom_eq import MomentumEquilibrium
from .nebc_base import (NEBCBase, NEBCRegistry, evaluate_nebc_model,
                        register_nebc, train_nebc_model)
from .neural_cube import NeuralCube
from .pc_hybrid import PredictiveCodingHybrid
from .simple_fa import StandardFA
from .sparse_eq import SparseEquilibrium
# Algorithm-Models (Migrated from algorithms/)
from .standard_eqprop import StandardEqProp
from .sto_fa import StochasticFA
from .temporal_resonance import TemporalResonanceEqProp
from .temporal_resonance import \
    TemporalResonanceEqProp as TemporalResonanceNetwork
from .ternary import TernaryEqProp
from .ternary import TernaryEqProp as TernaryWeightMLP
from .transformer_eqprop import TransformerEqProp

# Export registry
__all__ = [
    # Base
    "BioModel",
    "ModelConfig",
    "register_model",
    "ModelRegistry",
    "ModelSpec",
    "MODEL_REGISTRY",
    "NEBCBase",
    "NEBCRegistry",
    "register_nebc",
    "EqPropModel",
    "EquilibriumFunction",
    # Core
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "ModernConvEqProp",
    "SimpleConvEqProp",
    "TransformerEqProp",
    "CausalTransformerEqProp",
    "EqPropDiffusion",
    "LazyEqProp",
    "HomeostaticEqProp",
    "DirectFeedbackAlignmentEqProp",
    "NeuralCube",
    "TernaryWeightMLP",
    "TernaryEqProp",
    "ContrastiveHebbianLearning",
    "CHLAutoencoder",
    "DeepHebbianChain",
    "TemporalResonanceNetwork",
    "TemporalResonanceEqProp",
    "BackpropTransformerLM",
    "HolomorphicEP",
    "DirectedEP",
    "FiniteNudgeEP",
    # Algorithm-Models
    "StandardEqProp",
    "StandardFA",
    "AdaptiveFeedbackAlignment",
    "EquilibriumAlignment",
    "ContrastiveFeedbackAlignment",
    "LayerwiseEquilibriumFA",
    "EnergyGuidedFA",
    "PredictiveCodingHybrid",
    "SparseEquilibrium",
    "MomentumEquilibrium",
    "StochasticFA",
    "EnergyMinimizingFA",
    # FA Variants
    "FeedbackAlignmentEqProp",
    "AdaptiveFA",
    # Memory Efficient
    "MemoryEfficientLoopedMLP",
    "MemoryEfficientEqPropModel",
    "create_memory_efficient_model",
    # LM
    "FullEqPropLM",
    "EqPropAttentionOnlyLM",
    "RecurrentEqPropLM",
    "HybridEqPropLM",
    "LoopedMLPForLM",
    "get_eqprop_lm",
    "create_eqprop_lm",
    "list_eqprop_lm_variants",
]
