"""
Bioplausible Models

Clean, minimal model exports.

Usage:
    from bioplausible.models import LoopedMLP, ConvEqProp
    from bioplausible.models import create_model, list_models
"""

# Core EqProp models
from .looped_mlp import LoopedMLP

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
        AdaptiveFeedbackAlignment,
        ContrastiveFeedbackAlignment,
        FeedbackAlignmentEqProp,
        StochasticFA,
    )
except ImportError:
    FeedbackAlignmentEqProp = None
    AdaptiveFeedbackAlignment = None
    StochasticFA = None
    ContrastiveFeedbackAlignment = None

try:
    from .dfa_eqprop import DirectFeedbackAlignmentEqProp
except ImportError:
    DirectFeedbackAlignmentEqProp = None

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

# New Phase 0 additions
try:
    from .forward_forward import ForwardForwardNet
except ImportError:
    ForwardForwardNet = None

try:
    from .pepita import PEPITA
except ImportError:
    PEPITA = None

try:
    from .target_prop import DifferenceTargetProp
except ImportError:
    DifferenceTargetProp = None

try:
    from .three_factor import ThreeFactorHebbian
except ImportError:
    ThreeFactorHebbian = None

try:
    from .spiking_stdp import SpikingSTDP
except ImportError:
    SpikingSTDP = None

try:
    from .graph_eqprop import GraphEqProp
except ImportError:
    GraphEqProp = None

try:
    from .fabricpc_graph_pcn import FabricPCGraphPCN
except ImportError:
    FabricPCGraphPCN = None

# =============================================================================
# EquiTile: Scalable Local-Learning Architecture
# =============================================================================
# EquiTile: Scalable Local-Learning Architecture
from .equitile import (
    AblationStudy,
    AsyncConfig,
    AsyncEquiTile,
    BenchmarkRunner,
    CurriculumScheduler,
    DistributedConfig,
    DistributedEquiTile,
    DynamicEquiTile,
    EnhancedEquiTile,
    EnhancedEquiTileBuilder,
    EquiTile,
    EquiTileBuilder,
    EquiTileConfig,
    EquiTileEP,
    EquiTileProfiler,
    ExperimentTracker,
    InferenceContext,
    LearningMonitor,
    MemoryProfiler,
    MetricCollector,
    MixedPrecisionTrainer,
    MultiGPUConfig,
    MultiGPUEquiTile,
    NCCLCommunicator,
    TileGraph,
    TileGrowthManager,
    TileLayerNorm,
    TileMetrics,
    TileProcessor,
    TileResult,
    TileScheduler,
    TileState,
    TileTask,
    TrainingContext,
    VisualizationHelper,
    build_enhanced_model,
    build_model,
    create_ablation_study,
    create_async_model,
    create_distributed_model,
    create_dynamic_model,
    create_enhanced_model,
    create_fast_config,
    create_metric_collector,
    create_multigpu_model,
    create_production_config,
    create_profiler,
    create_research_config,
    create_tracker,
    create_visualization_helper,
    run_benchmark,
)

# Aliases for backward compatibility
# EquiTileEP is now a proper subclass imported from .equitile

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

try:
    from .custom_stack import CustomStackedModel
except ImportError:
    CustomStackedModel = None

# Aliases for validation track compatibility
AdaptiveFA = AdaptiveFeedbackAlignment


# Simple model registry
MODEL_REGISTRY = {
    "looped_mlp": LoopedMLP,
}

if ConvEqProp:
    MODEL_REGISTRY["conv_eqprop"] = ConvEqProp

if MemoryEfficientLoopedMLP:
    MODEL_REGISTRY["memory_efficient_mlp"] = MemoryEfficientLoopedMLP

if TransformerEqProp:
    MODEL_REGISTRY["transformer_eqprop"] = TransformerEqProp

if BackpropMLP:
    MODEL_REGISTRY["backprop_mlp"] = BackpropMLP

if TileEQ:
    MODEL_REGISTRY["tile_eq"] = TileEQ

if EquiTile:
    MODEL_REGISTRY["equitile"] = EquiTile

if ForwardForwardNet:
    MODEL_REGISTRY["forward_forward"] = ForwardForwardNet

if PEPITA:
    MODEL_REGISTRY["pepita"] = PEPITA

if DifferenceTargetProp:
    MODEL_REGISTRY["diff_target_prop"] = DifferenceTargetProp

if ThreeFactorHebbian:
    MODEL_REGISTRY["three_factor_hebbian"] = ThreeFactorHebbian

if SpikingSTDP:
    MODEL_REGISTRY["spiking_stdp"] = SpikingSTDP

if GraphEqProp:
    MODEL_REGISTRY["graph_eqprop"] = GraphEqProp

if FabricPCGraphPCN:
    MODEL_REGISTRY["fabricpc_graph_pcn"] = FabricPCGraphPCN


def create_model(name: str, **kwargs):
    """Create a model by name."""
    if name == "backprop":
        name = "backprop_mlp"
    if name == "eqprop" or name == "eqprop_mlp":
        name = "looped_mlp"

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    # Some models don't take num_layers
    if "num_layers" in kwargs:
        if name in ["looped_mlp", "conv_eqprop", "neural_cube", "standard_eqprop"]:
            del kwargs["num_layers"]

    # For backprop_mlp, num_layers defaults to 2 if not passed, but we pass it.

    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """List available models."""
    return list(MODEL_REGISTRY.keys())


__all__ = [
    # Core models
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    # Validation track models
    "NeuralCube",
    "DeepHebbianChain",
    "ContrastiveHebbianLearning",
    "LazyEqProp",
    "FiniteNudgeEP",
    "HolomorphicEP",
    "DirectedEP",
    "FeedbackAlignmentEqProp",
    "AdaptiveFeedbackAlignment",
    "DirectFeedbackAlignmentEqProp",
    "StochasticFA",
    "ContrastiveFeedbackAlignment",
    "EquilibriumAlignment",
    "CausalTransformerEqProp",
    "EqPropDiffusion",
    "ModernConvEqProp",
    "TileEQ",
    "EquiTile",
    "EquiTileEP",  # Alias for backward compatibility
    # New Phase 0 additions
    "ForwardForwardNet",
    "PEPITA",
    "DifferenceTargetProp",
    "ThreeFactorHebbian",
    "SpikingSTDP",
    "GraphEqProp",
    "FabricPCGraphPCN",
    # Additional validation models
    "HomeostaticEqProp",
    "TemporalResonanceEqProp",
    "TernaryEqProp",
    "StandardFA",
    "StandardEqProp",
    "MomentumEquilibrium",
    "SparseEquilibrium",
    "PredictiveCodingHybrid",
    "EnergyGuidedFA",
    "EnergyMinimizingFA",
    "LayerwiseEquilibriumFA",
    "DirectFeedbackAlignment",
    "CustomStackedModel",
    # LM variants
    "EqPropAttentionOnlyLM",
    "FullEqPropLM",
    "HybridEqPropLM",
    "LoopedMLPForLM",
    "RecurrentEqPropLM",
    "BackpropTransformerLM",
    "get_eqprop_lm",
    # Aliases
    "AdaptiveFA",
    # EquiTile core
    "TileGraph",
    "TileState",
    # 'EdgeParams', # Removed
    "EquiTileConfig",
    "create_production_config",
    "create_research_config",
    "create_fast_config",
    # EquiTile enhanced
    "EnhancedEquiTile",
    "TileLayerNorm",
    "CurriculumScheduler",
    "create_enhanced_model",
    # EquiTile dynamics
    "DynamicEquiTile",
    "TileGrowthManager",
    "TileMetrics",
    "create_dynamic_model",
    # EquiTile async
    "AsyncEquiTile",
    "AsyncConfig",
    "TileTask",
    "TileResult",
    "TileProcessor",
    "TileScheduler",
    "create_async_model",
    # EquiTile multi-GPU
    "MultiGPUEquiTile",
    "MultiGPUConfig",
    "NCCLCommunicator",
    "create_multigpu_model",
    # EquiTile distributed
    "DistributedEquiTile",
    "DistributedConfig",
    "MixedPrecisionTrainer",
    "create_distributed_model",
    # EquiTile profiler
    "EquiTileProfiler",
    "LearningMonitor",
    "MemoryProfiler",
    "BenchmarkRunner",
    "create_profiler",
    "run_benchmark",
    # EquiTile builder
    "EquiTileBuilder",
    "EnhancedEquiTileBuilder",
    "TrainingContext",
    "InferenceContext",
    "build_model",
    "build_enhanced_model",
    # EquiTile research
    "ExperimentTracker",
    "MetricCollector",
    "VisualizationHelper",
    "AblationStudy",
    "create_tracker",
    "create_metric_collector",
    "create_visualization_helper",
    "create_ablation_study",
    # Factory
    "create_model",
    "list_models",
]
