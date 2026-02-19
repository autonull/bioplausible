"""
Bioplausible: Bio-Plausible Learning Algorithms for PyTorch

A production-grade library implementing biologically plausible learning algorithms,
centered on Equilibrium Propagation with spectral normalization for stability.

Key Features:
    - Multiple acceleration backends: Triton, CuPy, torch.compile
    - Spectral normalization for stable equilibrium dynamics
    - Comprehensive algorithm zoo: EqProp, FA variants, Hebbian learning
    - MEP optimizers: Muon Equilibrium Propagation with strategy pattern
    - AutoScientist: Autonomous experiment runner

Quick Start:
    from bioplausible import LoopedMLP, SupervisedTrainer, ModelZoo, OptimizerZoo

    model = ModelZoo.get('looped_mlp', input_size=784, hidden_size=256, output_size=10)
    optimizer = OptimizerZoo.get('smep', model.parameters(), model=model)
    trainer = SupervisedTrainer(model, device='cuda')
    trainer.fit(train_loader, val_loader, epochs=10)

Acceleration Backends:
    - Triton: Fused GPU kernels (fastest, requires CUDA)
    - CuPy: NumPy-compatible GPU arrays
    - torch.compile: PyTorch 2.0+ JIT compilation
    - Pure PyTorch: Standard autograd (fallback)

MEP Integration:
    The MEP (Muon Equilibrium Propagation) optimizers provide strategy-based
    optimization with validated performance (91-94% MNIST in 3 epochs).
    
    from bioplausible import smep, smep_fast, muon_backprop
    
    optimizer = smep(model.parameters(), model=model, mode='ep')
    optimizer.step(x=x, target=y)
"""

from bioplausible.acceleration import (
    TRITON_AVAILABLE,
    check_cupy_available,
    compile_model,
    enable_tf32,
    get_optimal_backend,
)
from bioplausible.core import EqPropTrainer
from bioplausible.datasets import (
    CharDataset,
    create_data_loaders,
    get_lm_dataset,
    get_vision_dataset,
)
from bioplausible.generation import generate_from_dataset, generate_text
from bioplausible.kernel import HAS_CUPY, EqPropKernel
from bioplausible.models import (
    BackpropMLP,
    ConvEqProp,
    LoopedMLP,
    MemoryEfficientLoopedMLP,
    TransformerEqProp,
)
from bioplausible.sklearn_interface import EqPropClassifier
from bioplausible.utils import (
    ModelRegistry,
    count_parameters,
    create_model_preset,
    export_to_onnx,
    verify_spectral_norm,
)

enable_tf32()

# ============================================================================
# ZOO: Unified Model and Optimizer Registry
# ============================================================================
try:
    from bioplausible.zoo import (
        ModelZoo,
        OptimizerZoo,
        ModelSpec,
        OptimizerSpec,
        get_model,
        get_optimizer,
        list_models,
        list_optimizers,
    )
    from bioplausible.zoo.registry import initialize_zoo
    
    HAS_ZOO = True
except ImportError as e:
    import warnings
    warnings.warn(f"Zoo import failed: {e}")
    HAS_ZOO = False
    ModelZoo = None
    OptimizerZoo = None
    get_model = None
    get_optimizer = None
    list_models = None
    list_optimizers = None

# ============================================================================
# MEP: Muon Equilibrium Propagation Optimizers
# ============================================================================
try:
    # Import from presets directly (more reliable)
    from mep.presets import (
        smep,
        smep_fast,
        sdmep,
        local_ep,
        natural_ep,
        muon_backprop,
    )
    
    # Import core optimizer components
    from mep.optimizers import (
        CompositeOptimizer,
        EPGradient,
        MuonUpdate,
        SpectralConstraint,
        Settler,
        EnergyFunction,
        ModelInspector,
    )
    
    HAS_MEP = True
except ImportError as e:
    import warnings
    warnings.warn(f"MEP import failed: {e}")
    HAS_MEP = False
    smep = None
    smep_fast = None
    sdmep = None
    local_ep = None
    natural_ep = None
    muon_backprop = None
    CompositeOptimizer = None
    EPGradient = None
    MuonUpdate = None
    SpectralConstraint = None
    Settler = None
    EnergyFunction = None
    ModelInspector = None

# ============================================================================
# Hybrid Optimizer: Best of Bioplausible + MEP
# ============================================================================
try:
    from bioplausible.hybrid_optimizer import (
        HybridEqPropOptimizer,
        HybridConfig,
        create_hybrid_optimizer,
    )
    
    HAS_HYBRID = True
except ImportError as e:
    import warnings
    warnings.warn(f"Hybrid optimizer import failed: {e}")
    HAS_HYBRID = False
    HybridEqPropOptimizer = None
    HybridConfig = None
    create_hybrid_optimizer = None

# ============================================================================
# Experiments Package: Research utilities
# ============================================================================
try:
    from bioplausible.experiments import (
        ExperimentRunner,
        ExperimentResult,
        HyperparameterSearch,
        quick_comparison,
        benchmark_model,
        get_preset,
        list_presets,
        run_preset,
        ALL_PRESETS,
    )
    
    HAS_EXPERIMENTS = True
except ImportError as e:
    import warnings
    warnings.warn(f"Experiments import failed: {e}")
    HAS_EXPERIMENTS = False
    ExperimentRunner = None
    ExperimentResult = None
    HyperparameterSearch = None
    quick_comparison = None
    benchmark_model = None
    get_preset = None
    list_presets = None
    run_preset = None
    ALL_PRESETS = None

# ============================================================================
# Deployment: Export and inference utilities
# ============================================================================
try:
    from bioplausible.deployment import (
        ModelExporter,
        ModelLoader,
        InferenceEngine,
        export_model,
        load_model,
    )
    
    HAS_DEPLOYMENT = True
except ImportError as e:
    import warnings
    warnings.warn(f"Deployment import failed: {e}")
    HAS_DEPLOYMENT = False
    ModelExporter = None
    ModelLoader = None
    InferenceEngine = None
    export_model = None
    load_model = None

# ============================================================================
# Visualization: Plotting and dashboard utilities
# ============================================================================
try:
    from bioplausible.visualization_tools import (
        TrainingVisualizer,
        ResultsDashboard,
        visualize_results,
    )
    
    HAS_VISUALIZATION = True
except ImportError as e:
    import warnings
    warnings.warn(f"Visualization import failed: {e}")
    HAS_VISUALIZATION = False
    TrainingVisualizer = None
    ResultsDashboard = None
    visualize_results = None

# ============================================================================
# Analysis: Statistical analysis utilities
# ============================================================================
try:
    from bioplausible.analysis_tools import (
        ResultAnalyzer,
        StatisticalComparison,
        AnalysisReport,
        analyze_results,
    )
    
    HAS_ANALYSIS_TOOLS = True
except ImportError as e:
    import warnings
    warnings.warn(f"Analysis tools import failed: {e}")
    HAS_ANALYSIS_TOOLS = False
    ResultAnalyzer = None
    StatisticalComparison = None
    AnalysisReport = None
    analyze_results = None

try:
    from bioplausible.models.eqprop_lm_variants import (
        create_eqprop_lm,
        get_eqprop_lm,
        list_eqprop_lm_variants,
    )

    HAS_LM_VARIANTS = True
except ImportError:
    HAS_LM_VARIANTS = False
    get_eqprop_lm = None
    list_eqprop_lm_variants = None
    create_eqprop_lm = None

try:
    from bioplausible.models.base import BioModel as BaseAlgorithm
    from bioplausible.models.registry import MODEL_REGISTRY

    ALGORITHM_REGISTRY = {spec.name: spec.description for spec in MODEL_REGISTRY}

    from bioplausible.models.ada_fa import AdaptiveFeedbackAlignment
    from bioplausible.models.cf_align import ContrastiveFeedbackAlignment
    from bioplausible.models.eg_fa import EnergyGuidedFA
    from bioplausible.models.em_fa import EnergyMinimizingFA
    from bioplausible.models.eq_align import EquilibriumAlignment
    from bioplausible.models.leq_fa import LayerwiseEquilibriumFA
    from bioplausible.models.looped_mlp import BackpropMLP as BackpropBaseline
    from bioplausible.models.mom_eq import MomentumEquilibrium
    from bioplausible.models.pc_hybrid import PredictiveCodingHybrid
    from bioplausible.models.simple_fa import StandardFA
    from bioplausible.models.sparse_eq import SparseEquilibrium
    from bioplausible.models.standard_eqprop import StandardEqProp
    from bioplausible.models.sto_fa import StochasticFA

    HAS_BIOPLAUSIBLE = True
except ImportError as e:
    import warnings

    warnings.warn(f"Bio-plausible models import failed: {e}")
    HAS_BIOPLAUSIBLE = False
    BaseAlgorithm = None
    ALGORITHM_REGISTRY = {}

__version__ = "0.1.0"

__all__ = [
    # Core
    "EqPropTrainer",
    "EqPropKernel",
    "HAS_CUPY",
    # Models
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    # Datasets
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "create_data_loaders",
    # Generation
    "generate_text",
    "generate_from_dataset",
    # Acceleration
    "TRITON_AVAILABLE",
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    "enable_tf32",
    # Utils
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "create_model_preset",
    "ModelRegistry",
    # Sklearn interface
    "EqPropClassifier",
    # LM variants
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "HAS_LM_VARIANTS",
    # Base classes
    "BaseAlgorithm",
    "BackpropBaseline",
    # Algorithm models
    "StandardEqProp",
    "StandardFA",
    "EquilibriumAlignment",
    "AdaptiveFeedbackAlignment",
    "ContrastiveFeedbackAlignment",
    "LayerwiseEquilibriumFA",
    "PredictiveCodingHybrid",
    "EnergyGuidedFA",
    "SparseEquilibrium",
    "MomentumEquilibrium",
    "StochasticFA",
    "EnergyMinimizingFA",
    "HAS_BIOPLAUSIBLE",
    "ALGORITHM_REGISTRY",
    "MODEL_REGISTRY",
    # Zoo (NEW)
    "ModelZoo",
    "OptimizerZoo",
    "ModelSpec",
    "OptimizerSpec",
    "get_model",
    "get_optimizer",
    "list_models",
    "list_optimizers",
    "HAS_ZOO",
    # MEP Optimizers (NEW)
    "smep",
    "smep_fast",
    "sdmep",
    "local_ep",
    "natural_ep",
    "muon_backprop",
    "CompositeOptimizer",
    "EPGradient",
    "MuonUpdate",
    "SpectralConstraint",
    "Settler",
    "EnergyFunction",
    "ModelInspector",
    "HAS_MEP",
    # Hybrid Optimizer (NEW)
    "HybridEqPropOptimizer",
    "HybridConfig",
    "create_hybrid_optimizer",
    "HAS_HYBRID",
    # Experiments Package (NEW)
    "ExperimentRunner",
    "ExperimentResult",
    "HyperparameterSearch",
    "quick_comparison",
    "benchmark_model",
    "get_preset",
    "list_presets",
    "run_preset",
    "ALL_PRESETS",
    "HAS_EXPERIMENTS",
    # Deployment (NEW)
    "ModelExporter",
    "ModelLoader",
    "InferenceEngine",
    "export_model",
    "load_model",
    "HAS_DEPLOYMENT",
    # Visualization (NEW)
    "TrainingVisualizer",
    "ResultsDashboard",
    "visualize_results",
    "HAS_VISUALIZATION",
    # Analysis (NEW)
    "ResultAnalyzer",
    "StatisticalComparison",
    "AnalysisReport",
    "analyze_results",
    "HAS_ANALYSIS_TOOLS",
]


def get_scientist():
    """Lazy import of AutoScientist to avoid circular imports."""
    from bioplausible.scientist import AutoScientist

    return AutoScientist
