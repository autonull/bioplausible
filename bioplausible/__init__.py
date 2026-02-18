"""
Bioplausible: Bio-Plausible Learning Algorithms for PyTorch

A production-grade library implementing biologically plausible learning algorithms,
centered on Equilibrium Propagation with spectral normalization for stability.

Key Features:
    - Multiple acceleration backends: Triton, CuPy, torch.compile
    - Spectral normalization for stable equilibrium dynamics
    - Comprehensive algorithm zoo: EqProp, FA variants, Hebbian learning
    - AutoScientist: Autonomous experiment runner

Quick Start:
    from bioplausible import LoopedMLP, SupervisedTrainer

    model = LoopedMLP(784, 256, 10)
    trainer = SupervisedTrainer(model, device='cuda')
    trainer.fit(train_loader, val_loader, epochs=10)

Acceleration Backends:
    - Triton: Fused GPU kernels (fastest, requires CUDA)
    - CuPy: NumPy-compatible GPU arrays
    - torch.compile: PyTorch 2.0+ JIT compilation
    - Pure PyTorch: Standard autograd (fallback)
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
    "EqPropTrainer",
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    "EqPropKernel",
    "HAS_CUPY",
    "TRITON_AVAILABLE",
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    "enable_tf32",
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "generate_text",
    "generate_from_dataset",
    "create_data_loaders",
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "create_model_preset",
    "ModelRegistry",
    "EqPropClassifier",
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "HAS_LM_VARIANTS",
    "BaseAlgorithm",
    "BackpropBaseline",
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
]


def get_scientist():
    """Lazy import of AutoScientist to avoid circular imports."""
    from bioplausible.scientist import AutoScientist

    return AutoScientist
