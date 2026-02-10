"""
EqProp-Torch: Equilibrium Propagation for PyTorch

A production-grade library for training neural networks with Equilibrium Propagation,
featuring spectral normalization for stability and torch.compile for acceleration.
"""

from .acceleration import (check_cupy_available, compile_model, enable_tf32,
                           get_optimal_backend)

# Enable TF32 by default for Ampere+ GPUs (2-3x speedup)
enable_tf32()

from .core import EqPropTrainer
from .datasets import (CharDataset, create_data_loaders, get_lm_dataset,
                       get_vision_dataset)
from .generation import generate_from_dataset, generate_text
from .kernel import HAS_CUPY, EqPropKernel
from .models import (BackpropMLP, ConvEqProp, LoopedMLP,
                     MemoryEfficientLoopedMLP, TransformerEqProp)
from .sklearn_interface import EqPropClassifier
from .utils import (ModelRegistry, count_parameters, create_model_preset,
                    export_to_onnx, verify_spectral_norm)

# Language model variants (optional import - fails gracefully if dependencies missing)
try:
    from .lm_models import (create_eqprop_lm, get_eqprop_lm,
                            list_eqprop_lm_variants)

    HAS_LM_VARIANTS = True
except ImportError:
    HAS_LM_VARIANTS = False
    get_eqprop_lm = None
    list_eqprop_lm_variants = None

# Bio-plausible research algorithms
try:
    from .models.base import BioModel as BaseAlgorithm
    from .models.registry import MODEL_REGISTRY

    # Alias for compatibility
    ALGORITHM_REGISTRY = {spec.name: spec.description for spec in MODEL_REGISTRY}

    # Import key models
    from .models.ada_fa import AdaptiveFeedbackAlignment
    from .models.cf_align import ContrastiveFeedbackAlignment
    from .models.eg_fa import EnergyGuidedFA
    from .models.em_fa import EnergyMinimizingFA
    from .models.eq_align import EquilibriumAlignment
    from .models.leq_fa import LayerwiseEquilibriumFA
    # Pseudo-BackpropBaseline for compatibility if referenced
    from .models.looped_mlp import BackpropMLP as BackpropBaseline
    from .models.mom_eq import MomentumEquilibrium
    from .models.pc_hybrid import PredictiveCodingHybrid
    from .models.simple_fa import StandardFA
    from .models.sparse_eq import SparseEquilibrium
    from .models.standard_eqprop import StandardEqProp
    from .models.sto_fa import StochasticFA

    HAS_BIOPLAUSIBLE = True
except ImportError as e:
    import warnings

    warnings.warn(f"Bio-plausible models import failed: {e}")
    HAS_BIOPLAUSIBLE = False
    BaseAlgorithm = None
    ALGORITHM_REGISTRY = {}


__version__ = "0.1.0"
__all__ = [
    # Trainer
    "EqPropTrainer",
    # Models
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    # Kernel
    "EqPropKernel",
    "HAS_CUPY",
    # Utils
    "compile_model",
    "get_optimal_backend",
    "check_cupy_available",
    "enable_tf32",
    # Datasets
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "generate_text",
    "generate_from_dataset",
    "create_data_loaders",
    # Utils
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "create_model_preset",
    "ModelRegistry",
    "EqPropClassifier",
    # LM variants (if available)
    "get_eqprop_lm",
    "list_eqprop_lm_variants",
    "create_eqprop_lm",
    "HAS_LM_VARIANTS",
    # Bio-plausible research algorithms (if available) - first-class models
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
]
