"""
Model and Optimizer Registry Population

This module populates the ModelZoo and OptimizerZoo with all available
models from Bioplausible and optimizers from MEP.
"""

import torch.nn as nn
from typing import Any, Dict

from bioplausible.zoo import ModelZoo, OptimizerZoo, ModelSpec, OptimizerSpec


# ============================================================================
# MODEL REGISTRY POPULATION
# ============================================================================

def _populate_model_registry() -> None:
    """Populate the ModelZoo with all available models."""
    
    # Import models lazily to avoid circular imports
    from bioplausible.models import (
        LoopedMLP,
        BackpropMLP,
        ConvEqProp,
        MemoryEfficientLoopedMLP,
        TransformerEqProp,
    )
    
    # Try to import optional models
    try:
        from bioplausible.models.eqprop_lm_variants import (
            create_eqprop_lm,
        )
        HAS_LM = True
    except ImportError:
        HAS_LM = False
    
    try:
        from bioplausible.models.modern_conv_eqprop import ModernConvEqProp
        HAS_MODERN_CONV = True
    except ImportError:
        HAS_MODERN_CONV = False
    
    try:
        from bioplausible.models.neural_cube import NeuralCube
        HAS_NEURAL_CUBE = True
    except ImportError:
        HAS_NEURAL_CUBE = False
    
    try:
        from bioplausible.models.hebbian_chain import DeepHebbianChain
        HAS_HEBBIAN = True
    except ImportError:
        HAS_HEBBIAN = False
    
    try:
        from bioplausible.models.feedback_alignment import FeedbackAlignmentMLP
        HAS_FA = True
    except ImportError:
        HAS_FA = False
    
    try:
        from bioplausible.models.dfa_eqprop import DirectFeedbackAlignment
        HAS_DFA = True
    except ImportError:
        HAS_DFA = False
    
    try:
        from bioplausible.models.chl import ContrastiveHebbianLearning
        HAS_CHL = True
    except ImportError:
        HAS_CHL = False
    
    try:
        from bioplausible.models.pc_hybrid import PredictiveCodingHybrid
        HAS_PC = True
    except ImportError:
        HAS_PC = False
    
    try:
        from bioplausible.models.eqprop_diffusion import EqPropDiffusion
        HAS_DIFFUSION = True
    except ImportError:
        HAS_DIFFUSION = False
    
    try:
        from bioplausible.models.holomorphic_ep import HolomorphicEqProp
        HAS_HOLO = True
    except ImportError:
        HAS_HOLO = False
    
    try:
        from bioplausible.models.finite_nudge_ep import FiniteNudgeEqProp
        HAS_FINITE_NUDGE = True
    except ImportError:
        HAS_FINITE_NUDGE = False
    
    try:
        from bioplausible.models.lazy_eqprop import LazyEqProp
        HAS_LAZY = True
    except ImportError:
        HAS_LAZY = False
    
    try:
        from bioplausible.models.sparse_eq import SparseEqProp
        HAS_SPARSE = True
    except ImportError:
        HAS_SPARSE = False
    
    try:
        from bioplausible.models.mom_eq import MomentumEqProp
        HAS_MOM = True
    except ImportError:
        HAS_MOM = False
    
    try:
        from bioplausible.models.sto_fa import StochasticFA
        HAS_STO_FA = True
    except ImportError:
        HAS_STO_FA = False
    
    try:
        from bioplausible.models.ada_fa import AdaptiveFeedbackAlignment
        HAS_ADA_FA = True
    except ImportError:
        HAS_ADA_FA = False
    
    try:
        from bioplausible.models.cf_align import ContrastiveFeedbackAlignment
        HAS_CF_FA = True
    except ImportError:
        HAS_CF_FA = False
    
    try:
        from bioplausible.models.eg_fa import EnergyGuidedFA
        HAS_EG_FA = True
    except ImportError:
        HAS_EG_FA = False
    
    try:
        from bioplausible.models.em_fa import EnergyMinimizingFA
        HAS_EM_FA = True
    except ImportError:
        HAS_EM_FA = False
    
    try:
        from bioplausible.models.eq_align import EquilibriumAlignment
        HAS_EQ_ALIGN = True
    except ImportError:
        HAS_EQ_ALIGN = False
    
    try:
        from bioplausible.models.leq_fa import LayerwiseEquilibriumFA
        HAS_LEQ_FA = True
    except ImportError:
        HAS_LEQ_FA = False
    
    try:
        from bioplausible.models.homeostatic import HomeostaticEqProp
        HAS_HOMEOSTATIC = True
    except ImportError:
        HAS_HOMEOSTATIC = False
    
    try:
        from bioplausible.models.temporal_resonance import TemporalResonance
        HAS_TEMPORAL = True
    except ImportError:
        HAS_TEMPORAL = False
    
    try:
        from bioplausible.models.ternary import TernaryEqProp
        HAS_TERNARY = True
    except ImportError:
        HAS_TERNARY = False
    
    try:
        from bioplausible.models.deep_ep import DeepEqProp
        HAS_DEEP = True
    except ImportError:
        HAS_DEEP = False
    
    # Register EqProp models
    ModelZoo.register(ModelSpec(
        name="looped_mlp",
        category="eqprop",
        model_class=LoopedMLP,
        description="Standard looped MLP with spectral normalization (workhorse of the library)",
        default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        tags=["vision", "lm", "stable", "baseline"],
    ))
    
    ModelZoo.register(ModelSpec(
        name="backprop_mlp",
        category="eqprop",
        model_class=BackpropMLP,
        description="Backpropagation baseline for comparison",
        default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        tags=["vision", "baseline"],
    ))
    
    ModelZoo.register(ModelSpec(
        name="conv_eqprop",
        category="eqprop",
        model_class=ConvEqProp,
        description="Convolutional EqProp for vision tasks",
        default_params={"input_channels": 1, "hidden_channels": 32, "output_dim": 10},
        tags=["vision", "cnn"],
    ))

    ModelZoo.register(ModelSpec(
        name="memory_efficient_mlp",
        category="eqprop",
        model_class=MemoryEfficientLoopedMLP,
        description="Memory-efficient looped MLP with gradient checkpointing",
        default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10, },
        tags=["vision", "lm", "memory_efficient", "deep"],
    ))

    ModelZoo.register(ModelSpec(
        name="transformer_eqprop",
        category="eqprop",
        model_class=TransformerEqProp,
        description="Transformer with EqProp dynamics",
        default_params={"vocab_size": 1000, "hidden_dim": 128, "output_dim": 10, "num_layers": 4, "num_heads": 4},
        tags=["lm", "attention", "deep"],
    ))
    
    # Register advanced EqProp variants
    if HAS_MODERN_CONV:
        ModelZoo.register(ModelSpec(
            name="modern_conv_eqprop",
            category="eqprop",
            model_class=ModernConvEqProp,
            description="Modern ConvEqProp with residuals and GroupNorm (CIFAR-10 optimized)",
            default_params={"input_channels": 3, "output_dim": 10},
            tags=["vision", "cnn", "residual", "sota"],
        ))
    
    if HAS_NEURAL_CUBE:
        ModelZoo.register(ModelSpec(
            name="neural_cube",
            category="eqprop",
            model_class=NeuralCube,
            description="3D lattice topology with local connectivity",
            default_params={"input_size": 784, "cube_size": 8, "output_dim": 10},
            tags=["vision", "topology", "local"],
        ))
    
    if HAS_HOLO:
        ModelZoo.register(ModelSpec(
            name="holomorphic_eqprop",
            category="eqprop",
            model_class=HolomorphicEqProp,
            description="Complex-valued EqProp for exact gradient estimation (NeurIPS 2024)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "complex", "research"],
        ))
    
    if HAS_FINITE_NUDGE:
        ModelZoo.register(ModelSpec(
            name="finite_nudge_eqprop",
            category="eqprop",
            model_class=FiniteNudgeEqProp,
            description="Large beta nudge for robustness to noise",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10, "beta": 0.5},
            tags=["vision", "robust", "research"],
        ))
    
    if HAS_LAZY:
        ModelZoo.register(ModelSpec(
            name="lazy_eqprop",
            category="eqprop",
            model_class=LazyEqProp,
            description="Event-driven updates (97% FLOP reduction)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "efficient", "event_driven"],
        ))
    
    if HAS_SPARSE:
        ModelZoo.register(ModelSpec(
            name="sparse_eqprop",
            category="eqprop",
            model_class=SparseEqProp,
            description="Top-K sparsity during settling (biological energy constraints)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10, "k": 0.2},
            tags=["vision", "sparse", "biological"],
        ))
    
    if HAS_MOM:
        ModelZoo.register(ModelSpec(
            name="momentum_eqprop",
            category="eqprop",
            model_class=MomentumEqProp,
            description="Momentum term for faster settling convergence",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "fast"],
        ))
    
    if HAS_DEEP:
        ModelZoo.register(ModelSpec(
            name="deep_eqprop",
            category="eqprop",
            model_class=DeepEqProp,
            description="Deep EqProp with asymmetric weights",
            default_params={"input_size": 784, "hidden_sizes": [256, 256, 256], "output_size": 10},
            tags=["vision", "deep", "asymmetric"],
        ))
    
    # Register Feedback Alignment family
    if HAS_FA:
        ModelZoo.register(ModelSpec(
            name="feedback_alignment",
            category="feedback_alignment",
            model_class=FeedbackAlignmentMLP,
            description="Fixed random feedback weights",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "bio_plausible"],
        ))
    
    if HAS_DFA:
        ModelZoo.register(ModelSpec(
            name="direct_fa",
            category="feedback_alignment",
            model_class=DirectFeedbackAlignment,
            description="Direct feedback from output to all layers",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "bio_plausible", "skip_connection"],
        ))
    
    if HAS_STO_FA:
        ModelZoo.register(ModelSpec(
            name="stochastic_fa",
            category="feedback_alignment",
            model_class=StochasticFA,
            description="Stochastic noise in feedback weights",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10, "noise_std": 0.1},
            tags=["vision", "robust", "stochastic"],
        ))
    
    if HAS_ADA_FA:
        ModelZoo.register(ModelSpec(
            name="adaptive_fa",
            category="feedback_alignment",
            model_class=AdaptiveFeedbackAlignment,
            description="Feedback weights adapt to align with forward weights",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "adaptive"],
        ))
    
    if HAS_CF_FA:
        ModelZoo.register(ModelSpec(
            name="contrastive_fa",
            category="feedback_alignment",
            model_class=ContrastiveFeedbackAlignment,
            description="Contrastive learning + Feedback Alignment",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "contrastive"],
        ))
    
    if HAS_EG_FA:
        ModelZoo.register(ModelSpec(
            name="energy_guided_fa",
            category="feedback_alignment",
            model_class=EnergyGuidedFA,
            description="Energy-guided feedback alignment",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "energy_based"],
        ))
    
    if HAS_EM_FA:
        ModelZoo.register(ModelSpec(
            name="energy_minimizing_fa",
            category="feedback_alignment",
            model_class=EnergyMinimizingFA,
            description="Energy-minimizing feedback alignment",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "energy_based"],
        ))
    
    if HAS_EQ_ALIGN:
        ModelZoo.register(ModelSpec(
            name="equilibrium_alignment",
            category="feedback_alignment",
            model_class=EquilibriumAlignment,
            description="Equilibrium-based alignment",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "equilibrium"],
        ))
    
    if HAS_LEQ_FA:
        ModelZoo.register(ModelSpec(
            name="layerwise_equilibrium_fa",
            category="feedback_alignment",
            model_class=LayerwiseEquilibriumFA,
            description="Layerwise equilibrium with FA",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "layerwise", "equilibrium"],
        ))
    
    # Register Hebbian learning
    if HAS_HEBBIAN:
        ModelZoo.register(ModelSpec(
            name="hebbian_chain",
            category="hebbian",
            model_class=DeepHebbianChain,
            description="Deep Hebbian chain (works up to 500 layers with SN)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10, },
            tags=["vision", "deep", "local_learning"],
        ))
    
    if HAS_CHL:
        ModelZoo.register(ModelSpec(
            name="contrastive_hebbian",
            category="hebbian",
            model_class=ContrastiveHebbianLearning,
            description="Contrastive Hebbian Learning (precursor to EqProp)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "contrastive"],
        ))
    
    # Register Hybrid models
    if HAS_PC:
        ModelZoo.register(ModelSpec(
            name="predictive_coding_hybrid",
            category="hybrid",
            model_class=PredictiveCodingHybrid,
            description="EqProp (bottom-up) + Predictive Coding (top-down)",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "predictive_coding", "hybrid"],
        ))
    
    if HAS_DIFFUSION:
        ModelZoo.register(ModelSpec(
            name="eqprop_diffusion",
            category="hybrid",
            model_class=EqPropDiffusion,
            description="Energy-based denoising diffusion",
            default_params={"input_size": 784, "latent_size": 64, "timesteps": 100},
            tags=["generative", "diffusion", "energy_based"],
        ))
    
    # Register biological variants
    if HAS_HOMEOSTATIC:
        ModelZoo.register(ModelSpec(
            name="homeostatic_eqprop",
            category="eqprop",
            model_class=HomeostaticEqProp,
            description="Homeostatic regulation for biological plausibility",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "biological", "homeostatic"],
        ))
    
    if HAS_TEMPORAL:
        ModelZoo.register(ModelSpec(
            name="temporal_resonance",
            category="eqprop",
            model_class=TemporalResonance,
            description="Spike-timing dependent plasticity",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "spiking", "temporal"],
        ))
    
    if HAS_TERNARY:
        ModelZoo.register(ModelSpec(
            name="ternary_eqprop",
            category="eqprop",
            model_class=TernaryEqProp,
            description="Ternary weights {-1, 0, +1}",
            default_params={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
            tags=["vision", "quantized", "efficient"],
        ))


# ============================================================================
# OPTIMIZER REGISTRY POPULATION
# ============================================================================

def _populate_optimizer_registry() -> None:
    """Populate the OptimizerZoo with all available MEP optimizers."""
    
    # Import MEP optimizers
    try:
        from mep import (
            smep,
            smep_fast,
            sdmep,
            local_ep,
            natural_ep,
            muon_backprop,
        )
        from mep.optimizers import (
            CompositeOptimizer,
            EPGradient,
            MuonUpdate,
            SpectralConstraint,
        )
        HAS_MEP = True
    except ImportError:
        HAS_MEP = False
    
    if HAS_MEP:
        # Register MEP optimizers
        OptimizerZoo.register(OptimizerSpec(
            name="smep",
            category="ep",
            optimizer_class=smep,
            description="Spectral Muon Equilibrium Propagation (default, validated)",
            default_params={
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "mode": "ep",
                "settle_steps": 30,
                "settle_lr": 0.15,
                "beta": 0.5,
                "loss_type": "mse",
            },
            tags=["ep", "stable", "validated", "default"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="smep_fast",
            category="ep",
            optimizer_class=smep_fast,
            description="Fast SMEP with fewer settling steps (4-6x speedup)",
            default_params={
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "mode": "ep",
                "settle_steps": 10,
                "settle_lr": 0.2,
                "beta": 0.5,
                "loss_type": "mse",
            },
            tags=["ep", "fast"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="sdmep",
            category="ep",
            optimizer_class=sdmep,
            description="Spectral Dion-Muon EP (low-rank SVD for large models)",
            default_params={
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "mode": "ep",
                "settle_steps": 15,
                "settle_lr": 0.1,
                "beta": 0.3,
                "loss_type": "cross_entropy",
                "use_error_feedback": True,
            },
            tags=["ep", "low_rank", "large_models"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="local_ep",
            category="ep",
            optimizer_class=local_ep,
            description="Layer-local EP (biologically plausible, local learning)",
            default_params={
                "lr": 0.02,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "mode": "ep",
                "settle_steps": 20,
                "settle_lr": 0.05,
                "beta": 0.1,
                "loss_type": "mse",
            },
            tags=["ep", "local_learning", "bio_plausible"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="natural_ep",
            category="natural_gradient",
            optimizer_class=natural_ep,
            description="Natural gradient EP with Fisher whitening",
            default_params={
                "lr": 0.02,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "mode": "ep",
                "settle_steps": 20,
                "settle_lr": 0.05,
                "beta": 0.5,
                "loss_type": "mse",
            },
            tags=["ep", "natural_gradient", "fisher"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="muon_backprop",
            category="backprop",
            optimizer_class=muon_backprop,
            description="Muon optimizer with standard backprop (SGD/Adam replacement)",
            default_params={
                "lr": 0.02,
                "momentum": 0.9,
                "weight_decay": 0.0005,
                "ns_steps": 5,
                "gamma": 0.95,
            },
            tags=["backprop", "orthogonal", "drop_in"],
        ))
    
    # Register Bioplausible's native optimizers/training approaches
    try:
        from torch.optim import SGD, Adam, AdamW
        
        OptimizerZoo.register(OptimizerSpec(
            name="sgd",
            category="backprop",
            optimizer_class=SGD,
            description="Standard SGD with momentum",
            default_params={"lr": 0.01, "momentum": 0.9, "weight_decay": 0.0005},
            tags=["backprop", "baseline"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="adam",
            category="backprop",
            optimizer_class=Adam,
            description="Adam optimizer (default baseline)",
            default_params={"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.0},
            tags=["backprop", "adaptive", "baseline"],
        ))
        
        OptimizerZoo.register(OptimizerSpec(
            name="adamw",
            category="backprop",
            optimizer_class=AdamW,
            description="AdamW with decoupled weight decay",
            default_params={"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0.01},
            tags=["backprop", "adaptive", "decoupled_wd"],
        ))
    except ImportError:
        pass


# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_zoo() -> None:
    """Initialize the model and optimizer zoo."""
    _populate_model_registry()
    _populate_optimizer_registry()


# Auto-initialize on import
initialize_zoo()


__all__ = ["initialize_zoo"]
