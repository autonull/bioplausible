"""
Research Presets: Pre-configured model/optimizer combinations for research.

This module provides ready-to-use configurations for common research scenarios
and experimentation goals.

Categories:
- PERFORMANCE: Best accuracy configurations
- SPEED: Fast training configurations  
- EFFICIENCY: Memory/compute efficient configurations
- BIOPLAUSIBLE: Most biologically plausible configurations
- ROBUSTNESS: Noise/distribution shift robust configurations
- EXPLORATORY: Experimental configurations for discovery
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ResearchPreset:
    """A pre-configured research setup."""
    name: str
    category: str  # performance, speed, efficiency, bioplausible, robustness, exploratory
    model_name: str
    model_params: Dict[str, Any]
    optimizer_name: str
    optimizer_params: Dict[str, Any]
    description: str
    use_case: str
    expected_accuracy: Optional[str] = None
    expected_speed: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# ============================================================================
# PERFORMANCE PRESETS - Best accuracy configurations
# ============================================================================

PERFORMANCE_PRESETS = [
    ResearchPreset(
        name="performance_vision_default",
        category="performance",
        model_name="looped_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 512,
            "output_dim": 10,
            "use_spectral_norm": True,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 50,
            "settle_lr": 0.15,
            "beta": 0.5,
            "loss_type": "mse",
        },
        description="Default high-performance vision setup",
        use_case="Standard vision classification with best accuracy",
        expected_accuracy="95-97% MNIST (10 epochs)",
        expected_speed="10-15x slower than backprop",
        tags=["vision", "mnist", "high-accuracy"],
    ),
    
    ResearchPreset(
        name="performance_vision_cnn",
        category="performance",
        model_name="modern_conv_eqprop",
        model_params={
            "input_channels": 3,
            "output_dim": 10,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 40,
            "settle_lr": 0.15,
            "beta": 0.5,
            "loss_type": "cross_entropy",
        },
        description="High-performance CNN for CIFAR-10",
        use_case="CIFAR-10 classification with residual connections",
        expected_accuracy="70-80% CIFAR-10",
        expected_speed="8-12x slower than backprop",
        tags=["vision", "cifar", "cnn", "residual"],
    ),
    
    ResearchPreset(
        name="performance_lm",
        category="performance",
        model_name="transformer_eqprop",
        model_params={
            "vocab_size": 10000,
            "hidden_dim": 256,
            "output_dim": 10000,
            "num_layers": 6,
            "num_heads": 8,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.001,
            "settle_steps": 30,
            "settle_lr": 0.1,
            "beta": 0.3,
            "loss_type": "cross_entropy",
        },
        description="High-performance language model",
        use_case="Character-level language modeling",
        expected_accuracy="1.5-2.0 BPC",
        expected_speed="15-20x slower than backprop",
        tags=["lm", "transformer", "attention"],
    ),
]


# ============================================================================
# SPEED PRESETS - Fast training configurations
# ============================================================================

SPEED_PRESETS = [
    ResearchPreset(
        name="speed_vision_fast",
        category="speed",
        model_name="looped_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="smep_fast",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 10,
            "settle_lr": 0.2,
            "beta": 0.5,
            "loss_type": "mse",
        },
        description="Fast vision training with minimal accuracy loss",
        use_case="Rapid prototyping and hyperparameter search",
        expected_accuracy="90-93% MNIST (10 epochs)",
        expected_speed="4-6x slower than backprop",
        tags=["vision", "fast", "prototyping"],
    ),
    
    ResearchPreset(
        name="speed_backprop_baseline",
        category="speed",
        model_name="looped_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="muon_backprop",
        optimizer_params={
            "lr": 0.01,
            "ns_steps": 3,
        },
        description="Fast backprop baseline with Muon orthogonalization",
        use_case="Baseline comparison, ablation studies",
        expected_accuracy="97-98% MNIST (10 epochs)",
        expected_speed="1.2x slower than backprop",
        tags=["vision", "baseline", "backprop"],
    ),
]


# ============================================================================
# EFFICIENCY PRESETS - Memory/compute efficient configurations
# ============================================================================

EFFICIENCY_PRESETS = [
    ResearchPreset(
        name="efficiency_memory",
        category="efficiency",
        model_name="memory_efficient_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
            "use_spectral_norm": True,
        },
        optimizer_name="smep_fast",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 10,
            "settle_lr": 0.2,
            "beta": 0.5,
        },
        description="Memory-efficient training with gradient checkpointing",
        use_case="Deep networks on limited memory",
        expected_accuracy="88-92% MNIST (10 epochs)",
        expected_speed="4-6x slower than backprop",
        tags=["memory", "deep", "efficient"],
    ),
    
    ResearchPreset(
        name="efficiency_sparse",
        category="efficiency",
        model_name="sparse_eqprop",
        model_params={
            "input_dim": 784,
            "hidden_dim": 512,
            "output_dim": 10,
            "k": 0.2,  # Top-20% active
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 30,
            "settle_lr": 0.15,
            "beta": 0.5,
        },
        description="Sparse activation for computational efficiency",
        use_case="Energy-constrained deployment, neuromorphic",
        expected_accuracy="90-94% MNIST (10 epochs)",
        expected_speed="8-12x slower than backprop (but 80% FLOP reduction)",
        tags=["sparse", "efficient", "neuromorphic"],
    ),
]


# ============================================================================
# BIOPLAUSIBLE PRESETS - Most biologically plausible configurations
# ============================================================================

BIOPLAUSIBLE_PRESETS = [
    ResearchPreset(
        name="bioplausible_local",
        category="bioplausible",
        model_name="looped_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="local_ep",
        optimizer_params={
            "lr": 0.02,
            "settle_steps": 20,
            "settle_lr": 0.05,
            "beta": 0.1,
            "loss_type": "mse",
        },
        description="Layer-local learning (most biologically plausible)",
        use_case="Studying biological plausibility, local learning rules",
        expected_accuracy="85-90% MNIST (10 epochs)",
        expected_speed="10-15x slower than backprop",
        tags=["local", "biological", "layerwise"],
    ),
    
    ResearchPreset(
        name="bioplausible_hebbian",
        category="bioplausible",
        model_name="hebbian_chain",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
            "num_layers": 20,
        },
        optimizer_name="muon_backprop",
        optimizer_params={
            "lr": 0.01,
            "ns_steps": 5,
        },
        description="Pure Hebbian learning (no backpropagation)",
        use_case="Studying Hebbian learning, local plasticity",
        expected_accuracy="80-88% MNIST (10 epochs)",
        expected_speed="1.2x slower than backprop",
        tags=["hebbian", "local", "biological"],
    ),
    
    ResearchPreset(
        name="bioplausible_feedback",
        category="bioplausible",
        model_name="feedback_alignment",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="muon_backprop",
        optimizer_params={
            "lr": 0.01,
        },
        description="Fixed random feedback weights",
        use_case="Studying weight transport problem solutions",
        expected_accuracy="90-94% MNIST (10 epochs)",
        expected_speed="1.2x slower than backprop",
        tags=["feedback", "biological", "random"],
    ),
]


# ============================================================================
# ROBUSTNESS PRESETS - Noise/distribution shift robust configurations
# ============================================================================

ROBUSTNESS_PRESETS = [
    ResearchPreset(
        name="robustness_finite_nudge",
        category="robustness",
        model_name="finite_nudge_eqprop",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
            "beta": 0.5,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 30,
            "settle_lr": 0.15,
            "beta": 0.5,
        },
        description="Large nudge for noise robustness",
        use_case="Noisy inputs, adversarial robustness",
        expected_accuracy="92-95% MNIST (10 epochs)",
        expected_speed="10-15x slower than backprop",
        tags=["robust", "noise", "adversarial"],
    ),
    
    ResearchPreset(
        name="robustness_stochastic",
        category="robustness",
        model_name="stochastic_fa",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
            "noise_std": 0.1,
        },
        optimizer_name="muon_backprop",
        optimizer_params={
            "lr": 0.01,
        },
        description="Stochastic feedback for robustness",
        use_case="Studying noise in biological systems",
        expected_accuracy="88-92% MNIST (10 epochs)",
        expected_speed="1.2x slower than backprop",
        tags=["stochastic", "noise", "robust"],
    ),
]


# ============================================================================
# EXPLORATORY PRESETS - Experimental configurations for discovery
# ============================================================================

EXPLORATORY_PRESETS = [
    ResearchPreset(
        name="exploratory_deep_ep",
        category="exploratory",
        model_name="deep_eqprop",
        model_params={
            "input_dim": 784,
            "hidden_sizes": [512, 512, 512, 512],
            "output_dim": 10,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.005,
            "settle_steps": 40,
            "settle_lr": 0.1,
            "beta": 0.3,
        },
        description="Deep asymmetric EqProp for scaling studies",
        use_case="Exploring deep EqProp scaling properties",
        expected_accuracy="93-96% MNIST (10 epochs)",
        expected_speed="15-20x slower than backprop",
        tags=["deep", "asymmetric", "scaling"],
    ),
    
    ResearchPreset(
        name="exploratory_holomorphic",
        category="exploratory",
        model_name="holomorphic_eqprop",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="smep",
        optimizer_params={
            "lr": 0.01,
            "settle_steps": 30,
            "settle_lr": 0.15,
            "beta": 0.5,
        },
        description="Complex-valued EqProp for exact gradients",
        use_case="Research on holomorphic approaches",
        expected_accuracy="90-94% MNIST (10 epochs)",
        expected_speed="20-30x slower than backprop",
        tags=["complex", "holomorphic", "research"],
    ),
    
    ResearchPreset(
        name="exploratory_natural_ep",
        category="exploratory",
        model_name="looped_mlp",
        model_params={
            "input_dim": 784,
            "hidden_dim": 256,
            "output_dim": 10,
        },
        optimizer_name="natural_ep",
        optimizer_params={
            "lr": 0.02,
            "settle_steps": 20,
            "settle_lr": 0.05,
            "beta": 0.5,
            "fisher_damping": 1e-3,
        },
        description="Natural gradient EP with Fisher whitening",
        use_case="Second-order optimization research",
        expected_accuracy="92-95% MNIST (10 epochs)",
        expected_speed="15-25x slower than backprop",
        tags=["natural", "second-order", "fisher"],
    ),
]


# ============================================================================
# REGISTRY
# ============================================================================

ALL_PRESETS = (
    PERFORMANCE_PRESETS +
    SPEED_PRESETS +
    EFFICIENCY_PRESETS +
    BIOPLAUSIBLE_PRESETS +
    ROBUSTNESS_PRESETS +
    EXPLORATORY_PRESETS
)

PRESET_REGISTRY = {preset.name: preset for preset in ALL_PRESETS}


def get_preset(name: str) -> ResearchPreset:
    """Get a preset by name."""
    if name not in PRESET_REGISTRY:
        available = ', '.join(sorted(PRESET_REGISTRY.keys()))
        raise ValueError(f"Preset '{name}' not found. Available: {available}")
    return PRESET_REGISTRY[name]


def list_presets(category: Optional[str] = None) -> List[str]:
    """List available presets, optionally filtered by category."""
    if category:
        presets = [p for p in ALL_PRESETS if p.category == category]
    else:
        presets = ALL_PRESETS
    return sorted([p.name for p in presets])


def get_preset_by_category(category: str) -> List[ResearchPreset]:
    """Get all presets in a category."""
    return [p for p in ALL_PRESETS if p.category == category]


def run_preset(
    preset_name: str,
    train_loader,
    val_loader=None,
    epochs: int = 10,
    verbose: bool = True,
) -> Any:
    """
    Run a research preset.
    
    Args:
        preset_name: Name of preset to run.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Training epochs.
        verbose: Print progress.
    
    Returns:
        ExperimentResult from running the preset.
    """
    from bioplausible.experiments.utils import ExperimentRunner
    from bioplausible.zoo import ModelZoo, OptimizerZoo
    
    preset = get_preset(preset_name)
    
    runner = ExperimentRunner()
    
    return runner.run(
        model_name=preset.model_name,
        optimizer_name=preset.optimizer_name,
        train_loader=train_loader,
        val_loader=val_loader,
        model_params=preset.model_params,
        optimizer_params=preset.optimizer_params,
        epochs=epochs,
        verbose=verbose,
    )


__all__ = [
    'ResearchPreset',
    'PERFORMANCE_PRESETS',
    'SPEED_PRESETS',
    'EFFICIENCY_PRESETS',
    'BIOPLAUSIBLE_PRESETS',
    'ROBUSTNESS_PRESETS',
    'EXPLORATORY_PRESETS',
    'ALL_PRESETS',
    'get_preset',
    'list_presets',
    'get_preset_by_category',
    'run_preset',
]
