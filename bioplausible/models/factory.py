"""
Model Factory

Centralizes model creation logic for Experiment Runner and UI.
Now uses a registration system to allow for easier extension.
"""

from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

# Hybrid / Experimental Models
from bioplausible.models.ada_fa import AdaptiveFeedbackAlignment
from bioplausible.models.backprop_transformer_lm import BackpropTransformerLM
from bioplausible.models.base import ModelConfig
from bioplausible.models.cf_align import ContrastiveFeedbackAlignment
from bioplausible.models.deep_ep import DirectedEP
from bioplausible.models.eg_fa import EnergyGuidedFA
from bioplausible.models.em_fa import EnergyMinimizingFA
from bioplausible.models.eq_align import EquilibriumAlignment
from bioplausible.models.eqprop_diffusion import EqPropDiffusion
from bioplausible.models.eqprop_lm_variants import create_eqprop_lm
from bioplausible.models.finite_nudge_ep import FiniteNudgeEP
from bioplausible.models.hebbian_chain import DeepHebbianChain

# Advanced EqProp Models
from bioplausible.models.holomorphic_ep import HolomorphicEP
from bioplausible.models.leq_fa import LayerwiseEquilibriumFA

# Standard Models
from bioplausible.models.looped_mlp import BackpropMLP, LoopedMLP
from bioplausible.models.modern_conv_eqprop import ModernConvEqProp
from bioplausible.models.mom_eq import MomentumEquilibrium
from bioplausible.models.neural_cube import NeuralCube
from bioplausible.models.pc_hybrid import PredictiveCodingHybrid
from bioplausible.models.registry import ModelSpec
from bioplausible.models.simple_fa import StandardFA
from bioplausible.models.sparse_eq import SparseEquilibrium
from bioplausible.models.sto_fa import StochasticFA

# =============================================================================
# Registry Logic
# =============================================================================


class ModelFactoryRegistry:
    _builders: Dict[str, Callable] = {}

    @classmethod
    def register(cls, model_type: str):
        def decorator(func: Callable):
            cls._builders[model_type] = func
            return func

        return decorator

    @classmethod
    def get_builder(cls, model_type: str) -> Optional[Callable]:
        return cls._builders.get(model_type)


register_model_builder = ModelFactoryRegistry.register

# =============================================================================
# Helper Functions
# =============================================================================


def _make_config(
    name: str,
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    num_layers: int,
    spec: ModelSpec,
) -> ModelConfig:
    return ModelConfig(
        name=name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=[hidden_dim] * min(num_layers, 5),
        beta=0.1,  # Default, overridden by hyperopt if needed
        learning_rate=spec.default_lr,
        equilibrium_steps=20,  # Default, overridden by hyperopt if needed
        use_spectral_norm=True,
    )


# =============================================================================
# Builders
# =============================================================================


@register_model_builder("backprop")
def build_backprop(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    if task_type == "lm":
        return BackpropTransformerLM(
            vocab_size=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=256,
        ).to(device)
    else:
        return BackpropMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
        ).to(device)


@register_model_builder("eqprop_transformer")
def build_eqprop_transformer(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    return create_eqprop_lm(
        spec.variant,
        vocab_size=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_sn=True,
    ).to(device)


@register_model_builder("eqprop_mlp")
def build_eqprop_mlp(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    return LoopedMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_spectral_norm=True,
    ).to(device)


@register_model_builder("dfa")
def build_dfa(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type):
    config = _make_config(
        "feedback_alignment", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return StandardFA(config=config).to(device)


@register_model_builder("chl")
def build_chl(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type):
    config = _make_config(
        "cf_align", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return ContrastiveFeedbackAlignment(config=config).to(device)


@register_model_builder("deep_hebbian")
def build_deep_hebbian(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    return DeepHebbianChain(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        use_spectral_norm=True,
        hebbian_lr=0.001,
        use_oja=True,
    ).to(device)


@register_model_builder("holomorphic_ep")
def build_holomorphic_ep(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "holomorphic_ep", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return HolomorphicEP(config=config, device=device).to(device)


@register_model_builder("directed_ep")
def build_directed_ep(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "directed_ep", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return DirectedEP(config=config, device=device).to(device)


@register_model_builder("finite_nudge_ep")
def build_finite_nudge_ep(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "finite_nudge_ep", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return FiniteNudgeEP(config=config, device=device).to(device)


@register_model_builder("modern_conv_eqprop")
def build_modern_conv_eqprop(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    return ModernConvEqProp(
        eq_steps=30,  # Default
        hidden_channels=hidden_dim,
    ).to(device)


@register_model_builder("eqprop_diffusion")
def build_eqprop_diffusion(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    # input_dim is interpreted as channels for vision tasks
    channels = input_dim if input_dim is not None else 1

    # Heuristic for flattened inputs (e.g., from TrialRunner)
    if channels == 784:  # MNIST flattened
        channels = 1
    elif channels == 3072:  # CIFAR-10 flattened
        channels = 3
    elif channels > 10:
        # Generic heuristic: check if square (grayscale) or 3*square (RGB)
        side = int(channels**0.5)
        if side * side == channels:
            channels = 1
        elif (channels % 3 == 0) and (int((channels / 3) ** 0.5) ** 2 * 3 == channels):
            channels = 3

    return EqPropDiffusion(img_channels=channels, hidden_channels=hidden_dim).to(device)


@register_model_builder("neural_cube")
def build_neural_cube(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    # Derive cube size from hidden_dim approx
    # hidden_dim = n_neurons = cube_size^3
    cube_size = int(round(hidden_dim ** (1 / 3)))
    return NeuralCube(
        cube_size=max(4, cube_size),  # Min size 4
        input_dim=input_dim,
        output_dim=output_dim,
    ).to(device)


@register_model_builder("adaptive_feedback_alignment")
def build_adaptive_fa(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "adaptive_feedback_alignment",
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        spec,
    )
    return AdaptiveFeedbackAlignment(config=config).to(device)


@register_model_builder("eq_align")
def build_eq_align(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    return EquilibriumAlignment(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        max_steps=30,  # Default
        use_spectral_norm=True,
        learning_rate=spec.default_lr,
    ).to(device)


@register_model_builder("layerwise_equilibrium_fa")
def build_leq_fa(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "layerwise_equilibrium_fa", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return LayerwiseEquilibriumFA(config=config).to(device)


@register_model_builder("energy_guided_fa")
def build_eg_fa(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type):
    config = _make_config(
        "energy_guided_fa", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return EnergyGuidedFA(config=config).to(device)


@register_model_builder("predictive_coding_hybrid")
def build_pc_hybrid(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "predictive_coding_hybrid", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return PredictiveCodingHybrid(config=config).to(device)


@register_model_builder("sparse_equilibrium")
def build_sparse_eq(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "sparse_equilibrium", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return SparseEquilibrium(config=config).to(device)


@register_model_builder("momentum_equilibrium")
def build_mom_eq(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "momentum_equilibrium", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return MomentumEquilibrium(config=config).to(device)


@register_model_builder("stochastic_fa")
def build_sto_fa(
    spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type
):
    config = _make_config(
        "stochastic_fa", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return StochasticFA(config=config).to(device)


@register_model_builder("energy_minimizing_fa")
def build_em_fa(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type):
    config = _make_config(
        "energy_minimizing_fa", input_dim, output_dim, hidden_dim, num_layers, spec
    )
    return EnergyMinimizingFA(config=config).to(device)


# =============================================================================
# Main Factory Function
# =============================================================================


def create_model(
    spec: ModelSpec,
    input_dim: Optional[int],
    output_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 4,
    device: str = "cpu",
    task_type: str = "lm",  # "lm", "vision", "rl"
    **kwargs,
) -> nn.Module:
    """
    Factory method to create a model instance from a specification.
    """
    model_type = spec.model_type

    # Decide if we need embeddings (LM only, usually)
    # If input_dim is provided, we assume vector input (Vision/RL)
    # Exclude models that handle their own embeddings (Transformers)
    use_embedding = (
        (input_dim is None)
        and (task_type == "lm")
        and (model_type not in ["backprop", "eqprop_transformer"])
    )

    input_size = input_dim if input_dim is not None else hidden_dim

    # Get builder
    builder = ModelFactoryRegistry.get_builder(model_type)
    if not builder:
        raise ValueError(f"Unknown model type: {model_type}")

    # Build model
    model = builder(
        spec, input_size, output_dim, hidden_dim, num_layers, device, task_type
    )

    # Configure optional properties from kwargs
    if "gradient_method" in kwargs and hasattr(model, "gradient_method"):
        model.gradient_method = kwargs["gradient_method"]
    if "beta" in kwargs and hasattr(model, "beta"):
        model.beta = kwargs["beta"]

    # Attach embedding if needed
    embedding_layer = None
    if use_embedding:
        embedding_layer = nn.Embedding(output_dim, hidden_dim).to(device)
        model.embed = embedding_layer
        model.has_embed = True
    else:
        model.has_embed = False

    return model.to(device)


def load_weights(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    strict: bool = False,
    freeze_layers: bool = False,
):
    """
    Load weights from a checkpoint path.

    Args:
        model: Target model
        path: Path to .pt file
        device: Device to load onto
        strict: If True, require exact match of keys
        freeze_layers: If True, freeze all loaded layers (for transfer learning probe)
    """
    if not path:
        return

    try:
        print(f"Loading weights from {path}...")
        state_dict = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        if freeze_layers:
            print("Freezing loaded layers for transfer learning...")
            # Freeze everything that was loaded
            for name, param in model.named_parameters():
                if name in state_dict:
                    param.requires_grad = False
                else:
                    # Likely the head/probe
                    print(f"  -> {name} remains trainable")

    except Exception as e:
        print(f"Failed to load weights: {e}")
