"""
Model Factory

Centralizes model creation logic for Experiment Runner and UI.
Now uses a registration system to allow for easier extension.
"""

from typing import Callable, Dict, Optional, Any

import torch
import torch.nn as nn
from bioplausible.models.registry import ModelRegistry, ModelSpec


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

    # Handle Backprop variant switching
    if model_type == "backprop":
        if task_type == "lm":
            model_cls = ModelRegistry.get("backprop_transformer_lm")
        else:
            model_cls = ModelRegistry.get("backprop_mlp")
    else:
        try:
            model_cls = ModelRegistry.get(model_type)
        except ValueError:
            raise ValueError(f"Unknown model type: {model_type}")

    # Decide if we need embeddings (LM only, usually)
    # If input_dim is provided, we assume vector input (Vision/RL)
    # Exclude models that handle their own embeddings (Transformers)
    use_embedding = (
        (input_dim is None)
        and (task_type == "lm")
        and (model_type not in ["backprop", "eqprop_transformer"])
    )

    input_size = input_dim if input_dim is not None else hidden_dim

    # Use build method if available (BioModel and others implement it)
    if hasattr(model_cls, "build"):
        model = model_cls.build(
            spec,
            input_size,
            output_dim,
            hidden_dim,
            num_layers,
            device,
            task_type,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"Model class {model_cls.__name__} does not implement 'build' method."
        )

    # Attach embedding if needed (Generic BioModel on LM Task)
    if use_embedding:
        embedding_layer = nn.Embedding(output_dim, hidden_dim).to(device)
        model.embed = embedding_layer
        model.has_embed = True

        # Patch forward to use embedding
        original_forward = model.forward
        def forward_with_embed(x, **fw_kwargs):
            h = embedding_layer(x)
            return original_forward(h, **fw_kwargs)

        # Bind method to instance
        model.forward = forward_with_embed
    else:
        model.has_embed = False

    return model


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
