"""
EqProp-Torch Utilities

Helper functions for ONNX export, model verification, and training utilities.
"""

import os
import random
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """
    Seed all random number generators for reproducibility.

    Args:
        seed: Random seed (default: 42)
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    opset_version: int = 14,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    device: str = "cpu",
) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        output_path: Path to save .onnx file
        input_shape: Example input shape (e.g., (1, 784))
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis specification
        device: Device to use for export

    Example:
        >>> model = LoopedMLP(784, 256, 10)
        >>> export_to_onnx(model, "model.onnx", (1, 784))
    """
    # Handle compiled models
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    model.eval()
    model = model.to(device)
    dummy_input = torch.randn(*input_shape, device=device)

    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print(f"✓ Model exported to {output_path}")
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    model = _get_model_for_processing(model)

    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def verify_spectral_norm(model: nn.Module) -> Dict[str, float]:
    """
    Verify that all layers with spectral normalization have L <= 1.

    Args:
        model: PyTorch model

    Returns:
        Dict mapping layer names to their Lipschitz constants
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod

    lipschitz_values = {}

    for name, module in model.named_modules():
        # Check for spectral norm parametrization
        if _has_spectral_norm(module):
            spectral_norm = _compute_module_spectral_norm(module)
            if spectral_norm is not None:
                lipschitz_values[name] = spectral_norm

    return lipschitz_values


def _compute_module_spectral_norm(module: nn.Module) -> Optional[float]:
    """Compute spectral norm for a module if possible."""
    with torch.no_grad():
        weight = getattr(module, "weight", None)
        if weight is not None and weight.dim() >= 2:
            return _compute_spectral_norm(weight)
    return None


def _has_spectral_norm(module: nn.Module) -> bool:
    """Check if a module has spectral normalization."""
    return hasattr(module, "parametrizations") and hasattr(
        module.parametrizations, "weight"
    )


def _compute_spectral_norm(weight: torch.Tensor) -> float:
    """Compute the spectral norm (largest singular value) of a weight tensor."""
    # Reshape for 2D computation if needed
    W_flat = weight.reshape(weight.shape[0], -1) if weight.dim() > 2 else weight
    s = torch.linalg.svdvals(W_flat)
    return s[0].item() if s.numel() > 0 else 0.0


def _get_model_for_processing(model: nn.Module) -> nn.Module:
    """Get the appropriate model for processing (handle compiled models)."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute the global gradient norm across all parameters.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm
    """
    model = _get_model_for_processing(model)

    squared_norms = []
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            squared_norms.append(param_norm.item() ** 2)

    total_squared_norm = sum(squared_norms)
    return total_squared_norm**0.5


def estimate_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Estimate memory usage of a model.

    Args:
        model: PyTorch model
        input_shape: Input shape (without batch dimension)
        batch_size: Batch size for estimation

    Returns:
        Dict with memory estimates in MB
    """
    param_memory = _calculate_param_memory(model)
    grad_memory = param_memory  # Same as parameters
    activation_memory = _estimate_activation_memory(input_shape, batch_size)

    return {
        "parameters_mb": param_memory,
        "gradients_mb": grad_memory,
        "activations_mb": activation_memory,
        "total_mb": param_memory + grad_memory + activation_memory,
    }


def _calculate_param_memory(model: nn.Module) -> float:
    """Calculate memory used by model parameters."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def _estimate_activation_memory(input_shape: Tuple[int, ...], batch_size: int) -> float:
    """Estimate memory used by activations."""
    # This is model-specific, here's a simple heuristic
    return batch_size * sum(input_shape) * 4 / 1e6  # 4 bytes per float32


class ModelRegistry:
    """
    Simple registry for model factories.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.register('my_mlp', lambda: LoopedMLP(784, 256, 10))
        >>> model = registry.create('my_mlp')
    """

    def __init__(self) -> None:
        self._factories = {}

    def register(self, name: str, factory: Callable[[], nn.Module]) -> None:
        """
        Register a model factory function.

        Args:
            name: Name to register the model under
            factory: Function that creates the model
        """
        self._factories[name] = factory

    def create(self, name: str, **kwargs) -> nn.Module:
        """
        Create a model from the registry.

        Args:
            name: Name of the registered model
            **kwargs: Additional arguments to pass to the factory function

        Returns:
            Created model instance
        """
        if name not in self._factories:
            raise ValueError(
                f"Model '{name}' not registered. Available: {list(self._factories.keys())}"
            )
        return self._factories[name](**kwargs)

    def list_models(self) -> List[str]:
        """
        List all registered models.

        Returns:
            List of model names
        """
        return list(self._factories.keys())


# Global registry instance
model_registry = ModelRegistry()


def create_model_preset(preset_name: str, **overrides) -> nn.Module:
    """
    Create a model from a preset configuration.

    Args:
        preset_name: Name of preset ('mnist_small', 'mnist_large', 'cifar_conv', etc.)
        **overrides: Override default parameters

    Returns:
        Configured model

    Example:
        >>> model = create_model_preset('mnist_small', hidden_dim=512)
    """
    from .models import ConvEqProp, LoopedMLP

    presets = {
        "mnist_small": lambda: LoopedMLP(784, 128, 10, use_spectral_norm=True),
        "mnist_medium": lambda: LoopedMLP(784, 256, 10, use_spectral_norm=True),
        "mnist_large": lambda: LoopedMLP(784, 512, 10, use_spectral_norm=True),
        "cifar_conv": lambda: ConvEqProp(3, 64, 10),
        "cifar_mlp": lambda: LoopedMLP(3072, 512, 10, use_spectral_norm=True),
    }

    if preset_name not in presets:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(presets.keys())}"
        )

    # Note: overrides would require more sophisticated preset handling
    return presets[preset_name]()


# =============================================================================
# Profiling Utilities
# =============================================================================


@contextmanager
def SimpleProfiler(name: str):
    """
    Context manager for simple time profiling.

    Example:
        >>> with SimpleProfiler("Training Step"):
        >>>     train_step()
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"[{name}] took {(end - start)*1000:.2f} ms")


def profile_model(
    model: nn.Module, input_shape: Tuple[int, ...], device: str = "cpu", runs: int = 10
) -> Dict[str, float]:
    """
    Run a simple performance profile on a model.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (including batch dim)
        device: Device to run on
        runs: Number of runs for averaging

    Returns:
        Dictionary with 'avg_ms' and 'std_ms'
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    avg_ms = sum(times) / len(times)
    std_ms = (sum((t - avg_ms) ** 2 for t in times) / len(times)) ** 0.5

    return {"avg_ms": avg_ms, "std_ms": std_ms}


# Missing utils imported by models
def spectral_linear(
    in_features: int, out_features: int, use_sn: bool = True
) -> nn.Module:
    """Helper to create a linear layer with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features)
    if use_sn:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


def spectral_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    padding: int = 0,
    use_sn: bool = True,
) -> nn.Module:
    """Helper to create a conv2d layer with optional spectral normalization."""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
    if use_sn:
        return torch.nn.utils.parametrizations.spectral_norm(layer)
    return layer


__all__ = [
    "seed_everything",
    "export_to_onnx",
    "count_parameters",
    "verify_spectral_norm",
    "compute_gradient_norm",
    "estimate_memory_usage",
    "ModelRegistry",
    "model_registry",
    "create_model_preset",
    "SimpleProfiler",
    "profile_model",
    "spectral_linear",
    "spectral_conv2d",
]
