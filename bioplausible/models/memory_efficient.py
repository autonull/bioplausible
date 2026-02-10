"""
Memory-Efficient Models for O(1) Training

These models provide O(1) memory training by leveraging the NumPy/CuPy kernel backend.
"""

from typing import Any, Dict, Optional

import torch

from ..kernel import HAS_CUPY
from .eqprop_base import EqPropModel
from .looped_mlp import LoopedMLP


class MemoryEfficientLoopedMLP(LoopedMLP):
    """
    Memory-efficient version of LoopedMLP that defaults to O(1) memory kernel backend.

    This model uses the NumPy/CuPy kernel for O(1) memory training, making it suitable
    for deep networks where PyTorch autograd would consume O(N) memory.

    Example:
        >>> # O(1) memory model - ideal for deep networks
        >>> model = MemoryEfficientLoopedMLP(784, 256, 10)
        >>> # Automatically uses kernel backend for O(1) memory training
        >>> print(model.backend)  # 'kernel' if CUDA/CuPy available, else 'pytorch'
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        use_spectral_norm: bool = True,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_gpu_if_available: bool = True,
    ) -> None:
        """
        Initialize memory-efficient model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            use_spectral_norm: Whether to use spectral normalization
            max_steps: Maximum equilibrium steps
            gradient_method: Gradient computation method
            use_gpu_if_available: Whether to use GPU if available (requires CuPy)
        """
        # Determine backend based on availability
        if use_gpu_if_available and HAS_CUPY and torch.cuda.is_available():
            backend = "kernel"
        else:
            backend = (
                "pytorch" if HAS_CUPY else "pytorch"
            )  # Fallback to pytorch if CuPy not available

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
            max_steps=max_steps,
            gradient_method=gradient_method,
            backend=backend,
        )

        # Store the intended memory efficiency
        self.is_memory_efficient = self.backend == "kernel"

    def __repr__(self) -> str:
        backend_str = f", backend={self.backend}"
        efficiency_str = (
            ", O(1) memory" if self.is_memory_efficient else ", O(N) memory"
        )
        return (
            f"MemoryEfficientLoopedMLP(input={self.input_dim}, hidden={self.hidden_dim}, "
            f"output={self.output_dim}, steps={self.max_steps}, "
            f"spectral_norm={self.use_spectral_norm}{backend_str}{efficiency_str})"
        )


class MemoryEfficientEqPropModel(EqPropModel):
    """
    Base class for memory-efficient EqProp models that can leverage kernel backend.

    This class provides the foundation for building models that can switch between
    PyTorch autograd (O(N) memory) and kernel-based (O(1) memory) implementations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        gradient_method: str = "bptt",
        use_spectral_norm: bool = True,
        memory_efficient: bool = True,
        use_gpu: bool = True,
    ):
        """
        Initialize memory-efficient EqProp model.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            max_steps: Maximum equilibrium steps
            gradient_method: Gradient computation method
            use_spectral_norm: Whether to use spectral normalization
            memory_efficient: Whether to use O(1) memory kernel backend when possible
            use_gpu: Whether to use GPU for kernel backend
        """
        self.memory_efficient = memory_efficient
        self.use_gpu = use_gpu and HAS_CUPY and torch.cuda.is_available()

        # Set backend based on memory efficiency preference
        if memory_efficient and HAS_CUPY and self.use_gpu:
            self.backend = "kernel"
        else:
            self.backend = "pytorch"

        super().__init__(
            max_steps=max_steps,
            gradient_method=gradient_method,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            use_spectral_norm=use_spectral_norm,
        )

        # Initialize kernel engine if using memory-efficient backend
        if self.backend == "kernel":
            from ..kernel import EqPropKernel

            self._engine = EqPropKernel(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                max_steps=max_steps,
                use_spectral_norm=use_spectral_norm,
                use_gpu=self.use_gpu,
            )
        else:
            self._engine = None

    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Optional[Dict[str, float]]:
        """
        Perform a training step using the appropriate backend.

        Returns:
            Training metrics if using kernel backend, None otherwise (delegates to parent)
        """
        if self.backend == "kernel" and self._engine is not None:
            # Convert inputs to numpy/cupy for kernel
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            if isinstance(y, torch.Tensor):
                y_np = y.detach().cpu().numpy()
            else:
                y_np = y

            # Run kernel training step
            metrics = self._engine.train_step(x_np, y_np)
            return metrics

        # Delegate to parent implementation for PyTorch backend
        return super().train_step(x, y)

    def forward(self, x: torch.Tensor, steps: Optional[int] = None, **kwargs):
        """
        Forward pass using the appropriate backend.
        """
        if self.backend == "kernel" and self._engine is not None:
            # Use kernel for inference
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = x

            h_star, _, _ = self._engine.solve_equilibrium(x_np)
            logits_np = self._engine.compute_output(h_star)

            # Convert back to tensor on same device as input
            return torch.from_numpy(logits_np).to(x.device)

        # Use PyTorch implementation
        return super().forward(x, steps, **kwargs)


# Factory function for easy creation of memory-efficient models
def create_memory_efficient_model(
    model_type: str, input_dim: int, hidden_dim: int, output_dim: int, **kwargs
) -> Any:
    """
    Factory function to create memory-efficient models.

    Args:
        model_type: Type of model to create
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        **kwargs: Additional arguments

    Returns:
        Memory-efficient model instance
    """
    if model_type.lower() in ["loopedmlp", "memory_efficient", "o1_memory"]:
        return MemoryEfficientLoopedMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
        )
    else:
        raise ValueError(f"Unsupported memory-efficient model type: {model_type}")


__all__ = [
    "MemoryEfficientLoopedMLP",
    "MemoryEfficientEqPropModel",
    "create_memory_efficient_model",
]
