"""
EquiTile Deployment: Model Export and Optimization
===================================================

Tools for deploying EquiTile models:
- ONNX export for cross-platform inference
- Quantization for reduced memory and faster inference
- Model pruning for efficiency
- TorchScript/Torch.compile compilation

Examples
--------
>>> from bioplausible.equitile.deployment import EquiTileExporter
>>> exporter = EquiTileExporter(model)
>>> exporter.to_onnx("model.onnx", input_shape=(1, 784))
>>> exporter.quantize_dynamic()
>>> exporter.to_torchscript("model.pt")
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from .core import EquiTile


@dataclass
class ExportConfig:
    """Configuration for model export.

    Attributes
    ----------
    opset_version : int
        ONNX opset version
    do_constant_folding : bool
        Enable constant folding
    input_names : list of str
        Input tensor names
    output_names : list of str
        Output tensor names
    dynamic_axes : dict
        Dynamic axis configuration
    """

    opset_version: int = 14
    do_constant_folding: bool = True
    input_names: list[str] = None
    output_names: list[str] = None
    dynamic_axes: dict[str, dict[int, str]] = None

    def __post_init__(self) -> None:
        """Set defaults."""
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]
        if self.dynamic_axes is None:
            self.dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}


class EquiTileExporter:
    """Exporter for EquiTile models.

    Parameters
    ----------
    model : EquiTile
        Model to export
    config : ExportConfig, optional
        Export configuration
    """

    def __init__(
        self,
        model: EquiTile,
        config: ExportConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or ExportConfig()
        self.model.eval()

    def to_onnx(
        self,
        path: str,
        input_shape: tuple[int, ...],
        device: str = "cpu",
    ) -> str:
        """Export to ONNX format.

        Parameters
        ----------
        path : str
            Output path
        input_shape : tuple
            Input tensor shape
        device : str
            Device for export

        Returns
        -------
        str
            Path to exported model
        """
        self.model.to(device)
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)

        # Ensure path has .onnx extension
        path = str(path)
        if not path.endswith(".onnx"):
            path += ".onnx"

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            path,
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=self.config.dynamic_axes,
        )

        print(f"Model exported to {path}")
        return path

    def to_torchscript(
        self,
        path: str,
        input_shape: tuple[int, ...],
        method: Literal["trace", "script"] = "trace",
        device: str = "cpu",
    ) -> str:
        """Export to TorchScript format.

        Parameters
        ----------
        path : str
            Output path
        input_shape : tuple
            Input tensor shape
        method : str
            Export method: 'trace', 'script', or 'compile'
        device : str
            Device for export

        Returns
        -------
        str
            Path to exported model

        Notes
        -----
        - 'trace': Uses torch.jit.trace, good for fixed computation graphs
        - 'script': Uses torch.jit.script (deprecated in Python 3.14+)
        - 'compile': Uses torch.compile (recommended for Python 3.14+)
        """
        self.model.to(device)
        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)

        # Ensure path has .pt extension
        path = str(path)
        if not path.endswith(".pt"):
            path += ".pt"

        # Export
        if method == "trace":
            scripted_model = torch.jit.trace(self.model, dummy_input)
        elif method == "compile":
            # torch.compile returns an optimized module, save state dict instead
            compiled_model = torch.compile(self.model, mode="reduce-overhead")
            # Run once to trigger compilation
            _ = compiled_model(dummy_input)
            # Save state dict for compiled model
            torch.save(
                {
                    "model_state_dict": compiled_model.state_dict(),
                    "config": (
                        self.model.config if hasattr(self.model, "config") else None
                    ),
                    "compiled": True,
                },
                path,
            )
            print(f"Compiled model saved to {path}")
            return path
        else:
            # script method - use torch.jit.script with deprecation warning
            import warnings

            warnings.warn(
                "torch.jit.script is deprecated in Python 3.14+. "
                "Use method='compile' to use torch.compile instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            scripted_model = torch.jit.script(self.model)

        scripted_model.save(path)
        print(f"Model exported to {path}")
        return path

    def quantize_dynamic(
        self,
        dtype: Literal["qint8", "quint8", "qint32"] = "qint8",
    ) -> EquiTile:
        """Apply dynamic quantization.

        Parameters
        ----------
        dtype : str
            Quantization dtype

        Returns
        -------
        EquiTile
            Quantized model
        """
        # Quantize linear layers
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=getattr(torch, dtype),
        )

        print(f"Model quantized to {dtype}")
        return quantized_model

    def get_model_size(self, path: str | None = None) -> int:
        """Get model size in bytes.

        Parameters
        ----------
        path : str, optional
            Path to saved model

        Returns
        -------
        int
            Model size in bytes
        """
        if path is not None:
            return pathlib.Path(path).stat().st_size

        # Estimate from parameters
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return param_size + buffer_size

    def get_flops(
        self,
        input_shape: tuple[int, ...],
        device: str = "cpu",
    ) -> int:
        """Estimate FLOPs (requires torchinfo).

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape
        device : str
            Device

        Returns
        -------
        int
            Estimated FLOPs
        """
        try:
            from torchinfo import summary

            info = summary(self.model, input_size=input_shape, device=device, verbose=0)
            return info.total_mult_adds
        except ImportError:
            print("torchinfo not installed. Install with: pip install torchinfo")
            return -1

    def profile_memory(
        self,
        input_shape: tuple[int, ...],
        device: str = "cpu",
    ) -> dict[str, float]:
        """Profile memory usage.

        Parameters
        ----------
        input_shape : tuple
            Input tensor shape
        device : str
            Device

        Returns
        -------
        dict
            Memory statistics
        """
        self.model.to(device)

        # Create input
        dummy_input = torch.randn(input_shape, device=device)

        # Profile
        with torch.autograd.profiler.profile(use_cuda=(device == "cuda")) as prof:
            self.model(dummy_input)

        # Get memory stats
        if device == "cuda":
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**2
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**2
        else:
            memory_allocated = 0
            memory_reserved = 0

        return {
            "memory_allocated_mb": memory_allocated,
            "memory_reserved_mb": memory_reserved,
            "events": len(prof.key_averages()),
        }


class ModelPruner:
    """Pruner for EquiTile models.

    Parameters
    ----------
    model : EquiTile
        Model to prune
    """

    def __init__(self, model: EquiTile) -> None:
        self.model = model
        self.pruned_weights: dict[str, torch.Tensor] = {}

    def prune_by_magnitude(
        self,
        threshold: float = 0.01,
        layer_type: str = "linear",
    ) -> int:
        """Prune weights by magnitude.

        Parameters
        ----------
        threshold : float
            Pruning threshold
        layer_type : str
            Layer type to prune

        Returns
        -------
        int
            Number of pruned weights
        """
        pruned_count = 0

        for name, module in self.model.named_modules():
            if layer_type == "linear" and isinstance(module, nn.Linear):
                if hasattr(module, "weight") and module.weight is not None:
                    mask = torch.abs(module.weight) > threshold
                    pruned = (~mask).sum().item()
                    pruned_count += pruned

                    # Store pruned weights
                    self.pruned_weights[name] = module.weight.data.clone()

                    # Apply pruning
                    module.weight.data = module.weight.data * mask.float()

        print(f"Pruned {pruned_count} weights")
        return pruned_count

    def prune_by_importance(
        self,
        fraction: float = 0.1,
    ) -> int:
        """Prune weights by importance (tile importance).

        Parameters
        ----------
        fraction : float
            Fraction of weights to prune

        Returns
        -------
        int
            Number of pruned weights
        """
        if not hasattr(self.model, "tile_importance"):
            print("Model has no tile_importance attribute")
            return 0

        # Get importance scores
        importance = torch.sigmoid(self.model.tile_importance)

        # Find threshold
        threshold = torch.quantile(importance, fraction)

        # Prune low-importance tiles
        pruned_count = 0
        with torch.no_grad():
            for i, imp in enumerate(importance):
                if imp < threshold:
                    self.model.tile_importance[i] = 0.0
                    pruned_count += 1

        print(f"Pruned {pruned_count} tiles")
        return pruned_count

    def get_sparsity(self) -> dict[str, float]:
        """Get model sparsity.

        Returns
        -------
        dict
            Sparsity per layer
        """
        sparsity = {}

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "weight") and module.weight is not None:
                    total = module.weight.numel()
                    zeros = (module.weight == 0).sum().item()
                    sparsity[name] = zeros / total

        return sparsity


class DeploymentChecker:
    """Checker for deployment readiness.

    Parameters
    ----------
    model : EquiTile
        Model to check
    """

    def __init__(self, model: EquiTile) -> None:
        self.model = model
        self.model.eval()
        self.issues: list[str] = []

    def check(self) -> dict[str, Any]:
        """Run all checks.

        Returns
        -------
        dict
            Check results
        """
        self.issues.clear()

        results = {
            "ready": True,
            "issues": [],
            "recommendations": [],
        }

        # Check for training mode
        if self.model.training:
            self.issues.append("Model is in training mode")
            results["ready"] = False

        # Check for NaN/Inf weights
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                self.issues.append(f"NaN weights in {name}")
                results["ready"] = False
            if torch.isinf(param).any():
                self.issues.append(f"Inf weights in {name}")
                results["ready"] = False

        # Check model size
        param_count = sum(p.numel() for p in self.model.parameters())
        if param_count > 100_000_000:
            results["recommendations"].append(
                f"Large model ({param_count:,} params). Consider quantization."
            )

        results["issues"] = self.issues
        results["param_count"] = param_count

        return results

    def get_report(self) -> str:
        """Get deployment readiness report.

        Returns
        -------
        str
            Report string
        """
        results = self.check()

        report = []
        report.append("=" * 50)
        report.append("EquiTile Deployment Readiness Report")
        report.append("=" * 50)
        report.append("")

        if results["ready"]:
            report.append("✓ Model is ready for deployment")
        else:
            report.append("✗ Model is NOT ready for deployment")

        report.append("")
        report.append(f"Parameter count: {results.get('param_count', 0):,}")
        report.append("")

        if results["issues"]:
            report.append("Issues:")
            for issue in results["issues"]:
                report.append(f"  - {issue}")
            report.append("")

        if results["recommendations"]:
            report.append("Recommendations:")
            for rec in results["recommendations"]:
                report.append(f"  - {rec}")

        report.append("")
        report.append("=" * 50)

        return "\n".join(report)


# =============================================================================
# Factory Functions
# =============================================================================


def export_model(
    model: EquiTile,
    path: str,
    format: Literal["onnx", "torchscript"] = "onnx",
    input_shape: tuple[int, ...] = (1, 784),
) -> str:
    """Export model to specified format.

    Parameters
    ----------
    model : EquiTile
        Model to export
    path : str
        Output path
    format : str
        Export format
    input_shape : tuple
        Input shape

    Returns
    -------
    str
        Path to exported model
    """
    exporter = EquiTileExporter(model)

    if format == "onnx":
        return exporter.to_onnx(path, input_shape)
    elif format == "torchscript":
        return exporter.to_torchscript(path, input_shape)
    else:
        raise ValueError(f"Unknown format: {format}")


def quantize_model(
    model: EquiTile,
    dtype: str = "qint8",
) -> EquiTile:
    """Quantize model.

    Parameters
    ----------
    model : EquiTile
        Model to quantize
    dtype : str
        Quantization dtype

    Returns
    -------
    EquiTile
        Quantized model
    """
    exporter = EquiTileExporter(model)
    return exporter.quantize_dynamic(dtype)


def prune_model(
    model: EquiTile,
    threshold: float = 0.01,
) -> EquiTile:
    """Prune model.

    Parameters
    ----------
    model : EquiTile
        Model to prune
    threshold : float
        Pruning threshold

    Returns
    -------
    EquiTile
        Pruned model
    """
    pruner = ModelPruner(model)
    pruner.prune_by_magnitude(threshold)
    return model


def check_deployment(model: EquiTile) -> str:
    """Check deployment readiness.

    Parameters
    ----------
    model : EquiTile
        Model to check

    Returns
    -------
    str
        Deployment report
    """
    checker = DeploymentChecker(model)
    return checker.get_report()
