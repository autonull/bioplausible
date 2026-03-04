"""
Memory Profiler for EquiTile LM
================================

Tools for profiling memory usage and bandwidth:
- MemoryProfiler: Track GPU memory allocation
- BandwidthAnalyzer: Measure memory bandwidth utilization
- OperationCounter: Count FLOPs and memory operations

Example
-------
>>> from bioplausible.models.equitile.lm_demo.profiling import MemoryProfiler
>>> profiler = MemoryProfiler()
>>> with profiler.track("forward"):
...     output = model(input_ids)
>>> profiler.report()
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    operation: str


@dataclass
class ProfileResult:
    """Profiling results."""

    operation: str
    duration_ms: float
    memory_allocated_mb: float
    memory_freed_mb: float
    peak_memory_mb: float
    tensors_created: int
    tensors_deleted: int


class MemoryProfiler:
    """GPU memory profiler.

    Tracks memory allocation during model operations.

    Parameters
    ----------
    device : str
        Device to profile (default: cuda)
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.snapshots: List[MemorySnapshot] = []
        self.results: List[ProfileResult] = []
        self._start_allocated = 0
        self._start_time = 0

    def reset(self) -> None:
        """Reset profiler state."""
        self.snapshots.clear()
        self.results.clear()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_memory_mb(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.device.type != "cuda":
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}

        return {
            "allocated": torch.cuda.memory_allocated(self.device) / 1024 / 1024,
            "reserved": torch.cuda.memory_reserved(self.device) / 1024 / 1024,
            "max_allocated": torch.cuda.max_memory_allocated(self.device) / 1024 / 1024,
        }

    def snapshot(self, operation: str = "") -> MemorySnapshot:
        """Take a memory snapshot."""
        mem = self.get_memory_mb()
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_mb=mem["allocated"],
            reserved_mb=mem["reserved"],
            max_allocated_mb=mem["max_allocated"],
            operation=operation,
        )
        self.snapshots.append(snapshot)
        return snapshot

    @contextmanager
    def track(self, operation: str):
        """Context manager for tracking an operation.

        Parameters
        ----------
        operation : str
            Operation name
        """
        # Sync and snapshot before
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        self.snapshot(f"{operation}_start")
        mem_before = self.get_memory_mb()
        start_time = time.time()
        tensors_before = self._count_tensors()

        try:
            yield
        finally:
            # Sync and snapshot after
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

            end_time = time.time()
            mem_after = self.get_memory_mb()
            tensors_after = self._count_tensors()

            self.snapshot(f"{operation}_end")

            result = ProfileResult(
                operation=operation,
                duration_ms=(end_time - start_time) * 1000,
                memory_allocated_mb=max(
                    0, mem_after["allocated"] - mem_before["allocated"]
                ),
                memory_freed_mb=max(
                    0, mem_before["allocated"] - mem_after["allocated"]
                ),
                peak_memory_mb=mem_after["max_allocated"],
                tensors_created=max(0, tensors_after - tensors_before),
                tensors_deleted=max(0, tensors_before - tensors_after),
            )
            self.results.append(result)

    def _count_tensors(self) -> int:
        """Count active tensors on device."""
        import gc

        count = 0
        try:
            for obj in gc.get_objects():
                try:
                    if (
                        isinstance(obj, torch.Tensor)
                        and obj.device.type == self.device.type
                    ):
                        count += 1
                except (ReferenceError, AttributeError):
                    # Object was garbage collected or has no device
                    pass
        except RuntimeError:
            # GC changed during iteration
            pass
        return count

    def report(self) -> str:
        """Generate profiling report."""
        if not self.results:
            return "No profiling data available"

        lines = ["=" * 60, "Memory Profiling Report", "=" * 60, ""]

        for result in self.results:
            lines.append(f"Operation: {result.operation}")
            lines.append(f"  Duration: {result.duration_ms:.2f} ms")
            lines.append(f"  Memory Allocated: {result.memory_allocated_mb:.2f} MB")
            lines.append(f"  Memory Freed: {result.memory_freed_mb:.2f} MB")
            lines.append(f"  Peak Memory: {result.peak_memory_mb:.2f} MB")
            lines.append(f"  Tensors Created: {result.tensors_created}")
            lines.append(f"  Tensors Deleted: {result.tensors_deleted}")
            lines.append("")

        # Summary
        total_time = sum(r.duration_ms for r in self.results)
        total_allocated = sum(r.memory_allocated_mb for r in self.results)
        peak_memory = max(r.peak_memory_mb for r in self.results)

        lines.append("-" * 60)
        lines.append("Summary")
        lines.append("-" * 60)
        lines.append(f"Total Time: {total_time:.2f} ms")
        lines.append(f"Total Memory Allocated: {total_allocated:.2f} MB")
        lines.append(f"Peak Memory: {peak_memory:.2f} MB")
        lines.append(f"Operations Profiled: {len(self.results)}")

        return "\n".join(lines)


class BandwidthAnalyzer:
    """Analyze memory bandwidth utilization.

    Measures actual vs theoretical bandwidth.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Theoretical bandwidth (GB/s) - approximate values
        self.theoretical_bandwidth = {
            "RTX 3080": 760,
            "RTX 3090": 936,
            "RTX 4070": 504,
            "RTX 4090": 1008,
            "A100": 1555,
            "H100": 2000,
        }

    def get_gpu_name(self) -> str:
        """Get GPU name."""
        if self.device.type != "cuda":
            return "CPU"
        return torch.cuda.get_device_name(self.device)

    def get_theoretical_bandwidth(self) -> float:
        """Get theoretical bandwidth in GB/s."""
        gpu_name = self.get_gpu_name()
        for name, bw in self.theoretical_bandwidth.items():
            if name in gpu_name:
                return bw
        # Default estimate based on memory info
        if self.device.type == "cuda":
            info = torch.cuda.get_device_properties(self.device)
            # Rough estimate: bandwidth ~ 4-6x memory clock
            return 500  # Conservative default
        return 0

    def measure_read_bandwidth(
        self,
        tensor_size_mb: int = 100,
        iterations: int = 10,
    ) -> float:
        """Measure memory read bandwidth.

        Parameters
        ----------
        tensor_size_mb : int
            Size of tensor in MB
        iterations : int
            Number of iterations

        Returns
        -------
        float
            Bandwidth in GB/s
        """
        if self.device.type != "cuda":
            return 0

        # Create tensor
        size_bytes = tensor_size_mb * 1024 * 1024
        elements = size_bytes // 4  # float32
        tensor = torch.randn(elements, device=self.device)

        # Warmup
        for _ in range(3):
            _ = tensor * 2

        # Measure
        torch.cuda.synchronize()
        start = time.time()

        total_bytes = 0
        for _ in range(iterations):
            result = tensor * 2  # Read tensor
            _ = result.sum()  # Force computation
            total_bytes += size_bytes * 2  # Read + write

        torch.cuda.synchronize()
        elapsed = time.time() - start

        bandwidth = (total_bytes / 1024 / 1024 / 1024) / elapsed
        return bandwidth

    def measure_write_bandwidth(
        self,
        tensor_size_mb: int = 100,
        iterations: int = 10,
    ) -> float:
        """Measure memory write bandwidth."""
        if self.device.type != "cuda":
            return 0

        size_bytes = tensor_size_mb * 1024 * 1024
        elements = size_bytes // 4

        # Warmup
        tensor = torch.empty(elements, device=self.device)

        # Measure
        torch.cuda.synchronize()
        start = time.time()

        total_bytes = 0
        for _ in range(iterations):
            tensor = torch.randn(elements, device=self.device)
            total_bytes += size_bytes

        torch.cuda.synchronize()
        elapsed = time.time() - start

        bandwidth = (total_bytes / 1024 / 1024 / 1024) / elapsed
        return bandwidth

    def analyze_model_bandwidth(
        self,
        model: torch.nn.Module,
        input_tensor: Tensor,
        iterations: int = 10,
    ) -> Dict[str, float]:
        """Analyze bandwidth for model forward/backward.

        Parameters
        ----------
        model : nn.Module
            Model to analyze
        input_tensor : Tensor
            Input tensor
        iterations : int
            Number of iterations

        Returns
        -------
        dict
            Bandwidth analysis results
        """
        if self.device.type != "cuda":
            return {}

        model.train()
        model = model.to(self.device)
        input_tensor = input_tensor.to(self.device)

        # Estimate memory traffic
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(p.numel() * p.element_size() for p in model.buffers())
        total_param_bytes = param_bytes + buffer_bytes

        # Input/output bytes
        input_bytes = input_tensor.numel() * input_tensor.element_size()

        # Warmup
        for _ in range(3):
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            model.zero_grad()

        # Measure forward
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            output = model(input_tensor)
            _ = output.sum()

        torch.cuda.synchronize()
        forward_time = time.time() - start

        # Measure forward + backward
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            output = model(input_tensor)
            loss = output.sum()
            loss.backward()
            model.zero_grad()

        torch.cuda.synchronize()
        total_time = time.time() - start
        backward_time = total_time - forward_time

        # Calculate bandwidth
        # Forward: read params + read input + write output
        forward_bytes = total_param_bytes + input_bytes * 2
        forward_bandwidth = (
            forward_bytes * iterations / 1024 / 1024 / 1024
        ) / forward_time

        # Backward: read params + read grad + write grad
        backward_bytes = total_param_bytes * 2
        backward_bandwidth = (
            backward_bytes * iterations / 1024 / 1024 / 1024
        ) / backward_time

        theoretical = self.get_theoretical_bandwidth()

        return {
            "gpu_name": self.get_gpu_name(),
            "theoretical_bandwidth_gb_s": theoretical,
            "forward_bandwidth_gb_s": forward_bandwidth,
            "backward_bandwidth_gb_s": backward_bandwidth,
            "forward_utilization": (
                forward_bandwidth / theoretical * 100 if theoretical > 0 else 0
            ),
            "backward_utilization": (
                backward_bandwidth / theoretical * 100 if theoretical > 0 else 0
            ),
            "param_memory_mb": total_param_bytes / 1024 / 1024,
            "forward_time_ms": forward_time / iterations * 1000,
            "backward_time_ms": backward_time / iterations * 1000,
        }

    def report(self, analysis: Dict[str, float]) -> str:
        """Generate bandwidth analysis report."""
        if not analysis:
            return "No bandwidth data available"

        lines = ["=" * 60, "Bandwidth Analysis Report", "=" * 60, ""]

        lines.append(f"GPU: {analysis.get('gpu_name', 'Unknown')}")
        lines.append(
            f"Theoretical Bandwidth: {analysis.get('theoretical_bandwidth_gb_s', 0):.0f} GB/s"
        )
        lines.append("")

        lines.append("Forward Pass:")
        lines.append(
            f"  Bandwidth: {analysis.get('forward_bandwidth_gb_s', 0):.0f} GB/s"
        )
        lines.append(f"  Utilization: {analysis.get('forward_utilization', 0):.1f}%")
        lines.append(f"  Time: {analysis.get('forward_time_ms', 0):.2f} ms")
        lines.append("")

        lines.append("Backward Pass:")
        lines.append(
            f"  Bandwidth: {analysis.get('backward_bandwidth_gb_s', 0):.0f} GB/s"
        )
        lines.append(f"  Utilization: {analysis.get('backward_utilization', 0):.1f}%")
        lines.append(f"  Time: {analysis.get('backward_time_ms', 0):.2f} ms")
        lines.append("")

        lines.append(f"Parameter Memory: {analysis.get('param_memory_mb', 0):.0f} MB")

        # Recommendations
        lines.append("")
        lines.append("-" * 60)
        lines.append("Recommendations")
        lines.append("-" * 60)

        fwd_util = analysis.get("forward_utilization", 0)
        if fwd_util < 30:
            lines.append("- Low bandwidth utilization: Consider kernel fusion")
        elif fwd_util > 80:
            lines.append(
                "- High bandwidth utilization: Memory-bound, consider smaller batches"
            )

        bwd_util = analysis.get("backward_utilization", 0)
        if bwd_util < 30:
            lines.append("- Low backward bandwidth: Consider gradient checkpointing")

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def profile_memory(model: torch.nn.Module, input_tensor: Tensor) -> str:
    """Profile memory usage for a model forward/backward pass.

    Parameters
    ----------
    model : nn.Module
        Model to profile
    input_tensor : Tensor
        Input tensor

    Returns
    -------
    str
        Profiling report
    """
    profiler = MemoryProfiler()
    analyzer = BandwidthAnalyzer()

    # Profile memory
    with profiler.track("forward"):
        output = model(input_tensor)

    with profiler.track("backward"):
        loss = output.sum()
        loss.backward()

    # Analyze bandwidth
    analysis = analyzer.analyze_model_bandwidth(model, input_tensor)

    # Generate report
    report = profiler.report()
    report += "\n\n" + analyzer.report(analysis)

    return report
