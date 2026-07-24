#!/usr/bin/env python3
"""
EquiTile Comprehensive Benchmark Suite

Benchmarks:
- Multi-GPU scaling (NCCL)
- Mixed precision performance
- Tile dynamics overhead
- Enhanced EP convergence
- Async execution efficiency

Usage:
    python benchmarks/benchmark_equitile_comprehensive.py
"""

import json
import pathlib
import time

import torch
from bioplausible.models import (
    DynamicEquiTile,
    EnhancedEPConfig,
    EnhancedEquiTile,
    EquiTile,
    MultiGPUConfig,
    MultiGPUEquiTile,
    TileGrowthConfig,
)


def create_dataset(n_samples=1000, input_dim=64, output_dim=10):
    """Create classification dataset."""
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    for i in range(output_dim):
        mask = y == i
        X[mask] += i * 1.5
    return X, y


class BenchmarkResult:
    """Stores benchmark results."""

    def __init__(self, name: str):
        self.name = name
        self.metrics: dict[str, float] = {}
        self.config: dict = {}
        self.timestamp = time.time()

    def add_metric(self, key: str, value: float, unit: str = ""):
        self.metrics[key] = {"value": value, "unit": unit}

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metrics": self.metrics,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        lines = [f"\n{self.name}"]
        for key, data in self.metrics.items():
            unit = f" {data['unit']}" if data["unit"] else ""
            lines.append(f"  {key}: {data['value']:.4f}{unit}")
        return "\n".join(lines)


def benchmark_multigpu_scaling() -> list[BenchmarkResult]:
    """Benchmark multi-GPU scaling."""
    print("\n" + "=" * 70)
    print("Benchmark: Multi-GPU Scaling")
    print("=" * 70)

    results = []

    if not torch.cuda.is_available():
        print("  CUDA not available. Skipping multi-GPU benchmark.")
        return results

    n_gpus = torch.cuda.device_count()
    print(f"  Available GPUs: {n_gpus}")

    for n_devices in [1, min(2, n_gpus), min(4, n_gpus)]:
        if n_devices > n_gpus:
            continue

        model = EquiTile(
            neurons_per_tile=64,
            num_layers=4,
            tiles_per_layer=4,
            input_dim=64,
            output_dim=10,
        )

        multi_gpu = MultiGPUEquiTile(
            model,
            config=MultiGPUConfig(
                device_ids=list(range(n_devices)),
                async_execution=True,
            ),
        )

        X, y = create_dataset(n_samples=500, input_dim=64, output_dim=10)

        # Warmup
        multi_gpu.train_step(X[:64], y[:64])

        # Benchmark
        start = time.perf_counter()
        n_steps = 10
        for _ in range(n_steps):
            stats = multi_gpu.train_step(X[:64], y[:64])
        elapsed = time.perf_counter() - start

        result = BenchmarkResult(f"Multi-GPU ({n_devices} devices)")
        result.config["n_devices"] = n_devices
        result.add_metric("Time per step", elapsed / n_steps, "s")
        result.add_metric("Throughput", 64 * n_steps / elapsed, "samples/s")
        result.add_metric("Comm time", stats.get("comm_time", 0), "s")
        result.add_metric("Compute time", stats.get("compute_time", 0), "s")

        if n_devices == 1:
            elapsed_1gpu = elapsed

        results.append(result)
        print(result)

        if n_devices > 1:
            result.add_metric(
                "Speedup",
                elapsed_1gpu / elapsed,
                "x",
            )

        multi_gpu.destroy()

    return results


def benchmark_mixed_precision() -> list[BenchmarkResult]:
    """Benchmark mixed precision performance."""
    print("\n" + "=" * 70)
    print("Benchmark: Mixed Precision")
    print("=" * 70)

    results = []

    if not torch.cuda.is_available():
        print("  CUDA not available. Skipping mixed precision benchmark.")
        return results

    model_fp32 = EquiTile(
        neurons_per_tile=64,
        num_layers=4,
        tiles_per_layer=4,
        input_dim=64,
        output_dim=10,
    )

    X, y = create_dataset(n_samples=500, input_dim=64, output_dim=10)

    # FP32 baseline
    torch.cuda.empty_cache()
    start = time.perf_counter()
    n_steps = 10
    for _ in range(n_steps):
        with torch.cuda.amp.autocast(enabled=False):
            model_fp32.train_step(X[:64], y[:64])
    elapsed_fp32 = time.perf_counter() - start

    result_fp32 = BenchmarkResult("Mixed Precision (FP32)")
    result_fp32.add_metric("Time per step", elapsed_fp32 / n_steps, "s")
    result_fp32.add_metric("Throughput", 64 * n_steps / elapsed_fp32, "samples/s")
    results.append(result_fp32)
    print(result_fp32)

    # FP16
    model_fp16 = EquiTile(
        neurons_per_tile=64,
        num_layers=4,
        tiles_per_layer=4,
        input_dim=64,
        output_dim=10,
    )

    torch.cuda.empty_cache()
    start = time.perf_counter()
    for _ in range(n_steps):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            model_fp16.train_step(X[:64], y[:64])
    elapsed_fp16 = time.perf_counter() - start

    result_fp16 = BenchmarkResult("Mixed Precision (FP16)")
    result_fp16.add_metric("Time per step", elapsed_fp16 / n_steps, "s")
    result_fp16.add_metric("Throughput", 64 * n_steps / elapsed_fp16, "samples/s")
    result_fp16.add_metric("Speedup", elapsed_fp32 / elapsed_fp16, "x")
    results.append(result_fp16)
    print(result_fp16)

    return results


def benchmark_tile_dynamics() -> list[BenchmarkResult]:
    """Benchmark tile dynamics overhead."""
    print("\n" + "=" * 70)
    print("Benchmark: Tile Dynamics")
    print("=" * 70)

    results = []

    # Without dynamics
    model_static = EquiTile(
        neurons_per_tile=32,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
    )

    X, y = create_dataset(n_samples=500, input_dim=32, output_dim=4)

    start = time.perf_counter()
    n_steps = 20
    for _ in range(n_steps):
        model_static.train_step(X[:32], y[:32])
    elapsed_static = time.perf_counter() - start

    result_static = BenchmarkResult("Tile Dynamics (Static)")
    result_static.add_metric("Time per step", elapsed_static / n_steps, "s")
    result_static.add_metric("Final tiles", len(model_static.graph.tiles), "")
    results.append(result_static)
    print(result_static)

    # With dynamics
    from bioplausible.models import DynamicEquiTileConfig

    model_dynamic = EquiTile(
        neurons_per_tile=32,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
    )

    dynamic = DynamicEquiTile(
        model_dynamic,
        config=DynamicEquiTileConfig(
            growth=TileGrowthConfig(
                growth_enabled=True,
                prune_enabled=True,
                growth_threshold=0.3,
                prune_threshold=0.1,
                growth_cooldown=5,
                prune_cooldown=5,
            )
        ),
    )

    start = time.perf_counter()
    n_modifications = 0
    for _ in range(n_steps):
        model_dynamic.train_step(X[:32], y[:32])
        mod_stats = dynamic.step()
        n_modifications += sum(mod_stats.values())
    elapsed_dynamic = time.perf_counter() - start

    result_dynamic = BenchmarkResult("Tile Dynamics (Dynamic)")
    result_dynamic.add_metric("Time per step", elapsed_dynamic / n_steps, "s")
    result_dynamic.add_metric(
        "Overhead", (elapsed_dynamic - elapsed_static) / elapsed_static * 100, "%"
    )
    result_dynamic.add_metric("Final tiles", len(model_dynamic.graph.tiles), "")
    result_dynamic.add_metric("Modifications", n_modifications, "")
    results.append(result_dynamic)
    print(result_dynamic)

    return results


def benchmark_enhanced_ep() -> list[BenchmarkResult]:
    """Benchmark enhanced EP convergence."""
    print("\n" + "=" * 70)
    print("Benchmark: Enhanced EP Convergence")
    print("=" * 70)

    results = []

    # Standard EP
    model_standard = EquiTile(
        neurons_per_tile=32,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=10,
    )

    X, y = create_dataset(n_samples=500, input_dim=32, output_dim=4)

    losses_standard = []
    for _ in range(20):
        stats = model_standard.train_step(X[:32], y[:32])
        losses_standard.append(stats["loss"])

    result_standard = BenchmarkResult("Enhanced EP (Standard)")
    result_standard.add_metric("Initial loss", losses_standard[0], "")
    result_standard.add_metric("Final loss", losses_standard[-1], "")
    result_standard.add_metric(
        "Improvement", losses_standard[0] - losses_standard[-1], ""
    )
    results.append(result_standard)
    print(result_standard)

    # Enhanced EP with LayerNorm

    model_enhanced_base = EquiTile(
        neurons_per_tile=32,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=32,
        output_dim=4,
        mode="ep",
        beta=0.1,
        inference_steps=10,
    )

    model_enhanced = EnhancedEquiTile(
        model_enhanced_base,
        config=EnhancedEPConfig(
            use_layer_norm=True,
            use_curriculum=True,
            curriculum_stages=3,
            init_scheme="xavier",
        ),
    )

    losses_enhanced = []
    for _ in range(20):
        model_enhanced.get_curriculum_weights(X[:32], y[:32])
        stats = model_enhanced.train_step(X[:32], y[:32])
        losses_enhanced.append(stats["loss"])
        model_enhanced.curriculum.step(stats["loss"])

    result_enhanced = BenchmarkResult("Enhanced EP (LayerNorm + Curriculum)")
    result_enhanced.add_metric("Initial loss", losses_enhanced[0], "")
    result_enhanced.add_metric("Final loss", losses_enhanced[-1], "")
    result_enhanced.add_metric(
        "Improvement", losses_enhanced[0] - losses_enhanced[-1], ""
    )
    result_enhanced.add_metric(
        "Convergence gain",
        (losses_standard[-1] - losses_enhanced[-1]) / losses_standard[-1] * 100,
        "%",
    )
    results.append(result_enhanced)
    print(result_enhanced)

    return results


def benchmark_async_execution() -> list[BenchmarkResult]:
    """Benchmark async execution efficiency."""
    print("\n" + "=" * 70)
    print("Benchmark: Async Execution")
    print("=" * 70)

    results = []

    from bioplausible.models import AsyncConfig, AsyncEquiTile

    model = EquiTile(
        neurons_per_tile=32,
        num_layers=4,
        tiles_per_layer=4,
        input_dim=32,
        output_dim=4,
    )

    X, y = create_dataset(n_samples=500, input_dim=32, output_dim=4)

    # Sync execution
    start = time.perf_counter()
    n_steps = 10
    for _ in range(n_steps):
        model.train_step(X[:64], y[:64])
    elapsed_sync = time.perf_counter() - start

    result_sync = BenchmarkResult("Async Execution (Sync)")
    result_sync.add_metric("Time per step", elapsed_sync / n_steps, "s")
    result_sync.add_metric("Throughput", 64 * n_steps / elapsed_sync, "samples/s")
    results.append(result_sync)
    print(result_sync)

    # Async execution
    async_model = AsyncEquiTile(
        model,
        config=AsyncConfig(
            n_workers=4,
            use_processes=False,
        ),
    )

    start = time.perf_counter()
    with async_model.async_context():
        for _ in range(n_steps):
            async_model.train_step(X[:64], y[:64])
    elapsed_async = time.perf_counter() - start

    result_async = BenchmarkResult("Async Execution (Async, 4 workers)")
    result_async.add_metric("Time per step", elapsed_async / n_steps, "s")
    result_async.add_metric("Throughput", 64 * n_steps / elapsed_async, "samples/s")
    result_async.add_metric("Speedup", elapsed_sync / elapsed_async, "x")
    results.append(result_async)
    print(result_async)

    return results


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 70)
    print("EquiTile Comprehensive Benchmark Suite")
    print("=" * 70)

    all_results = []

    all_results.extend(benchmark_multigpu_scaling())
    all_results.extend(benchmark_mixed_precision())
    all_results.extend(benchmark_tile_dynamics())
    all_results.extend(benchmark_enhanced_ep())
    all_results.extend(benchmark_async_execution())

    # Summary
    print("\n" + "=" * 70)
    print("Benchmark Summary")
    print("=" * 70)

    for result in all_results:
        print(result)

    # Save results
    results_dict = {
        "timestamp": time.time(),
        "benchmarks": [r.to_dict() for r in all_results],
    }

    output_path = "benchmark_results.json"
    with pathlib.Path(output_path).open("w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    run_all_benchmarks()
