"""
EquiTile LM Benchmarks
======================

Performance benchmarks and comparisons:
- NanoGPT comparison (head-to-head)
- Parameter efficiency analysis
- FLOP efficiency analysis
- Rigorous statistical benchmarking

Usage
-----
>>> from bioplausible.equitile.benchmarks import compare_nanoGPT, run_rigorous_benchmark
>>> results = compare_nanoGPT(task="shakespeare", epochs=5)
>>> rigorous_results = run_rigorous_benchmark(num_runs=5)
"""

from .compare_nanoGPT import (
    NanoGPTConfig,
    NanoGPTModel,
    compare_nanoGPT,
    run_benchmark_comparison,
)
from .efficiency_analysis import (
    EfficiencyAnalyzer,
    FLOPEfficiencyResult,
    ParameterEfficiencyResult,
    analyze_flop_efficiency,
    analyze_parameter_efficiency,
)
from .rigorous import (
    BenchmarkConfig,
    BenchmarkResult,
    RigorousBenchmark,
    StatisticalMetrics,
    get_system_info,
    run_rigorous_benchmark,
    set_all_seeds,
)

__all__ = [
    # NanoGPT comparison
    "NanoGPTModel",
    "NanoGPTConfig",
    "compare_nanoGPT",
    "run_benchmark_comparison",
    # Efficiency analysis
    "EfficiencyAnalyzer",
    "ParameterEfficiencyResult",
    "FLOPEfficiencyResult",
    "analyze_parameter_efficiency",
    "analyze_flop_efficiency",
    # Rigorous benchmarking
    "RigorousBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "StatisticalMetrics",
    "run_rigorous_benchmark",
    "set_all_seeds",
    "get_system_info",
]
