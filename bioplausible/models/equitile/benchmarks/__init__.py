"""
EquiTile LM Benchmarks
======================

Performance benchmarks and comparisons:
- NanoGPT comparison (head-to-head)
- Parameter efficiency analysis
- FLOP efficiency analysis
- Throughput benchmarking

Usage
-----
>>> from bioplausible.models.equitile.benchmarks import compare_nanoGPT
>>> results = compare_nanoGPT(task="shakespeare", epochs=5)
"""

from .compare_nanoGPT import (
    NanoGPTModel,
    NanoGPTConfig,
    compare_nanoGPT,
    run_benchmark_comparison,
)

from .efficiency_analysis import (
    EfficiencyAnalyzer,
    ParameterEfficiencyResult,
    FLOPEfficiencyResult,
    analyze_parameter_efficiency,
    analyze_flop_efficiency,
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
]
