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

from .compare_nanoGPT import NanoGPTConfig
from .compare_nanoGPT import NanoGPTModel
from .compare_nanoGPT import compare_nanoGPT
from .compare_nanoGPT import run_benchmark_comparison
from .efficiency_analysis import EfficiencyAnalyzer
from .efficiency_analysis import FLOPEfficiencyResult
from .efficiency_analysis import ParameterEfficiencyResult
from .efficiency_analysis import analyze_flop_efficiency
from .efficiency_analysis import analyze_parameter_efficiency
from .rigorous import BenchmarkConfig
from .rigorous import BenchmarkResult
from .rigorous import RigorousBenchmark
from .rigorous import StatisticalMetrics
from .rigorous import get_system_info
from .rigorous import run_rigorous_benchmark
from .rigorous import set_all_seeds

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
