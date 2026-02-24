# EquiTile Benchmarks

This directory contains rigorous benchmarks for evaluating the performance and quality of EquiTile Language Models.

## Overview

EquiTile is designed for high-performance, sparse, and conditional computation. These benchmarks validate these claims against standard baselines (like NanoGPT) using rigorous statistical methods.

### Key Scripts

- `rigorous.py`: The main entry point for scientific-grade benchmarking.
- `compare_nanoGPT.py`: Implementation of the NanoGPT baseline for direct comparison.
- `efficiency_analysis.py`: Tools for analyzing FLOPs and parameter efficiency.

## Running Benchmarks

### Quick Start

To run the default rigorous benchmark comparing EquiTile to NanoGPT:

```bash
python -m bioplausible.models.equitile.benchmarks.rigorous
```

### Custom Configuration

You can customize the benchmark parameters using command-line arguments:

```bash
python -m bioplausible.models.equitile.benchmarks.rigorous \
    --num-runs 10 \
    --epochs 5 \
    --batch-size 64 \
    --seq-length 256 \
    --device cuda
```

Arguments:
- `--num-runs`: Number of independent runs for statistical significance (default: 5).
- `--seed`: Random seed for reproducibility (default: 42).
- `--epochs`: Number of training epochs per run (default: 3).
- `--batch-size`: Batch size (default: 32).
- `--seq-length`: Sequence length (default: 128).
- `--device`: Device to use (`auto`, `cuda`, `cpu`) (default: `auto`).

## Interpreting Results

The benchmark outputs a comprehensive report including:

1.  **Throughput Results**: Tokens/sec for both models with 95% Confidence Intervals (CI).
2.  **Speedup Analysis**: The speedup ratio with uncertainty bounds, t-statistic, and p-value for statistical significance.
3.  **Quality Metrics**: Validation perplexity (PPL) comparison.
4.  **Memory Efficiency**: Peak memory usage comparison.

Example Output:
```
RIGOROUS BENCHMARK REPORT
======================================================================
...
THROUGHPUT RESULTS
----------------------------------------------------------------------
NanoGPT:  12,500 ± 150 tok/s
          95% CI: [12,350, 12,650]
EquiTile: 18,750 ± 200 tok/s
          95% CI: [18,550, 18,950]

SPEEDUP ANALYSIS
----------------------------------------------------------------------
Speedup: 1.50x (95% CI: [1.48x, 1.52x])
Statistically significant: Yes (p < 0.001)
...
```

## Model Details

- **EquiTile Model**: Uses `bioplausible.models.equitile.lm_demo.fast_lm.FastLMEquiTile`. This is the high-performance implementation with Mixture of Tiles (MoT) and Flash Attention.
- **Baseline**: Uses `NanoGPTModel`, a faithful implementation of Karpathy's nanoGPT.

## Directory Structure

```
benchmarks/
├── rigorous.py           # Main benchmark runner
├── compare_nanoGPT.py    # Baseline implementation
├── efficiency_analysis.py # FLOPs/Params analysis tools
└── benchmark_results/    # Output directory for JSON results and reports
```
