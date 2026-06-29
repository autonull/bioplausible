# EquiTile Rigorous Benchmarks

This directory contains scripts for rigorously benchmarking the EquiTile model against baselines like NanoGPT.

## Overview

The benchmarking suite performs:
- **Head-to-head comparison**: EquiTile vs NanoGPT with matched parameter counts.
- **Statistical analysis**: Multiple runs to ensure statistical significance.
- **Metric tracking**: Throughput (tokens/sec), Perplexity, Memory Usage, and Training Time.

## Running Benchmarks

Use the provided shell script to run the benchmarks. This script sets up the necessary `PYTHONPATH`.

```bash
./run_benchmarks.sh [options]
```

### Arguments

- `--num-runs INT`: Number of runs for statistical significance (default: 5).
- `--seed INT`: Random seed for reproducibility (default: 42).
- `--epochs INT`: Number of training epochs (default: 3).
- `--batch-size INT`: Batch size (default: 32).
- `--seq-length INT`: Sequence length (default: 128).
- `--device STR`: Device to use ('auto', 'cuda', 'cpu') (default: 'auto').

### Example

Run a quick test on CPU:
```bash
./run_benchmarks.sh --num-runs 1 --epochs 1 --batch-size 4 --seq-length 32 --device cpu
```

Run a full benchmark on GPU:
```bash
./run_benchmarks.sh --num-runs 5 --epochs 3 --batch-size 64 --seq-length 256 --device cuda
```

## Interpreting Results

The script outputs a comprehensive report including:
- **Speedup**: How much faster EquiTile is compared to the baseline.
- **Throughput**: Tokens processed per second.
- **Perplexity (PPL)**: A measure of language model quality (lower is better).
- **Statistical Significance**: Whether the observed differences are statistically significant (p-value < 0.05).

The results are also saved as JSON files in the `benchmark_results` directory.
