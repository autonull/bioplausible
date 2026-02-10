#!/bin/bash
# Benchmark Runner Script
# Runs benchmark with GPU acceleration enabled by default

echo "🚀 Starting Bioplausible Benchmark"
echo "=================================="
echo ""
echo "Mode: Pareto multi-objective (default)"
echo "Seed: 42 (default)"
echo ""
echo "Options:"
echo "  --mode scalarized    : Use weighted single objective"
echo "  --seed-base N        : Set base random seed"
echo ""

# Remove old database
rm -f examples/shallow_benchmark.db

# Run benchmark
python3 examples/shallow_benchmark.py "$@"
