#!/bin/bash

# EquiTile Benchmark Runner
# Sets up environment and runs rigorous benchmarks

# Determine the repository root (4 levels up from this script location)
# Script is at: bioplausible/models/equitile/benchmarks/run_benchmarks.sh
REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"

# Set PYTHONPATH to include the repository root
export PYTHONPATH="$REPO_ROOT"

echo "Running EquiTile Benchmarks..."
echo "Using PYTHONPATH=$PYTHONPATH"

# Run the benchmark script
python "$REPO_ROOT/bioplausible/models/equitile/benchmarks/rigorous.py" "$@"
