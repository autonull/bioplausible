#!/bin/bash
# Run benchmarks for EquiTile

# Set PYTHONPATH to include the project root
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "========================================================"
echo "Running Rigorous Benchmark (Fast Run for Verification)"
echo "========================================================"
# Using small parameters for quick verification
python3 bioplausible/models/equitile/benchmarks/rigorous.py --num-runs 1 --epochs 1 --batch-size 8 --seq-length 64

echo ""
echo "========================================================"
echo "Running Mixture of Tiles (MoT) Benchmark"
echo "========================================================"
python3 bioplausible/models/equitile/benchmarks/mot_benchmark.py
