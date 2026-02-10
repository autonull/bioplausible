#!/usr/bin/env python3
"""
Run all consolidated experiments for TorEqProp.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_experiment(script_name, description=None):
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"❌ Script not found: {script_name}")
        return False

    print(f"\n{'='*60}")
    print(f"RUNNING: {description or script_name}")
    print(f"{'='*60}")

    try:
        # Run with current python executable
        subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"✅ {script_name} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}.")
        return False


def main():
    print("TorEqProp Consolidated Experiment Suite")
    print("=======================================")

    experiments = [
        ("sn_benchmark_datasets.py", "Spectral Normalization Multi-Dataset Benchmark"),
        ("sn_benchmark_model_size.py", "Spectral Normalization Model Size Benchmark"),
        ("sn_stress_test.py", "Spectral Normalization Stress Test"),
        ("track_a1_stability.py", "Architecture-Agnostic Stability Study"),
        ("cifar_breakthrough.py", "Track 34: CIFAR-10 Breakthrough"),
        ("memory_scaling_demo.py", "Track 35: Memory Scaling Demo"),
        ("energy_confidence.py", "Track 36: Energy-Based OOD Detection"),
        ("language_modeling.py", "Track 37: Language Modeling"),
        ("adaptive_compute.py", "Track 38: Adaptive Compute"),
        ("diffusion_mnist.py", "Track 39: Diffusion Models"),
        ("flop_analysis.py", "Track 40: FLOP Analysis"),
    ]

    success_count = 0

    for script, desc in experiments:
        if run_experiment(script, desc):
            success_count += 1

    print(
        f"\n\nSuite Complete: {success_count}/{len(experiments)} experiments ran successfully."
    )


if __name__ == "__main__":
    main()
