#!/usr/bin/env python3
"""
ASI-Evolve command-line entry point.

Usage:
    python main.py
    python main.py --experiment my_exp --steps 20
    python main.py --config path/to/config.yaml
"""

import argparse
import importlib.util
import sys
from pathlib import Path


def _bootstrap_package() -> None:
    """Register the repository root as the importable `asi_evolve` package."""
    project_root = Path(__file__).resolve().parent
    package_name = "asi_evolve"

    if package_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        package_name,
        project_root / "__init__.py",
        submodule_search_locations=[str(project_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to bootstrap the 'asi_evolve' package")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


_bootstrap_package()

from asi_evolve.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Evolve Framework - Automated Experiment Evolution"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: config.yaml)",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of evolution steps (default: 10)",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=3,
        help="Number of nodes to sample per step (default: 3)",
    )

    parser.add_argument(
        "--eval-script",
        type=str,
        default=None,
        help="Path to evaluation script",
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        config_path=args.config,
        experiment_name=args.experiment,
    )

    pipeline.run(
        max_steps=args.steps,
        eval_script=args.eval_script,
        sample_n=args.sample_n,
    )

    stats = pipeline.get_stats()
    print("\n=== Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    best = pipeline.get_best_node()
    if best:
        print("\n=== Best Node ===")
        print(f"Name: {best.name}")
        print(f"Score: {best.score:.4f}")
        print(f"Motivation: {best.motivation[:200]}...")


if __name__ == "__main__":
    main()
