"""
Comprehensive Algorithm Evaluation

1-hour thorough comparison of all novel hybrid algorithms.
"""

import sys

sys.path.insert(0, "/home/me/eqprop")

import json
import time
from datetime import datetime
from typing import Dict, List

import torch
from algorithms import ALGORITHM_REGISTRY
from experiments.shallow_search import ShallowSearcher, load_mnist_subset


class ComprehensiveEvaluator:
    """Thorough multi-task evaluation of all algorithms."""

    def __init__(self, time_budget_minutes: float = 60):
        self.time_budget = time_budget_minutes * 60  # Convert to seconds
        self.results = {}

    def run_evaluation(self):
        """Run comprehensive evaluation."""
        start_time = time.time()

        algorithms = list(ALGORITHM_REGISTRY.keys())

        print("=" * 70)
        print("COMPREHENSIVE ALGORITHM EVALUATION")
        print("=" * 70)
        print(f"Time Budget: {self.time_budget/60:.1f} minutes")
        print(f"Algorithms: {len(algorithms)}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()

        # Calculate time per algorithm
        time_per_algo = self.time_budget / len(algorithms)

        print(f"Time per algorithm: {time_per_algo:.1f}s (~{time_per_algo/60:.1f} min)")
        print()

        # MNIST evaluation
        print("Loading MNIST (10K train samples)...")
        train_loader, test_loader = load_mnist_subset(n_samples=10000)

        # Different parameter budgets
        param_budgets = [50_000, 100_000, 200_000]

        all_results = {}

        for budget in param_budgets:
            print(f"\n{'='*70}")
            print(f"PARAMETER BUDGET: {budget:,}")
            print(f"{'='*70}\n")

            searcher = ShallowSearcher(
                algorithms=algorithms,
                param_budget=budget,
            )

            results = searcher.ultra_shallow_eval(
                train_loader=train_loader,
                test_loader=test_loader,
                input_dim=784,
                output_dim=10,
                time_budget=time_per_algo,
            )

            searcher.print_summary()

            all_results[f"mnist_{budget}"] = results

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results/algorithm_comparison_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

        # Generate summary report
        self._generate_report(all_results, output_file.replace(".json", ".md"))

        elapsed = time.time() - start_time
        print(f"\nTotal evaluation time: {elapsed/60:.1f} minutes")

    def _generate_report(self, results: Dict, output_path: str):
        """Generate markdown report."""
        lines = [
            "# Comprehensive Algorithm Comparison",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
        ]

        # Best by parameter budget
        for config, res in results.items():
            task, budget = config.split("_")
            successful = [
                (name, data["test_acc"])
                for name, data in res.items()
                if data.get("success", False)
            ]

            if successful:
                best = max(successful, key=lambda x: x[1])
                lines.extend(
                    [
                        f"### {task.upper()} @ {budget} parameters",
                        f"**Winner**: {best[0]} ({best[1]:.3f} accuracy)",
                        "",
                    ]
                )

        # Detailed results
        lines.extend(
            [
                "## Detailed Results",
                "",
            ]
        )

        for config, res in results.items():
            task, budget = config.split("_")
            lines.extend(
                [
                    f"### {task.upper()} - {budget} params",
                    "",
                    "| Rank | Algorithm | Test Acc | Train Acc | Epochs | Time (s) |",
                    "|------|-----------|----------|-----------|--------|----------|",
                ]
            )

            successful = [
                (name, data) for name, data in res.items() if data.get("success", False)
            ]
            successful.sort(key=lambda x: x[1]["test_acc"], reverse=True)

            for i, (name, data) in enumerate(successful, 1):
                lines.append(
                    f"| {i} | {name} | {data['test_acc']:.3f} | "
                    f"{data['train_acc']:.3f} | {data['epochs']} | {data['time']:.1f} |"
                )

            lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

        print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive algorithm evaluation")
    parser.add_argument("--hours", type=float, default=1.0, help="Time budget in hours")
    args = parser.parse_args()

    evaluator = ComprehensiveEvaluator(time_budget_minutes=args.hours * 60)
    evaluator.run_evaluation()
