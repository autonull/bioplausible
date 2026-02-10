"""
Leaderboard Insights Generation

Automatically detects patterns and generates actionable recommendations.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np


def generate_insights(
    trials: List[Dict[str, Any]], pareto_ids: List[int]
) -> List[Tuple[str, str]]:
    """
    Generate automatic insights from trial data.

    Args:
        trials: List of trial dictionaries
        pareto_ids: List of Pareto-optimal trial IDs

    Returns:
        List of (insight_text, insight_type) tuples
    """
    insights = []

    if not trials:
        return insights

    # 1. Best hyperparameter ranges
    hp_insights = _analyze_hyperparameters(trials)
    insights.extend(hp_insights)

    # 2. Model family performance
    family_insights = _analyze_model_families(trials)
    insights.extend(family_insights)

    # 3. Efficiency insights
    efficiency_insights = _analyze_efficiency(trials)
    insights.extend(efficiency_insights)

    # 4. Pareto insights
    pareto_insights = _analyze_pareto_frontier(trials, pareto_ids)
    insights.extend(pareto_insights)

    return insights


def _analyze_hyperparameters(trials: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Analyze hyperparam ranges that lead to high accuracy."""
    insights = []

    # Get top 25% trials
    sorted_trials = sorted(trials, key=lambda t: t["accuracy"], reverse=True)
    top_k = max(1, len(sorted_trials) // 4)
    top_trials = sorted_trials[:top_k]

    # Collect hyperparameter values from top trials
    hp_values = defaultdict(list)
    for trial in top_trials:
        if "config" in trial and trial["config"]:
            for key, value in trial["config"].items():
                if isinstance(value, (int, float)) and key != "epochs":
                    hp_values[key].append(value)

    # Analyze ranges
    for param, values in hp_values.items():
        if len(values) >= 3:
            mean_val = np.mean(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Format based on magnitude
            if param == "lr":
                insight = f"Top models use learning rates around {mean_val:.2e} (range: [{min_val:.2e}, {max_val:.2e}])"
            elif param in ["beta", "gamma", "alpha"]:
                insight = f"Best {param} values: {mean_val:.3f} ± {np.std(values):.3f}"
            elif param in ["steps", "num_layers"]:
                insight = f"Optimal {param}: {int(mean_val)} (range: {int(min_val)}-{int(max_val)})"
            elif param == "hidden_dim":
                insight = f"High-performing models use hidden_dim ≈ {int(mean_val)} (range: {int(min_val)}-{int(max_val)})"
            else:
                insight = f"{param} sweet spot: {mean_val:.4f}"

            insights.append((insight, "tip"))

    return insights


def _analyze_model_families(trials: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Analyze relative performance of different model families."""
    insights = []

    # Group by model
    model_performance = defaultdict(list)
    for trial in trials:
        model_performance[trial["model_name"]].append(trial["accuracy"])

    # Find best and worst
    model_means = {
        model: np.mean(accs) for model, accs in model_performance.items() if accs
    }

    if model_means:
        best_model = max(model_means, key=model_means.get)
        best_acc = model_means[best_model]

        insights.append(
            (
                f"{best_model} achieves highest average accuracy: {best_acc*100:.2f}%",
                "success",
            )
        )

        # Compare models
        if len(model_means) >= 2:
            sorted_models = sorted(
                model_means.items(), key=lambda x: x[1], reverse=True
            )
            gap = (sorted_models[0][1] - sorted_models[1][1]) * 100
            if gap > 2.0:
                insights.append(
                    (
                        f"{sorted_models[0][0]} outperforms {sorted_models[1][0]} by {gap:.1f} percentage points",
                        "info",
                    )
                )

    return insights


def _analyze_efficiency(trials: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Analyze parameter efficiency and speed."""
    insights = []

    # Find high accuracy trials
    high_acc_trials = [t for t in trials if t["accuracy"] >= 0.80]

    if high_acc_trials:
        # Find smallest model achieving high accuracy
        smallest = min(high_acc_trials, key=lambda t: t["param_count"])
        insights.append(
            (
                f"Smallest model achieving 80%+ accuracy: {smallest['model_name']} with {smallest['param_count']:.2f}M params ({smallest['accuracy']*100:.1f}%)",
                "success",
            )
        )

        # Find fastest
        fastest = min(high_acc_trials, key=lambda t: t["iteration_time"])
        insights.append(
            (
                f"Fastest model at 80%+ accuracy: {fastest['model_name']} at {fastest['iteration_time']:.4f}s per iteration",
                "success",
            )
        )

    # Parameter efficiency
    sorted_by_acc = sorted(trials, key=lambda t: t["accuracy"], reverse=True)
    top_5 = sorted_by_acc[: min(5, len(sorted_by_acc))]
    avg_params_top5 = np.mean([t["param_count"] for t in top_5])

    insights.append((f"Top 5 models average {avg_params_top5:.2f}M parameters", "info"))

    return insights


def _analyze_pareto_frontier(
    trials: List[Dict[str, Any]], pareto_ids: List[int]
) -> List[Tuple[str, str]]:
    """Analyze the Pareto frontier."""
    insights = []

    pareto_trials = [t for t in trials if t["trial_id"] in pareto_ids]

    if pareto_trials:
        insights.append(
            (
                f"{len(pareto_trials)} trials on Pareto frontier (optimal trade-offs)",
                "success",
            )
        )

        # Analyze trade-offs
        if len(pareto_trials) >= 2:
            # Find accuracy-focused vs efficiency-focused
            by_acc = max(pareto_trials, key=lambda t: t["accuracy"])
            by_params = min(pareto_trials, key=lambda t: t["param_count"])

            if by_acc["trial_id"] != by_params["trial_id"]:
                acc_gain = (by_acc["accuracy"] - by_params["accuracy"]) * 100
                param_cost = by_acc["param_count"] - by_params["param_count"]

                insights.append(
                    (
                        f"Pareto trade-off: +{acc_gain:.1f}% accuracy costs +{param_cost:.2f}M parameters",
                        "info",
                    )
                )
    else:
        insights.append(
            ("No clear Pareto frontier - all models have similar trade-offs", "warning")
        )

    return insights


def detect_anomalies(trials: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Detect unexpected or anomalous results."""
    anomalies = []

    # Detect trials with unusually low accuracy
    accuracies = [t["accuracy"] for t in trials]
    if accuracies:
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        for trial in trials:
            if trial["accuracy"] < mean_acc - 2 * std_acc:
                anomalies.append(
                    (
                        f"Trial #{trial['trial_id']} ({trial['model_name']}) significantly underperformed: {trial['accuracy']*100:.1f}%",
                        "warning",
                    )
                )

    return anomalies
