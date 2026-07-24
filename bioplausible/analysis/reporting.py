"""
Advanced Reporting for Bioplausible Experiments.

This module generates comprehensive scientific reports from experiment data,
analyzing performance across models and tasks.
"""

import pathlib
from collections import defaultdict
from datetime import datetime

import numpy as np

from bioplausible.analysis.results import load_trials


def generate_experiment_report(
    db_path: str, tier: str, output_path: str = "experiment_report.md"
) -> str:
    """
    Generate a comprehensive analysis report for a given tier.

    Args:
        db_path: Path to SQLite database
        tier: Tier name to filter by (e.g. "smoke")
        output_path: Path to save the markdown report

    Returns:
        Content of the generated report
    """
    trials = load_trials(db_path)

    # Filter by tier (assuming task/tier attrs available)
    # Note: load_trials might need to be robust to missing attrs
    filtered_trials = []

    tasks_seen = set()
    models_seen = set()

    for t in trials:
        t_tier = t.get("user_attrs", {}).get("tier", "shallow")
        if t_tier == tier:
            filtered_trials.append(t)
            tasks_seen.add(t.get("user_attrs", {}).get("task", "unknown"))
            models_seen.add(t["model_name"])

    if not filtered_trials:
        return f"No trials found for tier: {tier}"

    # --- Analysis ---

    # 1. Performance Matrix (Model x Task)
    matrix = defaultdict(lambda: defaultdict(list))
    for t in filtered_trials:
        task = t.get("user_attrs", {}).get("task", "unknown")
        model = t["model_name"]
        matrix[model][task].append(t["accuracy"])

    # Compute aggregates
    agg_matrix = {}
    for model, tasks in matrix.items():
        agg_matrix[model] = {}
        for task, accs in tasks.items():
            agg_matrix[model][task] = {
                "mean": np.mean(accs),
                "std": np.std(accs),
                "best": np.max(accs),
                "count": len(accs),
            }

    # 2. Backprop Competitiveness
    # Identify baseline performance per task
    baselines = {}
    for task in tasks_seen:
        # Find backprop for this task
        bp_accs = matrix.get("Backprop Baseline", {}).get(task, [])
        if not bp_accs:
            # Fallback alias search
            for m in matrix.keys():
                if "backprop" in m.lower():
                    bp_accs = matrix[m].get(task, [])
                    break

        if bp_accs:
            baselines[task] = np.mean(bp_accs)
        else:
            baselines[task] = 0.0

    # Report Content Generation
    lines = []
    lines.append(f"# Bioplausible Experiment Report: {tier.title()} Tier")
    lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Scope**: {len(models_seen)} Models x {len(tasks_seen)} Tasks")
    lines.append(f"**Total Trials**: {len(filtered_trials)}\n")

    lines.append("## 1. Executive Summary")

    # Find Top Performers
    best_model = None
    best_avg_acc = -1.0

    for model, tasks in agg_matrix.items():
        # Average across all tasks
        model_avgs = [stats["mean"] for stats in tasks.values()]
        avg_score = np.mean(model_avgs) if model_avgs else 0.0

        if avg_score > best_avg_acc:
            best_avg_acc = avg_score
            best_model = model

    lines.append(
        f"The top performing algorithm across all tasks was **{best_model}** "
        f"with an average accuracy of **{best_avg_acc * 100:.2f}%**."
    )

    lines.append("\n## 2. Performance Matrix (Accuracy)")

    # Header
    sorted_tasks = sorted(list(tasks_seen))
    lines.append(
        "| Model | " + " | ".join([t.upper() for t in sorted_tasks]) + " | Mean |"
    )
    lines.append(
        "| :--- | " + " | ".join([":---:" for _ in sorted_tasks]) + " | :---: |"
    )

    sorted_models = sorted(agg_matrix.keys())

    for model in sorted_models:
        row = [f"**{model}**"]
        model_accs = []
        for task in sorted_tasks:
            stats = agg_matrix[model].get(task)
            if stats:
                val = stats["mean"] * 100
                model_accs.append(val)
                # Highlight if beats baseline
                base = baselines.get(task, 0) * 100
                if base > 0 and val > base:
                    cell = f"**{val:.2f}%** 🟢"
                elif base > 0 and val > base - 1.0:  # within 1%
                    cell = f"{val:.2f}% 🟡"
                else:
                    cell = f"{val:.2f}%"
            else:
                cell = "-"
            row.append(cell)

        mean_val = np.mean(model_accs) if model_accs else 0.0
        row.append(f"{mean_val:.2f}%")
        lines.append("| " + " | ".join(row) + " |")

    lines.append(
        "\n> **Legend**: 🟢 Beats Backprop Baseline | 🟡 Matches Baseline (within 1%)"
    )

    lines.append("\n## 3. Algorithm Insights")

    for task in sorted_tasks:
        lines.append(f"### Task: {task.upper()}")
        base = baselines.get(task, 0)

        competitors = []
        for model in sorted_models:
            if "backprop" in model.lower():
                continue
            stats = agg_matrix[model].get(task)
            if stats and stats["mean"] >= base:
                competitors.append((model, stats["mean"]))

        if competitors:
            competitors.sort(key=lambda x: x[1], reverse=True)
            lines.append(
                "The following biologically plausible algorithms "
                f"rivaled or beat Backprop ({base * 100:.2f}%):"
            )
            for m, acc in competitors:
                diff = (acc - base) * 100
                lines.append(f"- **{m}**: {acc * 100:.2f}% (+{diff:.2f}%)")
        else:
            lines.append(
                "No algorithms matched the Backprop baseline on this task yet."
            )

    # Save to file
    with pathlib.Path(output_path).open("w") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)
