"""
Analysis Core Logic

Decoupled from UI to enable headless CLI usage.
"""

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from bioplausible.hyperopt.comparison import (ComparisonMetric,
                                              compute_algorithm_rankings,
                                              group_trials_by_family)


def load_trials(db_path: str) -> List[Dict[str, Any]]:
    """
    Load all trials from Optuna SQLite database.
    """
    if not Path(db_path).exists():
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Query trials with study names
    trials = []
    # Note: hyperopt_logs table uses Optuna trial_id as PK so we match on trial_id
    cursor.execute("""
        SELECT
            t.trial_id,
            t.number,
            t.state,
            s.study_name
        FROM trials t
        JOIN studies s ON t.study_id = s.study_id
        WHERE t.state = 'COMPLETE'
        ORDER BY t.trial_id
    """)

    for row in cursor.fetchall():
        trial = dict(row)
        trial_id = trial["trial_id"]

        # Extract model name
        study_name = trial["study_name"]
        if study_name.startswith("shallow_"):
            model_name = study_name.replace("shallow_", "").replace("_", " ").title()
        else:
            model_name = study_name.replace("_", " ").title()
        trial["model_name"] = model_name

        # Load trial values
        cursor.execute(
            """
            SELECT objective, value
            FROM trial_values
            WHERE trial_id = ?
            ORDER BY objective
        """,
            (trial_id,),
        )

        values = cursor.fetchall()

        # Handle different objective counts
        # 2 Objectives: Accuracy (0), Loss (1)
        # 3 Objectives: Accuracy (0), Params (1), Time (2) - Legacy/scalarized?

        if len(values) >= 3:
            trial["accuracy"] = values[0]["value"]
            trial["param_count"] = values[1]["value"]
            trial["iteration_time"] = values[2]["value"]
        elif len(values) == 2:
            trial["accuracy"] = values[0]["value"]
            trial["final_loss"] = values[1]["value"]
            # Try to recover params/time from user attrs or trial params if logged differently?
            # Or just leave as defaults if not specialized objectives
            trial["param_count"] = 0.0  # Placeholder
            trial["iteration_time"] = 0.0  # Placeholder
        else:
            trial["accuracy"] = 0.0
            trial["param_count"] = 0.0
            trial["iteration_time"] = 0.0

        # Load params
        cursor.execute(
            """
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        """,
            (trial_id,),
        )

        params = cursor.fetchall()
        trial["config"] = {p["param_name"]: p["param_value"] for p in params}

        # Load user attributes (e.g. tier)
        # Optuna stores attributes in 'value_json' column
        cursor.execute(
            """
            SELECT key, value_json
            FROM trial_user_attributes
            WHERE trial_id = ?
        """,
            (trial_id,),
        )
        attrs = cursor.fetchall()

        user_attrs = {}
        import json

        for a in attrs:
            try:
                # Optuna stores values as JSON strings
                user_attrs[a["key"]] = json.loads(a["value_json"])
            except (json.JSONDecodeError, TypeError):
                user_attrs[a["key"]] = a["value_json"]

        trial["user_attrs"] = user_attrs

        # Merge with hyperopt_logs for detailed metrics (param_count, time)
        try:
            cursor.execute(
                """
                SELECT param_count, iteration_time
                FROM hyperopt_logs
                WHERE trial_id = ?
            """,
                (trial_id,),
            )
            row = cursor.fetchone()
            if row:
                if row["param_count"]:
                    trial["param_count"] = row["param_count"]
                if row["iteration_time"]:
                    trial["iteration_time"] = row["iteration_time"]
        except sqlite3.OperationalError:
            # Table might not exist in old DBs
            pass

        # Extract tier specifically for top-level access
        # Default to 'shallow' if missing (legacy compatibility)
        trial["tier"] = trial["user_attrs"].get("tier", "shallow")

        # Placeholders
        trial["final_loss"] = 0.0
        trial["perplexity"] = 0.0
        trial["status"] = "completed"
        trial["epochs_completed"] = trial["config"].get("epochs", 5)

        trials.append(trial)

    conn.close()
    return trials


def compute_statistics(trials: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics per model."""
    stats = defaultdict(
        lambda: {
            "accuracy": [],
            "param_count": [],
            "iteration_time": [],
        }
    )

    for trial in trials:
        model = trial["model_name"]
        stats[model]["accuracy"].append(trial["accuracy"])
        stats[model]["param_count"].append(trial["param_count"])
        stats[model]["iteration_time"].append(trial["iteration_time"])

    result = {}
    for model, metrics in stats.items():
        result[model] = {
            "accuracy_mean": float(np.mean(metrics["accuracy"])),
            "accuracy_std": float(np.std(metrics["accuracy"])),
            "param_count_mean": float(np.mean(metrics["param_count"])),
            "param_count_std": float(np.std(metrics["param_count"])),
            "time_mean": float(np.mean(metrics["iteration_time"])),
            "time_std": float(np.std(metrics["iteration_time"])),
            "n_trials": len(metrics["accuracy"]),
        }

    return result


def compute_pareto_frontier(trials: List[Dict[str, Any]]) -> List[int]:
    """Compute Pareto frontier trial IDs."""
    if not trials:
        return []

    pareto_ids = []

    for i, trial_a in enumerate(trials):
        is_dominated = False

        for j, trial_b in enumerate(trials):
            if i == j:
                continue

            better_acc = trial_b["accuracy"] >= trial_a["accuracy"]
            better_params = trial_b["param_count"] <= trial_a["param_count"]
            better_time = trial_b["iteration_time"] <= trial_a["iteration_time"]

            strictly_better = (
                trial_b["accuracy"] > trial_a["accuracy"]
                or trial_b["param_count"] < trial_a["param_count"]
                or trial_b["iteration_time"] < trial_a["iteration_time"]
            )

            if better_acc and better_params and better_time and strictly_better:
                is_dominated = True
                break

        if not is_dominated:
            pareto_ids.append(trial_a["trial_id"])

    return pareto_ids


def get_rankings(trials: List[Dict[str, Any]]) -> List[Any]:
    """Compute comprehensive rankings with gap analysis."""
    trials_by_family = group_trials_by_family(trials)
    rankings = compute_algorithm_rankings(
        trials_by_family, metric=ComparisonMetric.ACCURACY
    )

    baseline = next(
        (
            r
            for r in rankings
            if "backprop" in r.family.lower() or "baseline" in r.family.lower()
        ),
        None,
    )
    if baseline and baseline.best_value > 0:
        for r in rankings:
            gap = (baseline.best_value - r.best_value) / baseline.best_value * 100
            r.gap_to_baseline = gap

    return rankings


def load_trials_timeseries(db_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Load epoch-by-epoch metrics for all trials.
    Returns a dictionary mapping trial_id to a list of epoch metrics.
    """
    if not Path(db_path).exists():
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Check if tables exist first to avoid errors on empty DB
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='epoch_metrics'"
    )
    if not cursor.fetchone():
        conn.close()
        return {}

    cursor.execute("""
        SELECT
            trial_id, epoch, loss, accuracy, perplexity, time
        FROM epoch_metrics
        ORDER BY trial_id, epoch
    """)

    timeseries = defaultdict(list)
    for row in cursor.fetchall():
        trial_id = row["trial_id"]
        timeseries[trial_id].append(dict(row))

    conn.close()
    return dict(timeseries)


def print_rankings(rankings: List[Any]):
    """Print rankings table."""
    print(f"{'Rank':<6} {'Family':<20} {'Best Acc':<10} {'Gap':<10} {'Trials':>8}")
    print(f"{'-'*6} {'-'*20} {'-'*10} {'-'*10} {'-'*8}")

    for i, r in enumerate(rankings, 1):
        gap = f"{r.gap_to_baseline:+.1f}%" if r.gap_to_baseline is not None else "Base"
        if r.gap_to_baseline is None and i > 1:
            # If not baseline but no gap calc (e.g. baseline not found), show nothing or N/A
            gap = "N/A"

        print(
            f"#{i:<5} {r.family:<20} {r.best_value*100:6.2f}%    {gap:<10} {r.n_trials:8d}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Bioplausible Experiment Analysis")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # helper for filtering
    def filter_trials(trials, tier=None, task=None):
        filtered = []
        for t in trials:
            # Filter by Tier
            if tier and t.get("user_attrs", {}).get("tier") != tier:
                continue

            # Filter by Task
            pass

            filtered.append(t)
        return filtered

    # RANK command
    rank_parser = subparsers.add_parser("rank", help="Show algorithm rankings")
    rank_parser.add_argument("--db", default="bioplausible.db", help="Path to database")
    rank_parser.add_argument("--tier", help="Filter by patience tier")
    rank_parser.add_argument("--task", help="Filter by task (e.g. mnist, lm)")

    args = parser.parse_args()

    if args.command == "rank":
        trials = load_trials(args.db)

        # Filter
        if args.tier or args.task:
            filtered = []
            for t in trials:
                if args.tier:
                    # check explicit tier attr first, fallback to study name suffix
                    trial_tier = t.get("user_attrs", {}).get("tier")
                    if not trial_tier:
                        parts = t["study_name"].split("_")
                        if parts:
                            trial_tier = parts[-1]

                    if trial_tier != args.tier:
                        continue

                if args.task:
                    # check task in study name
                    # specific enough? usually model_task_tier
                    # strict check might be hard if model has underscores.
                    # simplified check: is task string in study_name?
                    if args.task not in t["study_name"]:
                        continue

                filtered.append(t)
            trials = filtered

        rankings = get_rankings(trials)
        print_rankings(rankings)

    else:
        parser.print_help()
