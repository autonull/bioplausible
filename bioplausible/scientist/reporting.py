"""
ScientistReporter: Generates publication-quality reports from experiment data.
"""

import json
import logging
import os
import sqlite3

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Headless mode
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.statistics import StatisticalAnalyzer
from bioplausible.visualization import ResultVisualizer

# ML Imports
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reporter")


class ScientistReporter:
    """
    Generates analysis reports from the experiment database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = HyperoptStorage(db_path)
        self.analyzer = StatisticalAnalyzer()

    def generate_report(self, output_dir: str):
        """
        Main entry point. Generates Markdown and Images.
        """
        start_time = datetime.now()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        images_dir = out_path / "images"
        images_dir.mkdir(exist_ok=True)

        self.visualizer = ResultVisualizer(output_dir=images_dir)

        logger.info(f"Loading data from {self.db_path}...")
        try:
            trials = self.storage.get_all_trials()
        except Exception as e:
            logger.error(f"Failed to load trials from DB: {e}")
            return

        # Filter completed
        trials = [t for t in trials if t.status == "completed"]
        if not trials:
            logger.warning("No completed trials found.")
            return

        # 1. Prepare Data
        df = self._prepare_dataframe(trials)

        # 2. Generate Plots
        self._safe_plot(self._plot_leaderboard, df)
        self._safe_plot(self._plot_tier_progress, df)
        self._safe_plot(self._plot_hyperparam_correlations, df)
        self._safe_plot(self._plot_pareto_frontier, df)
        self._safe_plot(self._plot_significance_matrix, df)

        # 3. ML Analysis
        insights = ""
        try:
            insights = self._run_ml_analysis(df, images_dir)
        except Exception as e:
            logger.error(f"ML Analysis failed: {e}")
            insights = f"_Machine Learning Analysis failed to run: {e}_"

        # 4. Write Markdown
        try:
            self._write_markdown(df, insights, out_path / "index.md")
        except Exception as e:
            logger.error(f"Failed to write markdown report: {e}")

        logger.info(f"Report generated in {output_dir} ({datetime.now() - start_time})")

    def _safe_plot(self, func, *args):
        """Wrapper to catch plotting errors without aborting report."""
        try:
            func(*args)
        except Exception as e:
            logger.error(f"Plotting error in {func.__name__}: {e}")

    def _prepare_dataframe(self, trials):
        """
        Flattens trials into a list of dicts (lightweight DataFrame).
        """
        data = []
        for t in trials:
            row = {
                "id": t.trial_id,
                "model": t.model_name,
                "accuracy": t.accuracy,
                "loss": t.final_loss,
                "params": t.param_count,
            }
            # Flatten config
            for k, v in t.config.items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v

            # Ensure task/tier exist
            if "task" not in row:
                row["task"] = "unknown"
            if "tier" not in row:
                row["tier"] = "unknown"

            data.append(row)
        return data

    def _plot_leaderboard(self, data):
        """Bar chart of Top Accuracy per Model per Task."""
        tasks = sorted(list(set(d["task"] for d in data)))
        for task in tasks:
            self.visualizer.plot_leaderboard(data, task)

    def _plot_tier_progress(self, data):
        """Count of trials per tier."""
        self.visualizer.plot_tier_progress(data)

    def _plot_hyperparam_correlations(self, data):
        """Scatter plots of Hyperparams vs Accuracy."""
        self.visualizer.plot_hyperparam_correlations(data)

    def _plot_pareto_frontier(self, data):
        """Pareto Frontier: Accuracy vs Parameters."""
        self.visualizer.plot_pareto_frontier(data)

    def _plot_significance_matrix(self, data):
        """Heatmap of P-values between models (T-Test)."""
        # Note: Visualizer expects p_values and labels. We compute them here using Analyzer.
        models = sorted(list(set(d["model"] for d in data)))
        n = len(models)
        p_values = np.ones((n, n))

        # Only consider Standard/Deep for valid stats
        valid_data = [d for d in data if d.get("tier") in ["standard", "deep"]]

        # If not enough data, skip
        if len(valid_data) < 5:
            return

        for i, m1 in enumerate(models):
            accs1 = [d["accuracy"] for d in valid_data if d["model"] == m1]
            if len(accs1) < 3:
                continue

            for j, m2 in enumerate(models):
                if i == j:
                    continue
                accs2 = [d["accuracy"] for d in valid_data if d["model"] == m2]
                if len(accs2) < 3:
                    continue

                stats = self.analyzer.compare_algorithms(accs1, accs2, names=(m1, m2))
                p_values[i, j] = stats.get("p_val", 1.0)

        self.visualizer.plot_significance_matrix(p_values, models)

    def _run_ml_analysis(self, data, img_dir):
        """
        Uses Decision Trees to find rules for high performance.
        """
        if not HAS_ML:
            return "ML Analysis libraries (scikit-learn) not installed."

        insights = []
        models = list(set(d["model"] for d in data))

        for model in models:
            m_data = [d for d in data if d["model"] == model]
            if len(m_data) < 10:
                continue

            exclude = {
                "id",
                "model",
                "accuracy",
                "loss",
                "task",
                "tier",
                "epochs",
                "batch_size",
                "params",
            }
            keys = set()
            for d in m_data:
                keys.update(d.keys())

            feature_keys = [k for k in keys if k not in exclude]

            X, y = [], []
            for d in m_data:
                row = []
                valid = True
                for k in feature_keys:
                    val = d.get(k)
                    if isinstance(val, (int, float)):
                        row.append(val)
                    else:
                        valid = False

                if valid:
                    X.append(row)
                    y.append(d["accuracy"])

            if not X:
                continue

            X = np.array(X)
            y = np.array(y)

            reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=3)
            reg.fit(X, y)

            rules = export_text(reg, feature_names=feature_keys)

            insights.append(f"### ML Insights for {model}")
            insights.append(f"**Key Drivers of Performance**:")

            imp = reg.feature_importances_
            indices = np.argsort(imp)[::-1]
            for i in indices[:3]:
                if imp[i] > 0.01:
                    insights.append(f"- **{feature_keys[i]}**: {imp[i]:.2%} importance")

            insights.append(
                f"\n**Decision Rules (Tree Structure):**\n```\n{rules}\n```\n"
            )

            plt.figure(figsize=(12, 6), dpi=100)
            plot_tree(
                reg, feature_names=feature_keys, filled=True, rounded=True, precision=3
            )
            plt.title(f"Decision Tree for {model}", fontsize=14)
            plt.savefig(img_dir / f"tree_{model}.png")
            plt.close()

        return "\n".join(insights)

    def _write_markdown(self, data, insights, path):
        """Writes the final report."""
        best_acc = 0.0
        best_model = "None"
        if data:
            best_entry = max(data, key=lambda x: x["accuracy"])
            best_acc = best_entry["accuracy"]
            best_model = best_entry["model"]

        lines = [
            f"# AutoScientist Discovery Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"",
            f"## 1. Executive Summary",
            f"The autonomous system has conducted **{len(data)}** experiments.",
            f"The current state-of-the-art model discovered is **{best_model}** with **{best_acc:.2%}** accuracy.",
            f"",
            f"### Global Leaderboard",
        ]

        tasks = sorted(list(set(d["task"] for d in data)))
        for t in tasks:
            lines.append(f"#### Task: {t.upper()}")
            lines.append(f"![Leaderboard {t}](images/leaderboard_{t}.png)")

        lines.append(f"## 2. Experimental Progress")
        lines.append(f"![Tier Progress](images/tier_progress.png)")

        lines.append(f"## 3. Scientific Validity")
        lines.append(f"### Efficiency Frontier")
        lines.append(f"![Pareto](images/pareto_frontier.png)")
        lines.append(f"### Statistical Significance (P-Values)")
        lines.append(f"![Significance](images/significance_matrix.png)")

        lines.append(f"## 4. Machine Learning Analysis")
        lines.append(
            f"The system trained internal models to understand what makes these algorithms work."
        )
        lines.append(insights)

        lines.append(f"## 5. Hyperparameter Correlations")
        lines.append(f"![LR Impact](images/impact_learning_rate.png)")
        lines.append(f"![Beta Impact](images/impact_beta.png)")

        with open(path, "w") as f:
            f.write("\n".join(lines))
