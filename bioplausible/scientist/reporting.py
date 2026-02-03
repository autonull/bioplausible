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

# ML Imports
try:
    from scipy.stats import ttest_ind
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

    def generate_report(self, output_dir: str):
        """
        Main entry point. Generates Markdown and Images.
        """
        start_time = datetime.now()
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        images_dir = out_path / "images"
        images_dir.mkdir(exist_ok=True)

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
        self._safe_plot(self._plot_leaderboard, df, images_dir)
        self._safe_plot(self._plot_tier_progress, df, images_dir)
        self._safe_plot(self._plot_hyperparam_correlations, df, images_dir)
        self._safe_plot(self._plot_pareto_frontier, df, images_dir)
        self._safe_plot(self._plot_significance_matrix, df, images_dir)

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

    def _plot_leaderboard(self, data, img_dir):
        """Bar chart of Top Accuracy per Model per Task."""
        tasks = sorted(list(set(d["task"] for d in data)))
        models = sorted(list(set(d["model"] for d in data)))

        for task in tasks:
            task_data = [d for d in data if d["task"] == task]
            if not task_data:
                continue

            best_accs = {}
            for m in models:
                m_data = [d["accuracy"] for d in task_data if d["model"] == m]
                if m_data:
                    best_accs[m] = max(m_data)

            if not best_accs:
                continue

            sorted_items = sorted(best_accs.items(), key=lambda x: x[1])
            names = [x[0] for x in sorted_items]
            vals = [x[1] for x in sorted_items]

            plt.figure(figsize=(10, 6), dpi=100)
            bars = plt.barh(names, vals, color="#4ecdc4")
            plt.title(f"Leaderboard: {task.upper()}", fontsize=14)
            plt.xlabel("Accuracy", fontsize=12)
            plt.xlim(0, 1.0)
            plt.grid(axis="x", linestyle="--", alpha=0.7)

            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{width:.2%}",
                    va="center",
                )

            plt.tight_layout()
            plt.savefig(img_dir / f"leaderboard_{task}.png")
            plt.close()

    def _plot_tier_progress(self, data, img_dir):
        """Count of trials per tier."""
        tiers = ["smoke", "shallow", "standard", "deep"]
        counts = defaultdict(int)
        for d in data:
            t = d.get("tier", "unknown")
            counts[t] += 1

        plt.figure(figsize=(8, 5), dpi=100)
        x = range(len(tiers))
        y = [counts[t] for t in tiers]
        plt.bar(x, y, color="#ff6b6b")
        plt.xticks(x, [t.title() for t in tiers])
        plt.title("Experimental Progress (Trial Counts)", fontsize=14)
        plt.ylabel("Number of Trials", fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(img_dir / "tier_progress.png")
        plt.close()

    def _plot_hyperparam_correlations(self, data, img_dir):
        """Scatter plots of Hyperparams vs Accuracy."""
        params = ["learning_rate", "beta", "weight_decay"]

        for param in params:
            vals = []
            accs = []
            for d in data:
                if param in d and isinstance(d[param], (int, float)):
                    vals.append(d[param])
                    accs.append(d["accuracy"])

            if not vals:
                continue

            plt.figure(figsize=(8, 5), dpi=100)
            plt.scatter(vals, accs, alpha=0.6, c=accs, cmap="viridis")
            plt.title(f"Impact of {param}", fontsize=14)
            plt.xlabel(param, fontsize=12)
            plt.ylabel("Accuracy", fontsize=12)
            if param == "learning_rate":
                plt.xscale("log")
            plt.colorbar(label="Accuracy")
            plt.grid(True, alpha=0.3)
            plt.savefig(img_dir / f"impact_{param}.png")
            plt.close()

    def _plot_pareto_frontier(self, data, img_dir):
        """Pareto Frontier: Accuracy vs Parameters."""
        plt.figure(figsize=(10, 6), dpi=100)

        models = list(set(d["model"] for d in data))

        for model in models:
            m_data = [d for d in data if d["model"] == model]
            x = [d.get("params", 0) for d in m_data]
            y = [d["accuracy"] for d in m_data]

            # Simple scatter
            plt.scatter(x, y, label=model, alpha=0.7)

        plt.title("Efficiency Frontier (Accuracy vs Scale)", fontsize=14)
        plt.xlabel("Parameters (Millions)", fontsize=12)
        plt.ylabel("Accuracy", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(img_dir / "pareto_frontier.png")
        plt.close()

    def _plot_significance_matrix(self, data, img_dir):
        """Heatmap of P-values between models (T-Test)."""
        if not HAS_ML:
            return

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

                _, p = ttest_ind(accs1, accs2, equal_var=False)
                p_values[i, j] = p

        plt.figure(figsize=(10, 8), dpi=100)
        sns.heatmap(
            p_values,
            xticklabels=models,
            yticklabels=models,
            annot=True,
            fmt=".2f",
            cmap="Blues_r",
            vmin=0,
            vmax=0.05,
        )
        plt.title("Statistical Significance (P-Values)", fontsize=14)
        plt.tight_layout()
        plt.savefig(img_dir / "significance_matrix.png")
        plt.close()

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
