"""
ScientistReporter: Generates publication-quality reports from experiment data.
"""

import os
import json
import logging
import sqlite3
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

from bioplausible.hyperopt.storage import HyperoptStorage

# ML Imports
try:
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
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
        trials = self.storage.get_all_trials()

        # Filter completed
        trials = [t for t in trials if t.status == "completed"]
        if not trials:
            logger.warning("No completed trials found.")
            return

        # 1. Prepare Data
        df = self._prepare_dataframe(trials)

        # 2. Generate Plots
        self._plot_leaderboard(df, images_dir)
        self._plot_tier_progress(df, images_dir)
        self._plot_hyperparam_correlations(df, images_dir)

        # 3. ML Analysis
        insights = self._run_ml_analysis(df, images_dir)

        # 4. Write Markdown
        self._write_markdown(df, insights, out_path / "index.md")

        logger.info(f"Report generated in {output_dir} ({datetime.now() - start_time})")

    def _prepare_dataframe(self, trials):
        """
        Flattens trials into a list of dicts (lightweight DataFrame).
        """
        data = []
        for t in trials:
            row = {
                'id': t.trial_id,
                'model': t.model_name,
                'accuracy': t.accuracy,
                'loss': t.final_loss,
            }
            # Flatten config
            for k, v in t.config.items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v

            # Ensure task/tier exist
            if 'task' not in row: row['task'] = 'unknown'
            if 'tier' not in row: row['tier'] = 'unknown'

            data.append(row)
        return data

    def _plot_leaderboard(self, data, img_dir):
        """Bar chart of Top Accuracy per Model per Task."""
        # Group by Task -> Model -> Max Acc
        tasks = sorted(list(set(d['task'] for d in data)))
        models = sorted(list(set(d['model'] for d in data)))

        # Prepare grid
        # We'll make one plot per task if many tasks, or one grouped bar chart

        for task in tasks:
            task_data = [d for d in data if d['task'] == task]
            if not task_data: continue

            best_accs = {}
            for m in models:
                m_data = [d['accuracy'] for d in task_data if d['model'] == m]
                if m_data:
                    best_accs[m] = max(m_data)

            if not best_accs: continue

            # Sort by acc
            sorted_items = sorted(best_accs.items(), key=lambda x: x[1])
            names = [x[0] for x in sorted_items]
            vals = [x[1] for x in sorted_items]

            plt.figure(figsize=(10, 6))
            bars = plt.barh(names, vals, color='#4ecdc4')
            plt.title(f"Leaderboard: {task.upper()}")
            plt.xlabel("Accuracy")
            plt.xlim(0, 1.0)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            # Add labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                         f'{width:.2%}', va='center')

            plt.tight_layout()
            plt.savefig(img_dir / f"leaderboard_{task}.png")
            plt.close()

    def _plot_tier_progress(self, data, img_dir):
        """Count of trials per tier."""
        tiers = ['smoke', 'shallow', 'standard', 'deep']
        counts = defaultdict(int)
        for d in data:
            t = d.get('tier', 'unknown')
            counts[t] += 1

        plt.figure(figsize=(8, 5))
        x = range(len(tiers))
        y = [counts[t] for t in tiers]
        plt.bar(x, y, color='#ff6b6b')
        plt.xticks(x, [t.title() for t in tiers])
        plt.title("Experimental Progress (Trial Counts)")
        plt.ylabel("Number of Trials")
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(img_dir / "tier_progress.png")
        plt.close()

    def _plot_hyperparam_correlations(self, data, img_dir):
        """Scatter plots of Hyperparams vs Accuracy."""
        # Focus on a few key params if they exist
        params = ['learning_rate', 'beta', 'weight_decay']

        for param in params:
            # Check if param exists in data
            vals = []
            accs = []
            for d in data:
                if param in d and isinstance(d[param], (int, float)):
                    vals.append(d[param])
                    accs.append(d['accuracy'])

            if not vals: continue

            plt.figure(figsize=(8, 5))
            plt.scatter(vals, accs, alpha=0.6, c=accs, cmap='viridis')
            plt.title(f"Impact of {param}")
            plt.xlabel(param)
            plt.ylabel("Accuracy")
            if param == 'learning_rate':
                plt.xscale('log')
            plt.colorbar(label='Accuracy')
            plt.grid(True, alpha=0.3)
            plt.savefig(img_dir / f"impact_{param}.png")
            plt.close()

    def _run_ml_analysis(self, data, img_dir):
        """
        Uses Decision Trees to find rules for high performance.
        """
        if not HAS_ML:
            return "ML Analysis libraries (scikit-learn) not installed."

        insights = []

        # Analyze per Model family to find model-specific rules
        models = list(set(d['model'] for d in data))

        for model in models:
            m_data = [d for d in data if d['model'] == model]
            if len(m_data) < 10: continue # Need enough data

            # Extract features
            # We filter for numerical/categorical features that are hyperparameters
            exclude = {'id', 'model', 'accuracy', 'loss', 'task', 'tier', 'epochs', 'batch_size'}

            feature_names = []
            X = []
            y = []

            # Identify keys present in this model's data
            keys = set()
            for d in m_data:
                keys.update(d.keys())

            feature_keys = [k for k in keys if k not in exclude]

            for d in m_data:
                row = []
                valid = True
                for k in feature_keys:
                    val = d.get(k)
                    if isinstance(val, (int, float)):
                        row.append(val)
                    else:
                        valid = False # Skip non-numeric for now (simple regressor)
                        # Could use LabelEncoder here if needed

                if valid:
                    X.append(row)
                    y.append(d['accuracy'])

            if not X: continue

            X = np.array(X)
            y = np.array(y)

            # Train Tree
            reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=3)
            reg.fit(X, y)

            # Export Rules
            rules = export_text(reg, feature_names=feature_keys)

            insights.append(f"### ML Insights for {model}")
            insights.append(f"**Key Drivers of Performance**:")

            # Feature Importance
            imp = reg.feature_importances_
            indices = np.argsort(imp)[::-1]
            for i in indices[:3]:
                if imp[i] > 0.01:
                    insights.append(f"- **{feature_keys[i]}**: {imp[i]:.2%} importance")

            insights.append(f"\n**Decision Rules (Tree Structure):**\n```\n{rules}\n```\n")

            # Plot Tree
            plt.figure(figsize=(12, 6))
            plot_tree(reg, feature_names=feature_keys, filled=True, rounded=True, precision=3)
            plt.title(f"Decision Tree for {model}")
            plt.savefig(img_dir / f"tree_{model}.png")
            plt.close()

        return "\n".join(insights)

    def _write_markdown(self, data, insights, path):
        """Writes the final report."""
        best_acc = 0.0
        best_model = "None"
        if data:
            best_entry = max(data, key=lambda x: x['accuracy'])
            best_acc = best_entry['accuracy']
            best_model = best_entry['model']

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

        # Add leaderboard images
        tasks = sorted(list(set(d['task'] for d in data)))
        for t in tasks:
            lines.append(f"#### Task: {t.upper()}")
            lines.append(f"![Leaderboard {t}](images/leaderboard_{t}.png)")

        lines.append(f"## 2. Experimental Progress")
        lines.append(f"![Tier Progress](images/tier_progress.png)")

        lines.append(f"## 3. Machine Learning Analysis")
        lines.append(f"The system trained internal models to understand what makes these algorithms work.")
        lines.append(insights)

        lines.append(f"## 4. Hyperparameter Correlations")
        lines.append(f"![LR Impact](images/impact_learning_rate.png)")
        lines.append(f"![Beta Impact](images/impact_beta.png)")

        with open(path, "w") as f:
            f.write("\n".join(lines))
