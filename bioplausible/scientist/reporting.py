"""
ScientistReporter: Generates publication-quality reports from experiment data.
"""

import logging
import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import matplotlib
import numpy as np
from scipy.stats import percentileofscore

# Headless mode must be set before pyplot import
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bioplausible.hyperopt.storage import HyperoptStorage  # noqa: E402
from bioplausible.models.registry import get_model_spec  # noqa: E402
from bioplausible.scientist.decisions import DecisionLogger  # noqa: E402
from bioplausible.statistics import StatisticalAnalyzer  # noqa: E402
from bioplausible.visualization import ResultVisualizer  # noqa: E402

# ML Imports
try:
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
    from sklearn.feature_extraction import DictVectorizer

    HAS_ML = True
except ImportError:
    HAS_ML = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reporter")

ALGORITHM_FAMILIES = {
    "BackpropMLP": "Backprop",
    "LoopedMLP": "Backprop",
    "ConvEqProp": "Energy-Based",
    "TransformerEqProp": "Energy-Based",
    "StandardEqProp": "Energy-Based",
    "EquilibriumAlignment": "Hebbian",
    "AdaptiveFeedbackAlignment": "Backprop-Variant",
    "ContrastiveFeedbackAlignment": "Backprop-Variant",
    "LayerwiseEquilibriumFA": "Hybrid",
    "PredictiveCodingHybrid": "Predictive Coding",
    "EnergyGuidedFA": "Hybrid",
    "SparseEquilibrium": "Hebbian",
    "MomentumEquilibrium": "Energy-Based",
    "StochasticFA": "Backprop-Variant",
    "EnergyMinimizingFA": "Energy-Based",
}


class ScientistReporter:
    """
    Generates analysis reports from the experiment database.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.storage = HyperoptStorage(db_path)
        self.decision_logger = DecisionLogger(db_path)
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
        raw_df = self._prepare_dataframe(trials)
        agg_df = self._aggregate_config_stats(raw_df)

        # 2. Generate Plots (Use Aggregated for Leaderboard, Raw for others)
        self._safe_plot(self._plot_leaderboard, agg_df)
        self._safe_plot(self._plot_efficiency_leaderboard, agg_df)
        self._safe_plot(self._plot_family_leaderboard, agg_df)
        self._safe_plot(self._plot_tier_progress, raw_df)
        self._safe_plot(self._plot_hyperparam_correlations, raw_df)
        self._safe_plot(self._plot_pareto_frontier, agg_df)  # Use agg to see stable points
        self._safe_plot(self._plot_significance_matrix, raw_df)  # Analyzer needs raw samples
        self._safe_plot(self._plot_convergence_speed, raw_df)
        self._safe_plot(self.visualizer.plot_task_difficulty, raw_df)

        # Convergence Metrics & Curves
        convergence_report = ""
        try:
            trajectories = self.storage.get_all_trajectories()
            if trajectories:
                self.visualizer.plot_convergence_curves(trajectories)
                convergence_report = self._compute_convergence_metrics(trajectories)
        except Exception as e:
            logger.warning(f"Could not load/plot trajectories: {e}")

        # 3. ML Analysis
        insights = ""
        robustness_analysis = ""
        try:
            insights, robustness_analysis = self._run_ml_analysis(raw_df, images_dir)
        except Exception as e:
            logger.error(f"ML Analysis failed: {e}")
            insights = f"_Machine Learning Analysis failed to run: {e}_"

        # 4. Generate Narrative (Explainability)
        narrative = self._generate_narrative(agg_df, raw_df)
        chronicle = self._generate_chronicle()
        bayesian_ranking = self._compute_bayesian_ranking(agg_df)
        family_analysis = self._analyze_family_strengths_by_domain(agg_df)

        # Export Best Config
        try:
            self._export_best_config(agg_df, out_path)
        except Exception as e:
            logger.error(f"Failed to export best config: {e}")

        # 5. Write Markdown
        try:
            self._write_markdown(agg_df, insights, narrative, bayesian_ranking, convergence_report, family_analysis, out_path / "index.md")
        except Exception as e:
            logger.error(f"Failed to write markdown report: {e}")

        # 5. Write LaTeX (Academic)
        try:
            self._generate_latex_report(agg_df, out_path)
        except Exception as e:
            logger.error(f"Failed to generate LaTeX report: {e}")

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
        Adds percentile ranks per task.
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

            # Assign Family
            row["family"] = ALGORITHM_FAMILIES.get(t.model_name, "Other")

            data.append(row)

        # Calculate percentiles per task
        tasks = set(d["task"] for d in data)
        for task in tasks:
            task_indices = [i for i, d in enumerate(data) if d["task"] == task]
            if not task_indices:
                continue
            scores = [data[i]["accuracy"] for i in task_indices]

            for i in task_indices:
                # pct score 0-100
                p = percentileofscore(scores, data[i]["accuracy"], kind='rank')
                data[i]["accuracy_percentile"] = p

        return data

    def _aggregate_config_stats(self, data: List[Dict]) -> List[Dict]:
        """
        Groups trials by configuration to handle repeats/folds.
        Returns a list of unique configurations with aggregated stats.
        """
        from collections import defaultdict
        import hashlib

        # Identify keys to exclude from hash (metadata/randomness)
        exclude_keys = {
            "id",
            "accuracy",
            "loss",
            "seed",
            "job_id",
            "fold",
            "is_verification",
            "verified_trial_id",
            "start_time",
            "end_time",
            "status",
            "accuracy_percentile",
        }

        grouped = defaultdict(list)

        for row in data:
            # Create a hash of the stable config
            config_items = []
            for k, v in sorted(row.items()):
                if k not in exclude_keys:
                    config_items.append((k, v))

            config_hash = hashlib.md5(json.dumps(config_items, sort_keys=True, default=str).encode()).hexdigest()
            grouped[config_hash].append(row)

        aggregated = []
        for config_hash, rows in grouped.items():
            accs = [r["accuracy"] for r in rows]
            losses = [r["loss"] for r in rows if r["loss"] is not None]
            percentiles = [r.get("accuracy_percentile", 0.0) for r in rows]

            # Use the first row as the base template
            agg_row = rows[0].copy()

            # Add stats
            agg_row["count"] = len(rows)
            agg_row["accuracy_mean"] = float(np.mean(accs))
            agg_row["accuracy_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            agg_row["accuracy_min"] = float(np.min(accs))
            agg_row["accuracy_max"] = float(np.max(accs))

            # CI 95% = 1.96 * std / sqrt(n)
            if len(accs) > 1:
                std = np.std(accs, ddof=1)
                n = len(accs)
                agg_row["accuracy_ci_95"] = float(1.96 * std / np.sqrt(n))
            else:
                agg_row["accuracy_ci_95"] = 0.0

            # Cross-task metric (mean percentile for this config)
            agg_row["accuracy_percentile_mean"] = float(np.mean(percentiles))

            if losses:
                agg_row["loss_mean"] = float(np.mean(losses))
                agg_row["loss_std"] = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
            else:
                agg_row["loss_mean"] = float("inf")

            # Remove instance-specific fields
            for k in ["id", "seed", "job_id", "fold", "accuracy", "loss", "accuracy_percentile"]:
                if k in agg_row:
                    del agg_row[k]

            # But map accuracy_mean to accuracy for backward compat in plots (leaderboard sorts by 'accuracy')
            agg_row["accuracy"] = agg_row["accuracy_mean"]
            agg_row["loss"] = agg_row["loss_mean"]

            aggregated.append(agg_row)

        return aggregated

    def _export_best_config(self, data, out_path: Path):
        """Exports the best model configuration to a JSON file."""
        if not data:
            return

        # Find best by accuracy
        best_entry = max(data, key=lambda x: x["accuracy"])

        # Save as JSON
        config_path = out_path / "best_config.json"
        with open(config_path, "w") as f:
            json.dump(best_entry, f, indent=4)
        logger.info(f"Best configuration saved to {config_path}")

    def _plot_leaderboard(self, data):
        """Bar chart of Top Accuracy per Model per Task."""
        tasks = sorted(list(set(d["task"] for d in data)))
        for task in tasks:
            self.visualizer.plot_leaderboard(data, task, use_std=True)

    def _plot_efficiency_leaderboard(self, data):
        """Bar chart of Efficiency (Acc/Params) per Model per Task."""
        tasks = sorted(list(set(d["task"] for d in data)))
        for task in tasks:
            self.visualizer.plot_leaderboard(data, task, use_std=False, metric="efficiency")

    def _plot_family_leaderboard(self, data):
        """Bar chart of Mean Accuracy per Algorithm Family."""
        self.visualizer.plot_family_leaderboard(data)

    def _plot_convergence_speed(self, data):
        """Plot speed of learning."""
        self.visualizer.plot_convergence_speed(data)

    def _plot_convergence_curves(self):
        """Plot convergence curves from trajectory data."""
        try:
            trajectories = self.storage.get_all_trajectories()
            if trajectories:
                self.visualizer.plot_convergence_curves(trajectories)
                return self._compute_convergence_metrics(trajectories)
        except Exception as e:
            logger.warning(f"Could not load/plot trajectories: {e}")
        return ""

    def _analyze_family_strengths_by_domain(self, data) -> str:
        """
        Analyzes performance of algorithm families per domain (Vision, RL, Language).
        """
        domains = {
            "Vision": ["mnist", "fashion_mnist", "cifar10", "cifar100", "svhn"],
            "RL": ["cartpole", "lunar_lander", "acrobot", "pendulum", "mountain_car"],
            "Language": ["char_ngram", "tiny_shakespeare", "wikitext2", "penn_treebank"],
        }

        # Map task to domain
        task_domain_map = {}
        for domain, tasks in domains.items():
            for t in tasks:
                task_domain_map[t] = domain

        # Aggregate
        domain_family_stats = defaultdict(lambda: defaultdict(list))

        for d in data:
            task = d["task"]
            family = d.get("family", "Other")
            acc = d["accuracy"]

            # Find domain (default to 'Other' if unknown)
            domain = "Other"
            for known_task, known_domain in task_domain_map.items():
                if known_task in task: # Partial match
                    domain = known_domain
                    break

            domain_family_stats[domain][family].append(acc)

        if not domain_family_stats:
            return ""

        lines = ["\n### Family Performance by Domain"]

        for domain in sorted(domain_family_stats.keys()):
            if domain == "Other" and len(domain_family_stats) > 1:
                continue

            lines.append(f"\n**{domain} Domain**")
            lines.append("| Family | Mean Accuracy | Top Model |")
            lines.append("|---|---|---|")

            fam_stats = []
            for fam, accs in domain_family_stats[domain].items():
                mean_acc = np.mean(accs)
                # Find top model for this family in this domain
                top_model = "N/A"
                best_acc = -1.0
                for d in data:
                    # Re-check logic: d must be in this domain and family
                    t_domain = "Other"
                    for kt, kd in task_domain_map.items():
                        if kt in d["task"]:
                            t_domain = kd
                            break

                    if t_domain == domain and d.get("family") == fam:
                        if d["accuracy"] > best_acc:
                            best_acc = d["accuracy"]
                            top_model = d["model"]

                fam_stats.append((fam, mean_acc, top_model))

            fam_stats.sort(key=lambda x: x[1], reverse=True)

            for fam, mean, top in fam_stats:
                lines.append(f"| {fam} | {mean:.2%} | {top} |")

        return "\n".join(lines)

    def _compute_convergence_metrics(self, trajectories) -> str:
        """
        Calculates convergence speed (epochs to 90% of max accuracy).
        Returns a markdown section.
        """
        # Group by model
        model_metrics = defaultdict(list)

        for t in trajectories:
            if not t.checkpoints:
                continue

            max_acc = max(ckpt.val_acc for ckpt in t.checkpoints)
            if max_acc < 0.1: # Skip failing models
                continue

            target = 0.9 * max_acc
            epoch_90 = None
            for ckpt in t.checkpoints:
                if ckpt.val_acc >= target:
                    epoch_90 = ckpt.epoch
                    break

            if epoch_90 is not None:
                model_metrics[t.model_name].append(epoch_90)

        if not model_metrics:
            return "_No convergence data available._"

        # Create Table
        lines = ["\n### Fastest Learners (Epochs to 90% Accuracy)"]
        lines.append("| Model | Mean Epochs | Min Epochs |")
        lines.append("|---|---|---|")

        sorted_models = []
        for m, epochs in model_metrics.items():
            sorted_models.append((m, np.mean(epochs), np.min(epochs)))

        sorted_models.sort(key=lambda x: x[1])

        for m, mean_e, min_e in sorted_models:
            lines.append(f"| {m} | {mean_e:.1f} | {min_e} |")

        return "\n".join(lines)

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
        """Heatmap of P-values between models (T-Test). Generates per-task matrices."""
        tasks = sorted(list(set(d["task"] for d in data)))

        for task in tasks:
            task_data = [d for d in data if d["task"] == task]

            # Only consider Standard/Deep for valid stats
            valid_data = [d for d in task_data if d.get("tier") in ["standard", "deep"]]

            # If not enough data, skip
            if len(valid_data) < 5:
                continue

            models = sorted(list(set(d["model"] for d in valid_data)))
            if len(models) < 2:
                continue

            n = len(models)
            p_values = np.ones((n, n))

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

            self.visualizer.plot_significance_matrix(p_values, models, save_name=f"significance_matrix_{task}.png")

    def _run_ml_analysis(self, data, img_dir):
        """
        Uses Decision Trees to find rules for high performance.
        Includes Global Analysis and Per-Model Analysis.
        """
        # Add Sensitivity Analysis
        sensitivity = self._analyze_sensitivity(data)
        robustness = ""
        if sensitivity:
             self.visualizer.plot_sensitivity_heatmap(sensitivity)
             robustness = self._analyze_robustness(sensitivity)

        if not HAS_ML:
            return "ML Analysis libraries (scikit-learn) not installed.", robustness

        insights = []

        # --- 1. Global Analysis ---
        insights.append("### Global Performance Analysis")
        insights.append(
            "A decision tree was trained on the entire dataset to identify which algorithms and tasks drive performance."
        )

        # Prepare Global Data
        # We want to use 'model', 'task' as categorical features, and maybe 'params'.
        # DictVectorizer handles string values as one-hot features.
        global_features = []
        global_y = []

        for d in data:
            # Select relevant global features
            feat = {
                "model": d.get("model", "unknown"),
                "task": d.get("task", "unknown"),
                "tier": d.get("tier", "unknown"),
                "params": d.get("params", 0),
            }
            # Add some common hyperparams if present
            if "lr" in d:
                feat["lr"] = d["lr"]
            if "beta" in d:
                feat["beta"] = d["beta"]

            global_features.append(feat)
            global_y.append(d["accuracy"])

        if len(global_features) > 10:
            vec = DictVectorizer(sparse=False)
            X_global = vec.fit_transform(global_features)
            y_global = np.array(global_y)
            feature_names = vec.get_feature_names_out()

            # Train Global Tree
            reg_global = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
            reg_global.fit(X_global, y_global)

            # Global Insights
            imp = reg_global.feature_importances_
            indices = np.argsort(imp)[::-1]
            insights.append("**Top Global Factors:**")
            for i in indices[:5]:
                if imp[i] > 0.01:
                    insights.append(
                        f"- **{feature_names[i]}**: {imp[i]:.2%} importance"
                    )

            rules_global = export_text(reg_global, feature_names=list(feature_names))
            insights.append(
                f"\n**Global Decision Rules:**\n```\n{rules_global}\n```\n"
            )

            # Global Plot
            plt.figure(figsize=(16, 8), dpi=100)
            plot_tree(
                reg_global,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                precision=3,
                fontsize=10,
            )
            plt.title("Global Performance Decision Tree", fontsize=16)
            plt.tight_layout()
            plt.savefig(img_dir / "tree_global.png")
            plt.close()

        # --- 2. Granular Analysis (Task -> Model) ---
        tasks = list(set(d.get("task", "unknown") for d in data))

        for task in tasks:
            task_data = [d for d in data if d.get("task") == task]
            if len(task_data) < 5:
                continue

            insights.append(f"\n### Deep Dive: {task.upper()}")

            # Per-Task Global Tree
            # ... (omitted for brevity, focusing on per-model within task)

            models = list(set(d["model"] for d in task_data))
            for model in models:
                m_data = [d for d in task_data if d["model"] == model]
                if len(m_data) < 5: # Lower threshold for granular analysis
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
                    "study_name",
                    "job_id",
                    "accuracy_std", "accuracy_min", "accuracy_max", "loss_std", "count",
                    "iteration_time", "val_loss", "val_accuracy", "val_perplexity", "time"
                }

                # Identify features (hyperparams)
                keys = set()
                for d in m_data:
                    keys.update(d.keys())
                feature_keys = sorted([k for k in keys if k not in exclude and not k.startswith("train_")])

                X, y = [], []
                valid_features = []

                # First pass: check which features are actually numeric and variable
                for k in feature_keys:
                    vals = [d.get(k) for d in m_data if d.get(k) is not None]
                    if not vals: continue
                    if all(isinstance(v, (int, float)) for v in vals):
                        # check variance
                        if np.std(vals) > 1e-9:
                            valid_features.append(k)

                if not valid_features:
                    continue

                for d in m_data:
                    row = []
                    valid = True
                    for k in valid_features:
                        val = d.get(k)
                        if isinstance(val, (int, float)):
                            row.append(val)
                        else:
                            valid = False

                    if valid:
                        X.append(row)
                        y.append(d["accuracy"])

                if len(X) < 5:
                    continue

                X = np.array(X)
                y = np.array(y)

                try:
                    reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2)
                    reg.fit(X, y)

                    rules = export_text(reg, feature_names=valid_features)

                    # Compute importance
                    imp = reg.feature_importances_
                    indices = np.argsort(imp)[::-1]
                    top_factors = []
                    for i in indices[:3]:
                        if imp[i] > 0.05:
                            top_factors.append(f"**{valid_features[i]}** ({imp[i]:.0%})")

                    if top_factors:
                        insights.append(f"**{model}** on {task}: Driven by {', '.join(top_factors)}")
                        insights.append(f"```\n{rules}\n```")

                        # Plot
                        plt.figure(figsize=(10, 6), dpi=100)
                        plot_tree(
                            reg,
                            feature_names=valid_features,
                            filled=True,
                            rounded=True,
                            precision=3,
                            fontsize=9
                        )
                        safe_model = model.replace(" ", "_").replace("/", "_")
                        plt.title(f"{model} on {task}", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(img_dir / f"tree_{task}_{safe_model}.png")
                        plt.close()
                except Exception as e:
                    logger.warning(f"Failed to train tree for {model} on {task}: {e}")

        return "\n".join(insights), robustness

    def _analyze_robustness(self, sensitivity) -> str:
        """
        Classifies models as Robust or Fragile based on sensitivity score.
        """
        if not sensitivity:
            return ""

        scores = []
        for model, params in sensitivity.items():
            # Mean sensitivity across parameters
            mean_sens = np.mean(list(params.values()))
            scores.append((model, mean_sens))

        scores.sort(key=lambda x: x[1])

        lines = ["\n### Model Robustness Analysis"]
        lines.append("Lower sensitivity score indicates more robust performance across hyperparameter changes.")
        lines.append("| Model | Sensitivity Score | Classification |")
        lines.append("|---|---|---|")

        for m, s in scores:
            cls = "Robust" if s < 0.1 else ("Sensitive" if s < 0.3 else "Fragile")
            lines.append(f"| {m} | {s:.3f} | {cls} |")

        return "\n".join(lines)

    def _analyze_sensitivity(self, data: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Computes sensitivity index (normalized variance explained) for hyperparameters.
        """
        sensitivity = {}
        models = list(set(d["model"] for d in data))

        ignore = {"id", "model", "task", "tier", "accuracy", "loss", "family", "accuracy_percentile", "job_id", "fold", "seed", "start_time", "end_time", "status", "param_count", "params", "accuracy_ci_95", "accuracy_std", "accuracy_min", "accuracy_max", "accuracy_percentile_mean", "loss_mean", "loss_std", "count", "iteration_time", "val_loss", "val_accuracy", "val_perplexity", "time", "is_pareto", "config"}

        for model in models:
            m_data = [d for d in data if d["model"] == model]
            if len(m_data) < 5:
                continue

            model_sens = {}
            # Find relevant keys
            keys = set().union(*(d.keys() for d in m_data)) - ignore

            for k in keys:
                vals = []
                accs = []
                for d in m_data:
                    v = d.get(k)
                    if v is not None and isinstance(v, (int, float, str)):
                        vals.append(v)
                        accs.append(d["accuracy"])

                if not vals or len(set(vals)) < 2:
                    continue

                # Calculate variance explained (Sensitivity Index)
                groups = defaultdict(list)
                if isinstance(vals[0], (int, float)) and len(set(vals)) > 5:
                    # Binning for continuous variables
                    try:
                        bins = np.linspace(min(vals), max(vals), 5)
                        digitized = np.digitize(vals, bins)
                        for i, d_idx in enumerate(digitized):
                            groups[d_idx].append(accs[i])
                    except Exception:
                        continue
                else:
                    for v, acc in zip(vals, accs):
                        groups[v].append(acc)

                # Var(E[Y|X]) / Var(Y)
                total_var = np.var(accs)
                if total_var < 1e-9:
                    continue

                group_means = [np.mean(g) for g in groups.values() if g]
                group_sizes = [len(g) for g in groups.values() if g]

                if not group_means:
                    continue

                grand_mean = np.average(group_means, weights=group_sizes)
                var_explained = np.average([(m - grand_mean)**2 for m in group_means], weights=group_sizes)

                model_sens[k] = var_explained / total_var

            if model_sens:
                sensitivity[model] = model_sens

        return sensitivity

    def _generate_latex_report(self, data, out_path: Path):
        """Generates a LaTeX paper with citations. Uses aggregated data."""
        tex_path = out_path / "report.tex"
        bib_path = out_path / "references.bib"

        # Check for tools
        has_pdflatex = shutil.which("pdflatex") is not None
        has_bibtex = shutil.which("bibtex") is not None

        if not has_pdflatex:
            logger.warning(
                "pdflatex not found. Skipping PDF compilation steps in script."
            )

        # 1. Generate BibTeX
        used_models = set(d["model"] for d in data)
        bib_content = set()
        for m_name in used_models:
            try:
                spec = get_model_spec(m_name)
                if spec.citation:
                    bib_content.add(spec.citation)
            except ValueError:
                pass

        with open(bib_path, "w") as f:
            f.write("\n\n".join(bib_content))

        # 2. Generate LaTeX
        best_acc = 0.0
        best_model = "None"
        best_entry = None
        if data:
            best_entry = max(data, key=lambda x: x["accuracy"])
            best_acc = best_entry["accuracy"]
            best_model = best_entry["model"]

        latex = []
        latex.append(r"\documentclass{article}")
        latex.append(r"\usepackage{graphicx}")
        latex.append(r"\usepackage{booktabs}")
        latex.append(r"\usepackage{hyperref}")
        latex.append(r"\usepackage{listings}")
        latex.append(r"\usepackage[margin=1in]{geometry}")
        latex.append(
            r"\title{Autonomous Discovery of Bio-Plausible Learning Algorithms}"
        )
        latex.append(r"\author{AutoScientist}")
        latex.append(r"\date{\today}")
        latex.append(r"\begin{document}")
        latex.append(r"\maketitle")

        latex.append(r"\begin{abstract}")
        latex.append(
            f"We present the results of an autonomous search for biologically "
            f"plausible learning algorithms. "
            f"Our system explored {len(data)} configurations across multiple tasks. "
            f"The top-performing model, {best_model}, achieved {best_acc*100:.2f}\\% "
            f"accuracy."
        )
        latex.append(r"\end{abstract}")

        latex.append(r"\section{Introduction}")
        latex.append(
            r"Deep learning relies on backpropagation, which is biologically "
            r"implausible. "
            r"Alternative algorithms such as Equilibrium Propagation "
            r"\cite{scellier2017equilibrium} and "
            r"Feedback Alignment \cite{lillicrap2016random} have been proposed."
        )

        latex.append(r"\section{Methodology}")
        latex.append(
            r"We utilized the AutoScientist framework to iteratively explore the "
            r"hyperparameter space. "
            r"Models were evaluated on tasks including Vision (MNIST/CIFAR) and "
            r"Language Modeling."
        )

        latex.append(r"\section{Chronicle of Discovery}")
        latex.append(r"The following log details the autonomous decisions made by the scientist.")
        latex.append(r"\begin{itemize}")

        logs = self.decision_logger.get_log(limit=50)
        for log in logs:
            safe_desc = log['description'].replace('_', r'\_').replace('%', r'\%')
            latex.append(f"\\item \\textbf{{{log['date_str']}}} [{log['event_type']}]: {safe_desc}")

        latex.append(r"\end{itemize}")

        latex.append(r"\section{Results}")

        # Leaderboard Table
        latex.append(r"\subsection{Leaderboard}")
        latex.append(r"\begin{table}[h]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{l c c}")
        latex.append(r"\toprule")
        latex.append(r"Model & Task & Score (Mean $\pm$ Std) \\")
        latex.append(r"\midrule")

        # Top models (already aggregated)
        data.sort(key=lambda x: x["accuracy"], reverse=True)
        seen = set()
        count = 0
        for d in data:
            key = (d["model"], d["task"])
            if key not in seen:
                acc = d["accuracy"]
                std = d.get("accuracy_std", 0)
                std_str = f" $\\pm$ {std*100:.2f}" if std > 0 else ""
                latex.append(
                    f"{d['model']} & {d['task']} & {acc*100:.2f}\\%{std_str} \\\\"
                )
                seen.add(key)
                count += 1
                if count >= 10:
                    break

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(
            r"\caption{Top performing algorithms. Scores include standard deviation where multiple trials exist.}"
        )
        latex.append(r"\end{table}")

        # Figures
        latex.append(r"\subsection{Analysis}")
        latex.append(r"\begin{figure}[h]")
        latex.append(r"\centering")
        latex.append(
            r"\includegraphics[width=0.8\textwidth]{images/pareto_frontier.png}"
        )
        latex.append(r"\caption{Pareto Frontier: Accuracy vs Parameter Efficiency.}")
        latex.append(r"\end{figure}")

        latex.append(r"\begin{figure}[h]")
        latex.append(r"\centering")
        latex.append(
            r"\includegraphics[width=0.8\textwidth]{images/significance_matrix.png}"
        )
        latex.append(r"\caption{Statistical Significance Matrix (P-Values).}")
        latex.append(r"\end{figure}")

        # Machine Learning Analysis (Added)
        latex.append(r"\section{Machine Learning Analysis}")
        latex.append(
            r"We utilized decision tree regression to interpret the experimental results."
        )

        # Global Tree
        global_tree_img = "images/tree_global.png"
        if (out_path / global_tree_img).exists():
            latex.append(r"\begin{figure}[h]")
            latex.append(r"\centering")
            latex.append(f"\\includegraphics[width=1.0\\textwidth]{{{global_tree_img}}}")
            latex.append(r"\caption{Global Decision Tree: Algorithm Comparison}")
            latex.append(r"\end{figure}")

        if best_entry:
            tree_img = f"images/tree_{best_model}.png"
            if (out_path / tree_img).exists():
                latex.append(r"\begin{figure}[h]")
                latex.append(r"\centering")
                latex.append(f"\\includegraphics[width=1.0\\textwidth]{{{tree_img}}}")
                latex.append(f"\\caption{{Decision Tree for Best Model ({best_model})}}")
                latex.append(r"\end{figure}")
            else:
                latex.append(
                    r"No decision tree visualization available for the best model."
                )

        latex.append(r"\bibliographystyle{plain}")
        latex.append(r"\bibliography{references}")

        # Appendix
        latex.append(r"\appendix")
        latex.append(r"\section{Best Configuration}")
        latex.append(r"The hyperparameters for the top performing model are:")
        latex.append(r"\begin{lstlisting}[basicstyle=\ttfamily\small, breaklines=true]")
        if best_entry:
            latex.append(json.dumps(best_entry, indent=2))
        latex.append(r"\end{lstlisting}")

        latex.append(r"\end{document}")

        with open(tex_path, "w") as f:
            f.write("\n".join(latex))

        # 3. Compile Script
        with open(out_path / "compile_report.sh", "w") as f:
            f.write("#!/bin/bash\n")
            if has_pdflatex and has_bibtex:
                f.write("pdflatex report.tex\n")
                f.write("bibtex report\n")
                f.write("pdflatex report.tex\n")
                f.write("pdflatex report.tex\n")
            else:
                f.write(
                    "echo 'pdflatex or bibtex not found. Please install TeX Live.'\n"
                )

        os.chmod(out_path / "compile_report.sh", 0o755)

    def _compute_bayesian_ranking(self, agg_data) -> str:
        """
        Ranks models by probability of superiority using Beta distribution sampling.
        Returns a Markdown table string.
        """
        if not agg_data:
            return "_No data for ranking._"

        # Group by model
        from collections import defaultdict
        model_stats = defaultdict(list)
        for d in agg_data:
            model = d["model"]
            model_stats[model].append(d)

        # Find best config per model to represent peak performance
        best_models = {}
        for m, configs in model_stats.items():
            best = max(configs, key=lambda x: x["accuracy"])
            best_models[m] = best

        models = sorted(best_models.keys())
        if len(models) < 2:
            return "_Insufficient models for Bayesian ranking._"

        # Sampling
        samples = {}
        for m in models:
            d = best_models[m]
            # Prior: Alpha=1, Beta=1. Posterior: Alpha=1+k, Beta=1+(n-k)
            n = d.get("count", 1)
            # Clip accuracy to valid range just in case
            acc = max(0.0, min(1.0, d["accuracy"]))
            k = int(acc * n)
            alpha = 1 + k
            beta = 1 + (n - k)
            # Draw samples
            samples[m] = np.random.beta(alpha, beta, 1000)

        # Compute pairwise probabilities
        ranking = []
        for m in models:
            wins = 0
            for opponent in models:
                if m == opponent:
                    continue
                # Probability m > opponent
                prob = np.mean(samples[m] > samples[opponent])
                if prob > 0.5:
                    wins += 1
            ranking.append((m, wins, np.mean(samples[m])))

        ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

        # Table
        lines = ["| Rank | Model | Win Score | Mean Est. Acc |"]
        lines.append("|---|---|---|---|")
        for i, (m, wins, mean_acc) in enumerate(ranking):
            lines.append(f"| {i+1} | **{m}** | {wins}/{len(models)-1} | {mean_acc:.2%} |")

        return "\n".join(lines)

    def _generate_chronicle(self) -> str:
        """Generates a Markdown journal of decisions."""
        logs = self.decision_logger.get_log(limit=200)
        if not logs:
            return "_No significant strategic decisions recorded yet._"

        lines = []
        lines.append("| Timestamp | Event | Description |")
        lines.append("|-----------|-------|-------------|")
        for log in logs:
            lines.append(f"| {log['date_str']} | **{log['event_type']}** | {log['description']} |")

        return "\n".join(lines)

    def _generate_narrative(self, agg_data, raw_data) -> str:
        """Generates plain English narrative explaining the results."""
        models = sorted(list(set(d["model"] for d in agg_data)))
        valid_raw = [d for d in raw_data if d.get("tier") in ["standard", "deep"]]

        narrative = []

        # 1. Overall Winner
        if not agg_data:
            return "No data available."

        best = max(agg_data, key=lambda x: x["accuracy"])
        std_info = ""
        if best.get("accuracy_std", 0) > 0:
            std_info = f" (±{best['accuracy_std']:.2%})"

        narrative.append(f"The top performing model is **{best['model']}**, achieving a score of **{best['accuracy']:.2%}**{std_info} (Accuracy or Proxy Metric).")

        # 2. Pairwise Comparisons (Significance)
        if len(models) > 1:
            narrative.append("\n### Key Comparisons")

            # Compare top 2 distinct models
            model_scores = {}
            for m in models:
                scores = [d["accuracy"] for d in valid_raw if d["model"] == m]
                if scores:
                    model_scores[m] = scores

            sorted_models = sorted(model_scores.keys(), key=lambda m: np.mean(model_scores[m]), reverse=True)

            if len(sorted_models) >= 2:
                m1, m2 = sorted_models[0], sorted_models[1]
                s1, s2 = model_scores[m1], model_scores[m2]

                if len(s1) > 2 and len(s2) > 2:
                    stats = self.analyzer.compare_algorithms(s1, s2, names=(m1, m2))
                    p = stats['p_val']
                    d = stats.get('cohens_d', 0.0)
                    diff = stats['mean_a'] - stats['mean_b']

                    sig_icon = ""
                    if p < 0.05:
                        if abs(d) > 0.8:
                            sig_icon = "✓ (Large Effect)"
                        elif abs(d) > 0.5:
                            sig_icon = "✓ (Medium Effect)"
                        else:
                            sig_icon = "~ (Small Effect)"
                    else:
                        sig_icon = "(ns)"

                    sig_str = "statistically significant" if p < 0.05 else "not statistically significant"
                    narrative.append(f"- **{m1} vs {m2}**: {m1} outperforms by {diff:.2%}. {sig_icon} (p={p:.4f}, d={d:.2f}).")

        return "\n".join(narrative)

    def _write_markdown(self, data, insights, narrative, bayesian_ranking, convergence_report, family_analysis, path):
        """Writes the final report."""
        chronicle = self._generate_chronicle()
        best_acc = 0.0
        best_model = "None"
        if data:
            best_entry = max(data, key=lambda x: x["accuracy"])
            best_acc = best_entry["accuracy"]
            best_model = best_entry["model"]

        lines = [
            "# AutoScientist Discovery Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 1. Executive Summary",
            f"The autonomous system has conducted **{len(data)}** experiments.",
            f"The current state-of-the-art model discovered is **{best_model}** "
            f"with **{best_acc:.2%}** accuracy.",
            "",
            "### Research Narrative",
            narrative,
            "",
            "### Chronicle of Discovery",
            chronicle,
            "",
            "### Global Leaderboard",
        ]

        lines.append(family_analysis)

        tasks = sorted(list(set(d["task"] for d in data)))
        for t in tasks:
            lines.append(f"#### Task: {t.upper()}")
            lines.append(f"![Leaderboard {t}](images/leaderboard_{t}.png)")
            lines.append(f"![Efficiency {t}](images/leaderboard_{t}_efficiency.png)")

        lines.append("## 2. Experimental Progress")
        lines.append("![Tier Progress](images/tier_progress.png)")

        lines.append("## 3. Scientific Validity")
        lines.append("### Bayesian Ranking (Probabilistic Superiority)")
        lines.append(bayesian_ranking)
        lines.append("### Efficiency Frontier")
        lines.append("![Pareto](images/pareto_frontier.png)")
        lines.append("### Convergence Speed")
        lines.append("![Convergence](images/convergence_speed.png)")
        lines.append(convergence_report)
        lines.append("### Task Difficulty Analysis")
        lines.append("![Task Difficulty](images/task_difficulty.png)")
        lines.append("### Statistical Significance (P-Values)")

        # Significance matrices are now per-task
        tasks = sorted(list(set(d["task"] for d in data)))
        for t in tasks:
             lines.append(f"#### {t.upper()}")
             lines.append(f"![Significance {t}](images/significance_matrix_{t}.png)")

        lines.append("## 4. Machine Learning Analysis")
        lines.append(
            "The system trained internal models to understand what makes these "
            "algorithms work."
        )
        lines.append(robustness_analysis)
        lines.append(insights)

        lines.append("## 5. Hyperparameter Correlations")
        lines.append("![LR Impact](images/impact_learning_rate.png)")
        lines.append("![Beta Impact](images/impact_beta.png)")

        with open(path, "w") as f:
            f.write("\n".join(lines))
