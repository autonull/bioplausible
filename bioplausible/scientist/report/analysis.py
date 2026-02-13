"""
Machine Learning Analysis for Scientist Reports.

Performs analysis on experiment results to identify factors driving performance
and rank models using statistical methods.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

    HAS_ML = True
except ImportError:
    HAS_ML = False

logger = logging.getLogger("MLAnalyzer")


class MLAnalyzer:
    """
    Performs Machine Learning analysis on experiment results.

    Attributes:
        output_dir (Path): Directory to save analysis plots.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize the MLAnalyzer.

        Args:
            output_dir (Path): Directory for output files.
        """
        self.output_dir = output_dir

    def run_analysis(self, data: List[Dict[str, Any]]) -> Tuple[str, str]:
        """
        Main entry point for analysis.

        Args:
            data (List[Dict]): List of experiment result dictionaries.

        Returns:
            Tuple[str, str]: A tuple containing (insights_markdown, robustness_markdown).
        """
        sensitivity = self._analyze_sensitivity(data)
        robustness = ""
        if sensitivity:
            robustness = self._analyze_robustness(sensitivity)

        # Append direct robustness metrics if available
        direct_robustness = self._analyze_direct_robustness(data)
        if direct_robustness:
            robustness += "\n" + direct_robustness

        if not HAS_ML:
            return "ML Analysis libraries (scikit-learn) not installed.", robustness

        insights: List[str] = []

        # --- 1. Global Analysis ---
        insights.append("### Global Performance Analysis")
        insights.append(
            "A decision tree was trained on the entire dataset to identify which algorithms and tasks drive performance."
        )

        global_features = []
        global_y = []

        for d in data:
            feat = {
                "model": d.get("model", "unknown"),
                "task": d.get("task", "unknown"),
                "tier": d.get("tier", "unknown"),
                "params": d.get("params", 0),
            }
            if "lr" in d:
                feat["lr"] = d["lr"]
            if "beta" in d:
                feat["beta"] = d["beta"]

            global_features.append(feat)
            global_y.append(d["accuracy"])

        if len(global_features) > 10:
            try:
                vec = DictVectorizer(sparse=False)
                X_global = vec.fit_transform(global_features)
                y_global = np.array(global_y)
                feature_names = vec.get_feature_names_out()

                reg_global = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
                reg_global.fit(X_global, y_global)

                imp = reg_global.feature_importances_
                indices = np.argsort(imp)[::-1]
                insights.append("**Top Global Factors:**")
                for i in indices[:5]:
                    if imp[i] > 0.01:
                        insights.append(
                            f"- **{feature_names[i]}**: {imp[i]:.2%} importance"
                        )

                rules_global = export_text(
                    reg_global, feature_names=list(feature_names)
                )
                insights.append(
                    f"\n**Global Decision Rules:**\n```\n{rules_global}\n```\n"
                )

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
                plt.savefig(self.output_dir / "tree_global.png")
                plt.close()
            except Exception as e:
                logger.error(f"Global ML analysis failed: {e}")

        # --- 2. Granular Analysis (Task -> Model) ---
        tasks = list(set(d.get("task", "unknown") for d in data))

        for task in tasks:
            task_data = [d for d in data if d.get("task") == task]
            if len(task_data) < 5:
                continue

            insights.append(f"\n### Deep Dive: {task.upper()}")

            models = list(set(d["model"] for d in task_data))
            for model in models:
                m_data = [d for d in task_data if d["model"] == model]
                if len(m_data) < 5:
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
                    "accuracy_std",
                    "accuracy_min",
                    "accuracy_max",
                    "loss_std",
                    "count",
                    "iteration_time",
                    "val_loss",
                    "val_accuracy",
                    "val_perplexity",
                    "time",
                    "is_pareto",
                    "config",
                }

                keys = set()
                for d in m_data:
                    keys.update(d.keys())
                feature_keys = sorted(
                    [k for k in keys if k not in exclude and not k.startswith("train_")]
                )

                X, y = [], []
                valid_features = []

                for k in feature_keys:
                    vals = [d.get(k) for d in m_data if d.get(k) is not None]
                    if not vals:
                        continue
                    if all(isinstance(v, (int, float)) for v in vals):
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

                X_arr = np.array(X)
                y_arr = np.array(y)

                try:
                    reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=2)
                    reg.fit(X_arr, y_arr)

                    rules = export_text(reg, feature_names=valid_features)
                    imp = reg.feature_importances_
                    indices = np.argsort(imp)[::-1]
                    top_factors = []
                    for i in indices[:3]:
                        if imp[i] > 0.05:
                            top_factors.append(
                                f"**{valid_features[i]}** ({imp[i]:.0%})"
                            )

                    if top_factors:
                        insights.append(
                            f"**{model}** on {task}: Driven by {', '.join(top_factors)}"
                        )
                        insights.append(f"```\n{rules}\n```")

                        plt.figure(figsize=(10, 6), dpi=100)
                        plot_tree(
                            reg,
                            feature_names=valid_features,
                            filled=True,
                            rounded=True,
                            precision=3,
                            fontsize=9,
                        )
                        safe_model = model.replace(" ", "_").replace("/", "_")
                        plt.title(f"{model} on {task}", fontsize=12)
                        plt.tight_layout()
                        plt.savefig(self.output_dir / f"tree_{task}_{safe_model}.png")
                        plt.close()
                except Exception as e:
                    logger.warning(f"Failed to train tree for {model} on {task}: {e}")

        return "\n".join(insights), robustness

    def _analyze_sensitivity(
        self, data: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze parameter sensitivity."""
        sensitivity: Dict[str, Dict[str, float]] = {}
        models = list(set(d["model"] for d in data))

        ignore = {
            "id",
            "model",
            "task",
            "tier",
            "accuracy",
            "loss",
            "family",
            "accuracy_percentile",
            "job_id",
            "fold",
            "seed",
            "start_time",
            "end_time",
            "status",
            "param_count",
            "params",
            "accuracy_ci_95",
            "accuracy_std",
            "accuracy_min",
            "accuracy_max",
            "accuracy_percentile_mean",
            "loss_mean",
            "loss_std",
            "count",
            "iteration_time",
            "val_loss",
            "val_accuracy",
            "val_perplexity",
            "time",
            "is_pareto",
            "config",
        }

        for model in models:
            m_data = [d for d in data if d["model"] == model]
            if len(m_data) < 5:
                continue

            model_sens = {}
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

                groups = defaultdict(list)
                if isinstance(vals[0], (int, float)) and len(set(vals)) > 5:
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

                total_var = np.var(accs)
                if total_var < 1e-9:
                    continue

                group_means = [np.mean(g) for g in groups.values() if g]
                group_sizes = [len(g) for g in groups.values() if g]

                if not group_means:
                    continue

                grand_mean = np.average(group_means, weights=group_sizes)
                var_explained = np.average(
                    [(m - grand_mean) ** 2 for m in group_means], weights=group_sizes
                )

                model_sens[k] = var_explained / total_var

            if model_sens:
                sensitivity[model] = model_sens

        return sensitivity

    def _analyze_robustness(self, sensitivity: Dict[str, Dict[str, float]]) -> str:
        """Generate robustness report from sensitivity data."""
        if not sensitivity:
            return ""

        scores = []
        for model, params in sensitivity.items():
            mean_sens = np.mean(list(params.values()))
            scores.append((model, mean_sens))

        scores.sort(key=lambda x: x[1])

        lines = ["\n### Hyperparameter Sensitivity Analysis"]
        lines.append(
            "Lower sensitivity score indicates more robust performance across hyperparameter changes."
        )
        lines.append("| Model | Sensitivity Score | Classification |")
        lines.append("|---|---|---|")

        for m, s in scores:
            cls = "Robust" if s < 0.1 else ("Sensitive" if s < 0.3 else "Fragile")
            lines.append(f"| {m} | {s:.3f} | {cls} |")

        return "\n".join(lines)

    def _analyze_direct_robustness(self, data: List[Dict[str, Any]]) -> str:
        """
        Analyze direct robustness metrics (Adversarial, Noise, OOD).
        """
        robust_data = [d for d in data if d.get("robustness_score") is not None]
        if not robust_data:
            return ""

        metrics = [
            "robustness_score",
            "noise_score",
            "perturbation_score",
            "ood_score",
            "adversarial_fgsm",
            "adversarial_pgd",
        ]

        model_stats = defaultdict(lambda: defaultdict(list))

        for d in robust_data:
            model = d.get("model") or d.get("model_name")
            if not model:
                continue
            for m in metrics:
                val = d.get(m)
                if val is not None:
                    model_stats[model][m].append(val)

        if not model_stats:
            return ""

        lines = ["\n### Adversarial & Noise Robustness"]
        lines.append(
            "Evaluation against noise injection, input perturbation, and adversarial attacks."
        )
        lines.append(
            "| Model | Overall | Noise | Perturb | OOD | Adv (FGSM) | Adv (PGD) |"
        )
        lines.append("|---|---|---|---|---|---|---|")

        sorted_models = sorted(
            model_stats.keys(),
            key=lambda x: (
                np.mean(model_stats[x]["robustness_score"])
                if model_stats[x]["robustness_score"]
                else 0
            ),
            reverse=True,
        )

        for model in sorted_models:
            stats = model_stats[model]
            row = [f"**{model}**"]

            for m in metrics:
                vals = stats.get(m, [])
                if vals:
                    mean_val = np.mean(vals)
                    row.append(f"{mean_val:.3f}")
                else:
                    row.append("-")

            lines.append(f"| {' | '.join(row)} |")

        return "\n".join(lines)


class BayesianRanker:
    """
    Ranks models by probability of superiority using Beta distribution sampling.
    """

    def rank_models(self, agg_data: List[Dict[str, Any]]) -> str:
        """
        Ranks models and returns Markdown table.

        Args:
            agg_data: Aggregated data with 'count', 'accuracy', 'model'.

        Returns:
            str: Markdown table of rankings.
        """
        if not agg_data:
            return "_No data for ranking._"

        model_stats = defaultdict(list)
        for d in agg_data:
            model = d["model"]
            model_stats[model].append(d)

        best_models = {}
        for m, configs in model_stats.items():
            best = max(configs, key=lambda x: x["accuracy"])
            best_models[m] = best

        models = sorted(best_models.keys())
        if len(models) < 2:
            return "_Insufficient models for Bayesian ranking._"

        samples = {}
        for m in models:
            d = best_models[m]
            n = d.get("count", 1)
            acc = max(0.0, min(1.0, d["accuracy"]))
            # Prior: Alpha=1, Beta=1. Posterior: Alpha=1+k, Beta=1+(n-k)
            k = int(acc * n)
            alpha = 1 + k
            beta = 1 + (n - k)
            samples[m] = np.random.beta(alpha, beta, 1000)

        ranking = []
        for m in models:
            wins = 0
            for opponent in models:
                if m == opponent:
                    continue
                prob = np.mean(samples[m] > samples[opponent])
                if prob > 0.5:
                    wins += 1
            ranking.append((m, wins, np.mean(samples[m])))

        ranking.sort(key=lambda x: (x[1], x[2]), reverse=True)

        lines = ["| Rank | Model | Win Score | Mean Est. Acc |"]
        lines.append("|---|---|---|---|")
        for i, (m, wins, mean_acc) in enumerate(ranking):
            lines.append(
                f"| {i+1} | **{m}** | {wins}/{len(models)-1} | {mean_acc:.2%} |"
            )

        return "\n".join(lines)
