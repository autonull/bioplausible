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

import matplotlib
import numpy as np

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
        self._safe_plot(self._plot_tier_progress, raw_df)
        self._safe_plot(self._plot_hyperparam_correlations, raw_df)
        self._safe_plot(self._plot_pareto_frontier, agg_df)  # Use agg to see stable points
        self._safe_plot(self._plot_significance_matrix, raw_df)  # Analyzer needs raw samples

        # 3. ML Analysis
        insights = ""
        try:
            insights = self._run_ml_analysis(raw_df, images_dir)
        except Exception as e:
            logger.error(f"ML Analysis failed: {e}")
            insights = f"_Machine Learning Analysis failed to run: {e}_"

        # 4. Generate Narrative (Explainability)
        narrative = self._generate_narrative(agg_df, raw_df)
        chronicle = self._generate_chronicle()

        # Export Best Config
        try:
            self._export_best_config(agg_df, out_path)
        except Exception as e:
            logger.error(f"Failed to export best config: {e}")

        # 5. Write Markdown
        try:
            self._write_markdown(agg_df, insights, narrative, out_path / "index.md")
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

            # Use the first row as the base template
            agg_row = rows[0].copy()

            # Add stats
            agg_row["count"] = len(rows)
            agg_row["accuracy_mean"] = float(np.mean(accs))
            agg_row["accuracy_std"] = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
            agg_row["accuracy_min"] = float(np.min(accs))
            agg_row["accuracy_max"] = float(np.max(accs))

            if losses:
                agg_row["loss_mean"] = float(np.mean(losses))
                agg_row["loss_std"] = float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0
            else:
                agg_row["loss_mean"] = float("inf")

            # Remove instance-specific fields
            for k in ["id", "seed", "job_id", "fold", "accuracy", "loss"]:
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
        # Note: Visualizer expects p_values and labels.
        # We compute them here using Analyzer.
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
        Includes Global Analysis and Per-Model Analysis.
        """
        if not HAS_ML:
            return "ML Analysis libraries (scikit-learn) not installed."

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

        # --- 2. Per-Model Analysis ---
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
                # Also exclude text fields that might confuse manual encoding fallback if we didn't use DictVectorizer
                "study_name",
                "job_id",
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
                        valid = False  # Skip non-numeric hyperparams for per-model regression

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
            insights.append("**Key Drivers of Performance**:")

            imp = reg.feature_importances_
            indices = np.argsort(imp)[::-1]
            for i in indices[:3]:
                if imp[i] > 0.01:
                    insights.append(
                        f"- **{feature_keys[i]}**: {imp[i]:.2%} importance"
                    )

            insights.append(
                f"\n**Decision Rules (Tree Structure):**\n```\n{rules}\n```\n"
            )

            plt.figure(figsize=(12, 6), dpi=100)
            plot_tree(
                reg,
                feature_names=feature_keys,
                filled=True,
                rounded=True,
                precision=3,
            )
            plt.title(f"Decision Tree for {model}", fontsize=14)
            plt.savefig(img_dir / f"tree_{model}.png")
            plt.close()

        return "\n".join(insights)

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
        latex.append(r"Model & Task & Accuracy (Mean $\pm$ Std) \\")
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

        narrative.append(f"The top performing model is **{best['model']}**, achieving **{best['accuracy']:.2%}** accuracy{std_info}.")

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
                    diff = stats['mean_a'] - stats['mean_b']

                    sig_str = "statistically significant" if p < 0.05 else "not statistically significant"
                    narrative.append(f"- **{m1} vs {m2}**: {m1} outperforms by {diff:.2%}. This difference is {sig_str} (p={p:.4f}).")

        return "\n".join(narrative)

    def _write_markdown(self, data, insights, narrative, path):
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

        tasks = sorted(list(set(d["task"] for d in data)))
        for t in tasks:
            lines.append(f"#### Task: {t.upper()}")
            lines.append(f"![Leaderboard {t}](images/leaderboard_{t}.png)")

        lines.append("## 2. Experimental Progress")
        lines.append("![Tier Progress](images/tier_progress.png)")

        lines.append("## 3. Scientific Validity")
        lines.append("### Efficiency Frontier")
        lines.append("![Pareto](images/pareto_frontier.png)")
        lines.append("### Statistical Significance (P-Values)")
        lines.append("![Significance](images/significance_matrix.png)")

        lines.append("## 4. Machine Learning Analysis")
        lines.append(
            "The system trained internal models to understand what makes these "
            "algorithms work."
        )
        lines.append(insights)

        lines.append("## 5. Hyperparameter Correlations")
        lines.append("![LR Impact](images/impact_learning_rate.png)")
        lines.append("![Beta Impact](images/impact_beta.png)")

        with open(path, "w") as f:
            f.write("\n".join(lines))
