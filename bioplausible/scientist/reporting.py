"""
ScientistReporter: Generates publication-quality reports from experiment data.
"""

import logging
import os
import shutil
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

# Headless mode must be set before pyplot import
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bioplausible.hyperopt.storage import HyperoptStorage  # noqa: E402
from bioplausible.models.registry import get_model_spec  # noqa: E402
from bioplausible.statistics import StatisticalAnalyzer  # noqa: E402
from bioplausible.visualization import ResultVisualizer  # noqa: E402

# ML Imports
try:
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

        # 4. Generate Narrative (Explainability)
        narrative = self._generate_narrative(df)

        # Export Best Config
        try:
            self._export_best_config(df, out_path)
        except Exception as e:
            logger.error(f"Failed to export best config: {e}")

        # 5. Write Markdown
        try:
            self._write_markdown(df, insights, narrative, out_path / "index.md")
        except Exception as e:
            logger.error(f"Failed to write markdown report: {e}")

        # 5. Write LaTeX (Academic)
        try:
            self._generate_latex_report(df, out_path)
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
            insights.append("**Key Drivers of Performance**:")

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
        """Generates a LaTeX paper with citations."""
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

        latex.append(r"\section{Results}")

        # Leaderboard Table
        latex.append(r"\subsection{Leaderboard}")
        latex.append(r"\begin{table}[h]")
        latex.append(r"\centering")
        latex.append(r"\begin{tabular}{l c c}")
        latex.append(r"\toprule")
        latex.append(r"Model & Task & Accuracy \\")
        latex.append(r"\midrule")

        # Top 5 models
        data.sort(key=lambda x: x["accuracy"], reverse=True)
        seen = set()
        count = 0
        for d in data:
            key = (d["model"], d["task"])
            if key not in seen:
                latex.append(
                    f"{d['model']} & {d['task']} & {d['accuracy']*100:.2f}\\% \\\\"
                )
                seen.add(key)
                count += 1
                if count >= 10:
                    break

        latex.append(r"\bottomrule")
        latex.append(r"\end{tabular}")
        latex.append(r"\caption{Top performing algorithms discovered by the system.}")
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
        latex.append(r"We trained decision trees to identify key performance drivers.")

        if best_entry:
            tree_img = f"images/tree_{best_model}.png"
            if (out_path / tree_img).exists():
                latex.append(r"\begin{figure}[h]")
                latex.append(r"\centering")
                latex.append(f"\\includegraphics[width=1.0\\textwidth]{{{tree_img}}}")
                latex.append(f"\\caption{{Decision Tree for {best_model}}}")
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

    def _generate_narrative(self, data) -> str:
        """Generates plain English narrative explaining the results."""
        models = sorted(list(set(d["model"] for d in data)))
        valid_data = [d for d in data if d.get("tier") in ["standard", "deep"]]

        narrative = []

        # 1. Overall Winner
        if not data:
            return "No data available."

        best = max(data, key=lambda x: x["accuracy"])
        narrative.append(f"The top performing model is **{best['model']}**, achieving **{best['accuracy']:.2%}** accuracy.")

        # 2. Pairwise Comparisons (Significance)
        if len(models) > 1:
            narrative.append("\n### Key Comparisons")

            # Compare top 2 distinct models
            model_scores = {}
            for m in models:
                scores = [d["accuracy"] for d in valid_data if d["model"] == m]
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
