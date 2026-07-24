"""
LaTeX Report Generator.

Generates publication-quality LaTeX reports from experimental data, including
leaderboards, analysis plots, and bibliography.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

from bioplausible.core.registry import Registry

logger = logging.getLogger("LatexGenerator")


class LatexGenerator:
    """Generates Academic LaTeX reports from experiment data."""

    def generate_report(
        self, data: List[Dict[str, Any]], logs: List[Dict[str, Any]], out_path: Path
    ) -> None:
        """
        Generates a LaTeX paper with citations. Uses aggregated data.

        Args:
            data (List[Dict]): Aggregated experiment results.
            logs (List[Dict]): Decision logs.
            out_path (Path): Directory to save the LaTeX files.
        """
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
        bib_content: Set[str] = set()
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
        best_entry: Optional[Dict[str, Any]] = None
        if data:
            best_entry = max(data, key=lambda x: x["accuracy"])
            best_acc = best_entry["accuracy"]
            best_model = best_entry["model"]

        latex: List[str] = []
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
        latex.append(
            r"The following log details the autonomous decisions made by the scientist."
        )
        latex.append(r"\begin{itemize}")

        for log in logs:
            safe_desc = log["description"].replace("_", r"\_").replace("%", r"\%")
            latex.append(
                (
                    f"\\item \\textbf{{{log['date_str']}}}"
                    f" [{log['event_type']}]: {safe_desc}"
                )
            )

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
        sorted_data = sorted(data, key=lambda x: x["accuracy"], reverse=True)
        seen: Set[Tuple[str, str]] = set()
        count = 0
        for d in sorted_data:
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
            r"\caption{Top performing algorithms. Scores include"
            r" standard deviation where multiple trials exist.}"
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

        # Tier Progress
        if (out_path / "images/tier_progress.png").exists():
            latex.append(r"\begin{figure}[h]")
            latex.append(r"\centering")
            latex.append(
                r"\includegraphics[width=0.8\textwidth]{images/tier_progress.png}"
            )
            latex.append(r"\caption{Experimental Tier Progression.}")
            latex.append(r"\end{figure}")

        # Convergence
        conv_plots = list(out_path.glob("images/convergence_curves*.png"))
        if conv_plots:
            latex.append(r"\begin{figure}[h]")
            latex.append(r"\centering")
            # Just take the first one for now to avoid clutter
            latex.append(
                (
                    f"\\includegraphics[width=0.8\\textwidth]{{"
                    f"images/{conv_plots[0].name}}}"
                )
            )
            latex.append(r"\caption{Convergence Curves (Accuracy vs Epochs).}")
            latex.append(r"\end{figure}")

        # Application Recommendations
        latex.append(r"\section{Application Recommendations}")
        recommendations = self._analyze_applications(data)
        latex.append(recommendations)

        # Machine Learning Analysis (Added)
        latex.append(r"\section{Machine Learning Analysis}")
        latex.append(
            r"We utilized decision tree regression to interpret"
            r" the experimental results."
        )

        # Global Tree
        global_tree_img = "images/tree_global.png"
        if (out_path / global_tree_img).exists():
            latex.append(r"\begin{figure}[h]")
            latex.append(r"\centering")
            latex.append(
                f"\\includegraphics[width=1.0\\textwidth]{{{global_tree_img}}}"
            )
            latex.append(r"\caption{Global Decision Tree: Algorithm Comparison}")
            latex.append(r"\end{figure}")

        if best_entry:
            tree_img = f"images/tree_{best_model}.png"
            if (out_path / tree_img).exists():
                latex.append(r"\begin{figure}[h]")
                latex.append(r"\centering")
                latex.append(f"\\includegraphics[width=1.0\\textwidth]{{{tree_img}}}")
                latex.append(
                    f"\\caption{{Decision Tree for Best Model ({best_model})}}"
                )
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

    def _analyze_applications(self, data: List[Dict[str, Any]]) -> str:
        """
        Generates application recommendations based on performance profiles.

        Args:
            data (List[Dict]): Experiment data.

        Returns:
            str: LaTeX formatted recommendations.
        """
        recs = []

        # 1. Critical Systems (High Accuracy, Low Variance)
        candidates = [d for d in data if d.get("tier") in ["standard", "deep"]]

        # Fallback if tier info missing in aggregated data
        if not candidates and data:
            candidates = [d for d in data if d.get("count", 1) >= 3]

        if candidates:
            # Sort by accuracy (desc), break ties with std (asc)
            crit_cand = sorted(
                candidates,
                key=lambda x: (x["accuracy"], -x.get("accuracy_std", 1.0)),
                reverse=True,
            )
            top_crit = crit_cand[0]
            recs.append(r"\subsection{Critical Infrastructure}")
            ident = top_crit.get("config_hash", top_crit.get("id", "N/A"))
            recs.append(
                f"For safety-critical applications requiring"
                f" maximum reliability, we recommend "
                f"\\textbf{{{top_crit['model']}}} (Config Hash: {ident})."
            )
            recs.append(
                f"It achieved the highest accuracy of"
                f" {top_crit['accuracy']*100:.2f}\\% "
                f"on the {top_crit['task']} task."
            )

        # 2. Edge Deployment (High Efficiency: Acc / Params)
        edge_cand = []
        for d in data:
            if d.get("params", 0) > 0 and d["accuracy"] > 0.5:  # Min functional
                score = d["accuracy"] / (d["params"] / 1e6)  # Acc per Million Params
                edge_cand.append((d, score))

        if edge_cand:
            top_edge = sorted(edge_cand, key=lambda x: x[1], reverse=True)[0][0]
            recs.append(r"\subsection{Edge & Embedded Systems}")
            recs.append(
                f"For resource-constrained environments,"
                f" \\textbf{{{top_edge['model']}}} "
                f"demonstrates superior efficiency."
            )
            recs.append(
                f"It achieves {top_edge['accuracy']*100:.1f}\\%"
                f" accuracy with only"
                f" {top_edge['params']/1e6:.2f}M parameters,"
                f" making it ideal for mobile deployment."
            )

        # 3. Online Learning (Fast Convergence)
        recs.append(r"\subsection{Real-time Adaptation}")
        recs.append(
            r"Algorithms from the \textbf{Hebbian} and"
            r" \textbf{Predictive Coding} families "
            r"are recommended for online learning tasks"
            r" due to their local update rules,"
            r" which avoid the global locking and memory overhead"
            r" of backpropagation-through-time."
        )

        return "\n".join(recs)
