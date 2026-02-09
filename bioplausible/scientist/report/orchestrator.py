"""
Report Orchestrator

Coordinaties the generation of comprehensive Scientist++ reports:
1. Research Synthesis (High-level insights, cross-algorithm analysis)
2. Modular Reporting (Visualizations, Leaderboards, detailed ML analysis)
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Optional

from bioplausible.scientist.synthesizer import ResearchSynthesizer
from bioplausible.scientist.report.composer import ReportComposer

logger = logging.getLogger("ReportOrchestrator")

class ReportOrchestrator:
    def __init__(self, db_path: str, output_dir: str = "reports"):
        self.db_path = db_path
        self.output_base_dir = output_dir

    def generate_reports(self):
        """
        Generates comprehensive Scientist++ reports.
        """
        logger.info("Generating Scientist++ Reports...")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_path = Path(self.output_base_dir) / f"run_{timestamp}"
        report_path.mkdir(parents=True, exist_ok=True)

        # 1. Generate high-level synthesis insights
        self._generate_synthesis(report_path, timestamp)

        # 2. Generate comprehensive report using Modular ReportComposer
        self._generate_modular_report(report_path)

        logger.info(f"\n{'='*60}")
        logger.info(f"Reports saved to: {report_path}")
        logger.info(f"  - FULL_REPORT.md: Main comprehensive report (includes Synthesis)")
        logger.info(f"  - images/: Visualizations and ML analysis")
        logger.info(f"  - synthesis/: Detailed research logs")
        logger.info(f"{'='*60}\n")

    def _generate_synthesis(self, report_path: Path, timestamp: str):
        logger.info("Generating research synthesis...")
        try:
            synthesizer = ResearchSynthesizer(self.db_path)
            synthesis_result = synthesizer.synthesize_full_report()

            # Create synthesis subdirectory
            synthesis_path = report_path / "synthesis"
            synthesis_path.mkdir(exist_ok=True)

            # Save Synthesis JSON
            with open(synthesis_path / "research_synthesis.json", "w") as f:
                json.dump(synthesis_result, f, indent=2)

            # Generate Synthesis Narrative
            self._write_synthesis_markdown(synthesis_path / "SYNTHESIS.md", synthesis_result, timestamp)

            logger.info("✓ Research synthesis generated (synthesis/)")
        except Exception as e:
            logger.error(f"Failed to generate synthesis: {e}", exc_info=True)

    def _write_synthesis_markdown(self, path: Path, synthesis_result: dict, timestamp: str):
        with open(path, "w") as f:
            f.write(f"# Research Synthesis\n")
            f.write(f"Generated: {timestamp}\n\n")

            # Cross-Algorithm Rankings
            f.write("## 🏆 Cross-Algorithm Performance Rankings\n\n")
            insights = synthesis_result.get("cross_algorithm_insights", {})
            if isinstance(insights, dict) and "rankings" in insights:
                f.write("| Rank | Model | Best Acc | Mean Acc | Std Dev | Trials |\n")
                f.write("|------|-------|----------|----------|---------|--------|\n")
                for i, r in enumerate(insights["rankings"][:10], 1):
                    f.write(
                        f"| {i} | {r['model']} | {r['best_accuracy']:.2%} | {r['mean_accuracy']:.2%} | {r['std']:.4f} | {r['trials']} |\n")
                f.write("\n")
            else:
                f.write(f"{insights}\n\n")

            # Statistical Significance
            sig = synthesis_result.get("statistical_significance", [])
            if sig and isinstance(sig, list) and len(sig) > 0 and isinstance(sig[0], dict):
                f.write("## 📏 Statistical Significance\n\n")
                f.write("| Winner | Loser | Mean Diff | P-Value | Confidence |\n")
                f.write("|--------|-------|-----------|---------|------------|\n")
                for s in sig[:10]:
                    f.write(f"| **{s['winner']}** | {s['loser']} | +{s['mean_diff']:.2%} | {s['p_value']:.4f} | {s['confidence']} |\n")
                f.write("\n")

            # Ablation Analysis
            ablations = synthesis_result.get("ablation_analysis", [])
            if ablations and isinstance(ablations, list):
                f.write("## 🔧 Ablation Studies (Mechanistic Insights)\n\n")
                f.write("| Model | Ablation | Value | Delta | Result |\n")
                f.write("|-------|----------|-------|-------|--------|\n")
                for a in ablations:
                    icon = "🟢" if a['delta'] > -0.01 else "🔴"
                    f.write(f"| {a['model']} | {a['ablation_param']} | {a['ablation_value']} | {a['delta']:+.2%} | {icon} |\n")
                f.write("\n")

            # Task-Specific Winners
            f.write("## 📊 Task-Specific Winners\n\n")
            task_winners = synthesis_result.get("task_specific_winners", {})
            if isinstance(task_winners, dict):
                for task, winners in task_winners.items():
                    f.write(f"### {task.replace('_', ' ').title()}\n")
                    for i, w in enumerate(winners, 1):
                        f.write(
                            f"{i}. **{w['model']}**: {w['accuracy']:.2%} ({w['params']:,} params)\n")
                    f.write("\n")

            # Efficiency Analysis
            f.write("## ⚡ Efficiency Analysis\n\n")
            efficiency = synthesis_result.get("efficiency_analysis", {})

            if "top_epoch_efficient" in efficiency:
                f.write("### Top Models by Epoch Efficiency (Accuracy / Epoch)\n")
                f.write(
                    "*Models that converge fastest - high accuracy with fewer epochs.*\n\n")
                f.write("| Model | Task | Accuracy | Epochs | Acc/Epoch |\n")
                f.write("|-------|------|----------|--------|----------|\n")
                for r in efficiency["top_epoch_efficient"][:5]:
                    eff = r['epoch_efficiency']
                    f.write(
                        f"| {r['model_name']} | {r['task_name']} | {r['accuracy']:.2%} | {r['num_epochs']} | {eff:.4f} |\n")
                f.write("\n")

            if "top_param_efficient" in efficiency:
                f.write("### Top Models by Parameter Efficiency (Accuracy / M-Params)\n")
                f.write("*Models that achieve high performance with fewer parameters.*\n\n")
                for r in efficiency["top_param_efficient"][:5]:
                    params_m = r['param_count'] / 1e6
                    f.write(
                        f"- **{r['model_name']}**: {r['accuracy']:.2%} with {params_m:.2f}M params (efficiency: {r['param_efficiency']:.2f})\n")
                f.write("\n")

            f.write("## ⚠️ Failure Analysis\n")
            fails = synthesis_result.get("failure_analysis", {})
            if isinstance(fails, dict):
                if "patterns" in fails and fails["patterns"]:
                    f.write("\n**Detected Patterns:**\n")
                    for p in fails["patterns"]:
                        f.write(f"- {p}\n")
                    f.write("\n")
                if "counts" in fails:
                    f.write("\n**Failure Counts:**\n")
                    for k, v in fails["counts"].items():
                        f.write(f"- **{k}**: {v} failures\n")
            else:
                f.write(f"{fails}\n\n")

            f.write("\n## 💡 Quick Wins & Suggestions\n\n")
            wins = synthesis_result.get("quick_wins", [])
            if isinstance(wins, list) and wins:
                for win in wins:
                    f.write(f"- {win}\n")
            else:
                f.write("All systems running smoothly. Continue exploration.\n")

            f.write("\n## 🔬 Research Gaps\n\n")
            gaps = synthesis_result.get("research_gaps", [])
            if isinstance(gaps, list) and gaps:
                for gap in gaps:
                    f.write(f"- {gap}\n")
            else:
                f.write("No major research gaps identified.\n")

    def _generate_modular_report(self, report_path: Path):
        logger.info("Generating modular analysis report...")
        try:
            with ReportComposer(self.db_path, str(report_path)) as composer:
                composer.generate_report()

            logger.info(
                "✓ Modular report generated (01_summary.md, 03_leaderboards.md, FULL_REPORT.md)")
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}", exc_info=True)
