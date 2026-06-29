import os
from typing import Any, Dict, List

import pandas as pd

from bioplausible.scientist.failure_tracker import FailureCategory, FailureTracker


class FailureManifestoGenerator:
    """
    Auto-generates reports/failure_manifesto.md from experiment DB.
    """

    def __init__(self, db_path: str):
        self.tracker = FailureTracker(db_path)

    def generate(self, output_path: str = "reports/failure_manifesto.md"):
        """
        Extracts failures from DB and groups them by algorithm and FailureCategory.
        Outputs a markdown manifesto report.
        """
        stats = self.tracker.get_failure_stats()
        recent_failures = self.tracker.get_recent_failures(limit=1000)

        # Build DataFrame for easier cross-tabulation
        fail_data = []
        for r in recent_failures:
            fail_data.append(
                {
                    "model": r.model_name,
                    "task": r.task_name,
                    "type": r.failure_type,
                    "epoch": r.failure_epoch,
                }
            )

        df = pd.DataFrame(fail_data)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Failure Modes Manifesto\n\n")
            f.write(
                "This document tracks the explicit failure modes encountered across different bioplausible algorithms.\n\n"
            )

            if df.empty:
                f.write("No failures logged yet.\n")
                return

            f.write("## Overall Failure Distribution\n\n")
            type_counts = df["type"].value_counts()
            f.write("| Failure Type | Count |\n")
            f.write("|--------------|-------|\n")
            for t, c in type_counts.items():
                f.write(f"| `{t}` | {c} |\n")
            f.write("\n")

            f.write("## Failures by Model and Type\n\n")

            cross_tab = pd.crosstab(df["model"], df["type"])

            # Format crosstab as markdown table
            cols = ["Model"] + list(cross_tab.columns)
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")

            for index, row in cross_tab.iterrows():
                row_vals = [str(index)] + [str(v) for v in row.values]
                f.write("| " + " | ".join(row_vals) + " |\n")
            f.write("\n")

            f.write("## Advanced Diagnostics\n\n")
            analysis = self.tracker.analyze_failure_patterns()

            if not analysis.get("recommendations"):
                f.write(
                    "No critical failure patterns detected requiring immediate intervention.\n"
                )
            else:
                for rec in analysis["recommendations"]:
                    sev = rec.get("severity", "info")
                    f.write(
                        f"### [Severity: {sev.upper()}] {rec.get('issue', 'Unknown Issue')}\n"
                    )
                    f.write(f"- **Recommendation**: {rec.get('suggestion')}\n")
                    if "affected_models" in rec:
                        f.write(
                            f"- **Affected Models**: {', '.join(rec['affected_models'])}\n"
                        )
                    if "details" in rec:
                        f.write(f"- **Details**: {rec['details']}\n")
                    f.write("\n")

        return output_path
