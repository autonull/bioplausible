from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TrackResult:
    """Result of a verification track."""

    track_id: int
    name: str
    status: str  # 'pass', 'fail', 'partial', 'stub'
    score: float  # 0-100
    metrics: Dict
    evidence: str  # Markdown evidence block
    time_seconds: float
    improvements: List[str] = field(default_factory=list)
    evidence_level: str = "smoke"  # 'smoke', 'directional', 'conclusive'
    limitations: List[str] = field(default_factory=list)
    reproducibility_hash: Optional[str] = None


class VerificationNotebook:
    """Generates a comprehensive markdown evidence notebook."""

    def __init__(self, title: str = "TorEqProp Verification Results"):
        self.title = title
        self.sections: List[str] = []
        self.start_time = datetime.now()
        self.track_results: List[TrackResult] = []

    def add_header(self, seed: int = 42):
        """Add title and metadata."""
        self.sections.append(f"# {self.title}\n")
        self.sections.append(
            f"**Generated**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self.sections.append(f"**Seed**: {seed} (deterministic)\n")
        self.sections.append(
            "**Reproducibility**: All experiments use fixed seeds for exact reproduction.\n"
        )
        self.sections.append("---\n")

    def add_section(self, title: str, content: str):
        self.sections.append(f"\n## {title}\n\n{content}\n")

    def add_subsection(self, title: str, content: str):
        self.sections.append(f"\n### {title}\n\n{content}\n")

    def add_table(self, headers: List[str], rows: List[List[str]]):
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        data_rows = "\n".join(
            "| " + " | ".join(str(c) for c in row) + " |" for row in rows
        )
        self.sections.append(f"\n{header_row}\n{separator}\n{data_rows}\n")

    def add_chart(self, title: str, data: Dict[str, float], max_width: int = 40):
        if not data:
            return
        max_val = max(abs(v) for v in data.values()) or 1
        scale = max_width / max_val

        lines = [f"\n**{title}**\n```"]
        max_label = max(len(str(k)) for k in data.keys())

        for label, value in data.items():
            bar_len = int(abs(value) * scale)
            bar = "█" * bar_len
            lines.append(f"{str(label):<{max_label}} │ {bar} {value:.3f}")

        lines.append("```\n")
        self.sections.append("\n".join(lines))

    def add_code_block(self, code: str, lang: str = ""):
        self.sections.append(f"\n```{lang}\n{code}\n```\n")

    def add_track_result(self, result: TrackResult):
        """Add a track result to the notebook."""
        self.track_results.append(result)

        status_icon = {"pass": "✅", "fail": "❌", "partial": "⚠️", "stub": "🔧"}.get(
            result.status, "❓"
        )
        evidence_icon = {"smoke": "🧪", "directional": "📊", "conclusive": "✅"}.get(
            result.evidence_level, "❓"
        )
        evidence_label = {
            "smoke": "Smoke Test",
            "directional": "Directional",
            "conclusive": "Conclusive",
        }.get(result.evidence_level, "Unknown")

        content = f"""
{status_icon} **Status**: {result.status.upper()} | **Score**: {result.score:.1f}/100 | **Time**: {result.time_seconds:.1f}s

{evidence_icon} **Evidence Level**: {evidence_label}

{result.evidence}
"""

        # Add limitations if any
        if result.limitations:
            content += "\n**Limitations**:\n"
            for lim in result.limitations:
                content += f"- {lim}\n"

        # Add reproducibility hash if available
        if result.reproducibility_hash:
            content += f"\n*Reproducibility Hash*: `{result.reproducibility_hash}`\n"

        self.add_section(f"Track {result.track_id}: {result.name}", content)

        # Add improvements if any
        if result.improvements:
            improvements_md = "\n".join(f"- {imp}" for imp in result.improvements)
            self.add_subsection("Areas for Improvement", improvements_md)

    def add_executive_summary(self):
        """Add executive summary based on all track results."""
        total = len(self.track_results)
        passed = sum(1 for r in self.track_results if r.status == "pass")
        partial = sum(1 for r in self.track_results if r.status == "partial")
        failed = sum(1 for r in self.track_results if r.status == "fail")
        stubs = sum(1 for r in self.track_results if r.status == "stub")

        avg_score = (
            np.mean([r.score for r in self.track_results]) if self.track_results else 0
        )
        total_time = sum(r.time_seconds for r in self.track_results)

        summary = f"""
## Executive Summary

**Verification completed in {total_time:.1f} seconds.**

### Overall Results

| Metric | Value |
|--------|-------|
| Tracks Verified | {total} |
| Passed | {passed} ✅ |
| Partial | {partial} ⚠️ |
| Failed | {failed} ❌ |
| Stubs (TODO) | {stubs} 🔧 |
| Average Score | {avg_score:.1f}/100 |

### Track Summary

| # | Track | Status | Score | Time |
|---|-------|--------|-------|------|
"""
        for r in self.track_results:
            icon = {"pass": "✅", "fail": "❌", "partial": "⚠️", "stub": "🔧"}.get(
                r.status, "❓"
            )
            summary += f"| {r.track_id} | {r.name} | {icon} | {r.score:.0f} | {r.time_seconds:.1f}s |\n"

        summary += "\n"

        # Insert at position 2 (after header)
        self.sections.insert(2, summary)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.add_executive_summary()
        with open(path, "w") as f:
            f.write("\n".join(self.sections))
        print(f"📓 Notebook saved to: {path}")


class ValidationTrack:
    """Base class for validation tracks to ensure consistent interface."""

    def __init__(
        self,
        name: str,
        track_id: int,
        description: str,
        category: str = "core",
        priority: str = "medium",
        tags: List[str] = None,
    ):
        self.name = name
        self.track_id = track_id
        self.description = description
        self.category = category
        self.priority = priority
        self.tags = tags or []

    def validate(self) -> Dict[str, Any]:
        """
        Execute the validation logic.

        Returns:
            Dict containing validation results. Should generally include:
            - success (bool)
            - score (0-100 float)
            - metrics (dict)
            - evidence (str)
            - improvements (list of str)
        """
        raise NotImplementedError("Subclasses must implement validate()")

    def __call__(self, verifier) -> TrackResult:
        """
        Callable interface compatible with the Verifier.
        """
        import time

        start_time = time.time()

        try:
            # Execute validation
            # Note: We might need 'verifier' in validate() if tracks depend on global config
            # For now, we assume tracks are self-contained or use global settings
            # If tracks need verifier props, we could pass it or set it on self
            self.verifier = verifier

            result_data = self.validate()

            elapsed = time.time() - start_time

            # Construct standard TrackResult
            # Handle minimal return format if just {'success': ...}

            status = "pass" if result_data.get("success", False) else "fail"
            score = result_data.get("score", 100.0 if status == "pass" else 0.0)
            metrics = result_data.get("metrics", {})
            evidence = result_data.get("evidence", "No evidence provided.")

            # If 'details' or custom fields are present, append to evidence
            if "details" in result_data:
                import json

                # Handle non-serializable objects in details if needed, or just str()
                evidence += f"\n\n**Details**:\n{str(result_data['details'])}"

            return TrackResult(
                track_id=self.track_id,
                name=self.name,
                status=status,
                score=score,
                metrics=metrics,
                evidence=evidence,
                time_seconds=elapsed,
                improvements=result_data.get("improvements", []),
                evidence_level=result_data.get("evidence_level", "directional"),
            )

        except Exception as e:
            import traceback

            traceback.print_exc()
            return TrackResult(
                track_id=self.track_id,
                name=self.name,
                status="fail",
                score=0.0,
                metrics={"error": str(e)},
                evidence=f"**Error**: {str(e)}\n\n```\n{traceback.format_exc()}\n```",
                time_seconds=time.time() - start_time,
            )
