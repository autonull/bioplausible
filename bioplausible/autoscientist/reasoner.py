"""
HypothesisReasoner: High-level meta-cognitive reasoning.

Analyses experiment history and KnowledgeBase to generate hypotheses.
Supports both rule-based reasoning and optional LLM integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bioplausible.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """A scientific hypothesis about what might work and why."""

    statement: str
    confidence: float = 0.5
    supporting_evidence: List[str] = field(default_factory=list)
    proposed_model: Optional[str] = None
    proposed_task: Optional[str] = None
    proposed_propagator: Optional[str] = None
    reasoning_chain: List[str] = field(default_factory=list)
    source: str = "rule-based"  # "rule-based" or "llm"


class HypothesisReasoner:
    """
    Generates hypotheses from experiment data and knowledge.

    Two modes:
    1. Rule-based: deterministic reasoning from known patterns
    2. LLM-augmented: uses language model for creative hypotheses (optional)
    """

    def __init__(
        self,
        knowledge_base: Optional[KnowledgeBase] = None,
        llm_backend: Optional[str] = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.llm_backend = llm_backend
        self._hypotheses: List[Hypothesis] = []

    def generate_hypotheses(
        self,
        recent_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses based on recent results and knowledge.

        Args:
            recent_results: Optional list of recent experiment metrics.

        Returns:
            List of generated hypotheses.
        """
        hypotheses = []

        # Rule 1: If a local-learning method works on vision, try it on LM
        hypotheses.extend(self._cross_domain_transfer_hypotheses(recent_results))

        # Rule 2: If bio-score is high but accuracy is low, try hybrid
        hypotheses.extend(self._bio_accuracy_tradeoff_hypotheses(recent_results))

        # Rule 3: If MEP works well, try different MEP variants
        hypotheses.extend(self._mep_variant_hypotheses(recent_results))

        self._hypotheses.extend(hypotheses)
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses

    def _cross_domain_transfer_hypotheses(
        self,
        recent_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Hypothesis]:
        """Hypothesis: transfer successful methods across domains."""
        hypotheses = []
        if not recent_results:
            return hypotheses

        # Simple rule: if a propagator works well on vision, try it on LM
        successful_propagators = set()
        for r in recent_results:
            if r.get("val_accuracy", 0) > 0.6:
                model = r.get("model", "")
                # Check if model was in vision domain
                if r.get("task") in ["mnist", "cifar10", "fashion_mnist"]:
                    successful_propagators.add(model)

        for prop in successful_propagators:
            hypotheses.append(
                Hypothesis(
                    statement=f"{prop} works on vision; should transfer to language",
                    confidence=0.6,
                    supporting_evidence=[
                        f"{prop} achieved {r.get('val_accuracy', 0):.2f} on vision"
                        for r in (recent_results or [])
                        if r.get("model") == prop
                    ],
                    proposed_model=prop,
                    proposed_task="tiny_shakespeare",
                    reasoning_chain=[
                        "Local learning rules are domain-agnostic",
                        "Success on vision suggests general-purpose credit assignment",
                    ],
                    source="rule-based",
                )
            )
        return hypotheses

    def _bio_accuracy_tradeoff_hypotheses(
        self,
        recent_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Hypothesis]:
        """Hypothesis: hybrid models balance bio-plausibility and accuracy."""
        hypotheses = []
        if not recent_results:
            return hypotheses

        for r in recent_results:
            acc = r.get("val_accuracy", 0)
            if acc < 0.5 and r.get("bio_score", 0) > 0.8:
                hypotheses.append(
                    Hypothesis(
                        statement=(
                            f"{r.get('model')} has high bio-plausibility "
                            f"but low accuracy ({acc:.2f}); "
                            "hybrid with backprop head may improve"
                        ),
                        confidence=0.7,
                        proposed_model=r.get("model"),
                        proposed_task=r.get("task"),
                        reasoning_chain=[
                            "Pure local learning may underfit complex patterns",
                            "Adding a global backprop head provides error signal",
                            "Hybrid models retain partial bio-plausibility",
                        ],
                        source="rule-based",
                    )
                )
        return hypotheses

    def _mep_variant_hypotheses(
        self,
        recent_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Hypothesis]:
        """Hypothesis: different MEP variants have different strengths."""
        from bioplausible.core.registry import ComponentCategory, Registry

        mep_propagators = Registry.query(
            category=ComponentCategory.PROPAGATOR,
            tags=["mep"],
        )

        hypotheses = []
        for p in mep_propagators:
            hypotheses.append(
                Hypothesis(
                    statement=f"Try {p['name']} on next experiment for comparison",
                    confidence=0.5,
                    proposed_propagator=p["name"],
                    reasoning_chain=[
                        (
                            f"{p['name']} has "
                            f"bio_score={p['metadata'].bio_plausibility_score}"
                        ),
                        "MEP variants offer different memory/accuracy tradeoffs",
                    ],
                    source="rule-based",
                )
            )
        return hypotheses

    def analyze_knowledge_base(self) -> List[str]:
        """Analyze KnowledgeBase for patterns and insights."""
        insights = []
        if not self.knowledge_base:
            return insights

        entries = self.knowledge_base.query(limit=50)
        if not entries:
            return insights

        # Analyze which models/propagators work best together
        model_perf = {}
        for entry in entries:
            metrics = entry.get("metrics", {})
            config = entry.get("config", {})
            model = config.get("model", "unknown")
            acc = metrics.get("val_accuracy", 0)
            if model not in model_perf:
                model_perf[model] = []
            model_perf[model].append(acc)

        for model, accs in model_perf.items():
            if accs:
                mean_acc = sum(accs) / len(accs)
                insights.append(
                    f"{model}: mean accuracy {mean_acc:.3f} across {len(accs)} runs"
                )

        return insights

    def get_top_hypotheses(self, n: int = 5) -> List[Hypothesis]:
        """Get the highest-confidence hypotheses."""
        sorted_hypotheses = sorted(
            self._hypotheses, key=lambda h: h.confidence, reverse=True
        )
        return sorted_hypotheses[:n]
