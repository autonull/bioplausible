"""
HypothesisReasoner: High-level meta-cognitive reasoning.

Analyses experiment history and KnowledgeBase to generate hypotheses.
Supports both rule-based reasoning and optional LLM integration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from bioplausible.knowledge import KnowledgeBase
from bioplausible.knowledge import KnowledgeEntry

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

        # Rule-based hypotheses
        hypotheses.extend(self._cross_domain_transfer_hypotheses(recent_results))
        hypotheses.extend(self._bio_accuracy_tradeoff_hypotheses(recent_results))
        hypotheses.extend(self._mep_variant_hypotheses(recent_results))

        # LLM-augmented hypotheses (if enabled)
        if self.llm_backend:
            try:
                hypotheses.extend(self._llm_hypotheses(recent_results))
            except Exception as e:
                logger.warning(f"LLM hypothesis generation failed: {e}")

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

        successful_propagators = set()
        for r in recent_results:
            if r.get("val_accuracy", 0) > 0.6:
                model = r.get("model", "")
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
        from bioplausible.core.registry import ComponentCategory
        from bioplausible.core.registry import Registry

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
                        f"{p['name']} has "
                        f"bio_score={p['metadata'].bio_plausibility_score}",
                        "MEP variants offer different memory/accuracy tradeoffs",
                    ],
                    source="rule-based",
                )
            )
        return hypotheses

    def _llm_hypotheses(
        self,
        recent_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using LLM (optional, local-first).

        Uses the LLM to suggest novel experiment ideas based on
        knowledge base patterns and recent results.
        """
        hypotheses = []

        insights = self.analyze_knowledge_base() if self.knowledge_base else []

        context = []
        if recent_results:
            context.append("Recent Results:")
            for r in recent_results[:5]:
                context.append(
                    f"  - {r.get('model', 'unknown')} on "
                    f"{r.get('task', 'unknown')}: {r.get('val_accuracy', 0):.2f}"
                )
        if insights:
            context.append("Knowledge Base Insights:")
            for i in insights[:5]:
                context.append(f"  - {i}")

        prompt = "\n".join(context)
        if not prompt:
            return hypotheses

        try:
            generator = LLMHypothesisGenerator(backend=self.llm_backend)
            llm_hypotheses = generator.generate(prompt)
            hypotheses.extend(llm_hypotheses)
        except Exception as e:
            logger.warning(f"Could not initialize LLM backend: {e}")

        return hypotheses

    def analyze_knowledge_base(self) -> List[str]:
        """Analyze KnowledgeBase for patterns and insights."""
        insights = []
        if not self.knowledge_base:
            return insights

        entries = self.knowledge_base.query(limit=50)
        if not entries:
            return insights

        model_perf = {}
        for entry in entries:
            if isinstance(entry, KnowledgeEntry):
                metrics = entry.metrics or {}
                model = entry.model_family or "unknown"
            else:
                metrics = entry.get("metrics", {})
                config = entry.get("config", {})
                model = (
                    config.get("model", "unknown")
                    if isinstance(metrics, dict)
                    else "unknown"
                )
            acc = metrics.get("val_accuracy", 0) if isinstance(metrics, dict) else 0
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


class LLMHypothesisGenerator:
    """
    Optional LLM-powered hypothesis generator for novel experiment ideas.

    Supports local-first backends:
    - 'openai': OpenAI API (requires API key)
    - 'local': Local model via llama.cpp or similar
    """

    def __init__(self, backend: str = "openai", api_key: Optional[str] = None):
        self.backend = backend
        self.api_key = api_key

    def generate(self, context: str) -> List[Hypothesis]:
        """
        Generate hypotheses from context using LLM.

        Args:
            context: Text context with experiment results and insights.

        Returns:
            List of hypotheses suggested by the LLM.
        """
        if self.backend == "openai" and self.api_key:
            return self._generate_openai(context)
        return self._fallback_hypotheses(context)

    def _generate_openai(self, context: str) -> List[Hypothesis]:
        """Generate using OpenAI API."""
        try:
            from openai import OpenAI

            client = OpenAI(api_key=self.api_key)

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a scientific research assistant. "
                            "Suggest novel experiments based on the "
                            "provided context. Return JSON with hypothesis "
                            "statements and confidence scores."
                        ),
                    },
                    {"role": "user", "content": context},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            if content:
                data = json.loads(content)
                hypotheses = []
                for item in data.get("hypotheses", []):
                    hypotheses.append(
                        Hypothesis(
                            statement=item.get("statement", ""),
                            confidence=item.get("confidence", 0.5),
                            proposed_model=item.get("model"),
                            proposed_task=item.get("task"),
                            proposed_propagator=item.get("propagator"),
                            reasoning_chain=item.get("reasoning", []),
                            source="llm",
                        )
                    )
                return hypotheses
        except Exception as e:
            logger.warning(f"OpenAI hypothesis generation failed: {e}")

        return []

    def _fallback_hypotheses(self, context: str) -> List[Hypothesis]:
        """Fallback when LLM is unavailable."""
        return [
            Hypothesis(
                statement="Try alternative architectures on underexplored tasks",
                confidence=0.3,
                source="rule-based",
                reasoning_chain=["LLM backend unavailable, using fallback heuristic"],
            ),
        ]


# Re-export for backward compatibility
__all__ = [
    "Hypothesis",
    "HypothesisReasoner",
    "LLMHypothesisGenerator",
]
