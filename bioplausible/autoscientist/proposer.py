"""
ExperimentProposer: Generates intelligent experiment batches.

Takes hypotheses from HypothesisReasoner and converts them into
concrete experiment proposals with configurations.
"""

from __future__ import annotations

import logging
from typing import Any

from bioplausible.autoscientist.bridge import AutoScientistBridge, ExperimentProposal
from bioplausible.autoscientist.reasoner import Hypothesis, HypothesisReasoner
from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


class ExperimentProposer:
    """
    Generates experiment batches from hypotheses.

    Supports:
    - Systematic search across model+propagator combinations
    - Targeted experiments based on specific hypotheses
    - Ablation studies (vary one parameter at a time)
    - Curriculum-based progression (easy tasks first)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        reasoner: HypothesisReasoner | None = None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.reasoner = reasoner or HypothesisReasoner(self.knowledge_base)
        self.bridge = AutoScientistBridge()

    def propose_batch(
        self,
        domain: str | None = None,
        n_proposals: int = 10,
        min_bio_score: float = 0.0,
    ) -> list[ExperimentProposal]:
        """
        Propose a batch of experiments.

        Combines systematic exploration with hypothesis-driven targeting.

        Args:
            domain: Optional domain filter.
            n_proposals: Number of proposals to generate.
            min_bio_score: Minimum bio-plausibility score.

        Returns:
            List of experiment proposals.
        """
        proposals = []

        # 1. Generate hypotheses
        hypotheses = self.reasoner.generate_hypotheses()

        # 2. Convert hypotheses to proposals
        for h in hypotheses[: n_proposals // 2]:
            proposal = self._hypothesis_to_proposal(h)
            if proposal:
                proposals.append(proposal)

        # 3. Fill remaining slots with systematic combinations
        remaining = n_proposals - len(proposals)
        if remaining > 0:
            systematic = self._systematic_proposals(domain, remaining, min_bio_score)
            proposals.extend(systematic)

        h_count = len(hypotheses)
        s_count = len(proposals) - h_count
        logger.info(
            "Proposed %d experiments (%d hypothesis-driven, %d systematic)",
            len(proposals),
            h_count,
            s_count,
        )
        return proposals

    def _hypothesis_to_proposal(
        self, hypothesis: Hypothesis
    ) -> ExperimentProposal | None:
        """Convert a hypothesis to an experiment proposal."""
        if not hypothesis.proposed_model and not hypothesis.proposed_propagator:
            return None

        return ExperimentProposal(
            hypothesis=hypothesis.statement,
            model=hypothesis.proposed_model or "MLP",
            task=hypothesis.proposed_task or "mnist",
            propagator=hypothesis.proposed_propagator,
            justification=(
                hypothesis.reasoning_chain[0] if hypothesis.reasoning_chain else ""
            ),
            expected_outcome=hypothesis.statement,
            priority=hypothesis.confidence,
            tags=["autoscientist", hypothesis.source],
        )

    def _systematic_proposals(
        self,
        domain: str | None = None,
        n: int = 5,
        min_bio_score: float = 0.0,
    ) -> list[ExperimentProposal]:
        """Generate systematic exploration proposals."""
        # Get models and propagators
        models = Registry.query(
            category=ComponentCategory.MODEL,
            min_bio_score=min_bio_score,
        )
        propagators = Registry.query(
            category=ComponentCategory.PROPAGATOR,
            min_bio_score=min_bio_score,
        )

        proposals = []
        for i in range(min(n, len(models) * len(propagators))):
            m_idx = i % len(models)
            p_idx = (i // len(models)) % len(propagators) if propagators else 0
            model = models[m_idx]
            propagator = propagators[p_idx] if propagators else None

            proposals.append(
                ExperimentProposal(
                    hypothesis=(
                        "Systematic exploration of model-propagator combinations"
                    ),
                    model=model["name"],
                    task="mnist",
                    propagator=propagator["name"] if propagator else None,
                    justification=(
                        f"Testing {model['name']} with "
                        f"{propagator['name'] if propagator else 'default'}: "
                        f"bio_score={model['metadata'].bio_plausibility_score}"
                    ),
                    priority=0.3,
                    tags=["systematic", "exploration"],
                )
            )

        return proposals

    def propose_ablation(
        self,
        model: str,
        base_config: dict[str, Any],
        parameters: list[str],
        values: list[list[Any]],
    ) -> list[ExperimentProposal]:
        """
        Propose ablation studies varying specific parameters.

        Args:
            model: Model name to ablate.
            base_config: Base configuration.
            parameters: Parameter names to vary.
            values: Values to try for each parameter.

        Returns:
            List of ablation proposals.
        """
        proposals = []
        for param, vals in zip(parameters, values):
            for v in vals:
                config = dict(base_config)
                config[param] = v
                proposals.append(
                    ExperimentProposal(
                        hypothesis=f"Ablation: effect of {param}={v} on {model}",
                        model=model,
                        task=base_config.get("task", "mnist"),
                        hyperparams={param: v},
                        priority=0.4,
                        tags=["ablation", param],
                    )
                )
        return proposals
