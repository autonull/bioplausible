"""
AutoScientist: The LLM-augmented meta-cognitive layer.

Ingests experiment logs + KnowledgeBase, proposes novel hypotheses,
performs high-level reasoning, symbolic analysis, and generates
intelligent experiment batches.

Distinct from Scientist (execution engine):
  - Scientist executes experiments reliably.
  - AutoScientist decides *what* to execute and *why*.
"""

from bioplausible.autoscientist.bridge import AutoScientistBridge
from bioplausible.autoscientist.bridge import ExperimentProposal
from bioplausible.autoscientist.campaign import AutoScientistCampaign
from bioplausible.autoscientist.proposer import ExperimentProposer
from bioplausible.autoscientist.reasoner import Hypothesis
from bioplausible.autoscientist.reasoner import HypothesisReasoner
from bioplausible.autoscientist.reasoner import LLMHypothesisGenerator

__all__ = [
    "AutoScientistCampaign",
    "ExperimentProposer",
    "HypothesisReasoner",
    "LLMHypothesisGenerator",
    "AutoScientistBridge",
    "ExperimentProposal",
    "Hypothesis",
]
