"""
Knowledge Package

Upgraded KnowledgeBase with SQLite + vector store (FAISS) for hybrid
structured + embedding search. Integrates surrogate models, symbolic
regression, and causal discovery.
"""

from bioplausible.knowledge.kb import (DEFAULT_KB, KnowledgeBase,
                                       KnowledgeEntry, LegacyKnowledgeBase,
                                       create_knowledge_base)
# Backward compatibility
from bioplausible.knowledge.seed import DEFAULT_KB as SEED_KB
from bioplausible.knowledge.seed import KNOWLEDGE_BASE_SEED

__all__ = [
    # New KnowledgeBase
    "KnowledgeBase",
    "KnowledgeEntry",
    "LegacyKnowledgeBase",
    "create_knowledge_base",
    "DEFAULT_KB",
    # Legacy
    "KNOWLEDGE_BASE_SEED",
    "SEED_KB",
]
