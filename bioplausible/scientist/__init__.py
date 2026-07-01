"""
Scientist: Autonomous Discovery Execution Engine.

This module manages continuous experiment execution for research automation.
Per TODO.md:
- Scientist = execution engine (task queuing, resource management, trial running)
- AutoScientist = LLM-augmented meta-reasoner (separate module)

Key Components:
    - Scientist: Main agent class running the discovery loop (alias: AutoScientist)
    - ScientistStrategy: Decides what experiments to run next
    - ExperimentState: Tracks progress and historical results
    - ExperimentTask: Individual experiment specification
    - DecisionLogger: Records scientific decisions

Usage:
    from bioplausible.scientist import Scientist

    scientist = Scientist(db_path="bioplausible.db")
    scientist.run()  # Start continuous discovery
"""

from bioplausible.scientist.core import AutoScientist, Scientist
from bioplausible.scientist.decisions import DecisionLogger
from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.strategy import ScientistStrategy
from bioplausible.scientist.task import ExperimentTask

__all__ = [
    "Scientist",
    "AutoScientist",  # Alias for backward compatibility
    "ExperimentState",
    "ScientistStrategy",
    "ExperimentTask",
    "DecisionLogger",
]
