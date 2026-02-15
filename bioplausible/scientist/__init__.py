"""
AutoScientist: Autonomous Discovery Agent

The Scientist module manages continuous experiment execution for research automation.

Key Components:
    - AutoScientist: Main agent class running the discovery loop
    - ScientistStrategy: Decides what experiments to run next
    - ExperimentState: Tracks progress and historical results
    - ExperimentTask: Individual experiment specification
    - DecisionLogger: Records scientific decisions

Usage:
    from bioplausible.scientist import AutoScientist

    scientist = AutoScientist(db_path="bioplausible.db")
    scientist.run()  # Start continuous discovery
"""

from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.decisions import DecisionLogger
from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.strategy import ScientistStrategy
from bioplausible.scientist.task import ExperimentTask

__all__ = [
    "AutoScientist",
    "ExperimentState",
    "ScientistStrategy",
    "ExperimentTask",
    "DecisionLogger",
]
