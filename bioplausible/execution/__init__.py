"""
Execution Engine: Autonomous Discovery Execution Engine.

This module manages continuous experiment execution for research automation.
Per TODO.md:
- ExecutionEngine = execution engine (task queuing, resource management, trial running)
- AutoScientist = LLM-augmented meta-reasoner (separate module)

Key Components:
    - ExecutionEngine: Main agent class running the discovery loop
    - ExecutionStrategy: Decides what experiments to run next
    - ExperimentState: Tracks progress and historical results
    - ExperimentTask: Individual experiment specification
    - DecisionLogger: Records scientific decisions

Usage:
    from bioplausible.execution import ExecutionEngine

    engine = ExecutionEngine(db_path="bioplausible.db")
    engine.run()  # Start continuous discovery
"""

from bioplausible.execution.decisions import DecisionLogger
from bioplausible.execution.engine import ExecutionEngine
from bioplausible.execution.state import ExperimentState
from bioplausible.execution.strategy import ExecutionStrategy
from bioplausible.execution.task import ExperimentTask

__all__ = [
    "DecisionLogger",
    "ExecutionEngine",
    "ExecutionStrategy",
    "ExperimentState",
    "ExperimentTask",
]
