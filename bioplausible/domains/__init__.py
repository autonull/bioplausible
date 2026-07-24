"""
Domains Package

Domain abstraction layer with standard interfaces for vision, LM, RL, graph,
tabular, time series, and scientific simulation.
"""

from bioplausible.domains.base import (
    Batch,
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)
from bioplausible.domains.graph import GraphTask
from bioplausible.domains.lm import LMTask
from bioplausible.domains.rl import RLTask
from bioplausible.domains.scientific import ScientificTask
from bioplausible.domains.tabular import TabularTask
from bioplausible.domains.timeseries import TimeSeriesTask
from bioplausible.domains.vision import VisionTask

# Registry for domain tasks
_DOMAIN_REGISTRY = {
    "vision": VisionTask,
    "lm": LMTask,
    "rl": RLTask,
    "graph": GraphTask,
    "tabular": TabularTask,
    "timeseries": TimeSeriesTask,
    "scientific": ScientificTask,
}


def create_domain_task(domain: str, name: str, **kwargs) -> DomainTask:
    """Create a domain task by name."""
    if domain not in _DOMAIN_REGISTRY:
        raise ValueError(
            f"Unknown domain: {domain}. Available: {list(_DOMAIN_REGISTRY.keys())}"
        )

    task_class = _DOMAIN_REGISTRY[domain]
    return task_class(name=name, **kwargs)


def register_domain_task(domain: str, task_class: type) -> None:
    """Register a new domain task class."""
    _DOMAIN_REGISTRY[domain] = task_class


def list_domains() -> list:
    """List available domains."""
    return list(_DOMAIN_REGISTRY.keys())


__all__ = [
    # Base classes
    "DomainTask",
    "DomainType",
    "DomainSpec",
    "TaskSplit",
    "Batch",
    "Metrics",
    # Concrete tasks
    "VisionTask",
    "LMTask",
    "RLTask",
    "GraphTask",
    "TabularTask",
    "TimeSeriesTask",
    "ScientificTask",
    # Factory
    "create_domain_task",
    "register_domain_task",
    "list_domains",
]
