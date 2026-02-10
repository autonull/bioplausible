from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Event:
    pass


@dataclass
class ProgressEvent(Event):
    epoch: int
    metrics: Dict[str, Any]


@dataclass
class CompletedEvent(Event):
    final_metrics: Dict[str, Any]


@dataclass
class PausedEvent(Event):
    pass
