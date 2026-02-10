from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTrainer(ABC):
    """Abstract base class for trainers."""

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device

    @abstractmethod
    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of training and return metrics."""
        pass
