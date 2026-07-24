"""
Base classes for Domain Abstraction Layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class DomainType(str, Enum):
    """Supported domain types."""

    VISION = "vision"
    LM = "lm"
    RL = "rl"
    GRAPH = "graph"
    TIMESERIES = "timeseries"
    TABULAR = "tabular"
    SCIENTIFIC = "scientific"
    CONTINUAL = "continual"
    MULTITASK = "multitask"
    CUSTOM = "custom"


class TaskSplit(str, Enum):
    """Data splits."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    TRAIN_VAL = "train_val"


@dataclass
class DomainSpec:
    """Specification for a domain."""

    name: str
    domain_type: DomainType
    description: str = ""
    default_metrics: list[str] = field(default_factory=list)
    typical_input_shape: tuple[int, ...] | None = None
    typical_output_shape: tuple[int, ...] | None = None
    supported_tasks: list[str] = field(default_factory=list)
    default_batch_size: int = 32
    default_lr: float = 1e-3
    requires_sequence: bool = False
    requires_spatial: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class Batch:
    """Standard batch format."""

    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> Batch:
        """Move batch to device."""
        return Batch(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            metadata=self.metadata,
        )

    @property
    def batch_size(self) -> int:
        return self.inputs.shape[0]


@dataclass
class Metrics:
    """Standardized metrics output."""

    loss: float
    accuracy: float | None = None
    perplexity: float | None = None
    custom: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        d = {"loss": self.loss}
        if self.accuracy is not None:
            d["accuracy"] = self.accuracy
        if self.perplexity is not None:
            d["perplexity"] = self.perplexity
        d.update(self.custom)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> Metrics:
        return cls(
            loss=d.get("loss", 0.0),
            accuracy=d.get("accuracy"),
            perplexity=d.get("perplexity"),
            custom={
                k: v
                for k, v in d.items()
                if k not in {"loss", "accuracy", "perplexity"}
            },
        )


class DomainTask(ABC):
    """
    Abstract base class for domain tasks.

    Each domain (vision, LM, RL, etc.) implements this interface
    to provide standardized data loading, evaluation, and metrics.
    """

    def __init__(
        self,
        name: str,
        device: str | torch.device = "cpu",
        batch_size: int = 32,
        **kwargs,
    ):
        self.name = name
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.kwargs = kwargs
        self._train_loader: DataLoader | None = None
        self._val_loader: DataLoader | None = None
        self._test_loader: DataLoader | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._setup_done = False

    @property
    @abstractmethod
    def domain_type(self) -> DomainType:
        """Return the domain type."""

    @property
    @abstractmethod
    def spec(self) -> DomainSpec:
        """Return domain specification."""

    @abstractmethod
    def setup(self) -> None:
        """Load datasets and prepare for training."""

    @abstractmethod
    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        """Get DataLoader for a split."""

    @property
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        if self._train_loader is None:
            self._train_loader = self.get_dataloader(TaskSplit.TRAIN)
        return self._train_loader

    @property
    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self._val_loader is None:
            self._val_loader = self.get_dataloader(TaskSplit.VAL)
        return self._val_loader

    @property
    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        if self._test_loader is None:
            self._test_loader = self.get_dataloader(TaskSplit.TEST)
        return self._test_loader

    def get_batch(self, split: TaskSplit = TaskSplit.TRAIN) -> Batch:
        """Get a single batch from the specified split."""
        loader = self.get_dataloader(split)
        inputs, targets = next(iter(loader))
        return Batch(inputs=inputs, targets=targets).to(self.device)

    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        if self._input_dim is None:
            self.setup()
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        if self._output_dim is None:
            self.setup()
        return self._output_dim

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
    ) -> Metrics:
        """Evaluate model on a split."""

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss (default: cross entropy)."""
        return torch.nn.functional.cross_entropy(outputs, targets)

    def compute_metrics(
        self, outputs: torch.Tensor, targets: torch.Tensor, loss: float
    ) -> Metrics:
        """Compute metrics from outputs and targets."""
        accuracy = (outputs.argmax(1) == targets).float().mean().item()
        return Metrics(loss=loss, accuracy=accuracy)

    def get_model_kwargs(self) -> dict[str, Any]:
        """Get keyword arguments for model construction."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }
