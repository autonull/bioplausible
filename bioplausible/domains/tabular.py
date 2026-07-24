"""
Tabular Domain Tasks

Standard tabular/structured data datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bioplausible.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)


class TabularTask(DomainTask):
    """Tabular/structured data domain tasks."""

    def __init__(
        self,
        name: str = "digits",
        dataset_name: str = "digits",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name

    @property
    def domain_type(self) -> DomainType:
        return DomainType.TABULAR

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.TABULAR,
            description=f"Tabular task: {self.dataset_name}",
            default_metrics=["accuracy", "loss"],
            supported_tasks=["classification", "regression"],
            default_batch_size=32,
            default_lr=1e-3,
            tags=["tabular", "structured-data"],
        )

    def setup(self) -> None:
        from sklearn.datasets import (
            load_breast_cancer,
            load_digits,
            load_iris,
            load_wine,
        )

        _DATASETS = {
            "digits": load_digits,
            "breast_cancer": load_breast_cancer,
            "wine": load_wine,
            "iris": load_iris,
        }

        if self.dataset_name not in _DATASETS:
            raise ValueError(
                f"Unknown tabular dataset: {self.dataset_name}. "
                f"Available: {list(_DATASETS.keys())}"
            )

        data = _DATASETS[self.dataset_name]()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)

        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

        # Split train/val
        n = int(0.8 * len(X))
        X_train, X_val = X[:n], X[n:]
        y_train, y_val = y[:n], y[n:]

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

        self._train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        self._val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self._test_loader = self._val_loader

        self._input_dim = X.shape[1]
        self._output_dim = len(data.target_names)
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        if not self._setup_done:
            self.setup()

        if split == TaskSplit.TRAIN:
            return self._train_loader
        return self._val_loader

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: int | None = None,
    ) -> Metrics:
        model.eval()
        loader = self.get_dataloader(split)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_samples += inputs.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        return Metrics(loss=avg_loss, accuracy=accuracy)
