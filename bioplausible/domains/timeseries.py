"""
Time Series Domain Tasks

Standard time series datasets and evaluation.
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


class TimeSeriesTask(DomainTask):
    """Time series domain tasks."""

    def __init__(
        self,
        name: str = "synthetic",
        dataset_name: str = "synthetic",
        seq_len: int = 32,
        horizon: int = 1,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.horizon = horizon

    @property
    def domain_type(self) -> DomainType:
        return DomainType.TIMESERIES

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.TIMESERIES,
            description=f"Time series task: {self.dataset_name}",
            default_metrics=["mse", "mae"],
            supported_tasks=["forecasting", "classification", "anomaly_detection"],
            default_batch_size=32,
            default_lr=1e-3,
            requires_sequence=True,
            tags=["timeseries", "forecasting"],
        )

    def setup(self) -> None:
        if self.dataset_name == "synthetic":
            self._setup_synthetic()
        else:
            raise ValueError(f"Unknown time series dataset: {self.dataset_name}")

    def _setup_synthetic(self) -> None:
        """Create synthetic sine wave forecasting data."""
        np.random.seed(42)
        t = np.linspace(0, 100, 5000)
        data = np.sin(t) + 0.1 * np.random.randn(5000)

        # Create sequences
        X, y = [], []
        for i in range(len(data) - self.seq_len - self.horizon + 1):
            X.append(data[i : i + self.seq_len])
            y.append(data[i + self.seq_len : i + self.seq_len + self.horizon])

        X = np.array(X, dtype=np.float32).reshape(-1, self.seq_len)
        y = np.array(y, dtype=np.float32).reshape(-1, self.horizon)

        # Split
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

        self._input_dim = self.seq_len
        self._output_dim = self.horizon
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

        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                total_mse += nn.functional.mse_loss(
                    outputs, targets
                ).item() * inputs.size(0)
                total_mae += nn.functional.l1_loss(
                    outputs, targets
                ).item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
        avg_mae = total_mae / total_samples if total_samples > 0 else 0.0

        return Metrics(
            loss=avg_mse,
            accuracy=0.0,
            custom={"mse": avg_mse, "mae": avg_mae},
        )

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(outputs, targets.float())
