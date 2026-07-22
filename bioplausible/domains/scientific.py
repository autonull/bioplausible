"""
Scientific Simulation Domain Tasks

Toy scientific simulation environments for testing bio-plausible learning
on physics-inspired problems.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from bioplausible.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)


class ScientificTask(DomainTask):
    """Scientific simulation domain tasks."""

    def __init__(
        self,
        name: str = "pendulum",
        dataset_name: str = "pendulum",
        n_samples: int = 5000,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name
        self.n_samples = n_samples

    @property
    def domain_type(self) -> DomainType:
        return DomainType.SCIENTIFIC

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.SCIENTIFIC,
            description=f"Scientific simulation: {self.dataset_name}",
            default_metrics=["mse", "accuracy"],
            supported_tasks=["regression", "simulation"],
            default_batch_size=64,
            default_lr=1e-3,
            tags=["scientific", "simulation", "physics"],
        )

    def setup(self) -> None:
        if self.dataset_name == "pendulum":
            self._setup_pendulum()
        elif self.dataset_name == "lorenz":
            self._setup_lorenz()
        else:
            raise ValueError(
                f"Unknown scientific dataset: {self.dataset_name}. "
                f"Available: pendulum, lorenz"
            )

    def _setup_pendulum(self) -> None:
        """Simple pendulum ODE: predict next state from current state."""
        np.random.seed(42)
        g = 9.81
        L = 1.0
        dt = 0.05

        theta = np.random.uniform(-np.pi, np.pi, self.n_samples)
        omega = np.random.uniform(-3.0, 3.0, self.n_samples)

        # Current state: [sin(theta), cos(theta), omega]
        X = np.column_stack([np.sin(theta), np.cos(theta), omega]).astype(np.float32)

        # Next state
        alpha = -(g / L) * np.sin(theta)
        omega_next = omega + alpha * dt
        theta_next = theta + omega * dt
        y = np.column_stack(
            [np.sin(theta_next), np.cos(theta_next), omega_next]
        ).astype(np.float32)

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
        self._output_dim = y.shape[1]
        self._setup_done = True

    def _setup_lorenz(self) -> None:
        """Lorenz system: predict next state from current state."""
        np.random.seed(42)

        def lorenz(x, y, z, sigma=10, rho=28, beta=8 / 3):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return dx, dy, dz

        dt = 0.01
        states = []
        x, y, z = 1.0, 1.0, 1.0
        for _ in range(self.n_samples):
            states.append([x, y, z])
            dx, dy, dz = lorenz(x, y, z)
            x += dx * dt
            y += dy * dt
            z += dz * dt

        states = np.array(states, dtype=np.float32)
        X = states[:-1]
        y = states[1:]

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
        self._output_dim = y.shape[1]
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
        max_batches: Optional[int] = None,
    ) -> Metrics:
        model.eval()
        loader = self.get_dataloader(split)

        total_mse = 0.0
        total_samples = 0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(loader):
                if max_batches and i >= max_batches:
                    break

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = self.compute_loss(outputs, targets)

                total_mse += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)

        avg_mse = total_mse / total_samples if total_samples > 0 else 0.0
        return Metrics(
            loss=avg_mse, accuracy=max(0.0, 1.0 - avg_mse), custom={"mse": avg_mse}
        )

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(outputs, targets.float())
