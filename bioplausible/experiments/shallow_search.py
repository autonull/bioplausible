"""
Shallow Search System

Quickly evaluate all algorithm variants to find top performers.
"""

import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from algorithms import ALGORITHM_REGISTRY, create_model
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ShallowSearcher:
    """Rapid algorithm comparison system."""

    def __init__(
        self,
        algorithms: List[str],
        param_budget: int = 100000,
        device: str = None,
    ):
        """
        Initialize searcher.

        Args:
            algorithms: List of algorithm names to compare
            param_budget: Target parameter count
            device: Device to run on
        """
        self.algorithms = algorithms
        self.param_budget = param_budget
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def _design_architecture(
        self,
        input_dim: int,
        output_dim: int,
        target_params: int,
    ) -> List[int]:
        """
        Design hidden layer sizes to hit parameter budget.

        Simple heuristic: 2-layer network with balanced sizes.
        """
        # For 2 hidden layers: params ≈ input*h1 + h1*h2 + h2*output
        # Approximate with h1 = h2 = h
        # params ≈ h*(input + h + output)
        h = int(np.sqrt(target_params / 2))

        # Clamp to reasonable range
        h = max(32, min(h, 1024))

        return [h, h]

    def ultra_shallow_eval(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        input_dim: int,
        output_dim: int,
        time_budget: float = 30.0,
    ) -> Dict[str, Dict]:
        """
        Ultra-shallow evaluation: 30 seconds per algorithm.

        Returns:
            Dict mapping algorithm name to metrics
        """
        results = {}

        hidden_dims = self._design_architecture(
            input_dim, output_dim, self.param_budget
        )

        print(f"\n{'='*60}")
        print(f"Ultra-Shallow Search (30s per algorithm)")
        print(f"Parameter Budget: {self.param_budget:,}")
        print(f"Architecture: {input_dim} → {hidden_dims} → {output_dim}")
        print(f"{'='*60}\n")

        for algo_name in self.algorithms:
            print(f"Testing {algo_name}...")
            start = time.time()

            try:
                # Create model
                model = create_model(
                    algo_name,
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=output_dim,
                    learning_rate=0.001,
                    equilibrium_steps=10,  # Fast for shallow
                )

                # Quick training
                train_acc, test_acc, epochs = self._quick_train(
                    model,
                    train_loader,
                    test_loader,
                    time_budget=time_budget,
                )

                elapsed = time.time() - start

                results[algo_name] = {
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "epochs": epochs,
                    "time": elapsed,
                    "params": model.get_num_params(),
                    "success": True,
                }

                print(
                    f"  ✓ {algo_name}: {test_acc:.3f} acc in {epochs} epochs ({elapsed:.1f}s)"
                )

            except Exception as e:
                print(f"  ✗ {algo_name}: FAILED - {e}")
                results[algo_name] = {
                    "success": False,
                    "error": str(e),
                }

        self.results = results
        return results

    def _quick_train(
        self,
        model,
        train_loader: DataLoader,
        test_loader: DataLoader,
        time_budget: float = 30.0,
    ) -> Tuple[float, float, int]:
        """
        Train model for limited time.

        Returns:
            (train_acc, test_acc, epochs_completed)
        """
        model.to(self.device)
        start_time = time.time()
        epoch = 0

        while (time.time() - start_time) < time_budget:
            model.train()

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                metrics = model.train_step(x_batch, y_batch)

                # Check time budget
                if (time.time() - start_time) >= time_budget:
                    break

            epoch += 1

        # Evaluate
        train_acc = self._evaluate(model, train_loader)
        test_acc = self._evaluate(model, test_loader)

        return train_acc, test_acc, epoch

    def _evaluate(self, model, loader: DataLoader) -> float:
        """Evaluate accuracy."""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output = model.forward(x_batch)
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

        return correct / total if total > 0 else 0.0

    def rank_algorithms(self) -> List[Tuple[str, float]]:
        """
        Rank algorithms by test accuracy.

        Returns:
            List of (algorithm_name, test_acc) sorted by performance
        """
        successful = [
            (name, res["test_acc"])
            for name, res in self.results.items()
            if res.get("success", False)
        ]

        return sorted(successful, key=lambda x: x[1], reverse=True)

    def print_summary(self):
        """Print ranking summary."""
        print(f"\n{'='*60}")
        print("Ultra-Shallow Search Results")
        print(f"{'='*60}\n")

        ranking = self.rank_algorithms()

        print(f"{'Rank':<6} {'Algorithm':<25} {'Test Acc':<10} {'Time (s)':<10}")
        print("-" * 60)

        for i, (name, acc) in enumerate(ranking, 1):
            time_taken = self.results[name]["time"]
            print(f"{i:<6} {name:<25} {acc:>8.3f}   {time_taken:>8.1f}")

        print(f"\n{'='*60}\n")


def load_mnist_subset(n_samples: int = 5000) -> Tuple[DataLoader, DataLoader]:
    """Load small MNIST subset for shallow search."""
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))]  # Flatten
    )

    train_data = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_data = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # Subset
    indices = torch.randperm(len(train_data))[:n_samples]
    train_subset = torch.utils.data.Subset(train_data, indices)

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Quick test
    algorithms = list(ALGORITHM_REGISTRY.keys())

    print("Loading MNIST...")
    train_loader, test_loader = load_mnist_subset(n_samples=5000)

    searcher = ShallowSearcher(
        algorithms=algorithms,
        param_budget=100_000,
    )

    results = searcher.ultra_shallow_eval(
        train_loader=train_loader,
        test_loader=test_loader,
        input_dim=784,
        output_dim=10,
        time_budget=30.0,
    )

    searcher.print_summary()
