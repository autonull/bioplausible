"""
Continual Learning Benchmarks for MEP.

This module provides benchmark scripts for evaluating continual learning
performance using sequential and permuted MNIST tasks.

Benchmarks:
1. Permuted MNIST: Same network trained on MNIST with different fixed permutations
2. Sequential MNIST: Sequential training on tasks with error feedback

Metrics:
- Final accuracy on all tasks
- Forgetting measure: max accuracy drop per task
- Forward transfer: improvement on new tasks due to prior learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import List, Dict, Any, Optional, Tuple
import random
import json
from pathlib import Path
from dataclasses import dataclass, asdict

from mep.presets import sdmep, smep
from mep.optimizers import CompositeOptimizer
from mep.optimizers.strategies.gradient import EPGradient, BackpropGradient
from mep.optimizers.strategies.update import DionUpdate, MuonUpdate
from mep.optimizers.strategies.feedback import ErrorFeedback, NoFeedback
from mep.optimizers.strategies.constraint import SpectralConstraint


@dataclass
class TaskResult:
    """Results for a single task."""
    task_id: int
    train_accuracy: float
    test_accuracy: float
    forgetting: float  # Max accuracy drop from peak


@dataclass
class ContinualLearningResult:
    """Results for continual learning benchmark."""
    benchmark_name: str
    num_tasks: int
    task_results: List[TaskResult]
    average_accuracy: float
    average_forgetting: float
    final_accuracy: float  # Accuracy on last task


class PermutedMNIST:
    """
    Permuted MNIST benchmark for continual learning.
    
    Each task uses the same MNIST data but with a different fixed
    random permutation of input pixels.
    """
    
    def __init__(
        self,
        num_tasks: int = 5,
        seed: int = 42,
        data_dir: str = "./data"
    ):
        self.num_tasks = num_tasks
        self.seed = seed
        self.data_dir = data_dir
        
        # Generate fixed permutations for each task
        rng = random.Random(seed)
        self.permutations = []
        for _ in range(num_tasks):
            perm = torch.tensor(rng.sample(range(784), 784))
            self.permutations.append(perm)
        
        self._train_data: Optional[TensorDataset] = None
        self._test_data: Optional[TensorDataset] = None
    
    def _load_data(self) -> Tuple[TensorDataset, TensorDataset]:
        """Load MNIST data."""
        if self._train_data is not None:
            return self._train_data, self._test_data
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        
        # Convert to tensors
        x_train = train_dataset.data.float().flatten(1) / 255.0
        y_train = train_dataset.targets
        x_test = test_dataset.data.float().flatten(1) / 255.0
        y_test = test_dataset.targets
        
        self._train_data = TensorDataset(x_train, y_train)
        self._test_data = TensorDataset(x_test, y_test)
        
        return self._train_data, self._test_data
    
    def get_task_dataloaders(
        self,
        task_id: int,
        batch_size: int = 128
    ) -> Tuple[DataLoader, DataLoader]:
        """Get dataloaders for a specific task with its permutation."""
        if task_id >= self.num_tasks:
            raise ValueError(f"Task ID {task_id} >= num_tasks {self.num_tasks}")
        
        train_data, test_data = self._load_data()
        perm = self.permutations[task_id]
        
        # Apply permutation
        x_train_perm = train_data.tensors[0][:, perm]
        x_test_perm = test_data.tensors[0][:, perm]
        
        train_ds = TensorDataset(x_train_perm, train_data.tensors[1])
        test_ds = TensorDataset(x_test_perm, test_data.tensors[1])
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        
        return train_loader, test_loader


class MLP(nn.Module):
    """Simple MLP for MNIST."""
    
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_mep_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    use_error_feedback: bool = True,
    error_beta: float = 0.9,
    mode: str = "ep"
) -> CompositeOptimizer:
    """Create MEP optimizer with optional error feedback."""
    
    if mode == "ep":
        gradient = EPGradient(
            beta=0.1,
            settle_steps=5,
            settle_lr=0.05,
            loss_type="cross_entropy"
        )
    else:  # backprop
        gradient = BackpropGradient()
    
    feedback = ErrorFeedback(beta=error_beta) if use_error_feedback else NoFeedback()
    
    optimizer = CompositeOptimizer(
        model.parameters(),
        gradient=gradient,
        update=DionUpdate(rank_frac=0.1),
        constraint=SpectralConstraint(gamma=0.95, timing="post_update"),
        feedback=feedback,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        model=model if mode == "ep" else None,
    )
    
    return optimizer


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    use_ep: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        output = model(x_batch)
        loss = F.cross_entropy(output, y_batch)
        
        if use_ep and isinstance(optimizer, CompositeOptimizer):
            # EP mode
            optimizer.step(x=x_batch, target=y_batch)
        else:
            # Backprop mode
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * x_batch.size(0)
        _, predicted = output.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
    
    return correct / total


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> float:
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            _, predicted = output.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return correct / total


def run_permuted_mnist_benchmark(
    num_tasks: int = 5,
    epochs_per_task: int = 5,
    lr: float = 0.01,
    use_error_feedback: bool = True,
    mode: str = "ep",
    device: Optional[torch.device] = None,
    seed: int = 42
) -> ContinualLearningResult:
    """
    Run Permuted MNIST continual learning benchmark.
    
    Args:
        num_tasks: Number of tasks (permutations).
        epochs_per_task: Epochs to train per task.
        lr: Learning rate.
        use_error_feedback: Whether to use error feedback.
        mode: 'ep' or 'backprop'.
        device: Device to use.
        seed: Random seed.
    
    Returns:
        ContinualLearningResult with metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    
    benchmark = PermutedMNIST(num_tasks=num_tasks, seed=seed)
    
    # Create single model that will be trained on all tasks
    model = MLP(hidden_dims=(256, 128)).to(device)
    
    # Track peak accuracy per task for forgetting measure
    peak_accuracies: Dict[int, float] = {}
    task_results: List[TaskResult] = []
    
    for task_id in range(num_tasks):
        print(f"Training on task {task_id + 1}/{num_tasks}...")
        
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)
        
        optimizer = create_mep_optimizer(
            model,
            lr=lr,
            use_error_feedback=use_error_feedback,
            mode=mode
        )
        
        # Train on this task
        for epoch in range(epochs_per_task):
            train_acc = train_epoch(model, optimizer, train_loader, device, mode == "ep")
        
        # Evaluate on current task
        current_acc = evaluate(model, test_loader, device)
        peak_accuracies[task_id] = current_acc
        
        # Evaluate on all previous tasks to measure forgetting
        all_accuracies = []
        for prev_task_id in range(task_id + 1):
            _, prev_test_loader = benchmark.get_task_dataloaders(prev_task_id)
            acc = evaluate(model, prev_test_loader, device)
            all_accuracies.append(acc)
            
            # Update forgetting for this task
            forgetting = peak_accuracies.get(prev_task_id, acc) - acc
            peak_accuracies[prev_task_id] = max(peak_accuracies.get(prev_task_id, 0), acc)
        
        avg_acc = sum(all_accuracies) / len(all_accuracies)
        avg_forgetting = sum(peak_accuracies[i] - acc for i, acc in enumerate(all_accuracies)) / len(all_accuracies)
        
        task_results.append(TaskResult(
            task_id=task_id,
            train_accuracy=train_acc,
            test_accuracy=current_acc,
            forgetting=avg_forgetting
        ))
        
        print(f"  Task {task_id + 1}: Accuracy = {current_acc:.4f}, "
              f"Avg Accuracy = {avg_acc:.4f}, Forgetting = {avg_forgetting:.4f}")
    
    # Final evaluation on all tasks
    final_accuracies = []
    for task_id in range(num_tasks):
        _, test_loader = benchmark.get_task_dataloaders(task_id)
        acc = evaluate(model, test_loader, device)
        final_accuracies.append(acc)
    
    return ContinualLearningResult(
        benchmark_name="Permuted MNIST",
        num_tasks=num_tasks,
        task_results=task_results,
        average_accuracy=sum(final_accuracies) / len(final_accuracies),
        average_forgetting=sum(peak_accuracies[i] - acc for i, acc in enumerate(final_accuracies)) / len(final_accuracies),
        final_accuracy=final_accuracies[-1] if final_accuracies else 0.0
    )


def run_comparison_benchmark(
    num_tasks: int = 5,
    epochs_per_task: int = 5,
    lr: float = 0.01,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Dict[str, ContinualLearningResult]:
    """
    Run comparison between MEP with error feedback and backprop baseline.

    Returns:
        Dictionary with results for each method.
    """
    results = {}

    print("=" * 60)
    print("Continual Learning Benchmark: Permuted MNIST")
    print(f"Tasks: {num_tasks}, Epochs per task: {epochs_per_task}")
    print("=" * 60)

    # MEP with error feedback
    print("\n[1/3] Running MEP with Error Feedback...")
    results["mep_error_feedback"] = run_permuted_mnist_benchmark(
        num_tasks=num_tasks,
        epochs_per_task=epochs_per_task,
        lr=lr,
        use_error_feedback=True,
        mode="ep",
        device=device,
        seed=seed
    )

    # Backprop baseline (no error feedback)
    print("\n[2/3] Running Backprop Baseline...")
    results["backprop_baseline"] = run_permuted_mnist_benchmark(
        num_tasks=num_tasks,
        epochs_per_task=epochs_per_task,
        lr=lr,
        use_error_feedback=False,
        mode="backprop",
        device=device,
        seed=seed + 1  # Different seed for variety
    )

    # EWC baseline
    print("\n[3/3] Running EWC Baseline...")
    try:
        from .ewc_baseline import run_ewc_benchmark
        results["ewc_baseline"] = run_ewc_benchmark(
            num_tasks=num_tasks,
            epochs_per_task=epochs_per_task,
            lr=lr,
            device=device,
            seed=seed
        )
    except ImportError:
        print("  Warning: EWC baseline not available, skipping...")

    return results


def print_comparison(results: Dict[str, ContinualLearningResult]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Method':<30} {'Avg Acc':<12} {'Forgetting':<12} {'Final Acc':<12}")
    print("-" * 60)
    
    for name, result in results.items():
        print(f"{name:<30} {result.average_accuracy:<12.4f} "
              f"{result.average_forgetting:<12.4f} {result.final_accuracy:<12.4f}")
    
    print("=" * 60)
    
    # Summary
    mep = results.get("mep_error_feedback")
    bp = results.get("backprop_baseline")
    
    if mep and bp:
        print(f"\nMEP reduces forgetting by: {(bp.average_forgetting - mep.average_forgetting):.4f}")
        print(f"MEP avg accuracy improvement: {(mep.average_accuracy - bp.average_accuracy):.4f}")


def save_results(
    results: Dict[str, ContinualLearningResult],
    output_path: str = "continual_learning_results.json"
) -> None:
    """Save results to JSON file."""
    data = {name: asdict(result) for name, result in results.items()}
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Continual Learning Benchmarks for MEP")
    parser.add_argument("--tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per task")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="cl_results.json", help="Output file")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    
    device = torch.device("cpu" if args.cpu else "cuda")
    print(f"Using device: {device}")
    
    results = run_comparison_benchmark(
        num_tasks=args.tasks,
        epochs_per_task=args.epochs,
        lr=args.lr,
        device=device,
        seed=args.seed
    )
    
    print_comparison(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
