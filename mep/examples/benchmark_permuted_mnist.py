#!/usr/bin/env python3
"""
Permuted MNIST Continual Learning Benchmark

Phase 2: Priority 3 - Continual Learning

Tests continual learning performance on Permuted MNIST:
- Train on task 1 (MNIST with permutation 1)
- Consolidate with EWC
- Train on task 2 (MNIST with permutation 2)
- Measure forgetting on task 1
- Repeat for multiple tasks

Metrics:
- Average accuracy across all tasks
- Forgetting measure (accuracy drop after learning new tasks)
- Forward transfer (improvement on new tasks)

Run: python examples/benchmark_permuted_mnist.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import time
import numpy as np

from mep import smep, muon_backprop
from mep.optimizers import EWCRegularizer, EPOptimizerWithEWC, O1MemoryEPv2


@dataclass
class BenchmarkResult:
    """Result for a single benchmark run."""
    method: str  # 'ep_ewc', 'ep_no_ewc', 'bp_ewc', 'bp_no_ewc'
    ewc_lambda: float
    
    # Per-task accuracies (after all tasks)
    final_accuracies: List[float]
    
    # Average accuracy across all tasks
    average_accuracy: float
    
    # Forgetting measure (max accuracy drop)
    forgetting_measure: float
    
    # Training time
    total_time_sec: float
    
    # Per-task training accuracy (during training)
    training_accuracies: List[List[float]]


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""
    timestamp: str
    num_tasks: int
    epochs_per_task: int
    batch_size: int
    hidden_dim: int
    
    results: List[BenchmarkResult]
    
    # Best method
    best_method: str
    best_accuracy: float
    best_forgetting: float


class MLP(nn.Module):
    """Simple MLP for MNIST."""
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def get_permuted_mnist(
    num_tasks: int = 5,
    batch_size: int = 32,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader, torch.Tensor]]:
    """
    Generate Permuted MNIST dataset.
    
    Args:
        num_tasks: Number of tasks (permutations).
        batch_size: Batch size for data loaders.
        seed: Random seed for reproducibility.
    
    Returns:
        List of (train_loader, test_loader, permutation) for each task.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Download MNIST if needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Flatten images
    train_images = train_dataset.data.float().flatten(1) / 255.0
    train_labels = train_dataset.targets
    test_images = test_dataset.data.float().flatten(1) / 255.0
    test_labels = test_dataset.targets
    
    # Generate random permutations
    permutations = []
    for _ in range(num_tasks):
        perm = torch.randperm(784)
        permutations.append(perm)
    
    # Create data loaders for each task
    task_data = []
    for task_id, perm in enumerate(permutations):
        # Apply permutation
        permuted_train = train_images[:, perm]
        permuted_test = test_images[:, perm]
        
        # Create datasets
        train_tensor = TensorDataset(permuted_train, train_labels)
        test_tensor = TensorDataset(permuted_test, test_labels)
        
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
        
        task_data.append((train_loader, test_loader, perm))
    
    return task_data


def compute_accuracy(model: nn.Module, data_loader: DataLoader, device: str) -> float:
    """Compute classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    return correct / total


def train_task_ep_ewc(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    task_id: int,
    epochs: int,
    device: str,
    ewc_lambda: float = 100.0,
    use_ewc: bool = True,
) -> Tuple[EPOptimizerWithEWC, List[float]]:
    """
    Train on a single task with EP + EWC.
    
    Returns:
        optimizer: Trained optimizer (contains EWC state).
        accuracies: List of test accuracies per epoch.
    """
    optimizer = EPOptimizerWithEWC(
        model.parameters(),
        model=model,
        lr=0.01,
        ewc_lambda=ewc_lambda,
        settle_steps=10,
        settle_lr=0.2,
        beta=0.5,
        use_analytic=True,
    )
    
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.step(x=x, target=y, task_id=task_id, use_ewc=use_ewc)
        
        # Evaluate
        acc = compute_accuracy(model, test_loader, device)
        accuracies.append(acc)
    
    return optimizer, accuracies


def train_task_bp_ewc(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    task_id: int,
    epochs: int,
    device: str,
    ewc_lambda: float = 100.0,
    use_ewc: bool = True,
) -> Tuple[torch.optim.Optimizer, EWCRegularizer, List[float]]:
    """
    Train on a single task with Backprop + EWC.
    
    Returns:
        optimizer: Trained optimizer.
        ewc: EWC regularizer (contains Fisher state).
        accuracies: List of test accuracies per epoch.
    """
    optimizer = muon_backprop(model.parameters(), lr=0.01)
    ewc = EWCRegularizer(model, ewc_lambda=ewc_lambda)
    criterion = nn.CrossEntropyLoss()
    
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            # Add EWC regularization
            if use_ewc and len(ewc.task_memories) > 0:
                ewc_loss = ewc.compute_ewc_loss()
                loss = loss + ewc_loss
            
            loss.backward()
            optimizer.step()
        
        # Evaluate
        acc = compute_accuracy(model, test_loader, device)
        accuracies.append(acc)
    
    return optimizer, ewc, accuracies


def run_benchmark(
    num_tasks: int = 5,
    epochs_per_task: int = 3,
    batch_size: int = 32,
    hidden_dim: int = 256,
    ewc_lambda: float = 100.0,
    seed: int = 42,
) -> BenchmarkSummary:
    """
    Run full continual learning benchmark.
    
    Tests:
    - EP + EWC
    - EP without EWC
    - Backprop + EWC
    - Backprop without EWC
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Tasks: {num_tasks}, Epochs: {epochs_per_task}, Batch: {batch_size}")
    print(f"EWC Lambda: {ewc_lambda}")
    print("=" * 60)
    
    # Get data
    task_data = get_permuted_mnist(num_tasks, batch_size, seed)
    
    results = []
    
    # Methods to test
    methods = [
        ('ep_ewc', True, True),
        ('ep_no_ewc', True, False),
        ('bp_ewc', False, True),
        ('bp_no_ewc', False, False),
    ]
    
    for method_name, use_ep, use_ewc in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print(f"{'='*60}")
        
        torch.manual_seed(seed)
        model = MLP(hidden_dim=hidden_dim).to(device)
        
        final_accuracies = []
        training_accuracies = []
        total_time = 0.0
        
        # Track accuracy after each task for forgetting computation
        task_accuracies_after = []  # List of accuracies on all previous tasks
        
        for task_id in range(num_tasks):
            train_loader, test_loader, _ = task_data[task_id]
            
            start = time.time()
            
            if use_ep:
                optimizer, accs = train_task_ep_ewc(
                    model, train_loader, test_loader, task_id,
                    epochs_per_task, device, ewc_lambda, use_ewc,
                )
                # Consolidate task
                if use_ewc:
                    optimizer.consolidate_task(train_loader, task_id, device)
            else:
                optimizer, ewc, accs = train_task_bp_ewc(
                    model, train_loader, test_loader, task_id,
                    epochs_per_task, device, ewc_lambda, use_ewc,
                )
                # Consolidate task
                if use_ewc:
                    ewc.update_fisher(train_loader, task_id, device)
            
            total_time += time.time() - start
            training_accuracies.append(accs)
            
            # Evaluate on all tasks seen so far
            current_accs = []
            for prev_task in range(task_id + 1):
                _, prev_test_loader, _ = task_data[prev_task]
                acc = compute_accuracy(model, prev_test_loader, device)
                current_accs.append(acc)
            
            task_accuracies_after.append(current_accs)
            print(f"  Task {task_id}: Train acc = {accs[-1]*100:.1f}%, "
                  f"Time = {time.time() - start:.1f}s")
        
        # Final accuracies on all tasks
        for task_id in range(num_tasks):
            _, test_loader, _ = task_data[task_id]
            acc = compute_accuracy(model, test_loader, device)
            final_accuracies.append(acc)
        
        # Compute forgetting measure
        # Forgetting = max accuracy drop from after training to final
        forgetting = 0.0
        for task_id in range(num_tasks):
            # Accuracy right after training on this task
            acc_after = task_accuracies_after[task_id][task_id]
            # Final accuracy on this task
            acc_final = final_accuracies[task_id]
            drop = acc_after - acc_final
            forgetting = max(forgetting, drop)
        
        avg_acc = sum(final_accuracies) / len(final_accuracies)
        
        result = BenchmarkResult(
            method=method_name,
            ewc_lambda=ewc_lambda,
            final_accuracies=final_accuracies,
            average_accuracy=avg_acc,
            forgetting_measure=forgetting,
            total_time_sec=total_time,
            training_accuracies=training_accuracies,
        )
        results.append(result)
        
        print(f"\n  Final Results:")
        print(f"    Average Accuracy: {avg_acc*100:.1f}%")
        print(f"    Forgetting: {forgetting*100:.1f}%")
        print(f"    Total Time: {total_time:.1f}s")
    
    # Find best method
    best_result = max(results, key=lambda r: r.average_accuracy)
    
    summary = BenchmarkSummary(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        num_tasks=num_tasks,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        results=results,
        best_method=best_result.method,
        best_accuracy=best_result.average_accuracy,
        best_forgetting=best_result.forgetting_measure,
    )
    
    return summary


def print_results_table(summary: BenchmarkSummary):
    """Print benchmark results table."""
    print("\n" + "=" * 80)
    print("CONTINUAL LEARNING BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Tasks: {summary.num_tasks}")
    print(f"  Epochs per task: {summary.epochs_per_task}")
    print(f"  Hidden dim: {summary.hidden_dim}")
    print(f"  EWC Lambda: {summary.results[0].ewc_lambda}")
    
    print(f"\n{'Method':<15} {'Avg Acc (%)':<15} {'Forgetting (%)':<15} {'Time (s)':<10}")
    print("-" * 60)
    
    for r in summary.results:
        print(f"{r.method:<15} {r.average_accuracy*100:<15.1f} "
              f"{r.forgetting_measure*100:<15.1f} {r.total_time_sec:<10.1f}")
    
    print("-" * 60)
    print(f"\nBest Method: {summary.best_method}")
    print(f"  Accuracy: {summary.best_accuracy*100:.1f}%")
    print(f"  Forgetting: {summary.best_forgetting*100:.1f}%")
    print("=" * 80)


def save_results(summary: BenchmarkSummary, filename: str = "continual_learning_results.json"):
    """Save results to JSON."""
    # Convert to dict, excluding training_accuracies for brevity
    data = {
        'timestamp': summary.timestamp,
        'num_tasks': summary.num_tasks,
        'epochs_per_task': summary.epochs_per_task,
        'batch_size': summary.batch_size,
        'hidden_dim': summary.hidden_dim,
        'results': [
            {
                'method': r.method,
                'ewc_lambda': r.ewc_lambda,
                'final_accuracies': r.final_accuracies,
                'average_accuracy': r.average_accuracy,
                'forgetting_measure': r.forgetting_measure,
                'total_time_sec': r.total_time_sec,
            }
            for r in summary.results
        ],
        'best_method': summary.best_method,
        'best_accuracy': summary.best_accuracy,
        'best_forgetting': summary.best_forgetting,
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")


def main():
    print("=" * 80)
    print("PHASE 2: CONTINUAL LEARNING BENCHMARK")
    print("Permuted MNIST with EP + EWC")
    print("=" * 80)
    
    # Configuration - reduced for faster testing
    num_tasks = 3  # Reduced from 5 for faster testing
    epochs_per_task = 2  # Reduced from 3
    batch_size = 64  # Increased for faster training
    hidden_dim = 128  # Reduced from 256
    ewc_lambda = 100.0
    
    print(f"\nConfiguration:")
    print(f"  Tasks: {num_tasks}")
    print(f"  Epochs per task: {epochs_per_task}")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  EWC Lambda: {ewc_lambda}")
    
    # Run benchmark
    summary = run_benchmark(
        num_tasks=num_tasks,
        epochs_per_task=epochs_per_task,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        ewc_lambda=ewc_lambda,
    )
    
    # Print results
    print_results_table(summary)
    
    # Save results
    save_results(summary)
    
    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    # Compare EP+EWC vs EP without EWC
    ep_ewc = next(r for r in summary.results if r.method == 'ep_ewc')
    ep_no_ewc = next(r for r in summary.results if r.method == 'ep_no_ewc')
    
    acc_improvement = ep_ewc.average_accuracy - ep_no_ewc.average_accuracy
    forgetting_reduction = ep_no_ewc.forgetting_measure - ep_ewc.forgetting_measure
    
    print(f"\nEP + EWC vs EP without EWC:")
    print(f"  Accuracy improvement: {acc_improvement*100:+.1f}%")
    print(f"  Forgetting reduction: {forgetting_reduction*100:+.1f}%")
    
    # Compare to backprop
    bp_ewc = next(r for r in summary.results if r.method == 'bp_ewc')
    
    acc_vs_bp = ep_ewc.average_accuracy - bp_ewc.average_accuracy
    forget_vs_bp = ep_ewc.forgetting_measure - bp_ewc.forgetting_measure
    
    print(f"\nEP + EWC vs Backprop + EWC:")
    print(f"  Accuracy difference: {acc_vs_bp*100:+.1f}%")
    print(f"  Forgetting difference: {forget_vs_bp*100:+.1f}%")
    
    # Success criteria
    print(f"\nSuccess Criteria (target: <15% forgetting):")
    if ep_ewc.forgetting_measure < 0.15:
        print(f"  ✅ EP + EWC achieves {ep_ewc.forgetting_measure*100:.1f}% forgetting")
    else:
        print(f"  ⚠️  EP + EWC has {ep_ewc.forgetting_measure*100:.1f}% forgetting (target: <15%)")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
