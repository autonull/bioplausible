"""
Comprehensive MEP Benchmark Suite

Compares MEP optimizers against standard optimizers (Adam, SGD, Muon)
on various tasks and datasets.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from mep.benchmarks.baselines import get_optimizer


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    dataset: str = "mnist"
    model: str = "mlp"
    epochs: int = 10
    batch_size: int = 128
    lr: float = 0.01
    weight_decay: float = 0.0005
    subset_train: int = 5000
    subset_test: int = 1000
    device: str = "cuda"


@dataclass
class EpochMetrics:
    """Metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time: float


@dataclass
class OptimizerResult:
    """Results for a single optimizer."""
    name: str
    metrics: List[EpochMetrics]
    total_time: float
    best_val_acc: float
    final_train_acc: float


def get_dataloaders(config: BenchmarkConfig) -> Tuple[DataLoader, DataLoader]:
    """Get data loaders for the specified dataset."""
    if config.dataset == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
        
    elif config.dataset == "fashion":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, transform=transform)
        
    elif config.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    # Create subsets for faster benchmarking
    train_indices = list(range(min(config.subset_train, len(train_dataset))))
    test_indices = list(range(min(config.subset_test, len(test_dataset))))
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def get_model(config: BenchmarkConfig, input_dim: int, num_classes: int) -> nn.Module:
    """Get model for the specified architecture."""
    if config.model == "mlp":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    elif config.model == "mlp_small":
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    elif config.model == "cnn":
        if config.dataset == "cifar10":
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
    else:
        raise ValueError(f"Unknown model: {config.model}")


def get_input_dim(config: BenchmarkConfig) -> int:
    """Get input dimension for the dataset."""
    if config.dataset in ["mnist", "fashion"]:
        return 784
    elif config.dataset == "cifar10":
        return 3072
    else:
        return 784


def get_num_classes(config: BenchmarkConfig) -> int:
    """Get number of classes for the dataset."""
    if config.dataset in ["mnist", "fashion", "cifar10"]:
        return 10
    else:
        return 10


def train_epoch(
    model: nn.Module,
    optimizer: Any,
    train_loader: DataLoader,
    device: torch.device,
    is_ep: bool,
    loss_fn: nn.Module
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        if is_ep:
            # EP mode: optimizer handles forward pass
            optimizer.step(x=x, target=y)
            
            # Compute loss for tracking
            with torch.no_grad():
                output = model(x)
                loss = loss_fn(output, y)
        else:
            # Standard backprop mode
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = loss_fn(output, y)
        
        total_loss += loss.item() * x.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


def run_benchmark(
    optimizer_name: str,
    config: BenchmarkConfig
) -> OptimizerResult:
    """Run benchmark for a single optimizer."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # Get data
    train_loader, test_loader = get_dataloaders(config)
    
    # Get model
    input_dim = get_input_dim(config)
    num_classes = get_num_classes(config)
    model = get_model(config, input_dim, num_classes).to(device)
    
    # Get optimizer
    optimizer, is_ep = get_optimizer(
        optimizer_name,
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta=0.5,
        settle_steps=10,
        settle_lr=0.05,
        loss_type='cross_entropy' if config.dataset in ['mnist', 'fashion', 'cifar10'] else 'mse'
    )
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Training loop
    metrics = []
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, is_ep, loss_fn)
        val_loss, val_acc = evaluate(model, test_loader, device, loss_fn)
        
        epoch_time = time.time() - epoch_start
        
        metrics.append(EpochMetrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time
        ))
        
        print(f"  {optimizer_name} Epoch {epoch+1}/{config.epochs}: "
              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Time={epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    return OptimizerResult(
        name=optimizer_name,
        metrics=metrics,
        total_time=total_time,
        best_val_acc=max(m.val_acc for m in metrics),
        final_train_acc=metrics[-1].train_acc if metrics else 0.0
    )


def run_all_benchmarks(config: BenchmarkConfig) -> Dict[str, OptimizerResult]:
    """Run benchmarks for all optimizers."""
    optimizers = ['sgd', 'adam', 'muon', 'eqprop', 'smep', 'sdmep']
    
    results = {}
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {opt_name.upper()}")
        print(f"{'='*60}")
        
        results[opt_name] = run_benchmark(opt_name, config)
    
    return results


def print_summary(results: Dict[str, OptimizerResult]) -> None:
    """Print summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Optimizer':<15} {'Best Val Acc':<15} {'Final Train Acc':<18} {'Total Time (s)':<15}")
    print("-"*80)
    
    # Sort by best validation accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1].best_val_acc, reverse=True)
    
    for name, result in sorted_results:
        print(f"{name:<15} {result.best_val_acc:<15.4f} {result.final_train_acc:<18.4f} {result.total_time:<15.2f}")
    
    print("="*80)
    
    # Find best performer
    best = sorted_results[0]
    print(f"\n🏆 Best performer: {best[0].upper()} with {best[1].best_val_acc:.2%} validation accuracy")


def save_results(results: Dict[str, OptimizerResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {}
    for name, result in results.items():
        result_dict = {
            'name': result.name,
            'metrics': [asdict(m) for m in result.metrics],
            'total_time': result.total_time,
            'best_val_acc': result.best_val_acc,
            'final_train_acc': result.final_train_acc
        }
        data[name] = result_dict
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description='MEP Benchmark Suite')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'mlp_small', 'cnn'],
                       help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--subset-train', type=int, default=5000,
                       help='Number of training samples')
    parser.add_argument('--subset-test', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    config = BenchmarkConfig(
        dataset=args.dataset,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=args.device
    )
    
    print("="*60)
    print("MEP BENCHMARK SUITE")
    print("="*60)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning Rate: {config.lr}")
    print(f"Device: {config.device}")
    print("="*60)
    
    results = run_all_benchmarks(config)
    print_summary(results)
    save_results(results, args.output)


if __name__ == '__main__':
    main()
