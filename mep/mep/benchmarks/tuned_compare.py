"""
Tuned MEP Benchmark Suite

Runs benchmarks with hyperparameters tuned for each optimizer type.
EP methods need different hyperparameters than backprop methods.
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
class OptimizerConfig:
    """Hyperparameters for an optimizer."""
    lr: float
    beta: float = 0.5
    settle_steps: int = 10
    settle_lr: float = 0.05
    loss_type: str = 'mse'
    ns_steps: int = 5
    gamma: float = 0.95
    error_beta: float = 0.9
    use_error_feedback: bool = True
    rank_frac: float = 0.2
    dion_thresh: int = 100000
    fisher_damping: float = 1e-3


# Tuned hyperparameters for different optimizers
OPTIMIZER_CONFIGS = {
    # Standard optimizers (backprop)
    'sgd': OptimizerConfig(lr=0.1),
    'adam': OptimizerConfig(lr=0.001),
    'adamw': OptimizerConfig(lr=0.001),
    'muon': OptimizerConfig(lr=0.02, gamma=0.95),
    
    # EP-based optimizers - OPTIMIZED for MNIST performance
    # Key insight: Higher beta and more settling steps dramatically improve convergence
    'eqprop': OptimizerConfig(
        lr=0.01,
        beta=0.5,  # Higher beta for stronger nudging
        settle_steps=30,  # More steps for proper settling
        settle_lr=0.15,  # Higher LR for faster convergence
        loss_type='mse',
        ns_steps=0,
        use_error_feedback=False
    ),
    'smep': OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=30,
        settle_lr=0.15,
        loss_type='mse',
        ns_steps=5,
        gamma=0.95,
        use_error_feedback=False
    ),
    'sdmep': OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=30,
        settle_lr=0.15,
        loss_type='mse',
        ns_steps=5,
        rank_frac=0.5,
        dion_thresh=200000,
        gamma=0.95,
        use_error_feedback=False
    ),
    'local_ep': OptimizerConfig(
        lr=0.01,
        beta=0.2,  # Lower beta for local EP
        settle_steps=15,
        settle_lr=0.02,
        loss_type='mse',
        use_error_feedback=False
    ),
    'natural_ep': OptimizerConfig(
        lr=0.01,
        beta=0.5,
        settle_steps=10,
        settle_lr=0.05,
        loss_type='mse',
        fisher_damping=1e-2,
        use_error_feedback=False
    ),
}


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
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
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
    if config.dataset in ["mnist", "fashion"]:
        return 784
    elif config.dataset == "cifar10":
        return 3072
    return 784


def get_num_classes(config: BenchmarkConfig) -> int:
    return 10


def train_epoch(
    model: nn.Module,
    optimizer: Any,
    train_loader: DataLoader,
    device: torch.device,
    is_ep: bool,
    opt_config: OptimizerConfig
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        if is_ep:
            # EP mode
            optimizer.step(x=x, target=y)
            with torch.no_grad():
                output = model(x)
                # Use MSE for EP training (more stable)
                if opt_config.loss_type == 'mse':
                    target_onehot = F.one_hot(y, num_classes=output.shape[1]).float()
                    loss = F.mse_loss(output, target_onehot)
                else:
                    loss = F.cross_entropy(output, y)
        else:
            # Standard backprop
            optimizer.zero_grad()
            output = model(x)
            loss = F.cross_entropy(output, y)
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
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = F.cross_entropy(output, y)
        
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
    
    train_loader, test_loader = get_dataloaders(config)
    input_dim = get_input_dim(config)
    num_classes = get_num_classes(config)
    model = get_model(config, input_dim, num_classes).to(device)
    
    # Get tuned config
    opt_config = OPTIMIZER_CONFIGS.get(optimizer_name, OptimizerConfig(lr=config.lr))
    
    # Get optimizer with tuned hyperparameters
    optimizer, is_ep = get_optimizer(
        optimizer_name,
        model,
        lr=opt_config.lr,
        weight_decay=config.weight_decay,
        beta=opt_config.beta,
        settle_steps=opt_config.settle_steps,
        settle_lr=opt_config.settle_lr,
        loss_type=opt_config.loss_type,
        ns_steps=opt_config.ns_steps,
        gamma=opt_config.gamma,
        error_beta=opt_config.error_beta,
        use_error_feedback=opt_config.use_error_feedback
    )
    
    metrics = []
    start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, is_ep, opt_config)
        val_loss, val_acc = evaluate(model, test_loader, device)
        
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


def run_all_benchmarks(config: BenchmarkConfig, optimizers: Optional[List[str]] = None) -> Dict[str, OptimizerResult]:
    """Run benchmarks for all optimizers."""
    if optimizers is None:
        optimizers = ['sgd', 'adam', 'muon', 'eqprop', 'smep', 'sdmep']

    results: Dict[str, OptimizerResult] = {}
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {opt_name.upper()} (LR={OPTIMIZER_CONFIGS[opt_name].lr})")
        print(f"{'='*60}")

        results[opt_name] = run_benchmark(opt_name, config)

    return results


def print_summary(results: Dict[str, OptimizerResult]) -> None:
    """Print summary table of results."""
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY (Tuned Hyperparameters)")
    print("="*90)
    print(f"{'Optimizer':<15} {'Best Val Acc':<15} {'Final Train Acc':<18} {'Total Time (s)':<15} {'LR':<10}")
    print("-"*90)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].best_val_acc, reverse=True)
    
    for name, result in sorted_results:
        lr = OPTIMIZER_CONFIGS.get(name, OptimizerConfig(lr=0.01)).lr
        print(f"{name:<15} {result.best_val_acc:<15.4f} {result.final_train_acc:<18.4f} {result.total_time:<15.2f} {lr:<10.5f}")
    
    print("="*90)
    
    best = sorted_results[0]
    print(f"\n🏆 Best performer: {best[0].upper()} with {best[1].best_val_acc:.2%} validation accuracy")
    
    # Show EP vs backprop comparison
    ep_opts = [r for r in sorted_results if r[0] in ['eqprop', 'smep', 'sdmep', 'local_ep', 'natural_ep']]
    bp_opts = [r for r in sorted_results if r[0] in ['sgd', 'adam', 'muon']]
    
    if ep_opts and bp_opts:
        best_ep = ep_opts[0]
        best_bp = bp_opts[0]
        print(f"\n📊 EP vs Backprop:")
        print(f"   Best EP:     {best_ep[0].upper()}: {best_ep[1].best_val_acc:.2%}")
        print(f"   Best Backprop: {best_bp[0].upper()}: {best_bp[1].best_val_acc:.2%}")
        print(f"   Gap: {best_bp[1].best_val_acc - best_ep[1].best_val_acc:.2%}")


def save_results(results: Dict[str, OptimizerResult], output_path: str) -> None:
    """Save results to JSON file."""
    data = {}
    for name, result in results.items():
        result_dict = {
            'name': result.name,
            'config': asdict(OPTIMIZER_CONFIGS.get(name, OptimizerConfig(lr=0.01))),
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
    parser = argparse.ArgumentParser(description='Tuned MEP Benchmark Suite')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion', 'cifar10'])
    parser.add_argument('--model', type=str, default='mlp',
                       choices=['mlp', 'mlp_small', 'cnn'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--subset-train', type=int, default=5000)
    parser.add_argument('--subset-test', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='tuned_benchmark_results.json')
    parser.add_argument('--optimizers', type=str, nargs='+', default=None,
                       help='Specific optimizers to benchmark')

    args = parser.parse_args()
    
    config = BenchmarkConfig(
        dataset=args.dataset,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        subset_train=args.subset_train,
        subset_test=args.subset_test,
        device=args.device
    )
    
    print("="*60)
    print("TUNED MEP BENCHMARK SUITE")
    print("="*60)
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Epochs: {config.epochs}")
    print(f"Device: {config.device}")
    print("="*60)
    
    results = run_all_benchmarks(config, args.optimizers)
    print_summary(results)
    save_results(results, args.output)


if __name__ == '__main__':
    main()
