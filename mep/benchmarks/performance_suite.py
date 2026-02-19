#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite

Tests EP optimizer performance across all configurations:
- Different settling steps (5, 10, 20, 30)
- Different learning rates
- Different network depths
- EP vs Backprop comparison

Run: python benchmarks/performance_suite.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

from mep import EPOptimizer, smep, smep_fast, muon_backprop


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    config_name: str
    mode: str
    depth: int
    settle_steps: int
    lr: float
    epoch: int
    
    # Performance
    time_per_epoch_sec: float
    samples_per_sec: float
    
    # Accuracy
    train_accuracy: float
    test_accuracy: float
    
    # Stability
    has_nan: bool
    has_inf: bool
    
    # Metadata
    timestamp: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of all benchmarks."""
    timestamp: str
    device: str
    total_tests: int
    passed: int
    failed: int
    
    results: List[BenchmarkResult]
    
    # Best configurations
    best_accuracy: BenchmarkResult
    fastest_converging: BenchmarkResult


def get_mnist_loaders(batch_size=128, num_workers=0):
    """Load MNIST dataset."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader, len(train_dataset), len(test_dataset)


class MLP(nn.Module):
    """Configurable MLP with working architecture."""
    
    def __init__(self, input_dim=784, hidden_dim=256, num_layers=3, output_dim=10):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        # Input layer with specified hidden dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers - use decreasing hidden dims for better performance
        current_dim = hidden_dim
        for i in range(num_layers - 2):
            next_dim = max(hidden_dim // (2 ** (i + 1)), 32)  # Decrease but min 32
            layers.append(nn.Linear(current_dim, next_dim))
            layers.append(nn.ReLU())
            current_dim = next_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


def evaluate(model, loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).flatten(1), y.to(device)  # Flatten images!
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    
    return correct / total


def check_stability(model):
    """Check for NaN/Inf in parameters."""
    has_nan = False
    has_inf = False
    
    for p in model.parameters():
        if p.grad is not None:
            has_nan = has_nan or torch.isnan(p.grad).any().item()
            has_inf = has_inf or torch.isinf(p.grad).any().item()
    
    return has_nan, has_inf


def benchmark_config(
    config_name: str,
    mode: str,
    depth: int,
    settle_steps: int,
    lr: float,
    epochs: int,
    train_loader,
    test_loader,
    train_size: int,
    test_size: int,
    device: str,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    
    # Create model
    model = MLP(input_dim=784, hidden_dim=256, num_layers=depth, output_dim=10).to(device)
    
    # Create optimizer
    if mode == 'ep':
        opt = EPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            lr=lr,
            settle_steps=settle_steps,
            settle_lr=0.15,
            beta=0.5,
            loss_type='mse',
            gradient_method='autograd',
        )
    else:  # backprop
        opt = muon_backprop(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device).flatten(1), y.to(device)  # Flatten images!
            
            if mode == 'ep':
                opt.step(x=x, target=y)
            else:
                opt.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                opt.step()
        
        # Check stability
        has_nan, has_inf = check_stability(model)
        if has_nan or has_inf:
            return BenchmarkResult(
                config_name=config_name,
                mode=mode,
                depth=depth,
                settle_steps=settle_steps,
                lr=lr,
                epoch=epochs,
                time_per_epoch_sec=0,
                samples_per_sec=0,
                train_accuracy=0,
                test_accuracy=0,
                has_nan=has_nan,
                has_inf=has_inf,
                timestamp=datetime.now().isoformat(),
            )
    
    elapsed = time.time() - start_time
    time_per_epoch = elapsed / epochs
    samples_per_sec = (train_size * epochs) / elapsed
    
    # Evaluate
    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    
    return BenchmarkResult(
        config_name=config_name,
        mode=mode,
        depth=depth,
        settle_steps=settle_steps,
        lr=lr,
        epoch=epochs,
        time_per_epoch_sec=time_per_epoch,
        samples_per_sec=samples_per_sec,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        has_nan=False,
        has_inf=False,
        timestamp=datetime.now().isoformat(),
    )


def run_performance_suite(epochs=3, quick=False):
    """Run full performance benchmark suite."""
    
    print("="*80)
    print("EP PERFORMANCE BENCHMARK SUITE")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Load data
    print("Loading MNIST...")
    train_loader, test_loader, train_size, test_size = get_mnist_loaders(batch_size=128)
    
    # Configurations to test
    configs = [
        # EP configurations
        {'name': 'EP_fast', 'mode': 'ep', 'depth': 3, 'settle_steps': 5, 'lr': 0.01},
        {'name': 'EP_default', 'mode': 'ep', 'depth': 3, 'settle_steps': 10, 'lr': 0.01},
        {'name': 'EP_accurate', 'mode': 'ep', 'depth': 3, 'settle_steps': 30, 'lr': 0.01},
        {'name': 'EP_deep', 'mode': 'ep', 'depth': 10, 'settle_steps': 10, 'lr': 0.01},
        
        # Backprop configurations
        {'name': 'BP_default', 'mode': 'backprop', 'depth': 3, 'settle_steps': 0, 'lr': 0.02},
        {'name': 'BP_deep', 'mode': 'backprop', 'depth': 10, 'settle_steps': 0, 'lr': 0.02},
    ]
    
    if quick:
        configs = configs[:3]  # Just first 3 for quick test
    
    results = []
    passed = 0
    failed = 0
    
    for cfg in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {cfg['name']}")
        print(f"  Mode: {cfg['mode']}, Depth: {cfg['depth']}, Settle: {cfg['settle_steps']}, LR: {cfg['lr']}")
        print(f"{'='*60}")
        
        try:
            result = benchmark_config(
                config_name=cfg['name'],
                mode=cfg['mode'],
                depth=cfg['depth'],
                settle_steps=cfg['settle_steps'],
                lr=cfg['lr'],
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                train_size=train_size,
                test_size=test_size,
                device=device,
            )
            
            results.append(result)
            
            if result.has_nan or result.has_inf:
                print(f"  ✗ FAILED: NaN/Inf detected")
                failed += 1
            else:
                print(f"  ✓ PASSED")
                print(f"    Test Accuracy: {result.test_accuracy*100:.1f}%")
                print(f"    Time/Epoch: {result.time_per_epoch_sec:.2f}s")
                print(f"    Samples/sec: {result.samples_per_sec:.0f}")
                passed += 1
                
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            results.append(BenchmarkResult(
                config_name=cfg['name'],
                mode=cfg['mode'],
                depth=cfg['depth'],
                settle_steps=cfg['settle_steps'],
                lr=cfg['lr'],
                epoch=epochs,
                time_per_epoch_sec=0,
                samples_per_sec=0,
                train_accuracy=0,
                test_accuracy=0,
                has_nan=False,
                has_inf=False,
                timestamp=datetime.now().isoformat(),
            ))
    
    # Find best configurations
    valid_results = [r for r in results if not r.has_nan and not r.has_inf and r.test_accuracy > 0]
    
    best_accuracy = max(valid_results, key=lambda r: r.test_accuracy) if valid_results else None
    fastest = min(valid_results, key=lambda r: r.time_per_epoch_sec) if valid_results else None
    
    # Summary
    summary = BenchmarkSummary(
        timestamp=datetime.now().isoformat(),
        device=device,
        total_tests=len(configs),
        passed=passed,
        failed=failed,
        results=results,
        best_accuracy=best_accuracy,
        fastest_converging=fastest,
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total: {len(configs)} tests, {passed} passed, {failed} failed")
    
    if best_accuracy:
        print(f"\nBest Accuracy: {best_accuracy.config_name}")
        print(f"  Test Accuracy: {best_accuracy.test_accuracy*100:.1f}%")
        print(f"  Time/Epoch: {best_accuracy.time_per_epoch_sec:.2f}s")
    
    if fastest:
        print(f"\nFastest: {fastest.config_name}")
        print(f"  Time/Epoch: {fastest.time_per_epoch_sec:.2f}s")
        print(f"  Test Accuracy: {fastest.test_accuracy*100:.1f}%")
    
    print("="*80)
    
    return summary


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='EP Performance Benchmark Suite')
    parser.add_argument('--quick', action='store_true', help='Quick test (3 configs)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--output', type=str, default='performance_results.json', help='Output file')
    
    args = parser.parse_args()
    
    summary = run_performance_suite(epochs=args.epochs, quick=args.quick)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
