"""
MEP Benchmark Runner

This module provides comprehensive benchmarking capabilities for MEP optimizers,
including YAML-based configuration, metrics tracking, and automated visualization.
"""

import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VIS_AVAILABLE = True
except ImportError:
    VIS_AVAILABLE = False

from mep.presets import smep, sdmep


@dataclass
class BenchmarkMetrics:
    """Container for benchmark metrics."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    epoch_time: float
    spectral_norm: Optional[float] = None
    energy_free: Optional[float] = None
    energy_nudged: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Container for complete benchmark results."""
    config: Dict[str, Any]
    optimizer_name: str
    metrics: List[BenchmarkMetrics] = field(default_factory=list)
    total_time: float = 0.0
    final_train_acc: float = 0.0
    final_val_acc: float = 0.0
    best_val_acc: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config,
            "optimizer_name": self.optimizer_name,
            "metrics": [asdict(m) for m in self.metrics],
            "total_time": self.total_time,
            "final_train_acc": self.final_train_acc,
            "final_val_acc": self.final_val_acc,
            "best_val_acc": self.best_val_acc,
        }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required for config loading. Install with: pip install PyYAML")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle defaults inheritance
    if 'defaults' in config:
        base_config: Dict[str, Any] = {}
        for default in config['defaults']:
            if isinstance(default, dict):
                for key, value in default.items():
                    if key == 'base' and value:
                        # Load base config
                        base_path = Path(config_path).parent / "base.yaml"
                        if base_path.exists():
                            with open(base_path, 'r') as f:
                                base_config = yaml.safe_load(f)
                        break
        
        # Merge configs (child overrides parent)
        config = _merge_configs(base_config, config)
        del config['defaults']
    
    return cast(Dict[str, Any], config)


def _merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def create_model(architecture: List[Dict[str, Any]], device: torch.device) -> nn.Module:
    """Create model from architecture specification."""
    layers: List[nn.Module] = []
    
    for layer_spec in architecture:
        # Make a copy to avoid modifying original config
        spec = layer_spec.copy()
        layer_type = spec.pop('type')
        
        if layer_type == 'Linear':
            layers.append(nn.Linear(**spec))
        elif layer_type == 'Conv2d':
            layers.append(nn.Conv2d(**spec))
        elif layer_type == 'ReLU':
            layers.append(nn.ReLU())
        elif layer_type == 'Sigmoid':
            layers.append(nn.Sigmoid())
        elif layer_type == 'Tanh':
            layers.append(nn.Tanh())
        elif layer_type == 'MaxPool2d':
            layers.append(nn.MaxPool2d(**spec))
        elif layer_type == 'AvgPool2d':
            layers.append(nn.AvgPool2d(**spec))
        elif layer_type == 'AdaptiveAvgPool2d':
            layers.append(nn.AdaptiveAvgPool2d(**spec))
        elif layer_type == 'Flatten':
            layers.append(nn.Flatten())
        elif layer_type == 'Dropout':
            layers.append(nn.Dropout(**spec))
        elif layer_type == 'BatchNorm2d':
            layers.append(nn.BatchNorm2d(**spec))
        elif layer_type == 'LayerNorm':
            layers.append(nn.LayerNorm(**spec))
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")
    
    return nn.Sequential(*layers).to(device)


def get_dataloader(
    dataset_name: str,
    batch_size: int,
    root: str = './data',
    subset_size: Optional[int] = None,
    num_workers: int = 4,
    device: Optional[torch.device] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders."""
    
    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    # Normalize based on dataset
    if dataset_name.upper() in ['MNIST', 'FASHIONMNIST']:
        mean, std = (0.5,), (0.5,)
    else:  # CIFAR10, CIFAR100
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Get dataset class
    dataset_map = {
        'MNIST': datasets.MNIST,
        'FashionMNIST': datasets.FashionMNIST,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
    }
    
    dataset_class = dataset_map.get(dataset_name.upper())
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_dataset = dataset_class(root, train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root, train=False, download=True, transform=transform_test)
    
    # Subset for faster benchmarking
    if subset_size and subset_size < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def create_optimizer(
    optimizer_name: str,
    model: nn.Module,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """Create optimizer from name and config."""
    
    opt_config = config.get('optimizer', {})
    ep_config = config.get('ep', {})
    dion_config = config.get('dion', {})
    
    # Get optimizer-specific overrides
    opt_overrides = config.get('optimizers', {}).get(optimizer_name, {})
    
    # Merge configs
    all_config = {**opt_config, **ep_config, **dion_config, **opt_overrides}
    
    if optimizer_name == 'SGD':
        return torch.optim.SGD(
            model.parameters(),
            lr=all_config.get('lr', 0.05),
            momentum=all_config.get('momentum', 0.9),
            weight_decay=all_config.get('weight_decay', 0.0005)
        )
    
    elif optimizer_name == 'Adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=all_config.get('lr', 0.001),
            weight_decay=all_config.get('weight_decay', 0.0)
        )
    
    elif optimizer_name == 'AdamW':
        return torch.optim.AdamW(
            model.parameters(),
            lr=all_config.get('lr', 0.001),
            weight_decay=all_config.get('weight_decay', 0.01)
        )
    
    elif optimizer_name == 'SMEP':
        return smep(
            model.parameters(),
            model=model,
            mode='ep',
            lr=all_config.get('lr', 0.02),
            beta=all_config.get('beta', 0.3),
            settle_steps=all_config.get('settle_steps', 15),
            settle_lr=all_config.get('settle_lr', 0.02),
            ns_steps=all_config.get('ns_steps', 5),
            use_spectral_constraint=all_config.get('use_spectral_constraint', True),
            gamma=all_config.get('gamma', 0.95),
            use_error_feedback=all_config.get('use_error_feedback', True),
            error_beta=all_config.get('error_beta', 0.9),
            loss_type=all_config.get('loss_type', 'cross_entropy'),
        )
    
    elif optimizer_name == 'SDMEP':
        return sdmep(
            model.parameters(),
            model=model,
            mode='ep',
            lr=all_config.get('lr', 0.02),
            beta=all_config.get('beta', 0.3),
            settle_steps=all_config.get('settle_steps', 15),
            settle_lr=all_config.get('settle_lr', 0.02),
            ns_steps=all_config.get('ns_steps', 5),
            use_spectral_constraint=all_config.get('use_spectral_constraint', True),
            gamma=all_config.get('gamma', 0.95),
            use_error_feedback=all_config.get('use_error_feedback', True),
            error_beta=all_config.get('error_beta', 0.9),
            loss_type=all_config.get('loss_type', 'cross_entropy'),
            rank_frac=all_config.get('rank_frac', 0.2),
            dion_thresh=all_config.get('dion_thresh', 100000),
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    is_ep: bool = False
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        if is_ep:
            # EP mode
            # Type ignore because 'step' arguments vary
            optimizer.step(x=x, target=y) # type: ignore
            
            # Compute loss for logging (extra forward pass)
            with torch.no_grad():
                output = model(x)
                loss = F.cross_entropy(output, y)
                pred = output.argmax(dim=1)
        else:
            # Standard backprop
            output = model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1)
        
        total_loss += loss.item() * x.size(0)
        correct += (pred == y).sum().item()
        total += x.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


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
        total += x.size(0)
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy


def get_spectral_norm(model: nn.Module, device: torch.device) -> float:
    """Estimate spectral norm of first weight layer."""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight
            if weight.ndim > 2:
                weight = weight.view(weight.shape[0], -1)
            
            # Power iteration
            u = torch.randn(weight.shape[0], device=device)
            v = torch.randn(weight.shape[1], device=device)
            for _ in range(5):
                v = F.normalize(torch.mv(weight.T, u), dim=0, eps=1e-8)
                u = F.normalize(torch.mv(weight, v), dim=0, eps=1e-8)
            
            return float(torch.dot(u, torch.mv(weight, v)).item())
    
    return 0.0


def run_benchmark(
    optimizer_name: str,
    config: Dict[str, Any],
    device: torch.device,
    verbose: bool = True
) -> BenchmarkResult:
    """Run complete benchmark for one optimizer."""
    
    # Create model
    model = create_model(config['model']['architecture'], device)
    
    # Create dataloaders
    train_loader, test_loader = get_dataloader(
        dataset_name=config['dataset']['name'],
        batch_size=config['training']['batch_size'],
        root=config['data']['root'],
        subset_size=config['dataset'].get('subset_size'),
        num_workers=config['data'].get('num_workers', 4)
    )
    
    # Create optimizer
    optimizer = create_optimizer(optimizer_name, model, config)
    is_ep = optimizer_name in ['SMEP', 'SDMEP']
    
    # Training loop
    result = BenchmarkResult(
        config=config,
        optimizer_name=optimizer_name
    )
    
    epochs = config['training']['epochs']
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, device, is_ep)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, test_loader, device)
        
        # Spectral norm (for monitoring)
        spec_norm = get_spectral_norm(model, device)
        
        epoch_time = time.time() - epoch_start
        
        # Record metrics
        metrics = BenchmarkMetrics(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_time=epoch_time,
            spectral_norm=spec_norm
        )
        result.metrics.append(metrics)
        
        # Update result stats
        result.final_train_acc = train_acc
        result.final_val_acc = val_acc
        result.best_val_acc = max(result.best_val_acc, val_acc)
        
        if verbose:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}s")
    
    result.total_time = time.time() - start_time
    
    return result


def plot_results(
    results: List[BenchmarkResult],
    save_dir: str,
    config: Dict[str, Any]
) -> None:
    """Generate comparison plots from benchmark results."""
    
    if not VIS_AVAILABLE:
        print("matplotlib/seaborn not available. Skipping plots.")
        return
    
    # Set style
    try:
        sns.set_style("whitegrid")
    except:
        pass
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Extract data
    optimizer_names = [r.optimizer_name for r in results]
    
    # Plot 1: Training curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training accuracy
    ax = axes[0, 0]
    for result in results:
        epochs = [m.epoch for m in result.metrics]
        train_accs = [m.train_acc for m in result.metrics]
        ax.plot(epochs, train_accs, marker='o', label=result.optimizer_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy Over Time')
    ax.legend()
    
    # Validation accuracy
    ax = axes[0, 1]
    for result in results:
        epochs = [m.epoch for m in result.metrics]
        val_accs = [m.val_acc for m in result.metrics]
        ax.plot(epochs, val_accs, marker='s', label=result.optimizer_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Validation Accuracy Over Time')
    ax.legend()
    
    # Training loss
    ax = axes[1, 0]
    for result in results:
        epochs = [m.epoch for m in result.metrics]
        train_losses = [m.train_loss for m in result.metrics]
        ax.plot(epochs, train_losses, marker='o', label=result.optimizer_name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    
    # Spectral norm (if available)
    ax = axes[1, 1]
    has_spectral = any(m.spectral_norm is not None for r in results for m in r.metrics)
    if has_spectral:
        for result in results:
            epochs = [m.epoch for m in result.metrics]
            spec_norms = [m.spectral_norm for m in result.metrics]
            ax.plot(epochs, spec_norms, marker='o', label=result.optimizer_name)
        ax.axhline(y=0.95, color='r', linestyle='--', label='Gamma=0.95')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Spectral Norm')
        ax.set_title('Spectral Norm Over Time')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No spectral norm data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Spectral Norm (N/A)')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(save_dir) / "training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Final comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(results))
    final_accs = [r.best_val_acc for r in results]
    times = [r.total_time for r in results]
    
    # Normalize times for display
    max_time = max(times) if times else 1.0
    if max_time == 0: max_time = 1.0
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x, final_accs, alpha=0.7, label='Best Val Accuracy', color='steelblue')
    bars2 = ax2.bar(x, [t/max_time*100 for t in times], alpha=0.7, label='Time (normalized)', color='coral')
    
    ax.set_xlabel('Optimizer')
    ax.set_ylabel('Best Validation Accuracy (%)', color='steelblue')
    ax2.set_ylabel('Time (normalized, %)', color='coral')
    ax.set_title('Optimizer Comparison: Accuracy vs Time')
    ax.set_xticks(list(x))
    ax.set_xticklabels(optimizer_names)
    
    # Add value labels
    for bar, acc, t in zip(bars1, final_accs, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = Path(save_dir) / "optimizer_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/")


def save_results(results: List[BenchmarkResult], save_dir: str, config: Dict[str, Any]) -> None:
    """Save results to JSON file."""
    save_path = Path(save_dir) / "results.json"
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "results": [r.to_dict() for r in results]
    }
    
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {save_path}")


def run_all_benchmarks(config: Dict[str, Any]) -> List[BenchmarkResult]:
    """Run benchmarks for all specified optimizers."""
    
    # Determine device
    device_str = config.get('experiment', {}).get('device', 'auto')
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Running benchmarks on {device}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Optimizers: {list(config.get('optimizers', {}).keys())}")
    
    results = []
    
    for optimizer_name in config.get('optimizers', {}).keys():
        print(f"\n{'='*50}")
        print(f"Benchmarking: {optimizer_name}")
        print('='*50)
        
        result = run_benchmark(
            optimizer_name=optimizer_name,
            config=config,
            device=device,
            verbose=config.get('logging', {}).get('verbose', True)
        )
        results.append(result)
    
    return results


def main() -> None:
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description='MEP Benchmark Runner')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--repeats', type=int, default=None,
                        help='Number of repeats for statistical significance')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.repeats:
        config['experiment']['repeats'] = args.repeats
    
    # Set output directory
    if args.output:
        config['logging']['save_dir'] = args.output
    
    save_dir = config['logging']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Run benchmarks
    all_results = []
    repeats = config.get('experiment', {}).get('repeats', 1)
    
    for repeat in range(repeats):
        print(f"\n{'#'*60}")
        print(f"# Repeat {repeat + 1}/{repeats}")
        print('#'*60)
        
        results = run_all_benchmarks(config)
        all_results.extend(results)
    
    # Aggregate results if multiple repeats
    if repeats > 1:
        # Group by optimizer and compute statistics
        # For simplicity, just use the last run for plots
        final_results = results
    else:
        final_results = all_results
    
    # Save results
    save_results(final_results, save_dir, config)
    
    # Generate plots
    if not args.no_plots:
        plot_results(final_results, save_dir, config)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for result in final_results:
        print(f"\n{result.optimizer_name}:")
        print(f"  Best Val Accuracy: {result.best_val_acc:.2f}%")
        print(f"  Total Time: {result.total_time:.2f}s")


if __name__ == '__main__':
    main()
