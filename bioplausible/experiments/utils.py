"""
Bioplausible Experimentation Utilities

Comprehensive utilities for experimentation, research, and discovery
of novel machine learning approaches using the Bioplausible framework.

Features:
- Model/Optimizer comparison utilities
- Experiment workflow helpers
- Hyperparameter search utilities
- Validation and benchmarking tools
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    model_name: str
    optimizer_name: str
    model_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    
    # Performance metrics
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    test_accuracy: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    
    # Timing
    training_time: float = 0.0  # seconds
    steps_per_second: float = 0.0
    
    # Resource usage
    num_parameters: int = 0
    memory_peak_mb: float = 0.0
    
    # Additional metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Get a summary string."""
        return (
            f"{self.model_name} + {self.optimizer_name}:\n"
            f"  Train Acc: {self.train_accuracy:.2f}%, Val Acc: {self.val_accuracy:.2f}%\n"
            f"  Training Time: {self.training_time:.1f}s, Steps/s: {self.steps_per_second:.1f}\n"
            f"  Parameters: {self.num_parameters:,}"
        )


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    model_name: str
    optimizer_name: str
    model_params: Dict[str, Any] = field(default_factory=dict)
    optimizer_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training config
    epochs: int = 10
    batches_per_epoch: int = 100
    eval_batches: int = 20
    device: str = 'auto'
    
    # Tracking
    track_metrics: bool = True
    verbose: bool = True


class ExperimentRunner:
    """
    Run controlled experiments for model/optimizer combinations.
    
    Example usage:
        runner = ExperimentRunner()
        
        # Run single experiment
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name='smep',
            train_loader=train_loader,
            val_loader=val_loader,
        )
        
        # Compare multiple optimizers
        results = runner.compare_optimizers(
            model_name='looped_mlp',
            optimizer_names=['smep', 'smep_fast', 'muon_backprop'],
            train_loader=train_loader,
            val_loader=val_loader,
        )
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def run(
        self,
        model_name: str,
        optimizer_name: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        model_params: Optional[Dict[str, Any]] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        epochs: int = 10,
        batches_per_epoch: int = 100,
        eval_batches: int = 20,
        verbose: bool = True,
    ) -> ExperimentResult:
        """
        Run a single experiment.
        
        Args:
            model_name: Name of model from Zoo.
            optimizer_name: Name of optimizer from Zoo.
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
            test_loader: Test data loader (optional).
            model_params: Override default model parameters.
            optimizer_params: Override default optimizer parameters.
            epochs: Number of training epochs.
            batches_per_epoch: Batches per epoch.
            eval_batches: Batches for evaluation.
            verbose: Print progress.
        
        Returns:
            ExperimentResult with metrics.
        """
        from bioplausible.zoo import ModelZoo, OptimizerZoo
        
        # Get model and optimizer
        model_params = model_params or {}
        optimizer_params = optimizer_params or {}
        
        model = ModelZoo.get(model_name, **model_params)
        model = model.to(self.device)
        
        optimizer = OptimizerZoo.get(
            optimizer_name,
            model.parameters(),
            model=model,
            **optimizer_params
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Training loop
        start_time = time.time()
        total_steps = 0
        train_losses = []
        train_correct = 0
        train_total = 0
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                if batch_idx >= batches_per_epoch:
                    break
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Flatten for MLP models
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)
                
                # Optimizer step
                try:
                    # MEP-style optimizers
                    optimizer.step(x=x, target=y)
                except TypeError:
                    # Standard PyTorch optimizers
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                # Track metrics
                with torch.no_grad():
                    output = model(x)
                    loss = nn.functional.cross_entropy(output, y).item()
                    epoch_loss += loss
                    train_losses.append(loss)
                    
                    pred = output.argmax(dim=1)
                    train_correct += (pred == y).sum().item()
                    train_total += y.shape[0]
                
                epoch_steps += 1
                total_steps += 1
            
            if verbose:
                avg_loss = epoch_loss / max(1, epoch_steps)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluation
        val_accuracy = 0.0
        val_loss = 0.0
        
        if val_loader is not None:
            val_accuracy, val_loss = self._evaluate(
                model, val_loader, eval_batches
            )
        
        test_accuracy = 0.0
        if test_loader is not None:
            test_accuracy, _ = self._evaluate(model, test_loader, eval_batches)
        
        # Create result
        result = ExperimentResult(
            model_name=model_name,
            optimizer_name=optimizer_name,
            model_params=model_params,
            optimizer_params=optimizer_params,
            train_accuracy=100.0 * train_correct / max(1, train_total),
            val_accuracy=val_accuracy,
            test_accuracy=test_accuracy,
            train_loss=np.mean(train_losses) if train_losses else 0.0,
            val_loss=val_loss,
            training_time=training_time,
            steps_per_second=total_steps / max(0.001, training_time),
            num_parameters=num_params,
        )
        
        if verbose:
            print(result.summary())
        
        return result
    
    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        max_batches: int = 20,
    ) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        batches = 0
        
        with torch.no_grad():
            for x, y in loader:
                if batches >= max_batches:
                    break
                
                x = x.to(self.device)
                y = y.to(self.device)
                
                if len(x.shape) > 2:
                    x = x.view(x.shape[0], -1)
                
                output = model(x)
                loss = nn.functional.cross_entropy(output, y).item()
                
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.shape[0]
                total_loss += loss
                batches += 1
        
        model.train()
        
        accuracy = 100.0 * correct / max(1, total)
        avg_loss = total_loss / max(1, batches)
        
        return accuracy, avg_loss
    
    def compare_optimizers(
        self,
        model_name: str,
        optimizer_names: List[str],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        model_params: Optional[Dict[str, Any]] = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> List[ExperimentResult]:
        """
        Compare multiple optimizers on the same model.
        
        Args:
            model_name: Name of model from Zoo.
            optimizer_names: List of optimizer names to compare.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model_params: Model parameters.
            epochs: Training epochs per optimizer.
            verbose: Print progress.
        
        Returns:
            List of ExperimentResult, sorted by validation accuracy.
        """
        results = []
        
        for opt_name in optimizer_names:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testing optimizer: {opt_name}")
                print(f"{'='*60}")
            
            result = self.run(
                model_name=model_name,
                optimizer_name=opt_name,
                train_loader=train_loader,
                val_loader=val_loader,
                model_params=model_params,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result)
        
        # Sort by validation accuracy
        results.sort(key=lambda r: r.val_accuracy, reverse=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print("COMPARISON RESULTS (sorted by val accuracy)")
            print(f"{'='*60}")
            for i, r in enumerate(results):
                print(f"{i+1}. {r.optimizer_name}: {r.val_accuracy:.2f}%")
        
        return results
    
    def compare_models(
        self,
        model_names: List[str],
        optimizer_name: str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer_params: Optional[Dict[str, Any]] = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> List[ExperimentResult]:
        """
        Compare multiple models with the same optimizer.
        
        Args:
            model_names: List of model names to compare.
            optimizer_name: Name of optimizer.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            optimizer_params: Optimizer parameters.
            epochs: Training epochs per model.
            verbose: Print progress.
        
        Returns:
            List of ExperimentResult, sorted by validation accuracy.
        """
        results = []
        
        for model_name in model_names:
            if verbose:
                print(f"\n{'='*60}")
                print(f"Testing model: {model_name}")
                print(f"{'='*60}")
            
            result = self.run(
                model_name=model_name,
                optimizer_name=optimizer_name,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer_params=optimizer_params,
                epochs=epochs,
                verbose=verbose,
            )
            results.append(result)
        
        # Sort by validation accuracy
        results.sort(key=lambda r: r.val_accuracy, reverse=True)
        
        if verbose:
            print(f"\n{'='*60}")
            print("COMPARISON RESULTS (sorted by val accuracy)")
            print(f"{'='*60}")
            for i, r in enumerate(results):
                print(f"{i+1}. {r.model_name}: {r.val_accuracy:.2f}%")
        
        return results


class HyperparameterSearch:
    """
    Hyperparameter search utilities.
    
    Example usage:
        search = HyperparameterSearch()
        
        # Grid search
        best_result = search.grid_search(
            model_name='looped_mlp',
            optimizer_name='smep',
            param_grid={
                'lr': [0.001, 0.01, 0.1],
                'settle_steps': [10, 30],
                'beta': [0.3, 0.5],
            },
            train_loader=train_loader,
            val_loader=val_loader,
        )
    """
    
    def __init__(self, device: str = 'auto'):
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def grid_search(
        self,
        model_name: str,
        optimizer_name: str,
        param_grid: Dict[str, List[Any]],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        model_params: Optional[Dict[str, Any]] = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], ExperimentResult]:
        """
        Perform grid search over optimizer hyperparameters.
        
        Args:
            model_name: Name of model.
            optimizer_name: Name of optimizer.
            param_grid: Dict of param_name -> list of values.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model_params: Fixed model parameters.
            epochs: Training epochs per configuration.
            verbose: Print progress.
        
        Returns:
            Tuple of (best_params, best_result).
        """
        from bioplausible.zoo import OptimizerZoo
        from itertools import product
        
        runner = ExperimentRunner(device=self.device)
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_configs = []
        for values in product(*param_values):
            config = dict(zip(param_names, values))
            all_configs.append(config)
        
        if verbose:
            print(f"Grid search: {len(all_configs)} configurations")
        
        best_result = None
        best_params = None
        best_accuracy = -1.0
        
        for i, config in enumerate(all_configs):
            if verbose:
                print(f"\n[{i+1}/{len(all_configs)}] Testing: {config}")
            
            # Merge with base optimizer params
            optimizer_params = {**(model_params or {}), **config}
            
            try:
                result = runner.run(
                    model_name=model_name,
                    optimizer_name=optimizer_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    epochs=epochs,
                    verbose=False,
                )
                
                if result.val_accuracy > best_accuracy:
                    best_accuracy = result.val_accuracy
                    best_params = config
                    best_result = result
                    
            except Exception as e:
                if verbose:
                    print(f"  Failed: {e}")
                continue
        
        if verbose:
            print(f"\nBest params: {best_params}")
            print(f"Best val accuracy: {best_accuracy:.2f}%")
        
        return best_params, best_result


def quick_comparison(
    model_name: str = 'looped_mlp',
    optimizer_names: Optional[List[str]] = None,
    epochs: int = 3,
    verbose: bool = True,
) -> List[ExperimentResult]:
    """
    Quick comparison of optimizers on MNIST.
    
    This is a convenience function for rapid experimentation.
    
    Args:
        model_name: Model to test.
        optimizer_names: Optimizers to compare (default: all MEP).
        epochs: Training epochs.
        verbose: Print progress.
    
    Returns:
        List of results sorted by accuracy.
    """
    from bioplausible.datasets import get_vision_dataset
    
    if optimizer_names is None:
        optimizer_names = ['smep', 'smep_fast', 'muon_backprop']
    
    # Load MNIST
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    runner = ExperimentRunner()
    
    return runner.compare_optimizers(
        model_name=model_name,
        optimizer_names=optimizer_names,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        verbose=verbose,
    )


def benchmark_model(
    model_name: str,
    optimizer_name: str = 'smep',
    epochs: int = 10,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Benchmark a model/optimizer combination on MNIST.
    
    Args:
        model_name: Model to benchmark.
        optimizer_name: Optimizer to use.
        epochs: Training epochs.
        verbose: Print progress.
    
    Returns:
        ExperimentResult with benchmark metrics.
    """
    from bioplausible.datasets import get_vision_dataset
    
    train_loader, val_loader, test_loader = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    runner = ExperimentRunner()
    
    return runner.run(
        model_name=model_name,
        optimizer_name=optimizer_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=epochs,
        verbose=verbose,
    )


__all__ = [
    'ExperimentResult',
    'ExperimentConfig',
    'ExperimentRunner',
    'HyperparameterSearch',
    'quick_comparison',
    'benchmark_model',
]
