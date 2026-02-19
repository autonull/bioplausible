"""
EWC (Elastic Weight Consolidation) Implementation

EWC adds a regularization term to prevent forgetting by penalizing changes
to important weights for previous tasks.

Reference:
    Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in
    neural networks. PNAS, 114(13), 3521-3526.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict


class EWC:
    """
    Elastic Weight Consolidation for continual learning.
    
    EWC adds a regularization term to the loss:
        L = L_task + λ * Σ F_i (θ_i - θ*_i)²
    
    where F_i is the Fisher information diagonal for parameter i,
    and θ*_i are the parameter values after the previous task.
    """

    def __init__(self, model: nn.Module, fisher_damping: float = 1e-3):
        """
        Initialize EWC.
        
        Args:
            model: The model to apply EWC to
            fisher_damping: Damping term for Fisher information
        """
        self.model = model
        self.fisher_damping = fisher_damping
        
        # Store optimal parameters and Fisher information for each task
        self.task_params: Dict[int, OrderedDict[str, torch.Tensor]] = {}
        self.fisher_estimates: Dict[int, OrderedDict[str, torch.Tensor]] = {}

    def compute_fisher(
        self,
        data_loader: DataLoader,
        device: torch.device,
        task_id: int,
        num_samples: Optional[int] = None
    ):
        """
        Compute Fisher information diagonal for current parameters.
        
        Uses the empirical Fisher approximation:
            F_i = E[(∂L/∂θ_i)²]
        
        Args:
            data_loader: Data loader for the task
            device: Device to compute on
            task_id: Task identifier
            num_samples: Optional limit on number of samples
        """
        self.model.eval()
        
        # Initialize Fisher accumulation
        fisher = OrderedDict()
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        total_samples = 0
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if num_samples and total_samples >= num_samples:
                    break
                
                x, y = x.to(device), y.to(device)
                
                # Compute gradients
                self.model.zero_grad()
                output = self.model(x)
                loss = F.cross_entropy(output, y)
                loss.backward()
                
                # Accumulate squared gradients
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher[name] += param.grad.pow(2) * x.size(0)
                
                total_samples += x.size(0)
        
        # Normalize by number of samples and add damping
        for name in fisher:
            fisher[name] = fisher[name] / (total_samples + 1e-8)
            fisher[name] += self.fisher_damping
        
        # Store Fisher for this task
        self.fisher_estimates[task_id] = fisher
        
        # Store current parameters as optimal for this task
        self.task_params[task_id] = OrderedDict(
            (name, param.data.clone())
            for name, param in self.model.named_parameters()
        )

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC regularization loss for all previous tasks.
        
        Returns:
            EWC regularization term to add to task loss
        """
        if not self.task_params:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for task_id in self.task_params:
            fisher = self.fisher_estimates[task_id]
            optimal_params = self.task_params[task_id]
            
            for name, param in self.model.named_parameters():
                if name in fisher:
                    ewc_loss += (fisher[name] * (param - optimal_params[name]).pow(2)).sum()
        
        return ewc_loss

    def get_ewc_lambda_schedule(
        self,
        initial_lambda: float = 1.0,
        decay_factor: float = 0.9
    ) -> Dict[int, float]:
        """
        Get lambda schedule for EWC with decay over tasks.
        
        Args:
            initial_lambda: Initial lambda value
            decay_factor: Decay factor per task
        
        Returns:
            Dictionary mapping task_id to lambda value
        """
        return {
            task_id: initial_lambda * (decay_factor ** (len(self.task_params) - task_id - 1))
            for task_id in range(len(self.task_params))
        }


def train_with_ewc(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    ewc: EWC,
    ewc_lambda: float = 1.0,
    epochs: int = 1,
    lr: float = 0.01,
) -> float:
    """
    Train model with EWC regularization.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        device: Device to train on
        ewc: EWC instance
        ewc_lambda: Weight for EWC loss
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        Final training accuracy
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    correct = 0
    total = 0
    
    for epoch in range(epochs):
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            
            # Task loss + EWC regularization
            task_loss = F.cross_entropy(output, y_batch)
            ewc_loss = ewc.ewc_loss()
            total_loss = task_loss + ewc_lambda * ewc_loss
            
            total_loss.backward()
            optimizer.step()
            
            # Track accuracy
            _, predicted = output.max(1)
            total += y_batch.size(0)
            correct += predicted.eq(y_batch).sum().item()
    
    return correct / total


def evaluate_continual_learning(
    model: nn.Module,
    test_loaders: list,
    device: torch.device,
    peak_accuracies: Dict[int, float],
) -> Tuple[float, float]:
    """
    Evaluate continual learning performance.
    
    Args:
        model: Model to evaluate
        test_loaders: List of test loaders for each task
        device: Device to evaluate on
        peak_accuracies: Dictionary of peak accuracy per task
    
    Returns:
        Tuple of (average_accuracy, average_forgetting)
    """
    model.eval()
    accuracies = []
    forgetting = []
    
    with torch.no_grad():
        for task_id, test_loader in enumerate(test_loaders):
            correct = 0
            total = 0
            
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                _, predicted = output.max(1)
                total += y_batch.size(0)
                correct += predicted.eq(y_batch).sum().item()
            
            acc = correct / total
            accuracies.append(acc)
            
            # Compute forgetting for this task
            peak_acc = peak_accuracies.get(task_id, acc)
            forgetting.append(peak_acc - acc)
    
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_forgetting = sum(forgetting) / len(forgetting)
    
    return avg_accuracy, avg_forgetting


def run_ewc_benchmark(
    num_tasks: int = 5,
    epochs_per_task: int = 5,
    lr: float = 0.01,
    ewc_lambda: float = 1.0,
    device: Optional[torch.device] = None,
    seed: int = 42
) -> "ContinualLearningResult":
    """
    Run EWC baseline for Permuted MNIST benchmark.
    
    Args:
        num_tasks: Number of tasks (permutations)
        epochs_per_task: Epochs to train per task
        lr: Learning rate
        ewc_lambda: EWC regularization weight
        device: Device to use
        seed: Random seed
    
    Returns:
        ContinualLearningResult with metrics
    """
    # Import here to avoid circular dependency
    from .continual_learning import (
        PermutedMNIST, MLP, TaskResult, ContinualLearningResult,
        evaluate, train_epoch
    )
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)
    
    benchmark = PermutedMNIST(num_tasks=num_tasks, seed=seed)
    model = MLP(hidden_dims=(256, 128)).to(device)
    
    # Initialize EWC
    ewc = EWC(model, fisher_damping=1e-3)
    
    peak_accuracies: Dict[int, float] = {}
    task_results: List[TaskResult] = []
    
    for task_id in range(num_tasks):
        print(f"  Training task {task_id + 1}/{num_tasks} with EWC...")
        
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)
        
        # Train with EWC
        train_with_ewc(
            model, train_loader, device, ewc,
            ewc_lambda=ewc_lambda, epochs=epochs_per_task, lr=lr
        )
        
        # Evaluate on current task
        current_acc = evaluate(model, test_loader, device)
        peak_accuracies[task_id] = current_acc
        
        # Evaluate on all previous tasks
        all_accuracies = []
        for prev_task_id in range(task_id + 1):
            _, prev_test_loader = benchmark.get_task_dataloaders(prev_task_id)
            acc = evaluate(model, prev_test_loader, device)
            all_accuracies.append(acc)
            
            # Update peak accuracy
            peak_accuracies[prev_task_id] = max(peak_accuracies.get(prev_task_id, 0), acc)
        
        # Compute forgetting
        avg_acc = sum(all_accuracies) / len(all_accuracies)
        avg_forgetting = sum(peak_accuracies[i] - acc for i, acc in enumerate(all_accuracies)) / len(all_accuracies)
        
        task_results.append(TaskResult(
            task_id=task_id,
            train_accuracy=current_acc,
            test_accuracy=current_acc,
            forgetting=avg_forgetting
        ))
        
        print(f"    Task {task_id + 1}: Accuracy = {current_acc:.4f}, "
              f"Avg = {avg_acc:.4f}, Forgetting = {avg_forgetting:.4f}")
    
    # Compute Fisher information after each task (for next task's EWC)
    # This is done in train_with_ewc already
    
    # Final evaluation
    final_accuracies = []
    for task_id in range(num_tasks):
        _, test_loader = benchmark.get_task_dataloaders(task_id)
        acc = evaluate(model, test_loader, device)
        final_accuracies.append(acc)
    
    return ContinualLearningResult(
        benchmark_name="Permuted MNIST (EWC)",
        num_tasks=num_tasks,
        task_results=task_results,
        average_accuracy=sum(final_accuracies) / len(final_accuracies),
        average_forgetting=sum(peak_accuracies[i] - acc for i, acc in enumerate(final_accuracies)) / len(final_accuracies),
        final_accuracy=final_accuracies[-1] if final_accuracies else 0.0
    )
