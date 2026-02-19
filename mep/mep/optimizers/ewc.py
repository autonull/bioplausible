"""
EWC (Elastic Weight Consolidation) for EP

Phase 2: Priority 3 - Continual Learning

EWC adds a regularization term to prevent catastrophic forgetting:
    L_total = L_current + λ × Σ F_i × (θ_i - θ*_i)²

where:
    F_i = Fisher information diagonal (parameter importance)
    θ*_i = Optimal parameters from previous task
    λ = EWC weight (typically 100-1000)

Author: Phase 2 Implementation
Created: 2026-03-04
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TaskMemory:
    """Stored information for a completed task."""
    task_id: int
    fisher: Dict[str, torch.Tensor]  # Fisher information diagonal
    optimal_params: Dict[str, torch.Tensor]  # Parameter values after training
    dataset_size: int  # Number of samples in task


class EWCRegularizer:
    """
    EWC regularization for continual learning.
    
    Computes Fisher information after each task and adds
    regularization term during training on subsequent tasks.
    
    Usage:
        ewc = EWCRegularizer(model, ewc_lambda=100)
        
        # After training on task 1
        ewc.update_fisher(train_loader, task_id=0)
        
        # During training on task 2+
        loss = criterion(output, target)
        ewc_loss = ewc.compute_ewc_loss()
        total_loss = loss + ewc_loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 100.0,
        fisher_damping: float = 1e-3,
    ):
        """
        Initialize EWC regularizer.
        
        Args:
            model: Neural network model.
            ewc_lambda: EWC regularization weight (higher = less forgetting).
            fisher_damping: Damping term for Fisher (prevents division by zero).
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_damping = fisher_damping
        
        # Store task memories
        self.task_memories: Dict[int, TaskMemory] = {}
        self._current_task: Optional[int] = None
    
    def update_fisher(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_id: int,
        device: str = 'cuda',
        loss_type: str = 'cross_entropy',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information after completing a task.
        
        Args:
            data_loader: DataLoader for task data.
            task_id: Task identifier.
            device: Device to compute on.
            loss_type: 'cross_entropy' or 'mse'.
        
        Returns:
            Fisher information diagonal for each parameter.
        """
        self.model.eval()
        
        # Initialize Fisher accumulators
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        total_samples = 0
        
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                x, y = batch[0].to(device), batch[1].to(device)
            else:
                x, y = batch.to(device), None
            
            # Forward pass with gradients enabled
            x.requires_grad_(False)  # Don't need input gradients
            
            with torch.enable_grad():
                output = self.model(x)
                
                # Compute loss
                if loss_type == 'cross_entropy':
                    if y is None:
                        # Unsupervised - use entropy
                        probs = F.softmax(output, dim=1)
                        loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                    else:
                        loss = F.cross_entropy(output, y)
                else:
                    # MSE - need target
                    if y is None:
                        continue
                    loss = F.mse_loss(output, y.float())
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                self.model.parameters(),
                retain_graph=False,
                allow_unused=True,
            )
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            batch_size = x.size(0)
            for (n, p), g in zip(self.model.named_parameters(), grads):
                if g is not None:
                    fisher[n] += (g ** 2) * batch_size
            
            total_samples += batch_size
        
        # Normalize by dataset size
        for n in fisher:
            fisher[n] /= total_samples
            fisher[n] += self.fisher_damping  # Damping
        
        # Store optimal parameters
        optimal_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        # Store task memory
        self.task_memories[task_id] = TaskMemory(
            task_id=task_id,
            fisher=fisher,
            optimal_params=optimal_params,
            dataset_size=total_samples,
        )
        
        self._current_task = task_id
        
        return fisher
    
    def compute_ewc_loss(self, include_all_tasks: bool = True) -> torch.Tensor:
        """
        Compute EWC regularization loss.
        
        Args:
            include_all_tasks: If True, include all previous tasks.
                              If False, only include most recent previous task.
        
        Returns:
            EWC regularization term (scalar tensor).
        """
        if not self.task_memories:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        # Determine which tasks to include
        if include_all_tasks:
            tasks_to_include = list(self.task_memories.keys())
        else:
            # Only most recent previous task
            if self._current_task is not None:
                prev_task = self._current_task - 1
                tasks_to_include = [prev_task] if prev_task in self.task_memories else []
            else:
                tasks_to_include = []
        
        for task_id in tasks_to_include:
            if task_id == self._current_task:
                # Don't regularize against current task
                continue
            
            memory = self.task_memories[task_id]
            
            for n, p in self.model.named_parameters():
                if n in memory.fisher and p.requires_grad:
                    fisher = memory.fisher[n]
                    optimal = memory.optimal_params[n]
                    
                    # EWC penalty: F_i × (θ_i - θ*_i)²
                    ewc_loss += (fisher * (p - optimal) ** 2).sum()
        
        # Scale by lambda and 0.5 (conventional factor)
        ewc_loss = ewc_loss * self.ewc_lambda * 0.5
        
        return ewc_loss
    
    def compute_ewc_loss_for_task(self, task_id: int) -> torch.Tensor:
        """
        Compute EWC loss for a specific previous task.
        
        Args:
            task_id: Task to compute regularization against.
        
        Returns:
            EWC regularization term.
        """
        if task_id not in self.task_memories:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        memory = self.task_memories[task_id]
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for n, p in self.model.named_parameters():
            if n in memory.fisher and p.requires_grad:
                fisher = memory.fisher[n]
                optimal = memory.optimal_params[n]
                ewc_loss += (fisher * (p - optimal) ** 2).sum()
        
        return ewc_loss * self.ewc_lambda * 0.5
    
    def get_forgetting_measure(self, task_id: int) -> Dict[str, float]:
        """
        Measure how much the model has forgotten from a task.
        
        Args:
            task_id: Task to measure forgetting for.
        
        Returns:
            Dictionary with forgetting metrics.
        """
        if task_id not in self.task_memories:
            return {'error': 'Task not found'}
        
        memory = self.task_memories[task_id]
        
        # Compute parameter drift
        total_drift = 0.0
        weighted_drift = 0.0
        param_count = 0
        
        for n, p in self.model.named_parameters():
            if n in memory.optimal_params:
                optimal = memory.optimal_params[n]
                fisher = memory.fisher[n]
                
                # Unweighted drift
                drift = (p - optimal).abs().mean().item()
                total_drift += drift
                param_count += 1
                
                # Fisher-weighted drift (importance-weighted forgetting)
                weighted_drift += (fisher * (p - optimal) ** 2).sum().item()
        
        return {
            'task_id': task_id,
            'avg_param_drift': total_drift / max(1, param_count),
            'weighted_drift': weighted_drift,
            'ewc_penalty': weighted_drift * self.ewc_lambda * 0.5,
        }
    
    def state_dict(self) -> Dict:
        """Save EWC state."""
        return {
            'task_memories': {
                tid: {
                    'fisher': mem.fisher,
                    'optimal_params': mem.optimal_params,
                    'dataset_size': mem.dataset_size,
                }
                for tid, mem in self.task_memories.items()
            },
            'current_task': self._current_task,
            'ewc_lambda': self.ewc_lambda,
        }
    
    def load_state_dict(self, state: Dict):
        """Load EWC state."""
        self.task_memories = {
            tid: TaskMemory(
                task_id=tid,
                fisher=mem['fisher'],
                optimal_params=mem['optimal_params'],
                dataset_size=mem['dataset_size'],
            )
            for tid, mem in state['task_memories'].items()
        }
        self._current_task = state['current_task']
        self.ewc_lambda = state['ewc_lambda']


class EPOptimizerWithEWC:
    """
    EP optimizer with integrated EWC regularization.
    
    Wraps smep/O1MemoryEPv2 and adds EWC loss to the contrast step.
    
    Usage:
        optimizer = EPOptimizerWithEWC(
            model.parameters(),
            model=model,
            ewc_lambda=100,
        )
        
        # Train on task 1
        for epoch in range(epochs):
            for x, y in train_loader:
                optimizer.step(x=x, target=y)
        
        # Consolidate task 1
        optimizer.consolidate_task(train_loader, task_id=0)
        
        # Train on task 2 (with EWC regularization)
        for epoch in range(epochs):
            for x, y in train_loader:
                optimizer.step(x=x, target=y, task_id=1)
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        ewc_lambda: float = 100.0,
        settle_steps: int = 10,
        settle_lr: float = 0.2,
        beta: float = 0.5,
        loss_type: str = 'cross_entropy',
        use_analytic: bool = True,
    ):
        """
        Initialize EP optimizer with EWC.
        
        Args:
            params: Parameters to optimize.
            model: Model instance.
            lr: Learning rate.
            ewc_lambda: EWC regularization weight.
            settle_steps: EP settling steps.
            settle_lr: Settling learning rate.
            beta: EP nudging strength.
            loss_type: Loss type for EP.
            use_analytic: Use analytic gradients (faster).
        """
        self.model = model
        self.lr = lr
        self.ewc = EWCRegularizer(model, ewc_lambda=ewc_lambda)
        
        if use_analytic:
            from mep.optimizers import O1MemoryEPv2
            self.ep_optimizer = O1MemoryEPv2(
                params,
                model=model,
                lr=lr,
                settle_steps=settle_steps,
                settle_lr=settle_lr,
                beta=beta,
                loss_type=loss_type,
            )
        else:
            from mep import smep
            self.ep_optimizer = smep(
                params,
                model=model,
                lr=lr,
                mode='ep',
                settle_steps=settle_steps,
                settle_lr=settle_lr,
                beta=beta,
                loss_type=loss_type,
            )
        
        self._current_task: Optional[int] = None
        self._ewc_weight = 1.0  # Can be adjusted dynamically
    
    def step(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        task_id: Optional[int] = None,
        use_ewc: bool = True,
    ):
        """
        Training step with optional EWC regularization.
        
        Args:
            x: Input tensor.
            target: Target tensor.
            task_id: Current task ID (for EWC).
            use_ewc: Whether to apply EWC regularization.
        """
        if task_id is not None:
            self._current_task = task_id
        
        # Get EWC loss if applicable
        ewc_loss = torch.tensor(0.0)
        if use_ewc and len(self.ewc.task_memories) > 0:
            ewc_loss = self.ewc.compute_ewc_loss()
        
        # EP step with EWC modification
        # We need to add EWC gradient to the contrast step
        
        if isinstance(self.ep_optimizer, type(self.ep_optimizer)):
            # For O1MemoryEPv2 or smep, we do a custom step
            self._ep_step_with_ewc(x, target, ewc_loss)
        else:
            # Fallback: standard EP step
            self.ep_optimizer.step(x=x, target=target)
    
    def _ep_step_with_ewc(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        ewc_loss: torch.Tensor,
    ):
        """Custom EP step with EWC gradient added to contrast."""
        from mep.optimizers import settle_manual_o1, energy_from_states_minimal
        
        # Settling phases (no EWC needed here)
        states_free = settle_manual_o1(
            self.model, x, None, beta=0.0,
            structure=self.ep_optimizer.structure,
            steps=self.ep_optimizer.settle_steps,
            lr=self.ep_optimizer.settle_lr,
            loss_type=self.ep_optimizer.loss_type,
        )
        
        states_nudged = settle_manual_o1(
            self.model, x, target, beta=self.ep_optimizer.beta,
            structure=self.ep_optimizer.structure,
            steps=self.ep_optimizer.settle_steps,
            lr=self.ep_optimizer.settle_lr,
            loss_type=self.ep_optimizer.loss_type,
        )
        
        # Contrast step with EWC
        E_free = energy_from_states_minimal(
            self.model, x, states_free, self.ep_optimizer.structure,
            None, 0.0, loss_type=self.ep_optimizer.loss_type,
        )
        
        E_nudged = energy_from_states_minimal(
            self.model, x, states_nudged, self.ep_optimizer.structure,
            target, self.ep_optimizer.beta, loss_type=self.ep_optimizer.loss_type,
        )
        
        contrast_loss = (E_nudged - E_free) / self.ep_optimizer.beta
        
        # Add EWC regularization
        total_loss = contrast_loss + ewc_loss
        
        # Compute gradients
        grads = torch.autograd.grad(
            total_loss,
            list(self.model.parameters()),
            retain_graph=False,
        )
        
        # Update parameters
        with torch.no_grad():
            for p, g in zip(self.model.parameters(), grads):
                p.sub_(g, alpha=self.lr)
    
    def consolidate_task(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_id: int,
        device: str = 'cuda',
    ):
        """
        Consolidate a completed task by computing Fisher information.
        
        Call this after training on each task.
        
        Args:
            data_loader: DataLoader for task data.
            task_id: Task identifier.
            device: Device to compute on.
        """
        self.ewc.update_fisher(data_loader, task_id, device)
        self._current_task = task_id
    
    def set_ewc_lambda(self, lambda_value: float):
        """Adjust EWC regularization weight."""
        self.ewc.ewc_lambda = lambda_value
    
    def get_forgetting(self, task_id: int) -> Dict[str, float]:
        """Get forgetting measure for a task."""
        return self.ewc.get_forgetting_measure(task_id)
