"""
Unified EP Optimizer

Refactored optimizer that consolidates all EP variants into a single,
well-parameterized interface.

Key parameters:
- mode: 'ep' or 'backprop'
- settle_steps: Number of settling iterations (default: 10)
- gradient_method: 'analytic' (fast) or 'autograd' (exact)
- ewc_lambda: EWC regularization strength (default: 0 = disabled)

Usage:
    # Simple EP (fast)
    opt = EPOptimizer(model.parameters(), model=model)
    
    # EP with EWC for continual learning
    opt = EPOptimizer(model.parameters(), model=model, ewc_lambda=100)
    
    # Backprop (for comparison)
    opt = EPOptimizer(model.parameters(), model=model, mode='backprop')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from .strategies import MuonUpdate, SpectralConstraint, NoConstraint
from .inspector import ModelInspector
from .energy import EnergyFunction


@dataclass
class EPConfig:
    """Configuration for EP optimizer."""
    # Core
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    
    # EP-specific
    mode: str = 'ep'  # 'ep' or 'backprop'
    beta: float = 0.5  # Nudging strength
    settle_steps: int = 10  # Settling iterations
    settle_lr: float = 0.2  # Settling learning rate
    gradient_method: str = 'analytic'  # 'analytic' or 'autograd'
    
    # EWC
    ewc_lambda: float = 0.0  # 0 = disabled
    
    # Muon
    ns_steps: int = 5  # Newton-Schulz iterations
    gamma: float = 0.95  # Spectral norm bound
    
    # Loss
    loss_type: str = 'cross_entropy'
    softmax_temperature: float = 1.0


class EWCState:
    """EWC state for continual learning."""
    
    def __init__(self, model: nn.Module, fisher_damping: float = 1e-3):
        self.model = model
        self.fisher_damping = fisher_damping
        self.task_memories: Dict[int, Dict[str, Any]] = {}
        self._current_task: Optional[int] = None
    
    def update_fisher(self, data_loader, task_id: int, device: str, loss_type: str):
        """Compute Fisher information after completing a task."""
        self.model.eval()
        
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
            
            with torch.enable_grad():
                output = self.model(x)
                if loss_type == 'cross_entropy':
                    if y is not None:
                        loss = F.cross_entropy(output, y)
                    else:
                        probs = F.softmax(output, dim=1)
                        loss = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                else:
                    if y is None:
                        continue
                    loss = F.mse_loss(output, y.float())
            
            grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False, allow_unused=True)
            
            batch_size = x.size(0)
            for (n, p), g in zip(self.model.named_parameters(), grads):
                if g is not None:
                    fisher[n] += (g ** 2) * batch_size
            
            total_samples += batch_size
        
        for n in fisher:
            fisher[n] /= total_samples
            fisher[n] += self.fisher_damping
        
        optimal_params = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        
        self.task_memories[task_id] = {
            'fisher': fisher,
            'optimal_params': optimal_params,
            'dataset_size': total_samples,
        }
        self._current_task = task_id
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.task_memories or self._current_task is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        ewc_loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        for task_id, memory in self.task_memories.items():
            if task_id == self._current_task:
                continue
            
            for n, p in self.model.named_parameters():
                if n in memory['fisher'] and p.requires_grad:
                    fisher = memory['fisher'][n]
                    optimal = memory['optimal_params'][n]
                    ewc_loss += (fisher * (p - optimal) ** 2).sum()
        
        return ewc_loss * 0.5


class EPOptimizer:
    """
    Unified EP Optimizer.
    
    Consolidates smep, smep_fast, O1MemoryEP, and EPOptimizerWithEWC
    into a single, well-parameterized interface.
    
    Parameters:
        params: Parameters to optimize.
        model: Model instance.
        lr: Learning rate (default: 0.01).
        momentum: Momentum factor (default: 0.9).
        weight_decay: Weight decay (default: 0.0005).
        
        # Mode
        mode: 'ep' or 'backprop' (default: 'ep').
        
        # EP settling
        beta: Nudging strength (default: 0.5).
        settle_steps: Settling iterations (default: 10).
        settle_lr: Settling learning rate (default: 0.2).
        gradient_method: 'analytic' (fast) or 'autograd' (default: 'analytic').
        
        # EWC for continual learning
        ewc_lambda: EWC regularization strength (default: 0 = disabled).
        
        # Muon orthogonalization
        ns_steps: Newton-Schulz iterations (default: 5).
        gamma: Spectral norm bound (default: 0.95).
        
        # Loss
        loss_type: 'cross_entropy' or 'mse' (default: 'cross_entropy').
    
    Examples:
        # Fast EP (default settings)
        opt = EPOptimizer(model.parameters(), model=model)
        
        # EP with EWC for continual learning
        opt = EPOptimizer(model.parameters(), model=model, ewc_lambda=100)
        
        # Backprop (for comparison)
        opt = EPOptimizer(model.parameters(), model=model, mode='backprop')
        
        # High-accuracy EP (more settling steps)
        opt = EPOptimizer(model.parameters(), model=model, settle_steps=30)
    """
    
    def __init__(
        self,
        params,
        model: Optional[nn.Module] = None,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        mode: str = 'ep',
        beta: float = 0.5,
        settle_steps: int = 10,
        settle_lr: float = 0.2,
        gradient_method: str = 'analytic',
        ewc_lambda: float = 0.0,
        ns_steps: int = 5,
        gamma: float = 0.95,
        loss_type: str = 'cross_entropy',
        softmax_temperature: float = 1.0,
    ):
        self.config = EPConfig(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            gradient_method='autograd',  # Must use autograd for settling (analytic is broken)
            ewc_lambda=ewc_lambda,
            ns_steps=ns_steps,
            gamma=gamma,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
        )
        
        self.model = model
        self.params = list(params)
        self.inspector = ModelInspector()
        
        # Structure only needed for EP mode
        if mode == 'ep':
            if model is None:
                raise ValueError("model is required for EP mode")
            self.structure = self.inspector.inspect(model)
            
            # Pre-compute structure info
            self._state_indices = []
            for i, item in enumerate(self.structure):
                if item['type'] in ('layer', 'attention'):
                    self._state_indices.append(i)
        else:
            self.structure = []
            self._state_indices = []
        
        # Momentum buffers
        self.buffers = [torch.zeros_like(p) for p in self.params]
        
        # EWC state (if enabled)
        self.ewc_state: Optional[EWCState] = None
        if ewc_lambda > 0:
            if model is None:
                raise ValueError("model is required for EWC")
            self.ewc_state = EWCState(model)
    
    def step(self, x: Optional[torch.Tensor] = None, target: Optional[torch.Tensor] = None, task_id: Optional[int] = None):
        """
        Perform optimization step.
        
        For EP mode:
            opt.step(x=x, target=y)
        
        For backprop mode (standard PyTorch pattern):
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            opt.step()  # x and target not needed
        
        Args:
            x: Input tensor (required for EP mode).
            target: Target tensor (required for EP mode).
            task_id: Task ID for EWC (optional).
        """
        if task_id is not None and self.ewc_state is not None:
            self.ewc_state._current_task = task_id
        
        if self.config.mode == 'backprop':
            self._backprop_step()
        else:
            if x is None or target is None:
                raise ValueError("EP mode requires x and target arguments")
            self._ep_step(x, target)
    
    def _backprop_step(self):
        """Backpropagation step - requires user to call model(x) and backward() first."""
        # For backprop, the standard pattern is:
        #   output = model(x)
        #   loss = criterion(output, y)
        #   loss.backward()
        #   optimizer.step()
        # So we just apply momentum + weight decay to existing gradients
        with torch.no_grad():
            for p, buf in zip(self.params, self.buffers):
                if p.grad is None:
                    continue
                buf.mul_(self.config.momentum).add_(p.grad)
                if self.config.weight_decay > 0:
                    buf.add_(p, alpha=self.config.weight_decay)
                p.sub_(buf, alpha=self.config.lr)
    
    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients (for backprop mode)."""
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()
    
    def _ep_step(self, x: torch.Tensor, target: Optional[torch.Tensor]):
        """EP step with configurable gradient method."""
        device = x.device
        batch_size = x.shape[0]
        
        # Prepare target
        target_vec = target
        if target is not None and self.config.loss_type == 'cross_entropy':
            if target.dim() > 1 and target.shape[1] > 1:
                target_vec = target.argmax(dim=1)
            else:
                target_vec = target.squeeze()
        
        # Free phase settling
        states_free = self._settle(x, None, target_vec)
        
        # Nudged phase settling
        beta = self.config.beta if target is not None else 0.0
        states_nudged = self._settle(x, target_vec, target_vec, beta=beta)
        
        # Compute EWC loss if enabled
        ewc_loss = torch.tensor(0.0, device=device)
        if self.ewc_state is not None:
            ewc_loss = self.ewc_state.compute_ewc_loss()
        
        # Contrast step - use_grad=True for parameter gradients
        E_free = self._energy_from_states(x, states_free, None, 0.0, use_grad=True)
        E_nudged = self._energy_from_states(x, states_nudged, target_vec, self.config.beta, use_grad=True)
        
        contrast_loss = (E_nudged - E_free) / self.config.beta
        total_loss = contrast_loss + ewc_loss
        
        grads = torch.autograd.grad(total_loss, self.params, retain_graph=False)
        
        with torch.no_grad():
            for p, g, buf in zip(self.params, grads, self.buffers):
                buf.mul_(self.config.momentum).add_(g)
                if self.config.weight_decay > 0:
                    buf.add_(p, alpha=self.config.weight_decay)
                p.sub_(buf, alpha=self.config.lr)
    
    def _settle(
        self,
        x: torch.Tensor,
        target_vec: Optional[torch.Tensor],
        original_target: Optional[torch.Tensor],
        beta: float = 0.0,
    ) -> List[torch.Tensor]:
        """Settling loop with configurable gradient method."""
        device = x.device
        
        # Capture initial states
        states = self._capture_states(x)
        momentum_buffers = [torch.zeros_like(s) for s in states]
        
        for _ in range(self.config.settle_steps):
            if self.config.gradient_method == 'analytic':
                grads = self._analytic_gradients(x, states, target_vec, beta)
            else:
                grads = self._autograd_gradients(x, states, target_vec, beta)
            
            with torch.no_grad():
                for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
                    buf.mul_(0.5).add_(g)
                    state.sub_(buf, alpha=self.config.settle_lr)
        
        return [s.detach() for s in states]
    
    def _capture_states(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Capture initial states."""
        states = []
        handles = []

        def hook(m, i, o):
            # Must set requires_grad=True for settling to work
            states.append(o.detach().float().clone().requires_grad_(True))

        for item in self.structure:
            if item['type'] in ('layer', 'attention'):
                handles.append(item['module'].register_forward_hook(hook))

        try:
            with torch.no_grad():
                model_output = self.model(x)
        finally:
            for h in handles:
                h.remove()

        return states
    
    def _analytic_gradients(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        target_vec: Optional[torch.Tensor],
        beta: float,
    ) -> List[torch.Tensor]:
        """Compute gradients analytically (fast)."""
        device = x.device
        batch_size = x.shape[0]
        grads = []
        prev = x
        state_idx = 0
        
        with torch.no_grad():
            for item in self.structure:
                if item['type'] == 'layer':
                    state = states[state_idx]
                    h = item['module'](prev)
                    
                    # dE/dstate = state - h
                    grad = (state - h) / batch_size
                    grads.append(grad)
                    
                    prev = state
                    state_idx += 1
                elif item['type'] == 'act':
                    prev = item['module'](prev)
            
            # Nudge gradient for last state
            if target_vec is not None and beta > 0 and grads:
                last_state = states[-1]
                
                if self.config.loss_type == 'cross_entropy':
                    state_sm = F.softmax(last_state / self.config.softmax_temperature, dim=1)
                    if target_vec.dim() == 1:
                        target_one_hot = F.one_hot(target_vec, num_classes=last_state.shape[1]).float()
                    else:
                        target_one_hot = target_vec.float()
                    nudge_grad = beta * (state_sm - target_one_hot) / batch_size
                else:
                    # MSE loss - target should match state shape
                    if target_vec.dim() == 1:
                        # For classification with MSE, convert to one-hot
                        target_one_hot = F.one_hot(target_vec, num_classes=last_state.shape[1]).float()
                    elif target_vec.shape != last_state.shape:
                        # Try to broadcast
                        target_one_hot = target_vec.expand_as(last_state)
                    else:
                        target_one_hot = target_vec
                    nudge_grad = beta * (last_state - target_one_hot) / batch_size
                
                grads[-1] = grads[-1] + nudge_grad
        
        return grads
    
    def _autograd_gradients(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        target_vec: Optional[torch.Tensor],
        beta: float,
    ) -> List[torch.Tensor]:
        """Compute gradients using autograd (exact)."""
        states_with_grad = [s.detach().clone().requires_grad_(True) for s in states]
        
        E = self._energy_from_states(x, states_with_grad, target_vec, beta, use_grad=True)
        
        grads = torch.autograd.grad(E, states_with_grad, retain_graph=False, allow_unused=True)
        return list(grads)
    
    def _energy_from_states(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        target_vec: Optional[torch.Tensor],
        beta: float,
        use_grad: bool = False,
    ) -> torch.Tensor:
        """Compute energy from settled states.
        
        Note: Internal energy is ALWAYS MSE (state consistency).
        The loss_type only affects the nudge term, not the internal energy.
        """
        device = x.device
        batch_size = x.shape[0]

        E = torch.tensor(0.0, device=device, dtype=torch.float32)
        prev = x
        state_idx = 0

        if use_grad:
            # Forward pass with gradients for parameter updates
            for item in self.structure:
                if item['type'] == 'layer':
                    if state_idx >= len(states):
                        break

                    state = states[state_idx]
                    h = item['module'](prev)

                    # Always use MSE for internal energy (state consistency)
                    E = E + 0.5 * F.mse_loss(h.float(), state.float(), reduction='sum') / batch_size

                    prev = state.to(x.dtype)
                    state_idx += 1
                elif item['type'] == 'act':
                    prev = item['module'](prev)

            # Nudge term - this is where loss_type matters
            if target_vec is not None and beta > 0:
                if self.config.loss_type == 'cross_entropy':
                    E = E + beta * F.cross_entropy(prev.float(), target_vec, reduction='sum', label_smoothing=0.1) / batch_size
                else:
                    # MSE loss - convert class indices to one-hot if needed
                    if target_vec.dim() == 1:
                        target_one_hot = F.one_hot(target_vec, num_classes=prev.shape[1]).float()
                    elif target_vec.shape != prev.shape:
                        target_one_hot = target_vec.expand_as(prev)
                    else:
                        target_one_hot = target_vec
                    E = E + beta * F.mse_loss(prev.float(), target_one_hot, reduction='sum') / batch_size
        else:
            # No grad - for settling iterations
            with torch.no_grad():
                for item in self.structure:
                    if item['type'] == 'layer':
                        if state_idx >= len(states):
                            break

                        state = states[state_idx]
                        h = item['module'](prev)

                        # Always use MSE for internal energy (state consistency)
                        E = E + 0.5 * F.mse_loss(h.float(), state.float(), reduction='sum') / batch_size

                        prev = state.to(x.dtype)
                        state_idx += 1
                    elif item['type'] == 'act':
                        prev = item['module'](prev)

                # Nudge term - this is where loss_type matters
                if target_vec is not None and beta > 0:
                    if self.config.loss_type == 'cross_entropy':
                        E = E + beta * F.cross_entropy(prev.float(), target_vec, reduction='sum', label_smoothing=0.1) / batch_size
                    else:
                        # MSE loss - convert class indices to one-hot if needed
                        if target_vec.dim() == 1:
                            target_one_hot = F.one_hot(target_vec, num_classes=prev.shape[1]).float()
                        elif target_vec.shape != prev.shape:
                            target_one_hot = target_vec.expand_as(prev)
                        else:
                            target_one_hot = target_vec
                        E = E + beta * F.mse_loss(prev.float(), target_one_hot, reduction='sum') / batch_size

        return E
    
    def consolidate_task(self, data_loader, task_id: int, device: str = 'cuda'):
        """
        Consolidate a completed task (for EWC).
        
        Call this after training on each task in continual learning.
        """
        if self.ewc_state is None:
            raise ValueError("EWC not enabled (ewc_lambda=0)")
        
        self.ewc_state.update_fisher(data_loader, task_id, device, self.config.loss_type)
    
    def get_forgetting(self, task_id: int) -> Dict[str, float]:
        """Get forgetting measure for a task."""
        if self.ewc_state is None or task_id not in self.ewc_state.task_memories:
            return {'error': 'Task not found or EWC not enabled'}
        
        memory = self.ewc_state.task_memories[task_id]
        total_drift = 0.0
        weighted_drift = 0.0
        param_count = 0
        
        for n, p in self.model.named_parameters():
            if n in memory['optimal_params']:
                optimal = memory['optimal_params'][n]
                drift = (p - optimal).abs().mean().item()
                total_drift += drift
                param_count += 1
                
                if n in memory['fisher']:
                    fisher = memory['fisher'][n]
                    weighted_drift += (fisher * (p - optimal) ** 2).sum().item()
        
        return {
            'task_id': task_id,
            'avg_param_drift': total_drift / max(1, param_count),
            'weighted_drift': weighted_drift,
            'ewc_penalty': weighted_drift * self.config.ewc_lambda * 0.5,
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state."""
        return {
            'config': self.config,
            'buffers': self.buffers,
            'ewc_state': self.ewc_state.state_dict() if self.ewc_state else None,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load optimizer state."""
        self.config = state['config']
        self.buffers = state['buffers']
        if state['ewc_state'] is not None and self.ewc_state is not None:
            self.ewc_state.load_state_dict(state['ewc_state'])


# Backward compatibility aliases
def smep(params, model, **kwargs):
    """Backward compatibility: smep preset."""
    return EPOptimizer(
        params, model=model,
        mode='ep',
        settle_steps=kwargs.get('settle_steps', 30),
        settle_lr=kwargs.get('settle_lr', 0.15),
        beta=kwargs.get('beta', 0.5),
        loss_type=kwargs.get('loss_type', 'mse'),
        lr=kwargs.get('lr', 0.01),
    )


def smep_fast(params, model, **kwargs):
    """Backward compatibility: smep_fast preset."""
    return EPOptimizer(
        params, model=model,
        mode='ep',
        settle_steps=kwargs.get('settle_steps', 10),
        settle_lr=kwargs.get('settle_lr', 0.2),
        beta=kwargs.get('beta', 0.5),
        loss_type=kwargs.get('loss_type', 'mse'),
        lr=kwargs.get('lr', 0.01),
    )


def sdmep(params, model, **kwargs):
    """Backward compatibility: sdmep preset (Dion low-rank)."""
    # Note: Dion update not yet integrated into unified optimizer
    # Falls back to standard Muon update
    return EPOptimizer(
        params, model=model,
        mode='ep',
        settle_steps=kwargs.get('settle_steps', 15),
        settle_lr=kwargs.get('settle_lr', 0.1),
        beta=kwargs.get('beta', 0.3),
        loss_type=kwargs.get('loss_type', 'cross_entropy'),
        lr=kwargs.get('lr', 0.01),
    )


def local_ep(params, model, **kwargs):
    """Backward compatibility: local_ep preset."""
    return EPOptimizer(
        params, model=model,
        mode='ep',
        settle_steps=kwargs.get('settle_steps', 20),
        settle_lr=kwargs.get('settle_lr', 0.05),
        beta=kwargs.get('beta', 0.1),
        loss_type=kwargs.get('loss_type', 'mse'),
        lr=kwargs.get('lr', 0.02),
    )


def natural_ep(params, model, **kwargs):
    """Backward compatibility: natural_ep preset."""
    # Note: Natural gradient (Fisher) not yet integrated
    # Falls back to standard EP
    return EPOptimizer(
        params, model=model,
        mode='ep',
        settle_steps=kwargs.get('settle_steps', 20),
        settle_lr=kwargs.get('settle_lr', 0.05),
        beta=kwargs.get('beta', 0.5),
        loss_type=kwargs.get('loss_type', 'mse'),
        lr=kwargs.get('lr', 0.02),
    )


def muon_backprop(params, model=None, **kwargs):
    """Backward compatibility: muon backprop preset."""
    return EPOptimizer(
        params,
        model=model,
        mode='backprop',
        lr=kwargs.get('lr', 0.02),
        momentum=kwargs.get('momentum', 0.9),
        weight_decay=kwargs.get('weight_decay', 0.0005),
        ns_steps=kwargs.get('ns_steps', 5),
        gamma=kwargs.get('gamma', 0.95),
    )
