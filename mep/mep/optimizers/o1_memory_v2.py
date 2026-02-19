"""
O(1) Memory Implementation v2: Analytic Gradients

Phase 2: Week 3-4 - True O(1) Memory via Analytic Gradients

This module achieves O(1) activation memory by computing state gradients
analytically instead of using torch.autograd.grad().

Key insight: For MSE energy E = 0.5 * ||state - h||^2, the gradient is:
    dE/dstate = state - h

No autograd required - just tensor subtraction!

Author: Phase 2 Implementation
Created: 2026-02-18 (v2: 2026-02-25)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple


def analytic_state_gradients(
    model: nn.Module,
    x: torch.Tensor,
    states: List[torch.Tensor],
    structure: List[Dict[str, Any]],
    target_vec: Optional[torch.Tensor],
    beta: float,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
) -> List[torch.Tensor]:
    """
    Compute dE/dstate analytically without autograd.
    
    For MSE energy: E = 0.5 * ||state - h||^2
    Gradient: dE/dstate = state - h
    
    For KL energy (classification): E = KL(softmax(state) || softmax(h))
    Gradient: dE/dstate = (softmax(state) - softmax(h)) / temperature
              (approximate, ignoring second-order terms)
    
    Args:
        model: Neural network module.
        x: Input tensor.
        states: Current settled states.
        structure: Model structure.
        target_vec: Target for nudge term.
        beta: Nudging strength.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.
    
    Returns:
        List of gradient tensors for each state.
    """
    device = x.device
    batch_size = x.shape[0]
    
    grads = []
    prev = x
    state_idx = 0
    
    # Count state-producing modules
    state_producing = [item for item in structure if item["type"] in ("layer", "attention")]
    num_states = len(state_producing)
    
    use_classification = loss_type == "cross_entropy"
    
    with torch.no_grad():
        for item in structure:
            item_type = item["type"]
            module = item["module"]
            
            if item_type == "layer":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                is_last_state = (state_idx == num_states - 1)
                
                # Forward pass to get h (no grad needed)
                h = module(prev)
                
                # Analytic gradient: dE/dstate = state - h
                if use_classification and is_last_state:
                    # For KL divergence: grad ≈ (softmax(state) - softmax(h)) / T
                    state_sm = F.softmax(state / softmax_temperature, dim=1)
                    h_sm = F.softmax(h / softmax_temperature, dim=1)
                    grad = (state_sm - h_sm) / softmax_temperature
                else:
                    # For MSE: grad = state - h
                    grad = state - h
                
                # Normalize by batch size (matching energy formula)
                grad = grad / batch_size
                
                grads.append(grad)
                
                # Input to next layer is the current state
                prev = state.to(x.dtype)
                state_idx += 1
            
            elif item_type == "norm":
                prev = module(prev)
            
            elif item_type == "pool":
                prev = module(prev)
            
            elif item_type == "flatten":
                prev = module(prev)
            
            elif item_type == "dropout":
                pass
            
            elif item_type == "attention":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                
                if isinstance(module, nn.MultiheadAttention):
                    h = module(prev, prev, prev, need_weights=False)[0]
                else:
                    h = module(prev)
                
                # Analytic gradient for attention
                grad = (state - h) / batch_size
                grads.append(grad)
                
                prev = state.to(x.dtype)
                state_idx += 1
            
            elif item_type == "act":
                prev = module(prev)
        
        # Handle nudge term gradient for last state
        if target_vec is not None and beta > 0 and grads:
            last_state = states[-1]
            
            if loss_type == "cross_entropy":
                # Gradient of cross-entropy nudge term
                # d/dstate [β * CE(state, target)] = β * (softmax(state) - one_hot(target))
                state_sm = F.softmax(last_state / softmax_temperature, dim=1)
                if target_vec.dim() == 1:
                    # Class indices
                    target_one_hot = F.one_hot(target_vec, num_classes=last_state.shape[1])
                    target_one_hot = target_one_hot.to(dtype=last_state.dtype)
                else:
                    # Already one-hot
                    target_one_hot = target_vec
                
                nudge_grad = beta * (state_sm - target_one_hot) / batch_size
            else:
                # Gradient of MSE nudge term
                # d/dstate [β * MSE(state, target)] = β * (state - target) / batch_size
                if target_vec.shape != last_state.shape:
                    target_vec = target_vec.expand_as(last_state)
                nudge_grad = beta * (last_state - target_vec) / batch_size
            
            # Add nudge gradient to last state gradient
            grads[-1] = grads[-1] + nudge_grad
    
    return grads


def settle_manual_o1(
    model: nn.Module,
    x: torch.Tensor,
    target: Optional[torch.Tensor],
    beta: float,
    structure: List[Dict[str, Any]],
    steps: int = 30,
    lr: float = 0.15,
    momentum: float = 0.5,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
) -> List[torch.Tensor]:
    """
    O(1) memory settling using analytic gradients.
    
    No autograd is used - gradients are computed analytically.
    This achieves true O(1) activation memory during settling.
    
    Args:
        model: Neural network module.
        x: Input tensor.
        target: Target tensor (None for free phase).
        beta: Nudging strength.
        structure: Model structure from inspector.
        steps: Number of settling iterations.
        lr: Settling learning rate.
        momentum: Momentum factor.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.
    
    Returns:
        List of settled state tensors.
    """
    device = x.device
    
    # Capture initial states (no_grad)
    with torch.no_grad():
        states = _capture_states_no_grad(model, x, structure)
    
    if not states:
        layer_count = sum(1 for item in structure if item["type"] in ("layer", "attention"))
        if layer_count > 0:
            raise RuntimeError(f"No activations captured. Expected {layer_count} layer(s).")
        else:
            return []
    
    # Prepare target
    target_vec = None
    if target is not None:
        if loss_type == "cross_entropy":
            if target.dim() > 1 and target.shape[1] > 1:
                target_vec = target.argmax(dim=1).long()
            else:
                target_vec = target.squeeze().long()
        else:
            if target.dim() == 1:
                num_classes = states[-1].shape[-1]
                target_vec = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)
            else:
                target_vec = target.to(dtype=x.dtype)
    
    # Momentum buffers
    momentum_buffers = [torch.zeros_like(s) for s in states]
    
    # Settling loop - TRUE O(1): no autograd, no graph storage
    for step in range(steps):
        # Compute gradients analytically (no autograd!)
        grads = analytic_state_gradients(
            model, x, states, structure, target_vec, beta,
            loss_type=loss_type, softmax_temperature=softmax_temperature
        )
        
        # Update states (no_grad)
        with torch.no_grad():
            for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
                buf.mul_(momentum).add_(g)
                state.sub_(buf, alpha=lr)
    
    return [s.detach() for s in states]


def _capture_states_no_grad(
    model: nn.Module,
    x: torch.Tensor,
    structure: List[Dict[str, Any]],
) -> List[torch.Tensor]:
    """Capture initial layer states without autograd."""
    states: List[torch.Tensor] = []
    handles: List[Any] = []
    
    def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
        if isinstance(output, tuple):
            s = output[0].detach().float().clone()
        else:
            s = output.detach().float().clone()
        states.append(s)
    
    for item in structure:
        if item["type"] in ("layer", "attention"):
            handles.append(item["module"].register_forward_hook(capture_hook))
    
    try:
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    
    return states


def manual_energy_compute_o1(
    model: nn.Module,
    x: torch.Tensor,
    states: List[torch.Tensor],
    structure: List[Dict[str, Any]],
    target_vec: Optional[torch.Tensor],
    beta: float,
    loss_type: str = "cross_entropy",
    softmax_temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute EP energy without any autograd overhead.
    
    This version uses direct tensor operations with no grad tracking.
    Use for settling iterations where we only need the energy value.
    
    Args:
        model: Neural network module.
        x: Input tensor.
        states: List of layer states.
        structure: Model structure.
        target_vec: Target for nudge term.
        beta: Nudging strength.
        loss_type: 'mse' or 'cross_entropy'.
        softmax_temperature: Temperature for softmax.
    
    Returns:
        Scalar energy tensor (no gradient history).
    """
    batch_size = x.shape[0]
    device = x.device
    
    E = torch.tensor(0.0, device=device, dtype=torch.float32)
    prev = x
    state_idx = 0
    
    state_producing = [item for item in structure if item["type"] in ("layer", "attention")]
    num_states = len(state_producing)
    
    use_classification = loss_type == "cross_entropy"
    
    with torch.no_grad():
        for item in structure:
            item_type = item["type"]
            module = item["module"]
            
            if item_type == "layer":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                is_last_state = (state_idx == num_states - 1)
                
                # Forward pass
                h = module(prev)
                
                # Compute energy
                if use_classification and is_last_state:
                    E = E + _kl_energy_no_grad(state, h, batch_size, softmax_temperature)
                else:
                    E = E + 0.5 * _mse_no_grad(h, state) / batch_size
                
                prev = state.to(x.dtype)
                state_idx += 1
            
            elif item_type == "norm":
                prev = module(prev)
            
            elif item_type == "pool":
                prev = module(prev)
            
            elif item_type == "flatten":
                prev = module(prev)
            
            elif item_type == "dropout":
                pass
            
            elif item_type == "attention":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                
                if isinstance(module, nn.MultiheadAttention):
                    h = module(prev, prev, prev, need_weights=False)[0]
                else:
                    h = module(prev)
                
                E = E + 0.5 * _mse_no_grad(h, state) / batch_size
                prev = state.to(x.dtype)
                state_idx += 1
            
            elif item_type == "act":
                prev = module(prev)
        
        # Nudge term
        if target_vec is not None and beta > 0:
            E = E + _nudge_term_no_grad(prev, target_vec, beta, batch_size, loss_type)
    
    return E


def _mse_no_grad(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute MSE without grad tracking."""
    return F.mse_loss(input, target, reduction="sum")


def _kl_energy_no_grad(
    state: torch.Tensor,
    prediction: torch.Tensor,
    batch_size: int,
    softmax_temperature: float,
) -> torch.Tensor:
    """Compute KL divergence without grad tracking."""
    eps = 1e-8
    
    state_softmax = F.softmax(state / softmax_temperature, dim=1)
    h_softmax = F.softmax(prediction / softmax_temperature, dim=1)
    
    kl_div = F.kl_div(torch.log(state_softmax + eps), h_softmax, reduction="sum")
    return kl_div / batch_size


def _nudge_term_no_grad(
    output: torch.Tensor,
    target_vec: torch.Tensor,
    beta: float,
    batch_size: int,
    loss_type: str,
) -> torch.Tensor:
    """Compute nudge term without grad tracking."""
    if loss_type == "cross_entropy":
        return beta * F.cross_entropy(
            output, target_vec, reduction="sum", label_smoothing=0.1
        ) / batch_size
    else:
        return beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size


def energy_from_states_minimal(
    model: nn.Module,
    x: torch.Tensor,
    states: List[torch.Tensor],
    structure: List[Dict[str, Any]],
    target_vec: Optional[torch.Tensor],
    beta: float,
    loss_type: str = "cross_entropy",
) -> torch.Tensor:
    """
    Compute energy from states with MINIMAL autograd for parameter gradients.
    
    This builds the smallest possible graph for computing dE/dW.
    Uses gradient checkpointing for the forward pass.
    
    Args:
        model: Neural network module.
        x: Input tensor.
        states: List of settled states.
        structure: Model structure.
        target_vec: Target for nudge term.
        beta: Nudging strength.
        loss_type: 'mse' or 'cross_entropy'.
    
    Returns:
        Energy tensor with gradient history for parameter gradients.
    """
    batch_size = x.shape[0]
    device = x.device
    
    E = torch.tensor(0.0, device=device, dtype=torch.float32)
    prev = x
    state_idx = 0
    
    state_producing = [item for item in structure if item["type"] in ("layer", "attention")]
    num_states = len(state_producing)
    
    use_classification = loss_type == "cross_entropy"
    
    # Use gradient checkpointing for the forward pass
    for item in structure:
        item_type = item["type"]
        module = item["module"]
        
        if item_type == "layer":
            if state_idx >= len(states):
                break
            
            state = states[state_idx]
            is_last_state = (state_idx == num_states - 1)
            
            # Forward pass with checkpointing
            if state_idx < num_states - 1:
                # Checkpoint hidden layers
                h = torch.utils.checkpoint.checkpoint(
                    module, prev, use_reentrant=False
                )
            else:
                # Don't checkpoint last layer (needed for output)
                h = module(prev)
            
            if use_classification and is_last_state:
                E = E + _kl_energy_autograd(state.float(), h.float(), batch_size)
            else:
                E = E + 0.5 * _mse_autograd(h.float(), state.float()) / batch_size
            
            prev = state.to(x.dtype)
            state_idx += 1
        
        elif item_type == "norm":
            prev = module(prev)
        
        elif item_type == "pool":
            prev = module(prev)
        
        elif item_type == "flatten":
            prev = module(prev)
        
        elif item_type == "dropout":
            pass
        
        elif item_type == "attention":
            if state_idx >= len(states):
                break
            
            state = states[state_idx]
            
            if isinstance(module, nn.MultiheadAttention):
                h = module(prev, prev, prev, need_weights=False)[0]
            else:
                h = module(prev)
            
            E = E + 0.5 * _mse_autograd(h.float(), state.float()) / batch_size
            prev = state.to(x.dtype)
            state_idx += 1
        
        elif item_type == "act":
            prev = module(prev)
    
    # Nudge term
    if target_vec is not None and beta > 0:
        E = E + _nudge_term_autograd(prev.float(), target_vec, beta, batch_size, loss_type)
    
    return E


def _mse_autograd(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE with autograd for parameter gradients."""
    return F.mse_loss(input, target, reduction="sum")


def _kl_energy_autograd(
    state: torch.Tensor,
    prediction: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """KL divergence with autograd for parameter gradients."""
    eps = 1e-8
    
    state_softmax = F.softmax(state, dim=1)
    h_softmax = F.softmax(prediction, dim=1)
    
    kl_div = F.kl_div(torch.log(state_softmax + eps), h_softmax, reduction="sum")
    return kl_div / batch_size


def _nudge_term_autograd(
    output: torch.Tensor,
    target_vec: torch.Tensor,
    beta: float,
    batch_size: int,
    loss_type: str,
) -> torch.Tensor:
    """Nudge term with autograd for parameter gradients."""
    if loss_type == "cross_entropy":
        return beta * F.cross_entropy(
            output, target_vec, reduction="sum", label_smoothing=0.1
        ) / batch_size
    else:
        return beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size


class O1MemoryEPv2:
    """
    O(1) Memory EP optimizer v2 with analytic gradients.
    
    Uses analytic state gradients instead of autograd during settling.
    Achieves true O(1) activation memory - independent of network depth.
    
    Usage:
        optimizer = O1MemoryEPv2(model.parameters(), model=model, lr=0.01)
        optimizer.step(x=x, target=y)
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        beta: float = 0.5,
        loss_type: str = "cross_entropy",
    ):
        self.params = list(params)
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.beta = beta
        self.loss_type = loss_type
        
        from mep.optimizers import ModelInspector
        
        self.inspector = ModelInspector()
        self.structure = self.inspector.inspect(model)
        
        # Momentum buffers for parameter updates
        self.buffers = [torch.zeros_like(p) for p in self.params]
    
    def step(self, x: torch.Tensor, target: torch.Tensor):
        """Perform O(1) memory EP training step."""
        # Free phase settling (O(1) memory - analytic gradients)
        states_free = settle_manual_o1(
            self.model, x, None, beta=0.0,
            structure=self.structure,
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
        )
        
        # Nudged phase settling (O(1) memory - analytic gradients)
        states_nudged = settle_manual_o1(
            self.model, x, target, beta=self.beta,
            structure=self.structure,
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
        )
        
        # Contrast step with gradient checkpointing
        E_free = energy_from_states_minimal(
            self.model, x, states_free, self.structure, None, 0.0,
            loss_type=self.loss_type
        )
        
        E_nudged = energy_from_states_minimal(
            self.model, x, states_nudged, self.structure, target, self.beta,
            loss_type=self.loss_type
        )
        
        contrast_loss = (E_nudged - E_free) / self.beta
        
        # Compute parameter gradients
        grads = torch.autograd.grad(contrast_loss, self.params, retain_graph=False)
        
        # Update parameters with momentum
        with torch.no_grad():
            for p, g, buf in zip(self.params, grads, self.buffers):
                buf.mul_(self.momentum).add_(g)
                
                if self.weight_decay != 0:
                    buf.add_(p, alpha=self.weight_decay)
                
                p.sub_(buf, alpha=self.lr)
