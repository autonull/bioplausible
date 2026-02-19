"""
Settling dynamics for Equilibrium Propagation.

This module handles the iterative settling of network activations
to minimize the energy function during free and nudged phases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Callable, Tuple


def _settle_step_compilable(
    states: List[torch.Tensor],
    momentum_buffers: List[torch.Tensor],
    energy: torch.Tensor,
    current_lr: float,
    momentum: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Perform a single settling step - compiled-friendly.
    
    This function is designed to be torch.compile-compatible by avoiding
    Python-side operations and using tensor operations only.
    """
    new_states = []
    new_buffers = []
    
    for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, 
                                             torch.autograd.grad(energy, states, 
                                                                retain_graph=False, 
                                                                allow_unused=True))):
        if g is None:
            new_states.append(state)
            new_buffers.append(buf)
        else:
            new_buf = buf * momentum + g
            new_state = state - new_buf * current_lr
            new_states.append(new_state)
            new_buffers.append(new_buf)
    
    return new_states, new_buffers


class Settler:
    """
    Settles network activations to minimize energy.

    Uses gradient-based optimization to find fixed points of the
    energy function during EP free and nudged phases.

    Key insight: Settling convergence is critical for EP performance.
    - Higher settle_lr (0.1-0.2) enables faster convergence
    - More settle_steps (30-50) ensures proper settling
    - Momentum (0.5) helps escape local minima
    """

    MOMENTUM = 0.5

    def __init__(
        self,
        steps: int = 30,  # Increased default from 20 to 30
        lr: float = 0.15,  # Increased default from 0.05 to 0.15
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
        tol: float = 1e-4,
        patience: int = 5,
        adaptive: bool = False,
    ):
        if steps <= 0:
            raise ValueError(f"Steps must be positive, got {steps}")
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if tol < 0:
            raise ValueError(f"Tolerance must be non-negative, got {tol}")
        if patience < 0:
            raise ValueError(f"Patience must be non-negative, got {patience}")

        self.steps = steps
        self.lr = lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
        self.tol = tol
        self.patience = patience
        self.adaptive = adaptive
        self.step_size_growth = 1.1
        self.step_size_decay = 0.5
    
    def settle(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Settle network activations to energy minimum.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor (None for free phase).
            beta: Nudging strength.
            energy_fn: Function to compute energy.
            structure: Model structure from inspector.
        
        Returns:
            List of settled state tensors for each layer.
        
        Raises:
            ValueError: If input is invalid.
            RuntimeError: If settling diverges.
        """
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        
        # Capture initial states
        states = self._capture_states(model, x, structure)
        
        if not states:
            layer_count = sum(1 for item in structure if item["type"] in ("layer", "attention"))
            if layer_count > 0:
                 raise RuntimeError(
                    f"No activations captured. Expected {layer_count} layer(s).\n"
                    f"Model: {type(model).__name__}, Structure: {len(structure)} items"
                )
            else:
                return [] # No states to settle
        
        # Prepare target
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, states[-1].shape[-1], states[-1].dtype)
        
        # Momentum buffers
        momentum_buffers = [torch.zeros_like(s) for s in states]
        
        # Settling loop
        prev_energy: Optional[float] = None
        patience_counter = 0
        current_lr = self.lr
        just_restored = False

        # Backup for adaptive steps
        states_backup = [s.clone() for s in states] if self.adaptive else None
        
        for step in range(self.steps):
            with torch.enable_grad():
                E = energy_fn(model, x, states, structure, target_vec, beta)
                
                # Check for divergence
                if torch.isnan(E) or torch.isinf(E):
                    raise RuntimeError(
                        f"Energy diverged at step {step}: E={E.item()}. "
                        f"Try reducing settle_lr, beta, or learning rate."
                    )
                
                current_energy = float(E.item())

                # Adaptive step size logic
                if self.adaptive and states_backup is not None:
                    if prev_energy is not None:
                        if current_energy > prev_energy:
                            # Energy increased: reject step
                            # Restore states from backup
                            with torch.no_grad():
                                for s, b in zip(states, states_backup):
                                    s.copy_(b)

                            # Decay LR
                            current_lr *= self.step_size_decay

                            # We must continue to re-evaluate at restored state
                            just_restored = True
                            continue
                        else:
                            # Energy decreased: accept step

                            # Grow LR slightly (with cap?) only if we didn't just restore
                            if not just_restored:
                                current_lr = min(current_lr * self.step_size_growth, self.lr * 10)

                            # Update backup
                            with torch.no_grad():
                                for s, b in zip(states, states_backup):
                                    b.copy_(s)
                    else:
                        # First step
                        with torch.no_grad():
                            for s, b in zip(states, states_backup):
                                b.copy_(s)

                # Early stopping
                # Skip check if we just restored (delta would be 0)
                if prev_energy is not None and not just_restored:
                    delta = abs(current_energy - prev_energy)
                    # Use both absolute and relative tolerance for robust convergence detection
                    rel_tol = self.tol * max(1.0, abs(prev_energy))
                    if delta < self.tol or delta < rel_tol:
                        patience_counter += 1
                    else:
                        patience_counter = 0

                    if patience_counter >= self.patience:
                        # Converged - energy stable for patience steps
                        break

                just_restored = False
                prev_energy = current_energy

                grads = torch.autograd.grad(E, states, retain_graph=False, allow_unused=True)

            # SGD step with momentum - use fused kernel if available
            with torch.no_grad():
                # Try to use fused CUDA kernel for efficiency
                try:
                    from ..cuda.kernels import fused_settle_step_inplace
                    if states[0].is_cuda:
                        fused_settle_step_inplace(
                            states, momentum_buffers, grads,
                            momentum=self.MOMENTUM, lr=current_lr
                        )
                    else:
                        # CPU fallback
                        for i, (state, g) in enumerate(zip(states, grads)):
                            if g is None:
                                continue
                            buf = momentum_buffers[i]
                            buf.mul_(self.MOMENTUM).add_(g)
                            state.sub_(buf, alpha=current_lr)
                except ImportError:
                    # Fallback if cuda module not available
                    for i, (state, g) in enumerate(zip(states, grads)):
                        if g is None:
                            continue
                        buf = momentum_buffers[i]
                        buf.mul_(self.MOMENTUM).add_(g)
                        state.sub_(buf, alpha=current_lr)

        return [s.detach() for s in states]
    
    def settle_with_graph(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Settle network keeping computation graph intact for gradient flow.
        """
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")
        
        # Capture initial states
        states = self._capture_states_fresh(model, x, structure)
        
        if not states:
            layer_count = sum(1 for item in structure if item["type"] in ("layer", "attention"))
            if layer_count > 0:
                raise RuntimeError(
                    f"No activations captured. Expected {layer_count} layer(s)."
                )
            else:
                return []
        
        # Prepare target
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, states[-1].shape[-1], states[-1].dtype)
        
        momentum_buffers = [torch.zeros_like(s) for s in states]
        
        prev_energy: Optional[float] = None
        patience_counter = 0
        current_lr = self.lr

        # Backup not easily supported for graph mode due to graph connections
        # For now, disable adaptive step size in graph mode or implement complex rollback
        if self.adaptive:
            import warnings
            warnings.warn("Adaptive settling is not supported in 'settle_with_graph'. Ignoring adaptive flag.")

        for step in range(self.steps):
            working_states = [s.detach().requires_grad_(True) for s in states]
            
            E = energy_fn(model, x, working_states, structure, target_vec, beta)
            
            if torch.isnan(E) or torch.isinf(E):
                raise RuntimeError(f"Energy diverged at step {step}: E={E.item()}")
            
            current_energy = float(E.item())

            if prev_energy is not None:
                delta = abs(current_energy - prev_energy)
                if delta < self.tol:
                    patience_counter += 1
                else:
                    patience_counter = 0

                if patience_counter >= self.patience:
                    break

            prev_energy = current_energy

            grads = torch.autograd.grad(E, working_states, retain_graph=False, allow_unused=True)
            
            # Update working states
            for i, (state, g) in enumerate(zip(working_states, grads)):
                if g is None:
                    continue
                buf = momentum_buffers[i]
                buf.mul_(self.MOMENTUM).add_(g)
                state = state - buf * current_lr
                working_states[i] = state
            
            # Copy back to states
            with torch.no_grad():
                for i, s in enumerate(working_states):
                    states[i] = s.detach().requires_grad_(False)
        
        return [s.detach() for s in states]
    
    def _capture_states_fresh(
        self,
        model: nn.Module,
        x: torch.Tensor,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Capture states as fresh tensors."""
        states: List[torch.Tensor] = []
        handles: List[Any] = []
        
        def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
            # Capture state in float32 for stability
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

    def _capture_states(
        self,
        model: nn.Module,
        x: torch.Tensor,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Capture initial layer states."""
        states: List[torch.Tensor] = []
        handles: List[Any] = []
        
        def capture_hook(module: nn.Module, inp: Any, output: Any) -> None:
            # Capture state in float32 for stability during settling updates
            if isinstance(output, tuple):
                s = output[0].detach().float().clone().requires_grad_(True)
            else:
                s = output.detach().float().clone().requires_grad_(True)
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
    
    def _prepare_target(
        self,
        target: torch.Tensor,
        num_classes: int,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert target to appropriate format."""
        if self.loss_type == "cross_entropy":
            if target.dim() > 1 and target.shape[1] > 1:
                return target.argmax(dim=1).long()
            return target.squeeze().long()
        else:
            if target.dim() == 1:
                return F.one_hot(target, num_classes=num_classes).to(dtype=dtype)
            return target.to(dtype=dtype)

    def settle_compiled(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
    ) -> List[torch.Tensor]:
        """
        Settle network activations using torch.compile for acceleration.

        This method uses a fixed number of settling steps (no early stopping)
        to enable torch.compile optimization. Best for repeated calls with
        similar models and inputs.

        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor (None for free phase).
            beta: Nudging strength.
            energy_fn: Function to compute energy.
            structure: Model structure from inspector.

        Returns:
            List of settled state tensors for each layer.

        Note:
            - Adaptive stepping and early stopping are disabled for compilation.
            - First call includes compilation overhead; subsequent calls are faster.
            - Use torch.compile on the energy_fn for maximum benefit.
        """
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")

        # Capture initial states
        states = self._capture_states(model, x, structure)

        if not states:
            layer_count = sum(1 for item in structure if item["type"] in ("layer", "attention"))
            if layer_count > 0:
                raise RuntimeError(
                    f"No activations captured. Expected {layer_count} layer(s)."
                )
            else:
                return []

        # Prepare target
        target_vec = None
        if target is not None:
            target_vec = self._prepare_target(target, states[-1].shape[-1], states[-1].dtype)

        # Momentum buffers
        momentum_buffers = [torch.zeros_like(s) for s in states]

        # Fixed settling loop - compiled
        states = self._settle_loop_fixed(
            model, x, states, momentum_buffers, target_vec, beta,
            energy_fn, structure, self.steps, self.lr
        )

        return [s.detach() for s in states]

    def _settle_loop_fixed(
        self,
        model: nn.Module,
        x: torch.Tensor,
        states: List[torch.Tensor],
        momentum_buffers: List[torch.Tensor],
        target_vec: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]],
        steps: int,
        lr: float,
    ) -> List[torch.Tensor]:
        """
        Fixed-step settling loop - designed for torch.compile.

        This method performs a fixed number of settling steps without
        Python-side control flow, enabling better compilation.
        """
        for _ in range(steps):
            with torch.enable_grad():
                E = energy_fn(model, x, states, structure, target_vec, beta)

            # SGD step with momentum
            with torch.no_grad():
                grads = torch.autograd.grad(E, states, retain_graph=False, allow_unused=True)
                for i, (state, g) in enumerate(zip(states, grads)):
                    if g is None:
                        continue
                    buf = momentum_buffers[i]
                    buf.mul_(self.MOMENTUM).add_(g)
                    state.sub_(buf, alpha=lr)

        return states


# Compiled helper function (can be used standalone)
@torch.compile(mode="reduce-overhead")
def _compiled_settle_step(
    states: List[torch.Tensor],
    momentum_buffers: List[torch.Tensor],
    model: nn.Module,
    x: torch.Tensor,
    target_vec: Optional[torch.Tensor],
    beta: float,
    energy_fn: Callable,
    structure: List[Dict[str, Any]],
    lr: float,
    momentum: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Compiled settling step - standalone function for torch.compile.

    This function wraps a single settling step and can be used
    with torch.compile for acceleration.
    """
    with torch.enable_grad():
        E = energy_fn(model, x, states, structure, target_vec, beta)

    new_states = []
    new_buffers = []

    with torch.no_grad():
        grads = torch.autograd.grad(E, states, retain_graph=False, allow_unused=True)
        for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
            if g is None:
                new_states.append(state)
                new_buffers.append(buf)
            else:
                new_buf = buf * momentum + g
                new_state = state - new_buf * lr
                new_states.append(new_state)
                new_buffers.append(new_buf)

    return new_states, new_buffers
