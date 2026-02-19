"""
Gradient computation strategies.

Implements various methods for computing gradients:
- Standard backpropagation
- Equilibrium Propagation (free/nudged contrast)
- Layer-local EP (biologically plausible)
- Natural gradient with Fisher whitening
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, Any, List, Dict
from .base import GradientStrategy


class BackpropGradient:
    """
    Standard backpropagation via .backward().
    
    This is the default gradient computation for conventional deep learning.
    """
    
    def __init__(self, loss_fn: Optional[nn.Module] = None):
        self.loss_fn = loss_fn
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        loss_fn: Optional[nn.Module] = None,
        **kwargs: Any
    ) -> None:
        """
        Compute gradients via standard backpropagation.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            loss_fn: Loss function (override instance default).
        """
        loss_fn = loss_fn or self.loss_fn
        if loss_fn is None:
            raise ValueError("loss_fn must be provided to BackpropGradient")
        
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()


class EPGradient:
    """
    Equilibrium Propagation via free/nudged phase contrast.

    Computes gradients as (E_nudged - E_free) / beta, where:
    - Free phase: network settles with beta=0
    - Nudged phase: network settles with target perturbation
    
    Default settings use adaptive settling with early stopping for efficiency.
    """

    def __init__(
        self,
        beta: float = 0.3,
        settle_steps: int = 15,
        settle_lr: float = 0.1,
        loss_type: str = "cross_entropy",
        softmax_temperature: float = 1.0,
        tol: float = 1e-3,
        patience: int = 3,
        adaptive: bool = True,
    ):
        if not (0 < beta <= 1):
            raise ValueError(f"Beta must be in (0, 1], got {beta}")
        if settle_steps <= 0:
            raise ValueError(f"Settle steps must be positive, got {settle_steps}")
        if settle_lr <= 0:
            raise ValueError(f"Settle learning rate must be positive, got {settle_lr}")

        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
        self.tol = tol
        self.patience = patience
        self.adaptive = adaptive
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable] = None,
        structure_fn: Optional[Callable] = None,
        **kwargs: Any
    ) -> None:
        """
        Compute EP gradients via free/nudged contrast.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            energy_fn: Function to compute energy given states.
            structure_fn: Function to extract model structure.
        """
        if target is None:
            raise ValueError("Target tensor is required for Equilibrium Propagation")
        if energy_fn is None:
            raise ValueError("energy_fn is required for Equilibrium Propagation")
        if structure_fn is None:
            raise ValueError("structure_fn is required for Equilibrium Propagation")

        structure = structure_fn(model)
        
        # Free phase (beta=0)
        states_free = self._settle(
            model, x, target=None, beta=0.0,
            energy_fn=energy_fn, structure=structure
        )
        
        # Nudged phase
        states_nudged = self._settle(
            model, x, target=target, beta=self.beta,
            energy_fn=energy_fn, structure=structure
        )
        
        # Apply contrast
        self._apply_contrast(
            model, x, target, states_free, states_nudged,
            energy_fn, structure
        )
    
    def _settle(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
        energy_fn: Callable,
        structure: List[Dict[str, Any]]
    ) -> List[torch.Tensor]:
        """Settle network to energy minimum."""
        from ..settling import Settler
        
        settler = Settler(
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
            tol=self.tol,
            patience=self.patience,
            adaptive=self.adaptive,
        )
        return settler.settle(model, x, target, beta, energy_fn, structure)
    
    def _apply_contrast(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        energy_fn: Callable,
        structure: List[Dict[str, Any]]
    ) -> None:
        """Apply EP gradient: (E_nudged - E_free) / beta."""
        # Prepare target
        # Use x.dtype (typically float32) to match the precision of the contrastive step
        target_vec = self._prepare_target(
            target, states_free[-1].shape[-1], x.dtype
        )
        
        # Ensure we run in full precision for the contrastive gradient step
        device_type = x.device.type
        # Only disable if valid device for autocast
        if device_type in ['cuda', 'cpu', 'xpu', 'hpu']:
            amp_context = torch.amp.autocast(device_type=device_type, enabled=False)
        else:
            from contextlib import nullcontext
            amp_context = nullcontext() # type: ignore

        with torch.enable_grad():
            with amp_context:
                # Compute energies
                E_free = energy_fn(model, x, states_free, structure, target_vec=None, beta=0.0)
                E_nudged = energy_fn(
                    model, x, states_nudged, structure, target_vec=target_vec, beta=self.beta
                )

                # Contrast loss
                loss = (E_nudged - E_free) / self.beta
                params = list(model.parameters())
                grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

        # Set gradients (overwrite any existing gradients)
        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g.detach()
    
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


class LocalEPGradient:
    """
    Layer-local EP gradients (biologically plausible).
    
    Each layer computes its own local energy gradient based on
    immediate input/output, without cross-layer gradient flow.
    """
    
    def __init__(
        self,
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
        tol: float = 1e-4,
        patience: int = 5,
        adaptive: bool = False,
    ):
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
        self.tol = tol
        self.patience = patience
        self.adaptive = adaptive
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable] = None,
        structure_fn: Optional[Callable] = None,
        **kwargs: Any
    ) -> None:
        """Compute layer-local EP gradients."""
        if target is None:
            raise ValueError("Target tensor is required for Local Equilibrium Propagation")
        if energy_fn is None:
             raise ValueError("energy_fn is required for Local Equilibrium Propagation")
        if structure_fn is None:
             raise ValueError("structure_fn is required for Local Equilibrium Propagation")

        from ..settling import Settler
        
        structure = structure_fn(model)
        settler = Settler(
            steps=self.settle_steps,
            lr=self.settle_lr,
            loss_type=self.loss_type,
            softmax_temperature=self.softmax_temperature,
            tol=self.tol,
            patience=self.patience,
            adaptive=self.adaptive,
        )
        
        # Free phase
        states_free = settler.settle(
            model, x, target=None, beta=0.0,
            energy_fn=energy_fn, structure=structure
        )
        
        # Nudged phase
        states_nudged = settler.settle(
            model, x, target=target, beta=self.beta,
            energy_fn=energy_fn, structure=structure
        )
        
        # Apply local contrast per layer
        self._apply_local_contrast(
            model, x, target, states_free, states_nudged, structure
        )
    
    def _apply_local_contrast(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: List[Dict[str, Any]]
    ) -> None:
        """Apply EP contrast independently per layer."""
        # Extract layer I/O
        io_free = self._get_layer_io(x, states_free, structure)
        io_nudged = self._get_layer_io(x, states_nudged, structure)
        
        map_free = {id(item["module"]): item for item in io_free}
        map_nudged = {id(item["module"]): item for item in io_nudged}
        batch_size = x.shape[0]
        
        inter_layer_params: List[nn.Parameter] = []

        for item in structure:
            module = item["module"]

            if item["type"] != "layer":
                # Collect parameters of non-layer modules (e.g., BN, Norm)
                # to be updated with the next layer
                inter_layer_params.extend(module.parameters())
                continue
            
            if id(module) not in map_free or id(module) not in map_nudged:
                continue
            
            # Free phase
            in_free = map_free[id(module)]["input"]
            out_free = map_free[id(module)]["output"].detach()
            
            # Nudged phase
            in_nudged = map_nudged[id(module)]["input"]
            out_nudged = map_nudged[id(module)]["output"].detach()
            
            # Update layer params AND accumulated inter-layer params
            module_params = list(module.parameters()) + inter_layer_params
            
            with torch.enable_grad():
                pred_free = module(in_free)
                E_free = 0.5 * F.mse_loss(pred_free, out_free, reduction="sum") / batch_size
                
                pred_nudged = module(in_nudged)
                E_nudged = 0.5 * F.mse_loss(pred_nudged, out_nudged, reduction="sum") / batch_size
                
                loss = (E_nudged - E_free) / self.beta
                grads = torch.autograd.grad(loss, module_params, retain_graph=False, allow_unused=True)

            for p, g in zip(module_params, grads):
                if g is not None:
                    p.grad = g.detach()

            # Clear inter-layer params after they have been assigned to a layer update
            inter_layer_params = []

        # Handle any remaining parameters (e.g., BN after last layer)
        if inter_layer_params:
            self._update_trailing_params(
                inter_layer_params, states_nudged[-1], target, structure, x.dtype
            )

    def _update_trailing_params(
        self,
        params: List[nn.Parameter],
        last_state: torch.Tensor,
        target: torch.Tensor,
        structure: List[Dict[str, Any]],
        dtype: torch.dtype
    ) -> None:
        """Update parameters of modules after the last layer."""
        # Identify modules after the last layer
        trailing_modules = []

        # Find the point after the last "layer" or "attention"
        start_idx = 0
        for i in range(len(structure) - 1, -1, -1):
            if structure[i]["type"] in ("layer", "attention"):
                start_idx = i + 1
                break

        for i in range(start_idx, len(structure)):
            trailing_modules.append(structure[i]["module"])

        if not trailing_modules:
            return

        # Compute loss on nudged state through trailing modules
        with torch.enable_grad():
            output = last_state.detach().requires_grad_(True)

            for mod in trailing_modules:
                output = mod(output)

            # Prepare target based on final output shape
            target_vec = self._prepare_target(
                target, output.shape[-1], dtype
            )

            if self.loss_type == "cross_entropy":
                loss = F.cross_entropy(output, target_vec, reduction="sum", label_smoothing=0.1) / output.shape[0]
            else:
                 # MSE - handle potential shape mismatch
                 if output.shape != target_vec.shape:
                      if output.numel() == target_vec.numel():
                           target_vec = target_vec.view_as(output)

                 loss = F.mse_loss(output, target_vec, reduction="sum") / output.shape[0]

            grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)

        for p, g in zip(params, grads):
            if g is not None:
                p.grad = g.detach()

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

    def _get_layer_io(
        self,
        x: torch.Tensor,
        states: List[torch.Tensor],
        structure: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract layer inputs and outputs."""
        io_list = []
        prev = x
        state_idx = 0
        
        for item in structure:
            if item["type"] == "layer":
                if state_idx >= len(states):
                    break
                module = item["module"]
                state = states[state_idx]
                io_list.append({"module": module, "input": prev, "output": state})
                prev = state
                state_idx += 1
            elif item["type"] in ("act", "norm", "pool", "flatten", "dropout"):
                prev = item["module"](prev)
            elif item["type"] == "attention":
                if state_idx >= len(states):
                    break
                module = item["module"]
                state = states[state_idx]
                io_list.append({"module": module, "input": prev, "output": state})
                prev = state
                state_idx += 1
        
        return io_list


class NaturalGradient:
    """
    Natural gradient with Fisher Information whitening.
    
    Wraps a base gradient strategy and applies Fisher-based whitening
    to account for the geometry of the parameter space.
    """
    
    def __init__(
        self,
        base_strategy: GradientStrategy,
        fisher_approx: str = "empirical",
        use_diagonal: bool = False,
    ):
        self.base_strategy = base_strategy
        self.fisher_approx = fisher_approx
        self.use_diagonal = use_diagonal
    
    def compute_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable] = None,
        structure_fn: Optional[Callable] = None,
        **kwargs: Any
    ) -> None:
        """
        Compute natural gradients with Fisher whitening.
        
        First computes base gradients, then captures Fisher information.
        """
        # Get structure if structure_fn provided
        structure = None
        if structure_fn is not None:
            structure = structure_fn(model)
        
        # Compute base gradients
        call_kwargs = kwargs.copy()
        if energy_fn is not None:
            call_kwargs['energy_fn'] = energy_fn
        if structure_fn is not None:
            call_kwargs['structure_fn'] = structure_fn

        self.base_strategy.compute_gradients(
            model, x, target, **call_kwargs
        )
        
        # Capture Fisher information for later use in update
        self._compute_fisher(model, x, target, energy_fn, structure)
    
    def _compute_fisher(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        energy_fn: Optional[Callable],
        structure: Optional[List[Dict[str, Any]]]
    ) -> None:
        """
        Compute Fisher Information Matrix blocks.
        
        Stores Fisher blocks in a way accessible to NaturalUpdate strategy.
        """
        # For empirical Fisher: F = sum(g @ g.T) over samples
        # We approximate using the batch-averaged gradient (rank-1 approximation per step)
        # This is a simplification but allows running without per-sample gradients.

        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()

                # Reshape to 2D (Out, In)
                if g.ndim > 2:
                    g = g.view(g.shape[0], -1)
                elif g.ndim < 2:
                    # Vectors (biases) - skip or treat as diagonal?
                    # NaturalGradient usually targets weights.
                    continue

                # Compute Fisher proxy
                # We assume whitening along the input dimension (In, In) covariance
                # F = g.T @ g  (In, In)

                if self.use_diagonal:
                    fisher = torch.sum(g**2, dim=0) # (In,)
                else:
                    fisher = g.T @ g # (In, In)

                # Store on parameter for FisherUpdate to consume
                setattr(p, "fisher", fisher)
