# type: ignore
"""
MEP Optimizers: Spectral Muon Equilibrium Propagation

This module provides optimizers for biologically plausible deep learning
using Equilibrium Propagation (EP) with geometry-aware updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Optional, Tuple, List, Iterable, Dict, Any, Callable

# Import CUDA kernels for accelerated operations
try:
    from .cuda.kernels import (
        newton_schulz_cuda,
        dion_update_cuda,
        spectral_norm_power_iteration_cuda,
        enforce_spectral_constraint_cuda,
    )
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# Type aliases
TensorOrNone = Optional[torch.Tensor]
ModuleOrNone = Optional[nn.Module]
StructureItem = Dict[str, Any]
Structure = List[StructureItem]
StateDict = Dict[str, Any]


class EPWrapper:
    """
    Wraps the model to add automatic free-phase settling in EP mode.

    Attributes:
        model: The wrapped neural network module.
        optimizer: The associated SMEPOptimizer instance.
        original_forward: The original forward method before wrapping.
        free_states: Cached states from the free phase.
        nudged_states: Cached states from the nudged phase.
        last_input: The most recent input tensor for EP workflow.
        last_target: The most recent target tensor for EP workflow.
    """

    def __init__(self, model: nn.Module, optimizer: "SMEPOptimizer") -> None:
        self.model: nn.Module = model
        self.optimizer: SMEPOptimizer = optimizer
        self.original_forward: Callable = model.forward
        self.free_states: List[torch.Tensor] = []
        self.nudged_states: List[torch.Tensor] = []
        self.last_input: TensorOrNone = None
        self.last_target: TensorOrNone = None

    def forward(
        self,
        x: torch.Tensor,
        phase: str = "free",
        target: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Forward pass with optional EP settling.

        Args:
            x: Input tensor.
            phase: 'free' for free-phase settling, 'nudged' for nudged phase.
            target: Target tensor for nudged phase.
            **kwargs: Additional arguments passed to original forward.

        Returns:
            Output tensor (last state from settling).
        """
        # If optimizer is in backprop mode, just pass through
        if self.optimizer.defaults["mode"] == "backprop":
            return self.original_forward(x, **kwargs)  # type: ignore[no-any-return]

        if phase == "free":
            self.last_input = x
            # Run settling
            states = self.optimizer._settle(
                self.model, x, target=None, beta=0.0, forward_fn=self.original_forward
            )
            self.free_states = states
            return states[-1]  # type: ignore[no-any-return]

        elif phase == "nudged":
            # Nudged phase
            self.last_target = target
            states = self.optimizer._settle(
                self.model,
                x,
                target=target,
                beta=self.optimizer.defaults["beta"],
                forward_fn=self.original_forward,
            )
            self.nudged_states = states
            return states[-1]  # type: ignore[no-any-return]

        else:
            return self.original_forward(x, **kwargs)  # type: ignore[no-any-return]


class SMEPOptimizer(Optimizer):
    """
    SMEP Optimizer: Spectral Muon Equilibrium Propagation

    A self-contained optimizer that:
    - Computes gradients via Equilibrium Propagation OR standard backprop
    - Applies Muon (Newton-Schulz) orthogonalization to weight updates
    - Optional: Error Feedback (continual learning)
    - Optional: Spectral Constraints (Lipschitz control)
    - Works with arbitrary PyTorch models (drop-in replacement for SGD/Adam)

    Attributes:
        model: Optional model instance for new EP API.
        ep_wrapper: EPWrapper instance for automatic settling.
    """

    # Constants for numerical stability
    EPSILON_NORM: float = 1e-6  # Small value to prevent division by zero
    EPSILON_SPECTRAL: float = 1e-6  # Epsilon for spectral norm power iteration
    SETTLING_MOMENTUM: float = 0.5  # Momentum for state settling optimizer
    SPECTRAL_POWER_ITER: int = 3  # Number of power iterations for spectral norm

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        model: ModuleOrNone = None,
        lr: float = 0.02,
        momentum: float = 0.9,
        wd: float = 0.0005,
        mode: str = "backprop",
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        ns_steps: int = 5,
        use_error_feedback: bool = True,
        error_beta: float = 0.9,
        use_spectral_constraint: bool = True,
        gamma: float = 0.95,
        spectral_timing: str = "post_update",
        spectral_lambda: float = 1.0,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
        max_grad_norm: float = 10.0,
    ) -> None:
        """
        Initialize SMEPOptimizer.

        Args:
            params: Iterable of parameters to optimize.
            model: Optional model instance. Required for new EP API (mode='ep').
            lr: Learning rate. Must be positive.
            momentum: Momentum factor. Must be in [0, 1).
            wd: Weight decay coefficient. Must be non-negative.
            mode: 'backprop' or 'ep'.
            beta: Nudging strength for EP. Must be in (0, 1].
            settle_steps: Number of settling steps for EP. Must be positive.
            settle_lr: Learning rate for settling optimization. Must be positive.
            ns_steps: Newton-Schulz iterations. Must be positive.
            use_error_feedback: Enable error feedback for continual learning.
            error_beta: Error feedback decay factor. Must be in [0, 1).
            use_spectral_constraint: Enable spectral constraint for stability.
            gamma: Spectral norm bound. Must be in (0, 1].
            spectral_timing: 'post_update', 'during_settling', or 'both'.
            spectral_lambda: Strength of spectral penalty during settling.
            loss_type: 'mse' for regression or 'cross_entropy' for classification.
            softmax_temperature: Temperature for softmax in classification (lower = sharper).
            max_grad_norm: Maximum gradient norm for clipping (prevents exploding gradients).

        Raises:
            ValueError: If any parameter validation fails.
        """
        # Input validation
        self._validate_hyperparameters(
            lr=lr,
            momentum=momentum,
            wd=wd,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            ns_steps=ns_steps,
            error_beta=error_beta,
            gamma=gamma,
            spectral_timing=spectral_timing,
            spectral_lambda=spectral_lambda,
        )

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            wd=wd,
            ns_steps=ns_steps,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            use_error_feedback=use_error_feedback,
            error_beta=error_beta,
            use_spectral_constraint=use_spectral_constraint,
            gamma=gamma,
            spectral_timing=spectral_timing,
            spectral_lambda=spectral_lambda,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

        # Cache for model structure to avoid repeated introspection
        self._model_structure_cache: Dict[int, Structure] = {}

        self.model: ModuleOrNone = model
        self.ep_wrapper: Optional[EPWrapper] = None

        # Check if already wrapped and unwrap if so (to avoid nesting)
        if self.model is not None:
            if hasattr(self.model.forward, "__self__") and isinstance(
                self.model.forward.__self__, EPWrapper
            ):
                old_wrapper = self.model.forward.__self__
                self.model.forward = old_wrapper.original_forward

        if mode == "ep" and self.model is not None:
            # One-time wrap
            self.ep_wrapper = EPWrapper(self.model, self)
            self.model.forward = self.ep_wrapper.forward

    @staticmethod
    def _validate_hyperparameters(
        lr: float,
        momentum: float,
        wd: float,
        mode: str,
        beta: float,
        settle_steps: int,
        settle_lr: float,
        ns_steps: int,
        error_beta: float,
        gamma: float,
        spectral_timing: str,
        spectral_lambda: float,
    ) -> None:
        """
        Validate optimizer hyperparameters.

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if not (0 <= momentum < 1):
            raise ValueError(f"Momentum must be in [0, 1), got {momentum}")
        if wd < 0:
            raise ValueError(f"Weight decay must be non-negative, got {wd}")
        if mode not in ["backprop", "ep"]:
            raise ValueError(f"Mode must be 'backprop' or 'ep', got '{mode}'")
        if beta <= 0 or beta > 1:
            raise ValueError(f"Beta must be in (0, 1], got {beta}")
        if settle_steps <= 0:
            raise ValueError(f"Settle steps must be positive, got {settle_steps}")
        if settle_lr <= 0:
            raise ValueError(f"Settle learning rate must be positive, got {settle_lr}")
        if ns_steps < 0:
            raise ValueError(f"Newton-Schulz steps must be non-negative, got {ns_steps}")
        if not (0 <= error_beta < 1):
            raise ValueError(f"Error beta must be in [0, 1), got {error_beta}")
        if not (0 < gamma <= 1):
            raise ValueError(f"Gamma must be in (0, 1], got {gamma}")
        if spectral_timing not in ["post_update", "during_settling", "both"]:
            raise ValueError(
                f"Spectral timing must be 'post_update', 'during_settling', or 'both', "
                f"got '{spectral_timing}'"
            )
        if spectral_lambda < 0:
            raise ValueError(f"Spectral lambda must be non-negative, got {spectral_lambda}")

    @torch.no_grad()
    def newton_schulz(self, G: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Newton-Schulz orthogonalization (Muon update).

        Uses CUDA-accelerated implementation when available.

        Args:
            G: Gradient tensor (must be 2D for orthogonalization).
            steps: Number of Newton-Schulz iterations.

        Returns:
            Orthogonalized gradient tensor.

        Raises:
            RuntimeError: If NaN or Inf detected in output.
        """
        if G.ndim != 2:
            return G

        # Use CUDA kernel if available and on GPU
        if CUDA_AVAILABLE and G.is_cuda:
            return newton_schulz_cuda(G, steps=steps, epsilon=self.EPSILON_NORM)

        # Fallback to CPU implementation
        r, c = G.shape

        # Handle rectangular matrices by transposing if needed
        transposed = False
        if r < c:
            transposed = True
            G = G.T
            r, c = c, r

        # Pre-normalize to ensure convergence (Frobenius)
        X = G.clone()
        norm = X.norm().clamp(min=1e-4, max=1e4) + self.EPSILON_NORM
        X.div_(norm)

        # NS Iteration: X = 0.5 * X * (3I - X^T X)
        identity = torch.eye(c, device=G.device, dtype=G.dtype)
        for _ in range(steps):
            A = X.T @ X
            X = 0.5 * X @ (3 * identity - A)

        # Do NOT restore scale. Return orthogonal matrix X.
        res = X

        if transposed:
            res = res.T

        # Numerical stability check
        if torch.isnan(res).any() or torch.isinf(res).any():
            raise RuntimeError(
                f"Newton-Schulz produced NaN/Inf. Input norm: {G.norm().item():.4f}. "
                f"Try reducing learning rate or Newton-Schulz steps."
            )

        return res  # type: ignore[no-any-return]

    @torch.no_grad()
    def get_spectral_norm(
        self, W: torch.Tensor, u: TensorOrNone, v: TensorOrNone, iter: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute spectral norm via power iteration.

        Uses CUDA-accelerated implementation when available.

        Args:
            W: Weight matrix.
            u: Left singular vector (cached).
            v: Right singular vector (cached).
            iter: Number of power iterations (defaults to class constant).

        Returns:
            Tuple of (spectral_norm, updated_u, updated_v).
        """
        if iter is None:
            iter = self.SPECTRAL_POWER_ITER

        # Use CUDA kernel if available and on GPU
        if CUDA_AVAILABLE and W.is_cuda:
            return spectral_norm_power_iteration_cuda(
                W, u, v, niter=iter, epsilon=self.EPSILON_SPECTRAL
            )

        # Fallback to CPU implementation
        if W.ndim > 2:
            W = W.view(W.shape[0], -1)

        h, w = W.shape
        if u is None:
            u = torch.randn(h, device=W.device, dtype=W.dtype)
            u /= u.norm() + self.EPSILON_NORM
        if v is None:
            v = torch.randn(w, device=W.device, dtype=W.dtype)
            v /= v.norm() + self.EPSILON_NORM

        for _ in range(iter):
            v = W.T @ u
            v /= v.norm() + self.EPSILON_SPECTRAL
            u = W @ v
            u /= u.norm() + self.EPSILON_SPECTRAL

        sigma = (u @ W @ v).abs()
        return sigma, u, v

    def _prepare_target(
        self, target: torch.Tensor, num_classes: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Convert target to one-hot if needed, matching dtype.

        Args:
            target: Target tensor (class indices or one-hot).
            num_classes: Number of output classes.
            dtype: Target dtype for one-hot encoding.

        Returns:
            One-hot encoded target tensor (MSE) or indices (CrossEntropy).
        """
        loss_type = self.defaults.get("loss_type", "mse")
        if loss_type == "cross_entropy":
            # For classification: ensure we have class indices (LongTensor)
            if target.dim() > 1 and target.shape[1] > 1:
                # Convert one-hot to class indices
                return target.argmax(dim=1).long()
            # Handle potential (N, 1) or (N,) indices
            return target.squeeze().long()
        else:
            # MSE: ensure one-hot if provided as indices
            if target.dim() == 1:
                one_hot = F.one_hot(target, num_classes=num_classes).to(dtype=dtype)
                return one_hot
            return target.to(dtype=dtype)

    def _inspect_model(self, model: nn.Module) -> Structure:
        """
        Extract sequence of layers and activations (cached).

        Supports:
        - Linear layers (nn.Linear)
        - Convolutional layers (nn.Conv1d, nn.Conv2d, nn.Conv3d)
        - Transformer layers (nn.MultiheadAttention, nn.TransformerEncoderLayer)
        - Normalization layers (nn.LayerNorm, nn.BatchNorm*)
        - Activations (nn.ReLU, nn.GELU, etc.)

        Args:
            model: Neural network to inspect.

        Returns:
            List of structure items with 'type' and 'module' keys.
        """
        model_id = id(model)
        if model_id in self._model_structure_cache:
            return self._model_structure_cache[model_id]

        structure: Structure = []
        for m in model.modules():
            # Convolutional layers
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                structure.append({"type": "layer", "module": m})
            # Transformer attention
            elif isinstance(m, nn.MultiheadAttention):
                structure.append({"type": "attention", "module": m})
            # Normalization layers (track but don't treat as full layers)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                structure.append({"type": "norm", "module": m})
            # Activations and other non-linearities
            elif isinstance(
                m,
                (
                    nn.ReLU,
                    nn.Sigmoid,
                    nn.Tanh,
                    nn.LeakyReLU,
                    nn.Softmax,
                    nn.Flatten,
                    nn.Dropout,
                    nn.GELU,
                    nn.SiLU,
                    nn.ELU,
                    nn.CELU,
                    nn.GLU,
                    nn.Hardswish,
                    nn.Mish,
                ),
            ):
                structure.append({"type": "act", "module": m})
            # Pooling layers
            elif isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                               nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                               nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d)):
                structure.append({"type": "pool", "module": m})

        self._model_structure_cache[model_id] = structure
        return structure

    def _compute_energy(
        self,
        model: nn.Module,
        x: torch.Tensor,
        states: List[torch.Tensor],
        structure: Structure,
        target_vec: Optional[torch.Tensor] = None,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute total energy: E = E_int + E_ext

        E_int = 0.5 * mean_over_batch sum_over_features || s_i - f_i(s_{i-1}) ||^2
        E_ext = beta * Loss(s_last, target)

        For CrossEntropy loss, the internal energy uses a softmax-aware formulation:
        - Hidden layers: standard MSE settling
        - Output layer: KL divergence between softmax(state) and softmax(prediction)
          This better matches the geometry of classification problems.

        Note: Energy is normalized by batch size to ensure batch-size invariance.

        Args:
            model: Neural network module.
            x: Input tensor.
            states: List of layer states from forward pass.
            structure: Model structure from _inspect_model.
            target_vec: Optional target vector for nudge term.
            beta: Nudging strength.

        Returns:
            Scalar energy tensor.

        Raises:
            RuntimeError: If energy becomes NaN or Inf.
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError(f"Batch size cannot be zero, got input shape {x.shape}")

        loss_type = self.defaults.get("loss_type", "mse")
        use_classification = loss_type == "cross_entropy"
        softmax_temp = self.defaults.get("softmax_temperature", 1.0)

        E = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        prev = x
        state_idx = 0

        # Identify the last layer for special handling in classification
        layer_modules = [item["module"] for item in structure if item["type"] == "layer"]
        last_layer_idx = len(layer_modules) - 1

        # Iterate through structure to reconstruct graph
        for item in structure:
            item_type = item["type"]
            module = item["module"]

            if item_type == "layer":
                if state_idx >= len(states):
                    break

                state = states[state_idx]
                is_last_layer = (state_idx == last_layer_idx)

                # Prediction from previous state
                h = module(prev)

                # Energy mismatch computation
                if use_classification and is_last_layer:
                    # For classification output layer: use KL divergence on softmax
                    # This matches the CrossEntropy geometry better than raw MSE
                    eps = 1e-8
                    # Correct: apply softmax first, then add eps for log stability
                    state_softmax = F.softmax(state / softmax_temp, dim=-1)
                    h_softmax = F.softmax(h / softmax_temp, dim=-1)
                    # KL divergence: D_KL(state_softmax || h_softmax)
                    kl_div = F.kl_div(
                        torch.log(state_softmax + eps), h_softmax, reduction="sum"
                    )
                    E = E + kl_div / batch_size
                else:
                    # Standard MSE for hidden layers or regression
                    E = E + 0.5 * F.mse_loss(h, state, reduction="sum") / batch_size

                # Next input base is current state
                prev = state
                state_idx += 1

            elif item_type == "norm":
                # Normalization layers: apply to current flow
                # These don't have learnable states in the EP sense
                prev = module(prev)

            elif item_type == "pool":
                # Pooling layers: apply to current flow
                prev = module(prev)

            elif item_type == "attention":
                # Multihead attention: treat as a layer for state tracking
                if state_idx >= len(states):
                    break

                state = states[state_idx]

                # For attention, we need to handle the multi-output case
                # module returns (attn_output, attn_weights) or just attn_output
                if isinstance(module, nn.MultiheadAttention):
                    # Self-attention with same Q=K=V=prev
                    # need_weights=False returns just the output
                    try:
                        h = module(prev, prev, prev, need_weights=False)[0]
                    except (RuntimeError, AssertionError):
                        # Fallback: skip attention energy for this step
                        prev = state
                        state_idx += 1
                        continue
                else:
                    h = module(prev)

                # MSE energy for attention states
                E = E + 0.5 * F.mse_loss(h, state, reduction="sum") / batch_size
                prev = h
                state_idx += 1

            elif item_type == "act":
                # Apply activation to current flow
                prev = module(prev)

        # Nudge term (consistent reduction with E_int)
        if target_vec is not None and beta > 0:
            output = prev
            if loss_type == "cross_entropy":
                # For classification: target_vec contains class indices (Long)
                # Use label smoothing for better numerical stability
                E = E + beta * F.cross_entropy(
                    output, target_vec, reduction="sum", label_smoothing=0.1
                ) / batch_size
            else:
                # MSE for regression
                E = E + beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size

        # Numerical stability check
        if torch.isnan(E) or torch.isinf(E):
            raise RuntimeError(
                f"Energy computation produced NaN/Inf. "
                f"Input shape: {x.shape}, States: {len(states)}, "
                f"Target: {target_vec.shape if target_vec is not None else None}"
            )

        return E

    def _settle(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        beta: float = 0.0,
        forward_fn: Optional[Callable] = None,
    ) -> List[torch.Tensor]:
        """
        Settle network activations to minimize energy.

        Args:
            model: Neural network to settle.
            x: Input tensor.
            target: Optional target for nudged settling.
            beta: Nudging strength.
            forward_fn: Optional forward function override.

        Returns:
            List of settled state tensors.

        Raises:
            ValueError: If input tensor is invalid.
            RuntimeError: If settling fails to capture activations or energy diverges.
        """
        # Input validation
        if x.numel() == 0:
            raise ValueError(f"Input tensor cannot be empty, got shape {x.shape}")
        if beta < 0 or beta > 1:
            raise ValueError(f"Beta must be in [0, 1], got {beta}")

        if forward_fn is None:
            forward_fn = model

        # Introspect model
        structure = self._inspect_model(model)

        # Capture initial states via forward pass
        states: List[torch.Tensor] = []
        handles: List[Any] = []

        def capture_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            # Handle tuple outputs (e.g., from MultiheadAttention)
            if isinstance(output, tuple):
                states.append(output[0].detach().clone().requires_grad_(True))
            else:
                states.append(output.detach().clone().requires_grad_(True))

        for item in structure:
            # Capture states for layers and attention modules
            if item["type"] in ("layer", "attention"):
                handles.append(item["module"].register_forward_hook(capture_hook))

        try:
            with torch.no_grad():
                forward_fn(x)
        finally:
            for h in handles:
                h.remove()

        if not states:
            # Provide helpful debugging information
            layer_count = sum(1 for item in structure if item["type"] == "layer")
            raise RuntimeError(
                f"No activations captured during settling. "
                f"Expected {layer_count} layer(s) but got 0.\n"
                f"Model type: {type(model).__name__}\n"
                f"Model structure contains {len(structure)} items: "
                f"{', '.join(item['type'] for item in structure[:5])}"
                f"{'...' if len(structure) > 5 else ''}\n"
                f"Ensure model contains nn.Linear or nn.Conv2d layers."
            )

        # Optimization loop
        # Manual SGD to relax states (avoids Optimizer overhead)
        lr = self.defaults["settle_lr"]
        momentum = self.SETTLING_MOMENTUM
        momentum_buffers = [torch.zeros_like(s) for s in states]

        # Pre-calculate spectral penalty if needed
        spectral_penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Check if any group needs spectral penalty
        needs_spectral = False
        for group in self.param_groups:
            timing = group.get(
                "spectral_timing", self.defaults.get("spectral_timing", "post_update")
            )
            if timing in ["during_settling", "both"] and group.get(
                "use_spectral_constraint", False
            ):
                needs_spectral = True
                break

        if needs_spectral:
            for group in self.param_groups:
                timing = group.get(
                    "spectral_timing", self.defaults.get("spectral_timing", "post_update")
                )
                if timing in ["during_settling", "both"] and group.get(
                    "use_spectral_constraint", False
                ):
                    for p in group["params"]:
                        if p.ndim >= 2:
                            state = self.state[p]
                            if "u_spec" not in state:
                                state["u_spec"] = None
                            if "v_spec" not in state:
                                state["v_spec"] = None

                            sigma, u, v = self.get_spectral_norm(
                                p, state["u_spec"], state["v_spec"]
                            )
                            # Update cached vectors
                            state["u_spec"] = u.detach()
                            state["v_spec"] = v.detach()

                            if sigma > group["gamma"]:
                                diff = sigma - group["gamma"]
                                spectral_penalty = spectral_penalty + group.get(
                                    "spectral_lambda", 1.0
                                ) * (diff**2)

        # Prepare target vector if needed
        target_vec: TensorOrNone = None
        if target is not None:
            # Use states[-1] dtype to ensure target matches computation precision
            target_vec = self._prepare_target(target, states[-1].shape[-1], dtype=states[-1].dtype)

        # Settling loop with energy monitoring
        prev_energy: Optional[float] = None
        for step in range(self.defaults["settle_steps"]):
            with torch.enable_grad():
                E = self._compute_energy(model, x, states, structure, target_vec, beta)
                if spectral_penalty > 0:
                    E = E + spectral_penalty

                # Check for energy divergence
                E_val = E.item()
                if torch.isnan(E) or torch.isinf(E):
                    raise RuntimeError(
                        f"Energy diverged at settling step {step}: E={E_val}. "
                        f"Try reducing settle_lr, beta, or learning rate."
                    )

                # Warn if energy increased significantly (optional monitoring)
                if prev_energy is not None and E_val > prev_energy * 1.5:
                    # Energy increased by >50%, could indicate instability
                    pass  # Could add warning here if needed

                prev_energy = E_val

                grads = torch.autograd.grad(E, states, retain_graph=False, allow_unused=True)

                # if beta > 0 and step == 0:
                #     grad_norms = [g.norm().item() if g is not None else -1.0 for g in grads]
                #     print(f"DEBUG: Beta={beta}, E={E.item()}, Len(states)={len(states)}, Grad Norms={grad_norms}")

            # Manual SGD step
            with torch.no_grad():
                for i, (state, g) in enumerate(zip(states, grads)):
                    if g is None:
                        continue

                    # Momentum update: v = m * v + g
                    buf = momentum_buffers[i]
                    buf.mul_(momentum).add_(g)

                    # State update: s = s - lr * v
                    state.sub_(buf, alpha=lr)

        return [s.detach() for s in states]

    def _apply_ep_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: Structure,
    ) -> None:
        """
        Compute and accumulate EP gradients given free and nudged states.

        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            states_free: States from free phase settling.
            states_nudged: States from nudged phase settling.
            structure: Model structure.

        Raises:
            RuntimeError: If gradient computation produces NaN/Inf.
        """
        # Prepare target (use helper, matching free state precision)
        target_vec = self._prepare_target(
            target, states_free[-1].shape[-1], dtype=states_free[-1].dtype
        )

        # E_free
        E_free = self._compute_energy(model, x, states_free, structure, target_vec=None, beta=0.0)

        # E_nudged
        E_nudged = self._compute_energy(
            model, x, states_nudged, structure, target_vec, beta=self.defaults["beta"]
        )

        params_list = list(model.parameters())

        # Optimize: Single backward pass for (E_nudged - E_free) / beta
        loss = (E_nudged - E_free) / self.defaults["beta"]
        grads = torch.autograd.grad(loss, params_list, retain_graph=False, allow_unused=True)

        # Gradient clipping for numerical stability
        max_grad_norm = self.defaults.get("max_grad_norm", 10.0)
        total_norm = torch.norm(
            torch.stack([torch.norm(g) for g in grads if g is not None])
        )
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads = [
                g.clone() * clip_coef if g is not None else None
                for g in grads
            ]

        for p, g in zip(model.parameters(), grads):
            if g is None:
                continue

            # Check for NaN/Inf in EP gradients
            if torch.isnan(g).any() or torch.isinf(g).any():
                raise RuntimeError(
                    "EP gradient computation produced NaN/Inf. "
                    f"Beta: {self.defaults['beta']}. "
                    f"Try reducing beta or learning rate."
                )

            if p.grad is None:
                p.grad = g.detach()
            else:
                p.grad.add_(g.detach())

    def _compute_ep_gradients(
        self, model: nn.Module, x: torch.Tensor, target: torch.Tensor
    ) -> None:
        """
        Compute EP gradients via free and nudged phase contrast.

        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
        """
        # 1. Inspect model structure
        structure = self._inspect_model(model)

        # Determine correct forward function (unwrap if necessary)
        # This prevents recursive settling if the model is wrapped with EPWrapper
        forward_fn = model
        if hasattr(model.forward, "__self__") and isinstance(
            model.forward.__self__, EPWrapper
        ):
            forward_fn = model.forward.__self__.original_forward

        # 2. Free Phase
        states_free = self._settle(
            model, x, target=None, beta=0.0, forward_fn=forward_fn
        )

        # 3. Nudged Phase
        states_nudged = self._settle(
            model, x, target=target, beta=self.defaults["beta"], forward_fn=forward_fn
        )

        # 4. Compute Gradients via Contrast
        self._apply_ep_gradients(
            model, x, target, states_free, states_nudged, structure
        )

    @torch.no_grad()
    def step(  # type: ignore[override]
        self,
        closure: Optional[Callable[[], float]] = None,
        x: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Optional[float]:
        """
        Perform optimization step.

        New EP API:
            optimizer = SMEPOptimizer(..., model=model, mode='ep')
            output = model(x)  # Triggers free phase settling
            optimizer.step(target=y)  # Triggers nudged phase & update

        Legacy EP API:
            optimizer = SMEPOptimizer(..., mode='ep')
            optimizer.step(x=x, target=y, model=model)

        Backprop API:
            optimizer = SMEPOptimizer(..., mode='backprop')
            loss.backward()
            optimizer.step()

        Args:
            closure: Optional closure for second-order optimization.
            x: Input tensor (required for EP mode if not using wrapped model).
            target: Target tensor (required for EP mode).
            model: Model instance (required for legacy EP API).

        Returns:
            Loss value if closure is provided, None otherwise.

        Raises:
            ValueError: If required arguments are missing for EP mode.
            RuntimeError: If NaN/Inf detected during optimization.
        """
        # Handle backward compatibility: if closure looks like a tensor,
        # it's actually x from legacy positional call: step(x, target, model)
        if closure is not None and not callable(closure):
            # Legacy positional call: step(x, target, model)
            # Arguments got shifted: closure=x, x=target, target=model, model=None
            # Need to unshift them
            model = target  # model was passed as target
            target = x  # target was passed as x
            x = closure  # x was passed as closure
            closure = None

        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # If mode='ep', compute gradients via EP
        mode = self.defaults["mode"]
        if mode == "ep":
            if x is None:
                # New mode: check if model wrapped and target present
                if self.model is None or self.ep_wrapper is None:
                    raise ValueError(
                        "For EP mode, pass model=model at optimizer creation "
                        "OR pass x, target, model to step()"
                    )

                if target is None:
                    raise ValueError("EP mode requires target=y in step(target=y)")

                x_input = self.ep_wrapper.last_input
                if x_input is None:
                    raise RuntimeError(
                        "In EP mode with wrapped model, you must call "
                        "model(x) before optimizer.step(target=y)"
                    )

                with torch.enable_grad():
                    # Nudged phase
                    self.ep_wrapper.forward(x_input, phase="nudged", target=target)
                    states_nudged = self.ep_wrapper.nudged_states
                    states_free = self.ep_wrapper.free_states

                    # Compute gradients
                    structure = self._inspect_model(self.model)
                    self._apply_ep_gradients(
                        self.model, x_input, target, states_free, states_nudged, structure
                    )

            else:
                # Legacy mode
                if target is None or model is None:
                    # If model was passed in init, use it as default
                    model = model or self.model
                    if model is None or target is None:
                        raise ValueError("mode='ep' requires x, target, and model arguments")

                # Temporarily enable gradients for EP computation
                with torch.enable_grad():
                    self._compute_ep_gradients(model, x, target)

        # Apply Muon updates (works for both backprop and EP gradients)
        for group in self.param_groups:
            # Initialize momentum buffers
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                if "error_buffer" not in state:
                    state["error_buffer"] = torch.zeros_like(p)
                if "u_spec" not in state:
                    state["u_spec"] = None
                if "v_spec" not in state:
                    state["v_spec"] = None

                g = p.grad.data

                # --- Error Feedback ---
                if group["use_error_feedback"]:
                    g_aug = g + group["error_beta"] * state["error_buffer"]
                else:
                    g_aug = g
                    # Clear buffer if unused
                    state["error_buffer"].zero_()

                update = g_aug.clone()

                # --- Update Calculation (Muon / Dion) ---
                if p.ndim >= 2:
                    orig_shape = p.shape
                    if p.ndim > 2:
                        g_flat = g_aug.view(g_aug.shape[0], -1)
                    else:
                        g_flat = g_aug

                    # Override hook for subclasses (e.g. Dion)
                    update_flat = self._compute_update(p, g_flat, group, state, g_aug, orig_shape)

                    if p.ndim > 2:
                        update = update_flat.view(orig_shape)
                    else:
                        update = update_flat
                else:
                    update = g

                # Check for NaN/Inf in update
                if torch.isnan(update).any() or torch.isinf(update).any():
                    raise RuntimeError(
                        f"Update produced NaN/Inf for parameter {p.shape}. "
                        f"Gradient norm: {g.norm().item():.4f}. "
                        f"Try reducing learning rate."
                    )

                # --- Momentum ---
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(update)

                # --- Weight Decay + Apply Update ---
                p.data.mul_(1 - group["lr"] * group["wd"])
                p.data.add_(buf, alpha=-group["lr"])

                # --- Spectral Constraint ---
                spectral_timing = group.get("spectral_timing", "post_update")
                if (
                    group["use_spectral_constraint"]
                    and p.ndim >= 2
                    and spectral_timing in ["post_update", "both"]
                ):
                    sigma, u, v = self.get_spectral_norm(p.data, state["u_spec"], state["v_spec"])
                    state["u_spec"] = u
                    state["v_spec"] = v
                    if sigma > group["gamma"]:
                        p.data.mul_(group["gamma"] / sigma)

        return loss

    def _compute_update(
        self,
        p: nn.Parameter,
        g_flat: torch.Tensor,
        group: Dict[str, Any],
        state: StateDict,
        g_aug: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Default update strategy: Pure Muon (Newton-Schulz).

        Subclasses (SDMEP) can override this to add Dion logic.

        Args:
            p: Parameter tensor.
            g_flat: Flattened gradient.
            group: Parameter group dict.
            state: Optimizer state dict.
            g_aug: Augmented gradient (with error feedback).
            orig_shape: Original parameter shape.

        Returns:
            Updated gradient tensor.
        """
        # Default: Muon for everything
        update_flat = self.newton_schulz(g_flat, group["ns_steps"])

        # When using pure Muon, error buffer is zeroed (no residual)
        state["error_buffer"].zero_()

        return update_flat


class SDMEPOptimizer(SMEPOptimizer):
    """
    SDMEP Optimizer: Spectral Dion-Muon Equilibrium Propagation

    This is SMEP but with Dion (Low-Rank) updates for large matrices.

    For large weight matrices (numel > dion_thresh), uses low-rank SVD
    (Dion) with error feedback. For smaller matrices, uses full-rank
    Newton-Schulz (Muon).
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        model: ModuleOrNone = None,
        lr: float = 0.02,
        momentum: float = 0.9,
        wd: float = 0.0005,
        gamma: float = 0.95,
        rank_frac: float = 0.2,
        error_beta: float = 0.9,
        dion_thresh: int = 100000,
        mode: str = "backprop",
        beta: float = 0.5,
        settle_steps: int = 20,
        settle_lr: float = 0.05,
        ns_steps: int = 5,
        use_error_feedback: bool = True,
        use_spectral_constraint: bool = True,
        loss_type: str = "mse",
        softmax_temperature: float = 1.0,
        max_grad_norm: float = 10.0,
    ) -> None:
        """
        Initialize SDMEPOptimizer.

        Args:
            params: Iterable of parameters to optimize.
            model: Optional model instance.
            lr: Learning rate.
            momentum: Momentum factor.
            wd: Weight decay coefficient.
            gamma: Spectral norm bound.
            rank_frac: Fraction of singular values to retain for Dion.
            error_beta: Error feedback decay factor.
            dion_thresh: Parameter count threshold for Dion vs Muon.
            mode: 'backprop' or 'ep'.
            beta: Nudging strength for EP.
            settle_steps: Number of settling steps.
            settle_lr: Settling learning rate.
            ns_steps: Newton-Schulz iterations.
            use_error_feedback: Enable error feedback.
            use_spectral_constraint: Enable spectral constraint.
            loss_type: 'mse' for regression or 'cross_entropy' for classification.
            softmax_temperature: Temperature for softmax in classification.
            max_grad_norm: Maximum gradient norm for clipping.
        """
        # Validate Dion-specific parameters
        if not (0 < rank_frac <= 1):
            raise ValueError(f"rank_frac must be in (0, 1], got {rank_frac}")
        if dion_thresh <= 0:
            raise ValueError(f"dion_thresh must be positive, got {dion_thresh}")

        # Call parent (SMEPOptimizer) constructor
        super().__init__(
            params=params,
            model=model,
            lr=lr,
            momentum=momentum,
            wd=wd,
            mode=mode,
            beta=beta,
            settle_steps=settle_steps,
            settle_lr=settle_lr,
            ns_steps=ns_steps,
            use_error_feedback=use_error_feedback,
            error_beta=error_beta,
            use_spectral_constraint=use_spectral_constraint,
            gamma=gamma,
            loss_type=loss_type,
            softmax_temperature=softmax_temperature,
            max_grad_norm=max_grad_norm,
        )

        # Add SDMEP-specific parameters to defaults
        for group in self.param_groups:
            group["rank_frac"] = rank_frac
            group["dion_thresh"] = dion_thresh

    def _compute_update(
        self,
        p: nn.Parameter,
        g_flat: torch.Tensor,
        group: Dict[str, Any],
        state: StateDict,
        g_aug: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Override: Use Dion for large matrices, Muon for small ones.

        Uses CUDA-accelerated SVD when available.
        Includes adaptive rank selection and numerical safeguards.

        Args:
            p: Parameter tensor.
            g_flat: Flattened gradient.
            group: Parameter group dict.
            state: Optimizer state dict.
            g_aug: Augmented gradient.
            orig_shape: Original parameter shape.

        Returns:
            Updated gradient tensor.
        """
        if p.numel() > group["dion_thresh"]:
            # --- DION (Low-rank SVD) ---
            # Adaptive rank selection based on gradient properties
            base_rank = max(1, int(min(g_flat.shape) * group["rank_frac"]))
            
            # Clamp rank to valid range
            max_possible_rank = min(g_flat.shape)
            rank = min(base_rank, max_possible_rank)
            
            # Ensure rank is at least 1 and at most matrix dimensions
            rank = max(1, min(rank, max_possible_rank))

            try:
                # Gradient clipping before SVD for numerical stability
                grad_norm = g_flat.norm()
                max_grad_norm = group.get("max_grad_norm", 10.0)
                if grad_norm > max_grad_norm:
                    g_flat = g_flat * (max_grad_norm / (grad_norm + 1e-8))

                # Use CUDA kernel if available and on GPU
                if CUDA_AVAILABLE and g_flat.is_cuda:
                    # Use accelerated Dion update with error feedback
                    error_buf = state["error_buffer"] if group["use_error_feedback"] else None
                    update_lowrank, new_error_buf = dion_update_cuda(
                        g_flat,
                        rank=rank,
                        error_buffer=error_buf,
                        error_beta=group["error_beta"],
                    )
                    # Update error buffer if using error feedback
                    if group["use_error_feedback"] and new_error_buf is not None:
                        # Clip error buffer to prevent accumulation
                        error_max = group.get("max_grad_norm", 10.0) * 2
                        new_error_buf = new_error_buf.clamp(-error_max, error_max)
                        state["error_buffer"].copy_(new_error_buf)
                else:
                    # Fallback to CPU implementation
                    # U: (M, r), S: (r,), V: (N, r)
                    U, S, V = torch.svd_lowrank(g_flat, q=rank)

                    # Reconstruct: U @ V.T (scale-invariant, analogous to Muon)
                    # This ignores the singular values S (gradient magnitude)
                    update_lowrank = U @ V.T

                    # Error Feedback: Update buffer with residual
                    if group["use_error_feedback"]:
                        residual = g_flat - update_lowrank
                        # Clip residual to prevent explosion
                        error_max = group.get("max_grad_norm", 10.0) * 2
                        residual = residual.clamp(-error_max, error_max)
                        state["error_buffer"].mul_(group["error_beta"]).add_(residual)

                # Final numerical check
                if torch.isnan(update_lowrank).any() or torch.isinf(update_lowrank).any():
                    # Fallback to Muon if Dion produces NaN/Inf
                    update_flat = self.newton_schulz(g_flat, group["ns_steps"])
                    state["error_buffer"].zero_()
                    return update_flat

                return update_lowrank.view(g_flat.shape)

            except (RuntimeError, torch.linalg.LinAlgError) as e:
                # Fallback to Muon if SVD fails
                update_flat = self.newton_schulz(g_flat, group["ns_steps"])
                state["error_buffer"].zero_()
                return update_flat
        else:
            # --- MUON (Newton-Schulz) ---
            # Use parent implementation
            update_flat = self.newton_schulz(g_flat, group["ns_steps"])

            # Full rank -> no error
            state["error_buffer"].zero_()

            return update_flat


class LocalEPMuon(SMEPOptimizer):
    """
    EP with layer-local Newton-Schulz orthogonalization.

    This optimizer enforces **biological plausibility** by preventing global
    gradient communication. Instead of backpropagating errors across layers
    (or using global contrastive signals), each layer computes its own local
    energy gradient based on its immediate input and output.

    Key Features:
    - **Layer-Local Updates:** Each layer updates independently using only
      local information.
    - **No Weight Transport:** Avoids the need for symmetric feedback weights.
    - **Contrastive Hebbian Learning:** Updates are driven by the difference
      between free and nudged phase activities.

    Usage:
        optimizer = LocalEPMuon(model.parameters(), model=model, mode='ep', beta=0.1)
        # Standard EP loop
        output = model(x)
        optimizer.step(target=y)
    """

    def _compute_update(
        self,
        p: nn.Parameter,
        g_flat: torch.Tensor,
        group: Dict[str, Any],
        state: StateDict,
        g_aug: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Compute local update using only this layer's gradients.

        Args:
            p: Parameter tensor.
            g_flat: Flattened gradient.
            group: Parameter group dict.
            state: Optimizer state dict.
            g_aug: Augmented gradient.
            orig_shape: Original parameter shape.

        Returns:
            Newton-Schulz orthogonalized gradient.
        """
        # Only use gradients from this layer's immediate context
        # No access to cross-layer gradients
        return self.newton_schulz(g_flat, group["ns_steps"])

    def _get_layer_io(
        self, model: nn.Module, x: torch.Tensor, states: List[torch.Tensor], structure: Structure
    ) -> List[Dict[str, Any]]:
        """
        Extract layer inputs and outputs from states.

        Args:
            model: Neural network module.
            x: Input tensor.
            states: List of layer states.
            structure: Model structure.

        Returns:
            List of dicts with 'module', 'input', 'output' keys.
        """
        io_list: List[Dict[str, Any]] = []
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
            elif item["type"] == "act":
                prev = item["module"](prev)
        return io_list

    def _apply_ep_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: Structure,
    ) -> None:
        """
        Compute layer-local EP gradients independently.

        Each layer updates based on local energy contrast without
        cross-layer gradient accumulation.

        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            states_free: States from free phase.
            states_nudged: States from nudged phase.
            structure: Model structure.
        """
        io_free = self._get_layer_io(model, x, states_free, structure)
        io_nudged = self._get_layer_io(model, x, states_nudged, structure)

        # Create map from module id to IO data
        map_free: Dict[int, Dict[str, Any]] = {id(item["module"]): item for item in io_free}
        map_nudged: Dict[int, Dict[str, Any]] = {id(item["module"]): item for item in io_nudged}

        # Iterate structure to process layers
        for item in structure:
            if item["type"] != "layer":
                continue
            module = item["module"]

            if id(module) not in map_free or id(module) not in map_nudged:
                continue

            free_data = map_free[id(module)]
            nudged_data = map_nudged[id(module)]

            # --- Free Phase Local Gradient ---
            # E_local = 0.5 * || output - module(input) ||^2
            # Detach input/output to isolate this layer's gradient
            in_free = free_data["input"].detach()
            out_free = free_data["output"].detach()

            # --- Nudged Phase Local Gradient ---
            in_nudged = nudged_data["input"].detach()
            out_nudged = nudged_data["output"].detach()

            module_params = list(module.parameters())

            # Recompute forward pass for this layer to build graph
            with torch.enable_grad():
                # Free Phase Energy
                pred_free = module(in_free)
                E_free = 0.5 * F.mse_loss(pred_free, out_free, reduction="sum") / x.shape[0]

                # Nudged Phase Energy
                pred_nudged = module(in_nudged)
                E_nudged = 0.5 * F.mse_loss(pred_nudged, out_nudged, reduction="sum") / x.shape[0]

                # Combined Local Gradient
                loss = (E_nudged - E_free) / self.defaults["beta"]
                grads = torch.autograd.grad(
                    loss, module_params, retain_graph=False, allow_unused=True
                )

            # --- Update Gradients ---
            for p, g in zip(module_params, grads):
                if g is None:
                    continue

                # Check for NaN/Inf
                if torch.isnan(g).any() or torch.isinf(g).any():
                    raise RuntimeError(
                        f"LocalEPMuon produced NaN/Inf for layer {module}. "
                        f"Try reducing beta or learning rate."
                    )

                if p.grad is None:
                    p.grad = g.detach()
                else:
                    p.grad.add_(g.detach())


class NaturalEPMuon(SMEPOptimizer):
    """
    EP with Natural Gradient descent on the energy landscape.

    This optimizer uses the **Fisher Information Matrix** induced by the EP
    energy function to perform updates in the natural parameter space, rather
    than Euclidean space. This accounts for the geometry of the probability
    distribution over energy states.

    Key Features:
    - **Geometric Optimization:** Handles "sloppy" directions in parameter space.
    - **Fisher Approximation:** Uses empirical Fisher information from free-phase
      gradients.
    - **Whitening:** Applies whitening to the gradient before the Muon update.

    Args:
        fisher_approx (str): Approximation method for Fisher Information.
            Currently supports 'empirical'. Default: 'empirical'.

    Usage:
        optimizer = NaturalEPMuon(model.parameters(), model=model, mode='ep',
                                  fisher_approx='empirical')
        optimizer.step(target=y)
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        fisher_approx: str = "empirical",
        use_diagonal_fisher: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize NaturalEPMuon.

        Args:
            params: Iterable of parameters.
            fisher_approx: Fisher approximation method ('empirical').
            use_diagonal_fisher: Use diagonal approximation for Fisher matrix (faster).
            **kwargs: Additional arguments passed to SMEPOptimizer.
        """
        if fisher_approx not in ["empirical"]:
            raise ValueError(
                f"Unknown Fisher approximation: {fisher_approx}. " f"Supported: 'empirical'"
            )
        super().__init__(params, **kwargs)
        self.fisher_approx = fisher_approx
        self.use_diagonal_fisher = use_diagonal_fisher

    def _compute_update(
        self,
        p: nn.Parameter,
        g_flat: torch.Tensor,
        group: Dict[str, Any],
        state: StateDict,
        g_aug: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Compute natural gradient update with Fisher whitening.

        Args:
            p: Parameter tensor.
            g_flat: Flattened gradient.
            group: Parameter group dict.
            state: Optimizer state dict.
            g_aug: Augmented gradient.
            orig_shape: Original parameter shape.

        Returns:
            Whitened and orthogonalized gradient.
        """
        # Approximate Fisher from EP energy landscape
        fisher_block = self._compute_fisher_block(p, state, g_flat)

        if fisher_block is not None:
            damping = 1e-3
            if self.use_diagonal_fisher:
                # Diagonal whitening: g / (F + eps)
                F = fisher_block + damping
                whitened = g_flat / F.unsqueeze(0)
            else:
                # Whitening: solve(F + eps*I, g^T)^T -> g @ (F + eps*I)^-1
                # Use a larger epsilon for stability with low-rank empirical Fisher
                F = fisher_block + damping * torch.eye(
                    fisher_block.shape[0], device=fisher_block.device
                )
                try:
                    whitened = torch.linalg.solve(F, g_flat.T).T
                    if torch.isnan(whitened).any():
                        whitened = g_flat
                except RuntimeError:
                    # Fallback to raw gradient if solve fails
                    whitened = g_flat
        else:
            whitened = g_flat

        return self.newton_schulz(whitened, group["ns_steps"])

    def _compute_fisher_block(
        self, p: nn.Parameter, state: StateDict, g_flat: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute Fisher Information Matrix block for a parameter.

        Args:
            p: Parameter tensor.
            state: Optimizer state dict.
            g_flat: Flattened gradient.

        Returns:
            Fisher matrix block or None if not available.
        """
        # Retrieve stored grad_free
        grad_free = state.get("grad_free")
        if grad_free is None:
            return None

        # Flatten grad_free if needed
        if grad_free.shape != g_flat.shape:
            grad_free = grad_free.view(g_flat.shape)

        if self.use_diagonal_fisher:
            # Diagonal approximation: sum(g**2, dim=0)
            F = (grad_free**2).sum(dim=0)
        else:
            # Empirical Fisher approximation: F = g_free^T @ g_free
            # This gives (Cols, Cols).
            F = grad_free.T @ grad_free

        # Normalize by norm to keep scale reasonable
        F = F / (grad_free.norm() + 1e-6)
        return F  # type: ignore[no-any-return]

    def _apply_ep_gradients(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target: torch.Tensor,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
        structure: Structure,
    ) -> None:
        """
        Compute EP gradients and cache free-phase gradients for Fisher.

        Args:
            model: Neural network module.
            x: Input tensor.
            target: Target tensor.
            states_free: States from free phase.
            states_nudged: States from nudged phase.
            structure: Model structure.
        """
        # 1. Standard EP gradients
        super()._apply_ep_gradients(model, x, target, states_free, states_nudged, structure)

        # 2. Capture Free Phase Gradients for Fisher
        # Re-compute E_free
        E_free = self._compute_energy(model, x, states_free, structure, target_vec=None, beta=0.0)

        params_list = list(model.parameters())
        grads_free = torch.autograd.grad(E_free, params_list, retain_graph=False, allow_unused=True)

        for p, g_free in zip(params_list, grads_free):
            if g_free is not None:
                if p.ndim >= 2:
                    self.state[p]["grad_free"] = g_free.detach()
