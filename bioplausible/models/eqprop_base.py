from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from .nebc_base import NEBCBase
from .triton_kernel import TritonEqPropOps


class EquilibriumFunction(autograd.Function):
    """
    Implicit differentiation for Equilibrium Propagation models.

    Implements O(1) memory backpropagation using the equilibrium property:
    dL/dtheta = dL/dh * dh/dtheta
    where dh/dtheta = (I - J)^-1 * df/dtheta

    The backward pass solves for the adjoint state delta:
    delta = (I - J^T)^-1 * dL/dh
    via fixed-point iteration:
    delta_{t+1} = J^T * delta_t + dL/dh
    """

    @staticmethod
    def forward(
        ctx: Any,
        model: nn.Module,
        x_transformed: torch.Tensor,
        h_init: torch.Tensor,
        *params: torch.Tensor,
    ) -> torch.Tensor:
        ctx.model = model

        # Optimization: Freeze Spectral Norm during loop
        should_freeze_sn = getattr(model, "use_spectral_norm", False) and model.training
        remaining_steps = model.max_steps

        # 1. Find fixed point (no gradient tracking needed for the loop itself)
        # We assume h_init is close to the fixed point if we are continuing from previous state,
        # or we iterate enough steps to converge.
        with torch.no_grad():
            h = h_init

            if should_freeze_sn and remaining_steps > 0:
                # Warmup step
                h = model.forward_step(h, x_transformed)
                remaining_steps -= 1
                model.eval()

            try:
                for _ in range(remaining_steps):
                    h = model.forward_step(h, x_transformed)
            finally:
                if should_freeze_sn:
                    model.train()

        # Save tensors for backward
        # Note: We must save params to ensure autograd knows they participate in the graph
        ctx.save_for_backward(h, x_transformed, *params)
        return h

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], ...]:
        h_star, x_transformed, *params = ctx.saved_tensors
        model = ctx.model

        # Capture training state
        was_training = model.training
        # Set to eval to prevent buffer updates (e.g. Spectral Norm) during backward fixed-point iteration
        # This is critical because Spectral Norm updates 'u' and 'v' buffers in .train() mode,
        # which would cause in-place modification errors or incorrect gradients during the backward loop.
        model.eval()

        try:
            # 2. Compute adjoint state (delta) via fixed-point iteration
            # Initial guess for delta is dL/dh (grad_output)
            # OPTIMIZATION: Remove unnecessary clone (grad_output is read-only here)
            delta = grad_output

            # Use detached X for the VJP loop to avoid any graph entanglement with input gradients yet
            x_transformed_detached = x_transformed.detach()

            # Check if model has _forward_step_impl (uncompiled) to avoid torch.compile overhead in loop
            forward_fn = getattr(model, "_forward_step_impl", model.forward_step)

            # Iterate to equilibrium for the backward pass (solving for delta)
            # delta_{t+1} = (df/dh)^T * delta_t + grad_output
            for _ in range(model.max_steps):
                with torch.enable_grad():
                    # Create a new leaf for h_star at each step for local VJP calc
                    h_star_loop = h_star.detach().requires_grad_(True)

                    # Compute f(h, x)
                    f_h = forward_fn(h_star_loop, x_transformed_detached)

                    # VJP: v = (df/dh)^T @ delta
                    # retain_graph=False ensures we free the f_h graph immediately.
                    # We detach delta because for the purpose of the VJP, delta is a constant vector.
                    vjp = autograd.grad(
                        f_h,
                        h_star_loop,
                        grad_outputs=delta.detach(),
                        retain_graph=False,
                        create_graph=False,
                    )[0]

                    # Update delta
                    # Crucial: detach delta to prevent graph growth during the fixed-point iteration
                    # The VJP loop is purely for finding the value of the adjoint state.
                    delta = (vjp + grad_output).detach()

            # 3. Compute gradients for parameters and input using the converged delta
            delta = delta.detach()

            with torch.enable_grad():
                h_star_detached = h_star.detach()

                # A. Compute gradients for parameters
                # dL/dtheta = (df/dtheta)^T @ delta

                # CRITICAL: Detach x_transformed here.
                # If we don't detach, autograd will trace d(f_h)/d(x) * d(x)/d(theta)
                # effectively double-counting the gradient for params that affect x_transformed.
                x_detached = x_transformed.detach()

                params_with_grad = [p for p in params if p.requires_grad]
                grads_params_list = [None] * len(params)

                if params_with_grad:
                    # Re-run forward step to build graph from params to f_h
                    # Use uncompiled function here too for consistency.
                    f_h_params = forward_fn(h_star_detached, x_detached)

                    computed_grads = autograd.grad(
                        f_h_params,
                        params,
                        grad_outputs=delta,
                        allow_unused=True,
                        retain_graph=False,
                    )
                    grads_params_list = list(computed_grads)

                # B. Compute gradients for input (x_transformed)
                # dL/dx = (df/dx)^T @ delta
                grad_x = None
                if x_transformed.requires_grad:
                    # Use attached x_transformed to get gradients w.r.t input
                    f_h_x = model.forward_step(h_star_detached, x_transformed)
                    grad_x = autograd.grad(
                        f_h_x, x_transformed, grad_outputs=delta, retain_graph=False
                    )[0]

        finally:
            # Restore original training state
            model.train(was_training)

        # Return gradients corresponding to inputs of forward:
        # ctx, model, x_transformed, h_init, *params
        # model and h_init don't get gradients
        return (None, grad_x, None, *grads_params_list)


class EqPropModel(NEBCBase):
    """
    Abstract base class for Equilibrium Propagation models.
    """

    def __init__(self, max_steps: int = 30, gradient_method: str = "bptt", **kwargs):
        """
        Args:
            max_steps: Number of equilibrium steps
            gradient_method: 'bptt', 'equilibrium' (implicit diff), or 'contrastive' (Hebbian)
        """
        input_dim = kwargs.get("input_dim", 0)
        hidden_dim = kwargs.get("hidden_dim", 0)
        output_dim = kwargs.get("output_dim", 0)

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=kwargs.get("use_spectral_norm", True),
            lipschitz_mode=kwargs.get("lipschitz_mode", "power_iteration"),
        )
        self.max_steps = max_steps
        self.gradient_method = gradient_method

        # Contrastive Hebbian specific params
        self.beta = kwargs.get("beta", 0.1)
        self.hebbian_lr = kwargs.get("learning_rate", 0.001)
        self.internal_optimizer = None

    @abstractmethod
    def _build_layers(self):
        """Build layers. Required by NEBCBase, implemented by subclasses."""
        pass

    @abstractmethod
    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        """Single equilibrium iteration step."""
        pass

    @abstractmethod
    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize the hidden state tensor based on input x."""
        pass

    @abstractmethod
    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """Transform raw input x into the form used in the loop."""
        pass

    @abstractmethod
    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden state to output."""
        pass

    def get_hebbian_pairs(
        self, h: torch.Tensor, x: torch.Tensor
    ) -> List[Tuple[nn.Module, torch.Tensor, torch.Tensor]]:
        """
        Return list of (layer_module, input, output_target) for Hebbian updates.

        This defines the topology for contrastive learning.
        For a layer y = f(W, u), we typically return (layer, u, y).
        The generic update will compute gradients of (layer(u) * y).sum().

        Args:
            h: Hidden state at equilibrium
            x: Raw input

        Returns:
            List of tuples: (layer, input_to_layer, target_output_of_layer)
        """
        raise NotImplementedError(
            "Subclasses must implement get_hebbian_pairs for generic contrastive learning."
        )

    def contrastive_update(
        self,
        h_free: torch.Tensor,
        h_nudged: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        """
        Perform generic contrastive Hebbian update using 'get_hebbian_pairs'.

        Implements: Delta W ~ grad(Layer(x) @ y_nudged) - grad(Layer(x) @ y_free)
        """
        batch_size = x.shape[0]
        scale = 1.0 / (self.beta * batch_size)

        # 1. Get pairs for Free and Nudged states
        # Note: We recompute 'transform_input' or similar if needed, but 'get_hebbian_pairs'
        # usually takes raw x and h.
        pairs_free = self.get_hebbian_pairs(h_free, x)
        pairs_nudged = self.get_hebbian_pairs(h_nudged, x)

        # 2. Aggregate Proxy Losses and Compute Gradients Once
        # Optimization: Sum proxy losses to reduce autograd overhead
        total_loss_free = 0.0
        total_loss_nudged = 0.0

        for (layer, inp_f, tgt_f), (_, inp_n, tgt_n) in zip(pairs_free, pairs_nudged):
            # Free Phase
            # Detach inputs to prevent backprop through layers (preserve local learning)
            out_f = layer(inp_f.detach())
            total_loss_free = total_loss_free + torch.sum(out_f * tgt_f.detach())

            # Nudged Phase
            out_n = layer(inp_n.detach())
            total_loss_nudged = total_loss_nudged + torch.sum(out_n * tgt_n.detach())

        # Compute gradients for all parameters at once
        params = list(self.parameters())
        grads_f = autograd.grad(
            total_loss_free, params, retain_graph=True, allow_unused=True
        )
        grads_n = autograd.grad(
            total_loss_nudged, params, retain_graph=True, allow_unused=True
        )

        # Apply update
        for param, gf, gn in zip(params, grads_f, grads_n):
            if param.requires_grad:
                # Delta W ~ (Nudged - Free)
                g_update = 0.0
                if gn is not None:
                    g_update += gn
                if gf is not None:
                    g_update -= gf

                if isinstance(g_update, float) and g_update == 0.0:
                    continue

                grad_term = scale * g_update

                if param.grad is None:
                    param.grad = grad_term
                else:
                    param.grad.add_(grad_term)

        # 3. Output Layer (Standard Backprop on Nudged or Free?)
        # Standard EqProp: W_out update is just gradient of Cost function at Free phase.
        logits = self._output_projection(h_free)
        loss = F.cross_entropy(logits, y)

        # Update W_out (supervised component).
        # We use autograd.grad on loss, but only apply it to parameters that haven't been updated
        # by the Hebbian phase (i.e., parameters with .grad is None).
        # This assumes W_out is not part of the Hebbian dynamics.

        grads_loss = autograd.grad(loss, self.parameters(), allow_unused=True)
        for param, g in zip(self.parameters(), grads_loss):
            if g is not None:
                if param.grad is None:
                    # This param wasn't updated by Hebbian loop -> Must be W_out or similar
                    param.grad = g
                else:
                    # Already has Hebbian grad -> Do not add Loss grad (unless hybrid?)
                    # Pure EqProp: Internal weights only update via Hebbian.
                    pass

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Perform a single training step.
        If gradient_method is 'contrastive', this runs the EqProp loop manually.
        Otherwise, it returns None to let SupervisedTrainer handle BPTT/Implicit.
        """
        if self.gradient_method != "contrastive":
            return None  # Delegate to standard trainer

        # Initialize optimizer on first call
        if self.internal_optimizer is None:
            self.internal_optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hebbian_lr
            )

        self.internal_optimizer.zero_grad()

        # 1. Free Phase
        with torch.no_grad():
            h_free = self._initialize_hidden_state(x)
            x_transformed = self._transform_input(x)

            for _ in range(self.max_steps):
                h_free = self.forward_step(h_free, x_transformed)

            logits_free = self._output_projection(h_free)

        # 2. Nudged Phase
        # We need to compute gradients of the loss w.r.t h to nudge
        # But for 'contrastive', we typically nudge via a top-down drive or explicit gradient injection

        # Enable grad just for the nudge calculation
        h_nudged = h_free.clone().detach().requires_grad_(True)

        # Run one step to connect h to output (if needed) or just project
        # Ideally we settle in the nudged phase with a constant nudge.
        # Nudge term: - beta * dL/dh

        # Calculate dL/dh at equilibrium
        logits_nudge_init = self._output_projection(h_nudged)
        loss = F.cross_entropy(logits_nudge_init, y)
        grads_h = autograd.grad(loss, h_nudged)[0]

        # Stability Check 1: Gradients
        if torch.isnan(grads_h).any() or torch.isinf(grads_h).any():
            print("Warning: EqProp divergence detected (NaN gradients). Skipping step.")
            return {"loss": 100.0, "accuracy": 0.1}

        # Nudged dynamics: h <- forward_step(h) - beta * dL/dh
        # Note: In continuous time, dot_h = -h + sigma(...) - beta * dL/dh
        # In discrete step: h_new = forward_step(h) - beta * dL/dh

        # We perform fixed point iteration with the nudge
        # Nudge should be constant if dL/dh is approx constant locally, or updated?
        # Standard EqProp keeps the nudge target fixed (y) but dL/dh changes as h changes.

        with torch.no_grad():
            h_nudged = h_free.clone()

            # Simple implementation: Apply constant nudge derived from free phase error?
            # Or recompute nudge each step?
            # Scellier 2017: weakly clamp output units.
            # Here output is a projection. We inject gradient.

            # We'll use a constant nudge vector derived from free phase for stability/speed
            nudge_vec = -self.beta * grads_h

            for _ in range(
                self.max_steps // 2
            ):  # Typically fewer steps for nudged phase
                # h = f(h) + nudge
                h_next = self.forward_step(h_nudged, x_transformed)
                h_nudged = h_next + nudge_vec

            logits_nudged = self._output_projection(h_nudged)

        # 3. Weight Update
        self.contrastive_update(h_free, h_nudged, x, y)

        self.internal_optimizer.step()

        # Compute metrics
        with torch.no_grad():
            if torch.isnan(logits_free).any():
                print("Warning: Model collapse (NaN logits).")
                acc = 0.1
                loss_val = 100.0
            else:
                acc = (logits_free.argmax(dim=1) == y).float().mean().item()
                loss_val = F.cross_entropy(logits_free, y).item()

        return {"loss": loss_val, "accuracy": acc}

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, List[torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, Any]],
    ]:
        """
        Forward pass: iterate to equilibrium.

        Args:
            x: Input tensor
            steps: Override number of iteration steps
            return_trajectory: If True, return all hidden states
            return_dynamics: If True, return detailed convergence metrics

        Returns:
            Output logits
            (optionally) trajectory of hidden states or dynamics dict
        """
        steps = steps or self.max_steps

        # Initialize
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        if (
            return_trajectory
            or return_dynamics
            or self.gradient_method in ["bptt", "contrastive"]
        ):
            # Standard unrolling (BPTT, Analysis, or Contrastive Inference)
            # OPTIMIZATION: Preallocate trajectory buffer
            if return_trajectory:
                trajectory = [None] * (steps + 1)
                trajectory[0] = h
            else:
                trajectory = None
            deltas = [] if return_dynamics else None

            # Optimization: Freeze Spectral Norm during loop to prevent graph breaks
            should_freeze_sn = (
                getattr(self, "use_spectral_norm", False) and self.training
            )
            remaining_steps = steps
            current_steps = 1  # Start at 1 because index 0 is initial state

            if should_freeze_sn and remaining_steps > 0:
                # Warmup step (update SN stats)
                h_new = self.forward_step(h, x_transformed)
                if return_dynamics:
                    # OPTIMIZATION: Use torch.dist for consistency with main loop (max norm)
                    deltas.append(torch.dist(h_new, h, p=float('inf')).item())
                h = h_new
                if return_trajectory:
                    trajectory[current_steps] = h
                    current_steps += 1
                remaining_steps -= 1
                # Switch to eval for the rest of the loop
                self.eval()

            try:
                for step_idx in range(remaining_steps):
                    h_new = self.forward_step(h, x_transformed)

                    if return_dynamics:
                        # OPTIMIZATION: Use torch.dist to avoid intermediate allocations
                        delta = torch.dist(h_new, h, p=float('inf')).item()
                        deltas.append(delta)

                    if step_idx > 5:
                        convergence_threshold = 1e-4 if step_idx > 10 else 2e-4
                        # OPTIMIZATION: Use torch.dist
                        if torch.dist(h_new, h, p=float('inf')).item() < convergence_threshold:
                            h = h_new
                            if return_trajectory:
                                trajectory[current_steps] = h
                                # Fill remaining slots with same value or truncate?
                                # Usually trajectory is expected entirely.
                                # But preallocation size was constant.
                                # If we break early, we should slice the result?
                                # Original behavior was append, so len < steps+1.
                                # So we should slice trajectory at end.
                            current_steps += 1
                            break

                    h = h_new
                    if return_trajectory:
                        trajectory[current_steps] = h
                        current_steps += 1
            finally:
                if should_freeze_sn:
                    self.train()

            out = self._output_projection(h)

            if return_dynamics:
                return out, {
                    "trajectory": trajectory[:current_steps] if return_trajectory else None,
                    "deltas": deltas,
                    "final_delta": deltas[-1] if deltas else 0.0,
                }

            if return_trajectory:
                return out, trajectory[:current_steps]
            return out

        elif self.gradient_method == "equilibrium":
            # O(1) memory implicit differentiation
            # We must pass params to apply so they are captured by ctx for backward
            # Note: We use list(self.parameters()) to get all parameters including weight_orig
            params = list(self.parameters())
            h_star = EquilibriumFunction.apply(self, x_transformed, h, *params)
            out = self._output_projection(h_star)
            return out

        else:
            raise ValueError(f"Unknown gradient_method: {self.gradient_method}")

    def inject_noise_and_relax(
        self,
        x: torch.Tensor,
        noise_level: float = 1.0,
        injection_step: int = 15,
        total_steps: int = 30,
    ) -> Dict[str, float]:
        """Demonstrate self-healing: inject noise and measure damping."""
        h = self._initialize_hidden_state(x)
        x_transformed = self._transform_input(x)

        # Run to injection point
        for _ in range(injection_step):
            h = self.forward_step(h, x_transformed)

        # Inject noise
        h_clean = h.clone()
        h_noisy = h + torch.randn_like(h) * noise_level

        initial_noise_norm = (h_noisy - h_clean).norm().item() / h.numel() ** 0.5

        # Run remaining steps
        steps_remaining = total_steps - injection_step
        for _ in range(steps_remaining):
            h_noisy = self.forward_step(h_noisy, x_transformed)
            h_clean = self.forward_step(h_clean, x_transformed)

        final_noise_norm = (h_noisy - h_clean).norm().item() / h.numel() ** 0.5

        ratio = (
            final_noise_norm / initial_noise_norm if initial_noise_norm > 1e-9 else 0.0
        )

        return {
            "initial_noise": initial_noise_norm,
            "final_noise": final_noise_norm,
            "damping_ratio": ratio,
            "damping_percent": (1 - ratio) * 100,
        }
