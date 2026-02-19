"""
Energy function computation for Equilibrium Propagation.

This module defines the energy function used in EP:
    E = E_internal + E_external

where:
    E_internal = 0.5 * Σ ||s_i - f_i(s_{i-1})||²  (state consistency)
    E_external = β * L(s_last, y)                 (task loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional


class EnergyFunction:
    """
    Computes the EP energy function.
    
    The energy measures how well the network states satisfy:
    1. Internal consistency (each layer matches its prediction)
    2. External constraint (output matches target, when nudged)
    """
    
    def __init__(
        self,
        loss_type: str = "mse",  # MSE for stable EP energy computation
        softmax_temperature: float = 1.0,
    ):
        self.loss_type = loss_type
        self.softmax_temperature = softmax_temperature
    
    def __call__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        states: List[torch.Tensor],
        structure: List[Dict[str, Any]],
        target_vec: Optional[torch.Tensor] = None,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute total energy: E = E_int + E_ext.
        
        Args:
            model: Neural network module.
            x: Input tensor.
            states: List of layer states.
            structure: Model structure.
            target_vec: Target for nudge term (None for free phase).
            beta: Nudging strength.
        
        Returns:
            Scalar energy tensor.
        """
        batch_size = x.shape[0]
        if batch_size == 0:
            raise ValueError(f"Batch size cannot be zero, got input shape {x.shape}")
        
        use_classification = self.loss_type == "cross_entropy"
        
        # Accumulate energy in float32 for stability
        E = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        prev = x
        state_idx = 0
        
        # Find all modules that produce a state (layer or attention)
        state_producing_modules = [
            item for item in structure
            if item["type"] in ("layer", "attention")
        ]
        num_states = len(state_producing_modules)
        
        if len(states) != num_states:
            raise ValueError(
                f"Number of states ({len(states)}) does not match number of state-producing layers ({num_states}). "
                f"Structure has {len(structure)} items."
            )

        for item in structure:
            item_type = item["type"]
            module = item["module"]

            if item_type == "layer":
                if state_idx >= len(states):
                    break

                state = states[state_idx]
                is_last_state = (state_idx == num_states - 1)

                # Compute prediction from previous layer
                h = module(prev)

                if use_classification and is_last_state:
                    # KL divergence for classification output
                    # Cast h to float32 for stable energy calculation
                    E = E + self._kl_energy(state.float(), h.float(), batch_size)
                else:
                    # MSE for hidden layers and regression
                    self._validate_shapes(h, state, f"Layer {state_idx} ({type(module).__name__})")
                    # Compute MSE in float32
                    E = E + 0.5 * self._safe_mse(h.float(), state.float()) / batch_size

                # The input to the next layer is the current state (relaxed variable)
                # Ensure dtype matches input x to prevent type mismatch with weights
                # (especially when states were settled in AMP but contrast runs in FP32)
                prev = state.to(x.dtype)
                state_idx += 1

            elif item_type == "norm":
                prev = module(prev)

            elif item_type == "pool":
                prev = module(prev)

            elif item_type == "flatten":
                prev = module(prev)

            elif item_type == "dropout":
                # Skip dropout during energy computation - it breaks settling convergence
                # because stochastic masking prevents finding a fixed point
                # Dropout should only be used during standard forward passes, not EP settling
                pass

            elif item_type == "attention":
                if state_idx >= len(states):
                    break
                
                state = states[state_idx]
                
                if isinstance(module, nn.MultiheadAttention):
                    try:
                        # For MHA, we usually use the first output (attn_output)
                        # Assuming self-attention: query=key=value=prev
                        # Note: MHA expects (L, N, E) or (N, L, E) depending on batch_first.
                        # We assume inputs are compatible.
                        h = module(prev, prev, prev, need_weights=False)[0]
                    except (RuntimeError, AssertionError):
                        # Fallback if shapes don't match or other error
                        # Use state as placeholder to continue flow?
                        # Or raise error? raising is better for robustness.
                        # But for now let's assume it works or use prev.
                        # Actually, if we fail to compute h, energy computation is invalid.
                        raise RuntimeError("Failed to compute MultiheadAttention output during energy calculation.")
                else:
                    h = module(prev)
                
                self._validate_shapes(h, state, f"Attention Layer {state_idx}")
                # Compute MSE in float32
                E = E + 0.5 * self._safe_mse(h.float(), state.float()) / batch_size
                prev = state.to(x.dtype)
                state_idx += 1
            
            elif item_type == "act":
                prev = module(prev)

        # Nudge term
        if target_vec is not None and beta > 0:
            # Nudge term in float32
            E = E + self._nudge_term(prev.float(), target_vec, beta, batch_size)
        
        # Stability check
        if torch.isnan(E) or torch.isinf(E):
            raise RuntimeError(
                f"Energy computation produced NaN/Inf. "
                f"Input: {x.shape}, States: {len(states)}, "
                f"Target: {target_vec.shape if target_vec is not None else None}"
            )
        
        return E
    
    def _validate_shapes(self, h: torch.Tensor, state: torch.Tensor, context: str) -> None:
        """Ensure shapes match between prediction and state."""
        if h.shape != state.shape:
             raise ValueError(
                f"Shape mismatch at {context}: Prediction {h.shape} vs State {state.shape}. "
                "Check model architecture and layer types."
             )

    def _safe_mse(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
         """Compute MSE safely."""
         return F.mse_loss(input, target, reduction="sum")

    def _kl_energy(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute KL divergence energy for classification output.
        
        Uses softmax-aware formulation:
            E = D_KL(softmax(state) || softmax(prediction))
        """
        eps = 1e-8
        self._validate_shapes(prediction, state, "KL Divergence")

        # Assume class dimension is 1 (standard for PyTorch: N, C, ...)
        state_softmax = F.softmax(state / self.softmax_temperature, dim=1)
        h_softmax = F.softmax(prediction / self.softmax_temperature, dim=1)
        
        kl_div = F.kl_div(
            torch.log(state_softmax + eps), h_softmax, reduction="sum"
        )
        return kl_div / batch_size
    
    def _nudge_term(
        self,
        output: torch.Tensor,
        target_vec: torch.Tensor,
        beta: float,
        batch_size: int
    ) -> torch.Tensor:
        """
        Compute external nudge term.
        
        For classification: CrossEntropy with label smoothing
        For regression: MSE
        """
        if self.loss_type == "cross_entropy":
            # target_vec contains class indices
            if output.shape[0] != target_vec.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: Output {output.shape[0]}, Target {target_vec.shape[0]}"
                )

            return beta * F.cross_entropy(
                output, target_vec, reduction="sum", label_smoothing=0.1
            ) / batch_size
        else:
            # MSE for regression
            # target_vec might need reshape to match output?
            if output.shape != target_vec.shape:
                # Try squeezing target_vec if it has extra dim 1
                 if output.shape == target_vec.squeeze().shape:
                     target_vec = target_vec.squeeze()
                 elif output.squeeze().shape == target_vec.shape:
                     output = output.squeeze()

            if output.shape != target_vec.shape:
                 raise ValueError(
                    f"Shape mismatch in nudge term: Output {output.shape}, Target {target_vec.shape}"
                )

            return beta * F.mse_loss(output, target_vec, reduction="sum") / batch_size
