from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

from .eqprop_base import EqPropModel
from .nebc_base import register_nebc
from .registry import register_model


@register_model("eq_align")
@register_nebc("eq_align")
class EquilibriumAlignment(EqPropModel):
    """
    Equilibrium Alignment (EqAlign) - Native Implementation.

    Combines Equilibrium Propagation's fixed-point dynamics with
    Feedback Alignment (FA) training signals.

    Architecture:
        Similar to LoopedMLP (Recurrent Fixed-Point Net).
        h = tanh(W_in @ x + W_rec @ h)

    Training (Custom train_step):
        Uses Direct Feedback Alignment (DFA) on the equilibrium state.
        Instead of BPTT or Contrastive updates, we backpropagate the error
        through a fixed random feedback matrix B directly to the hidden state.

        delta_h = B @ (y_pred - y)

    This avoids the weight transport problem (using W.T) and BPTT memory costs.
    """

    algorithm_name = "EquilibriumAlignment"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        use_spectral_norm: bool = True,
        learning_rate: float = 0.001,
        **kwargs,
    ):
        self.learning_rate = learning_rate
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=max_steps,
            use_spectral_norm=use_spectral_norm,
            **kwargs,
        )

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=30,
            use_spectral_norm=True,
            learning_rate=spec.default_lr,
        ).to(device)

    def _build_layers(self):
        # Forward weights
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_rec = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)

        if self.use_spectral_norm:
            self.W_in = spectral_norm(self.W_in)
            self.W_rec = spectral_norm(self.W_rec)
            self.W_out = spectral_norm(self.W_out)

        # Feedback weights (Fixed, Random)
        # B maps from Output -> Hidden
        # Shape: (Hidden, Output) effectively, but we use it as error @ B
        # error: (Batch, Out). B: (Out, Hidden). -> (Batch, Hidden).
        self.B_out = nn.Parameter(
            torch.randn(self.output_dim, self.hidden_dim) * 0.1, requires_grad=False
        )

    def _initialize_hidden_state(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.zeros(
            (batch_size, self.hidden_dim), device=x.device, dtype=x.dtype
        )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_in(x)

    def forward_step(
        self, h: torch.Tensor, x_transformed: torch.Tensor
    ) -> torch.Tensor:
        return torch.tanh(x_transformed + self.W_rec(h))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """
        Custom training step using Equilibrium Feedback Alignment.
        """
        # 1. Forward to Equilibrium
        # We use the internal forward() which handles loops
        # We need the final hidden state h_star for updates

        # Manually run forward to keep access to h_star easily without hooks
        with torch.no_grad():
            x_transformed = self._transform_input(x)
            h = self._initialize_hidden_state(x)
            for _ in range(self.max_steps):
                h = self.forward_step(h, x_transformed)
            h_star = h

            # Compute Output
            logits = self._output_projection(h_star)

            # Loss
            loss = nn.functional.cross_entropy(logits, y)

            # Accuracy
            acc = (logits.argmax(dim=1) == y).float().mean().item()

            # 2. Compute Error
            if y.dim() == 1:
                target = nn.functional.one_hot(y, self.output_dim).float()
            else:
                target = y

            # error = logits - target  # (Batch, Out) / Batch_Size if reduction needed?
            # CE Gradient is softmax(logits) - target.
            # logits above are raw scores.
            # wait, cross_entropy combines log_softmax + nll_loss.
            # Gradient of CE w.r.t logits is (softmax(logits) - target).
            probs = torch.softmax(logits, dim=1)
            delta_out = probs - target

        # 3. Backward (Feedback Alignment)
        # Propagate error to hidden state using B
        # delta_h = delta_out @ B_out
        # B_out shape: (Out, Hidden). delta_out: (Batch, Out).
        # We want (Batch, Hidden).
        delta_h = torch.mm(delta_out, self.B_out)  # (Batch, Hidden)

        # Apply derivative of nonlinearity (tanh) at h_star
        # h' = 1 - h^2
        delta_h = delta_h * (1 - h_star**2)

        # 4. Compute Gradients
        # dW_out = delta_out.T @ h_star
        # dW_rec = delta_h.T @ h_star
        # dW_in  = delta_h.T @ x

        batch_size = x.size(0)

        grad_W_out = torch.mm(delta_out.T, h_star) / batch_size
        grad_W_rec = torch.mm(delta_h.T, h_star) / batch_size
        grad_W_in = torch.mm(delta_h.T, x) / batch_size

        grad_b_out = delta_out.mean(0)
        grad_b_rec = delta_h.mean(0)  # Bias for recurrent part effectively

        # 5. Update Weights (Manual SGD)
        # Note: We must update the original parameters, not the spectral_norm-wrapped ones.

        def update_layer(layer, grad_w, grad_b=None):
            # Handle Spectral Norm wrapping
            if hasattr(layer, "parametrizations"):
                weight_param = layer.parametrizations.weight.original
            else:
                weight_param = layer.weight

            weight_param.data -= self.learning_rate * grad_w

            if layer.bias is not None and grad_b is not None:
                layer.bias.data -= self.learning_rate * grad_b

        update_layer(self.W_out, grad_W_out, grad_b_out)
        update_layer(
            self.W_rec, grad_W_rec, grad_b_rec
        )  # W_rec usually has no bias in this formulation
        # Update bias for both layers using the same gradient
        update_layer(self.W_in, grad_W_in, grad_b_rec)

        return {"loss": loss.item(), "accuracy": acc}
