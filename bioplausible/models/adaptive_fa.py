from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .nebc_base import NEBCBase, register_nebc


@register_nebc("adaptive_fa")
class AdaptiveFA(NEBCBase):
    """
    Adaptive Feedback Alignment (AFA) as a native NEBC model.

    Ported from `algorithms/ada_fa.py` to be compatible with standard EqPropTrainer
    and model interfaces.
    """

    algorithm_name = "AdaptiveFA"

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        activation: str = "relu",
        learning_rate: float = 0.001,
        feedback_lr_scale: float = 0.001,
        **kwargs,
    ):
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.feedback_lr_scale = feedback_lr_scale

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            **kwargs,
        )

        # Standard optimizers (if used via train_step)
        # Note: NEBCBase doesn't store optimizers. We'll handle updates manually
        # in train_step or assume external optimizer handles standard params.
        # But AFA has custom update for Feedback weights.

    def _build_layers(self):
        self.layers = nn.ModuleList()
        self.feedback_weights = nn.ParameterList()

        # Dimensions: [In, Hidden, ..., Hidden, Out]
        dims = (
            [self.input_dim]
            + [self.hidden_dim] * (self.num_layers - 1)
            + [self.output_dim]
        )

        for i in range(len(dims) - 1):
            # Forward Layer
            layer = nn.Linear(dims[i], dims[i + 1])
            self.layers.append(layer)

            # Feedback Weight (B)
            # B maps from layer i+1 (Out) to layer i (In)
            # Standard backprop uses W.T (In, Out) -> (Out, In) effectively in computation
            # Here we store B explicitly.
            # Shape matching: error @ B. error: [Batch, Out]. B: [Out, In]. -> [Batch, In].
            B = torch.randn(dims[i + 1], dims[i]) * 0.1
            self.feedback_weights.append(nn.Parameter(B, requires_grad=True))

        if self.activation_name == "relu":
            self.act = nn.ReLU()
        elif self.activation_name == "tanh":
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.act(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Custom training step for AFA."""
        # 1. Forward
        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Activation for hidden layers only
            if i < len(self.layers) - 1:
                h = self.act(h)
            activations.append(h)

        output = activations[-1]
        loss = nn.functional.cross_entropy(output, y)

        # 2. Backward (Manual)
        if y.dim() == 1:
            target = nn.functional.one_hot(y, self.output_dim).float()
        else:
            target = y

        error = output - target

        # We manually update weights, so we don't need autograd for them
        # BUT standard PyTorch training loop might want to call optimizer.step().
        # If we use train_step, EqPropTrainer delegates everything to us.

        with torch.no_grad():
            for i in reversed(range(len(self.layers))):
                # activations[i] is input to layer i
                # activations[i+1] is output of layer i
                h_prev = activations[i]

                # Compute gradient for hidden state h_curr (activations[i+1])
                # For output layer, it's just error
                if i == len(self.layers) - 1:
                    grad_h = error
                else:
                    # Propagate error from layer above (i+1) down to layer i
                    # using FEEDBACK weights of layer i+1
                    # Accessing feedback_weights[i+1]
                    B = self.feedback_weights[i + 1]
                    grad_h = torch.mm(error, B)

                    # Apply activation derivative of layer i output
                    h_curr = activations[i + 1]
                    if self.activation_name == "relu":
                        grad_h = grad_h * (h_curr > 0).float()
                    elif self.activation_name == "tanh":
                        grad_h = grad_h * (1 - h_curr**2)

                # Compute gradient for weights W_i
                # dL/dW_i = grad_h.T @ h_prev
                # grad_h: [Batch, Out_i]. h_prev: [Batch, In_i].
                # result: [Out_i, In_i]. Matches W_i.
                grad_W = torch.mm(grad_h.T, h_prev) / x.size(0)
                grad_b = grad_h.mean(0)

                # Apply update (SGD/Adam would be better, but simple update here)
                self.layers[i].weight.data -= self.learning_rate * grad_W
                if self.layers[i].bias is not None:
                    self.layers[i].bias.data -= self.learning_rate * grad_b

                # Update Feedback weights (B) to align with W
                # Match logic from reference implementation.
                # Update B[i+1] to align with W[i+1].
                if i < len(self.layers) - 1:
                    # Access layer i+1
                    W_next = self.layers[i + 1].weight.data
                    B_next = self.feedback_weights[i + 1].data

                    # Diff
                    diff = W_next - B_next
                    # Update
                    self.feedback_weights[i + 1].data += (
                        self.learning_rate * self.feedback_lr_scale * diff
                    )

                # Prepare error for next iteration (which is layer below)
                error = grad_h

        pred = output.argmax(dim=1)
        acc = (pred == y).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}
