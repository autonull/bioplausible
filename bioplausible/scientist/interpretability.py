"""
Interpretability Tools for Bioplausible Models.

Provides methods to visualize feature importance (saliency maps, integrated gradients)
and decision boundaries, aiding in the analysis of model behavior and failure modes.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class FeatureAttribution:
    """
    Computes feature attribution (saliency) maps for models.

    Supports both standard backprop models and Equilibrium Propagation models
    (via energy gradients if applicable).
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Initialize the FeatureAttribution tool.

        Args:
            model (nn.Module): The PyTorch model to interpret.
        """
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        # Detect device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def compute_saliency(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute standard Saliency Map (Vanilla Gradient).

        Calculates |dL / dx|, representing the sensitivity of the loss
        to input features.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Saliency map (same shape as input).
        """
        self.model.eval()
        x = x.clone().detach().to(self.device)
        x.requires_grad = True

        # Forward
        logits = self._forward_pass(x)

        # Loss
        loss = self.loss_fn(logits, y.to(self.device))

        # Backward
        self.model.zero_grad()
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        # Saliency = magnitude of gradient
        if x.grad is None:
            return torch.zeros_like(x)

        saliency = x.grad.abs()
        return saliency

    def compute_integrated_gradients(
        self, x: torch.Tensor, y: torch.Tensor, steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients (approximate).

        IG(x) = (x - x_baseline) * Integral(gradients)
        Provides a more robust attribution than simple saliency by integrating
        gradients along a path from a baseline (zero) to the input.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target labels.
            steps (int): Number of integration steps.

        Returns:
            torch.Tensor: Integrated Gradients map.
        """
        self.model.eval()
        x = x.to(self.device)
        baseline = torch.zeros_like(x)

        # Accumulate gradients
        grads_accum = torch.zeros_like(x)

        # Generate path inputs
        # We iterate manually to handle memory efficiently
        for i in range(steps):
            alpha = i / steps
            x_step = baseline + alpha * (x - baseline)
            x_step = x_step.clone().detach().requires_grad_(True)

            logits = self._forward_pass(x_step)
            loss = self.loss_fn(logits, y.to(self.device))

            self.model.zero_grad()
            if x_step.grad is not None:
                x_step.grad.zero_()
            loss.backward()

            if x_step.grad is not None:
                grads_accum += x_step.grad

        # Average and multiply
        avg_grad = grads_accum / steps
        ig = (x - baseline) * avg_grad
        return ig

    def _forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """
        Helper to handle model quirks (flattening etc).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Model output logits.
        """
        # Improved flattening logic
        # Check if model has explicit input_dim
        input_dim = getattr(self.model, "input_dim", None)

        should_flatten = False
        if x.dim() > 2:
            if input_dim is not None:
                # If total elements match input_dim, flatten
                if x[0].numel() == input_dim:
                    should_flatten = True
            elif "Conv" not in type(self.model).__name__:
                # Fallback heuristic
                should_flatten = True

        if should_flatten:
            x_flat = x.view(x.size(0), -1)
            return self.model(x_flat)

        return self.model(x)


def visualize_decision_boundary(
    model: nn.Module,
    data_range: Tuple[float, float] = (-1.5, 1.5),
    steps: int = 100,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D decision boundary grid for a 2-input model (or PCA reduced).

    Args:
        model (nn.Module): The model to visualize.
        data_range (Tuple[float, float]): Range of the grid (min, max).
        steps (int): Resolution of the grid.
        device (str): Device to run inference on.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - xx: Grid X coordinates.
            - yy: Grid Y coordinates.
            - preds: Predicted class labels shaped like the grid.
    """
    model.eval()
    x_range = np.linspace(data_range[0], data_range[1], steps)
    y_range = np.linspace(data_range[0], data_range[1], steps)
    xx, yy = np.meshgrid(x_range, y_range)

    # Flatten grid for batch inference
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid).to(device)

    with torch.no_grad():
        logits = model(grid_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()

    return xx, yy, preds.reshape(xx.shape)
