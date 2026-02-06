"""
Interpretability Tools for Bioplausible Models.

Provides methods to visualize feature importance and decision boundaries.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List

class FeatureAttribution:
    """
    Computes feature attribution (saliency) maps for models.
    Supports both standard backprop models and Equilibrium Propagation models
    (via energy gradients if applicable).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device if list(model.parameters()) else "cpu"

    def compute_saliency(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute standard Saliency Map (Vanilla Gradient).
        |dL / dx|
        """
        self.model.eval()
        x = x.clone().detach().to(self.device)
        x.requires_grad = True

        # Forward
        logits = self._forward_pass(x)

        # Loss
        loss = nn.CrossEntropyLoss()(logits, y.to(self.device))

        # Backward
        self.model.zero_grad()
        loss.backward()

        # Saliency = magnitude of gradient
        saliency = x.grad.abs()

        # If image (C, H, W), take max over channels to get (H, W) or keep channels?
        # Usually for viz we want (H, W). Returns raw tensor for now.
        return saliency

    def compute_integrated_gradients(self, x: torch.Tensor, y: torch.Tensor, steps: int = 50) -> torch.Tensor:
        """
        Compute Integrated Gradients (approximate).
        IG(x) = (x - x_baseline) * Integral(gradients)
        """
        self.model.eval()
        x = x.to(self.device)
        baseline = torch.zeros_like(x)

        # Generate path inputs
        alphas = torch.linspace(0, 1, steps).to(self.device)
        # Shape: (Steps, C, H, W) or (Steps, Dim)

        # Accumulate gradients
        grads_accum = torch.zeros_like(x)

        for alpha in alphas:
            x_step = baseline + alpha * (x - baseline)
            x_step = x_step.clone().detach().requires_grad_(True)

            logits = self._forward_pass(x_step)
            loss = nn.CrossEntropyLoss()(logits, y.to(self.device))

            self.model.zero_grad()
            loss.backward()

            if x_step.grad is not None:
                grads_accum += x_step.grad

        # Average and multiply
        avg_grad = grads_accum / steps
        ig = (x - baseline) * avg_grad
        return ig

    def _forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to handle model quirks (flattening etc)."""
        # We need to replicate what the trainer does for input shaping
        # But here 'x' is already shaped as expected by the caller (likely batch)
        # If model expects flattened, we must flatten.

        # Heuristic: Check model input dimension vs x
        # If x is image (B, C, H, W) and model is MLP (B, D), flatten.
        if x.dim() > 2 and "Conv" not in type(self.model).__name__:
             x_flat = x.view(x.size(0), -1)
             return self.model(x_flat)
        return self.model(x)

def visualize_decision_boundary(model, data_range=(-1.5, 1.5), steps=100, device="cpu"):
    """
    Generate a 2D decision boundary grid for a 2-input model (or PCA reduced).
    Returns (xx, yy, predictions).
    """
    model.eval()
    x = np.linspace(data_range[0], data_range[1], steps)
    y = np.linspace(data_range[0], data_range[1], steps)
    xx, yy = np.meshgrid(x, y)

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.FloatTensor(grid).to(device)

    with torch.no_grad():
        logits = model(grid_tensor)
        preds = logits.argmax(1).cpu().numpy()

    return xx, yy, preds.reshape(xx.shape)
