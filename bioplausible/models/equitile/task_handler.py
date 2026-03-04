from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class TaskHandler:
    """Handles task-specific loss, gradient, and metric computations."""

    def __init__(
        self,
        task_type: Literal["classification", "regression", "binary", "multilabel"],
        output_dim: int,
    ) -> None:
        self.task_type = task_type
        self.output_dim = output_dim

    def compute_loss(self, logits: Tensor, y: Tensor) -> Tensor:
        """Compute task-specific loss."""
        if self.task_type == "regression":
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            loss = F.mse_loss(logits, y_target)
        elif self.task_type == "binary":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
        elif self.task_type == "multilabel":
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
        else:  # classification
            loss = F.cross_entropy(logits, y)
        return loss

    def compute_loss_and_grad(self, logits: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute task-specific loss and gradient of loss w.r.t logits."""
        loss = self.compute_loss(logits, y)

        if self.task_type == "regression":
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            grad = logits - y_target
        elif self.task_type == "binary":
            grad = (
                (logits.sigmoid() - y.float()).unsqueeze(-1)
                if y.dim() < logits.dim()
                else (logits.sigmoid() - y.float())
            )
        elif self.task_type == "multilabel":
            grad = logits.sigmoid() - y.float()
        else:  # classification
            probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(y, self.output_dim).float().to(logits.device)
            grad = probs - target_onehot
        return loss, grad

    def compute_metrics(self, logits: Tensor, y: Tensor) -> float:
        """Compute task-specific accuracy metric."""
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            return 0.0

        with torch.no_grad():
            if self.task_type == "regression":
                # For regression, accuracy is R^2
                y_float = y.float()
                # Ensure y_float matches logits shape for squeeze if needed
                y_flat = y_float.view(-1)
                logits_flat = logits.view(-1)

                ss_res = ((y_flat - logits_flat) ** 2).sum()
                ss_tot = ((y_flat - y_flat.mean()) ** 2).sum()
                accuracy = (1 - (ss_res / (ss_tot + 1e-8))).item()
            elif self.task_type in ("binary", "multilabel"):
                # Unified logic for binary/multilabel
                preds = (logits.sigmoid() > 0.5).long()
                if self.task_type == "binary":
                    # Handle potentially squeezed y
                    y_target = y.view_as(preds)
                    accuracy = (preds == y_target).float().mean().item()
                else:
                    # Multilabel: exact match per sample
                    accuracy = (preds == y).all(dim=-1).float().mean().item()
            else:
                accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
        return accuracy
