"""
Training safety mechanisms to prevent NaN, inf, and gradient explosions.
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    """Safety configuration for training."""

    max_grad_norm: float = 10.0
    nan_check_frequency: int = 10  # Check every N batches
    lr_reduction_on_nan: float = 0.5
    max_nan_retries: int = 3
    enable_anomaly_detection: bool = False


class SafetyWrapper:
    """
    Wraps training to catch and handle numerical instabilities.

    Usage:
        safety = SafetyWrapper()
        for batch in dataloader:
            loss = model.forward(batch)
            success, info = safety.safe_backward_and_step(loss, optimizer, model)
            if not success:
                if safety.should_abort():
                    raise RuntimeError("Training aborted due to repeated failures")
                safety.handle_failure(optimizer)
    """

    def __init__(self, config: SafetyConfig = None):
        self.config = config or SafetyConfig()
        self.consecutive_failures = 0
        self.total_failures = 0
        self.step_count = 0

        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)

    def safe_backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        clip_norm: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform backward pass + optimizer step with safety checks.

        Args:
            loss: Loss tensor
            optimizer: Optimizer instance
            model: Model instance
            clip_norm: Optional gradient clipping override

        Returns:
            (success, info):
                success=True if step completed successfully
                info=dict with metrics (on success) or error info (on failure)
        """
        self.step_count += 1

        # 1. Check loss validity BEFORE backward
        if not torch.isfinite(loss):
            self.consecutive_failures += 1
            self.total_failures += 1
            return False, {
                "error": "loss_nan_or_inf",
                "loss_value": float(loss),
                "step": self.step_count,
            }

        # 2. Backward pass
        try:
            loss.backward()
        except RuntimeError as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            optimizer.zero_grad()  # Clean up partial gradients
            return False, {
                "error": "backward_failed",
                "exception": str(e),
                "step": self.step_count,
            }

        # 3. Check gradients for NaN/Inf
        total_norm = 0.0
        has_nan = False
        nan_param_names = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                if not torch.isfinite(param_norm):
                    has_nan = True
                    nan_param_names.append(name)
                    break
                total_norm += param_norm.item() ** 2

        if has_nan:
            self.consecutive_failures += 1
            self.total_failures += 1
            optimizer.zero_grad()
            logger.warning(f"NaN gradient detected in parameters: {nan_param_names}")
            return False, {
                "error": "grad_nan",
                "grad_norm": float("nan"),
                "nan_params": nan_param_names,
                "step": self.step_count,
            }

        total_norm = total_norm**0.5

        # 4. Clip gradients
        clip_value = clip_norm if clip_norm is not None else self.config.max_grad_norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # 5. Step optimizer
        try:
            optimizer.step()
            optimizer.zero_grad()
        except RuntimeError as e:
            self.consecutive_failures += 1
            self.total_failures += 1
            return False, {
                "error": "optimizer_step_failed",
                "exception": str(e),
                "step": self.step_count,
            }

        # Success! Reset failure counter
        self.consecutive_failures = 0

        return True, {
            "grad_norm": total_norm,
            "loss": float(loss),
            "step": self.step_count,
        }

    def should_abort(self) -> bool:
        """Check if training should be aborted due to repeated failures."""
        return self.consecutive_failures >= self.config.max_nan_retries

    def handle_failure(self, optimizer: torch.optim.Optimizer):
        """
        Handle a training failure by reducing learning rate.
        Call this after safe_backward_and_step returns False.
        """
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            new_lr = old_lr * self.config.lr_reduction_on_nan
            param_group["lr"] = new_lr
            logger.warning(
                f"Reduced LR from {old_lr:.2e} to {new_lr:.2e} "
                f"(failure {self.consecutive_failures}/{self.config.max_nan_retries})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            "total_steps": self.step_count,
            "total_failures": self.total_failures,
            "consecutive_failures": self.consecutive_failures,
            "failure_rate": (
                self.total_failures / self.step_count if self.step_count > 0 else 0.0
            ),
        }
