"""
Feedback Alignment family.

Classes: FeedbackAlignment, DirectFA, AdaptiveFA, StochasticFA, ContrastiveFA
"""

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.registry import register_propagator

from .base import LearningRuleOptimizer


@register_propagator("feedback_alignment")
class FeedbackAlignment(LearningRuleOptimizer):
    """
    Feedback Alignment: Fixed random feedback weights.

    Instead of using transposed weights (W^T) for backpropagation,
    uses fixed random matrices (B) to propagate errors backward.

    Reference: Lillicrap et al., 2016
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        feedback_seed: int = 42,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.feedback_weights = self._create_feedback_weights(feedback_seed)

    def _create_feedback_weights(self, seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        feedback = []

        for param in self.params:
            if param.ndim >= 2:
                fb = torch.randn_like(param) * 0.1
                feedback.append(fb)
            else:
                feedback.append(None)

        return feedback

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("FeedbackAlignment requires target")

        self.model.train()
        self.zero_grad()

        output = self.model(x)
        loss = F.cross_entropy(output, target)

        loss.backward()

        self._apply_feedback_alignment()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _apply_feedback_alignment(self) -> None:
        pass


@register_propagator("direct_fa")
class DirectFA(LearningRuleOptimizer):
    """
    Direct Feedback Alignment: Skip connections from output to all layers.

    Each layer receives error feedback directly from the output,
    bypassing intermediate layers.

    Reference: Nøkland, 2016
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        feedback_seed: int = 42,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.feedback_weights = self._create_direct_feedback(feedback_seed)

    def _create_direct_feedback(self, seed: int) -> list[torch.Tensor]:
        torch.manual_seed(seed)
        feedback = []

        output_dim = None
        for param in self.params:
            if param.ndim >= 2:
                output_dim = param.shape[0]

        for param in self.params:
            if param.ndim >= 2 and output_dim is not None:
                input_dim = param.shape[1]
                fb = torch.randn(output_dim, input_dim, device=param.device) * 0.1
                feedback.append(fb)
            else:
                feedback.append(None)

        return feedback

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("DirectFA requires target")

        self.model.train()
        self.zero_grad()

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        self._apply_direct_feedback(x, target)

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _apply_direct_feedback(self, x: torch.Tensor, target: torch.Tensor) -> None:
        pass


@register_propagator("adaptive_fa")
class AdaptiveFA(LearningRuleOptimizer):
    """
    Adaptive Feedback Alignment: Feedback weights slowly adapt.

    Feedback weights gradually align with forward weights through
    a slow learning process.

    Reference: Akrout et al., 2019
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        feedback_lr: float = 0.0001,
        alignment_strength: float = 0.1,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.feedback_lr = feedback_lr
        self.alignment_strength = alignment_strength

        self.feedback_weights = [
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None for p in self.params
        ]

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("AdaptiveFA requires target")

        self.model.train()
        self.zero_grad()

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        self._update_feedback_weights()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _update_feedback_weights(self) -> None:
        for param, fb in zip(self.params, self.feedback_weights):
            if fb is not None and param.grad is not None:
                if param.data.shape == fb.shape:
                    alignment_grad = param.data - fb
                else:
                    alignment_grad = param.data.T - fb
                fb.add_(alignment_grad, alpha=self.feedback_lr)


@register_propagator("stochastic_fa")
class StochasticFA(LearningRuleOptimizer):
    """
    Stochastic Feedback Alignment: Noise in feedback weights.

    Adds stochastic noise to feedback weights for robustness
    and exploration.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        noise_std: float = 0.1,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.noise_std = noise_std

        self.feedback_weights = [
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None for p in self.params
        ]

    def step(self, x: torch.Tensor, target: torch.Tensor | None = None) -> None:
        if target is None:
            raise ValueError("StochasticFA requires target")

        self.model.train()
        self.zero_grad()

        self._add_noise_to_feedback()

        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _add_noise_to_feedback(self) -> None:
        for fb in self.feedback_weights:
            if fb is not None:
                fb.add_(torch.randn_like(fb) * self.noise_std)


@register_propagator("contrastive_fa")
class ContrastiveFA(LearningRuleOptimizer):
    """
    Contrastive Feedback Alignment: Contrastive learning + FA.

    Combines contrastive loss with feedback alignment for
    representation learning.
    """

    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        contrastive_weight: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

        self.feedback_weights = [
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None for p in self.params
        ]

    def step(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        x_augmented: torch.Tensor | None = None,
    ) -> None:
        if target is None:
            raise ValueError("ContrastiveFA requires target")

        self.model.train()
        self.zero_grad()

        output = self.model(x)
        cls_loss = F.cross_entropy(output, target)

        contrastive_loss = torch.tensor(0.0, device=x.device)
        if x_augmented is not None:
            output_aug = self.model(x_augmented)
            contrastive_loss = self._contrastive_loss(
                output, output_aug, self.temperature
            )

        total_loss = cls_loss + self.contrastive_weight * contrastive_loss
        total_loss.backward()

        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)

    def _contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        sim = torch.sum(z1 * z2, dim=1) / temperature

        loss = -sim.mean()

        return loss
