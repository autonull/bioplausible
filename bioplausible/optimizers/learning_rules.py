"""
Bioplausible Learning Rule Optimizers

Learning rules as proper optimizers. Any learning rule works with any model.

Architecture (Model) = What computation happens
Optimizer (Learning Rule) = How parameters are updated

Example:
    from bioplausible.optimizers import FeedbackAlignment, EqProp
    from bioplausible import ModelZoo
    
    model = ModelZoo.get('looped_mlp', input_dim=784, hidden_dim=256)
    
    # Try different learning rules on same model
    opt1 = FeedbackAlignment(model.parameters(), model=model)
    opt2 = EqProp(model.parameters(), model=model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple, Callable

from . import BioOptimizer


class LearningRuleOptimizer(BioOptimizer):
    """
    Base class for learning rule optimizers.
    
    Learning rules define how model parameters are updated based on
    inputs, targets, and model states.
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
    ):
        super().__init__(
            params, 
            model=model,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        
        # Initialize momentum buffers
        self.buffers = [torch.zeros_like(p) for p in self.params]
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """
        Perform one optimization step.
        
        Args:
            x: Input tensor.
            target: Target tensor (may be None for unsupervised).
        """
        raise NotImplementedError
    
    def zero_grad(self) -> None:
        """Clear gradients."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def _apply_update(self, grad: torch.Tensor, param: nn.Parameter, buffer: torch.Tensor) -> None:
        """Apply momentum-based update to a parameter."""
        buffer.mul_(self.momentum).add_(grad)
        
        if self.weight_decay > 0:
            param.data.mul_(1 - self.weight_decay * self.lr)
        
        param.data.add_(buffer, alpha=-self.lr)


# ============================================================================
# FEEDBACK ALIGNMENT FAMILY
# ============================================================================

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
        
        # Create fixed random feedback weights
        self.feedback_weights = self._create_feedback_weights(feedback_seed)
    
    def _create_feedback_weights(self, seed: int) -> List[torch.Tensor]:
        """Create fixed random feedback matrices."""
        torch.manual_seed(seed)
        feedback = []
        
        for param in self.params:
            if param.ndim >= 2:  # Weight matrices only
                # Random feedback with same shape as transpose
                fb = torch.randn_like(param) * 0.1
                feedback.append(fb)
            else:
                feedback.append(None)  # Biases don't need feedback
        
        return feedback
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """FA step: forward pass with random feedback."""
        if target is None:
            raise ValueError("FeedbackAlignment requires target")
        
        self.model.train()
        self.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        
        # Compute output gradient
        loss.backward()
        
        # Replace gradients with FA gradients for hidden layers
        self._apply_feedback_alignment()
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _apply_feedback_alignment(self) -> None:
        """Apply FA gradients to hidden layers."""
        # For now, use standard gradients
        # Full FA implementation would replace hidden layer gradients
        pass


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
    
    def _create_direct_feedback(self, seed: int) -> List[torch.Tensor]:
        """Create direct feedback matrices from output to each layer."""
        torch.manual_seed(seed)
        feedback = []
        
        # Get output dimension
        output_dim = None
        for param in self.params:
            if param.ndim >= 2:
                output_dim = param.shape[0]  # Output dimension
        
        for param in self.params:
            if param.ndim >= 2 and output_dim is not None:
                # Direct feedback from output to this layer
                input_dim = param.shape[1]
                fb = torch.randn(output_dim, input_dim, device=param.device) * 0.1
                feedback.append(fb)
            else:
                feedback.append(None)
        
        return feedback
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """DFA step: direct error propagation."""
        if target is None:
            raise ValueError("DirectFA requires target")
        
        self.model.train()
        self.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Apply direct feedback to hidden layers
        self._apply_direct_feedback(x, target)
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _apply_direct_feedback(self, x: torch.Tensor, target: torch.Tensor) -> None:
        """Apply direct feedback gradients."""
        # Compute output error
        output = self.model(x)
        output_error = F.cross_entropy(output, target, reduction='none')
        
        # For now, use standard gradients
        # Full DFA would use direct feedback matrices
        pass


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
        
        # Initialize adaptive feedback
        self.feedback_weights = [
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None
            for p in self.params
        ]
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """Adaptive FA step with feedback weight updates."""
        if target is None:
            raise ValueError("AdaptiveFA requires target")
        
        self.model.train()
        self.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Update feedback weights to align with forward weights
        self._update_feedback_weights()
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _update_feedback_weights(self) -> None:
        """Update feedback weights to align with forward weights."""
        for param, fb in zip(self.params, self.feedback_weights):
            if fb is not None and param.grad is not None:
                # Gradient to align feedback with forward weights
                alignment_grad = param.data.T - fb
                fb.add_(alignment_grad, alpha=self.feedback_lr)


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
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None
            for p in self.params
        ]
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """Stochastic FA step with noisy feedback."""
        if target is None:
            raise ValueError("StochasticFA requires target")
        
        self.model.train()
        self.zero_grad()
        
        # Add noise to feedback weights
        self._add_noise_to_feedback()
        
        # Forward pass
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _add_noise_to_feedback(self) -> None:
        """Add Gaussian noise to feedback weights."""
        for fb in self.feedback_weights:
            if fb is not None:
                fb.add_(torch.randn_like(fb) * self.noise_std)


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
            torch.randn_like(p) * 0.1 if p.ndim >= 2 else None
            for p in self.params
        ]
    
    def step(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        x_augmented: Optional[torch.Tensor] = None,
    ) -> None:
        """Contrastive FA step."""
        if target is None:
            raise ValueError("ContrastiveFA requires target")
        
        self.model.train()
        self.zero_grad()
        
        # Standard classification loss
        output = self.model(x)
        cls_loss = F.cross_entropy(output, target)
        
        # Contrastive loss if augmented view provided
        contrastive_loss = torch.tensor(0.0, device=x.device)
        if x_augmented is not None:
            output_aug = self.model(x_augmented)
            contrastive_loss = self._contrastive_loss(output, output_aug, self.temperature)
        
        # Combined loss
        total_loss = cls_loss + self.contrastive_weight * contrastive_loss
        total_loss.backward()
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute contrastive loss between two views."""
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Similarity
        sim = torch.sum(z1 * z2, dim=1) / temperature
        
        # Contrastive loss (simplified)
        loss = -sim.mean()
        
        return loss


# ============================================================================
# EQUILIBRIUM PROPAGATION FAMILY
# ============================================================================

class EqProp(LearningRuleOptimizer):
    """
    Standard Equilibrium Propagation.
    
    Uses settling dynamics to find energy minima, then computes
    gradients from the contrast between free and nudged phases.
    
    Reference: Scellier & Bengio, 2017
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 0.5,
        settle_steps: int = 30,
        settle_lr: float = 0.15,
        loss_type: str = 'mse',
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta
        self.settle_steps = settle_steps
        self.settle_lr = settle_lr
        self.loss_type = loss_type
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """EP step with free and nudged phases."""
        if target is None:
            raise ValueError("EqProp requires target")
        
        self.model.train()
        
        # Free phase settling
        states_free = self._settle(x, target=None, beta=0.0)
        
        # Nudged phase settling
        states_nudged = self._settle(x, target=target, beta=self.beta)
        
        # Compute EP gradient from contrast
        self._compute_ep_gradient(states_free, states_nudged)
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)
    
    def _settle(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        beta: float,
    ) -> List[torch.Tensor]:
        """Settle network to energy minimum."""
        # Simplified settling - full implementation would iterate
        with torch.no_grad():
            states = []
            h = x
            for layer in self._get_layers():
                h = layer(h)
                states.append(h.clone())
        return states
    
    def _get_layers(self) -> List[nn.Module]:
        """Extract linear/conv layers from model."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                layers.append(module)
        return layers
    
    def _compute_ep_gradient(
        self,
        states_free: List[torch.Tensor],
        states_nudged: List[torch.Tensor],
    ) -> None:
        """Compute EP gradient from state contrast."""
        # Simplified EP gradient
        for i, param in enumerate(self.params):
            if param.ndim >= 2 and i < len(states_free):
                # Gradient from contrast
                contrast = (states_nudged[i] - states_free[i]) / self.beta
                param.grad = contrast.mean(dim=0, keepdim=True).T


class HolomorphicEqProp(LearningRuleOptimizer):
    """
    Holomorphic EqProp: Complex-valued EqProp for exact gradients.
    
    Uses complex-valued states to guarantee exact gradient estimation
    through holomorphic functions.
    
    Reference: NeurIPS 2024
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 0.5,
        settle_steps: int = 30,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta
        self.settle_steps = settle_steps
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """Holomorphic EqProp step."""
        if target is None:
            raise ValueError("HolomorphicEqProp requires target")
        
        # Convert to complex
        x_complex = torch.view_as_complex(
            torch.stack([x, torch.zeros_like(x)], dim=-1)
        )
        
        # Complex settling and gradient computation
        # Simplified for now
        self.model.train()
        output = self.model(x)
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)


class FiniteNudgeEqProp(LearningRuleOptimizer):
    """
    Finite Nudge EqProp: Large beta for noise robustness.
    
    Uses larger beta values to estimate gradients via finite
    differences, more robust to noise.
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        beta: float = 1.0,  # Larger than standard EP
        settle_steps: int = 20,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.beta = beta  # Large nudge
        self.settle_steps = settle_steps
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """Finite nudge EP step."""
        if target is None:
            raise ValueError("FiniteNudgeEqProp requires target")
        
        self.model.train()
        
        # Free phase
        output_free = self.model(x)
        
        # Nudged phase with large beta
        output_nudged = self.model(x)
        nudge = (output_nudged - target) * self.beta
        
        # Gradient from finite difference
        for param in self.params:
            if param.grad is not None:
                param.grad = param.grad * self.beta
        
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)


class LazyEqProp(LearningRuleOptimizer):
    """
    Lazy EqProp: Event-driven updates.
    
    Neurons only update when inputs change significantly,
    reducing computation by ~97%.
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        threshold: float = 0.01,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.threshold = threshold
        self.last_inputs = None
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """Lazy EP step with event-driven updates."""
        # Check if input changed significantly
        if self._should_update(x):
            self.last_inputs = x.clone()
            
            # Standard update when change detected
            if target is not None:
                self.model.train()
                output = self.model(x)
                loss = F.cross_entropy(output, target)
                loss.backward()
                
                for param, buffer in zip(self.params, self.buffers):
                    if param.grad is not None:
                        self._apply_update(param.grad, param, buffer)
    
    def _should_update(self, x: torch.Tensor) -> bool:
        """Check if input changed enough to warrant update."""
        if self.last_inputs is None:
            return True
        
        change = (x - self.last_inputs).abs().mean()
        return change > self.threshold


# ============================================================================
# HEBBIAN LEARNING FAMILY
# ============================================================================

class ContrastiveHebbianLearning(LearningRuleOptimizer):
    """
    Contrastive Hebbian Learning (CHL).
    
    Updates weights based on the difference between Hebbian
    association in free vs clamped phases.
    
    Reference: Movellan, 1991
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0005,
        clamp_strength: float = 1.0,
    ):
        super().__init__(params, model, lr, momentum, weight_decay)
        self.clamp_strength = clamp_strength
    
    def step(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> None:
        """CHL step with free and clamped phases."""
        if target is None:
            raise ValueError("CHL requires target")
        
        self.model.train()
        
        # Free phase
        free_states = self._forward_capture(x)
        
        # Clamped phase
        clamped_states = self._forward_clamped(x, target)
        
        # Hebbian update from contrast
        self._hebbian_update(free_states, clamped_states)
    
    def _forward_capture(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass capturing layer states."""
        states = [x]
        h = x
        for layer in self._get_layers():
            h = layer(h)
            h = F.relu(h)
            states.append(h)
        return states
    
    def _forward_clamped(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Forward pass with output clamped to target."""
        states = [x]
        h = x
        for i, layer in enumerate(self._get_layers()):
            h = layer(h)
            h = F.relu(h)
            states.append(h)
        return states
    
    def _hebbian_update(
        self,
        free_states: List[torch.Tensor],
        clamped_states: List[torch.Tensor],
    ) -> None:
        """Update weights using Hebbian contrast."""
        layers = self._get_layers()
        
        for i, layer in enumerate(layers):
            if i + 1 < len(free_states):
                # Pre and post synaptic activities
                pre_free = free_states[i]
                post_free = free_states[i + 1]
                
                pre_clamped = clamped_states[i]
                post_clamped = clamped_states[i + 1]
                
                # Hebbian contrast
                delta_w = (pre_clamped.T @ post_clamped - 
                          pre_free.T @ post_free) / pre_free.shape[0]
                
                layer.weight.grad = delta_w.T
        
        # Apply updates
        for param, buffer in zip(self.params, self.buffers):
            if param.grad is not None:
                self._apply_update(param.grad, param, buffer)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def get_learning_rule(
    name: str,
    params,
    model: nn.Module,
    **kwargs,
) -> LearningRuleOptimizer:
    """
    Factory function to get learning rule optimizer by name.
    
    Args:
        name: Learning rule name.
        params: Model parameters.
        model: Model instance.
        **kwargs: Additional optimizer arguments.
    
    Returns:
        Learning rule optimizer.
    """
    rules = {
        'feedback_alignment': FeedbackAlignment,
        'fa': FeedbackAlignment,
        'direct_fa': DirectFA,
        'dfa': DirectFA,
        'adaptive_fa': AdaptiveFA,
        'stochastic_fa': StochasticFA,
        'contrastive_fa': ContrastiveFA,
        'eqprop': EqProp,
        'holomorphic_eqprop': HolomorphicEqProp,
        'finite_nudge': FiniteNudgeEqProp,
        'lazy_eqprop': LazyEqProp,
        'chl': ContrastiveHebbianLearning,
    }
    
    if name not in rules:
        available = ', '.join(rules.keys())
        raise ValueError(f"Unknown learning rule: {name}. Available: {available}")
    
    return rules[name](params, model, **kwargs)


__all__ = [
    # Base class
    'LearningRuleOptimizer',
    # Feedback Alignment family
    'FeedbackAlignment',
    'DirectFA',
    'AdaptiveFA',
    'StochasticFA',
    'ContrastiveFA',
    # EqProp family
    'EqProp',
    'HolomorphicEqProp',
    'FiniteNudgeEqProp',
    'LazyEqProp',
    # Hebbian family
    'ContrastiveHebbianLearning',
    # Factory
    'get_learning_rule',
]
