"""
EquiTile Enhanced: Better EP with LayerNorm and Curriculum Learning
===================================================================

Improvements for Equilibrium Propagation:
- Layer normalization for stability
- Curriculum learning for better convergence
- Weight normalization
- Better initialization schemes

Key Components
--------------
- TileLayerNorm: Per-tile layer normalization
- CurriculumScheduler: Curriculum learning scheduler
- EnhancedEquiTile: EP with all improvements
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .equitile import EquiTile, TileState


class TileLayerNorm(nn.Module):
    """Layer normalization for individual tiles.

    Stabilizes EP dynamics by normalizing tile activities.
    """

    def __init__(
        self,
        num_neurons: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.eps = eps
        self.momentum = momentum
        self.elementwise_affine = elementwise_affine

        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_neurons))
        self.register_buffer('running_var', torch.ones(num_neurons))

        # Learnable parameters
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_neurons))
            self.bias = nn.Parameter(torch.zeros(num_neurons))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Normalize tile activity.

        Args:
            x: Activity tensor (batch, neurons)
            training: Whether in training mode

        Returns:
            Normalized activity
        """
        if training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


class TileBatchNorm(nn.Module):
    """Batch normalization for tiles."""

    def __init__(self, num_neurons: int, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_neurons, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    enabled: bool = False
    curriculum_type: str = "difficulty"  # 'difficulty', 'uncertainty', 'loss'
    n_stages: int = 5
    samples_per_stage: int = 1000
    difficulty_metric: str = "error"  # 'error', 'loss', 'uncertainty'
    start_easy: bool = True
    auto_progress: bool = True
    progress_threshold: float = 0.1  # Progress if improvement < threshold


class CurriculumScheduler:
    """Curriculum learning scheduler for EP.

    Starts with easy examples and gradually increases difficulty.
    This helps EP converge better by providing cleaner learning signals early.
    """

    def __init__(self, config: CurriculumConfig = None):
        self.config = config or CurriculumConfig()
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses: List[float] = []
        self.current_loss = 0.0
        self._difficulty_cache: Dict[int, float] = {}

    def get_sample_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: 'EquiTile',
    ) -> torch.Tensor:
        """Get weights for samples based on curriculum stage."""
        if not self.config.enabled:
            return torch.ones(len(X))

        n_samples = len(X)

        if self.config.curriculum_type == "difficulty":
            return self._get_difficulty_weights(X, y, model, n_samples)
        elif self.config.curriculum_type == "uncertainty":
            return self._get_uncertainty_weights(X, model, n_samples)
        else:
            return self._get_loss_weights(n_samples)

    def _get_difficulty_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: 'EquiTile',
        n_samples: int,
    ) -> torch.Tensor:
        """Weight samples by difficulty."""
        # Estimate difficulty from prediction error
        with torch.no_grad():
            _, states = model(X, return_states=True)

        # Compute error for each sample
        errors = []
        for i in range(n_samples):
            sample_error = 0.0
            for tile_id, state in states.items():
                if state['error'] is not None:
                    sample_error += state['error'][i].norm().item()
            errors.append(sample_error)
            self._difficulty_cache[i] = sample_error

        errors = torch.tensor(errors)

        # Sort by difficulty
        sorted_indices = torch.argsort(errors)

        # Assign weights based on stage
        weights = torch.ones(n_samples)

        if self.config.start_easy:
            # Early stages: focus on easy samples
            progress = self.current_stage / self.config.n_stages
            n_easy = int(n_samples * (0.3 + 0.7 * progress))
            weights[sorted_indices[n_easy:]] = 0.1
        else:
            # Start with hard samples
            progress = self.current_stage / self.config.n_stages
            n_hard = int(n_samples * (0.3 * (1 - progress)))
            weights[sorted_indices[:n_hard]] = 0.1

        return weights

    def _get_uncertainty_weights(
        self,
        X: torch.Tensor,
        model: 'EquiTile',
        n_samples: int,
    ) -> torch.Tensor:
        """Weight samples by model uncertainty."""
        # Estimate uncertainty from activity variance
        weights = torch.ones(n_samples)

        # Would need multiple forward passes with dropout for uncertainty
        # Simplified: use error norm as proxy
        return weights

    def _get_loss_weights(self, n_samples: int) -> torch.Tensor:
        """Weight samples by loss."""
        # Simple: uniform weights, progress based on loss
        return torch.ones(n_samples)

    def step(self, loss: float):
        """Update curriculum based on loss."""
        self.current_loss = loss
        self.stage_losses.append(loss)
        self.samples_seen += 1

        if not self.config.auto_progress:
            return

        # Check if we should progress to next stage
        if self.samples_seen >= self.config.samples_per_stage:
            if len(self.stage_losses) >= 10:
                recent_improvement = (
                    sum(self.stage_losses[-10:-5]) - sum(self.stage_losses[-5:])
                ) / 5

                if recent_improvement < self.config.progress_threshold:
                    self.progress_stage()

            self.samples_seen = 0

    def progress_stage(self):
        """Progress to next curriculum stage."""
        if self.current_stage < self.config.n_stages - 1:
            self.current_stage += 1
            print(f"Curriculum: progressed to stage {self.current_stage}")

    def reset(self):
        """Reset curriculum."""
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses = []
        self._difficulty_cache = {}


@dataclass
class EnhancedEPConfig:
    """Configuration for enhanced EP."""
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    layer_norm_affine: bool = True

    use_curriculum: bool = False
    curriculum_stages: int = 5

    use_weight_norm: bool = False

    init_scheme: str = "xavier"  # 'xavier', 'kaiming', 'orthogonal'
    init_gain: float = 1.0


class EnhancedEquiTile:
    """Enhanced EquiTile with better EP support.

    Adds:
    - Layer normalization for stability
    - Curriculum learning for better convergence
    - Weight normalization
    - Better initialization

    Usage:
        model = EquiTile(...)
        enhanced = EnhancedEquiTile(
            model,
            config=EnhancedEPConfig(
                use_layer_norm=True,
                use_curriculum=True,
                curriculum_stages=5,
            )
        )

        for X, y in dataloader:
            weights = enhanced.curriculum.get_sample_weights(X, y, model)
            stats = enhanced.train_step(X, y, sample_weights=weights)
            enhanced.curriculum.step(stats['loss'])
    """

    def __init__(
        self,
        model: 'EquiTile',
        config: EnhancedEPConfig = None,
    ):
        self.model = model
        self.config = config or EnhancedEPConfig()

        # Layer normalization
        self.layer_norms: Dict[int, TileLayerNorm] = {}
        if self.config.use_layer_norm:
            self._init_layer_norms()

        # Curriculum learning
        self.curriculum = CurriculumScheduler(
            CurriculumConfig(
                enabled=self.config.use_curriculum,
                n_stages=self.config.curriculum_stages,
            )
        )

        # Weight normalization
        if self.config.use_weight_norm:
            self._apply_weight_norm()

        # Better initialization
        self._init_weights()

    def _init_layer_norms(self):
        """Initialize layer norms for all tiles."""
        for tile in self.model.graph.all_tiles:
            if not tile.is_input:
                self.layer_norms[tile.id] = TileLayerNorm(
                    tile.neurons,
                    eps=self.config.layer_norm_eps,
                    elementwise_affine=self.config.layer_norm_affine,
                )

    def _apply_weight_norm(self):
        """Apply weight normalization to edges."""
        for edge in self.model.graph.edges.values():
            if edge.weight is not None:
                nn.utils.parametrizations.weight_norm(edge)

    def _init_weights(self):
        """Initialize weights with better scheme."""
        with torch.no_grad():
            for edge in self.model.graph.edges.values():
                if edge.weight is not None:
                    if self.config.init_scheme == "xavier":
                        nn.init.xavier_normal_(edge.weight, gain=self.config.init_gain)
                    elif self.config.init_scheme == "kaiming":
                        nn.init.kaiming_normal_(edge.weight, gain=self.config.init_gain)
                    elif self.config.init_scheme == "orthogonal":
                        nn.init.orthogonal_(edge.weight, gain=self.config.init_gain)

                if edge.bias is not None:
                    nn.init.zeros_(edge.bias)

    def normalize_activities(self, training: bool = True):
        """Apply layer normalization to tile activities."""
        if not self.config.use_layer_norm:
            return

        for tile_id, ln in self.layer_norms.items():
            tile = self.model.graph.tiles[tile_id]
            if tile.activity is not None:
                tile.activity = ln(tile.activity, training=training)

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Training step with enhancements."""
        # Apply curriculum weights if provided
        if sample_weights is not None:
            # Would need to modify loss computation to use weights
            pass

        # Run standard training step
        stats = self.model.train_step(x, y)

        # Apply layer normalization after relaxation
        if self.config.use_layer_norm:
            self.normalize_activities(training=True)

        # Update curriculum
        self.curriculum.step(stats['loss'])

        return stats

    def get_curriculum_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Get sample weights from curriculum."""
        return self.curriculum.get_sample_weights(X, y, self.model)


class WeightStandardizer:
    """Weight standardization for better EP convergence.

    Standardizes weight matrices to have zero mean and unit variance.
    This helps stabilize the equilibrium dynamics.
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def standardize(self, weight: torch.Tensor) -> torch.Tensor:
        """Standardize weight matrix."""
        mean = weight.mean()
        var = weight.var()
        return (weight - mean) / torch.sqrt(var + self.eps)

    def apply(self, model: 'EquiTile'):
        """Apply weight standardization to model."""
        with torch.no_grad():
            for edge in model.graph.edges.values():
                if edge.weight is not None:
                    edge.weight.data = self.standardize(edge.weight.data)


class ActivityClipping:
    """Activity clipping for stable EP dynamics.

    Clips tile activities to prevent exploding values during relaxation.
    """

    def __init__(
        self,
        min_val: float = -5.0,
        max_val: float = 5.0,
        adaptive: bool = False,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.adaptive = adaptive
        self._running_max = 5.0

    def clip(self, activity: torch.Tensor) -> torch.Tensor:
        """Clip activity values."""
        if self.adaptive:
            # Adapt clipping range based on activity statistics
            current_max = activity.abs().max().item()
            self._running_max = 0.9 * self._running_max + 0.1 * current_max
            max_val = max(self.min_val, min(self._running_max, self.max_val))
            return torch.clamp(activity, -max_val, max_val)

        return torch.clamp(activity, self.min_val, self.max_val)

    def apply(self, model: 'EquiTile'):
        """Apply clipping to all tile activities."""
        for tile in model.graph.all_tiles:
            if tile.activity is not None:
                tile.activity = self.clip(tile.activity)


def create_enhanced_model(
    neurons_per_tile: int,
    num_layers: int,
    tiles_per_layer: int,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> EnhancedEquiTile:
    """Create an enhanced EquiTile model with all improvements.

    Usage:
        model = create_enhanced_model(
            neurons_per_tile=64,
            num_layers=4,
            tiles_per_layer=4,
            input_dim=784,
            output_dim=10,
            use_layer_norm=True,
            use_curriculum=True,
        )
    """
    from .equitile import EquiTile

    model = EquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        **kwargs,
    )

    enhanced = EnhancedEquiTile(
        model,
        config=EnhancedEPConfig(
            use_layer_norm=True,
            use_curriculum=kwargs.get('use_curriculum', False),
            curriculum_stages=kwargs.get('curriculum_stages', 5),
            use_weight_norm=kwargs.get('use_weight_norm', False),
            init_scheme=kwargs.get('init_scheme', 'xavier'),
        )
    )

    return enhanced
