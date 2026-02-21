"""
EquiTile Enhanced: LayerNorm and Curriculum Learning
=====================================================

Enhanced features for Equilibrium Propagation research:
- TileLayerNorm: Per-tile layer normalization
- CurriculumScheduler: Progressive difficulty training
- EnhancedEquiTile: Wrapper with all enhancements
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .core import EquiTile


# =============================================================================
# Layer Normalization
# =============================================================================

class TileLayerNorm(nn.Module):
    """Layer normalization for individual tiles.

    Stabilizes EP dynamics by normalizing tile activities.

    Parameters
    ----------
    num_neurons : int
        Number of neurons to normalize
    eps : float
        Epsilon for numerical stability
    elementwise_affine : bool
        Whether to learn scale and shift parameters
    """

    def __init__(
        self,
        num_neurons: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.eps = eps
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

        Parameters
        ----------
        x : Tensor
            Activity tensor (batch, neurons)
        training : bool
            Whether in training mode

        Returns
        -------
        Tensor
            Normalized activity
        """
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


# =============================================================================
# Curriculum Learning
# =============================================================================

@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""
    enabled: bool = False
    n_stages: int = 5
    samples_per_stage: int = 1000
    start_easy: bool = True
    auto_progress: bool = True
    progress_threshold: float = 0.1


class CurriculumScheduler:
    """Curriculum learning scheduler for EP.

    Starts with easy examples and gradually increases difficulty.
    This helps EP converge better by providing cleaner learning signals early.

    Parameters
    ----------
    config : CurriculumConfig
        Curriculum configuration
    """

    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses: List[float] = []
        self._difficulty_cache: Dict[int, float] = {}

    def get_sample_weights(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        model: 'EquiTile',
    ) -> torch.Tensor:
        """Get weights for samples based on curriculum stage.

        Parameters
        ----------
        X : Tensor
            Input features
        y : Tensor
            Targets
        model : EquiTile
            The model

        Returns
        -------
        Tensor
            Sample weights
        """
        if not self.config.enabled:
            return torch.ones(len(X))

        n_samples = len(X)

        # Estimate difficulty from prediction error
        with torch.no_grad():
            _, states = model(X, return_states=True)

        errors = []
        for i in range(n_samples):
            sample_error = 0.0
            for tile_id, state in states.items():
                if state['error'] is not None:
                    sample_error += state['error'][i].norm().item()
            errors.append(sample_error)
            self._difficulty_cache[i] = sample_error

        errors = torch.tensor(errors)
        sorted_indices = torch.argsort(errors)

        # Assign weights based on stage
        weights = torch.ones(n_samples)
        progress = self.current_stage / self.config.n_stages

        if self.config.start_easy:
            # Early stages: focus on easy samples
            n_easy = int(n_samples * (0.3 + 0.7 * progress))
            weights[sorted_indices[n_easy:]] = 0.1
        else:
            # Start with hard samples
            n_hard = int(n_samples * (0.3 * (1 - progress)))
            weights[sorted_indices[:n_hard]] = 0.1

        return weights

    def step(self, loss: float):
        """Update curriculum based on loss.

        Parameters
        ----------
        loss : float
            Current training loss
        """
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

    def reset(self):
        """Reset curriculum."""
        self.current_stage = 0
        self.samples_seen = 0
        self.stage_losses = []
        self._difficulty_cache = {}


# =============================================================================
# Enhanced EquiTile
# =============================================================================

@dataclass
class EnhancedEPConfig:
    """Configuration for enhanced EP."""
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-5
    layer_norm_affine: bool = True

    use_curriculum: bool = False
    curriculum_stages: int = 5

    use_weight_norm: bool = False
    init_scheme: Literal["xavier", "kaiming", "orthogonal"] = "xavier"
    init_gain: float = 1.0


class EnhancedEquiTile:
    """Enhanced EquiTile with better EP support.

    Adds:
    - Layer normalization for stability
    - Curriculum learning for better convergence
    - Weight normalization
    - Better initialization

    Parameters
    ----------
    model : EquiTile
        Base EquiTile model
    config : EnhancedEPConfig, optional
        Enhancement configuration

    Examples
    --------
    >>> model = EquiTile(mode='ep', ...)
    >>> enhanced = EnhancedEquiTile(
    ...     model,
    ...     config=EnhancedEPConfig(
    ...         use_layer_norm=True,
    ...         use_curriculum=True,
    ...     )
    ... )
    """

    def __init__(
        self,
        model: 'EquiTile',
        config: Optional[EnhancedEPConfig] = None,
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
        """Apply layer normalization to tile activities.

        Parameters
        ----------
        training : bool
            Whether in training mode
        """
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
        """Training step with enhancements.

        Parameters
        ----------
        x : Tensor
            Input features
        y : Tensor
            Targets
        sample_weights : Tensor, optional
            Sample weights from curriculum

        Returns
        -------
        Dict[str, float]
            Training statistics
        """
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
        """Get sample weights from curriculum.

        Parameters
        ----------
        X : Tensor
            Input features
        y : Tensor
            Targets

        Returns
        -------
        Tensor
            Sample weights
        """
        return self.curriculum.get_sample_weights(X, y, self.model)


# =============================================================================
# Factory Functions
# =============================================================================

def create_enhanced_model(
    neurons_per_tile: int = 64,
    num_layers: int = 4,
    tiles_per_layer: int = 4,
    input_dim: int = 784,
    output_dim: int = 10,
    use_layer_norm: bool = True,
    use_curriculum: bool = True,
    **kwargs,
) -> EnhancedEquiTile:
    """Create an enhanced EquiTile model.

    Parameters
    ----------
    neurons_per_tile : int
        Neurons per tile
    num_layers : int
        Number of layers
    tiles_per_layer : int
        Tiles per layer
    input_dim : int
        Input dimension
    output_dim : int
        Output dimension
    use_layer_norm : bool
        Enable layer normalization
    use_curriculum : bool
        Enable curriculum learning

    Returns
    -------
    EnhancedEquiTile
        Enhanced model
    """
    from .core import EquiTile

    model = EquiTile(
        neurons_per_tile=neurons_per_tile,
        num_layers=num_layers,
        tiles_per_layer=tiles_per_layer,
        input_dim=input_dim,
        output_dim=output_dim,
        mode='ep',
        **kwargs,
    )

    enhanced = EnhancedEquiTile(
        model,
        config=EnhancedEPConfig(
            use_layer_norm=use_layer_norm,
            use_curriculum=use_curriculum,
        )
    )

    return enhanced
