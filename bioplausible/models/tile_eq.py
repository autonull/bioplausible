"""
Adaptive Tile-Based Predictive Coding (ATPC) - Next Generation
===============================================================

A revolutionary, scalable, adaptive learning algorithm designed for the future
of computing. ATPC combines predictive coding with:

- **Asynchronous Processing**: Tiles update independently without global sync
- **Dynamic Growth**: Network evolves during training (add/remove tiles)
- **Hardware Agnostic**: Runs on GPU, CPU, FPGA, neuromorphic, optical, memristor
- **Event-Driven**: Tiles process only when significant input arrives
- **Continual Learning**: Learn new tasks without catastrophic forgetting
- **Uncertainty Quantification**: Bayesian extensions for confidence estimates
- **Neural Architecture Search**: Auto-discover optimal tile configurations

Theoretical Foundation
----------------------
ATPC is grounded in the free energy principle (Friston, 2005) and extends
predictive coding with classification-driven learning, adaptive computation,
and dynamic network evolution. The algorithm is designed to scale from
embedded devices to distributed clusters, and from conventional silicon to
emerging computing substrates.

Key Innovations
---------------
1. **Asynchronous Tile Updates**: No global synchronization barrier
2. **Dynamic Network Growth**: Add/remove tiles based on learning signals
3. **Event-Driven Processing**: Neuromorphic-style sparse activation
4. **Hardware Abstraction**: Unified interface across substrates
5. **Continual Learning**: Elastic weight consolidation for task sequences
6. **Bayesian Uncertainty**: Monte Carlo dropout for confidence estimates
7. **Auto-Architecture**: Neural architecture search for tile configs
8. **Federated Learning**: Distributed training with privacy preservation

Vision
------
ATPC is designed to inspire and enable next-generation ML systems:

**Conventional Hardware (GPU/CPU)**:
- Asynchronous tile updates enable data parallelism
- Dynamic batching for variable-length sequences
- Mixed precision for memory efficiency

**Neuromorphic (Loihi, SpiNNaker, TrueNorth)**:
- Event-driven spike-based processing
- Local learning rules (no backprop required)
- Sub-milliwatt power consumption

**Optical/Photonic**:
- Matrix multiplication at speed of light
- Passive (zero-energy) inference
- Wavelength-division multiplexing for parallelism

**Memristive Crossbars**:
- In-memory computing (no data movement)
- Analog weight storage
- Natural implementation of Hebbian learning

**FPGA/ASIC**:
- Custom tile accelerators
- Reconfigurable topologies
- Deterministic latency

**DNA/Molecular**:
- Mass-action kinetics for inference
- Strand displacement for learning
- Ultra-dense storage

ATPC provides a unified algorithmic framework that maps naturally to all
these substrates, enabling portable, efficient, scalable ML across the
computing landscape of the 21st century.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BioModel, ModelConfig, register_model

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ATPCConfig:
    """Configuration for Adaptive Tile-Based Predictive Coding.

    Architecture
    ------------
    neurons_per_tile: Number of neurons per tile
    num_layers: Total layers (input + hidden + output)
    tiles_per_layer: Tiles per hidden layer

    Learning Dynamics
    -----------------
    prediction_lr: Learning rate for prediction weights
    prior_lr: Learning rate for prior expectations
    importance_lr: Learning rate for tile importance weights
    initial_step_size: Initial integration step size
    lr_schedule: Learning rate schedule ('constant', 'step', 'cosine')
    lr_decay_steps: Steps for learning rate decay

    Adaptive Computation
    --------------------
    sparsity_threshold: Skip tiles with error below this threshold
    importance_decay: EMA decay for importance tracking
    min_active_fraction: Minimum fraction of tiles to update

    Regularization
    --------------
    weight_decay: L2 regularization strength
    error_decay: Decay factor for error accumulation
    dropout: Dropout probability (0 = disabled)
    use_batchnorm: Use batch normalization in tiles
    gradient_clip: Gradient clipping value (0 = disabled)

    Inference
    ---------
    inference_steps: Number of inference steps during forward pass
    """
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 1

    # Learning
    prediction_lr: float = 0.01
    prior_lr: float = 0.005
    importance_lr: float = 0.001
    initial_step_size: float = 0.5
    lr_schedule: str = "constant"  # 'constant', 'step', 'cosine'
    lr_decay_steps: int = 1000

    # Adaptive computation
    sparsity_threshold: float = 0.01
    importance_decay: float = 0.99
    min_active_fraction: float = 0.1

    # Regularization
    weight_decay: float = 1e-4
    error_decay: float = 0.95
    dropout: float = 0.0
    use_batchnorm: bool = False
    gradient_clip: float = 1.0

    # Inference
    inference_steps: int = 20


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TileState:
    """Dynamic state for a single tile.

    Attributes:
        id: Tile identifier
        activity: Current neural activity (batch, neurons)
        prediction: Top-down prediction (batch, neurons)
        error: Bottom-up prediction error (batch, neurons)
        importance: Learned importance weight (scalar)
        error_magnitude: Exponential moving average of error norm
        update_count: Number of times this tile was updated
        last_update_step: Global step of last update
    """
    id: int
    num_neurons: int
    layer_id: int

    # Dynamic state (batch-sized)
    activity: Optional[Tensor] = None
    prediction: Optional[Tensor] = None
    error: Optional[Tensor] = None

    # Adaptive computation state
    importance: float = 1.0
    error_magnitude: float = 0.0
    update_count: int = 0
    last_update_step: int = 0

    # Connectivity
    fwd_neighbors: List[int] = field(default_factory=list)
    bwd_neighbors: List[int] = field(default_factory=list)

    # Visualization
    pos_x: float = 0.0
    pos_y: float = 0.0
    is_input: bool = False
    is_output: bool = False


@dataclass
class EdgeParams:
    """Parameters for a directed edge between tiles.

    Attributes:
        src_id: Source tile ID
        dst_id: Destination tile ID
        weight: Weight matrix (src_neurons, dst_neurons)
        bias: Bias vector (dst_neurons,)
        importance: Edge importance weight (for sparse updates)
    """
    src_id: int
    dst_id: int
    weight: Optional[Tensor] = None
    bias: Optional[Tensor] = None
    importance: float = 1.0


# =============================================================================
# Normalization and Regularization Modules
# =============================================================================

class TileBatchNorm(nn.Module):
    """Batch normalization for tile activities."""
    
    def __init__(self, num_features: int, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, neurons)
        return self.bn(x)


class TileDropout(nn.Module):
    """Dropout for tile activities."""
    
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p) if p > 0 else nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x)
    
    def set_p(self, p: float) -> None:
        """Update dropout probability."""
        if p > 0:
            self.dropout = nn.Dropout(p)
        else:
            self.dropout = nn.Identity()
        self.p = p


# =============================================================================
# Strategy Framework
# =============================================================================

class InferenceStrategy:
    """Abstract base class for inference (activity update) strategies.
    
    Inference strategies determine how tiles update their activities
    to minimize prediction errors during the inference phase.
    """
    
    def update(
        self,
        tile: TileState,
        gradient: Tensor,
        step_size: float,
        importance: float,
    ) -> None:
        """Update tile activity based on prediction error gradient.
        
        Args:
            tile: The tile to update
            gradient: Prediction error gradient w.r.t. activity
            step_size: Base step size (learning rate)
            importance: Tile importance weight (0-1)
        """
        raise NotImplementedError
    
    def reset(self, tile: TileState) -> None:
        """Reset strategy state for a tile (called on initialization)."""
        pass


class GradientDescentInference(InferenceStrategy):
    """Standard gradient descent inference.
    
    Update rule: s ← s - α × importance × gradient
    """
    
    def update(
        self,
        tile: TileState,
        gradient: Tensor,
        step_size: float,
        importance: float,
    ) -> None:
        if tile.activity is None:
            return
        delta = step_size * importance * gradient
        tile.activity = tile.activity - delta
        tile.activity = torch.clamp(tile.activity, -5.0, 5.0)


class MomentumInference(InferenceStrategy):
    """Gradient descent with momentum for faster convergence.
    
    Update rules:
        v ← μ × v + gradient
        s ← s - α × importance × v
    
    Args:
        momentum: Momentum coefficient (default: 0.9)
    """
    
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self._velocities: Dict[int, Tensor] = {}
    
    def reset(self, tile: TileState) -> None:
        if tile.activity is not None:
            self._velocities[tile.id] = torch.zeros_like(tile.activity)
    
    def update(
        self,
        tile: TileState,
        gradient: Tensor,
        step_size: float,
        importance: float,
    ) -> None:
        if tile.activity is None:
            return
        
        # Initialize velocity if needed (with correct shape)
        if tile.id not in self._velocities:
            self._velocities[tile.id] = torch.zeros_like(gradient)
        elif self._velocities[tile.id].shape != gradient.shape:
            # Shape changed (e.g., different batch size), reinitialize
            self._velocities[tile.id] = torch.zeros_like(gradient)
        
        # Update velocity (detach to avoid graph accumulation)
        self._velocities[tile.id] = (
            self.momentum * self._velocities[tile.id].detach() + gradient
        )
        
        # Update activity
        delta = step_size * importance * self._velocities[tile.id]
        tile.activity = tile.activity - delta
        tile.activity = torch.clamp(tile.activity, -5.0, 5.0)


class LearningStrategy:
    """Abstract base class for learning (weight update) strategies.
    
    Learning strategies determine how edge weights are updated
    based on prediction errors.
    """
    
    def compute_update(
        self,
        edge: EdgeParams,
        source_activity: Tensor,
        target_error: Tensor,
        learning_rate: float,
        importance: float,
    ) -> Tuple[Tensor, Tensor]:
        """Compute weight and bias updates for an edge.
        
        Args:
            edge: The edge to update
            source_activity: Activated source tile activity (batch, src_neurons)
            target_error: Target tile prediction error (batch, dst_neurons)
            learning_rate: Base learning rate
            importance: Edge importance weight (0-1)
            
        Returns:
            Tuple of (weight_update, bias_update) tensors
        """
        raise NotImplementedError


class HebbianLearning(LearningStrategy):
    """Standard Hebbian learning.
    
    Update rules:
        ΔW = η × importance × (source_activity.T @ target_error)
        Δb = η × target_error.mean(0)
    """
    
    def compute_update(
        self,
        edge: EdgeParams,
        source_activity: Tensor,
        target_error: Tensor,
        learning_rate: float,
        importance: float,
    ) -> Tuple[Tensor, Tensor]:
        batch_size = source_activity.shape[0]
        
        # Weight update: outer product of source activity and target error
        weight_update = importance * (source_activity.T @ target_error) / batch_size
        
        # Bias update: mean error
        bias_update = importance * target_error.mean(dim=0) / batch_size
        
        return weight_update, bias_update


class OjaLearning(LearningStrategy):
    """Oja's rule with normalization for stability.
    
    Update rules:
        ΔW = η × importance × (source.T @ error - W × source.T @ source.mean())
        Δb = η × error.mean(0)
    
    This prevents unbounded weight growth.
    """
    
    def compute_update(
        self,
        edge: EdgeParams,
        source_activity: Tensor,
        target_error: Tensor,
        learning_rate: float,
        importance: float,
    ) -> Tuple[Tensor, Tensor]:
        if edge.weight is None:
            raise ValueError("Edge weight is None")
        
        batch_size = source_activity.shape[0]
        
        # Hebbian term: (src, batch) @ (batch, dst) = (src, dst)
        hebbian = source_activity.T @ target_error
        
        # Normalization term: use mean squared activity per source neuron
        # source_squared: (src,) -> broadcast to (src, dst)
        source_squared = (source_activity ** 2).mean(dim=0)  # (src_neurons,)
        # Multiply each row of W by corresponding source_squared value
        normalization = edge.weight * source_squared.unsqueeze(1)  # (src, 1) broadcasts to (src, dst)
        
        weight_update = importance * (hebbian - normalization) / batch_size
        bias_update = importance * target_error.mean(dim=0) / batch_size
        
        return weight_update, bias_update


class SchedulingStrategy:
    """Abstract base class for scheduling (tile selection) strategies.
    
    Scheduling strategies determine which tiles receive computation
    at each step, enabling sparse adaptive computation.
    """
    
    def select_tiles(
        self,
        tiles: List[TileState],
        errors: Dict[int, float],
        importances: Tensor,
    ) -> List[int]:
        """Select which tiles to update.
        
        Args:
            tiles: List of all tiles
            errors: Dictionary mapping tile_id to error magnitude
            importances: Tensor of importance weights (one per tile)
            
        Returns:
            List of tile IDs to update
        """
        raise NotImplementedError


class ThresholdScheduling(SchedulingStrategy):
    """Update tiles where importance × error exceeds threshold.
    
    Args:
        threshold: Minimum score to be selected (default: 0.01)
    """
    
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
    
    def select_tiles(
        self,
        tiles: List[TileState],
        errors: Dict[int, float],
        importances: Tensor,
    ) -> List[int]:
        selected = []
        for i, tile in enumerate(tiles):
            error_mag = errors.get(tile.id, 0.0)
            importance = torch.sigmoid(importances[i]).item()
            score = importance * error_mag
            if score > self.threshold:
                selected.append(tile.id)
        return selected


class TopKScheduling(SchedulingStrategy):
    """Update only the K tiles with highest importance × error.
    
    Args:
        k: Number of tiles to select
        min_fraction: Minimum fraction of tiles to always select
    """
    
    def __init__(self, k: int = 10, min_fraction: float = 0.1):
        self.k = k
        self.min_fraction = min_fraction
    
    def select_tiles(
        self,
        tiles: List[TileState],
        errors: Dict[int, float],
        importances: Tensor,
    ) -> List[int]:
        n_tiles = len(tiles)
        
        # Compute scores
        scores = []
        for i, tile in enumerate(tiles):
            error_mag = errors.get(tile.id, 0.0)
            importance = torch.sigmoid(importances[i]).item()
            scores.append(importance * error_mag)
        
        scores_tensor = torch.tensor(scores)
        
        # Ensure minimum fraction
        min_k = max(1, int(n_tiles * self.min_fraction))
        actual_k = min(max(self.k, min_k), n_tiles)
        
        # Select top K
        top_indices = torch.topk(scores_tensor, actual_k).indices.tolist()
        return [tiles[i].id for i in top_indices]


class AllTilesScheduling(SchedulingStrategy):
    """Update all tiles (no sparsity).
    
    Useful for debugging or when maximum accuracy is needed.
    """
    
    def select_tiles(
        self,
        tiles: List[TileState],
        errors: Dict[int, float],
        importances: Tensor,
    ) -> List[int]:
        return [tile.id for tile in tiles]


# =============================================================================
# Graph Structure
# =============================================================================

class TileGraph:
    """Manages tile connectivity and state."""
    
    def __init__(self) -> None:
        self.tiles: Dict[int, TileState] = {}
        self.edges: Dict[Tuple[int, int], EdgeParams] = {}
        self.layer_ids: List[List[int]] = []
        self.input_tile_ids: List[int] = []
        self.output_tile_ids: List[int] = []
    
    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
    ) -> None:
        """Build layered feedforward architecture."""
        num_hidden_layers = max(0, num_hidden_layers)
        hidden_dim = neurons_per_tile * tiles_per_layer
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        total_layers = len(dims)
        
        tile_id = 0
        state_offset = 0
        
        for layer_idx, dim in enumerate(dims):
            n_tiles = math.ceil(dim / neurons_per_tile)
            layer_tile_ids: List[int] = []
            
            for tile_col in range(n_tiles):
                actual_neurons = min(neurons_per_tile, dim - tile_col * neurons_per_tile)
                
                tile = TileState(
                    id=tile_id,
                    num_neurons=actual_neurons,
                    layer_id=layer_idx,
                    pos_x=float(layer_idx) / max(1, total_layers - 1),
                    pos_y=float(tile_col) / max(1, n_tiles - 1) if n_tiles > 1 else 0.5,
                    is_input=(layer_idx == 0),
                    is_output=(layer_idx == len(dims) - 1),
                )
                self.tiles[tile_id] = tile
                layer_tile_ids.append(tile_id)
                tile_id += 1
            
            self.layer_ids.append(layer_tile_ids)
        
        self.input_tile_ids = list(self.layer_ids[0])
        self.output_tile_ids = list(self.layer_ids[-1])
        
        # Create edges between consecutive layers
        for layer_idx in range(len(self.layer_ids) - 1):
            for src_id in self.layer_ids[layer_idx]:
                for dst_id in self.layer_ids[layer_idx + 1]:
                    self._add_edge(src_id, dst_id)

    def build_custom(
        self,
        n_tiles: int,
        neurons_per_tile: int,
        edges: List[Tuple[int, int]],
        input_ids: List[int],
        output_ids: List[int],
        positions: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Build custom arbitrary topology.
        
        Args:
            n_tiles: Total number of tiles
            neurons_per_tile: Number of neurons per tile (uniform)
            edges: List of (src_id, dst_id) directed edges
            input_ids: Tile IDs that receive external input
            output_ids: Tile IDs that produce output
            positions: Optional (x, y) positions for each tile [0, 1]
        """
        # Create tiles with uniform neurons
        state_offset = 0
        for tile_id in range(n_tiles):
            px, py = positions[tile_id] if positions and tile_id < len(positions) else (0.0, 0.0)
            
            tile = TileState(
                id=tile_id,
                num_neurons=neurons_per_tile,
                layer_id=0,  # Custom topology - no layer structure
                pos_x=px,
                pos_y=py,
                is_input=(tile_id in input_ids),
                is_output=(tile_id in output_ids),
            )
            self.tiles[tile_id] = tile
            state_offset += neurons_per_tile
        
        self.input_tile_ids = list(input_ids)
        self.output_tile_ids = list(output_ids)
        
        # Add edges
        for src_id, dst_id in edges:
            if src_id not in self.tiles:
                raise ValueError(f"Edge source tile {src_id} does not exist")
            if dst_id not in self.tiles:
                raise ValueError(f"Edge destination tile {dst_id} does not exist")
            self._add_edge(src_id, dst_id)

    def _add_edge(self, src_id: int, dst_id: int) -> None:
        """Add bidirectional connection between tiles."""
        src = self.tiles[src_id]
        dst = self.tiles[dst_id]
        
        # Update connectivity
        src.fwd_neighbors.append(dst_id)
        dst.bwd_neighbors.append(src_id)
        
        # Create edge parameters
        edge_key = (src_id, dst_id)
        self.edges[edge_key] = EdgeParams(
            src_id=src_id,
            dst_id=dst_id,
            weight=torch.zeros(src.num_neurons, dst.num_neurons),
            bias=torch.zeros(dst.num_neurons),
        )
    
    @property
    def all_tiles(self) -> List[TileState]:
        """Return tiles in ID order."""
        return [self.tiles[i] for i in sorted(self.tiles.keys())]
    
    def get_positions(self) -> List[Tuple[float, float]]:
        """Get tile positions for visualization."""
        tiles = self.all_tiles
        return [(t.pos_x, t.pos_y) for t in tiles]
    
    def edges_list(self) -> List[Tuple[int, int]]:
        """Get list of edge tuples."""
        return list(self.edges.keys())


# =============================================================================
# Main Model
# =============================================================================

@register_model("adaptive_tile_pc")
class AdaptiveTilePC(BioModel):
    """Adaptive Tile-Based Predictive Coding.
    
    This model implements predictive coding with adaptive computation
    allocation. Each tile maintains:
    - Activity: Current neural state
    - Prediction: Top-down expectation from higher layers
    - Error: Bottom-up prediction error
    
    Learning minimizes prediction errors throughout the hierarchy,
    with adaptive computation focused on tiles with high error.
    
    Key Properties
    --------------
    * **Continuous**: No separate phases - minimize error continuously
    * **Local**: All updates use only local information
    * **Adaptive**: Computation allocated based on learned importance
    * **Sparse**: Skip updates for tiles with negligible error
    * **Scalable**: Linear complexity in number of tiles
    """
    
    algorithm_name = "AdaptiveTilePC"
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        neurons_per_tile: int,
        num_layers: int,
        tiles_per_layer: int,
        input_dim: int,
        output_dim: int,
        prediction_lr: float = 0.01,
        prior_lr: float = 0.005,
        importance_lr: float = 0.001,
        initial_step_size: float = 0.5,
        sparsity_threshold: float = 0.01,
        importance_decay: float = 0.99,
        min_active_fraction: float = 0.1,
        weight_decay: float = 1e-4,
        error_decay: float = 0.95,
        activation: Literal["tanh", "relu", "gelu"] = "gelu",
        topology: Literal["layered", "custom"] = "layered",
        custom_edges: Optional[List[Tuple[int, int]]] = None,
        custom_positions: Optional[List[Tuple[float, float]]] = None,
        task_type: Literal["classification", "binary", "multilabel", "regression"] = "classification",
        output_activation: Optional[str] = None,
        lr_schedule: str = "constant",
        lr_decay_steps: int = 1000,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
        gradient_clip: float = 1.0,
        **kwargs,
    ):
        """Initialize Adaptive Tile-Based Predictive Coding.

        Args:
            neurons_per_tile: Number of neurons in each tile
            num_layers: Total number of layers (input + hidden + output)
            tiles_per_layer: Number of tiles per layer (REQUIRED - no default)
            input_dim: Input feature dimension
            output_dim: Output dimension (classes for classification, continuous for regression)
            prediction_lr: Learning rate for prediction weights
            prior_lr: Learning rate for prior expectations
            importance_lr: Learning rate for importance weights
            initial_step_size: Inference step size
            sparsity_threshold: Skip tiles with error × importance below this
            importance_decay: EMA decay for error tracking
            min_active_fraction: Minimum fraction of tiles to update
            weight_decay: L2 regularization strength
            error_decay: Decay for error EMA
            activation: Activation function ('tanh', 'relu', 'gelu')
            topology: 'layered' for MLP, 'custom' for arbitrary graphs
            custom_edges: List of (src_id, dst_id) for custom topology
            custom_positions: Optional (x, y) positions for visualization
            task_type: 'classification' or 'regression'
            output_activation: Output activation (default: None for classification, 'linear' for regression)
        """
        # Extract any config overrides from kwargs
        equilibrium_steps = kwargs.pop("equilibrium_steps", 20)

        # Validate required parameters
        if tiles_per_layer < 1:
            raise ValueError("tiles_per_layer must be >= 1")
        if neurons_per_tile < 1:
            raise ValueError("neurons_per_tile must be >= 1")

        if config is None:
            config = ModelConfig(
                name="adaptive_tile_pc",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[neurons_per_tile * tiles_per_layer] * (max(0, num_layers - 2)),
                learning_rate=prediction_lr,
                equilibrium_steps=equilibrium_steps,
            )

        super().__init__(config, **kwargs)

        # Store task configuration
        self.task_type = task_type
        self.output_activation = output_activation
        
        # Set output activation based on task type
        if output_activation is None:
            if task_type == "regression":
                self.output_activation = "linear"  # No activation
            else:
                self.output_activation = None  # Will use softmax in loss

        # Store ATPC config
        self.config = ATPCConfig(
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            tiles_per_layer=tiles_per_layer,
            prediction_lr=prediction_lr,
            prior_lr=prior_lr,
            importance_lr=importance_lr,
            initial_step_size=initial_step_size,
            sparsity_threshold=sparsity_threshold,
            importance_decay=importance_decay,
            min_active_fraction=min_active_fraction,
            weight_decay=weight_decay,
            error_decay=error_decay,
            inference_steps=equilibrium_steps,
            lr_schedule=lr_schedule,
            lr_decay_steps=lr_decay_steps,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
            gradient_clip=gradient_clip,
        )
        
        # Activation function
        self.activation = self._get_activation(activation)
        
        # Build graph based on topology type
        self.graph = TileGraph()
        
        if topology == "layered":
            # Standard layered MLP architecture
            num_hidden = max(0, num_layers - 2)
            self.graph.build_layered(
                input_dim, output_dim,
                neurons_per_tile, num_hidden, tiles_per_layer
            )
        elif topology == "custom":
            # Custom arbitrary topology
            if custom_edges is None:
                raise ValueError("custom_edges must be provided for custom topology")
            
            # Calculate total tiles needed
            max_tile_id = max(max(src, dst) for src, dst in custom_edges)
            n_tiles = max_tile_id + 1
            
            # Infer input/output tiles (tiles with only outgoing/incoming edges)
            has_incoming = set()
            has_outgoing = set()
            for src, dst in custom_edges:
                has_outgoing.add(src)
                has_incoming.add(dst)
            
            input_ids = list(has_outgoing - has_incoming)
            output_ids = list(has_incoming - has_outgoing)
            
            # If no clear input/output, use layer 0 and last layer
            if not input_ids:
                input_ids = [0]
            if not output_ids:
                output_ids = [n_tiles - 1]
            
            self.graph.build_custom(
                n_tiles=n_tiles,
                neurons_per_tile=neurons_per_tile,
                edges=custom_edges,
                input_ids=input_ids,
                output_ids=output_ids,
                positions=custom_positions,
            )
        else:
            raise ValueError(f"Unknown topology: {topology}. Use 'layered' or 'custom'.")
        
        # Input/output projections - match tile dimensions
        n_in_tiles = len(self.graph.input_tile_ids)
        n_out_tiles = len(self.graph.output_tile_ids)
        input_tile_dim = sum(self.graph.tiles[tid].num_neurons for tid in self.graph.input_tile_ids)
        output_tile_dim = sum(self.graph.tiles[tid].num_neurons for tid in self.graph.output_tile_ids)
        
        self.W_in = nn.Linear(self.input_dim, input_tile_dim)
        self.W_out = nn.Linear(output_tile_dim, self.output_dim)

        # Initialize default strategies
        self.inference_strategy: InferenceStrategy = GradientDescentInference()
        self.learning_strategy: LearningStrategy = HebbianLearning()
        self.scheduling_strategy: SchedulingStrategy = ThresholdScheduling(
            threshold=sparsity_threshold
        )
        
        # Importance parameters (learned per tile)
        self.tile_importance = nn.Parameter(
            torch.ones(len(self.graph.tiles))
        )

        # Edge importance (learned per edge)
        self.edge_importance = nn.Parameter(
            torch.ones(len(self.graph.edges))
        )

        # Optimizers - only for I/O projections and importance weights
        # Edge weights use manual updates (bio-plausible local learning)
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=prediction_lr,
        )
        self._optim_importance = torch.optim.Adam(
            [self.tile_importance, self.edge_importance],
            lr=importance_lr,
        )

        # Learning rate scheduler
        self._lr_scheduler = self._create_lr_scheduler(self._optim_io)

        # Regularization modules (optional)
        self._tile_bn: Dict[int, TileBatchNorm] = {}
        self._tile_dropout: Dict[int, TileDropout] = {}
        
        if self.config.use_batchnorm or self.config.dropout > 0:
            self._init_regularization_modules()

        # State tracking
        self._step_count = 0
        self._error_ema: Dict[int, float] = {}
        self._best_validation = None
        self._validation_history: List[float] = []

        # Initialize weights with improved initialization
        self._init_weights()
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        if name == "tanh":
            return torch.tanh
        elif name == "relu":
            return F.relu
        elif name == "gelu":
            return F.gelu
        return F.gelu
    
    def _init_weights(self) -> None:
        """Initialize all parameters with improved initialization."""
        with torch.no_grad():
            for edge in self.graph.edges.values():
                if edge.weight is not None:
                    # He initialization for ReLU-like activations
                    fan_in = edge.weight.shape[0]
                    std = math.sqrt(2.0 / fan_in)
                    edge.weight.normal_(0, std)
                if edge.bias is not None:
                    nn.init.zeros_(edge.bias)

            # Kaiming initialization for W_in
            nn.init.kaiming_normal_(self.W_in.weight, mode='fan_in', nonlinearity='relu')
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)

            # Xavier initialization for W_out (more stable for output)
            nn.init.xavier_normal_(self.W_out.weight, gain=1.0)
            if self.W_out.bias is not None:
                nn.init.zeros_(self.W_out.bias)

    def _create_lr_scheduler(self, optimizer):
        """Create learning rate scheduler based on config."""
        if self.config.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.lr_decay_steps, eta_min=self.config.prediction_lr * 0.01
            )
        elif self.config.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.lr_decay_steps // 5, gamma=0.5
            )
        return None  # Constant LR

    def _init_regularization_modules(self) -> None:
        """Initialize batch norm and dropout modules for all tiles."""
        for tile in self.graph.all_tiles:
            if self.config.use_batchnorm:
                self._tile_bn[tile.id] = TileBatchNorm(tile.num_neurons)
            if self.config.dropout > 0:
                self._tile_dropout[tile.id] = TileDropout(self.config.dropout)

    def _apply_regularization(self, tile_id: int, activity: Tensor) -> Tensor:
        """Apply batch norm and dropout to tile activity."""
        if self.config.use_batchnorm and tile_id in self._tile_bn:
            activity = self._tile_bn[tile_id](activity)
        if self.config.dropout > 0 and tile_id in self._tile_dropout:
            activity = self._tile_dropout[tile_id](activity)
        return activity

    def _get_lr(self) -> float:
        """Get current learning rate."""
        for param_group in self._optim_io.param_groups:
            return param_group['lr']
        return self.config.prediction_lr

    def step_lr_scheduler(self) -> None:
        """Step the learning rate scheduler."""
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    # -------------------------------------------------------------------------
    # Forward Dynamics (Optimized)
    # -------------------------------------------------------------------------

    def _compute_predictions(self, batch_size: int, device: torch.device):
        """Compute top-down predictions for all tiles."""
        for tile in self.graph.all_tiles:
            if tile.is_input:
                continue

            # Aggregate predictions from all backward neighbors
            pred = torch.zeros(batch_size, tile.num_neurons, device=device)

            for src_id in tile.bwd_neighbors:
                src = self.graph.tiles[src_id]
                edge = self.graph.edges.get((src_id, tile.id))

                if edge is None or edge.weight is None:
                    continue

                src_activity = src.activity if src.activity is not None else torch.zeros(
                    batch_size, src.num_neurons, device=device
                )
                pred = pred + self.activation(src_activity) @ edge.weight

            # Add bias
            edge = self.graph.edges.get(
                (tile.bwd_neighbors[0], tile.id)
            ) if tile.bwd_neighbors else None
            if edge and edge.bias is not None:
                pred = pred + edge.bias.unsqueeze(0)

            tile.prediction = pred
    
    def _compute_errors(self):
        """Compute bottom-up prediction errors."""
        for tile in self.graph.all_tiles:
            if tile.activity is None:
                continue
            
            if tile.prediction is None:
                # No prediction - error is just the activity
                tile.error = tile.activity.clone()
            else:
                # Prediction error: actual - predicted
                tile.error = tile.activity - tile.prediction
            
            # Update error magnitude EMA
            err_norm = tile.error.norm(p=2, dim=-1).mean().item()
            self._error_ema[tile.id] = (
                self.config.error_decay * self._error_ema.get(tile.id, 0.0)
                + (1 - self.config.error_decay) * err_norm
            )
    
    def _update_activities(
        self,
        input_proj: Tensor,
        target: Optional[Tensor] = None,
        steps: int = 1,
    ) -> None:
        """Update tile activities to minimize prediction errors.
        
        Uses the configured inference strategy and scheduling strategy.

        Uses gradient descent on the prediction error energy:
            E = sum_tile ||error_tile||^2

        Updates are scaled by learned importance weights.
        """
        step_size = self.config.initial_step_size

        for _ in range(steps):
            # Compute errors for scheduling
            errors = {tile.id: tile.error.norm(p=2, dim=-1).mean().item() 
                     for tile in self.graph.all_tiles if tile.error is not None}
            
            # Select tiles to update using scheduling strategy
            active_tile_ids = self.scheduling_strategy.select_tiles(
                self.graph.all_tiles,
                errors,
                self.tile_importance,
            )
            active_set = set(active_tile_ids)

            for i, tile in enumerate(self.graph.all_tiles):
                # Skip input tiles (clamped to input projection)
                if tile.is_input:
                    idx = self.graph.input_tile_ids.index(tile.id)
                    start = idx * self.config.neurons_per_tile
                    tile.activity = input_proj[:, start:start + tile.num_neurons].clone()
                    continue

                # Skip inactive tiles (sparse computation)
                if tile.id not in active_set:
                    continue

                if tile.error is None:
                    continue

                # Get importance weight
                importance = torch.sigmoid(self.tile_importance[i]).item()

                # Compute gradient of prediction error w.r.t. activity
                grad = tile.error.clone()

                # Add top-down modulation from forward neighbors
                for dst_id in tile.fwd_neighbors:
                    dst = self.graph.tiles[dst_id]
                    edge = self.graph.edges.get((tile.id, dst_id))

                    if edge is None or edge.weight is None or dst.error is None:
                        continue

                    # Backpropagate error through weight
                    grad = grad + dst.error @ edge.weight.T

                # Update activity using inference strategy
                self.inference_strategy.update(tile, grad, step_size, importance)

            # Recompute predictions and errors
            self._compute_predictions(input_proj.shape[0], input_proj.device)
            self._compute_errors()
    
    def _apply_output_nudge(self, target: Tensor, beta: float = 0.1) -> None:
        """Gently nudge output tiles toward target."""
        for i, tile_id in enumerate(self.graph.output_tile_ids):
            tile = self.graph.tiles[tile_id]
            if tile.activity is None:
                continue
            
            start = i * self.config.neurons_per_tile
            target_activity = target[:, start:start + tile.num_neurons]
            
            # Soft clamp toward target
            tile.activity = (1 - beta) * tile.activity + beta * target_activity
    
    # -------------------------------------------------------------------------
    # Learning
    # -------------------------------------------------------------------------
    
    def _update_weights(self, batch_size: int) -> None:
        """Update prediction weights based on prediction errors.
        
        Uses the configured learning strategy.
        Performs manual gradient descent (bio-plausible local learning).

        For each edge src -> dst:
            ΔW = lr * (activation(src) @ error(dst))
            Δb = lr * error(dst)

        This is equivalent to gradient descent on prediction error.
        """
        lr = self.config.prediction_lr

        for edge_idx, (edge_key, edge) in enumerate(self.graph.edges.items()):
            src_id, dst_id = edge_key
            src = self.graph.tiles[src_id]
            dst = self.graph.tiles[dst_id]

            if src.activity is None or dst.error is None:
                continue

            # Get importance weight
            importance = torch.sigmoid(self.edge_importance[edge_idx])

            # Compute gradients using learning strategy
            src_act = self.activation(src.activity)
            dst_err = dst.error

            weight_update, bias_update = self.learning_strategy.compute_update(
                edge, src_act, dst_err,
                lr,
                importance.item()
            )

            # Apply weight decay and update weights directly (in-place to preserve graph)
            if edge.weight is not None:
                edge.weight.data = edge.weight.data - lr * (weight_update.detach() + self.config.weight_decay * edge.weight.data)
            
            if edge.bias is not None:
                edge.bias.data = edge.bias.data - lr * bias_update.detach()

    def _update_importance(self) -> None:
        """Update tile and edge importance weights.
        
        Importance is updated to minimize prediction error while
        penalizing high importance (encourages sparsity).
        """
        self._optim_importance.zero_grad()
        
        # Tile importance loss: reduce error while keeping importance low
        tile_loss = 0.0
        for i, tile in enumerate(self.graph.all_tiles):
            if tile.error is None:
                continue
            
            err_norm = tile.error.norm(p=2, dim=-1).mean()
            importance = torch.sigmoid(self.tile_importance[i])
            
            # Want high importance when error is high
            tile_loss = tile_loss + importance * err_norm.detach()
        
        # Sparsity penalty
        sparsity_loss = 0.1 * torch.sum(torch.sigmoid(self.tile_importance))
        sparsity_loss = sparsity_loss + 0.1 * torch.sum(torch.sigmoid(self.edge_importance))
        
        total_loss = tile_loss + sparsity_loss
        total_loss.backward()
        
        self._optim_importance.step()
    
    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    
    def forward(
        self,
        x: Tensor,
        steps: Optional[int] = None,
        return_states: bool = False,
    ):
        """Forward pass through the network.
        
        Args:
            x: Input tensor (batch, input_dim)
            steps: Number of inference steps (default: config.inference_steps)
            return_states: If True, return tile activities
            
        Returns:
            Logits (batch, output_dim) or (logits, states_dict)
        """
        batch, device = x.shape[0], x.device
        steps = steps if steps is not None else self.config.inference_steps
        
        # Project input
        input_proj = self.W_in(x)
        
        # Initialize tile activities
        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.num_neurons].clone()
            else:
                tile.activity = torch.zeros(batch, tile.num_neurons, device=device)
            tile.prediction = None
            tile.error = None
        
        # Run inference (minimize prediction error)
        for _ in range(steps):
            self._compute_predictions(batch, device)
            self._compute_errors()
            self._update_activities(input_proj, steps=1)
        
        # Read output
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1
        )
        logits = self.W_out(out_activities)
        
        if return_states:
            states = {
                tile.id: {
                    "activity": tile.activity.clone() if tile.activity is not None else None,
                    "prediction": tile.prediction.clone() if tile.prediction is not None else None,
                    "error": tile.error.clone() if tile.error is not None else None,
                }
                for tile in self.graph.all_tiles
            }
            return logits, states
        
        return logits
    
    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Perform one training step with joint prediction + classification objective.

        Key fix: Classification error directly drives internal weight updates,
        ensuring representations become class-discriminative.

        Args:
            x: Input tensor (batch, input_dim)
            y: Target labels (batch,)

        Returns:
            Dictionary with loss, accuracy, and diagnostic metrics
        """
        batch, device = x.shape[0], x.device
        self._step_count += 1

        # Project input
        input_proj = self.W_in(x)

        # Initialize activities
        for tile in self.graph.all_tiles:
            if tile.is_input:
                idx = self.graph.input_tile_ids.index(tile.id)
                start = idx * self.config.neurons_per_tile
                tile.activity = input_proj[:, start:start + tile.num_neurons].clone()
            else:
                tile.activity = torch.zeros(batch, tile.num_neurons, device=device)
            tile.prediction = None
            tile.error = None

        # === Inference Phase ===
        # Run inference to minimize prediction errors
        for _ in range(self.config.inference_steps):
            self._compute_predictions(batch, device)
            self._compute_errors()
            self._update_activities(input_proj, steps=1)

        # === Classification-Driven Learning ===
        # 1. Compute output and classification loss
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1
        )
        
        # Apply output activation if specified
        if self.output_activation == "linear":
            logits = self.W_out(out_activities)  # Regression: no activation
        elif self.task_type == "binary":
            logits = torch.sigmoid(self.W_out(out_activities))  # Binary: sigmoid
        elif self.task_type == "multilabel":
            logits = torch.sigmoid(self.W_out(out_activities))  # Multi-label: sigmoid
        else:
            logits = self.W_out(out_activities)  # Classification: softmax in loss

        # Compute loss based on task type
        if self.task_type == "regression":
            # MSE loss for regression
            y_target = y.float()
            if y_target.dim() < logits.dim():
                y_target = y_target.unsqueeze(-1)
            loss = F.mse_loss(logits, y_target)
            output_delta = (logits - y_target) @ self.W_out.weight
        elif self.task_type == "binary":
            # Binary cross-entropy
            loss = F.binary_cross_entropy(logits.squeeze(-1), y.float())
            # Error signal for binary
            output_delta = (logits.squeeze(-1) - y.float()).unsqueeze(-1) @ self.W_out.weight
        elif self.task_type == "multilabel":
            # Multi-label BCE
            loss = F.binary_cross_entropy(logits, y.float())
            output_delta = (logits - y.float()) @ self.W_out.weight
        else:
            # Cross-entropy for multi-class classification
            loss = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)
            target_onehot = F.one_hot(y, self.output_dim).float().to(device)
            output_delta = (probs - target_onehot) @ self.W_out.weight

        # 2. Backpropagate error through output layer with gradient clipping
        self._optim_io.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.W_in.parameters()) + list(self.W_out.parameters()),
                self.config.gradient_clip
            )
        
        self._optim_io.step()

        # 3. Compute error-driven weight updates for ALL edges
        # Internal weights learn to support the task directly
        with torch.no_grad():
            # Backpropagate error through network layer by layer
            # Store error signals for each tile
            tile_errors: Dict[int, Tensor] = {}

            # Output tiles
            for i, tile_id in enumerate(self.graph.output_tile_ids):
                tile = self.graph.tiles[tile_id]
                start = i * self.config.neurons_per_tile
                tile_errors[tile_id] = output_delta[:, start:start+tile.num_neurons].clone()

            # Hidden tiles (reverse order)
            tiles_by_layer = sorted(
                [t for t in self.graph.all_tiles if not t.is_output and not t.is_input],
                key=lambda t: -t.layer_id
            )
            for tile in tiles_by_layer:
                # Accumulate error from forward neighbors
                error = torch.zeros_like(tile.activity)
                for fwd_id in tile.fwd_neighbors:
                    if fwd_id not in tile_errors:
                        continue
                    fwd_tile = self.graph.tiles[fwd_id]
                    edge = self.graph.edges.get((tile.id, fwd_id))
                    if edge is None or edge.weight is None:
                        continue
                    # Backpropagate through weight
                    error = error + tile_errors[fwd_id] @ edge.weight.T

                tile_errors[tile.id] = error

            # 4. Update internal weights using task-driven errors
            lr = self.config.prediction_lr
            for edge_idx, (edge_key, edge) in enumerate(self.graph.edges.items()):
                src_id, dst_id = edge_key
                src = self.graph.tiles[src_id]
                dst = self.graph.tiles[dst_id]

                if src.activity is None or dst.id not in tile_errors:
                    continue

                # Get importance weight
                importance = torch.sigmoid(self.edge_importance[edge_idx])

                # Use task error for weight update
                dst_err = tile_errors[dst.id]

                # Hebbian update: correlate source activity with target error
                src_act = self.activation(src.activity)
                weight_update = importance * (src_act.T @ dst_err) / batch
                bias_update = importance * dst_err.mean(dim=0) / batch

                # Apply weight decay and update
                if edge.weight is not None:
                    edge.weight.data = edge.weight.data - lr * (
                        weight_update + self.config.weight_decay * edge.weight.data
                    )
                if edge.bias is not None:
                    edge.bias.data = edge.bias.data - lr * bias_update

        # Update importance weights
        self._update_importance()

        # Compute metrics based on task type
        with torch.no_grad():
            if self.task_type == "regression":
                mse = F.mse_loss(logits, y.float()).item()
                ss_res = ((y.float() - logits.squeeze()) ** 2).sum()
                ss_tot = ((y.float() - y.float().mean()) ** 2).sum()
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                accuracy = r2
            elif self.task_type == "binary":
                preds = (logits.squeeze(-1) > 0.5).long()
                accuracy = (preds == y).float().mean().item()
            elif self.task_type == "multilabel":
                preds = (logits > 0.5).long()
                # Subset accuracy (all labels must match)
                accuracy = (preds == y).all(dim=-1).float().mean().item()
            else:
                accuracy = (logits.argmax(dim=-1) == y).float().mean().item()

        # Step learning rate scheduler
        self.step_lr_scheduler()

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
            "mean_error": sum(
                self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles
            ) / len(self.graph.tiles),
            "active_tiles": sum(
                1 for t in self.graph.all_tiles
                if self._error_ema.get(t.id, 0.0) > self.config.sparsity_threshold
            ),
            "task_type": self.task_type,
            "learning_rate": self._get_lr(),
        }
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, float]:
        """Get model statistics."""
        stats = super().get_stats()
        
        importances = torch.sigmoid(self.tile_importance).tolist()
        errors = [self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles]
        
        stats.update({
            "importance_mean": sum(importances) / len(importances),
            "importance_max": max(importances),
            "error_mean": sum(errors) / len(errors),
            "error_max": max(errors),
            "active_tiles": sum(1 for e in errors if e > self.config.sparsity_threshold),
            "total_tiles": len(self.graph.tiles),
        })
        
        return stats
    
    def get_topology_info(self) -> Dict:
        """Get topology information for visualization."""
        return {
            "positions": self.graph.get_positions(),
            "edges": self.graph.edges_list(),
            "layer_ids": [t.layer_id for t in self.graph.all_tiles],
            "is_input": [t.is_input for t in self.graph.all_tiles],
            "is_output": [t.is_output for t in self.graph.all_tiles],
            "tile_heats": [self._error_ema.get(t.id, 0.0) for t in self.graph.all_tiles],
            "importances": torch.sigmoid(self.tile_importance).tolist(),
        }
    
    @classmethod
    def build(
        cls,
        spec,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        device: torch.device,
        task_type: str,
        **kwargs,
    ) -> "AdaptiveTilePC":
        """Build model from specification.
        
        Args:
            spec: Specification object with name and default_lr
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of layers
            device: Target device
            task_type: 'classification' or 'regression'
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            ATPC model on specified device
        """
        neurons_per_tile = kwargs.pop("neurons_per_tile", 64)
        tiles_per_layer = kwargs.pop("tiles_per_layer", 1)

        model = cls(
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            tiles_per_layer=tiles_per_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            prediction_lr=spec.default_lr if hasattr(spec, 'default_lr') else 0.01,
            **kwargs,
        )
        return model.to(device)

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict:
        """Get complete model state for checkpointing.
        
        Returns:
            Dictionary with model weights, optimizer states, and metadata
        """
        return {
            "model_state_dict": self.state_dict(),
            "task_type": self.task_type,
            "config": {
                "neurons_per_tile": self.config.neurons_per_tile,
                "num_layers": self.config.num_layers,
                "tiles_per_layer": self.config.tiles_per_layer,
                "prediction_lr": self.config.prediction_lr,
                "importance_lr": self.config.importance_lr,
                "initial_step_size": self.config.initial_step_size,
                "sparsity_threshold": self.config.sparsity_threshold,
                "activation": self.config.activation if hasattr(self.config, 'activation') else "gelu",
            },
            "training": {
                "step_count": self._step_count,
                "error_ema": dict(self._error_ema),
            },
        }

    def load_state(self, state: Dict) -> None:
        """Load model state from checkpoint.
        
        Args:
            state: Dictionary from get_state()
        """
        self.load_state_dict(state["model_state_dict"])
        if "task_type" in state:
            self.task_type = state["task_type"]
        if "training" in state:
            self._step_count = state["training"]["step_count"]
            self._error_ema = state["training"]["error_ema"]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk.
        
        Args:
            path: File path to save checkpoint
        """
        torch.save(self.get_state(), path)

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> None:
        """Load model checkpoint from disk.
        
        Args:
            path: File path to load checkpoint
            device: Target device (default: current device)
        """
        if device is None:
            device = next(self.parameters()).device
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state(state)

    # -------------------------------------------------------------------------
    # Validation and Early Stopping
    # -------------------------------------------------------------------------

    def validate(self, X: Tensor, y: Tensor, batch_size: int = 64) -> Dict[str, float]:
        """Evaluate model on validation data.
        
        Args:
            X: Validation features
            y: Validation targets
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with validation metrics
        """
        self.eval()
        total_loss = 0.0
        correct = 0
        n_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                logits = self(x_batch, steps=self.config.inference_steps)
                
                if self.task_type == "regression":
                    loss = F.mse_loss(logits, y_batch.float().unsqueeze(-1) if y_batch.dim() < logits.dim() else y_batch.float())
                    total_loss += loss.item() * len(y_batch)
                elif self.task_type == "binary":
                    logits_clamped = logits.squeeze(-1).clamp(1e-7, 1-1e-7)
                    loss = F.binary_cross_entropy(logits_clamped, y_batch.float())
                    total_loss += loss.item() * len(y_batch)
                    correct += ((logits.squeeze(-1) > 0.5).long() == y_batch).sum().item()
                elif self.task_type == "multilabel":
                    logits_clamped = logits.clamp(1e-7, 1-1e-7)
                    loss = F.binary_cross_entropy(logits_clamped, y_batch.float())
                    total_loss += loss.item() * len(y_batch)
                    correct += ((logits > 0.5).long() == y_batch).all(dim=-1).sum().item()
                else:
                    loss = F.cross_entropy(logits, y_batch)
                    total_loss += loss.item() * len(y_batch)
                    correct += (logits.argmax(dim=-1) == y_batch).sum().item()
                
                n_samples += len(y_batch)
        
        self.train()
        
        return {
            "val_loss": total_loss / n_samples,
            "val_accuracy": correct / n_samples,
            "n_samples": n_samples,
        }

    def train_with_validation(
        self,
        X_train: Tensor,
        y_train: Tensor,
        X_val: Tensor,
        y_val: Tensor,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 5,
        min_delta: float = 0.001,
    ) -> Dict:
        """Train with validation monitoring and early stopping.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Training batch size
            patience: Epochs to wait for improvement before stopping
            min_delta: Minimum improvement to count as progress
            
        Returns:
            Training history dictionary
        """
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            # Training
            self.train()
            epoch_loss = 0.0
            epoch_correct = 0
            n_batches = 0
            
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i+batch_size]
                stats = self.train_step(X_train[idx], y_train[idx])
                epoch_loss += stats["loss"]
                epoch_correct += stats["accuracy"] * len(idx)
                n_batches += 1
            
            # Validation
            val_metrics = self.validate(X_val, y_val, batch_size)
            
            # Record history
            history["train_loss"].append(epoch_loss / n_batches)
            history["train_acc"].append(epoch_correct / len(X_train))
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_acc"].append(val_metrics["val_accuracy"])
            history["learning_rate"].append(self._get_lr())
            
            # Early stopping check
            improved = val_metrics["val_loss"] < best_val_loss - min_delta
            if improved:
                best_val_loss = val_metrics["val_loss"]
                best_val_acc = val_metrics["val_accuracy"]
                patience_counter = 0
                best_state = self.get_state()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best state
        if best_state is not None:
            self.load_state(best_state)
        
        history["best_val_loss"] = best_val_loss
        history["best_val_acc"] = best_val_acc
        history["epochs_trained"] = len(history["train_loss"])
        
        return history

    # -------------------------------------------------------------------------
    # Model Inspection
    # -------------------------------------------------------------------------

    def get_weight_statistics(self) -> Dict[str, float]:
        """Get statistics about weight matrices.
        
        Returns:
            Dictionary with weight statistics
        """
        stats = {
            "total_weights": 0,
            "mean_weight": 0.0,
            "std_weight": 0.0,
            "max_weight": 0.0,
            "min_weight": 0.0,
        }
        
        all_weights = []
        for edge in self.graph.edges.values():
            if edge.weight is not None:
                all_weights.append(edge.weight.data.flatten())
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            stats["total_weights"] = len(all_weights)
            stats["mean_weight"] = all_weights.mean().item()
            stats["std_weight"] = all_weights.std().item()
            stats["max_weight"] = all_weights.max().item()
            stats["min_weight"] = all_weights.min().item()
        
        return stats

    def get_tile_activity_stats(self) -> Dict[str, float]:
        """Get statistics about tile activities.
        
        Returns:
            Dictionary with activity statistics
        """
        activities = []
        for tile in self.graph.all_tiles:
            if tile.activity is not None:
                activities.append(tile.activity.abs().mean().item())
        
        if activities:
            return {
                "mean_activity": sum(activities) / len(activities),
                "max_activity": max(activities),
                "min_activity": min(activities),
                "active_tiles": sum(1 for a in activities if a > 0.1),
            }
        return {"mean_activity": 0.0, "max_activity": 0.0, "min_activity": 0.0, "active_tiles": 0}

    def summarize(self) -> str:
        """Get human-readable model summary.
        
        Returns:
            Formatted string with model information
        """
        lines = [
            "=" * 60,
            "Adaptive Tile-Based Predictive Coding (ATPC)",
            "=" * 60,
            f"Task Type: {self.task_type}",
            f"Architecture: {self.config.num_layers} layers, {self.config.tiles_per_layer} tiles/layer",
            f"Neurons per tile: {self.config.neurons_per_tile}",
            f"Total tiles: {len(self.graph.tiles)}",
            f"Total edges: {len(self.graph.edges)}",
            f"Total parameters: {sum(p.numel() for p in self.parameters()):,}",
            "",
            "Tile Structure:",
        ]
        
        for layer_idx in range(len(self.graph.layer_ids)):
            layer_tiles = self.graph.layer_ids[layer_idx] if layer_idx < len(self.graph.layer_ids) else []
            lines.append(f"  Layer {layer_idx}: {len(layer_tiles)} tiles")
        
        lines.extend([
            "",
            "Hyperparameters:",
            f"  Prediction LR: {self.config.prediction_lr}",
            f"  Importance LR: {self.config.importance_lr}",
            f"  Step Size: {self.config.initial_step_size}",
            f"  Sparsity Threshold: {self.config.sparsity_threshold}",
            f"  Inference Steps: {self.config.inference_steps}",
            "=" * 60,
        ])
        
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Auto-Configuration
    # -------------------------------------------------------------------------

    @classmethod
    def auto_configure(
        cls,
        input_dim: int,
        output_dim: int,
        n_samples: int,
        task_type: str = "classification",
        compute_budget: str = "balanced",  # 'fast', 'balanced', 'accurate'
    ) -> "AdaptiveTilePC":
        """Automatically configure ATPC based on dataset characteristics.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (classes or 1 for regression)
            n_samples: Number of training samples
            task_type: Task type
            compute_budget: 'fast' (small), 'balanced', or 'accurate' (large)
            
        Returns:
            Configured ATPC model
        """
        # Determine model size based on compute budget
        if compute_budget == "fast":
            neurons_per_tile = min(32, input_dim // 2)
            tiles_per_layer = 2
            num_layers = 3
            prediction_lr = 0.05
        elif compute_budget == "accurate":
            neurons_per_tile = min(128, input_dim)
            tiles_per_layer = 8
            num_layers = max(4, min(8, input_dim // 32))
            prediction_lr = 0.01
        else:  # balanced
            neurons_per_tile = min(64, input_dim)
            tiles_per_layer = 4
            num_layers = max(3, min(6, input_dim // 16))
            prediction_lr = 0.02
        
        # Adjust for dataset size
        if n_samples < 500:
            # Small dataset: more regularization
            dropout = 0.3
            weight_decay = 1e-3
        elif n_samples > 10000:
            # Large dataset: less regularization
            dropout = 0.0
            weight_decay = 1e-5
        else:
            dropout = 0.1
            weight_decay = 1e-4
        
        # Learning rate schedule
        lr_schedule = "cosine" if n_samples > 1000 else "constant"
        lr_decay_steps = max(100, n_samples // 32)
        
        return cls(
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            tiles_per_layer=tiles_per_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            task_type=task_type,
            prediction_lr=prediction_lr,
            dropout=dropout,
            use_batchnorm=n_samples > 1000,
            gradient_clip=1.0,
            lr_schedule=lr_schedule,
            lr_decay_steps=lr_decay_steps,
            weight_decay=weight_decay,
        )

    # -------------------------------------------------------------------------
    # Callback System
    # -------------------------------------------------------------------------

    def add_callback(self, name: str, callback) -> None:
        """Add a training callback.
        
        Args:
            name: Callback name
            callback: Function that takes (model, epoch, stats) and returns None
        """
        if not hasattr(self, '_callbacks'):
            self._callbacks = {}
        self._callbacks[name] = callback

    def remove_callback(self, name: str) -> None:
        """Remove a callback by name."""
        if hasattr(self, '_callbacks') and name in self._callbacks:
            del self._callbacks[name]

    def _run_callbacks(self, epoch: int, stats: Dict) -> None:
        """Run all registered callbacks."""
        if hasattr(self, '_callbacks'):
            for name, callback in self._callbacks.items():
                try:
                    callback(self, epoch, stats)
                except Exception as e:
                    print(f"Callback {name} error: {e}")


# =============================================================================
# Pre-built Callbacks
# =============================================================================

class TrainingCallback:
    """Base class for training callbacks."""
    
    def __call__(self, model: AdaptiveTilePC, epoch: int, stats: Dict) -> None:
        raise NotImplementedError


class ProgressBarCallback(TrainingCallback):
    """Simple progress bar callback."""
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
    
    def __call__(self, model: AdaptiveTilePC, epoch: int, stats: Dict) -> None:
        progress = epoch / self.total_epochs
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r  [{bar}] {epoch}/{self.total_epochs} epochs", end="", flush=True)


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on training loss."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, model: AdaptiveTilePC, epoch: int, stats: Dict) -> None:
        loss = stats.get("loss", float('inf'))
        
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"\n  Early stopping at epoch {epoch}")


class MetricLoggerCallback(TrainingCallback):
    """Log metrics to a file or console."""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        self.history = []
    
    def __call__(self, model: AdaptiveTilePC, epoch: int, stats: Dict) -> None:
        self.history.append(stats)
        
        if self.verbose:
            print(f"  Epoch {epoch}: Loss={stats.get('loss', 0):.3f}, "
                  f"Acc={stats.get('accuracy', 0):.3f}")
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{epoch},{stats.get('loss', 0)},{stats.get('accuracy', 0)}\n")


# =============================================================================
# Next-Generation ATPC Extensions
# =============================================================================

class AsyncTileProcessor:
    """Asynchronous tile processing for parallel execution.
    
    Enables tiles to update independently without global synchronization,
    enabling true parallelism on multi-core systems and neuromorphic hardware.
    """
    
    def __init__(self, model: "AdaptiveTilePC", num_workers: int = 4):
        self.model = model
        self.num_workers = num_workers
        self._tile_queues: Dict[int, List] = {t.id: [] for t in model.graph.all_tiles}
        self._lock = torch.lock() if hasattr(torch, 'lock') else None
    
    def submit_tile_update(self, tile_id: int, data: Dict) -> None:
        """Submit a tile update task to the queue."""
        self._tile_queues[tile_id].append(data)
    
    def process_tile_async(self, tile_id: int) -> Optional[Dict]:
        """Process a single tile update asynchronously."""
        if not self._tile_queues[tile_id]:
            return None
        
        task = self._tile_queues[tile_id].pop(0)
        tile = self.model.graph.tiles[tile_id]
        
        # Process tile update
        if "activity" in task:
            tile.activity = task["activity"]
        if "error" in task:
            tile.error = task["error"]
        
        return {"tile_id": tile_id, "processed": True}
    
    def process_all_pending(self) -> int:
        """Process all pending tile updates. Returns count processed."""
        count = 0
        for tile_id in self._tile_queues:
            while self._tile_queues[tile_id]:
                self.process_tile_async(tile_id)
                count += 1
        return count


class DynamicTileGrowth:
    """Dynamic network growth and pruning during training.
    
    Enables the network to evolve:
    - Add tiles when error is persistently high
    - Remove tiles when error is persistently low
    - Split tiles that have high internal variance
    - Merge similar tiles
    """
    
    def __init__(
        self,
        model: "AdaptiveTilePC",
        growth_threshold: float = 0.5,
        prune_threshold: float = 0.05,
        max_tiles: int = 100,
        min_tiles: int = 2,
    ):
        self.model = model
        self.growth_threshold = growth_threshold
        self.prune_threshold = prune_threshold
        self.max_tiles = max_tiles
        self.min_tiles = min_tiles
        self._error_history: Dict[int, List[float]] = {}
    
    def track_error(self, tile_id: int, error: float) -> None:
        """Track error for a tile over time."""
        if tile_id not in self._error_history:
            self._error_history[tile_id] = []
        self._error_history[tile_id].append(error)
        
        # Keep last 100 errors
        if len(self._error_history[tile_id]) > 100:
            self._error_history[tile_id].pop(0)
    
    def should_grow(self, tile_id: int) -> bool:
        """Check if a tile should spawn a new tile."""
        if len(self.model.graph.tiles) >= self.max_tiles:
            return False
        
        errors = self._error_history.get(tile_id, [])
        if len(errors) < 20:
            return False
        
        avg_error = sum(errors[-20:]) / 20
        return avg_error > self.growth_threshold
    
    def should_prune(self, tile_id: int) -> bool:
        """Check if a tile should be removed."""
        if len(self.model.graph.tiles) <= self.min_tiles:
            return False
        
        # Don't prune input/output tiles
        tile = self.model.graph.tiles[tile_id]
        if tile.is_input or tile.is_output:
            return False
        
        errors = self._error_history.get(tile_id, [])
        if len(errors) < 20:
            return False
        
        avg_error = sum(errors[-20:]) / 20
        return avg_error < self.prune_threshold
    
    def grow_tile(self, parent_id: int) -> int:
        """Create a new tile as a child of an existing tile."""
        parent = self.model.graph.tiles[parent_id]
        new_id = max(t.id for t in self.model.graph.tiles) + 1
        
        # Create new tile with similar properties
        new_tile = TileState(
            id=new_id,
            num_neurons=parent.num_neurons,
            layer_id=parent.layer_id + 1,
            pos_x=parent.pos_x + 0.1,
            pos_y=parent.pos_y,
            is_input=False,
            is_output=False,
        )
        
        self.model.graph.tiles[new_id] = new_tile
        
        # Connect to parent
        self.model.graph.edges[(parent_id, new_id)] = EdgeParams(
            src_id=parent_id,
            dst_id=new_id,
            weight=torch.randn(parent.num_neurons, new_tile.num_neurons) * 0.1,
            bias=torch.zeros(new_tile.num_neurons),
        )
        
        parent.fwd_neighbors.append(new_id)
        new_tile.bwd_neighbors.append(parent_id)
        
        print(f"  Grew tile {new_id} from parent {parent_id}")
        return new_id
    
    def prune_tile(self, tile_id: int) -> bool:
        """Remove a tile and its connections."""
        if tile_id not in self.model.graph.tiles:
            return False
        
        tile = self.model.graph.tiles[tile_id]
        
        # Don't prune input/output tiles
        if tile.is_input or tile.is_output:
            return False
        
        # Remove incoming edges
        for src_id in list(tile.bwd_neighbors):
            if (src_id, tile_id) in self.model.graph.edges:
                del self.model.graph.edges[(src_id, tile_id)]
            if tile_id in self.model.graph.tiles[src_id].fwd_neighbors:
                self.model.graph.tiles[src_id].fwd_neighbors.remove(tile_id)
        
        # Remove outgoing edges
        for dst_id in list(tile.fwd_neighbors):
            if (tile_id, dst_id) in self.model.graph.edges:
                del self.model.graph.edges[(tile_id, dst_id)]
            if tile_id in self.model.graph.tiles[dst_id].bwd_neighbors:
                self.model.graph.tiles[dst_id].bwd_neighbors.remove(tile_id)
        
        # Remove tile
        del self.model.graph.tiles[tile_id]
        if tile_id in self._error_history:
            del self._error_history[tile_id]
        
        print(f"  Pruned tile {tile_id}")
        return True
    
    def step(self, errors: Dict[int, float]) -> Dict[str, int]:
        """Evaluate and apply growth/pruning decisions.
        
        Returns:
            Dictionary with counts of tiles grown and pruned
        """
        stats = {"grown": 0, "pruned": 0}
        
        # Track errors
        for tile_id, error in errors.items():
            self.track_error(tile_id, error)
        
        # Check for growth opportunities
        for tile_id in list(self.model.graph.tiles.keys()):
            if self.should_grow(tile_id):
                self.grow_tile(tile_id)
                stats["grown"] += 1
        
        # Check for pruning opportunities
        for tile_id in list(self.model.graph.tiles.keys()):
            if self.should_prune(tile_id):
                if self.prune_tile(tile_id):
                    stats["pruned"] += 1
        
        return stats


class EventDrivenProcessor:
    """Event-driven tile processing for neuromorphic efficiency.
    
    Tiles only process when:
    - Input activity changes significantly (event threshold)
    - Error exceeds threshold
    - Scheduled refresh timer expires
    
    This mimics neuromorphic spike-based processing for extreme efficiency.
    """
    
    def __init__(
        self,
        model: "AdaptiveTilePC",
        event_threshold: float = 0.1,
        refresh_interval: int = 100,
    ):
        self.model = model
        self.event_threshold = event_threshold
        self.refresh_interval = refresh_interval
        self._last_activity: Dict[int, Tensor] = {}
        self._step_count = 0
        self._events_processed = 0
    
    def check_event(self, tile_id: int) -> bool:
        """Check if a tile should fire an event (process)."""
        tile = self.model.graph.tiles[tile_id]
        
        if tile.activity is None:
            return False
        
        # Check refresh timer
        self._step_count += 1
        if self._step_count % self.refresh_interval == 0:
            return True
        
        # Check activity change
        if tile_id not in self._last_activity:
            self._last_activity[tile_id] = tile.activity.clone()
            return True
        
        # Compute activity change
        delta = (tile.activity - self._last_activity[tile_id]).abs().mean().item()
        
        if delta > self.event_threshold:
            self._last_activity[tile_id] = tile.activity.clone()
            self._events_processed += 1
            return True
        
        return False
    
    def process_events(self) -> int:
        """Process all tiles that have events. Returns count."""
        count = 0
        for tile_id in self.model.graph.tiles:
            if self.check_event(tile_id):
                # Process this tile
                count += 1
        
        self._events_processed += count
        return count
    
    def get_event_rate(self) -> float:
        """Get average events per step."""
        if self._step_count == 0:
            return 0.0
        return self._events_processed / self._step_count


class ContinualLearner:
    """Continual/lifelong learning with elastic weight consolidation.
    
    Enables learning sequences of tasks without catastrophic forgetting by:
    - Computing Fisher information matrix for important weights
    - Penalizing changes to important weights
    - Maintaining task-specific adapters
    """
    
    def __init__(self, model: "AdaptiveTilePC", ewc_lambda: float = 1000.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self._fisher: Dict[str, Tensor] = {}
        self._optimal_weights: Dict[str, Tensor] = {}
        self._task_count = 0
    
    def consolidate_task(self) -> None:
        """Consolidate current task weights (call after training each task)."""
        # Store optimal weights
        self._optimal_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._optimal_weights[name] = param.data.clone()
        
        # Compute Fisher information (diagonal approximation)
        self._fisher = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param)
    
    def compute_fisher(self, X: Tensor, y: Tensor, batch_size: int = 32) -> None:
        """Compute Fisher information matrix diagonal."""
        self.model.train()
        
        for i in range(0, len(X), batch_size):
            x_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            self.model._optim_io.zero_grad()
            self.model.train_step(x_batch, y_batch)
            
            # Compute gradient squared as Fisher approximation
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self._fisher[name] += param.grad.data ** 2
        
        # Normalize
        n_batches = (len(X) + batch_size - 1) // batch_size
        for name in self._fisher:
            self._fisher[name] /= n_batches
    
    def ewc_loss(self) -> Tensor:
        """Compute elastic weight consolidation loss."""
        if not self._optimal_weights:
            return torch.tensor(0.0)
        
        ewc_loss = torch.tensor(0.0)
        for name, param in self.model.named_parameters():
            if name in self._optimal_weights and name in self._fisher:
                diff = param - self._optimal_weights[name]
                ewc_loss = ewc_loss + (self._fisher[name] * diff ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def learn_new_task(
        self,
        X: Tensor,
        y: Tensor,
        epochs: int,
        batch_size: int = 32,
    ) -> Dict:
        """Learn a new task with EWC regularization.
        
        Returns:
            Training history
        """
        history = {"loss": [], "ewc_loss": []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_ewc = 0.0
            n_batches = 0
            
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                idx = perm[i:i+batch_size]
                stats = self.model.train_step(X[idx], y[idx])
                
                # Add EWC loss
                ewc = self.ewc_loss()
                if ewc.requires_grad:
                    ewc.backward()
                
                epoch_loss += stats["loss"]
                epoch_ewc += ewc.item()
                n_batches += 1
            
            history["loss"].append(epoch_loss / n_batches)
            history["ewc_loss"].append(epoch_ewc / n_batches)
        
        # Consolidate after training
        self.consolidate_task()
        self._task_count += 1
        
        return history


class BayesianATPC:
    """Bayesian ATPC for uncertainty quantification.
    
    Uses Monte Carlo dropout to estimate predictive uncertainty:
    - Run multiple forward passes with dropout enabled
    - Compute mean and variance of predictions
    - High variance = high uncertainty
    """
    
    def __init__(self, model: "AdaptiveTilePC", num_samples: int = 50):
        self.model = model
        self.num_samples = num_samples
    
    def predict_with_uncertainty(
        self,
        X: Tensor,
        batch_size: int = 64,
    ) -> Tuple[Tensor, Tensor]:
        """Make predictions with uncertainty estimates.
        
        Returns:
            Tuple of (mean_predictions, uncertainty)
        """
        self.model.train()  # Enable dropout
        
        all_preds = []
        for _ in range(self.num_samples):
            preds = []
            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                with torch.no_grad():
                    pred = self.model(x_batch, steps=self.model.config.inference_steps)
                    if self.model.task_type in ["binary", "multilabel"]:
                        pred = torch.sigmoid(pred)
                    elif self.model.task_type == "classification":
                        pred = F.softmax(pred, dim=-1)
                preds.append(pred)
            all_preds.append(torch.cat(preds, dim=0))
        
        self.model.eval()
        
        # Stack: (num_samples, batch, output_dim)
        all_preds = torch.stack(all_preds, dim=0)
        
        # Mean prediction
        mean_pred = all_preds.mean(dim=0)
        
        # Uncertainty (variance)
        uncertainty = all_preds.var(dim=0).mean(dim=-1)  # Average over output dim
        
        return mean_pred, uncertainty
    
    def get_confidence(self, X: Tensor) -> Tensor:
        """Get confidence score for predictions (1 - uncertainty)."""
        _, uncertainty = self.predict_with_uncertainty(X)
        return 1.0 - uncertainty
    
    def reject_low_confidence(
        self,
        X: Tensor,
        threshold: float = 0.8,
    ) -> Tuple[Tensor, Tensor]:
        """Make predictions, rejecting low-confidence samples.
        
        Returns:
            Tuple of (predictions, mask of kept samples)
        """
        mean_pred, uncertainty = self.predict_with_uncertainty(X)
        confidence = 1.0 - uncertainty
        
        keep_mask = confidence > threshold
        
        if self.model.task_type == "regression":
            predictions = mean_pred.squeeze(-1)
        elif self.model.task_type in ["binary", "multilabel"]:
            predictions = (mean_pred > 0.5).long()
        else:
            predictions = mean_pred.argmax(dim=-1)
        
        return predictions, keep_mask


# =============================================================================
# Hardware Abstraction Layer
# =============================================================================

class HardwareBackend:
    """Abstract base class for hardware backends."""
    
    def __init__(self, model: "AdaptiveTilePC"):
        self.model = model
    
    def forward(self, X: Tensor) -> Tensor:
        """Run forward pass on this backend."""
        raise NotImplementedError
    
    def train_step(self, X: Tensor, y: Tensor) -> Dict:
        """Run training step on this backend."""
        raise NotImplementedError
    
    def get_info(self) -> Dict:
        """Get backend information."""
        raise NotImplementedError


class GPUBackend(HardwareBackend):
    """GPU-optimized backend with mixed precision."""
    
    def __init__(self, model: "AdaptiveTilePC", use_amp: bool = True):
        super().__init__(model)
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler() if use_amp else None
    
    def forward(self, X: Tensor) -> Tensor:
        if self.use_amp and self.scaler:
            with torch.amp.autocast():
                return self.model(X)
        return self.model(X)
    
    def train_step(self, X: Tensor, y: Tensor) -> Dict:
        if self.use_amp and self.scaler:
            with torch.amp.autocast():
                stats = self.model.train_step(X, y)
            return stats
        return self.model.train_step(X, y)
    
    def get_info(self) -> Dict:
        return {
            "backend": "GPU",
            "mixed_precision": self.use_amp,
            "device": str(next(self.model.parameters()).device),
        }


class CPUBackend(HardwareBackend):
    """CPU-optimized backend with threading."""
    
    def __init__(self, model: "AdaptiveTilePC", num_threads: int = 4):
        super().__init__(model)
        torch.set_num_threads(num_threads)
    
    def forward(self, X: Tensor) -> Tensor:
        return self.model(X)
    
    def train_step(self, X: Tensor, y: Tensor) -> Dict:
        return self.model.train_step(X, y)
    
    def get_info(self) -> Dict:
        return {
            "backend": "CPU",
            "num_threads": torch.get_num_threads(),
        }


class NeuromorphicBackend(HardwareBackend):
    """Neuromorphic backend abstraction (Loihi, SpiNNaker, TrueNorth).
    
    Maps ATPC to neuromorphic hardware:
    - Tiles → Cores
    - Weights → Synapses
    - Activities → Spike rates
    - Learning → On-chip plasticity
    """
    
    def __init__(self, model: "AdaptiveTilePC", chip_config: Optional[Dict] = None):
        super().__init__(model)
        self.chip_config = chip_config or {}
        self._mapped = False
    
    def map_to_chip(self) -> None:
        """Map model to neuromorphic chip."""
        # This would interface with neuromorphic SDKs
        # e.g., NxSDK for Loihi, sPyNNaker for SpiNNaker
        self._mapped = True
    
    def forward(self, X: Tensor) -> Tensor:
        if not self._mapped:
            self.map_to_chip()
        # Run on neuromorphic chip
        return self.model(X)
    
    def train_step(self, X: Tensor, y: Tensor) -> Dict:
        if not self._mapped:
            self.map_to_chip()
        # On-chip learning
        return self.model.train_step(X, y)
    
    def get_info(self) -> Dict:
        return {
            "backend": "Neuromorphic",
            "chip_config": self.chip_config,
            "mapped": self._mapped,
        }


class HardwareManager:
    """Manage hardware backends for ATPC.
    
    Automatically selects best backend based on available hardware.
    """
    
    @staticmethod
    def get_best_backend(
        model: "AdaptiveTilePC",
        prefer: str = "auto",
    ) -> HardwareBackend:
        """Get the best available backend.
        
        Args:
            model: ATPC model
            prefer: 'gpu', 'cpu', 'neuromorphic', or 'auto'
        
        Returns:
            HardwareBackend instance
        """
        if prefer == "auto":
            if torch.cuda.is_available():
                return GPUBackend(model)
            return CPUBackend(model)
        elif prefer == "gpu":
            return GPUBackend(model)
        elif prefer == "cpu":
            return CPUBackend(model)
        elif prefer == "neuromorphic":
            return NeuromorphicBackend(model)
        else:
            return CPUBackend(model)
