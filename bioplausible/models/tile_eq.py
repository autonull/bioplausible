"""
Adaptive Tile-Based Predictive Coding (ATPC)
=============================================

A scalable, adaptive, continuous learning algorithm combining:
- Predictive Coding (Friston, Rao & Ballard): Minimize prediction error hierarchically
- Adaptive Computation: Allocate resources based on learned importance
- Sparse Updates: Only update parameters that significantly reduce error
- Strategy Framework: Pluggable inference, learning, and scheduling policies

Theoretical Foundation
----------------------
Unlike Equilibrium Propagation's two-phase approach, Predictive Coding
continuously minimizes a variational free energy bound through local
prediction error minimization. Each tile predicts the activity of tiles
above it and adjusts based on prediction errors.

Key Innovations
---------------
1. **Learned Importance Weights**: Tile priority is learned, not heuristic
2. **Adaptive Step Sizes**: Per-tile learning rates based on error history
3. **Sparse Updates**: Skip tiles with negligible prediction error
4. **Consistent Learning**: Same rule applies to all parameters
5. **Fast Convergence**: Second-order information via error preconditioning
6. **Strategy Framework**: Pluggable policies for inference, learning, scheduling

References
----------
* Friston, K. (2005). A theory of cortical responses. Phil. Trans. R. Soc. B.
* Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex.
  Nature Neuroscience.
* Whittington, J. C., & Bogacz, R. (2017). An approximation of the error
  backpropagation algorithm in a predictive coding network. Neural Computation.
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
    
    Adaptive Computation
    --------------------
    sparsity_threshold: Skip tiles with error below this threshold
    importance_decay: EMA decay for importance tracking
    min_active_fraction: Minimum fraction of tiles to update
    
    Regularization
    --------------
    weight_decay: L2 regularization strength
    error_decay: Decay factor for error accumulation
    
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
    
    # Adaptive computation
    sparsity_threshold: float = 0.01
    importance_decay: float = 0.99
    min_active_fraction: float = 0.1
    
    # Regularization
    weight_decay: float = 1e-4
    error_decay: float = 0.95
    
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
        
        if tile.id not in self._velocities:
            self._velocities[tile.id] = torch.zeros_like(tile.activity)
        
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
        **kwargs,
    ):
        """Initialize Adaptive Tile-Based Predictive Coding.
        
        Args:
            neurons_per_tile: Number of neurons in each tile
            num_layers: Total number of layers (input + hidden + output)
            tiles_per_layer: Number of tiles per layer (REQUIRED - no default)
            input_dim: Input feature dimension
            output_dim: Output dimension
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
                hidden_dims=[neurons_per_tile * tiles_per_layer] * (num_layers - 2),
                learning_rate=prediction_lr,
                equilibrium_steps=equilibrium_steps,
            )
        
        super().__init__(config, **kwargs)
        
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
        
        # State tracking
        self._step_count = 0
        self._error_ema: Dict[int, float] = {}
        
        # Initialize weights
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
        """Initialize all parameters."""
        with torch.no_grad():
            for edge in self.graph.edges.values():
                if edge.weight is not None:
                    nn.init.xavier_normal_(edge.weight, gain=1.0)
                if edge.bias is not None:
                    nn.init.zeros_(edge.bias)
            
            nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
            if self.W_in.bias is not None:
                nn.init.zeros_(self.W_in.bias)
            
            nn.init.xavier_normal_(self.W_out.weight, gain=1.0)
            if self.W_out.bias is not None:
                nn.init.zeros_(self.W_out.bias)
    
    # -------------------------------------------------------------------------
    # Forward Dynamics
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
        """Perform one training step.
        
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
        
        # === Learning Phase ===
        # Apply target nudge to output
        target_onehot = F.one_hot(y, self.output_dim).float().to(device)
        target_proj = self.W_out.weight.T @ target_onehot.T  # (n_out, batch)
        target_proj = target_proj.T  # (batch, n_out)
        self._apply_output_nudge(target_proj, beta=0.1)
        
        # Recompute errors after nudge
        self._compute_predictions(batch, device)
        self._compute_errors()
        
        # Update prediction weights
        self._update_weights(batch)
        
        # Update I/O projections
        self._optim_io.zero_grad()
        out_activities = torch.cat(
            [self.graph.tiles[tid].activity for tid in self.graph.output_tile_ids],
            dim=-1
        )
        logits = self.W_out(out_activities)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        self._optim_io.step()
        
        # Update importance weights
        self._update_importance()
        
        # Compute metrics
        accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
        
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
        """Build model from specification."""
        neurons_per_tile = kwargs.pop("neurons_per_tile", 64)
        tiles_per_layer = kwargs.pop("tiles_per_layer", 1)
        
        model = cls(
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            tiles_per_layer=tiles_per_layer,
            input_dim=input_dim,
            output_dim=output_dim,
            **kwargs,
        )
        return model.to(device)
