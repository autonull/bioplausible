"""
TileEQ — Tile-Based Adaptive Equilibrium Propagation
======================================================

A biologically plausible learning implementation where neurons are grouped into
fixed-size "tiles" with adaptive compute allocation. Hot tiles (high kinetic
energy / blame) receive more relaxation steps; cold tiles are skipped.

Key Features
------------
* **Single contiguous memory**: All biases + weights stored in one flat parameter buffer
* **Bidirectional Hopfield dynamics**: Each tile receives signals from both
  lower (bwd_neighbors) and higher (fwd_neighbors) tiles
* **Adaptive scheduling**: Heat-based priority system for efficient compute allocation
* **Exact EP weight updates**: Follows Scellier & Bengio (2017) contrastive Hebbian rule
* **Error diffusion**: Spreads blame to neighbours weighted by connection strength

References
----------
* Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap
  between Energy-Based Models and Backpropagation. Frontiers in Computational Neuroscience.
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
# Configuration Dataclasses
# =============================================================================

@dataclass
class TileEQConfig:
    """Configuration for TileEQ model architecture and training.

    Args:
        neurons_per_tile: Number of neurons in each tile (default: 64)
        num_layers: Total number of layers including input/output (default: 4)
        tiles_per_layer: Number of tiles per hidden layer (default: 1)
        beta: Nudging strength for EP (0=no nudge, 1=full clamp) (default: 0.1)
        epsilon: Convergence threshold for relaxation (default: 1e-4)
        dt: Euler integration step size (larger = faster convergence) (default: 0.5)
        diffusion_rate: Rate of error diffusion between tiles (default: 0.15)
        diffusion_every_k: Apply error diffusion every k training steps (default: 5)
        activation: Activation function ('tanh' or 'relu') (default: 'tanh')
        heat_weights: Weights for (kinetic, entropy, blame, age) heat components
        tau_high: High heat threshold for bucket scheduling (default: 0.5)
        tau_low: Low heat threshold below which tiles may be skipped (default: 0.05)
        tau_max: Maximum heat threshold (adapted during training) (default: 1.0)
        internal_lr: Learning rate for EP memory updates (default: auto = lr * 0.1)
    """
    neurons_per_tile: int = 64
    num_layers: int = 4
    tiles_per_layer: int = 1
    beta: float = 0.1
    epsilon: float = 1e-4
    dt: float = 0.5
    diffusion_rate: float = 0.15
    diffusion_every_k: int = 5
    activation: Literal["tanh", "relu"] = "tanh"
    heat_weights: Tuple[float, float, float, float] = (1.0, 0.5, 1.2, 0.3)
    tau_high: float = 0.5
    tau_low: float = 0.05
    tau_max: float = 1.0
    internal_lr: Optional[float] = None


# =============================================================================
# Tile Data Structures
# =============================================================================

@dataclass
class TileDescriptor:
    """Metadata for a single tile.

    All offsets are into shared state/error tensors or the memory buffer.

    Attributes:
        id: Unique tile identifier
        num_neurons: Number of neurons in this tile
        state_offset: Offset into (batch, total_state_size) state tensor
        error_offset: Offset into error tensor (same as state_offset)
        bias_offset: Offset into memory buffer for bias parameters
        heat: Current heat value (priority for compute allocation)
        last_update_step: Last global step when this tile was updated
        fwd_neighbors: IDs of tiles this tile connects to (forward direction)
        bwd_neighbors: IDs of tiles that connect to this tile (backward direction)
        weight_offsets_fwd: Memory offsets for forward weight matrices
        weight_shapes_fwd: Shapes of forward weight matrices
        is_input: Whether this tile receives external input
        is_output: Whether this tile produces output
        pos_x: Normalized x position for visualization (0-1)
        pos_y: Normalized y position for visualization (0-1)
        layer_id: Layer index in layered architectures
    """
    id: int
    num_neurons: int
    state_offset: int
    error_offset: int
    bias_offset: int

    # Mutable per-step properties
    heat: float = 0.0
    last_update_step: int = 0

    # Graph connectivity
    fwd_neighbors: List[int] = field(default_factory=list)
    bwd_neighbors: List[int] = field(default_factory=list)
    weight_offsets_fwd: List[int] = field(default_factory=list)
    weight_shapes_fwd: List[Tuple[int, int]] = field(default_factory=list)
    weight_offsets_bwd: List[int] = field(default_factory=list)
    weight_shapes_bwd: List[Tuple[int, int]] = field(default_factory=list)

    # Tile role
    is_input: bool = False
    is_output: bool = False

    # Visualization layout (normalized 0-1 coordinates)
    pos_x: float = 0.0
    pos_y: float = 0.0
    layer_id: int = 0


# =============================================================================
# Memory Management
# =============================================================================

class MemoryBlock:
    """Zero-copy slice views into the single flat parameter buffer.

    Provides efficient access to biases and weights stored in a contiguous
    memory buffer. Dynamic tensors (states, errors) are passed separately
    and not stored in the buffer.

    Args:
        buffer: The flat nn.Parameter buffer containing all biases and weights
        tiles: List of tile descriptors for offset calculations
    """

    __slots__ = ('buffer', 'tiles')

    def __init__(self, buffer: nn.Parameter, tiles: List[TileDescriptor]):
        self.buffer = buffer
        self.tiles = tiles

    # ---------------------------------------------------------------------
    # Dynamic (batch-sized) tensor views
    # ---------------------------------------------------------------------

    def state_view(self, states: Tensor, tile_id: int) -> Tensor:
        """Get a view of the state tensor for a specific tile.

        Args:
            states: Full state tensor of shape (batch, total_state_size)
            tile_id: ID of the tile to view

        Returns:
            View of shape (batch, num_neurons) for the specified tile
        """
        tile = self.tiles[tile_id]
        return states[:, tile.state_offset:tile.state_offset + tile.num_neurons]

    def error_view(self, errors: Tensor, tile_id: int) -> Tensor:
        """Get a view of the error tensor for a specific tile.

        Args:
            errors: Full error tensor of shape (batch, total_state_size)
            tile_id: ID of the tile to view

        Returns:
            View of shape (batch, num_neurons) for the specified tile
        """
        tile = self.tiles[tile_id]
        return errors[:, tile.error_offset:tile.error_offset + tile.num_neurons]

    # ---------------------------------------------------------------------
    # Static (1-D) views into the parameter buffer
    # ---------------------------------------------------------------------

    def bias_view(self, tile_id: int) -> Tensor:
        """Get a view of the bias parameters for a specific tile.

        Args:
            tile_id: ID of the tile to view

        Returns:
            View of shape (num_neurons,) containing bias parameters
        """
        tile = self.tiles[tile_id]
        return self.buffer[tile.bias_offset:tile.bias_offset + tile.num_neurons]

    def weight_view(self, src_id: int, dst_id: int) -> Tensor:
        """Get a view of the weight matrix for edge src→dst.

        Args:
            src_id: Source tile ID
            dst_id: Destination tile ID

        Returns:
            View of shape (N_src, N_dst) containing weight parameters

        Raises:
            KeyError: If no edge exists from src_id to dst_id
        """
        src_tile = self.tiles[src_id]
        try:
            idx = src_tile.fwd_neighbors.index(dst_id)
            offset = src_tile.weight_offsets_fwd[idx]
            shape = src_tile.weight_shapes_fwd[idx]
        except ValueError:
            raise KeyError(f"No edge from tile {src_id} to {dst_id}")

        end_offset = offset + shape[0] * shape[1]
        return self.buffer[offset:end_offset].view(shape)


# =============================================================================
# Graph Construction
# =============================================================================

class TileGraph:
    """Constructs and maintains the tile connectivity graph.

    Supports both layered MLP topologies and arbitrary graph structures.
    Manages tile descriptors, edge connectivity, and memory layout.

    Attributes:
        tiles: List of all tile descriptors
        layer_ids: List of tile IDs per layer (empty for non-layered graphs)
        input_tile_ids: IDs of tiles receiving external input
        output_tile_ids: IDs of tiles producing output
        total_buffer_size: Total size of the parameter buffer (biases + weights)
        total_state_size: Total state size per batch item
    """

    def __init__(self) -> None:
        self.tiles: List[TileDescriptor] = []
        self.layer_ids: List[List[int]] = []
        self.input_tile_ids: List[int] = []
        self.output_tile_ids: List[int] = []
        self.total_buffer_size: int = 0
        self.total_state_size: int = 0

    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
    ) -> None:
        """Build a layered MLP topology.

        Creates a feedforward architecture with input → hidden layers → output.
        All tiles have the same number of neurons (neurons_per_tile).

        Args:
            input_dim: Dimensionality of input features
            output_dim: Dimensionality of output predictions
            neurons_per_tile: Number of neurons in each tile
            num_hidden_layers: Number of hidden layers (0 = direct input→output)
            tiles_per_layer: Number of tiles per hidden layer (default: 1)
        """
        # Ensure at least input → output (2 layers minimum)
        num_hidden_layers = max(0, num_hidden_layers)

        # Build layer dimensions: [input_dim, hidden, ..., output_dim]
        hidden_dim = neurons_per_tile * tiles_per_layer
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        total_layers = len(dims)

        # Pass 1: Create tiles
        current_id = 0
        state_offset = 0
        bias_offset = 0

        for layer_idx, dim in enumerate(dims):
            n_tiles = math.ceil(dim / neurons_per_tile)
            layer_tile_ids: List[int] = []

            for tile_col in range(n_tiles):
                tile = TileDescriptor(
                    id=current_id,
                    num_neurons=neurons_per_tile,
                    pos_x=float(layer_idx) / max(1, total_layers - 1),
                    pos_y=float(tile_col) / max(1, n_tiles - 1) if n_tiles > 1 else 0.5,
                    layer_id=layer_idx,
                    state_offset=state_offset,
                    error_offset=state_offset,
                    bias_offset=bias_offset,
                    is_input=(layer_idx == 0),
                    is_output=(layer_idx == len(dims) - 1),
                )
                self.tiles.append(tile)
                layer_tile_ids.append(current_id)
                current_id += 1
                state_offset += neurons_per_tile
                bias_offset += neurons_per_tile

            self.layer_ids.append(layer_tile_ids)

        self.total_state_size = state_offset
        self.input_tile_ids = list(self.layer_ids[0])
        self.output_tile_ids = list(self.layer_ids[-1])

        # Pass 2: Add directed edges between consecutive layers
        weight_offset = bias_offset
        for layer_idx in range(len(self.layer_ids) - 1):
            for src_id in self.layer_ids[layer_idx]:
                for dst_id in self.layer_ids[layer_idx + 1]:
                    self._add_edge(src_id, dst_id, weight_offset)
                    weight_offset += self.tiles[src_id].num_neurons * self.tiles[dst_id].num_neurons

        self.total_buffer_size = weight_offset

    def _add_edge(self, src_id: int, dst_id: int, weight_offset: int) -> None:
        """Add a directed edge between two tiles.

        Args:
            src_id: Source tile ID
            dst_id: Destination tile ID
            weight_offset: Starting offset in memory buffer for weights
        """
        src_tile = self.tiles[src_id]
        dst_tile = self.tiles[dst_id]
        shape = (src_tile.num_neurons, dst_tile.num_neurons)

        # Forward edge (src → dst)
        src_tile.fwd_neighbors.append(dst_id)
        src_tile.weight_offsets_fwd.append(weight_offset)
        src_tile.weight_shapes_fwd.append(shape)

        # Backward edge (dst ← src) - reuses same weight buffer region
        dst_tile.bwd_neighbors.append(src_id)
        dst_tile.weight_offsets_bwd.append(weight_offset)
        dst_tile.weight_shapes_bwd.append(shape)

    def edges(self) -> List[Tuple[int, int]]:
        """Return list of all directed edges as (src_id, dst_id) tuples."""
        return [(tile.id, dst) for tile in self.tiles for dst in tile.fwd_neighbors]

    def get_positions(self) -> List[Tuple[float, float]]:
        """Return (pos_x, pos_y) for each tile in tile-id order."""
        return [(tile.pos_x, tile.pos_y) for tile in self.tiles]

    @classmethod
    def from_edges(
        cls,
        n_tiles: int,
        neurons_per_tile: int,
        fwd_edges: List[Tuple[int, int]],
        input_ids: List[int],
        output_ids: List[int],
        positions: Optional[List[Tuple[float, float]]] = None,
    ) -> "TileGraph":
        """Build an arbitrary topology from an explicit edge list.

        Args:
            n_tiles: Total number of tiles
            neurons_per_tile: Number of neurons in each tile (uniform)
            fwd_edges: List of (src_id, dst_id) directed edges
            input_ids: Tile IDs that receive external input
            output_ids: Tile IDs that produce output
            positions: Optional list of (x, y) coordinates in [0, 1] for visualization

        Returns:
            A new TileGraph instance with the specified topology
        """
        graph = cls()
        state_offset = 0
        bias_offset = 0

        # Create tiles
        for i in range(n_tiles):
            px, py = positions[i] if positions else (0.0, float(i) / max(1, n_tiles - 1))
            tile = TileDescriptor(
                id=i,
                num_neurons=neurons_per_tile,
                state_offset=state_offset,
                error_offset=state_offset,
                bias_offset=bias_offset,
                is_input=(i in input_ids),
                is_output=(i in output_ids),
                pos_x=px,
                pos_y=py,
            )
            graph.tiles.append(tile)
            state_offset += neurons_per_tile
            bias_offset += neurons_per_tile

        graph.total_state_size = state_offset
        graph.input_tile_ids = list(input_ids)
        graph.output_tile_ids = list(output_ids)
        graph.layer_ids = []  # Not a layered architecture

        # Wire edges
        weight_offset = bias_offset
        for src_id, dst_id in fwd_edges:
            graph._add_edge(src_id, dst_id, weight_offset)
            weight_offset += graph.tiles[src_id].num_neurons * graph.tiles[dst_id].num_neurons

        graph.total_buffer_size = weight_offset
        return graph


# =============================================================================
# Heat-Based Scheduling
# =============================================================================

class HeatScheduler:
    """Per-tile heat metric drives adaptive compute allocation.

    Heat is computed from four components:
    - Kinetic: State change magnitude (how much the tile is changing)
    - Entropy: Activation diversity (how varied the activations are)
    - Blame: Error magnitude (how much this tile contributes to loss)
    - Age: Steps since last update (prevents starvation)

    Tiles are bucketed by heat and allocated compute proportionally.
    Hot tiles get more relaxation steps; cold tiles may be skipped.

    Args:
        graph: The tile graph to schedule
        weights: Tuple of (kinetic, entropy, blame, age) weight coefficients
        tau_high: High heat threshold (tiles above get max steps)
        tau_low: Low heat threshold (tiles below may be skipped)
        tau_max: Maximum heat for bucket normalization (adapted during training)
    """

    # Step budget per bucket (index 0 = coldest, 7 = hottest)
    _BUCKET_FRACS: Tuple[float, ...] = (
        0, 1 / 16, 1 / 8, 1 / 4, 3 / 8, 1 / 2, 3 / 4, 1
    )

    def __init__(
        self,
        graph: TileGraph,
        weights: Tuple[float, float, float, float],
        tau_high: float,
        tau_low: float,
        tau_max: float,
    ):
        self.graph = graph
        self.w_kinetic, self.w_entropy, self.w_blame, self.w_age = weights
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.tau_max = max(1e-3, tau_max)
        self._epoch_max_heat = 0.0

    @property
    def epoch_max_heat(self) -> float:
        """Maximum heat observed in the current epoch."""
        return self._epoch_max_heat

    def update(
        self,
        tile: TileDescriptor,
        s_old: Tensor,
        s_new: Tensor,
        err: Tensor,
        step: int,
    ) -> None:
        """Update heat for a tile after relaxation.

        Args:
            tile: The tile to update
            s_old: State before relaxation (batch, num_neurons)
            s_new: State after relaxation (batch, num_neurons)
            err: Error vector for this tile (batch, num_neurons)
            step: Current global training step
        """
        # Kinetic: how much did the state change?
        kinetic = (s_new - s_old).abs().mean().item()

        # Entropy: activation diversity (softmax entropy)
        p = F.softmax(s_new.detach(), dim=-1)
        entropy = -(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item()

        # Blame: error magnitude normalized by neuron count
        blame = err.detach().norm(p=2, dim=-1).mean().item() / max(1, tile.num_neurons)

        # Age: steps since last update (prevents starvation)
        age = float(step - tile.last_update_step)

        # Compute weighted heat
        tile.heat = (
            self.w_kinetic * kinetic
            + self.w_entropy * entropy
            + self.w_blame * blame
            + self.w_age * age
        )
        tile.last_update_step = step
        self._epoch_max_heat = max(self._epoch_max_heat, tile.heat)

    def adapt_threshold(self) -> None:
        """Adapt tau_max at epoch end to prevent cold-collapse.

        Uses exponential moving average to track maximum heat,
        ensuring the scheduling remains sensitive as training progresses.
        """
        self.tau_max = 0.9 * self.tau_max + 0.1 * self._epoch_max_heat
        self.tau_max = max(1e-3, self.tau_max)
        self._epoch_max_heat = 0.0

    def schedule(self, max_steps: int) -> List[Tuple[int, int]]:
        """Generate a compute schedule for all tiles.

        Args:
            max_steps: Maximum relaxation steps available

        Returns:
            List of (tile_id, n_steps) tuples sorted by heat (hottest first)
        """
        tasks: List[Tuple[int, int]] = []
        tau = max(1e-6, self.tau_max)

        for tile in self.graph.tiles:
            # Bucket based on normalized heat
            bucket = max(0, min(7, int(tile.heat / tau * 8)))
            steps = int(self._BUCKET_FRACS[bucket] * max_steps)
            if steps > 0:
                tasks.append((tile.id, steps))

        # Sort by heat (hottest first)
        tasks.sort(key=lambda kv: self.graph.tiles[kv[0]].heat, reverse=True)

        # Cold-start: if all tiles are cold, schedule all with max steps
        if not tasks:
            tasks = [(tile.id, max_steps) for tile in self.graph.tiles]

        return tasks


# =============================================================================
# Main Model
# =============================================================================

@register_model("tile_eq")
class TileEQ(BioModel):
    """Tile-based Equilibrium Propagation.

    This model implements Equilibrium Propagation (Scellier & Bengio, 2017)
    with tile-based adaptive compute allocation. Neurons are grouped into
    fixed-size tiles, and a heat metric drives adaptive scheduling: hot
    tiles receive more relaxation steps while cold tiles are skipped.

    Key Properties
    --------------
    * Single contiguous `self.memory` parameter stores all biases + weights
    * Bidirectional Hopfield dynamics: each tile sees signals from both
      lower (bwd_neighbors) and higher (fwd_neighbors) tiles
    * EP weight update via local contrastive Hebbian rule:
      ΔW = (1/β·B) · (Φ(s_free)ᵀ @ s_src_free − Φ(s_nud)ᵀ @ s_src_nud)
    * Error diffusion spreads blame to neighbours weighted by ||W||_F

    Args:
        config: Optional model configuration (created from kwargs if not provided)
        neurons_per_tile: Number of neurons per tile (default: 64)
        num_layers: Total layers including input/output (default: 4)
        tiles_per_layer: Tiles per hidden layer (default: 1)
        beta: Nudging strength (default: 0.1)
        epsilon: Convergence threshold (default: 1e-4)
        dt: Euler step size (default: 0.5)
        diffusion_rate: Error diffusion rate (default: 0.15)
        diffusion_every_k: Diffusion frequency in steps (default: 5)
        activation: Activation function 'tanh' or 'relu' (default: 'tanh')
        heat_weights: Heat component weights (default: (1.0, 0.5, 1.2, 0.3))
        tau_high: High heat threshold (default: 0.5)
        tau_low: Low heat threshold (default: 0.05)
        tau_max: Max heat threshold (default: 1.0)
        internal_lr: EP memory learning rate (default: auto)
    """

    algorithm_name = "TileEQ"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        neurons_per_tile: int = 64,
        num_layers: int = 4,
        tiles_per_layer: int = 1,
        beta: float = 0.1,
        epsilon: float = 1e-4,
        dt: float = 0.5,
        diffusion_rate: float = 0.15,
        diffusion_every_k: int = 5,
        activation: Literal["tanh", "relu"] = "tanh",
        heat_weights: Tuple[float, float, float, float] = (1.0, 0.5, 1.2, 0.3),
        tau_high: float = 0.5,
        tau_low: float = 0.05,
        tau_max: float = 1.0,
        internal_lr: Optional[float] = None,
        **kwargs,
    ):
        # Extract equilibrium_steps before super() so it ends up in config
        eq_steps = kwargs.pop("max_steps", kwargs.pop("equilibrium_steps", 30))

        # Handle config creation
        if config is None:
            # Extract input/output dims for config
            input_dim = kwargs.pop("input_dim", 0)
            output_dim = kwargs.pop("output_dim", 0)
            learning_rate = kwargs.pop("learning_rate", 0.001)

            config = ModelConfig(
                name="tile_eq",
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=[neurons_per_tile * tiles_per_layer] * (num_layers - 2),
                learning_rate=learning_rate,
                equilibrium_steps=eq_steps,
            )

        super().__init__(config, **kwargs)

        # Override config.equilibrium_steps with our value
        self.config.equilibrium_steps = eq_steps
        self.config.max_steps = eq_steps

        # Store hyperparameters
        self.beta = beta
        self.epsilon = epsilon
        self.dt = dt
        self.diffusion_rate = diffusion_rate
        self.diffusion_every_k = diffusion_every_k
        self.neurons_per_tile = neurons_per_tile
        self.num_layers = num_layers
        self.tiles_per_layer = tiles_per_layer

        # Activation function
        self.phi = torch.tanh if activation == "tanh" else F.relu

        # Build tile graph
        self.graph = TileGraph()
        num_hidden = max(0, num_layers - 2)
        self.graph.build_layered(
            self.input_dim, self.output_dim, neurons_per_tile, num_hidden, tiles_per_layer
        )

        # Single flat parameter buffer (biases + weights)
        self.memory = nn.Parameter(torch.zeros(self.graph.total_buffer_size))
        self.mem_block = MemoryBlock(self.memory, self.graph.tiles)

        # Heat scheduler
        self.scheduler = HeatScheduler(
            self.graph, heat_weights, tau_high, tau_low, tau_max
        )

        # I/O projections (trained via standard backprop)
        # W_in: maps from external input dim to total input tile state size
        n_in_tiles = max(1, len(self.graph.input_tile_ids))
        n_out_tiles = max(1, len(self.graph.output_tile_ids))
        input_tile_dim = n_in_tiles * neurons_per_tile
        output_tile_dim = n_out_tiles * neurons_per_tile

        self.W_in = nn.Linear(self.input_dim, input_tile_dim)
        self.W_out = nn.Linear(output_tile_dim, self.output_dim)

        # Separate optimizers for EP memory and I/O projections
        _ep_lr = internal_lr if internal_lr is not None else self.config.learning_rate * 0.1
        self._optim_internal = torch.optim.Adam([self.memory], lr=_ep_lr)
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=self.config.learning_rate,
        )

        # Persistent state for error diffusion
        self._train_step_count = 0
        self._persistent_errors: Optional[Tensor] = None

        # Initialize weights
        self._init_weights()

    # -------------------------------------------------------------------------
    # Weight Initialization
    # -------------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize all weights and biases.

        Weights: Orthogonal initialization scaled by 2.0/sqrt(fan_in)
        Biases: Zero initialization
        W_in: Kaiming uniform for strong input projection
        """
        with torch.no_grad():
            # Initialize tile-to-tile weights
            for src_id, dst_id in self.graph.edges():
                W = self.mem_block.weight_view(src_id, dst_id)
                nn.init.orthogonal_(W)
                # Scale for good activation range with tanh (0.5-1.5 at output)
                scale = 2.0 / math.sqrt(max(1, W.shape[0]))
                W.mul_(scale)

            # Zero all biases
            for tile in self.graph.tiles:
                self.mem_block.bias_view(tile.id).zero_()

        # W_in: kaiming uniform for strong input projection
        nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
        if self.W_in.bias is not None:
            fan_in = self.W_in.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.W_in.bias, -bound, bound)

    # -------------------------------------------------------------------------
    # Factory Method
    # -------------------------------------------------------------------------

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
    ) -> "TileEQ":
        """Build a TileEQ model from a spec.

        Args:
            spec: Specification object with name and default_lr attributes
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            num_layers: Total number of layers
            device: Target device
            task_type: Task type (classification/regression)
            **kwargs: Additional arguments passed to constructor

        Returns:
            A new TileEQ instance on the specified device
        """
        neurons_per_tile = kwargs.pop("neurons_per_tile", 64)
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 6),
            learning_rate=spec.default_lr,
            extra=kwargs,
        )
        model = cls(
            config=config,
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            **kwargs,
        )
        return model.to(device)

    # -------------------------------------------------------------------------
    # Tile Dynamics
    # -------------------------------------------------------------------------

    def _tile_step(
        self,
        tile_id: int,
        states: Tensor,
        input_proj: Optional[Tensor],
    ) -> Tensor:
        """Perform a single Euler integration step for one tile.

        Input tiles are hard-clamped to the W_in projection (classical EP),
        ensuring feedforward signal reaches hidden and output tiles quickly.

        Args:
            tile_id: ID of the tile to update
            states: Full state tensor (batch, total_state_size)
            input_proj: Projected input from W_in (or None)

        Returns:
            New state tensor (batch, num_neurons) for this tile
        """
        tile = self.graph.tiles[tile_id]

        # Input tiles: hard-clamp to W_in projection (classical EP)
        if input_proj is not None and tile.is_input:
            idx = self.graph.input_tile_ids.index(tile_id)
            start = idx * self.neurons_per_tile
            clamped = input_proj[:, start:start + self.neurons_per_tile]
            return torch.clamp(clamped, -5.0, 5.0)

        # Start with bias contribution
        net = self.mem_block.bias_view(tile_id).unsqueeze(0)  # (1, N)

        # Bottom-up: contribution from upstream tiles via W^T
        for src_id in tile.bwd_neighbors:
            W = self.mem_block.weight_view(src_id, tile_id)  # (N_src, N)
            s_src = self.mem_block.state_view(states, src_id)
            net = net + self.phi(s_src) @ W  # (batch, N)

        # Top-down: contribution from downstream tiles via W
        for dst_id in tile.fwd_neighbors:
            W = self.mem_block.weight_view(tile_id, dst_id)  # (N, N_dst)
            s_dst = self.mem_block.state_view(states, dst_id)
            net = net + self.phi(s_dst) @ W.T  # (batch, N)

        # Euler integration: ds/dt = -s + net
        s = self.mem_block.state_view(states, tile_id)
        s_new = s + self.dt * (-s + net)

        return torch.clamp(s_new, -5.0, 5.0)

    def _relax_tile(
        self,
        tile_id: int,
        n_steps: int,
        states: Tensor,
        errors: Tensor,
        global_step: int,
        input_proj: Optional[Tensor],
    ) -> float:
        """Relax a single tile for n_steps.

        Args:
            tile_id: ID of the tile to relax
            n_steps: Number of relaxation steps
            states: Full state tensor
            errors: Full error tensor
            global_step: Current global training step
            input_proj: Projected input from W_in

        Returns:
            Maximum state change observed during relaxation
        """
        tile = self.graph.tiles[tile_id]
        max_delta = 0.0
        s_old = self.mem_block.state_view(states, tile_id)
        s_new = s_old

        for _ in range(n_steps):
            s_old = self.mem_block.state_view(states, tile_id).clone()
            s_new = self._tile_step(tile_id, states, input_proj)

            # Write back in-place
            states[:, tile.state_offset:tile.state_offset + tile.num_neurons] = s_new

            # Track convergence
            delta = (s_new - s_old).abs().mean().item()
            max_delta = max(max_delta, delta)
            if delta < self.epsilon:
                break

        # Update heat metric
        err_v = self.mem_block.error_view(errors, tile_id)
        self.scheduler.update(tile, s_old, s_new, err_v, global_step)

        return max_delta

    def _relax_graph(
        self,
        states: Tensor,
        errors: Tensor,
        global_step: int,
        max_steps: int,
        input_proj: Optional[Tensor] = None,
    ) -> Tuple[bool, int]:
        """Relax all tiles according to the heat schedule.

        Args:
            states: Full state tensor
            errors: Full error tensor
            global_step: Current global training step
            max_steps: Maximum relaxation steps
            input_proj: Projected input from W_in

        Returns:
            Tuple of (converged, steps_used)
        """
        schedule = self.scheduler.schedule(max_steps)

        for micro in range(max_steps):
            max_delta = 0.0
            for tile_id, _ in schedule:
                delta = self._relax_tile(tile_id, 1, states, errors, global_step + micro, input_proj)
                max_delta = max(max_delta, delta)

            # Early stopping if converged (after minimum steps)
            if max_delta < self.epsilon and micro > 5:
                return True, global_step + micro + 1

        return False, global_step + max_steps

    # -------------------------------------------------------------------------
    # Nudging (Nudged Phase)
    # -------------------------------------------------------------------------

    def _read_outputs(self, states: Tensor) -> Tensor:
        """Read concatenated output from all output tiles.

        Args:
            states: Full state tensor

        Returns:
            Concatenated output activations (batch, N_out_total)
        """
        return torch.cat(
            [self.mem_block.state_view(states, tid) for tid in self.graph.output_tile_ids],
            dim=-1,
        )

    def _nudge_target_in_state_space(
        self,
        out_acts: Tensor,
        target_onehot: Tensor,
    ) -> Tensor:
        """Map target one-hot into output-tile neuron space.

        Uses W_out pseudo-inverse direction to back-project the class target
        into the neuron activation space. The result is scaled to match the
        magnitude of free-phase activations.

        Args:
            out_acts: Free-phase output activations (batch, N_out_total)
            target_onehot: Target labels as one-hot (batch, output_dim)

        Returns:
            Target neuron activations (batch, N_out_total) for clamping
        """
        # Back-project using W_out^T (pseudo-inverse direction)
        target_proj = target_onehot @ self.W_out.weight  # (batch, N_out_total)

        # Scale to match free-phase activation magnitude
        free_norm = out_acts.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tgt_norm = target_proj.norm(dim=1, keepdim=True).clamp(min=1e-6)
        target_proj = target_proj * (free_norm / tgt_norm)

        return target_proj

    def _relax_graph_nudged(
        self,
        states: Tensor,
        errors: Tensor,
        global_step: int,
        max_steps: int,
        input_proj: Tensor,
        clamped_output: Tensor,
    ) -> None:
        """Relax the graph with output tiles clamped toward target.

        This is the canonical EP nudged phase: output neurons are softly
        clamped toward the target with strength beta, allowing the nudge
        to permeate through all tiles.

        Args:
            states: Full state tensor
            errors: Full error tensor
            global_step: Current global training step
            max_steps: Number of nudged relaxation steps
            input_proj: Projected input from W_in
            clamped_output: Target neuron activations for output tiles
        """
        beta = min(self.beta, 0.9)  # Prevent full override at beta==1

        for micro in range(max_steps):
            # Relax all tiles
            for tile_id, _ in self.scheduler.schedule(max_steps):
                self._relax_tile(tile_id, 1, states, errors, global_step + micro, input_proj)

            # Soft-clamp output tiles after each micro-step
            for idx, tile_id in enumerate(self.graph.output_tile_ids):
                tile = self.graph.tiles[tile_id]
                start = idx * self.neurons_per_tile
                chunk = clamped_output[:, start:start + self.neurons_per_tile]

                s = states[:, tile.state_offset:tile.state_offset + tile.num_neurons]
                states[:, tile.state_offset:tile.state_offset + tile.num_neurons] = (
                    (1.0 - beta) * s + beta * chunk
                )

    # -------------------------------------------------------------------------
    # EP Weight Updates
    # -------------------------------------------------------------------------

    def compute_ep_updates(
        self,
        free_states: Tensor,
        nudged_states: Tensor,
        batch_size: int,
    ) -> None:
        """Compute and accumulate EP gradient updates into memory.grad.

        Implements the exact Scellier & Bengio (2017) contrastive Hebbian rule:
            ΔW_{ij} = (1/β·B) · [φ(s_src_n)ᵀ @ φ(s_dst_n) − φ(s_src_f)ᵀ @ φ(s_dst_f)]

        Both source and destination states have φ applied for symmetric Hopfield energy.
        The sign (nudged − free) reinforces the nudged correlation pattern.

        Args:
            free_states: States from free phase equilibrium
            nudged_states: States from nudged phase equilibrium
            batch_size: Batch size for scaling
        """
        grad_acc = torch.zeros_like(self.memory.data)
        scale = 1.0 / (self.beta * batch_size)

        # Weight gradients: ΔW = nudged_corr − free_corr
        for src_id, dst_id in self.graph.edges():
            # Apply φ to both states for symmetric Hopfield energy
            phi_src_f = self.phi(self.mem_block.state_view(free_states, src_id))
            phi_dst_f = self.phi(self.mem_block.state_view(free_states, dst_id))
            phi_src_n = self.phi(self.mem_block.state_view(nudged_states, src_id))
            phi_dst_n = self.phi(self.mem_block.state_view(nudged_states, dst_id))

            # Contrastive Hebbian: reinforce nudged pattern
            dW = scale * (phi_src_n.T @ phi_dst_n - phi_src_f.T @ phi_dst_f)

            # Accumulate gradient at correct memory offset
            src_tile = self.graph.tiles[src_id]
            ei = src_tile.fwd_neighbors.index(dst_id)
            offset = src_tile.weight_offsets_fwd[ei]
            size = dW.numel()
            if size > 0:
                grad_acc[offset:offset + size] += dW.view(-1)

        # Bias gradients: nudged − free (drive biases toward nudged state)
        for tile in self.graph.tiles:
            s_f = self.mem_block.state_view(free_states, tile.id)
            s_n = self.mem_block.state_view(nudged_states, tile.id)
            db = scale * (s_n - s_f).sum(0).view(-1)
            if db.numel() > 0:
                grad_acc[tile.bias_offset:tile.bias_offset + tile.num_neurons] += db

        # Accumulate into existing gradient if present
        if self.memory.grad is None:
            self.memory.grad = grad_acc
        else:
            self.memory.grad += grad_acc

    # -------------------------------------------------------------------------
    # Error Diffusion
    # -------------------------------------------------------------------------

    def diffuse_errors(self, errors: Tensor) -> Tensor:
        """Diffuse accumulated error to neighboring tiles.

        Error is spilled to forward neighbors weighted by the Frobenius
        norm of each connection. This helps propagate blame information
        through the network.

        Args:
            errors: Current error tensor (batch, total_state_size)

        Returns:
            New error tensor with diffused errors
        """
        new_errors = errors.clone()

        for src in self.graph.tiles:
            err_src = self.mem_block.error_view(errors, src.id)

            # Compute weight for each forward edge based on Frobenius norm
            total_norm = 1e-9
            norms: Dict[int, float] = {}
            for dst_id in src.fwd_neighbors:
                norm = self.mem_block.weight_view(src.id, dst_id).norm(p="fro").item()
                norms[dst_id] = norm
                total_norm += norm

            # Deposit spill to each forward neighbor
            for dst_id, norm in norms.items():
                fraction = self.diffusion_rate * (norm / total_norm)
                spill = err_src * fraction
                dst_tile = self.graph.tiles[dst_id]

                if spill.shape[-1] == dst_tile.num_neurons:
                    new_errors[:, dst_tile.error_offset:dst_tile.error_offset + dst_tile.num_neurons] += spill

            # Decay source error
            new_errors[:, src.error_offset:src.error_offset + src.num_neurons] *= (
                1.0 - self.diffusion_rate
            )

        return new_errors

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ):
        """Forward pass through the network.

        Args:
            x: Input tensor (batch, input_dim)
            steps: Number of relaxation steps (default: config.equilibrium_steps)
            return_trajectory: If True, return full state trajectory
            return_dynamics: If True, return convergence diagnostics

        Returns:
            If return_dynamics: (logits, {"trajectory", "deltas", "final_delta"})
            If return_trajectory: (logits, trajectory)
            Otherwise: logits only
        """
        batch, device = x.shape[0], x.device
        states = torch.zeros(batch, self.graph.total_state_size, device=device)
        errors = torch.zeros_like(states)

        input_proj = self.W_in(x)
        max_steps = steps if steps is not None else self.config.equilibrium_steps

        trajectory: List[Tensor] = []
        deltas: List[float] = []

        # Relax to equilibrium
        for _ in range(max_steps):
            prev = states.clone()
            self._relax_graph(states, errors, 0, 1, input_proj)

            if return_dynamics or return_trajectory:
                delta = torch.dist(states, prev, p=2).item()
                deltas.append(delta)
            if return_trajectory:
                trajectory.append(states.clone())

        # Read output and compute logits
        logits = self.W_out(self._read_outputs(states))

        if return_dynamics:
            return logits, {
                "trajectory": trajectory if return_trajectory else None,
                "deltas": deltas,
                "final_delta": deltas[-1] if deltas else 0.0,
            }
        if return_trajectory:
            return logits, trajectory
        return logits

    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]:
        """Perform one training step.

        Implements the full EP training cycle:
        1. Free phase: relax to equilibrium with input clamped
        2. Nudged phase: relax with output clamped toward target
        3. EP weight update: contrastive Hebbian learning
        4. Error diffusion: spread blame to neighbors
        5. I/O update: standard backprop for W_in and W_out

        Args:
            x: Input tensor (batch, input_dim)
            y: Target labels (batch,) for classification

        Returns:
            Dictionary with 'loss' and 'accuracy' metrics
        """
        batch, device = x.shape[0], x.device
        eq_steps = self.config.equilibrium_steps

        # Initialize state and error buffers
        states = torch.zeros(batch, self.graph.total_state_size, device=device)

        # Create or resize persistent error buffer if needed
        if (
            self._persistent_errors is None
            or self._persistent_errors.shape != states.shape
            or self._persistent_errors.device != device
        ):
            self._persistent_errors = torch.zeros_like(states)
        errors = self._persistent_errors

        # -----------------------------------------------------------------
        # 1. Free Phase: Relax to equilibrium with input clamped
        # -----------------------------------------------------------------
        with torch.no_grad():
            input_proj = self.W_in(x)
            self._relax_graph(states, errors, 0, eq_steps, input_proj)
            free_states = states.clone()

            # Compute loss and accuracy
            out_free = self._read_outputs(free_states)
            logits_free = self.W_out(out_free)
            loss = F.cross_entropy(logits_free, y).item()
            acc = (logits_free.argmax(1) == y).float().mean().item()

            # Check for numerical issues
            if not torch.isfinite(logits_free).all():
                return {"loss": 100.0, "accuracy": 0.0}

        # -----------------------------------------------------------------
        # 2. Nudged Phase: Relax with output clamped toward target
        # -----------------------------------------------------------------
        with torch.no_grad():
            target = F.one_hot(y, self.output_dim).float().to(device)
            out_free = self._read_outputs(free_states)
            clamped_out = self._nudge_target_in_state_space(out_free, target)

            states = free_states.clone()
            nudge_steps = max(eq_steps, 10)
            self._relax_graph_nudged(states, errors, 0, nudge_steps, input_proj, clamped_out)
            nudged_states = states.clone()

        # -----------------------------------------------------------------
        # 3. EP Internal Weight Update
        # -----------------------------------------------------------------
        self._optim_internal.zero_grad()
        self.compute_ep_updates(free_states, nudged_states, batch)
        self._optim_internal.step()

        # -----------------------------------------------------------------
        # 4. Error Diffusion
        # -----------------------------------------------------------------
        with torch.no_grad():
            self._persistent_errors = errors + (nudged_states - free_states)
            self._train_step_count += 1

            if self._train_step_count % self.diffusion_every_k == 0:
                self._persistent_errors = self.diffuse_errors(self._persistent_errors)

            # Decay to prevent error accumulation
            self._persistent_errors.mul_(0.99)

        # -----------------------------------------------------------------
        # 5. I/O Projection Update (Standard Backprop)
        # -----------------------------------------------------------------
        self._optim_io.zero_grad()
        logits_bp = self.W_out(self._read_outputs(free_states.detach()))
        F.cross_entropy(logits_bp, y).backward()
        self._optim_io.step()

        # Adapt heat threshold for next epoch
        self.scheduler.adapt_threshold()

        return {"loss": loss, "accuracy": acc}

    # -------------------------------------------------------------------------
    # Reporting & Introspection
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Get model statistics for monitoring and reporting.

        Returns:
            Dictionary with heat statistics, active tile count, and thresholds
        """
        stats = super().get_stats()

        heats = [tile.heat for tile in self.graph.tiles]
        n_tiles = len(heats)
        active = sum(1 for h in heats if h > self.scheduler.tau_low)

        stats.update({
            "heat_mean": sum(heats) / n_tiles if n_tiles else 0.0,
            "heat_max": max(heats) if heats else 0.0,
            "heat_min": min(heats) if heats else 0.0,
            "active_tiles": active,
            "active_fraction": active / n_tiles if n_tiles else 0.0,
            "tau_max": self.scheduler.tau_max,
            "tau_low": self.scheduler.tau_low,
            "total_tiles": n_tiles,
        })

        return stats

    def get_topology_info(self) -> Dict:
        """Get topology information for visualization.

        Returns:
            Dictionary with positions, edges, layer IDs, and tile roles
        """
        return {
            "positions": self.graph.get_positions(),
            "edges": self.graph.edges(),
            "layer_ids": [tile.layer_id for tile in self.graph.tiles],
            "is_input": [tile.is_input for tile in self.graph.tiles],
            "is_output": [tile.is_output for tile in self.graph.tiles],
            "tile_heats": [tile.heat for tile in self.graph.tiles],
        }

    def get_weight_views(self) -> Dict[str, Tensor]:
        """Get all weight matrices as named views.

        Returns:
            Dictionary mapping "Tile {src} → {dst}" to weight tensors
        """
        weights = {}
        with torch.no_grad():
            for src_id, dst_id in self.graph.edges():
                name = f"Tile {src_id} → {dst_id}"
                weights[name] = self.mem_block.weight_view(src_id, dst_id).clone()
        return weights

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def get_state(self) -> Dict:
        """Get complete model state for checkpointing.

        Returns:
            Dictionary containing model weights, optimizer states, and training metadata
        """
        return {
            "model_state_dict": self.state_dict(),
            "optim_internal_state_dict": self._optim_internal.state_dict(),
            "optim_io_state_dict": self._optim_io.state_dict(),
            "train_step_count": self._train_step_count,
            "scheduler_tau_max": self.scheduler.tau_max,
            "tile_heats": {tile.id: tile.heat for tile in self.graph.tiles},
            "config": {
                "neurons_per_tile": self.neurons_per_tile,
                "num_layers": self.num_layers,
                "tiles_per_layer": self.tiles_per_layer,
                "beta": self.beta,
                "epsilon": self.epsilon,
                "dt": self.dt,
                "diffusion_rate": self.diffusion_rate,
                "diffusion_every_k": self.diffusion_every_k,
                "heat_weights": self.scheduler._BUCKET_FRACS,  # Note: internal constant
                "tau_high": self.scheduler.tau_high,
                "tau_low": self.scheduler.tau_low,
            },
        }

    def load_state(self, state: Dict) -> None:
        """Load complete model state from checkpoint.

        Args:
            state: Dictionary from get_state() containing all model data
        """
        # Load model weights
        self.load_state_dict(state["model_state_dict"])

        # Load optimizer states
        self._optim_internal.load_state_dict(state["optim_internal_state_dict"])
        self._optim_io.load_state_dict(state["optim_io_state_dict"])

        # Restore training metadata
        self._train_step_count = state["train_step_count"]
        self.scheduler.tau_max = state["scheduler_tau_max"]

        # Restore tile heats
        for tile in self.graph.tiles:
            if tile.id in state["tile_heats"]:
                tile.heat = state["tile_heats"][tile.id]

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint to disk.

        Args:
            path: File path to save checkpoint
        """
        torch.save(self.get_state(), path)

    def load_checkpoint(self, path: str, device: Optional[torch.device] = None) -> None:
        """Load model checkpoint from disk.

        Args:
            path: File path to load checkpoint from
            device: Target device (default: current device)
        """
        if device is None:
            device = next(self.parameters()).device
        state = torch.load(path, map_location=device, weights_only=True)
        self.load_state(state)

    # -------------------------------------------------------------------------
    # Evaluation Utilities
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: Tensor, steps: Optional[int] = None) -> Tensor:
        """Make predictions without returning diagnostics.

        Args:
            x: Input tensor (batch, input_dim)
            steps: Number of relaxation steps (default: config.equilibrium_steps)

        Returns:
            Logits tensor (batch, output_dim)
        """
        return self.forward(x, steps=steps)

    @torch.no_grad()
    def predict_class(self, x: Tensor, steps: Optional[int] = None) -> Tensor:
        """Predict class labels.

        Args:
            x: Input tensor (batch, input_dim)
            steps: Number of relaxation steps

        Returns:
            Predicted class indices (batch,)
        """
        logits = self.predict(x, steps=steps)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def evaluate(self, x: Tensor, y: Tensor, steps: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model on a batch.

        Args:
            x: Input tensor (batch, input_dim)
            y: Target labels (batch,)
            steps: Number of relaxation steps

        Returns:
            Dictionary with loss and accuracy metrics
        """
        logits = self.predict(x, steps=steps)
        loss = F.cross_entropy(logits, y).item()
        accuracy = (logits.argmax(dim=-1) == y).float().mean().item()
        return {"loss": loss, "accuracy": accuracy}

    # -------------------------------------------------------------------------
    # Architecture Introspection
    # -------------------------------------------------------------------------

    def get_architecture_info(self) -> Dict:
        """Get detailed architecture information.

        Returns:
            Dictionary with architecture details
        """
        return {
            "total_tiles": len(self.graph.tiles),
            "neurons_per_tile": self.neurons_per_tile,
            "num_layers": self.num_layers,
            "tiles_per_layer": self.tiles_per_layer,
            "total_state_size": self.graph.total_state_size,
            "total_params": self.graph.total_buffer_size,
            "input_tiles": self.graph.input_tile_ids,
            "output_tiles": self.graph.output_tile_ids,
            "num_edges": len(self.graph.edges()),
            "layer_structure": self.graph.layer_ids,
        }
