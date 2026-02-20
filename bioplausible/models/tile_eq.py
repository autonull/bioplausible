"""
TileEQ — Tile-Based Adaptive Equilibrium Propagation
======================================================
Neurons are grouped into fixed-size "tiles". A heat metric drives
adaptive compute allocation: hot tiles (high kinetic energy / blame)
receive more relaxation steps; cold tiles are skipped. Weight updates
follow the exact Scellier & Bengio (2017) contrastive Hebbian rule,
applied locally to each tile pair.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BioModel, ModelConfig, register_model


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TileDescriptor:
    """Metadata for one tile; all offsets are into the shared state/error tensors."""
    id: int
    num_neurons: int

    # Into the (batch, total_state_size) tensors
    state_offset: int
    error_offset: int   # same as state_offset (separate errors tensor)

    # Into self.memory (1-D parameter)
    bias_offset: int

    # Mutable per-step properties
    heat: float = 0.0
    last_update_step: int = 0

    # Graph connectivity (only fwd edges store weights; bwd is the transpose)
    fwd_neighbors: List[int] = field(default_factory=list)
    weight_offsets_fwd: List[int] = field(default_factory=list)
    weight_shapes_fwd: List[Tuple[int, int]] = field(default_factory=list)

    bwd_neighbors: List[int] = field(default_factory=list)
    # bwd edges reuse the same weight block as the corresponding fwd edge
    weight_offsets_bwd: List[int] = field(default_factory=list)
    weight_shapes_bwd: List[Tuple[int, int]] = field(default_factory=list)

    is_input: bool = False
    is_output: bool = False

    # Optional layout position for visualization (normalised 0-1)
    pos_x: float = 0.0
    pos_y: float = 0.0
    layer_id: int = 0


class MemoryBlock:
    """Zero-copy slice views into the single flat parameter buffer."""

    def __init__(self, buffer: nn.Parameter, tiles: List[TileDescriptor]):
        self.buffer = buffer
        self.tiles = tiles

    # ------------------------------------------------------------------
    # Dynamic (batch-sized) tensors — NOT stored in the buffer
    # ------------------------------------------------------------------

    def state_view(self, states: torch.Tensor, tile_id: int) -> torch.Tensor:
        t = self.tiles[tile_id]
        return states[:, t.state_offset : t.state_offset + t.num_neurons]

    def error_view(self, errors: torch.Tensor, tile_id: int) -> torch.Tensor:
        t = self.tiles[tile_id]
        return errors[:, t.error_offset : t.error_offset + t.num_neurons]

    # ------------------------------------------------------------------
    # Static (1-D) views into self.buffer (the nn.Parameter)
    # ------------------------------------------------------------------

    def bias_view(self, tile_id: int) -> torch.Tensor:
        t = self.tiles[tile_id]
        return self.buffer[t.bias_offset : t.bias_offset + t.num_neurons]

    def weight_view(self, src_id: int, dst_id: int) -> torch.Tensor:
        """Return a (N_src, N_dst) view of the weight block for edge src→dst."""
        src = self.tiles[src_id]
        try:
            idx = src.fwd_neighbors.index(dst_id)
            off = src.weight_offsets_fwd[idx]
            shape = src.weight_shapes_fwd[idx]
        except ValueError:
            raise KeyError(f"No edge from tile {src_id} to {dst_id}")
        return self.buffer[off : off + shape[0] * shape[1]].view(shape)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class TileGraph:
    """Constructs and holds the TileDescriptor list and edge table."""

    def __init__(self):
        self.tiles: List[TileDescriptor] = []
        self.layer_ids: List[List[int]] = []
        self.input_tile_ids: List[int] = []
        self.output_tile_ids: List[int] = []
        self.total_buffer_size = 0   # biases + weights
        self.total_state_size = 0    # states per batch item

    def build_layered(
        self,
        input_dim: int,
        output_dim: int,
        neurons_per_tile: int,
        num_hidden_layers: int,
        tiles_per_layer: int = 1,
    ):
        """Layered MLP topology.  All tiles have *neurons_per_tile* neurons."""
        # Clamp so we always have at least input → output (2 layers).
        num_hidden_layers = max(0, num_hidden_layers)
        
        # Build layer dimensions: [input_dim, hidden1, hidden2, ..., output_dim]
        # the number of tiles is derived from these dimensions
        hidden_dim = neurons_per_tile * tiles_per_layer
        dims = (
            [input_dim]
            + [hidden_dim] * num_hidden_layers
            + [output_dim]
        )
        total_layers = len(dims)

        current_id = 0
        state_offset = 0
        bias_offset = 0

        # Pass 1: create tiles
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

        # Pass 2: add directed edges between consecutive layers.
        # Weights start immediately after the bias region.
        weight_offset = bias_offset
        for layer_idx in range(len(self.layer_ids) - 1):
            for src_id in self.layer_ids[layer_idx]:
                for dst_id in self.layer_ids[layer_idx + 1]:
                    src_t = self.tiles[src_id]
                    dst_t = self.tiles[dst_id]
                    shape = (src_t.num_neurons, dst_t.num_neurons)
                    size = shape[0] * shape[1]

                    # Forward (src → dst)
                    src_t.fwd_neighbors.append(dst_id)
                    src_t.weight_offsets_fwd.append(weight_offset)
                    src_t.weight_shapes_fwd.append(shape)

                    # Backward (dst ← src), same buffer region
                    dst_t.bwd_neighbors.append(src_id)
                    dst_t.weight_offsets_bwd.append(weight_offset)
                    dst_t.weight_shapes_bwd.append(shape)

                    weight_offset += size

        self.total_buffer_size = weight_offset

    def edges(self) -> List[Tuple[int, int]]:
        return [(t.id, dst) for t in self.tiles for dst in t.fwd_neighbors]

    def get_positions(self) -> List[Tuple[float, float]]:
        """Return (pos_x, pos_y) for each tile in tile-id order (for visualization)."""
        return [(t.pos_x, t.pos_y) for t in self.tiles]

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
        """Build an arbitrary topology from explicit edge list.

        Args:
            n_tiles: total number of tiles.
            neurons_per_tile: neurons in every tile (uniform for now).
            fwd_edges: list of (src_id, dst_id) directed edges.
            input_ids: tile ids that receive external input.
            output_ids: tile ids from which output is read.
            positions: optional list of (x, y) in [0, 1] for layout viz.
        """
        g = cls()
        state_offset = 0
        bias_offset = 0

        # Create tiles
        for i in range(n_tiles):
            px, py = (positions[i] if positions else (0.0, float(i) / max(1, n_tiles - 1)))
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
            g.tiles.append(tile)
            state_offset += neurons_per_tile
            bias_offset += neurons_per_tile

        g.total_state_size = state_offset
        g.input_tile_ids = list(input_ids)
        g.output_tile_ids = list(output_ids)
        g.layer_ids = []  # not layered

        # Wire edges
        weight_offset = bias_offset
        for src_id, dst_id in fwd_edges:
            src_t = g.tiles[src_id]
            dst_t = g.tiles[dst_id]
            shape = (src_t.num_neurons, dst_t.num_neurons)
            size = shape[0] * shape[1]

            src_t.fwd_neighbors.append(dst_id)
            src_t.weight_offsets_fwd.append(weight_offset)
            src_t.weight_shapes_fwd.append(shape)

            dst_t.bwd_neighbors.append(src_id)
            dst_t.weight_offsets_bwd.append(weight_offset)
            dst_t.weight_shapes_bwd.append(shape)

            weight_offset += size

        g.total_buffer_size = weight_offset
        return g

# ---------------------------------------------------------------------------
# Heat / scheduling
# ---------------------------------------------------------------------------

class HeatScheduler:
    """Per-tile heat metric drives adaptive compute allocation."""

    # Step budget per bucket (index 0 = coldest, 7 = hottest)
    _BUCKET_FRACS = [0, 1/16, 1/8, 1/4, 3/8, 1/2, 3/4, 1]

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
        return self._epoch_max_heat

    def update(
        self,
        tile: TileDescriptor,
        s_old: torch.Tensor,
        s_new: torch.Tensor,
        err: torch.Tensor,
        step: int,
    ):
        kinetic = (s_new - s_old).abs().mean().item()

        p = F.softmax(s_new.detach(), dim=-1)
        entropy = -(p * torch.log(p + 1e-9)).sum(dim=-1).mean().item()

        blame = err.detach().norm(p=2, dim=-1).mean().item() / max(1, tile.num_neurons)

        age = float(step - tile.last_update_step)

        tile.heat = (
            self.w_kinetic * kinetic
            + self.w_entropy * entropy
            + self.w_blame * blame
            + self.w_age * age
        )
        tile.last_update_step = step
        self._epoch_max_heat = max(self._epoch_max_heat, tile.heat)

    def adapt_threshold(self):
        """Call at epoch end to prevent cold-collapse."""
        self.tau_max = 0.9 * self.tau_max + 0.1 * self._epoch_max_heat
        self.tau_max = max(1e-3, self.tau_max)
        self._epoch_max_heat = 0.0

    def schedule(self, max_steps: int) -> List[Tuple[int, int]]:
        """Return [(tile_id, n_steps), …] sorted hottest first."""
        tasks: List[Tuple[int, int]] = []
        tau = max(1e-6, self.tau_max)
        for tile in self.graph.tiles:
            bucket = max(0, min(7, int(tile.heat / tau * 8)))
            steps = int(self._BUCKET_FRACS[bucket] * max_steps)
            if steps > 0:
                tasks.append((tile.id, steps))

        tasks.sort(key=lambda kv: self.graph.tiles[kv[0]].heat, reverse=True)

        # If everything is cold (start of training), schedule all tiles.
        if not tasks:
            tasks = [(t.id, max_steps) for t in self.graph.tiles]
        return tasks


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@register_model("tile_eq")
class TileEQ(BioModel):
    """
    Tile-based Equilibrium Propagation.

    Key properties
    --------------
    * Single contiguous `self.memory` parameter stores all biases + weights.
    * Bidirectional Hopfield dynamics: each tile sees signals from both
      lower (bwd_neighbors) and higher (fwd_neighbors) tiles.
    * EP weight update via local contrastive Hebbian rule:
        ΔW = (1/β·B) · (Φ(s_free)ᵀ @ s_src_free
                        − Φ(s_nud)ᵀ @ s_src_nud)
    * Error diffusion spreads blame to neighbours weighted by ‖W‖_F.
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        *,
        neurons_per_tile: int = 64,
        num_layers: int = 4,          # total layers including I/O
        tiles_per_layer: int = 1,
        beta: float = 0.1,
        epsilon: float = 1e-4,
        dt: float = 0.5,              # Euler step size – larger dt = faster convergence
        diffusion_rate: float = 0.15,
        diffusion_every_k: int = 5,
        activation: str = "tanh",
        heat_weights: Tuple[float, float, float, float] = (1.0, 0.5, 1.2, 0.3),
        tau_high: float = 0.5,
        tau_low: float = 0.05,
        tau_max: float = 1.0,
        internal_lr: Optional[float] = None,   # EP memory LR (default: LR/10)
        **kwargs,
    ):
        # Pull out equilibrium_steps / max_steps from kwargs before super()
        # so they end up in config.equilibrium_steps rather than extra.
        eq_steps = kwargs.pop("max_steps", kwargs.pop("equilibrium_steps", 30))
        super().__init__(config, **kwargs)

        # Override config.equilibrium_steps with our value
        self.config.equilibrium_steps = eq_steps
        self.config.max_steps = eq_steps

        self.beta = beta
        self.epsilon = epsilon
        self.dt = dt
        self.diffusion_rate = diffusion_rate
        self.diffusion_every_k = diffusion_every_k
        self.neurons_per_tile = neurons_per_tile
        self.num_layers = num_layers
        self.tiles_per_layer = tiles_per_layer

        self.phi = torch.tanh if activation == "tanh" else F.relu

        # Build tile graph
        self.graph = TileGraph()
        num_hidden = max(0, num_layers - 2)
        self.graph.build_layered(
            self.input_dim, self.output_dim, neurons_per_tile, num_hidden, self.tiles_per_layer
        )

        # Single flat parameter buffer (biases + weights)
        self.memory = nn.Parameter(torch.zeros(self.graph.total_buffer_size))
        self.mem_block = MemoryBlock(self.memory, self.graph.tiles)

        # Heat scheduler
        self.scheduler = HeatScheduler(
            self.graph, heat_weights, tau_high, tau_low, tau_max
        )

        # I/O projections (trained via normal backprop)
        n_in_tiles = max(1, len(self.graph.input_tile_ids))
        n_out_tiles = max(1, len(self.graph.output_tile_ids))
        self.W_in = nn.Linear(self.input_dim, n_in_tiles * neurons_per_tile)
        self.W_out = nn.Linear(n_out_tiles * neurons_per_tile, self.output_dim)

        # Persistent optimizers  (created once, so Adam keeps its momentum state)
        # Use a separate (smaller) LR for EP memory updates to avoid destabilising
        # the feedforward representation learned by W_in/W_out.
        _ep_lr = internal_lr if internal_lr is not None else self.config.learning_rate * 0.1
        self._optim_internal = torch.optim.Adam(
            [self.memory], lr=_ep_lr
        )
        self._optim_io = torch.optim.Adam(
            list(self.W_in.parameters()) + list(self.W_out.parameters()),
            lr=self.config.learning_rate,
        )

        # Step counter for diffusion cadence
        self._train_step_count = 0
        self._persistent_errors: Optional[torch.Tensor] = None

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        with torch.no_grad():
            for src_id, dst_id in self.graph.edges():
                W = self.mem_block.weight_view(src_id, dst_id)
                nn.init.orthogonal_(W)
                # Use 2.0/sqrt(fan_in) so each layer amplifies rather than attenuates.
                # With tanh non-linearity this keeps activations in a good range (0.5-1.5)
                # at the output tiles, providing sufficient signal for W_out to learn.
                scale = 2.0 / math.sqrt(max(1, W.shape[0]))
                W.mul_(scale)
            for tile in self.graph.tiles:
                self.mem_block.bias_view(tile.id).zero_()
        # W_in: kaiming uniform for strong input projection
        nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
        if self.W_in.bias is not None:
            fan_in = self.W_in.weight.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.W_in.bias, -bound, bound)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs):
        neurons_per_tile = kwargs.pop("neurons_per_tile", 64)
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 6),
            learning_rate=spec.default_lr,
            extra=kwargs,
        )
        return cls(
            config=config,
            neurons_per_tile=neurons_per_tile,
            num_layers=num_layers,
            **kwargs,
        ).to(device)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def _tile_step(
        self,
        tile_id: int,
        states: torch.Tensor,
        input_proj: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single Euler step for one tile.  Returns new state (batch, N).

        Input tiles: hard-clamped to the W_in projection (classical EP).
        This ensures the feedforward signal immediately reaches hidden and output
        tiles, so they settle to non-trivial equilibria within a few Euler steps.
        """
        tile = self.graph.tiles[tile_id]
        s = self.mem_block.state_view(states, tile_id)

        # Classical EP: input tiles are pinned to W_in output (no Euler update).
        if input_proj is not None and tile.is_input:
            idx = self.graph.input_tile_ids.index(tile_id)
            start = idx * self.neurons_per_tile
            clamped = input_proj[:, start : start + self.neurons_per_tile]
            # Hard clamp: immediately write back so neighbours see it next step
            return torch.clamp(clamped, -5.0, 5.0)

        net = self.mem_block.bias_view(tile_id).unsqueeze(0)  # (1, N)

        # Bottom-up: contribution from each upstream tile via W^T (note: W is src×dst)
        for src_id in tile.bwd_neighbors:
            W = self.mem_block.weight_view(src_id, tile_id)   # (N_src, N)
            s_src = self.mem_block.state_view(states, src_id)
            net = net + self.phi(s_src) @ W                   # (batch, N)

        # Top-down: contribution from each downstream tile via W (transposed)
        for dst_id in tile.fwd_neighbors:
            W = self.mem_block.weight_view(tile_id, dst_id)   # (N, N_dst)
            s_dst = self.mem_block.state_view(states, dst_id)
            net = net + self.phi(s_dst) @ W.T                 # (batch, N)

        s_new = s + self.dt * (-s + net)
        return torch.clamp(s_new, -5.0, 5.0)

    def _relax_tile(
        self,
        tile_id: int,
        n_steps: int,
        states: torch.Tensor,
        errors: torch.Tensor,
        global_step: int,
        input_proj: Optional[torch.Tensor],
    ) -> float:
        tile = self.graph.tiles[tile_id]
        max_delta = 0.0
        s_old = self.mem_block.state_view(states, tile_id)

        for _ in range(n_steps):
            s_old = self.mem_block.state_view(states, tile_id).clone()
            s_new = self._tile_step(tile_id, states, input_proj)

            # Write back in-place
            states[:, tile.state_offset : tile.state_offset + tile.num_neurons] = s_new

            delta = (s_new - s_old).abs().mean().item()
            max_delta = max(max_delta, delta)
            if delta < self.epsilon:
                break

        err_v = self.mem_block.error_view(errors, tile_id)
        self.scheduler.update(tile, s_old, s_new, err_v, global_step)
        return max_delta

    def _relax_graph(
        self,
        states: torch.Tensor,
        errors: torch.Tensor,
        global_step: int,
        max_steps: int,
        input_proj: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, int]:
        schedule = self.scheduler.schedule(max_steps)
        converged = False

        for micro in range(max_steps):
            max_delta = 0.0
            for tile_id, _ in schedule:
                d = self._relax_tile(tile_id, 1, states, errors, global_step + micro, input_proj)
                max_delta = max(max_delta, d)
            if max_delta < self.epsilon and micro > 5:
                converged = True
                break

        return converged, global_step + max_steps

    # ------------------------------------------------------------------
    # Nudging
    # ------------------------------------------------------------------

    def _read_outputs(self, states: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.mem_block.state_view(states, tid) for tid in self.graph.output_tile_ids],
            dim=-1,
        )

    def _nudge_target_in_state_space(
        self,
        out_acts: torch.Tensor,
        target_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Map a target one-hot into output-tile neuron space using W_out pseudo-inverse.

        Returns a tensor of shape (batch, N_out_total) representing the desired
        neuron activations that would produce the target logits.  This is used
        as the clamped target for the output tiles during the nudged phase.
        """
        # Use W_out.T (pseudo-inverse direction) to back-project class target.
        # Normalise so the norms are comparable to free-phase neuron activations.
        target_proj = target_onehot @ self.W_out.weight  # (batch, N_out_total)
        # Scale to match the magnitude of the free-phase output activations
        free_norm  = out_acts.norm(dim=1, keepdim=True).clamp(min=1e-6)
        tgt_norm   = target_proj.norm(dim=1, keepdim=True).clamp(min=1e-6)
        target_proj = target_proj * (free_norm / tgt_norm)
        return target_proj

    def _relax_graph_nudged(
        self,
        states: torch.Tensor,
        errors: torch.Tensor,
        global_step: int,
        max_steps: int,
        input_proj: torch.Tensor,
        clamped_output: torch.Tensor,
    ):
        """Like _relax_graph but clamps output tiles toward `clamped_output` each step.

        This is the canonical EP nudged phase: maintain a soft clamp on output
        neurons with strength self.beta.  The clamp is a linear interpolation:
          s_out = (1 - beta) * s_relaxed + beta * clamped_output
        at each micro-step, ensuring the nudge permeates deeply into all tiles.
        """
        schedule = self.scheduler.schedule(max_steps)
        beta = min(self.beta, 0.9)  # Guard against full override at beta==1

        for micro in range(max_steps):
            for tile_id, _ in schedule:
                d = self._relax_tile(tile_id, 1, states, errors, global_step + micro, input_proj)

            # Soft-clamp output tiles toward target after each full micro-step
            for idx, tile_id in enumerate(self.graph.output_tile_ids):
                tile = self.graph.tiles[tile_id]
                start = idx * self.neurons_per_tile
                chunk = clamped_output[:, start : start + tile.num_neurons]
                s = states[:, tile.state_offset : tile.state_offset + tile.num_neurons]
                states[:, tile.state_offset : tile.state_offset + tile.num_neurons] = (
                    (1.0 - beta) * s + beta * chunk
                )

    # ------------------------------------------------------------------
    # EP weight update
    # ------------------------------------------------------------------

    def compute_ep_updates(
        self,
        free_states: torch.Tensor,
        nudged_states: torch.Tensor,
        batch_size: int,
    ):
        """Accumulate EP gradients into self.memory.grad.

        Follows exact Scellier & Bengio (2017) Eqn. 13:
            ΔW_{ij} = (1/β·B) · [φ(s_src_n)ᵀ @ φ(s_dst_n) − φ(s_src_f)ᵀ @ φ(s_dst_f)]
        Gradient stored so Adam.step() does W += lr * ΔW.
        Both source and destination states have φ applied (symmetric energy).
        Sign is nudged − free: the nudge brings the network towards target,
        so we want to reinforce the nudged correlation pattern.
        """
        grad_acc = torch.zeros_like(self.memory.data)
        scale = 1.0 / (self.beta * batch_size)

        # --- Weight gradients (nudged − free, both φ-activated) ---
        for src_id, dst_id in self.graph.edges():
            # Apply φ to both states for symmetric Hopfield energy
            phi_src_f = self.phi(self.mem_block.state_view(free_states,   src_id))
            phi_dst_f = self.phi(self.mem_block.state_view(free_states,   dst_id))
            phi_src_n = self.phi(self.mem_block.state_view(nudged_states, src_id))
            phi_dst_n = self.phi(self.mem_block.state_view(nudged_states, dst_id))

            # ΔW = nudged_corr − free_corr   (positive → reinforce nudged pattern)
            dW = scale * (phi_src_n.T @ phi_dst_n - phi_src_f.T @ phi_dst_f)  # (N_src, N_dst)

            src_t = self.graph.tiles[src_id]
            ei = src_t.fwd_neighbors.index(dst_id)
            off = src_t.weight_offsets_fwd[ei]
            sz = dW.numel()
            if sz > 0:
                grad_acc[off : off + sz] += dW.view(-1)

        # --- Bias gradients: nudged − free (drive biases toward nudged state) ---
        for tile in self.graph.tiles:
            s_f = self.mem_block.state_view(free_states,   tile.id)
            s_n = self.mem_block.state_view(nudged_states, tile.id)
            db = scale * (s_n - s_f).sum(0).view(-1)  # (N,)  nudged − free
            if db.numel() > 0:
                grad_acc[tile.bias_offset : tile.bias_offset + tile.num_neurons] += db

        if self.memory.grad is None:
            self.memory.grad = grad_acc
        else:
            self.memory.grad += grad_acc

    # ------------------------------------------------------------------
    # Error diffusion
    # ------------------------------------------------------------------

    def diffuse_errors(self, errors: torch.Tensor) -> torch.Tensor:
        """Spill accumulated error to neighbours (fwd direction only to avoid double-counting)."""
        new_errors = errors.clone()
        for src in self.graph.tiles:
            err_src = self.mem_block.error_view(errors, src.id)

            # Weight by Frobenius norm of each forward edge
            total_norm = 1e-9
            norms: Dict[int, float] = {}
            for dst_id in src.fwd_neighbors:
                n = self.mem_block.weight_view(src.id, dst_id).norm(p="fro").item()
                norms[dst_id] = n
                total_norm += n

            # Deposit spill (pre-decay snapshot of err_src)
            for dst_id, n in norms.items():
                fraction = self.diffusion_rate * (n / total_norm)
                spill = err_src * fraction
                dst = self.graph.tiles[dst_id]
                if spill.shape[-1] == dst.num_neurons:
                    new_errors[
                        :, dst.error_offset : dst.error_offset + dst.num_neurons
                    ] += spill

            # Decay source
            new_errors[
                :, src.error_offset : src.error_offset + src.num_neurons
            ] *= 1.0 - self.diffusion_rate

        return new_errors

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        return_trajectory: bool = False,
        return_dynamics: bool = False,
    ):
        batch, device = x.shape[0], x.device
        states = torch.zeros(batch, self.graph.total_state_size, device=device)
        errors = torch.zeros_like(states)

        input_proj = self.W_in(x)
        max_steps = steps if steps is not None else self.config.equilibrium_steps

        trajectory: List[torch.Tensor] = []
        deltas: List[float] = []

        for _ in range(max_steps):
            prev = states.clone()
            self._relax_graph(states, errors, 0, 1, input_proj)

            if return_dynamics or return_trajectory:
                delta = torch.dist(states, prev, p=2).item()
                deltas.append(delta)
            if return_trajectory:
                trajectory.append(states.clone())

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

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        batch, device = x.shape[0], x.device
        eq = self.config.equilibrium_steps

        states = torch.zeros(batch, self.graph.total_state_size, device=device)
        if (
            self._persistent_errors is None
            or self._persistent_errors.shape != states.shape
            or self._persistent_errors.device != device
        ):
            self._persistent_errors = torch.zeros_like(states)
        errors = self._persistent_errors

        # ----------------------------------------------------------------
        # 1. Free phase
        # ----------------------------------------------------------------
        with torch.no_grad():
            input_proj = self.W_in(x)
            self._relax_graph(states, errors, 0, eq, input_proj)
            free_states = states.clone()

            out_free = self._read_outputs(free_states)
            logits_free = self.W_out(out_free)
            loss = F.cross_entropy(logits_free, y).item()
            acc = (logits_free.argmax(1) == y).float().mean().item()

            if not torch.isfinite(logits_free).all():
                return {"loss": 100.0, "accuracy": 0.0}

        # ----------------------------------------------------------------
        # 2. Nudged phase — canonical EP: hold output clamped toward target
        # ----------------------------------------------------------------
        with torch.no_grad():
            target = F.one_hot(y, self.output_dim).float().to(device)
            # Back-project target into output-tile neuron space
            out_free = self._read_outputs(free_states)
            clamped_out = self._nudge_target_in_state_space(out_free, target)

            states = free_states.clone()
            # Use >= eq steps for nudged phase to ensure proper contrasting equilibrium
            nudge_steps = max(eq, 10)
            self._relax_graph_nudged(states, errors, 0, nudge_steps, input_proj, clamped_out)
            nudged_states = states.clone()

        # ----------------------------------------------------------------
        # 3. EP internal weight update
        # ----------------------------------------------------------------
        self._optim_internal.zero_grad()
        self.compute_ep_updates(free_states, nudged_states, batch)
        self._optim_internal.step()

        # ----------------------------------------------------------------
        # 4. Error diffusion
        # ----------------------------------------------------------------
        with torch.no_grad():
            self._persistent_errors = errors + (nudged_states - free_states)
            self._train_step_count += 1
            if self._train_step_count % self.diffusion_every_k == 0:
                self._persistent_errors = self.diffuse_errors(self._persistent_errors)
            self._persistent_errors.mul_(0.99)

        # ----------------------------------------------------------------
        # 5. I/O projection update via standard cross-entropy backprop
        # ----------------------------------------------------------------
        self._optim_io.zero_grad()
        logits_bp = self.W_out(self._read_outputs(free_states.detach()))
        F.cross_entropy(logits_bp, y).backward()
        self._optim_io.step()

        self.scheduler.adapt_threshold()
        return {"loss": loss, "accuracy": acc}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        stats = super().get_stats()
        heats = [t.heat for t in self.graph.tiles]
        n = len(heats)
        active = sum(1 for h in heats if h > self.scheduler.tau_low)
        stats.update({
            "heat_mean": sum(heats) / n if n else 0.0,
            "heat_max": max(heats) if heats else 0.0,
            "active_tiles": active,
            "active_fraction": active / n if n else 0.0,
            "tau_max": self.scheduler.tau_max,
        })
        return stats
