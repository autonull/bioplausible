"""
EquiTile Graph: Graph Neural Networks with EquiTile
====================================================

Extends EquiTile for graph-structured data:
- GraphEquiTile: Graph neural network with tile-based message passing
- Graph attention mechanisms
- Support for node/graph classification
- Integration with networkx and torch_geometric

Examples
--------
>>> from bioplausible.equitile.graph import GraphEquiTile, GraphEquiTileConfig
>>> config = GraphEquiTileConfig(
...     node_features=10,
...     hidden_dim=64,
...     num_classes=2,
...     num_layers=3,
... )
>>> model = GraphEquiTile(config)
>>> output = model(node_features, edge_index)
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Literal
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.equitile.config import EquiTileConfig
from bioplausible.equitile.core import EquiTile
from bioplausible.zoo.base import BioModel
from bioplausible.zoo.base import ModelConfig
from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.zoo.base import register_model

if TYPE_CHECKING:
    from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GraphEquiTileConfig:
    """Configuration for Graph EquiTile.

    Graph Settings
    --------------
    node_features : int
        Number of input node features
    hidden_dim : int
        Hidden dimension
    num_classes : int
        Number of output classes

    Architecture
    ------------
    num_layers : int
        Number of GNN layers
    neurons_per_tile : int
        Neurons per tile
    tiles_per_layer : int
        Tiles per layer
    attention_heads : int
        Number of attention heads

    Aggregation
    -----------
    aggregation : str
        Aggregation method: 'mean', 'sum', 'max', 'attention'
    readout : str
        Graph readout: 'mean', 'sum', 'max', 'attention'

    Learning
    --------
    learning_rate : float
        Base learning rate
    dropout : float
        Dropout probability
    """

    # Graph settings
    node_features: int = 10
    hidden_dim: int = 64
    num_classes: int = 2

    # Architecture
    num_layers: int = 3
    neurons_per_tile: int = 32
    tiles_per_layer: int = 4
    attention_heads: int = 4

    # Aggregation
    aggregation: Literal["mean", "sum", "max", "attention"] = "mean"
    readout: Literal["mean", "sum", "max", "attention"] = "mean"

    # Learning
    learning_rate: float = 1e-3
    dropout: float = 0.1
    activation: Literal["tanh", "relu", "gelu", "silu"] = "gelu"
    equitile_kwargs: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Graph Operations
# =============================================================================


def aggregate_messages(
    messages: Tensor,
    edge_index: Tensor,
    num_nodes: int,
    method: str = "mean",
) -> Tensor:
    """Aggregate messages from neighbors.

    Parameters
    ----------
    messages : torch.Tensor
        Messages (num_edges, dim)
    edge_index : torch.Tensor
        Edge indices (2, num_edges)
    num_nodes : int
        Number of nodes
    method : str
        Aggregation method

    Returns
    -------
    torch.Tensor
        Aggregated messages (num_nodes, dim)
    """
    if method == "mean":
        return scatter_mean(messages, edge_index[0], dim=0, dim_size=num_nodes)
    elif method == "sum":
        return scatter_sum(messages, edge_index[0], dim=0, dim_size=num_nodes)
    elif method == "max":
        return scatter_max(messages, edge_index[0], dim=0, dim_size=num_nodes)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tensor:
    """Scatter mean aggregation."""
    if dim_size is None:
        dim_size = index.max().item() + 1

    out = src.new_zeros((dim_size,) + src.shape[1:])
    count = src.new_zeros(dim_size)

    out.index_add_(dim, index, src)
    count.index_add_(0, index, src.new_ones(src.shape[0]))

    # Avoid division by zero
    count = count.clamp(min=1)
    return out / count.unsqueeze(-1)


def scatter_sum(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tensor:
    """Scatter sum aggregation."""
    if dim_size is None:
        dim_size = index.max().item() + 1

    out = src.new_zeros((dim_size,) + src.shape[1:])
    out.index_add_(dim, index, src)
    return out


def scatter_max(
    src: Tensor, index: Tensor, dim: int = 0, dim_size: Optional[int] = None
) -> Tensor:
    """Scatter max aggregation."""
    if dim_size is None:
        dim_size = index.max().item() + 1

    out = src.new_full((dim_size,) + src.shape[1:], float("-inf"))
    out.index_reduce_(dim, index, src, reduce="amax")

    # Replace -inf with 0
    out[out == float("-inf")] = 0
    return out


# =============================================================================
# Graph Attention Layer
# =============================================================================


class GraphAttentionLayer(nn.Module):
    """Graph attention layer for EquiTile.

    Parameters
    ----------
    in_features : int
        Input feature dimension
    out_features : int
        Output feature dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads

        assert (
            out_features % num_heads == 0
        ), "out_features must be divisible by num_heads"

        # Linear projections
        self.q_proj = nn.Linear(in_features, out_features)
        self.k_proj = nn.Linear(in_features, out_features)
        self.v_proj = nn.Linear(in_features, out_features)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (num_nodes, in_features)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)

        Returns
        -------
        torch.Tensor
            Output features (num_nodes, out_features)
        """
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]

        # Project to Q, K, V
        q = self.q_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(node_features).view(num_nodes, self.num_heads, self.head_dim)

        # Get edge features
        src_idx, dst_idx = edge_index[0], edge_index[1]
        q_dst = q[dst_idx]  # (num_edges, heads, head_dim)
        k_src = k[src_idx]
        v_src = v[src_idx]

        # Compute attention scores
        scores = (q_dst * k_src).sum(dim=-1) * self.scale  # (num_edges, heads)
        scores = self.dropout(F.softmax(scores, dim=0))

        # Apply attention to values
        messages = scores.unsqueeze(-1) * v_src  # (num_edges, heads, head_dim)

        # Aggregate messages
        output = aggregate_messages(
            messages.view(num_edges, -1),
            edge_index,
            num_nodes,
            method="sum",
        )

        return output.view(num_nodes, -1)


# =============================================================================
# Graph EquiTile Layer
# =============================================================================


class GraphEquiTileLayer(nn.Module):
    """Graph EquiTile layer with tile-based message passing.

    Parameters
    ----------
    config : GraphEquiTileConfig
        Configuration
    """

    def __init__(self, config: GraphEquiTileConfig) -> None:
        super().__init__()
        self.config = config

        # Graph attention
        self.attention = GraphAttentionLayer(
            in_features=config.hidden_dim,
            out_features=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout,
        )

        # Layer norm
        self.norm = nn.LayerNorm(config.hidden_dim)

        # Tile integration (Using Core EquiTile)
        layer_equitile_kwargs = config.equitile_kwargs.copy()
        layer_equitile_kwargs.update(
            {
                "neurons_per_tile": config.neurons_per_tile,
                "num_layers": 2,  # Input -> Output (Simple feedforward block)
                "tiles_per_layer": config.tiles_per_layer,
                "learning_rate": config.learning_rate,
                "dropout": config.dropout,
                "activation": config.activation,
            }
        )

        equitile_config = EquiTileConfig(**layer_equitile_kwargs)

        self.equitile = EquiTile(
            config=equitile_config,
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
        )

        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim),
        )

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices

        Returns
        -------
        torch.Tensor
            Output features
        """
        # Graph attention with residual
        attn_output = self.attention(node_features, edge_index)
        node_features = node_features + self.dropout(attn_output)
        node_features = self.norm(node_features)

        # Tile-based processing
        # Note: EquiTile expects (batch, dim), here we use (num_nodes, dim)
        tile_output = self.equitile(node_features)
        node_features = node_features + tile_output

        # Feedforward with residual
        ffn_output = self.ffn(node_features)
        node_features = node_features + ffn_output

        return node_features


# =============================================================================
# Graph EquiTile
# =============================================================================


@register_model("graph_equitile",
    domains=[Domain.GRAPH],
    locality_level=LocalityLevel.LOCAL,
    bio_plausibility_score=0.75,
    requires_backward=False,
    credit_assignment_type="hebbian",
    family="equitile",
)
class GraphEquiTile(BioModel):
    """Graph EquiTile for graph-structured data.

    Combines graph attention with EquiTile's tile-based
    message passing for node and graph classification.

    Parameters
    ----------
    config : GraphEquiTileConfig, optional
        Configuration
    **kwargs
        Additional configuration parameters

    Examples
    --------
    >>> config = GraphEquiTileConfig(
    ...     node_features=10,
    ...     hidden_dim=64,
    ...     num_classes=2,
    ... )
    >>> model = GraphEquiTile(config)
    >>> output = model(node_features, edge_index)
    """

    algorithm_name = "GraphEquiTile"

    def __init__(
        self,
        config: Optional[GraphEquiTileConfig] = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = GraphEquiTileConfig(**kwargs)

        super().__init__(
            ModelConfig(
                name="graph_equitile",
                input_dim=config.node_features,
                output_dim=config.num_classes,
            )
        )

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.node_features, config.hidden_dim)

        # Graph layers
        self.layers = nn.ModuleList(
            [GraphEquiTileLayer(config) for _ in range(config.num_layers)]
        )

        # Output projection
        if config.readout == "attention":
            self.readout_attention = nn.Linear(config.hidden_dim, 1)
        self.output_proj = nn.Linear(config.hidden_dim, config.num_classes)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
        return_node_embeddings: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """Forward pass.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features (num_nodes, node_features)
        edge_index : torch.Tensor
            Edge indices (2, num_edges)
        batch : torch.Tensor, optional
            Batch indices for each node
        return_node_embeddings : bool
            If True, return node embeddings as well

        Returns
        -------
        torch.Tensor or tuple
            Graph predictions, or (predictions, node_embeddings)
        """
        # Input projection
        x = self.input_proj(node_features)

        # Graph layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Graph readout
        if batch is not None:
            # Batched graphs
            if self.config.readout == "attention":
                attention = torch.sigmoid(self.readout_attention(x))
                graph_features = scatter_mean(x * attention, batch, dim=0)
            elif self.config.readout == "mean":
                graph_features = scatter_mean(x, batch, dim=0)
            elif self.config.readout == "sum":
                graph_features = scatter_sum(x, batch, dim=0)
            elif self.config.readout == "max":
                graph_features = scatter_max(x, batch, dim=0)
            else:
                graph_features = x
        else:
            # Single graph
            graph_features = x.mean(dim=0, keepdim=True)

        # Output projection
        logits = self.output_proj(graph_features)

        if return_node_embeddings:
            return logits, x
        return logits

    def train_step(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Dict[str, float]:
        """Perform one training step.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        labels : torch.Tensor
            Labels (graph or node level)
        batch : torch.Tensor, optional
            Batch indices

        Returns
        -------
        dict
            Training statistics
        """
        # Forward pass
        logits = self.forward(node_features, edge_index, batch)

        # Compute loss
        if labels.dim() == 0 or labels.shape[0] == logits.shape[0]:
            # Graph classification
            loss = F.cross_entropy(logits, labels)
        else:
            # Node classification
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

        # Update
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            if labels.dim() == 0 or labels.shape[0] == logits.shape[0]:
                accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
            else:
                accuracy = (
                    (
                        logits.view(-1, logits.shape[-1]).argmax(dim=-1)
                        == labels.view(-1)
                    )
                    .float()
                    .mean()
                    .item()
                )

        return {
            "loss": loss.item(),
            "accuracy": accuracy,
        }

    def predict(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """Make predictions.

        Parameters
        ----------
        node_features : torch.Tensor
            Node features
        edge_index : torch.Tensor
            Edge indices
        batch : torch.Tensor, optional
            Batch indices

        Returns
        -------
        torch.Tensor
            Predicted class labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(node_features, edge_index, batch)
            return logits.argmax(dim=-1)


# =============================================================================
# Graph Utilities
# =============================================================================


def create_graph_from_edges(
    edge_index: Tensor,
    node_features: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor, int]:
    """Create graph data structures.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices
    node_features : torch.Tensor, optional
        Node features
    num_nodes : int, optional
        Number of nodes

    Returns
    -------
    tuple
        (node_features, edge_index, num_nodes)
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    if node_features is None:
        node_features = torch.randn(num_nodes, 10)  # Default features

    return node_features, edge_index, num_nodes


def add_self_loops(
    edge_index: Tensor,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """Add self-loops to edge index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices
    num_nodes : int, optional
        Number of nodes

    Returns
    -------
    torch.Tensor
        Edge indices with self-loops
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1

    # Create self-loop indices
    self_loop = (
        torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    )

    # Concatenate
    return torch.cat([edge_index, self_loop], dim=1)


# =============================================================================
# Factory Functions
# =============================================================================


def create_graph_model(
    node_features: int,
    num_classes: int,
    hidden_dim: int = 64,
    num_layers: int = 3,
    **kwargs: Any,
) -> GraphEquiTile:
    """Create GraphEquiTile model.

    Parameters
    ----------
    node_features : int
        Node feature dimension
    num_classes : int
        Number of classes
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of layers
    **kwargs
        Additional arguments

    Returns
    -------
    GraphEquiTile
        Graph model
    """
    config = GraphEquiTileConfig(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        **kwargs,
    )
    return GraphEquiTile(config)


def create_molecule_model(
    atom_features: int = 9,
    num_classes: int = 2,
    **kwargs: Any,
) -> GraphEquiTile:
    """Create GraphEquiTile for molecular property prediction.

    Parameters
    ----------
    atom_features : int
        Atom feature dimension
    num_classes : int
        Number of classes
    **kwargs
        Additional arguments

    Returns
    -------
    GraphEquiTile
        Molecule model
    """
    return create_graph_model(
        node_features=atom_features,
        num_classes=num_classes,
        hidden_dim=128,
        num_layers=4,
        attention_heads=4,
        **kwargs,
    )


def create_social_graph_model(
    user_features: int = 16,
    num_classes: int = 2,
    **kwargs: Any,
) -> GraphEquiTile:
    """Create GraphEquiTile for social network analysis.

    Parameters
    ----------
    user_features : int
        User feature dimension
    num_classes : int
        Number of classes
    **kwargs
        Additional arguments

    Returns
    -------
    GraphEquiTile
        Social graph model
    """
    return create_graph_model(
        node_features=user_features,
        num_classes=num_classes,
        hidden_dim=64,
        num_layers=3,
        aggregation="attention",
        **kwargs,
    )
