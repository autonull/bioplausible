"""
Combined Predictive Coding Models
==================================

Aggregates all predictive coding models into a single module for the model zoo.
"""

from __future__ import annotations

from typing import Dict
from typing import Optional

import torch
import torch.nn as nn

from ..base import BioModel
from ..base import ModelConfig
from ..base import register_model

# ============================================================================
# fabricpc_graph_pcn.py - FabricPCGraphPCN
# ============================================================================


@register_model("fabricpc_graph_pcn")
class FabricPCGraphPCN(BioModel):
    """Predictive Coding model using FabricPC graph topology."""

    algorithm_name = "FabricPC Graph PCN"

    def __init__(
        self,
        config: ModelConfig | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        **kwargs,
    ) -> None:
        if config is None:
            config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim if input_dim is not None else 784,
                output_dim=output_dim if output_dim is not None else 10,
                hidden_dims=[hidden_dim] if hidden_dim is not None else [256],
                extra=kwargs,
            )
        super().__init__(config)

        from bioplausible.graph.initialization import initialize_params
        from bioplausible.graph.nodes import Linear
        from bioplausible.graph.nodes import ReLU
        from bioplausible.graph.topology import Edge
        from bioplausible.graph.topology import TaskMap
        from bioplausible.graph.topology import graph

        hidden_dims = self.config.hidden_dims or [self.hidden_dim]
        dims = [self.input_dim] + list(hidden_dims) + [self.output_dim]

        nodes = []
        edges = []

        input_node = Linear(shape=(dims[0], dims[0]), name="input")
        nodes.append(input_node)

        prev_node = input_node
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            linear = Linear(shape=(d_in, d_out), name=f"linear_{i}")
            nodes.append(linear)
            edges.append(Edge(source=prev_node, target=linear.slot("input")))

            if i < len(dims) - 2:
                act = ReLU(name=f"relu_{i}")
                nodes.append(act)
                edges.append(Edge(source=linear, target=act.slot("input")))
                prev_node = act
            else:
                prev_node = linear

        output_node = prev_node

        from bioplausible.graph.inference import InferenceSGD

        extra = self.config.extra or {}
        infer_steps = extra.get("infer_steps", 20)
        eta_infer = extra.get("eta_infer", 0.05)

        self.structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        )

        self._params: dict[str, dict[str, torch.Tensor]] = initialize_params(
            self.structure, rng_key=0
        )

        self._mode = extra.get("mode", "pcn")
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> FabricPCGraphPCN:
        self._device = device
        for node_name in self._params:
            for param_name in self._params[node_name]:
                self._params[node_name][param_name] = self._params[node_name][
                    param_name
                ].to(device)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        from bioplausible.graph.training import _feedforward

        activities = _feedforward(self.structure, self._params, x)
        return activities[self.structure.task_map.y.name]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        from torch.utils.data import DataLoader
        from torch.utils.data import TensorDataset

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=x.shape[0], shuffle=False)

        if self._mode == "backprop":
            from bioplausible.graph.training import train_backprop as trainer
        else:
            from bioplausible.graph.training import train_pcn as trainer

        extra = self.config.extra or {}
        results = trainer(
            self.structure,
            self._params,
            loader,
            epochs=1,
            lr=self.config.learning_rate,
            device=self._device,
            infer_steps=extra.get("infer_steps", 20),
            eta_infer=extra.get("eta_infer", 0.05),
        )

        return {
            "loss": results["train_loss"],
            "accuracy": results["train_acc"],
        }

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            learning_rate=getattr(spec, "default_lr", 0.001),
            extra=kwargs,
        )
        model = cls(config=config).to(device)
        return model


# ============================================================================
# pc_hybrid.py - PredictiveCodingHybrid
# ============================================================================


@register_model("predictive_coding_hybrid")
class PredictiveCodingHybrid(BioModel):
    """Layers predict inputs; FA propagates prediction errors."""

    def __init__(self, config: Optional[ModelConfig] = None, **kwargs):
        super().__init__(config, **kwargs)

        if not hasattr(self, "layers") or len(self.layers) == 0:
            self.layers = nn.ModuleList()
            hidden_dims = (
                self.config.hidden_dims
                if self.config.hidden_dims
                else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
            )
            dims = [self.input_dim] + hidden_dims + [self.output_dim]

            for i in range(len(dims) - 1):
                layer = nn.Linear(dims[i], dims[i + 1])
                layer = self.apply_spectral_norm(layer)
                self.layers.append(layer)

            self.to(kwargs.get("device", "cpu"))

        self.criterion = nn.CrossEntropyLoss()

        self.top_down = nn.ModuleList()
        hidden_dims = (
            self.config.hidden_dims
            if self.config.hidden_dims
            else [self.hidden_dim] if hasattr(self, "hidden_dim") else []
        )
        dims = [self.input_dim] + hidden_dims + [self.output_dim]

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i + 1], dims[i])
            self.top_down.append(layer)

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.config.learning_rate
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()

        activations = [x]
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
            activations.append(h)

        output = activations[-1]
        loss_cls = self.criterion(output, y)

        pc_loss = 0
        for i in range(len(self.layers)):
            upper = activations[i + 1].detach()
            lower_target = activations[i].detach()

            prediction = self.top_down[i](upper)
            pc_loss += nn.functional.mse_loss(prediction, lower_target)

        total_loss = loss_cls + 0.1 * pc_loss
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "accuracy": (output.argmax(1) == y).float().mean().item(),
        }

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        config = ModelConfig(
            name=spec.name,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=[hidden_dim] * min(num_layers, 5),
            extra=kwargs,
        )
        return cls(config=config).to(device)
