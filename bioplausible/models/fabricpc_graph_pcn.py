# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.

"""FabricPCGraphPCN — BioModel wrapper for the FabricPC graph API.

Wraps GraphStructure + train_pcn into a @register_model("fabricpc_graph_pcn")
model that integrates seamlessly with Bioplausible's factory/trainer/demo
infrastructure.
"""

from __future__ import annotations

import torch

from bioplausible.graph.initialization import initialize_params
from bioplausible.graph.nodes import Linear, ReLU
from bioplausible.graph.topology import Edge, TaskMap, graph
from bioplausible.graph.training import train_backprop, train_pcn
from bioplausible.models.base import BioModel, ModelConfig, register_model


@register_model("fabricpc_graph_pcn")
class FabricPCGraphPCN(BioModel):
    """Predictive Coding model using FabricPC graph topology.

    Builds a GraphStructure (MLP) from config, then trains via PC
    (energy minimization + local weight updates). Can also run
    backprop on the same graph for comparison.

    Configurable via ``config.extra``:
        mode: "pcn" (default) or "backprop"
        infer_steps: PC settling steps (default 20)
        eta_infer: PC inference LR (default 0.05)
    """

    algorithm_name = "FabricPC Graph PCN"

    def __init__(
        self,
        config: ModelConfig | None = None,
        input_dim: int | None = None,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        **kwargs,
    ) -> None:
        # Handle config-less call (create_model path) with reasonable MNIST defaults
        if config is None:
            config = ModelConfig(
                name=self.algorithm_name,
                input_dim=input_dim if input_dim is not None else 784,
                output_dim=output_dim if output_dim is not None else 10,
                hidden_dims=[hidden_dim] if hidden_dim is not None else [256],
                extra=kwargs,
            )
        super().__init__(config)

        # Build graph structure from config dimensions
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

        # Initialize parameters
        self._params: dict[str, dict[str, torch.Tensor]] = initialize_params(
            self.structure, rng_key=0
        )

        # Mode
        self._mode = extra.get("mode", "pcn")
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> FabricPCGraphPCN:
        """Move all params to device."""
        self._device = device
        for node_name in self._params:
            for param_name in self._params[node_name]:
                self._params[node_name][param_name] = self._params[node_name][
                    param_name
                ].to(device)
        return self

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Feedforward pass for evaluation."""
        from bioplausible.graph.training import _feedforward

        activities = _feedforward(self.structure, self._params, x)
        return activities[self.structure.task_map.y.name]

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        """Single training step.

        Delegates to train_pcn or train_backprop depending on config mode.
        Returns {"loss", "accuracy"}.
        """
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=x.shape[0], shuffle=False)

        if self._mode == "backprop":
            results = train_backprop(
                self.structure,
                self._params,
                loader,
                epochs=1,
                lr=self.config.learning_rate,
                device=self._device,
            )
        else:
            extra = self.config.extra or {}
            results = train_pcn(
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
        """Factory-style build method for create_model compatibility."""
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
