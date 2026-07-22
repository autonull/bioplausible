"""
Graph Domain Tasks

Standard graph datasets (Cora, PubMed, CiteSeer, etc.)
"""

from typing import Optional

import torch
import torch.nn as nn

from bioplausible.domains.base import (
    DomainSpec,
    DomainTask,
    DomainType,
    Metrics,
    TaskSplit,
)


class GraphTask(DomainTask):
    """Graph domain tasks."""

    def __init__(self, name: str = "cora", dataset_name: str = "cora", **kwargs):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name

    @property
    def domain_type(self) -> DomainType:
        return DomainType.GRAPH

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.GRAPH,
            description=f"Graph task: {self.dataset_name}",
            default_metrics=["accuracy", "loss"],
            supported_tasks=[
                "node_classification",
                "link_prediction",
                "graph_classification",
            ],
            default_batch_size=1,  # Full graph
            default_lr=1e-2,
            tags=["graph", "gnn"],
        )

    def setup(self) -> None:
        try:
            from torch_geometric.datasets import Planetoid
        except ImportError:
            raise ImportError(
                "torch-geometric required for graph tasks. "
                "Install with: pip install torch-geometric"
            )

        dataset = Planetoid(root="./data", name=self.dataset_name.capitalize())
        data = dataset[0]

        self._data = data
        self._input_dim = dataset.num_features
        self._output_dim = dataset.num_classes
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> None:
        # Graph tasks typically use full graph
        return None

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: Optional[int] = None,
    ) -> Metrics:
        model.eval()
        data = self._data.to(self.device)

        with torch.no_grad():
            out = model(data.x, data.edge_index)

            if split == TaskSplit.TRAIN:
                mask = data.train_mask
            elif split == TaskSplit.VAL:
                mask = data.val_mask
            else:
                mask = data.test_mask

            pred = out[mask].argmax(1)
            acc = (pred == data.y[mask]).float().mean().item()
            loss = torch.nn.functional.cross_entropy(out[mask], data.y[mask]).item()

        return Metrics(loss=loss, accuracy=acc)
