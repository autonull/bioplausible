import torch
import torch.nn as nn
from typing import Dict, Tuple
from bioplausible.hyperopt.tasks import BaseTask
from bioplausible.training.base import BaseTrainer

class GraphTask(BaseTask):
    """Node classification on Cora / Citeseer / PubMed via torch-geometric."""
    def __init__(self, name: str = "cora", device: str = "cpu", quick_mode: bool = False):
        super().__init__(name, device, quick_mode)
        self.data = None
        self._input_dim = None
        self._output_dim = None

    @property
    def task_type(self) -> str:
        return "graph"

    def setup(self):
        try:
            from torch_geometric.datasets import Planetoid
        except ImportError:
            raise ImportError("torch-geometric required. pip install torch-geometric")
        
        dataset = Planetoid(root='/tmp/' + self.name, name=self.name.capitalize())
        self.data = dataset[0].to(self.device)
        self._input_dim = dataset.num_node_features
        self._output_dim = dataset.num_classes
        
    def get_batch(self, split: str = "train", batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("Call setup() first.")
        return self.data, self.data.y

    def create_trainer(self, model: nn.Module, **kwargs) -> BaseTrainer:
        from bioplausible.training.supervised import SupervisedTrainer
        return SupervisedTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(self, logits: torch.Tensor, y: torch.Tensor, loss: float) -> Dict[str, float]:
        if isinstance(logits, tuple): 
            logits = logits[0]
        acc = (logits.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}
