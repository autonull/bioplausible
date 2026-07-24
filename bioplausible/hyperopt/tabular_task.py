from typing import Dict
from typing import Tuple

import torch
import torch.nn as nn

from bioplausible.hyperopt.tasks import BaseTask
from bioplausible.hyperopt.tasks import _TaskTrainer

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass


class TabularTask(BaseTask):
    """Tabular classification or regression datasets."""

    def __init__(
        self, name: str = "breast_cancer", device: str = "cpu", quick_mode: bool = False
    ):
        super().__init__(name, device, quick_mode)
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None

    @property
    def task_type(self) -> str:
        return "tabular"

    def setup(self):
        if self.name == "california_housing":
            data = fetch_california_housing()
            self._output_dim = 1
        else:
            data = load_breast_cancer()
            self._output_dim = 2

        X = data.data
        y = data.target

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        self.train_x = torch.from_numpy(X_train).float().to(self.device)
        self.val_x = torch.from_numpy(X_val).float().to(self.device)

        if self._output_dim == 1:
            self.train_y = (
                torch.from_numpy(y_train).float().unsqueeze(1).to(self.device)
            )
            self.val_y = torch.from_numpy(y_val).float().unsqueeze(1).to(self.device)
        else:
            self.train_y = torch.from_numpy(y_train).long().to(self.device)
            self.val_y = torch.from_numpy(y_val).long().to(self.device)

        self._input_dim = self.train_x.shape[1]

    def get_batch(
        self, split: str = "train", batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_x = self.train_x if split == "train" else self.val_x
        dataset_y = self.train_y if split == "train" else self.val_y

        idx = torch.randint(0, len(dataset_x), (batch_size,))
        return dataset_x[idx], dataset_y[idx]

    def create_trainer(self, model: nn.Module, **kwargs) -> _TaskTrainer:
        if "device" in kwargs:
            del kwargs["device"]
        return _TaskTrainer(model, self, device=self.device, **kwargs)

    def compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor, loss: float
    ) -> Dict[str, float]:
        if self._output_dim == 1:
            mse = torch.nn.functional.mse_loss(logits, y).item()
            return {"loss": loss, "mse": mse}
        else:
            acc = (logits.argmax(1) == y).float().mean().item()
            return {"loss": loss, "accuracy": acc}
