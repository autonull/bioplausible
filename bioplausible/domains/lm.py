"""
Language Modeling Domain Tasks

Standard LM datasets (Shakespeare, WikiText, etc.)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from bioplausible.domains.base import (
    DomainTask,
    DomainType,
    DomainSpec,
    TaskSplit,
    Metrics,
)


class LMTask(DomainTask):
    """Language modeling domain tasks."""

    def __init__(
        self,
        name: str = "tiny_shakespeare",
        dataset_name: str = "tiny_shakespeare",
        seq_len: int = 128,
        vocab_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    @property
    def domain_type(self) -> DomainType:
        return DomainType.LM

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.LM,
            description=f"Language modeling: {self.dataset_name}",
            default_metrics=["perplexity", "loss", "accuracy"],
            supported_tasks=["language_modeling", "text_generation"],
            default_batch_size=32,
            default_lr=3e-4,
            requires_sequence=True,
            tags=["nlp", "language"],
        )

    def setup(self) -> None:
        """Load LM dataset."""
        # Use existing dataset loading from bioplausible.datasets
        from bioplausible.datasets import get_lm_dataset

        dataset = get_lm_dataset(self.dataset_name, seq_len=self.seq_len)
        data = dataset.data
        self.vocab_size = self.vocab_size or dataset.vocab_size
        self._output_dim = self.vocab_size
        self._input_dim = self.seq_len  # token indices

        # Split train/val
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # Create DataLoaders
        self._train_loader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )
        self._val_loader = DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False
        )
        self._test_loader = self._val_loader
        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> DataLoader:
        if not self._setup_done:
            self.setup()

        if split == TaskSplit.TRAIN:
            return self._train_loader
        else:
            return self._val_loader

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: Optional[int] = None,
    ) -> Metrics:
        import numpy as np

        model.eval()
        loader = self.get_dataloader(split)

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if max_batches and i >= max_batches:
                    break

                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                else:
                    inputs = batch[:, :-1]
                    targets = batch[:, 1:]

                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                # Reshape for cross entropy
                if outputs.dim() == 3:
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    targets = targets.reshape(-1)

                loss = self.compute_loss(outputs, targets)

                total_loss += loss.item() * targets.numel()
                total_correct += (outputs.argmax(1) == targets).sum().item()
                total_tokens += targets.numel()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        perplexity = np.exp(min(avg_loss, 10))

        return Metrics(loss=avg_loss, accuracy=accuracy, perplexity=perplexity)

    def compute_loss(
        self, outputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(outputs, targets)
