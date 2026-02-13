"""
Character N-Gram task for basic language modeling sanity checks.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from bioplausible.hyperopt.tasks import BaseTask
from bioplausible.training.supervised import SupervisedTrainer


class CharNGramTask(BaseTask):
    """
    Synthetic task: Predict next character from previous N chars.
    Dataset: Deterministic repeating patterns or simple probabilistic grammar.
    Input: [B, SeqLen] (indices)
    Output: [B, VocabSize] (logits for last char)
    """

    def __init__(
        self,
        name="char_ngram",
        device="cpu",
        quick_mode=False,
        vocab_size=27,
        context_len=3,
    ):
        super().__init__(name, device, quick_mode)
        self.vocab_size = vocab_size
        self.context_len = context_len
        self._input_dim = None  # Embeddings
        self._output_dim = vocab_size

        # Pattern: a->b->c->d... z->padding
        self.pattern = torch.arange(vocab_size)

    @property
    def task_type(self) -> str:
        return "lm"

    def setup(self):
        # No external data needed
        pass

    def get_batch(
        self, split="train", batch_size=32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random start indices
        starts = torch.randint(0, self.vocab_size - self.context_len, (batch_size,))

        x_list = []
        y_list = []

        for s in starts:
            # Sequence: s, s+1, s+2...
            # Use modulo to wrap around
            seq = (
                torch.arange(s.item(), s.item() + self.context_len + 1)
            ) % self.vocab_size
            x_list.append(seq[:-1])
            y_list.append(seq[-1])

        x = torch.stack(x_list).to(self.device).long()
        y = torch.stack(y_list).to(self.device).long()
        return x, y

    def create_trainer(self, model: nn.Module, **kwargs):
        # Filter kwargs to remove task-specific args if needed
        return SupervisedTrainer(model, self, device=self.device, **kwargs)
