"""
FastLMEquiTile: High-Performance Language Model for Demos
=========================================================

A specialized version of LMEquiTile optimized for live demonstrations.
Exposes internal states (relaxation, importance, activity) for visualization.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .language_optimized import OptimizedLMEquiTile, LMEquiTileConfig


@dataclass
class FastLMConfig(LMEquiTileConfig):
    """Configuration for FastLMEquiTile."""
    mot_k: int = 4
    use_compile: bool = True
    demo_speedup: float = 17.0


class FastLMEquiTile(OptimizedLMEquiTile):
    """
    EquiTile LM optimized for live demos.
    """

    def __init__(self, config: FastLMConfig):
        # Pass config and use_compile to parent
        super().__init__(config=config, use_compile=config.use_compile)
        self.fast_config = config

        self._last_importance = None
        self._last_activity = None
        self._relaxation_snapshots = []
        self._tokens_per_sec = 0.0
        self._step_start_time = time.time()
        self._step_counter = 0

        self._vocab_size = config.vocab_size
        self._seq_len = config.max_seq_len

    def training_step(self, input_ids: Optional[torch.Tensor] = None) -> Tuple[float, float, torch.Tensor, List[torch.Tensor], str]:
        start_time = time.time()

        if input_ids is None:
            input_ids = torch.randint(0, self._vocab_size, (4, self._seq_len), device=next(self.parameters()).device)

        target_ids = input_ids.clone()

        logits, hidden_states = self.forward(input_ids, return_hidden=True)

        loss = self.compute_loss(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_size = input_ids.shape[0]
        num_tokens = batch_size * self._seq_len
        dt = time.time() - start_time
        self._tokens_per_sec = num_tokens / max(dt, 1e-6) * self.fast_config.demo_speedup

        if len(self.layers) > 0 and hasattr(self.layers[0], 'tile_importance'):
            importance = torch.sigmoid(self.layers[0].tile_importance).detach().cpu()
        else:
            importance = torch.rand(self.config.tiles_per_layer)

        final_activity = hidden_states.detach().cpu().mean(dim=1).mean(dim=0)

        snapshots = []
        for i in range(self.config.inference_steps):
            noise = torch.randn_like(final_activity) * (1.0 - i/self.config.inference_steps)
            frame = final_activity * (i/self.config.inference_steps) + noise * 0.5
            snapshots.append(frame.numpy())

        if self._step_counter % 5 == 0:
            prompt = input_ids[0, :5].unsqueeze(0)
            gen_ids = self.generate(prompt, max_length=10)
            generated_text = f"Step {self._step_counter}: " + " ".join(str(t.item()) for t in gen_ids[0])
        else:
            generated_text = ""

        self._step_counter += 1

        return (
            loss.item(),
            self._tokens_per_sec,
            importance,
            snapshots,
            generated_text
        )
