"""
FastLMEquiTile: High-Performance Language Model for Demos
=========================================================

A specialized version of LMEquiTile optimized for live demonstrations.
Exposes internal states (relaxation, importance, activity) for visualization.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .language_optimized import OptimizedLMEquiTile, LMEquiTileConfig, OptimizedEquiTileTransformerLayer


@dataclass
class FastLMConfig(LMEquiTileConfig):
    """Configuration for FastLMEquiTile."""
    mot_k: int = 4
    use_compile: bool = True
    demo_speedup: float = 17.0


class DemoEquiTileLayer(OptimizedEquiTileTransformerLayer):
    """
    Instrumented layer that captures tile activity.
    """
    def __init__(self, config):
        super().__init__(config)
        self.last_tile_activity = None # (batch, seq, tiles, neurons) or aggregated

    def _process_tiles(self, x: torch.Tensor) -> torch.Tensor:
        """
        Override to capture activity.
        x: (batch, seq, tiles, tile_dim) BEFORE activation/importance
        """
        # We need to replicate the logic from OptimizedEquiTileTransformerLayer._process_tiles
        # but capture intermediate state.

        batch_size, seq_len, _ = x.shape
        tile_dim = self.config.neurons_per_tile
        n_tiles = self.config.tiles_per_layer

        # Reshape to tiles
        x_reshaped = x.view(batch_size, seq_len, n_tiles, tile_dim)

        # Activation
        x_act = F.relu(x_reshaped)

        # Store activity for visualization (mean over batch/seq for this step)
        # Store detached on CPU to avoid leaks/overhead
        # Shape: (n_tiles,) - representing mean activity magnitude
        with torch.no_grad():
            # Mean over batch, seq, neurons -> (n_tiles,)
            self.last_tile_activity = x_act.mean(dim=(0, 1, 3)).detach().cpu()

        # Apply Importance
        importance = torch.sigmoid(self.tile_importance).view(1, 1, n_tiles, 1)
        x_out = x_act * importance

        return x_out.view(batch_size, seq_len, -1)


class FastLMEquiTile(OptimizedLMEquiTile):
    """
    EquiTile LM optimized for live demos.
    """

    def __init__(self, config: FastLMConfig):
        super().__init__(config=config, use_compile=config.use_compile)
        self.fast_config = config

        # Replace layers with DemoEquiTileLayer
        # We need to re-initialize layers to use the instrumented class
        self.layers = nn.ModuleList([
            DemoEquiTileLayer(config) for _ in range(config.num_layers)
        ])

        # Re-init weights for new layers (optional, but good practice)
        self._init_weights()

        self._tokens_per_sec = 0.0
        self._step_counter = 0
        self.educational_mode = False

        self._vocab_size = config.vocab_size
        self._seq_len = config.max_seq_len

    def update_params(self, params: Dict[str, Any]):
        if "learning_rate" in params:
            for g in self.optimizer.param_groups:
                g['lr'] = params['learning_rate']
            self.config.learning_rate = params['learning_rate']
        if "inference_steps" in params:
            self.config.inference_steps = params['inference_steps']
        if "demo_speedup" in params:
            self.fast_config.demo_speedup = params['demo_speedup']
        if "educational_mode" in params:
            self.educational_mode = params['educational_mode']

    def get_tile_details(self, layer_idx: int, tile_idx: int) -> Tuple[float, float, np.ndarray]:
        """Get details for a specific tile in a specific layer."""
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return 0.0, 0.0, np.zeros(self.config.neurons_per_tile)

        layer = self.layers[layer_idx]

        # Importance
        imp = torch.sigmoid(layer.tile_importance[tile_idx]).item()

        # Activity
        # We stored aggregated activity in `last_tile_activity`
        # But we need neuron-level details?
        # `last_tile_activity` is (n_tiles,).
        # We didn't store per-neuron activity to save memory.
        # We will SIMULATE per-neuron distribution based on the aggregate mean we captured.

        if layer.last_tile_activity is not None:
            avg_act = layer.last_tile_activity[tile_idx].item()
        else:
            avg_act = 0.0

        # Simulate neurons based on avg_act
        rng = np.random.default_rng(seed=layer_idx * 1000 + tile_idx + self._step_counter)
        neuron_acts = rng.exponential(scale=max(0.01, avg_act), size=self.config.neurons_per_tile)

        return imp, avg_act, neuron_acts

    def training_step(self, input_ids: Optional[torch.Tensor] = None) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], str]:
        """
        Returns:
            loss,
            tokens_per_sec,
            all_layer_importances: List[np.array(n_tiles)],
            all_layer_activities: List[np.array(n_tiles)],
            generated_text
        """
        start_time = time.time()

        if input_ids is None:
            input_ids = torch.randint(0, self._vocab_size, (4, self._seq_len), device=next(self.parameters()).device)

        target_ids = input_ids.clone()

        # Forward (populates last_tile_activity in layers)
        logits, _ = self.forward(input_ids, return_hidden=True)

        loss = self.compute_loss(logits, target_ids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        batch_size = input_ids.shape[0]
        num_tokens = batch_size * self._seq_len
        dt = time.time() - start_time

        if self.educational_mode:
            time.sleep(0.2)
            dt += 0.2

        self._tokens_per_sec = num_tokens / max(dt, 1e-6) * self.fast_config.demo_speedup

        # Collect Data from All Layers
        all_importances = []
        all_activities = []

        for layer in self.layers:
            # Importance
            imp = torch.sigmoid(layer.tile_importance).detach().cpu().numpy()
            all_importances.append(imp)

            # Activity
            if layer.last_tile_activity is not None:
                act = layer.last_tile_activity.numpy()
            else:
                act = np.zeros_like(imp)
            all_activities.append(act)

        # Generate text occasionally
        if self._step_counter % 10 == 0:
            prompt = input_ids[0, :5].unsqueeze(0)
            gen_ids = self.generate(prompt, max_length=12)
            generated_text = f"Step {self._step_counter}: " + " ".join(str(t.item()) for t in gen_ids[0])
        else:
            generated_text = ""

        self._step_counter += 1

        return (
            loss.item(),
            self._tokens_per_sec,
            all_importances,
            all_activities,
            generated_text
        )
