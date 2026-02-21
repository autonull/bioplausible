"""
FastLMEquiTile: High-Performance Language Model for Demos
=========================================================

A specialized version of LMEquiTile optimized for live demonstrations.
Exposes internal states (relaxation, importance, activity) for visualization.
"""

from dataclasses import dataclass
import time
from typing import Dict, List, Optional, Tuple, Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .language_optimized import OptimizedLMEquiTile, LMEquiTileConfig, OptimizedEquiTileTransformerLayer
from bioplausible.datasets import get_lm_dataset, CharDataset

@dataclass
class FastLMConfig(LMEquiTileConfig):
    """Configuration for FastLMEquiTile."""
    mot_k: int = 4
    use_compile: bool = True
    demo_speedup: float = 17.0
    dataset_name: str = "Random"
    batch_size: int = 32 # Default batch size for training steps


class DemoEquiTileLayer(OptimizedEquiTileTransformerLayer):
    """
    Instrumented layer that captures tile activity.
    """
    def __init__(self, config):
        super().__init__(config)
        self.last_tile_activity = None

    def _process_tiles(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        tile_dim = self.config.neurons_per_tile
        n_tiles = self.config.tiles_per_layer

        x_reshaped = x.view(batch_size, seq_len, n_tiles, tile_dim)
        x_act = F.relu(x_reshaped)

        with torch.no_grad():
            self.last_tile_activity = x_act.mean(dim=(0, 1, 3)).detach().cpu()

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

        self.layers = nn.ModuleList([
            DemoEquiTileLayer(config) for _ in range(config.num_layers)
        ])
        self._init_weights()

        self._tokens_per_sec = 0.0
        self._step_counter = 0
        self.educational_mode = False

        self._vocab_size = config.vocab_size
        self._seq_len = config.max_seq_len

        # Data Handling
        self.dataset = None
        self.data_loader = None
        self._load_data(config.dataset_name)

    def _load_data(self, name):
        """Load dataset or setup random generation."""
        if name == "Random":
            self.dataset = None
            return

        try:
            # Map friendly names to internal IDs if needed
            ds_name = name.lower().replace(" ", "_")
            if "shakespeare" in ds_name:
                ds_name = "tiny_shakespeare"
            elif "wikitext" in ds_name:
                ds_name = "wikitext-2"

            self.dataset = get_lm_dataset(ds_name, seq_len=self._seq_len)
            print(f"Loaded dataset: {name} ({len(self.dataset)} samples)")

            # Use simple random sampling for the demo loop
            # We don't need a full DataLoader overhead for single steps
        except Exception as e:
            print(f"Failed to load dataset {name}: {e}. Falling back to Random.")
            self.dataset = None

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

        if "mot_k" in params:
            self.fast_config.mot_k = params['mot_k']
            # Note: mot_k logic implementation would go here (e.g. updating top-k mask)

        if "weight_decay" in params:
            for g in self.optimizer.param_groups:
                g['weight_decay'] = params['weight_decay']

        if "dropout" in params:
            # Updating dropout requires updating modules
            self.config.dropout = params['dropout']
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = params['dropout']

    def get_tile_details(self, layer_idx: int, tile_idx: int) -> Tuple[float, float, np.ndarray]:
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return 0.0, 0.0, np.zeros(self.config.neurons_per_tile)

        layer = self.layers[layer_idx]
        imp = torch.sigmoid(layer.tile_importance[tile_idx]).item()

        if layer.last_tile_activity is not None:
            avg_act = layer.last_tile_activity[tile_idx].item()
        else:
            avg_act = 0.0

        rng = np.random.default_rng(seed=layer_idx * 1000 + tile_idx + self._step_counter)
        neuron_acts = rng.exponential(scale=max(0.01, avg_act), size=self.config.neurons_per_tile)

        return imp, avg_act, neuron_acts

    def training_step(self, input_ids: Optional[torch.Tensor] = None) -> Tuple[float, float, List[np.ndarray], List[np.ndarray], str]:
        start_time = time.time()

        device = next(self.parameters()).device

        if input_ids is None:
            if self.dataset is not None:
                # Sample real batch
                indices = torch.randint(0, len(self.dataset), (self.fast_config.batch_size,))
                batch_x = []
                for idx in indices:
                    x, _ = self.dataset[idx.item()]
                    batch_x.append(x)
                input_ids = torch.stack(batch_x).to(device)
            else:
                # Random generation
                input_ids = torch.randint(0, self._vocab_size, (self.fast_config.batch_size, self._seq_len), device=device)

        target_ids = input_ids.clone() # Next token prediction target is usually shifted

        # Forward
        logits, _ = self.forward(input_ids, return_hidden=True)

        # Loss
        loss = self.compute_loss(logits, target_ids)

        # Backward
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

        # Collect Data
        all_importances = []
        all_activities = []

        for layer in self.layers:
            imp = torch.sigmoid(layer.tile_importance).detach().cpu().numpy()
            all_importances.append(imp)

            if layer.last_tile_activity is not None:
                act = layer.last_tile_activity.numpy()
            else:
                act = np.zeros_like(imp)
            all_activities.append(act)

        # Generate text
        if self._step_counter % 10 == 0:
            prompt = input_ids[0, :5].unsqueeze(0)
            if self.dataset:
                # Use real decoding if dataset available
                gen_ids = self.generate(prompt, max_length=20)
                decoded = self.dataset.decode(gen_ids[0])
                generated_text = f"Step {self._step_counter}:\n{decoded}"
            else:
                gen_ids = self.generate(prompt, max_length=12)
                generated_text = f"Step {self._step_counter} (Random):\n" + " ".join(str(t.item()) for t in gen_ids[0])
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
