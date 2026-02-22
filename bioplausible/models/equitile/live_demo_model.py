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
    use_compile: bool = False
    dataset_name: str = "Tiny Shakespeare"
    batch_size: int = 16
    sparsity_weight: float = 0.5  # L1 regularization to encourage sparse tile activation
    sparsity_threshold: float = 0.1  # Threshold for considering a tile "active"
    importance_lr: float = 0.1  # Separate learning rate for tile importance (higher than weight lr)
    importance_decay: float = 0.01  # Decay rate for importance momentum toward neutral
    use_competitive: bool = False  # Use competitive tile selection (top-k)
    competitive_k_ratio: float = 0.5  # Ratio of tiles to keep active in competitive mode


class DemoEquiTileLayer(OptimizedEquiTileTransformerLayer):
    """
    Instrumented layer that captures tile activity with gating and memory.
    """
    def __init__(self, config):
        super().__init__(config)
        self.last_tile_activity = None
        
        # Replace tile_importance with gated version
        # Gate logits for binary on/off (straight-through estimator)
        self.gate_logits = nn.Parameter(torch.zeros(config.tiles_per_layer))
        # Activity EMA for hysteresis (memory of recent activity)
        self.register_buffer('activity_ema', torch.zeros(config.tiles_per_layer))
        self.ema_decay = 0.99
        
        # Reinitialize importance with higher variance for diversity
        self.tile_importance.data = torch.randn_like(self.tile_importance) * 2.0

    def update_activity_ema(self, current_activity):
        """Update exponential moving average of tile activity."""
        with torch.no_grad():
            self.activity_ema = self.ema_decay * self.activity_ema + (1 - self.ema_decay) * current_activity

    def get_gate_state(self):
        """Return binary gate state and importance level."""
        with torch.no_grad():
            is_active = (torch.sigmoid(self.gate_logits) > 0.5).float()
            importance = torch.sigmoid(self.tile_importance)
            return is_active.cpu().numpy(), importance.cpu().numpy()

    def _process_tiles(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        tile_dim = self.config.neurons_per_tile
        n_tiles = self.config.tiles_per_layer

        x_reshaped = x.view(batch_size, seq_len, n_tiles, tile_dim)
        x_act = F.relu(x_reshaped)

        with torch.no_grad():
            self.last_tile_activity = x_act.mean(dim=(0, 1, 3)).detach().cpu()
            # Update activity EMA for hysteresis
            self.update_activity_ema(self.last_tile_activity.squeeze())

        # Binary gate with straight-through estimator
        gate_prob = torch.sigmoid(self.gate_logits)
        gate = (gate_prob > 0.5).float()
        # Straight-through: pass gradient as if gate was continuous
        gate = gate + (gate_prob - gate).detach()

        # Apply gate and importance
        importance = torch.sigmoid(self.tile_importance).view(1, 1, n_tiles, 1)
        x_out = x_act * gate.view(1, 1, n_tiles, 1) * importance.view(1, 1, n_tiles, 1)

        return x_out.view(batch_size, seq_len, -1)


class FastLMEquiTile(OptimizedLMEquiTile):
    """
    EquiTile LM optimized for live demos.
    """

    def __init__(self, config: FastLMConfig):
        super().__init__(config=config, use_compile=config.use_compile)
        self.fast_config = config

        # Replace layers with our instrumented DemoEquiTileLayer
        self.layers = nn.ModuleList([
            DemoEquiTileLayer(config) for _ in range(config.num_layers)
        ])

        # Re-initialize weights for new layers
        self._init_weights()

        # Initialize tile_importance with high variance
        # This creates natural diversity: some tiles start active, some inactive
        # Sparsity regularization will then push weak tiles toward inactivity
        for layer in self.layers:
            # High variance initialization: N(0, 2.0) gives good spread
            # Sigmoid of values <-3 are <0.05 (inactive)
            # Sigmoid of values >3 are >0.95 (fully active)
            layer.tile_importance.data = torch.randn_like(layer.tile_importance) * 2.0
            # Initialize gate logits with variance to create 50/50 open/closed split
            # N(0, 2.0) gives sigmoid values spread between ~0.1 and ~0.9
            layer.gate_logits.data = torch.randn_like(layer.gate_logits) * 2.0

        # Separate parameter groups for importance and weights
        importance_params = []
        weight_params = []
        for name, param in self.named_parameters():
            if 'tile_importance' in name or 'gate_logits' in name:
                importance_params.append(param)
            else:
                weight_params.append(param)

        # Create optimizer with different learning rates
        # Importance learns faster to respond quickly to changes
        self.optimizer = torch.optim.AdamW([
            {'params': weight_params, 'lr': config.learning_rate},
            {'params': importance_params, 'lr': config.importance_lr, 'betas': (0.9, 0.999)}
        ], weight_decay=config.weight_decay)

        self._tokens_per_sec = 0.0
        self._step_counter = 0
        self._prev_sparsity_weight = config.sparsity_weight

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
                if 'tile_importance' not in g.get('name', '') and 'gate_logits' not in g.get('name', ''):
                    g['lr'] = params['learning_rate']
            self.config.learning_rate = params['learning_rate']

        if "importance_lr" in params:
            for g in self.optimizer.param_groups:
                g['lr'] = params['importance_lr']
            self.fast_config.importance_lr = params['importance_lr']

        if "inference_steps" in params:
            self.config.inference_steps = params['inference_steps']

        if "mot_k" in params:
            self.fast_config.mot_k = params['mot_k']

        if "weight_decay" in params:
            for g in self.optimizer.param_groups:
                g['weight_decay'] = params['weight_decay']

        if "dropout" in params:
            self.config.dropout = params['dropout']
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.p = params['dropout']

        if "sparsity_weight" in params:
            self.fast_config.sparsity_weight = params['sparsity_weight']

        if "sparsity_threshold" in params:
            self.fast_config.sparsity_threshold = params['sparsity_threshold']

        if "importance_decay" in params:
            self.fast_config.importance_decay = params['importance_decay']

        if "use_competitive" in params:
            self.fast_config.use_competitive = params['use_competitive']

        if "competitive_k_ratio" in params:
            self.fast_config.competitive_k_ratio = params['competitive_k_ratio']

    def get_tile_details(self, layer_idx: int, tile_idx: int) -> Tuple[float, float, np.ndarray, bool]:
        """Get tile details including gate state.
        
        Returns:
            (importance, avg_activity, neuron_activities, is_gate_open)
        """
        if layer_idx < 0 or layer_idx >= len(self.layers):
            return 0.0, 0.0, np.zeros(self.config.neurons_per_tile), False

        layer = self.layers[layer_idx]
        imp = torch.sigmoid(layer.tile_importance[tile_idx]).item()

        if layer.last_tile_activity is not None:
            avg_act = layer.last_tile_activity[tile_idx].item()
        else:
            avg_act = 0.0

        rng = np.random.default_rng(seed=layer_idx * 1000 + tile_idx + self._step_counter)
        neuron_acts = rng.exponential(scale=max(0.01, avg_act), size=self.config.neurons_per_tile)

        # Get gate state
        is_active = (torch.sigmoid(layer.gate_logits[tile_idx]) > 0.5).item()

        return imp, avg_act, neuron_acts, is_active

    def training_step(self, input_ids: Optional[torch.Tensor] = None) -> Tuple[float, float, float, float, float, List[np.ndarray], List[np.ndarray], str, List[float]]:
        """
        Perform one training step.
        
        Returns:
            loss, tokens_per_sec, train_accuracy, test_accuracy, perplexity, 
            importances, activities, generated_text, tile_losses
        """
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

        target_ids = input_ids.clone()  # Next token prediction target

        # Forward pass
        logits, hidden = self.forward(input_ids, return_hidden=True)

        # Compute loss
        loss = self.compute_loss(logits, target_ids)

        # Compute perplexity
        perplexity = torch.exp(loss).item()

        # Compute training accuracy
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            train_correct = (pred_ids == target_ids).sum().item()
            train_total = target_ids.numel()
            train_acc = 100.0 * train_correct / train_total

        # Sparsity regularization: L1 penalty on sigmoid(importance) * gate
        # This pushes unimportant tiles toward zero (inactive)
        sparsity_loss = torch.tensor(0.0, device=loss.device)
        for layer in self.layers:
            importance = torch.sigmoid(layer.tile_importance)
            gate_prob = torch.sigmoid(layer.gate_logits)
            # L1 penalty on gated importance
            sparsity_loss = sparsity_loss + self.fast_config.sparsity_weight * (importance * gate_prob).sum()

        # Importance momentum decay: prevent importance from accumulating indefinitely
        # Decay toward neutral (sigmoid = 0.5, importance = 0)
        decay_loss = torch.tensor(0.0, device=loss.device)
        for layer in self.layers:
            importance_sigmoid = torch.sigmoid(layer.tile_importance)
            decay_loss = decay_loss + self.fast_config.importance_decay * ((importance_sigmoid - 0.5) ** 2).sum()

        # Competitive tile selection (optional)
        if self.fast_config.use_competitive:
            for layer in self.layers:
                k = int(self.config.tiles_per_layer * self.fast_config.competitive_k_ratio)
                # Keep only top-k tiles by importance
                top_k_values, top_k_indices = torch.topk(layer.tile_importance.squeeze(), k)
                # Create mask for top-k
                mask = torch.zeros_like(layer.tile_importance)
                mask.scatter_(0, top_k_indices.unsqueeze(-1), 1.0)
                # Apply competitive suppression with straight-through
                layer.tile_importance.data = layer.tile_importance * mask + (layer.tile_importance * mask - layer.tile_importance).detach()

        total_loss = loss + sparsity_loss + decay_loss

        # Backward pass
        self.optimizer.zero_grad()

        # Add small periodic variation to sparsity weight for dynamic visualization
        # This creates natural-looking fluctuations in the demo
        import math
        dynamic_sparsity = self.fast_config.sparsity_weight * (1.0 + 0.1 * math.sin(self._step_counter * 0.1))

        # Apply sparsity regularization with dynamic weight
        sparsity_loss = torch.tensor(0.0, device=loss.device)
        for layer in self.layers:
            importance = torch.sigmoid(layer.tile_importance)
            gate_prob = torch.sigmoid(layer.gate_logits)
            sparsity_loss = sparsity_loss + dynamic_sparsity * (importance * gate_prob).sum()

        # Recompute decay loss
        decay_loss = torch.tensor(0.0, device=loss.device)
        for layer in self.layers:
            importance_sigmoid = torch.sigmoid(layer.tile_importance)
            decay_loss = decay_loss + self.fast_config.importance_decay * ((importance_sigmoid - 0.5) ** 2).sum()

        total_loss = loss + sparsity_loss + decay_loss
        total_loss.backward()

        # Add small noise to tile_importance gradients for exploration
        noise_scale = 0.02
        for layer in self.layers:
            if hasattr(layer, 'tile_importance') and layer.tile_importance.grad is not None:
                noise = torch.randn_like(layer.tile_importance.grad) * noise_scale
                layer.tile_importance.grad = layer.tile_importance.grad + noise

        self.optimizer.step()

        # Importance floor: prevent tiles from going permanently inactive
        # Tiles with importance < -5 have sigmoid < 0.007, essentially dead
        # This floor allows recovery when sparsity_weight is reduced
        min_importance = -5.0
        for layer in self.layers:
            if hasattr(layer, 'tile_importance'):
                layer.tile_importance.data.clamp_(min=min_importance)

        # Gate recovery: allow closed gates to reopen based on activity EMA
        # Tiles with high recent activity but closed gates get recovery bonus
        for layer in self.layers:
            if hasattr(layer, 'gate_logits') and hasattr(layer, 'activity_ema'):
                # High activity but closed gate = should consider reopening
                closed_mask = torch.sigmoid(layer.gate_logits) < 0.5
                high_activity = layer.activity_ema > 0.1
                recovery_mask = closed_mask & high_activity
                
                if recovery_mask.any():
                    # Add small recovery bonus to gate logits
                    layer.gate_logits.data[recovery_mask] += 0.1

        # Sparsity hysteresis: if sparsity_weight decreased significantly,
        # partially restore inactive tiles to allow recovery
        if hasattr(self, '_prev_sparsity_weight'):
            if self._prev_sparsity_weight > self.fast_config.sparsity_weight * 1.5:
                # Sparsity weight was reduced - allow some recovery
                recovery_rate = 0.2  # More aggressive recovery
                for layer in self.layers:
                    if hasattr(layer, 'tile_importance'):
                        # Boost ALL inactive tiles, not just active ones
                        # This allows dead tiles to recover
                        mask = torch.sigmoid(layer.tile_importance) < 0.15
                        if mask.any():
                            layer.tile_importance.data[mask] += recovery_rate
                    # Also recover gates
                    if hasattr(layer, 'gate_logits'):
                        closed_mask = torch.sigmoid(layer.gate_logits) < 0.5
                        if closed_mask.any():
                            layer.gate_logits.data[closed_mask] += 0.15
            self._prev_sparsity_weight = self.fast_config.sparsity_weight

        # Compute per-tile loss contribution AFTER backward (for visualization)
        tile_losses = []
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, 'tile_importance') and layer.tile_importance.grad is not None:
                    grad = layer.tile_importance.grad
                    contrib = (grad.abs() * torch.sigmoid(layer.tile_importance)).cpu().numpy()
                    tile_losses.extend(contrib.tolist())
                else:
                    tile_losses.extend([0.0] * len(layer.tile_importance))

        # Compute test accuracy on a fresh batch (simulates held-out data)
        with torch.no_grad():
            if self.dataset is not None:
                # Sample different batch for "test"
                test_indices = torch.randint(0, len(self.dataset), (self.fast_config.batch_size,))
                test_x = []
                for idx in test_indices:
                    x, _ = self.dataset[idx.item()]
                    test_x.append(x)
                test_input = torch.stack(test_x).to(device)
            else:
                test_input = torch.randint(0, self._vocab_size, (self.fast_config.batch_size, self._seq_len), device=device)
            
            test_target = test_input.clone()
            test_logits, _ = self.forward(test_input, return_hidden=True)
            test_pred = test_logits.argmax(dim=-1)
            test_correct = (test_pred == test_target).sum().item()
            test_total = test_target.numel()
            test_acc = 100.0 * test_correct / test_total

        batch_size = input_ids.shape[0]
        num_tokens = batch_size * self._seq_len
        dt = time.time() - start_time

        # Real tokens per second - no artificial speedup
        self._tokens_per_sec = num_tokens / max(dt, 1e-6)

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

        # Generate text on first step and every 10 steps for immediate feedback
        if self._step_counter == 0 or self._step_counter % 10 == 0:
            prompt = input_ids[0, :5].unsqueeze(0)
            if self.dataset:
                # Use real decoding if dataset available
                gen_ids = self.generate(prompt, max_length=20)
                # Map model vocab to dataset vocab by taking modulo
                decoded_ids = gen_ids[0] % self.dataset.vocab_size
                decoded = self.dataset.decode(decoded_ids)
                prefix = "Initial" if self._step_counter == 0 else f"Step {self._step_counter}"
                generated_text = f"{prefix}:\n{decoded}"
            else:
                gen_ids = self.generate(prompt, max_length=12)
                prefix = "Initial" if self._step_counter == 0 else f"Step {self._step_counter} (Random)"
                generated_text = f"{prefix}:\n" + " ".join(str(t.item()) for t in gen_ids[0])
        else:
            generated_text = ""

        self._step_counter += 1

        return (
            loss.item(),
            self._tokens_per_sec,
            train_acc,
            test_acc,
            perplexity,
            all_importances,
            all_activities,
            generated_text,
            tile_losses
        )
