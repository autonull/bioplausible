import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bioplausible.datasets import create_data_loaders, get_lm_dataset
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec


class LiveModelWrapper:
    """
    Wraps any BioModel to provide a unified interface for the Live UI.
    Handles data loading, optimization, and state capture for visualization.
    """

    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        model_instance: Optional[nn.Module] = None,
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        try:
            self.spec = get_model_spec(model_name)
        except ValueError:
            # Fallback spec if unknown
            from bioplausible.models.registry import ModelSpec

            self.spec = ModelSpec(
                name=model_name, description="External Model", model_type="custom"
            )

        # Training state
        self.step_counter = 0
        self.tokens_per_sec = 0.0
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

        # Visualization state
        self.layer_activities = {}  # Map layer_idx -> reduced activity (cpu numpy)
        self.layer_full_activities = (
            {}
        )  # Map layer_idx -> full tensor (cpu numpy) for inspection
        self.layer_importances = {}  # Map layer_idx -> importance tensor (cpu numpy)
        self.layer_names = []
        self.layer_sizes = []  # List of ints

        # Setup
        self._setup_data()

        if model_instance:
            self.model = model_instance.to(self.device)
            # Infer properties if possible
            # But mostly we rely on hooks
        else:
            self._setup_model()

        self._setup_optimizer()
        self._attach_hooks()

    def _setup_data(self):
        """Initialize datasets and loaders based on task."""
        self.task_type = "lm" if "lm" in (self.spec.task_compat or []) else "vision"

        # Override based on config if provided
        if "task_type" in self.config:
            self.task_type = self.config["task_type"]

        print(f"Setting up data for task: {self.task_type}")

        if self.task_type == "lm":
            dataset_name = self.config.get("dataset_name", "Tiny Shakespeare")
            seq_len = self.config.get("max_seq_len", 64)
            self.dataset = get_lm_dataset(
                dataset_name.lower().replace(" ", "_"), seq_len=seq_len
            )
            self.vocab_size = self.dataset.vocab_size
            self.output_dim = self.vocab_size
            self.input_dim = None  # For embedding models

        elif self.task_type == "vision":
            dataset_name = self.config.get("dataset_name", "mnist").lower()
            batch_size = self.config.get("batch_size", 64)

            # For MLPs, we might need flattened input
            self.flatten = (
                "mlp" in self.model_name.lower()
                or "backprop" in self.model_name.lower()
            )

            self.train_loader, self.test_loader = create_data_loaders(
                dataset_name, batch_size=batch_size, flatten=self.flatten
            )

            # Determine dimensions from dataset
            sample, _ = self.train_loader.dataset[0]
            self.input_dim = sample.numel() if self.flatten else sample.shape

            # Number of classes
            if dataset_name in ["cifar10", "mnist", "fashion_mnist", "svhn"]:
                self.output_dim = 10
            elif dataset_name == "cifar100":
                self.output_dim = 100
            else:
                self.output_dim = 10

            self.data_iter = iter(self.train_loader)

    def _setup_model(self):
        """Create the model using the factory."""
        print(f"Creating model: {self.model_name}")

        # Prepare kwargs
        kwargs = {
            "num_layers": self.config.get("num_layers", 4),
            "hidden_dim": self.config.get("neurons_per_tile", 128),  # Map to hidden_dim
            "dropout": self.config.get("dropout", 0.0),
        }

        # Special handling for EquiTile parameters
        if "equitile" in self.model_name.lower():
            kwargs["tiles_per_layer"] = self.config.get("tiles_per_layer", 16)
            # For EquiTile, 'neurons_per_tile' is passed as hidden_dim to factory,
            # but EquiTile logic handles it.

        self.model = create_model(
            self.spec,
            input_dim=(
                self.input_dim if isinstance(self.input_dim, int) else None
            ),  # Pass int if flattened
            output_dim=self.output_dim,
            device=str(self.device),
            task_type=self.task_type,
            **kwargs,
        )

    def _setup_optimizer(self):
        """Setup optimizer and loss function."""
        # Some models (like EqProp) manage their own optimization in train_step
        # But we create one for BPTT fallback or outer loop

        lr = self.config.get("learning_rate", self.spec.default_lr)
        weight_decay = self.config.get("weight_decay", 1e-4)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if self.task_type == "lm" or self.task_type == "vision":
            self.criterion = nn.CrossEntropyLoss()

    def _attach_hooks(self):
        """
        Attach forward hooks to capture activations for visualization.
        Heuristic: Hook all Linear, Conv2d, or special layers.
        """
        self.layer_names = []
        self.layer_sizes = []
        self.hook_handles = []

        # Helper to decide if a module is "visualizable"
        def is_visualizable(module):
            return isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTMCell, nn.GRUCell))

        # Recursive search or iteration?
        # Many models have a .layers or .net structure.

        candidates = []

        if hasattr(self.model, "layers"):
            # EquiTile, Transformers
            for i, layer in enumerate(self.model.layers):
                candidates.append((f"Layer {i}", layer))
        elif hasattr(self.model, "net"):
            # MLP Sequential
            for i, layer in enumerate(self.model.net):
                if is_visualizable(layer):
                    candidates.append(
                        (f"Layer {i} ({layer.__class__.__name__})", layer)
                    )
        else:
            # Fallback: traverse named modules
            for name, module in self.model.named_modules():
                if is_visualizable(module) and module is not self.model:
                    # Avoid duplicates (if container is also visualizable?)
                    candidates.append((name, module))

        # Attach hooks
        for idx, (name, module) in enumerate(candidates):
            self.layer_names.append(name)

            # Determine size
            size = 0
            if isinstance(module, nn.Linear):
                size = module.out_features
            elif isinstance(module, nn.Conv2d):
                size = module.out_channels  # Visualize channels as tiles?
            elif hasattr(module, "hidden_size"):
                size = module.hidden_size
            elif hasattr(module, "config") and hasattr(
                module.config, "tiles_per_layer"
            ):
                size = module.config.tiles_per_layer
            else:
                size = 64  # Default?

            self.layer_sizes.append(size)

            # Define hook
            def get_hook(layer_idx):
                def hook(mod, inp, out):
                    # Store activation stats
                    with torch.no_grad():
                        # Handle tuple outputs (RNNs, Transformers)
                        if isinstance(out, tuple):
                            act = out[0]
                        else:
                            act = out

                        # Store full activity for inspection (CPU numpy)
                        # We take the first sample in batch to avoid huge memory usage,
                        # or mean over batch?
                        # Inspection usually wants to see structure of one sample.
                        # Let's take mean over batch for stability.
                        if act.dim() > 0:
                            full_act = act.mean(dim=0).detach().cpu().numpy()
                        else:
                            full_act = act.detach().cpu().numpy()

                        self.layer_full_activities[layer_idx] = full_act

                        # Reduce for grid visualization
                        if act.dim() > 1:
                            # Flatten if spatial (Conv)
                            if act.dim() == 4:  # [B, C, H, W] -> [C]
                                reduced = act.mean(dim=[0, 2, 3])
                            elif act.dim() == 3:  # [B, Seq, Hidden] -> [Hidden]
                                reduced = act.mean(dim=[0, 1])
                            else:  # [B, Hidden] -> [Hidden]
                                reduced = act.mean(dim=0)
                        else:
                            reduced = act

                        self.layer_activities[layer_idx] = (
                            reduced.detach().cpu().numpy()
                        )

                        # Importances
                        if hasattr(mod, "tile_importance"):
                            imp = (
                                torch.sigmoid(mod.tile_importance)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            self.layer_importances[layer_idx] = imp
                        else:
                            # Default importance = 1.0
                            self.layer_importances[layer_idx] = np.ones_like(
                                self.layer_activities[layer_idx]
                            )

                return hook

            handle = module.register_forward_hook(get_hook(idx))
            self.hook_handles.append(handle)

        print(f"Attached hooks to {len(self.layer_sizes)} layers: {self.layer_sizes}")

    def update_params(self, params: Dict[str, Any]):
        """Update training parameters."""
        if "learning_rate" in params:
            for g in self.optimizer.param_groups:
                g["lr"] = params["learning_rate"]

    def training_step(self) -> Dict[str, Any]:
        """
        Run one step of training and return metrics.
        Returns generic dict.
        """
        start_time = time.time()

        # 1. Get Batch
        if self.task_type == "lm":
            # Sample random batch from dataset
            batch_size = self.config.get("batch_size", 16)
            indices = torch.randint(0, len(self.dataset), (batch_size,))
            batch_x = []
            for idx in indices:
                x, _ = self.dataset[idx.item()]
                batch_x.append(x)
            x = torch.stack(batch_x).to(self.device)
            y = (
                x.clone()
            )  # Next token prediction target (usually handled by model or shifted)

            # Standard LM target: input x[:, :-1], target x[:, 1:]
            # But CharDataset returns x (seq), y (seq+1 shifted).
            # Let's check CharDataset.__getitem__
            # It returns x=data[i:i+seq], y=data[i+1:i+seq+1]

            # The indices sampling above gets (x, y) tuples
            batch_y = []
            batch_x_tensors = []
            for idx in indices:
                bx, by = self.dataset[idx.item()]
                batch_x_tensors.append(bx)
                batch_y.append(by)

            x = torch.stack(batch_x_tensors).to(self.device)
            y = torch.stack(batch_y).to(self.device)

        else:  # Vision
            try:
                x, y = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.train_loader)
                x, y = next(self.data_iter)

            x, y = x.to(self.device), y.to(self.device)
            if isinstance(self.input_dim, int) and self.flatten:
                x = x.view(x.size(0), -1)

        # 2. Forward & Train
        loss_val = 0.0
        acc_val = 0.0

        # Check for built-in train_step (EqProp)
        metrics = None
        if hasattr(self.model, "train_step"):
            # This handles forward/backward/opt internally
            metrics = self.model.train_step(x, y)

        if metrics:
            loss_val = metrics.get("loss", 0.0)
            acc_val = metrics.get("accuracy", 0.0) * 100.0
            perplexity = np.exp(loss_val) if self.task_type == "lm" else 0.0
        else:
            # Standard BPTT
            self.optimizer.zero_grad()

            # Forward
            if self.task_type == "lm":
                # LM models usually return logits, hidden or (logits, loss)
                # BioModel convention: forward(x) -> logits
                logits = self.model(x)

                # Check shape: [B, S, V] vs [B*S, V]
                if logits.dim() == 3:
                    B, S, V = logits.shape
                    logits_flat = logits.view(-1, V)
                    y_flat = y.view(-1)
                    loss = self.criterion(logits_flat, y_flat)
                else:
                    loss = self.criterion(logits, y)

            else:
                logits = self.model(x)
                loss = self.criterion(logits, y)

            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            perplexity = np.exp(loss_val) if self.task_type == "lm" else 0.0

            # Accuracy
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = (pred == y.view_as(pred)).sum().item()
                total = y.numel()
                acc_val = 100.0 * correct / total

        # 3. Metrics
        dt = time.time() - start_time
        num_items = x.numel()  # tokens or pixels
        self.tokens_per_sec = num_items / max(dt, 1e-6)
        self.step_counter += 1

        # 4. Text Generation (if LM)
        gen_text = ""
        if self.task_type == "lm" and (
            self.step_counter == 1 or self.step_counter % 10 == 0
        ):
            if hasattr(self.model, "generate"):
                # Use model's generate
                try:
                    seed = x[0, :5].unsqueeze(0)
                    gen_ids = self.model.generate(seed, max_length=20)
                    if isinstance(gen_ids, torch.Tensor):
                        decoded = self.dataset.decode(gen_ids[0])
                        gen_text = f"Step {self.step_counter}:\n{decoded}"
                except Exception as e:
                    gen_text = f"Gen Error: {str(e)}"

        # 5. Collect Visualization Data
        # Sort by layer index
        sorted_keys = sorted(self.layer_activities.keys())
        all_activities = [self.layer_activities[k] for k in sorted_keys]
        all_importances = [
            self.layer_importances.get(k, np.ones_like(self.layer_activities[k]))
            for k in sorted_keys
        ]

        # If no hooks fired (e.g. EqProp might not trigger hooks if forward is custom?), handle gracefully
        if not all_activities and hasattr(self.model, "layers"):
            # Try to pull from layers manually if they have 'state'
            pass

        return {
            "loss": loss_val,
            "tps": self.tokens_per_sec,
            "train_acc": acc_val,
            "test_acc": 0.0,  # TODO: Implement test step
            "perplexity": perplexity,
            "importances": all_importances,
            "activities": all_activities,
            "gen_text": gen_text,
            "tile_losses": [],  # Not generic yet
            "step": self.step_counter,
            "layer_sizes": self.layer_sizes,
            "layer_names": self.layer_names,
        }

    def get_tile_details(
        self, layer_idx: int, tile_idx: int
    ) -> Tuple[float, float, np.ndarray, bool]:
        """
        Get details for a specific unit/tile.
        Returns: (importance, avg_activity, detailed_activity, is_active)
        """
        imp = 1.0
        act = 0.0
        neurons = np.zeros(10)
        is_active = True

        # Importance
        if layer_idx in self.layer_importances:
            imps = self.layer_importances[layer_idx]
            if tile_idx < len(imps):
                imp = float(imps[tile_idx])

        # Activity (Scalar)
        if layer_idx in self.layer_activities:
            acts = self.layer_activities[layer_idx]
            if tile_idx < len(acts):
                act = float(acts[tile_idx])

        # Detailed Activity (Neurons inside tile?)
        # For MLP, a "tile" is a neuron. So details is just the scalar value.
        # For EquiTile, a tile has neurons.
        # For Conv, a tile is a channel. Details could be the spatial map?

        if layer_idx in self.layer_full_activities:
            full = self.layer_full_activities[layer_idx]
            # Shape depends on layer type
            # MLP: [Hidden] -> tile is index. No inner structure.
            # Conv: [C, H, W] -> tile is C index. Inner is [H, W].
            # EquiTile: [NumTiles, NeuronsPerTile] -> tile is index. Inner is [NeuronsPerTile].

            if full.ndim == 1:
                # MLP or 1D
                neurons = np.array([act])
            elif full.ndim == 2:
                # [NumTiles, Neurons] (EquiTile) or [Seq, Hidden]?
                # If EquiTile, full is likely [Tiles, Neurons] if we reshaped?
                # But hooks capture output of layer.
                # EquiTile layer output is [Batch, Seq, Hidden]. Reduced to [Hidden].
                # If Hidden = Tiles * NeuronsPerTile, we need to know NeuronsPerTile.

                # Check config
                neurons_per_tile = self.config.get("neurons_per_tile", 1)
                tiles_per_layer = self.config.get("tiles_per_layer", len(full))

                if (
                    full.shape[0] == tiles_per_layer
                    and full.shape[1] == neurons_per_tile
                ):
                    neurons = full[tile_idx]
                elif full.shape[0] > tile_idx:
                    neurons = full[tile_idx]  # Treat row as details?

            elif full.ndim == 3:
                # Conv [C, H, W]
                if tile_idx < full.shape[0]:
                    spatial = full[tile_idx]  # [H, W]
                    neurons = spatial.flatten()

        return imp, act, neurons, is_active

    def save_checkpoint(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.config,
                "step": self.step_counter,
            },
            path,
        )

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=False)
        self.step_counter = ckpt.get("step", 0)
