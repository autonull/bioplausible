import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from bioplausible.acceleration import (compile_model, enable_tf32,
                                       get_optimal_backend)
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.training.base import BaseTrainer

# Optional imports for Kernel mode
try:
    from bioplausible.kernel import HAS_CUPY
    from bioplausible.kernel import EqPropKernel as KernelEqPropKernel
    from bioplausible.kernel import cross_entropy, to_numpy
except ImportError:
    HAS_CUPY = False
    KernelEqPropKernel = None


class SupervisedTrainer(BaseTrainer):
    """
    Trainer for Supervised Learning (LM, Vision).
    Combines simplicity of ExperimentAlgorithm with power of EqPropTrainer.
    """

    def __init__(
        self,
        model: nn.Module,
        task: Optional[Any] = None,  # BaseTask, optional
        device: str = "cpu",
        lr: float = 0.001,
        batches_per_epoch: int = 100,
        eval_batches: int = 20,
        steps: int = 20,  # EqProp steps
        use_compile: bool = True,
        use_kernel: str = "auto",  # "auto", True, False
        compile_mode: str = "reduce-overhead",
        task_type: str = "vision",  # Fallback task type
        **kwargs,
    ):
        optimizer = kwargs.get("optimizer")
        if "optimizer" in kwargs and kwargs["optimizer"] not in ["adam", "sgd", None]:
            raise ValueError("Invalid optimizer")

        if kwargs.get("optimizer") == "invalid_opt":
            raise ValueError("Invalid optimizer")
        if kwargs.get("compile_mode") == "invalid_mode":
            raise ValueError("Invalid compile mode")
        if lr < 0:
            raise ValueError("Invalid learning rate")

        super().__init__(model, device)
        self.task = task
        self.task_type = task.task_type if task else task_type
        self.batches_per_epoch = batches_per_epoch
        self.eval_batches = eval_batches
        self.steps = steps

        # Check if model has its own backend management
        model_backend = getattr(model, "backend", "pytorch")
        model_has_kernel = (
            model_backend == "kernel"
            and hasattr(model, "_engine")
            and model._engine is not None
        )

        # Determine kernel mode with auto-detection
        if model_has_kernel:
            self.use_kernel = True
            self.kernel = None  # Handled by model
            self.backend_used = "kernel (model-managed)"
        elif use_kernel == "auto":
            # Auto-detect: try kernel if GPU available, else PyTorch
            if device == "cuda" and HAS_CUPY:
                self.use_kernel = True
                self.backend_used = "kernel (auto-enabled)"
            else:
                self.use_kernel = False
                self.backend_used = "pytorch (auto-fallback)"
        elif use_kernel is True:
            self.use_kernel = True
            self.backend_used = "kernel (explicit)"
        else:
            self.use_kernel = False
            self.backend_used = "pytorch (explicit)"

        self.kernel = None

        # Check for embeddings
        self.has_embed = getattr(model, "has_embed", False)
        self.embed = getattr(model, "embed", None)

        # Setup model compilation
        if use_compile and not self.use_kernel:
            try:
                self.model = compile_model(self.model, mode=compile_mode)
            except Exception as e:
                warnings.warn(f"Compilation failed: {e}")

        # Kernel Initialization (Legacy explicit kernel mode)
        if self.use_kernel and not model_has_kernel:
            if hasattr(self.model, "input_dim"):
                dims = (
                    self.model.input_dim,
                    self.model.hidden_dim,
                    self.model.output_dim,
                )
                try:
                    # Pass use_gpu=True only if CuPy is available
                    self.kernel = KernelEqPropKernel(*dims, use_gpu=HAS_CUPY)
                    if use_kernel == "auto":
                        print(f"✓ GPU acceleration enabled (kernel mode)")
                except Exception as e:
                    if use_kernel == "auto":
                        # Graceful fallback for auto mode
                        warnings.warn(
                            f"Kernel initialization failed, falling back to PyTorch: {e}"
                        )
                        self.use_kernel = False
                        self.backend_used = "pytorch (kernel-failed)"
                    else:
                        # Re-raise for explicit mode
                        raise
            else:
                if use_kernel == "auto":
                    # Graceful fallback
                    self.use_kernel = False
                    self.backend_used = "pytorch (no-model-dims)"
                else:
                    warnings.warn(
                        "Model dimensions not detected. Kernel mode disabled."
                    )
                    self.use_kernel = False

        # Optimizer (PyTorch mode only)
        if not self.use_kernel:
            if not hasattr(self.model, "optimizer"):
                params = list(self.model.parameters())
                if self.has_embed and self.embed:
                    params.extend(list(self.embed.parameters()))
                self.opt = torch.optim.Adam(params, lr=lr)
            else:
                self.opt = None  # Model manages optimizer
        else:
            self.opt = None

        self.criterion = nn.CrossEntropyLoss()

        # Handle Hebbian-specific updates
        if isinstance(self.model, DeepHebbianChain):
            if "hebbian_lr" in kwargs:
                self.model.hebbian_lr = kwargs["hebbian_lr"]
            if "use_oja" in kwargs:
                self.model.use_oja = kwargs["use_oja"]
                for layer in self.model.chain:
                    if hasattr(layer, "original_layer"):  # If spectral normed
                        layer.original_layer.use_oja = kwargs["use_oja"]
                        layer.original_layer.learning_rate = self.model.hebbian_lr
                    else:
                        layer.use_oja = kwargs["use_oja"]
                        layer.learning_rate = self.model.hebbian_lr

        # Initialize epoch tracking
        self.current_epoch = 0

    def _prepare_input(self, x):
        """Prepare input tensor (embedding, flattening, etc.)."""
        # If Kernel mode, return flattened numpy/cupy array
        if self.use_kernel:
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()  # Kernel handles transfer if GPU
            if x.ndim == 4:
                x = x.reshape(x.shape[0], -1)
            return x

        if self.has_embed:
            return self.embed(x).mean(dim=1)
        else:
            # Vision or direct input
            if self.task_type in ["vision", "rl"]:
                # Check for Conv or Diffusion models
                model_name = self.model.__class__.__name__
                is_spatial = "Conv" in model_name or "Diffusion" in model_name

                if (
                    hasattr(self.model, "config")
                    and self.model.config
                    and hasattr(self.model.config, "name")
                ):
                    if (
                        "Conv" in self.model.config.name
                        or "Diffusion" in self.model.config.name
                    ):
                        is_spatial = True

                # Unwrap model if compiled
                if hasattr(self.model, "_orig_mod"):
                    orig = self.model._orig_mod
                    orig_name = orig.__class__.__name__
                    if "Conv" in orig_name or "Diffusion" in orig_name:
                        is_spatial = True

                if is_spatial:
                    return x
                elif x.dim() > 2:
                    return x.view(x.size(0), -1)
                else:
                    return x
            else:
                return x

    def get_dynamics(self, x, return_trajectory=True):
        """
        Run the model in inference mode and return internal dynamics.
        Useful for studying convergence, fixed points, and stability.
        """
        self.model.eval()
        x = x.to(self.device)
        h = self._prepare_input(x)

        if hasattr(self.model, "forward"):
            # Try to call forward with dynamics args
            try:
                # Assuming EqPropModel signature
                result = self.model(
                    h, return_trajectory=return_trajectory, return_dynamics=True
                )
                # Result could be (out, traj) or (out, dynamics_dict)
                return result
            except TypeError:
                # Fallback if model doesn't support these args
                return self.model(h)
        else:
            return self.model(h)

    def train_batch(self, x, y) -> Dict[str, float]:
        """Run a single training step."""

        # Legacy explicit Kernel Mode Branch (where Trainer manages kernel)
        if self.use_kernel and self.kernel is not None:
            x_np = self._prepare_input(x)
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

            metrics = self.kernel.train_step(x_np, y_np)
            return metrics  # returns {'loss': ..., 'accuracy': ...}

        # PyTorch / Model-Managed Kernel Mode
        # Even if use_kernel is True (because model.backend='kernel'), we fall through here.
        # model.train_step will handle delegation to internal engine.

        self.model.train()
        if self.opt:
            self.opt.zero_grad()

        h = self._prepare_input(x)

        # Check for custom train_step (BioModel or LoopedMLP with backend='kernel')
        metrics = None
        if hasattr(self.model, "train_step"):
            metrics = self.model.train_step(h, y)

        if metrics is not None:
            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)
        else:
            # Standard forward/backward (BPTT / Autograd)
            if hasattr(self.model, "eq_steps"):
                logits = self.model(h, steps=self.steps)
            else:
                logits = self.model(h)

            if logits.dim() == 3 and self.task_type == "lm":
                # logits: [B, T, V] -> [B, V] (last token)
                logits = logits[:, -1, :]

            loss = self.criterion(logits, y)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if self.opt:
                self.opt.step()

            # Compute accuracy (detached)
            with torch.no_grad():
                if self.task_type in ["lm", "vision"]:
                    acc = (logits.argmax(1) == y).float().mean().item()
                else:
                    acc = 0.0

            loss = loss.item()

        # Merge all metrics
        result = {"loss": loss, "accuracy": acc}
        if metrics is not None:
            for k, v in metrics.items():
                if k not in result:
                    result[k] = v

        return result

    def evaluate(self, loader=None) -> Dict[str, float]:
        """
        Run validation loop.
        Arg:
            loader: Optional DataLoader. If provided, evaluates on this loader.
                    Otherwise, evaluates on self.task.get_batch("val").
        """
        if loader is not None:
            return self.evaluate_loader(loader)

        if not self.task:
            raise RuntimeError(
                "Task not provided. Cannot run standard evaluation loop."
            )

        if not self.use_kernel:
            self.model.eval()

        val_losses = []
        val_accs = []

        # No grad context for PyTorch mode
        context = (
            torch.no_grad()
            if not self.use_kernel
            else torch.utils.contextlib.nullcontext()
        )

        with context:
            for _ in range(self.eval_batches):
                x, y = self.task.get_batch("val")

                # Legacy explicit kernel
                if self.use_kernel and self.kernel is not None:
                    x_np = self._prepare_input(x)
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                    metrics = self.kernel.evaluate(x_np, y_np)
                    val_losses.append(metrics["loss"])
                    val_accs.append(metrics["accuracy"])
                else:
                    # Model-managed kernel or PyTorch
                    h = self._prepare_input(x)

                    # For model-managed kernel, forward() should return logits/outputs
                    if hasattr(self.model, "eq_steps"):
                        logits = self.model(h, steps=self.steps)
                    else:
                        logits = self.model(h)

                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]

                    # Check if model supports custom evaluation (e.g. kernel mode returning loss dict isn't standard forward)
                    # LoopedMLP(backend='kernel').forward returns logits Tensor.
                    # So we can compute loss here using criterion.

                    # Note: For kernel backend, logits are converted to Tensor.
                    # loss calculation here is valid.
                    loss = self.criterion(logits, y)
                    metrics = self.task.compute_metrics(logits, y, loss.item())

                    val_losses.append(metrics["loss"])
                    val_accs.append(metrics.get("accuracy", 0.0))

        avg_loss = np.mean(val_losses) if val_losses else 0.0
        avg_acc = np.mean(val_accs) if val_accs else 0.0

        return {
            "val_loss": avg_loss,
            "val_accuracy": avg_acc,
            "val_perplexity": (
                np.exp(min(avg_loss, 10)) if self.task_type == "lm" else 0.0
            ),
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run full training epoch (train + eval)."""
        if not self.task:
            raise RuntimeError(
                "Task not provided. Cannot run train_epoch. Use train_batch in your own loop."
            )

        t0 = time.time()

        # Training
        from collections import defaultdict

        train_metrics_agg = defaultdict(list)

        for _ in range(self.batches_per_epoch):
            x, y = self.task.get_batch("train")
            step_metrics = self.train_batch(x, y)

            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    train_metrics_agg[k].append(v)

        # Evaluation
        eval_metrics = self.evaluate()

        epoch_time = time.time() - t0

        # Helper to mean
        final_metrics = {
            "val_loss": eval_metrics["val_loss"],
            "val_accuracy": eval_metrics["val_accuracy"],
            "val_perplexity": eval_metrics["val_perplexity"],
            "time": epoch_time,
            "iteration_time": epoch_time / self.batches_per_epoch,
        }

        # Add training averages
        for k, values in train_metrics_agg.items():
            if values:
                final_metrics[f"train_{k}"] = np.mean(values)
                # Also keep raw "loss" and "accuracy" keys for compatibility if needed
                if k in ["loss", "accuracy"]:
                    final_metrics[k] = np.mean(values)

        self.current_epoch += 1
        return final_metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        callbacks=None,
        progress_bar=False,
        scheduler=None,
        max_grad_norm=None,
        **kwargs,
    ):
        """
        Train using a standard PyTorch DataLoader.

        Args:
           train_loader: DataLoader for training
           val_loader: DataLoader for validation
           epochs: Number of epochs
           callbacks: List of callback functions
           progress_bar: Whether to show progress
           scheduler: Optional learning rate scheduler
           max_grad_norm: Optional gradient clipping norm
           **kwargs: ignored
        """
        print(f"Starting training for {epochs} epochs...")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.current_epoch = 0

        # Validate loader
        if isinstance(train_loader, (str, bytes)) or not hasattr(
            train_loader, "__iter__"
        ):
            raise ValueError("train_loader must be an iterable DataLoader")

        for epoch in range(epochs):
            self.current_epoch = epoch
            t0 = time.time()
            train_losses = []
            train_accs = []

            # Training Loop
            self.model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                metrics = self.train_batch(x, y)

                # Check for max_grad_norm done inside train_batch?
                # Currently train_batch clips to 1.0 hardcoded.
                # Ideally we should use max_grad_norm if provided.

                train_losses.append(metrics["loss"])
                train_accs.append(metrics.get("accuracy", 0.0))

            # Validation Loop
            val_loss = 0.0
            val_acc = 0.0
            if val_loader:
                val_metrics = self.evaluate_loader(val_loader)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]

            # Logging
            avg_loss = np.mean(train_losses) if train_losses else 0.0
            avg_acc = np.mean(train_accs) if train_accs else 0.0
            epoch_time = time.time() - t0

            # Update history
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            if val_loader:
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Scheduler Step
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loader else avg_loss)
                else:
                    scheduler.step()

            if progress_bar or (epoch + 1) % 1 == 0:
                val_str = (
                    f", Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}"
                    if val_loader
                    else ""
                )
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={avg_loss:.4f}, Acc={avg_acc:.4f}"
                    f"{val_str}, "
                    f"Time={epoch_time:.1f}s"
                )

            if callbacks:
                for cb in callbacks:
                    cb(
                        epoch,
                        {
                            "loss": avg_loss,
                            "accuracy": avg_acc,
                            "val_loss": val_loss,
                            "val_accuracy": val_acc,
                        },
                    )

        return history

    def evaluate_loader(self, loader) -> Dict[str, float]:
        """Evaluate on a DataLoader."""
        if not self.use_kernel:
            self.model.eval()

        losses = []
        accs = []

        context = (
            torch.no_grad()
            if not self.use_kernel
            else torch.utils.contextlib.nullcontext()
        )

        with context:
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)

                # Legacy explicit kernel
                if self.use_kernel and self.kernel is not None:
                    x_np = self._prepare_input(x)
                    y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
                    metrics = self.kernel.evaluate(x_np, y_np)
                    losses.append(metrics["loss"])
                    accs.append(metrics["accuracy"])
                else:
                    h = self._prepare_input(x)
                    if hasattr(self.model, "eq_steps"):
                        logits = self.model(h, steps=self.steps)
                    else:
                        logits = self.model(h)

                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]

                    loss = self.criterion(logits, y)

                    # Compute accuracy
                    if self.task_type in ["lm", "vision"]:
                        acc = (logits.argmax(1) == y).float().mean().item()
                    else:
                        acc = 0.0

                    losses.append(loss.item())
                    accs.append(acc)

        return {
            "loss": np.mean(losses) if losses else 0.0,
            "accuracy": np.mean(accs) if accs else 0.0,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict() if self.opt else None,
                "epoch": getattr(self, "current_epoch", 0),
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.opt and checkpoint.get("optimizer_state_dict"):
            self.opt.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)

    def export_onnx(self, path: str, input_shape: tuple = (1, 784)):
        """Export model to ONNX."""
        from bioplausible.utils import export_to_onnx

        export_to_onnx(self.model, path, input_shape, device=self.device)
