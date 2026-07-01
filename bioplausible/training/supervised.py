import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from bioplausible.acceleration import compile_model
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.scientist.safety import SafetyConfig, SafetyWrapper
from bioplausible.tracking import ExperimentTracker
from bioplausible.training.base import BaseTrainer

# Optional imports for Kernel mode
try:
    from bioplausible.kernel import HAS_CUPY
    from bioplausible.kernel import EqPropKernel as KernelEqPropKernel
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
        tracker: Optional[ExperimentTracker] = None,
        grad_clip: Optional[float] = None,
        safety_config: Optional[SafetyConfig] = None,
        scheduler_type: Optional[str] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        track_energy: bool = False,
        ablation_tags: Optional[Dict[str, Any]] = None,
        output_dir: str = "",
        **kwargs,
    ):
        self.track_energy = track_energy
        from omegaconf import OmegaConf

        if hasattr(ablation_tags, "_is_dict") or hasattr(ablation_tags, "__iter__"):
            try:
                self.ablation_tags = OmegaConf.to_container(ablation_tags, resolve=True)
            except Exception:
                self.ablation_tags = (
                    dict(ablation_tags)
                    if isinstance(ablation_tags, dict)
                    else ablation_tags
                )
        else:
            self.ablation_tags = ablation_tags or {}
        self.output_dir = output_dir

        if (
            "optimizer" in kwargs
            and isinstance(kwargs["optimizer"], str)
            and kwargs["optimizer"]
            not in [
                "adam",
                "sgd",
                "rmsprop",
                "adamw",
                None,
            ]
        ):
            raise ValueError(f"Invalid optimizer string: {kwargs['optimizer']}")

        valid_compile_modes = ["default", "reduce-overhead", "max-autotune", None]
        if compile_mode not in valid_compile_modes:
            raise ValueError(f"Invalid compile mode: {compile_mode}")
        if lr < 0:
            raise ValueError("Invalid learning rate")

        super().__init__(model, device)
        self.tracker = tracker
        self.task = task
        self.task_type = task.task_type if task else task_type
        self.batches_per_epoch = batches_per_epoch
        self.eval_batches = eval_batches
        self.steps = steps
        self.grad_clip = grad_clip
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Initialize Safety Wrapper
        if safety_config:
            self.safety = SafetyWrapper(safety_config)
            print(f"🛡️ Safety enabled (grad_clip={safety_config.max_grad_norm})")
        else:
            self.safety = None

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
            # If model has custom train_step, use PyTorch and skip kernel auto-detect
            if hasattr(model, "train_step") and callable(model.train_step):
                self.use_kernel = False
                self.backend_used = "pytorch (custom-train-step)"
            elif device == "cuda" and HAS_CUPY:
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
                        print("✓ GPU acceleration enabled (kernel mode)")
                except Exception as e:
                    if use_kernel == "auto":
                        msg = (
                            "Kernel initialization failed,"
                            f" falling back to PyTorch: {e}"
                        )
                        warnings.warn(msg)
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
            # Check if optimizer instance is passed directly
            if "optimizer" in kwargs and not isinstance(kwargs["optimizer"], str):
                self.opt = kwargs["optimizer"]
            elif not hasattr(self.model, "optimizer"):
                params = list(self.model.parameters())
                if self.has_embed and self.embed:
                    params.extend(list(self.embed.parameters()))

                opt_name = kwargs.get("optimizer", "adam")
                weight_decay = kwargs.get("weight_decay", 0.0)
                momentum = kwargs.get("momentum", 0.0)

                if opt_name == "sgd":
                    self.opt = torch.optim.SGD(
                        params, lr=lr, momentum=momentum, weight_decay=weight_decay
                    )
                elif opt_name == "rmsprop":
                    self.opt = torch.optim.RMSprop(
                        params, lr=lr, weight_decay=weight_decay, momentum=momentum
                    )
                elif opt_name == "adamw":
                    self.opt = torch.optim.AdamW(
                        params, lr=lr, weight_decay=weight_decay
                    )
                else:
                    self.opt = torch.optim.Adam(
                        params, lr=lr, weight_decay=weight_decay
                    )
            else:
                self.opt = None  # Model manages optimizer
        else:
            self.opt = None

        # Scheduler
        self.scheduler = None
        if self.opt and self.scheduler_type:
            if self.scheduler_type == "cosine":
                T_max = self.scheduler_kwargs.get("T_max", 50)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.opt, T_max=T_max
                )
            elif self.scheduler_type == "step":
                step_size = self.scheduler_kwargs.get("step_size", 30)
                gamma = self.scheduler_kwargs.get("gamma", 0.1)
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.opt, step_size=step_size, gamma=gamma
                )
            elif self.scheduler_type == "plateau":
                patience = self.scheduler_kwargs.get("patience", 10)
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, patience=patience
                )
            print(f"📅 Scheduler enabled: {self.scheduler_type}")

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
        self.samples_seen = 0

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
            if x.dtype in [
                torch.float32, torch.float64, torch.float16, torch.bfloat16,
            ]:
                return x
            return self.embed(x).mean(dim=1)

        # Add handling for LM that doesn't define embeddings inside the model
        if (
            self.task_type == "lm"
            and isinstance(x, torch.Tensor)
            and x.dtype in [torch.long, torch.int]
        ):
            # For standard MLPs forced into LM tasks without embedding:
            # Instead of casting categorical indices to float, we must use a
            # proper one-hot or embedding. We pass the tensor as-is and let
            # the model fail or handle it if it implements an embed.
            return x

        else:
            # Vision or direct input
            if self.task_type in ["vision", "rl", "lm"]:
                # Unwrap model if compiled for name check
                model_to_check = self.model
                if hasattr(self.model, "_orig_mod"):
                    model_to_check = self.model._orig_mod

                model_name = model_to_check.__class__.__name__

                # For LM with Transformer, keep as Long/Int for embedding
                is_transformer = "Transformer" in model_name or "GPT" in model_name

                # Cast to float if tensor AND not a transformer/LM that needs indices
                if (
                    not is_transformer
                    and isinstance(x, torch.Tensor)
                    and x.dtype
                    not in [torch.float32, torch.float64, torch.float16, torch.bfloat16]
                ):
                    x = x.float()

                # Check for Conv or Diffusion models
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
        # Update samples seen
        if hasattr(x, "shape"):
            self.samples_seen += x.shape[0]
        elif hasattr(x, "__len__"):
            self.samples_seen += len(x)

        # Legacy explicit Kernel Mode Branch (where Trainer manages kernel)
        if self.use_kernel and self.kernel is not None:
            x_np = self._prepare_input(x)
            y_np = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

            metrics = self.kernel.train_step(x_np, y_np)
            return metrics  # returns {'loss': ..., 'accuracy': ...}

        # PyTorch / Model-Managed Kernel Mode
        # Even if use_kernel is True (model.backend='kernel'), we fall through.
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
        elif (
            self.opt
            and hasattr(self.opt, "step")
            and (
                "target" in self.opt.step.__code__.co_varnames
                or "y" in self.opt.step.__code__.co_varnames
                or self.opt.__class__.__name__ == "CompositeOptimizer"
            )
        ):
            # MEP / Bio-plausible optimizer that handles backward internally
            # Expects step(x, target) or similar
            if self.opt.__class__.__name__ == "CompositeOptimizer":
                metrics_opt = self.opt.step(x=h, target=y)
            elif "target" in self.opt.step.__code__.co_varnames:
                metrics_opt = self.opt.step(x=h, target=y)
            else:
                metrics_opt = self.opt.step(x=h, y=y)  # Some variants use y

            if metrics_opt is None:
                metrics_opt = {}
            loss = metrics_opt.get("loss", 0.0)
            acc = metrics_opt.get("accuracy", 0.0)
            # Re-fetch accuracy if not provided by optimizer but model ran forward
            if acc == 0.0 and self.task_type in ["lm", "vision"]:
                with torch.no_grad():
                    logits = self.model(h)
                    if logits.dim() == 3 and self.task_type == "lm":
                        logits = logits[:, -1, :]
                    acc = (logits.argmax(1) == y).float().mean().item()
        else:
            # Standard forward/backward (BPTT / Autograd)
            if hasattr(self.model, "eq_steps"):
                logits = self.model(h, steps=self.steps)
            else:
                logits = self.model(h)

            if logits.dim() == 3 and self.task_type == "lm":
                # logits: [B, T, V] -> [B, V] (last token)
                logits = logits[:, -1, :]

            loss_val = self.criterion(logits, y)

            # --- SAFETY WRAPPER INTEGRATION ---
            if hasattr(self, "safety") and self.safety and self.opt:
                success, info = self.safety.safe_backward_and_step(
                    loss_val, self.opt, self.model, getattr(self, "grad_clip", None)
                )

                loss = info.get("loss", float(loss_val))

                if not success:
                    if self.safety.should_abort():
                        raise RuntimeError(
                            f"Training aborted by SafetyWrapper: {info.get('error')}"
                        )

                    self.safety.handle_failure(self.opt)
                    # Return fail metrics (high loss, 0 acc)
                    return {"loss": float(loss), "accuracy": 0.0}
            else:
                # Fallback to standard
                loss_val.backward()

                # Apply Grad Clipping
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                if self.opt:
                    self.opt.step()
                loss = loss_val.item()

            # Compute accuracy (detached)
            with torch.no_grad():
                if self.task_type in ["lm", "vision"]:
                    acc = (logits.argmax(1) == y).float().mean().item()
                else:
                    acc = 0.0

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

        import contextlib

        # No grad context for PyTorch mode
        context = torch.no_grad() if not self.use_kernel else contextlib.nullcontext()

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
                elif hasattr(self.model, "val_step"):
                    # Custom validation step (e.g. for Diffusion/Generative models)
                    h = self._prepare_input(x)
                    metrics = self.model.val_step(h, y)
                    val_losses.append(metrics.get("loss", 0.0))
                    val_accs.append(metrics.get("accuracy", 0.0))
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

                    # Check if model supports custom evaluation (e.g. kernel mode
                    # returning loss dict isn't standard forward)
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
                "Task not provided. Cannot run train_epoch."
                " Use train_batch in your own loop."
            )

        t0 = time.time()

        # Training
        from collections import defaultdict

        train_metrics_agg = defaultdict(list)

        for _ in range(self.batches_per_epoch):
            x, y = self.task.get_batch("train")

            if self.track_energy:
                from bioplausible.energy import EnergyTracker
                from bioplausible.models.registry import get_model_spec

                requires_backward = True
                if hasattr(self.model, "algorithm_name"):
                    try:
                        spec = get_model_spec(self.model.algorithm_name)
                        requires_backward = spec.requires_backward
                    except ValueError:
                        pass

                with EnergyTracker(
                    self.model, requires_backward=requires_backward
                ) as et:
                    step_metrics = self.train_batch(x, y)

                # Update metrics with profile data
                if et.profile:
                    step_metrics["energy_proxy"] = et.profile.energy_proxy
                    step_metrics["forward_flops"] = et.profile.forward_flops
                    step_metrics["backward_flops"] = et.profile.backward_flops
                    step_metrics["wall_time_ms"] = et.profile.wall_time_ms
                    step_metrics["peak_memory_mb"] = et.profile.peak_memory_mb
                    step_metrics["requires_backward"] = int(
                        et.profile.requires_backward
                    )
                    self.last_profile = et.profile
            else:
                step_metrics = self.train_batch(x, y)

            for k, v in step_metrics.items():
                if isinstance(v, (int, float)):
                    train_metrics_agg[k].append(v)

        # Evaluation
        eval_metrics = self.evaluate()

        epoch_time = time.time() - t0

        # Scheduler Step
        if self.scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(eval_metrics["val_loss"])
            else:
                self.scheduler.step()

            # Log current LR
            if self.opt:
                eval_metrics["learning_rate"] = self.opt.param_groups[0]["lr"]

        # Helper to mean
        final_metrics = {
            "val_loss": eval_metrics["val_loss"],
            "val_accuracy": eval_metrics["val_accuracy"],
            "val_perplexity": eval_metrics["val_perplexity"],
            "time": epoch_time,
            "iteration_time": epoch_time / self.batches_per_epoch,
            "samples_seen": self.samples_seen,
        }

        if "learning_rate" in eval_metrics:
            final_metrics["learning_rate"] = eval_metrics["learning_rate"]

        # Add training averages
        for k, values in train_metrics_agg.items():
            if values:
                final_metrics[f"train_{k}"] = np.mean(values)
                # Keep "loss", "accuracy", energy keys for compatibility
                if k in [
                    "loss",
                    "accuracy",
                    "energy_proxy",
                    "forward_flops",
                    "backward_flops",
                    "wall_time_ms",
                    "peak_memory_mb",
                    "requires_backward",
                ]:
                    final_metrics[k] = np.mean(values)

        if self.tracker:
            self.tracker.log_metrics(final_metrics, step=self.current_epoch)

        if self.output_dir:
            import json
            import os

            from omegaconf import OmegaConf

            os.makedirs(self.output_dir, exist_ok=True)

            clean_tags = self.ablation_tags
            if hasattr(clean_tags, "_is_dict") or hasattr(clean_tags, "__iter__"):
                try:
                    clean_tags = OmegaConf.to_container(clean_tags, resolve=True)
                except Exception:
                    clean_tags = (
                        dict(clean_tags) if isinstance(clean_tags, dict) else clean_tags
                    )
            else:
                pass

            log_line = {
                "epoch": self.current_epoch,
                "model": getattr(
                    self.model, "algorithm_name", self.model.__class__.__name__
                ),
                "task": self.task_type,
                "val_accuracy": eval_metrics.get("val_accuracy", 0.0),
                "val_loss": eval_metrics.get("val_loss", 0.0),
                "tags": clean_tags,
            }
            if self.track_energy and hasattr(self, "last_profile"):
                log_line.update(
                    {
                        "forward_flops": self.last_profile.forward_flops,
                        "backward_flops": self.last_profile.backward_flops,
                        "energy_proxy": self.last_profile.energy_proxy,
                        "wall_time_ms": self.last_profile.wall_time_ms,
                        "peak_memory_mb": self.last_profile.peak_memory_mb,
                        "requires_backward": self.last_profile.requires_backward,
                    }
                )

            with open(os.path.join(self.output_dir, "runs.jsonl"), "a") as f:
                f.write(json.dumps(log_line) + "\\n")

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
        early_stopping_patience=None,
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
            early_stopping_patience: Optional patience for early stopping
                (epochs with no improvement)
           **kwargs: ignored
        """
        print(f"Starting training for {epochs} epochs...")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.current_epoch = 0

        # Early Stopping State
        best_val_loss = float("inf")
        patience_counter = 0

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

            # Tracker Integration
            if self.tracker:
                metrics = {
                    "train_loss": avg_loss,
                    "train_accuracy": avg_acc,
                    "time": epoch_time,
                }
                if val_loader:
                    metrics["val_loss"] = val_loss
                    metrics["val_accuracy"] = val_acc
                self.tracker.log_metrics(metrics, step=epoch)

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

            # Early Stopping Check
            if val_loader and early_stopping_patience:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"Early stopping triggered after {epoch+1} epochs.")
                        history["stopped_early"] = True
                        break

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
                elif hasattr(self.model, "val_step"):
                    # Custom validation step
                    h = self._prepare_input(x)
                    metrics = self.model.val_step(h, y)
                    losses.append(metrics.get("loss", 0.0))
                    accs.append(metrics.get("accuracy", 0.0))
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
