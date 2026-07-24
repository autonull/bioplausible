"""
Optimized Training Loop for FastLMEquiTile
===========================================

Implements state-of-the-art training optimizations:
- Mixed Precision (AMP) with loss scaling
- Gradient checkpointing for memory efficiency
- Cosine learning rate schedule with warmup
- Gradient accumulation for effective large batches
- AdamW with decoupled weight decay
- Token masking for variable-length sequences

Example
-------
>>> from bioplausible.equitile.lm_demo import LMTrainer, TrainingConfig
>>> config = TrainingConfig(
...     epochs=10,
...     learning_rate=3e-4,
...     warmup_steps=100,
...     use_amp=True,
...     gradient_accumulation_steps=4,
... )
>>> trainer = LMTrainer(model, config, device='cuda')
>>> trainer.train(train_loader, val_loader)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import torch

# Use new torch.amp API (2.0+) or fallback to deprecated cuda.amp
try:
    from torch.amp import GradScaler
    from torch.amp import autocast
except ImportError:
    from torch.cuda.amp import GradScaler
    from torch.cuda.amp import autocast

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import DataLoader

    from .fast_lm import FastLMEquiTile


# =============================================================================
# Training Configuration
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for LM training.

    Training Loop
    -------------
    epochs : int
        Number of training epochs
    learning_rate : float
        Peak learning rate
    warmup_steps : int
        Warmup steps for LR schedule
    weight_decay : float
        Weight decay for AdamW

    Optimization
    ------------
    use_amp : bool
        Use automatic mixed precision
    gradient_accumulation_steps : int
        Steps to accumulate gradients
    gradient_clip : float
        Gradient clipping norm

    Schedule
    --------
    lr_schedule : str
        LR schedule type ('cosine', 'linear', 'constant')
    min_lr_ratio : float
        Minimum LR ratio for cosine schedule

    Checkpointing
    -------------
    checkpoint_dir : str
        Directory for checkpoints
    save_every : int
        Save checkpoint every N steps
    eval_every : int
        Evaluate every N steps

    Logging
    -------
    log_every : int
        Log metrics every N steps
    generate_every : int
        Generate samples every N steps

    Hardware
    --------
    device : str
        Device to train on
    num_workers : int
        Number of data workers
    """

    # Training loop
    epochs: int = 10
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1

    # Optimization
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clip: float = 1.0

    # Schedule
    lr_schedule: str = "cosine"
    min_lr_ratio: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 500
    eval_every: int = 100

    # Logging
    log_every: int = 10
    generate_every: int = 200

    # Hardware
    device: str = "auto"
    num_workers: int = 4

    def __post_init__(self) -> None:
        """Validate and set defaults."""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Training Metrics
# =============================================================================


@dataclass
class TrainingMetrics:
    """Training metrics tracker.

    Tracks and aggregates training statistics for logging and visualization.
    """

    # Loss tracking
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_perplexity: List[float] = field(default_factory=list)
    val_perplexity: List[float] = field(default_factory=list)

    # Learning rate
    learning_rates: List[float] = field(default_factory=list)

    # Throughput
    tokens_per_second: List[float] = field(default_factory=list)
    samples_per_second: List[float] = field(default_factory=list)

    # Steps
    global_step: int = 0
    epoch: int = 0

    # Best metrics
    best_val_loss: float = float("inf")
    best_val_step: int = 0

    # Tile statistics (for analysis)
    tile_importance_history: List[List[float]] = field(default_factory=list)

    def update(
        self,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        lr: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        samples_per_sec: Optional[float] = None,
    ) -> None:
        """Update metrics."""
        if train_loss is not None:
            self.train_loss.append(train_loss)
            self.train_perplexity.append(
                math.exp(train_loss) if train_loss > 0 else float("inf")
            )

        if val_loss is not None:
            self.val_loss.append(val_loss)
            self.val_perplexity.append(
                math.exp(val_loss) if val_loss > 0 else float("inf")
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_step = self.global_step

        if lr is not None:
            self.learning_rates.append(lr)

        if tokens_per_sec is not None:
            self.tokens_per_second.append(tokens_per_sec)

        if samples_per_sec is not None:
            self.samples_per_second.append(samples_per_sec)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
            "current_train_loss": self.train_loss[-1] if self.train_loss else None,
            "current_val_loss": self.val_loss[-1] if self.val_loss else None,
            "current_train_ppl": (
                self.train_perplexity[-1] if self.train_perplexity else None
            ),
            "current_val_ppl": self.val_perplexity[-1] if self.val_perplexity else None,
        }

    def save(self, path: str) -> None:
        """Save metrics to file."""
        data = {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_perplexity": self.train_perplexity,
            "val_perplexity": self.val_perplexity,
            "learning_rates": self.learning_rates,
            "tokens_per_second": self.tokens_per_second,
            "samples_per_second": self.samples_per_second,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_step": self.best_val_step,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingMetrics":
        """Load metrics from file."""
        with open(path, "r") as f:
            data = json.load(f)

        metrics = cls()
        metrics.train_loss = data.get("train_loss", [])
        metrics.val_loss = data.get("val_loss", [])
        metrics.train_perplexity = data.get("train_perplexity", [])
        metrics.val_perplexity = data.get("val_perplexity", [])
        metrics.learning_rates = data.get("learning_rates", [])
        metrics.tokens_per_second = data.get("tokens_per_second", [])
        metrics.samples_per_second = data.get("samples_per_second", [])
        metrics.global_step = data.get("global_step", 0)
        metrics.epoch = data.get("epoch", 0)
        metrics.best_val_loss = data.get("best_val_loss", float("inf"))
        metrics.best_val_step = data.get("best_val_step", 0)

        return metrics


# =============================================================================
# Learning Rate Schedules
# =============================================================================


class LRScheduler:
    """Learning rate scheduler with warmup.

    Supports cosine, linear, and constant schedules.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    peak_lr : float
        Peak learning rate
    warmup_steps : int
        Number of warmup steps
    total_steps : int
        Total training steps
    schedule_type : str
        Schedule type ('cosine', 'linear', 'constant')
    min_lr_ratio : float
        Minimum LR ratio for cosine/linear
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        warmup_steps: int,
        total_steps: int,
        schedule_type: str = "cosine",
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        self.min_lr = peak_lr * min_lr_ratio
        self.current_step = 0

    def step(self) -> float:
        """Update learning rate and return new value."""
        self.current_step += 1
        lr = self._get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def _get_lr(self) -> float:
        """Calculate learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.peak_lr * (self.current_step / max(1, self.warmup_steps))

        # Calculate progress through decay phase
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, progress)

        if self.schedule_type == "cosine":
            # Cosine decay
            return self.min_lr + (self.peak_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        elif self.schedule_type == "linear":
            # Linear decay
            return self.peak_lr - (self.peak_lr - self.min_lr) * progress
        else:
            # Constant
            return self.peak_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self._get_lr()


# =============================================================================
# LM Trainer
# =============================================================================


class LMTrainer:
    """Trainer for FastLMEquiTile with optimizations.

    Implements:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient checkpointing
    - Learning rate scheduling
    - Checkpointing and resume
    - Real-time metrics tracking

    Parameters
    ----------
    model : FastLMEquiTile
        Model to train
    config : TrainingConfig
        Training configuration
    """

    def __init__(
        self,
        model: FastLMEquiTile,
        config: TrainingConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model = self.model.to(self.device)

        # Mixed precision
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Metrics
        self.metrics = TrainingMetrics()

        # Callbacks
        self.on_step_callbacks: List[Callable] = []
        self.on_epoch_callbacks: List[Callable] = []

        # Generation prompt (set during training)
        self.gen_prompt: Optional[str] = None
        self.tokenizer = None

    def set_tokenizer(self, tokenizer) -> None:
        """Set tokenizer for generation."""
        self.tokenizer = tokenizer

    def set_generation_prompt(self, prompt: str) -> None:
        """Set prompt for periodic generation."""
        self.gen_prompt = prompt

    def add_on_step_callback(self, callback: Callable) -> None:
        """Add callback to be called after each step."""
        self.on_step_callbacks.append(callback)

    def add_on_epoch_callback(self, callback: Callable) -> None:
        """Add callback to be called after each epoch."""
        self.on_epoch_callbacks.append(callback)

    def _get_lr_scheduler(self, total_steps: int) -> LRScheduler:
        """Create learning rate scheduler."""
        return LRScheduler(
            self.optimizer,
            peak_lr=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps,
            schedule_type=self.config.lr_schedule,
            min_lr_ratio=self.config.min_lr_ratio,
        )

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
        max_batches: Optional[int] = None,
    ) -> float:
        """Evaluate model on validation set.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
        max_batches : int, optional
            Maximum batches to evaluate

        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(val_loader):
            if max_batches and batch_idx >= max_batches:
                break

            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)

            # Forward pass
            if self.use_amp:
                with autocast():
                    logits = self.model(input_ids)
                    loss = self.model.compute_loss(logits, target_ids)
            else:
                logits = self.model(input_ids)
                loss = self.model.compute_loss(logits, target_ids)

            total_loss += loss.item()
            n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def generate_sample(
        self,
        prompt: Optional[str] = None,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> str:
        """Generate text sample.

        Parameters
        ----------
        prompt : str, optional
            Generation prompt
        max_length : int
            Maximum generation length
        temperature : float
            Sampling temperature
        top_k : int
            Top-k sampling

        Returns
        -------
        str
            Generated text
        """
        if self.tokenizer is None:
            return "[No tokenizer set]"

        # Default prompt
        if prompt is None:
            prompt = "The "

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        output_ids = self.model.generate(
            input_tensor,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )

        # Decode
        generated = self.tokenizer.decode(output_ids[0].tolist())
        return generated

    def train_step(
        self,
        input_ids: Tensor,
        target_ids: Tensor,
    ) -> float:
        """Perform single training step with gradient accumulation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs
        target_ids : torch.Tensor
            Target token IDs

        Returns
        -------
        float
            Loss value
        """
        # Mixed precision forward pass
        if self.use_amp:
            with autocast():
                logits = self.model(input_ids)
                loss = self.model.compute_loss(logits, target_ids)
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Scaled backward
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = self.model.compute_loss(logits, target_ids)
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def optimizer_step(self) -> None:
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscales gradients and clips
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip,
            )
            self.optimizer.step()

        self.optimizer.zero_grad()

    def save_checkpoint(
        self,
        path: str,
        extra_data: Optional[Dict] = None,
    ) -> None:
        """Save training checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path
        extra_data : dict, optional
            Additional data to save
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": self.metrics.get_summary(),
            "config": vars(self.config),
            "global_step": self.metrics.global_step,
            "epoch": self.metrics.epoch,
        }

        if extra_data:
            checkpoint.update(extra_data)

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.metrics.global_step = checkpoint.get("global_step", 0)
        self.metrics.epoch = checkpoint.get("epoch", 0)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None,
    ) -> TrainingMetrics:
        """Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        resume_from : str, optional
            Checkpoint to resume from

        Returns
        -------
        TrainingMetrics
            Training metrics
        """
        # Resume if requested
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed from checkpoint: {resume_from}")

        # Setup
        self.model.train()
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.epochs

        # Learning rate scheduler
        scheduler = self._get_lr_scheduler(total_steps)

        # Create checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training loop
        start_time = time.time()

        for epoch in range(self.config.epochs):
            self.metrics.epoch = epoch + 1
            epoch_start = time.time()
            epoch_loss = 0.0
            n_steps = 0

            for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
                # Move to device
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)

                # Training step
                loss = self.train_step(input_ids, target_ids)
                epoch_loss += loss
                n_steps += 1

                # Gradient accumulation step
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer_step()
                    lr = scheduler.step()
                    self.metrics.global_step += 1

                    # Calculate throughput
                    elapsed = time.time() - start_time
                    tokens_processed = self.metrics.global_step * input_ids.numel()
                    tokens_per_sec = tokens_processed / max(0.001, elapsed)
                    samples_per_sec = (
                        self.metrics.global_step
                        * input_ids.shape[0]
                        / max(0.001, elapsed)
                    )

                    # Update metrics
                    self.metrics.update(
                        train_loss=loss,
                        lr=lr,
                        tokens_per_sec=tokens_per_sec,
                        samples_per_sec=samples_per_sec,
                    )

                    # Logging
                    if self.metrics.global_step % self.config.log_every == 0:
                        elapsed_epoch = time.time() - epoch_start
                        print(
                            f"Epoch {epoch + 1}/{self.config.epochs} | "
                            f"Step {self.metrics.global_step} | "
                            f"Loss: {loss:.4f} | "
                            f"PPL: {math.exp(loss):.2f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:.0f} | "
                            f"Time: {elapsed_epoch:.1f}s"
                        )

                    # Validation
                    if (
                        val_loader
                        and self.metrics.global_step % self.config.eval_every == 0
                    ):
                        val_loss = self.evaluate(val_loader)
                        self.metrics.update(val_loss=val_loss)
                        print(
                            f"  Validation Loss: {val_loss:.4f} | PPL: {math.exp(val_loss):.2f}"
                        )

                        # Save best checkpoint
                        if val_loss < self.metrics.best_val_loss:
                            self.save_checkpoint(
                                checkpoint_dir / "best_model.pt",
                                extra_data={"val_loss": val_loss},
                            )

                    # Generation
                    if (
                        self.gen_prompt
                        and self.metrics.global_step % self.config.generate_every == 0
                    ):
                        generated = self.generate_sample(
                            self.gen_prompt,
                            max_length=100,
                        )
                        print(f"  Generated: {generated[:80]}...")

                    # Checkpoint
                    if self.metrics.global_step % self.config.save_every == 0:
                        self.save_checkpoint(
                            checkpoint_dir
                            / f"checkpoint_{self.metrics.global_step}.pt",
                        )

                    # Callbacks
                    for callback in self.on_step_callbacks:
                        callback(self, self.metrics)

            # Epoch summary
            avg_loss = epoch_loss / max(1, n_steps)
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Epoch callbacks
            for callback in self.on_epoch_callbacks:
                callback(self, self.metrics)

        # Save final checkpoint
        self.save_checkpoint(checkpoint_dir / "final_model.pt")

        # Save metrics
        self.metrics.save(checkpoint_dir / "metrics.json")

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time / 60:.1f} minutes")
        print(
            f"Best validation loss: {self.metrics.best_val_loss:.4f} (step {self.metrics.best_val_step})"
        )

        return self.metrics


# =============================================================================
# Convenience Functions
# =============================================================================


def train_model(
    model: FastLMEquiTile,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    tokenizer=None,
    config: Optional[TrainingConfig] = None,
    resume_from: Optional[str] = None,
) -> TrainingMetrics:
    """Convenience function to train a model.

    Parameters
    ----------
    model : FastLMEquiTile
        Model to train
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader, optional
        Validation data loader
    tokenizer : Tokenizer, optional
        Tokenizer for generation
    config : TrainingConfig, optional
        Training configuration
    resume_from : str, optional
        Checkpoint to resume from

    Returns
    -------
    TrainingMetrics
        Training metrics
    """
    if config is None:
        config = TrainingConfig()

    trainer = LMTrainer(model, config)
    trainer.set_tokenizer(tokenizer)
    trainer.set_generation_prompt("The ")

    return trainer.train(train_loader, val_loader, resume_from)


def create_training_config(
    epochs: int = 10,
    learning_rate: float = 3e-4,
    batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    use_amp: bool = True,
    checkpoint_dir: str = "checkpoints",
    **kwargs,
) -> TrainingConfig:
    """Create training configuration with common settings.

    Parameters
    ----------
    epochs : int
        Number of epochs
    learning_rate : float
        Peak learning rate
    batch_size : int
        Batch size
    gradient_accumulation_steps : int
        Gradient accumulation steps
    use_amp : bool
        Use mixed precision
    checkpoint_dir : str
        Checkpoint directory
    **kwargs
        Additional configuration options

    Returns
    -------
    TrainingConfig
        Training configuration
    """
    return TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_amp=use_amp,
        checkpoint_dir=checkpoint_dir,
        **kwargs,
    )
