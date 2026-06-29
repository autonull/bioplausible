"""
BioLightningModule

Wraps bioplausible models and optimizers into a PyTorch Lightning
Module with manual optimization support for EqProp, Hebbian,
FeedbackAlignment, and MEP-style optimizers.
"""

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn

from bioplausible.models import create_model
from bioplausible.optimizers import create_optimizer

# Standard optimizers that follow PyTorch conventions
STANDARD_OPTIMIZERS = {"adam", "adamw", "sgd", "rmsprop"}


class BioLightningModule(pl.LightningModule):
    """
    LightningModule for biologically plausible learning rules.

    Because EqProp, Hebbian, and MEP optimizers do not follow the
    standard ``loss.backward()`` paradigm, this module disables
    *automatic* optimization and implements a manual ``training_step``
    that delegates to the model/optimizer native interfaces.

    Example:
        >>> module = BioLightningModule(
        ...     model_name="backprop_mlp",
        ...     optimizer_name="adam",
        ...     input_dim=784,
        ...     output_dim=10,
        ...     hidden_dim=256,
        ... )
        >>> trainer = Trainer(max_epochs=10)
        >>> trainer.fit(module, train_loader, val_loader)
    """

    def __init__(self, model_name: str, optimizer_name: str, **hparams):
        """
        Args:
            model_name: Registered model name (e.g. ``backprop_mlp``,
                ``looped_mlp``, ``equitile``).
            optimizer_name: Registered optimizer name (e.g. ``adam``,
                ``smep``, ``feedback_alignment``).
            **hparams: Forwarded to model constructor.
                Common keys: ``input_dim``, ``output_dim``, ``hidden_dim``,
                ``lr``, etc.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer_name = optimizer_name

        # Build model using string name
        self.model = create_model(model_name, **hparams)

        # Determine if we need manual optimization
        is_bio_optimizer = optimizer_name.lower() not in STANDARD_OPTIMIZERS
        self.automatic_optimization = not is_bio_optimizer

        # Store optimizer reference for manual stepping
        self._optimizer = None

        # Cache for energy-based metrics
        self._last_energy: float = 0.0

    def configure_optimizers(self):
        """Create and store the bioplausible optimizer."""
        self._optimizer = create_optimizer(
            self.model, self.optimizer_name, lr=self.hparams.get("lr", 1e-3)
        )
        return self._optimizer

    def configure_model(self) -> None:
        """Configure the model for training."""
        self.model.train()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass – delegates to model."""
        return self.model(x, **kwargs)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Handles three cases:
        1. Model has custom train_step (e.g., EqProp models)
        2. Bio-plausible optimizer (x/target kwargs)
        3. Standard PyTorch optimizer (automatic by PL)
        """
        x, y = batch

        # Flatten vision inputs for MLP-style models
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        opt = self._optimizer

        # For bio-optimizers with manual optimization
        if not self.automatic_optimization:
            opt.zero_grad()

            # Check for model-specific train_step (EqProp models, etc.)
            if hasattr(self.model, "train_step"):
                metrics = self.model.train_step(x, y)
            else:
                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits, y)
                acc = (logits.argmax(dim=1) == y).float().mean()
                metrics = {"loss": loss, "accuracy": acc.item()}

            # Step optimizer manually
            opt.step()

            if metrics is None:
                metrics = {}

            loss = metrics.get("loss", 0.0)
            acc = metrics.get("accuracy", 0.0)

            self.log("train_loss", loss, prog_bar=True, on_step=True)
            self.log("train_acc", acc, prog_bar=True, on_step=True)

            return metrics.get("loss", torch.tensor(0.0))

        # Standard PyTorch training - return loss for automatic backward
        if hasattr(self.model, "train_step"):
            metrics = self.model.train_step(x, y)
        else:
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)

            acc = (logits.argmax(dim=1) == y).float().mean()
            metrics = {"loss": loss, "accuracy": acc.item()}

        if metrics is None:
            metrics = {}

        loss = metrics.get("loss", 0.0)
        acc = metrics.get("accuracy", 0.0)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True)

        return metrics.get("loss", torch.tensor(0.0))

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Validation step – standard forward + metric computation."""
        x, y = batch
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Test step – identical to validation."""
        return self.validation_step(batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        """Hook for epoch-level logging."""
        pass
