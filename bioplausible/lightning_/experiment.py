"""
Lightning-powered experiment runner for AutoScientist.

This module provides PL-based alternatives for the trial execution
in bioplausible.scientist.core.AutoScientist.
"""

import logging
from typing import Any
from typing import Dict
from typing import Optional

from pytorch_lightning import Trainer

from bioplausible.lightning_.module import BioLightningModule
from bioplausible.lightning_.strategies import build_trainer

logger = logging.getLogger("AutoScientist.PL")


def run_pl_trial(
    model_name: str,
    optimizer_name: str,
    config: Dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    quick_mode: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Execute a single trial using PyTorch Lightning.

    Args:
        model_name: Registered model name.
        optimizer_name: Registered optimizer name.
        config: Hyperparameter dict (lr, hidden_dim, etc.).
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        quick_mode: If True, run for fewer epochs.

    Returns:
        Metrics dict or None on failure.
    """
    epochs = config.get("epochs", 10)
    if quick_mode:
        epochs = min(epochs, 3)

    module = BioLightningModule(
        model_name=model_name,
        optimizer_name=optimizer_name,
        **config,
    )

    trainer = Trainer(
        max_epochs=epochs,
        enable_progress_bar=True,
        logger=False,
    )

    try:
        trainer.fit(module, train_loader, val_loader)
        metrics = trainer.callback_metrics
        if "val_acc" in metrics:
            val_acc = metrics["val_acc"]
            if hasattr(val_acc, "item"):
                val_acc = val_acc.item()
            val_loss = metrics.get("val_loss", 0)
            if hasattr(val_loss, "item"):
                val_loss = val_loss.item()
            return {
                "accuracy": val_acc,
                "loss": val_loss,
            }
        return {"accuracy": 0.0, "loss": 0.0}
    except Exception as e:
        logger.error(f"PL trial failed: {e}", exc_info=True)
        return None


def run_pl_trial_with_wandb(
    model_name: str,
    optimizer_name: str,
    config: Dict[str, Any],
    train_loader: Any,
    val_loader: Any,
    run_name: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """
    Execute a PL trial with W&B logging.

    Args:
        model_name: Registered model name.
        optimizer_name: Registered optimizer name.
        config: Hyperparameter dict.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        run_name: Optional W&B run name.

    Returns:
        Metrics dict or None on failure.
    """
    epochs = config.get("epochs", 10)

    module = BioLightningModule(
        model_name=model_name,
        optimizer_name=optimizer_name,
        **config,
    )

    trainer = build_trainer(
        optimizer_name=optimizer_name,
        max_epochs=epochs,
        enable_wandb=True,
    )

    try:
        trainer.fit(module, train_loader, val_loader)
        metrics = trainer.callback_metrics
        if "val_acc" in metrics:
            val_acc = metrics["val_acc"]
            if hasattr(val_acc, "item"):
                val_acc = val_acc.item()
            val_loss = metrics.get("val_loss", 0)
            if hasattr(val_loss, "item"):
                val_loss = val_loss.item()
            return {
                "accuracy": val_acc,
                "loss": val_loss,
            }
        return {"accuracy": 0.0, "loss": 0.0}
    except Exception as e:
        logger.error(f"PL+W&B trial failed: {e}", exc_info=True)
        return None
