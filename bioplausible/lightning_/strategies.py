"""
Hardware Acceleration Strategies

Builds PL Trainers with correct precision and distributed strategy
for biologically plausible models.
"""

from typing import List
from typing import Optional
from typing import Union

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger


class BioPrecisionMixin:
    """
    Mixin that forces full 32-bit precision when a bio-plausible
    optimizer is detected (EqProp, Hebbian, etc.).
    """

    BIO_KEYWORDS = ("eqprop", "hebbian", "chl", "ep_", "feedback", "smep")

    @classmethod
    def resolve_precision(cls, optimizer_name: str, requested: Optional[str]) -> str:
        """
        Determine the correct precision string.

        Args:
            optimizer_name: Name of the optimizer.
            requested: User-requested precision (e.g. bf16-mixed).

        Returns:
            Final precision string for the PL Trainer.
        """
        if requested is None:
            requested = "bf16-mixed"

        if any(kw in optimizer_name.lower() for kw in cls.BIO_KEYWORDS):
            return "32-true"
        return requested


def build_trainer(
    optimizer_name: str,
    precision: Optional[str] = None,
    max_epochs: int = 10,
    gradient_clip_val: Optional[float] = None,
    accumulate_grad_batches: int = 1,
    strategy: str = "auto",
    devices: Union[int, str] = "auto",
    accelerator: str = "auto",
    callbacks: Optional[List] = None,
    enable_wandb: bool = False,
    wandb_project: str = "bioplausible-benchmarks",
    **trainer_kwargs,
) -> Trainer:
    """
    Construct a PL Trainer with bio-plausible-safe defaults.

    Args:
        optimizer_name: Name of the optimizer (used to pin precision).
        precision: Desired precision. Overridden for sensitive optimizers.
        max_epochs: Training epochs.
        gradient_clip_val: Gradient clipping value.
        accumulate_grad_batches: Simulate large batch sizes.
        strategy: Distributed strategy (ddp, ddp_spawn, etc.).
        devices: Number of devices or "auto".
        accelerator: Accelerator type (gpu, cpu, etc.).
        callbacks: Additional callbacks.
        enable_wandb: Whether to attach WandbLogger.
        wandb_project: W&B project name.
        **trainer_kwargs: Additional arguments forwarded to Trainer.

    Returns:
        Configured PyTorch Lightning Trainer.
    """
    precision = BioPrecisionMixin.resolve_precision(optimizer_name, precision)

    default_callbacks = [
        EarlyStopping(monitor="val_acc", patience=3, mode="max"),
        ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=3),
    ]

    # Add RichProgressBar only if progress bar is enabled
    progress_bar_enabled = trainer_kwargs.get("enable_progress_bar", True)
    if progress_bar_enabled:
        default_callbacks.append(RichProgressBar())

    if callbacks:
        default_callbacks.extend(callbacks)

    logger = None
    if enable_wandb:
        logger = WandbLogger(project=wandb_project, log_model=True)

    trainer = Trainer(
        max_epochs=max_epochs,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        strategy=strategy,
        devices=devices,
        accelerator=accelerator,
        callbacks=default_callbacks,
        logger=logger,
        **trainer_kwargs,
    )

    return trainer
