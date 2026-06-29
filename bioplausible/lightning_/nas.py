"""
Neural Architecture Search for AutoScientist.

Samples model names and optimizer names via Optuna to discover
Pareto-optimal combinations for each task.
"""

from typing import Any, Dict, List

import optuna
from pytorch_lightning import Trainer

from bioplausible.lightning_.module import BioLightningModule
from bioplausible.models.registry import list_model_specs
from bioplausible.optimizers import list_optimizers


def get_plausible_model_names() -> List[str]:
    """Return bio-plausible model names."""
    return [
        spec.name
        for spec in list_model_specs()
        if spec.learning_rule_class in ("equilibrium", "hebbian", "forward-only")
    ]


def get_bio_optimizer_names() -> List[str]:
    """Return bio-plausible optimizer names."""
    keywords = ("eqprop", "smep", "hebbian", "fa", "chl")
    return [
        name for name in list_optimizers() if any(kw in name.lower() for kw in keywords)
    ]


def create_nas_objective(
    train_loader,
    val_loader,
    max_epochs: int = 10,
) -> callable:
    """
    Create an Optuna objective that samples model + optimizer names.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        max_epochs: Max training epochs per trial.

    Returns:
        Objective function.
    """
    model_names = get_plausible_model_names()
    optimizer_names = get_bio_optimizer_names()

    def objective(trial: optuna.trial.Trial) -> float:
        # Sample model and optimizer
        model_name = trial.suggest_categorical("model_name", model_names)
        optimizer_name = trial.suggest_categorical("optimizer_name", optimizer_names)

        # Sample hyperparameters
        hparams = {
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 64, 512, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        }

        module = BioLightningModule(
            model_name=model_name, optimizer_name=optimizer_name, **hparams
        )

        trainer = Trainer(
            max_epochs=max_epochs,
            enable_progress_bar=False,
            logger=False,
            callbacks=[],
        )

        try:
            trainer.fit(module, train_loader, val_loader)
            metrics = trainer.callback_metrics
            acc = metrics.get("val_acc", 0.0).item() if "val_acc" in metrics else 0.0
            trial.set_user_attr("model_name", model_name)
            trial.set_user_attr("optimizer_name", optimizer_name)
            return acc
        except Exception:
            return 0.0

    return objective


def run_nas_search(
    train_loader,
    val_loader,
    n_trials: int = 50,
    max_epochs: int = 10,
) -> Dict[str, Any]:
    """
    Run a NAS search over model + optimizer combinations.

    Args:
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        n_trials: Number of Optuna trials.
        max_epochs: Max training epochs.

    Returns:
        Best configuration.
    """
    objective = create_nas_objective(train_loader, val_loader, max_epochs)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_trial.params)
