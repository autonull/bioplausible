"""
HPO Integration: Optuna + PyTorch Lightning

Replaces the legacy HyperparameterSearch with scalable,
pruning-aware hyperparameter optimisation via PyTorch Lightning.
"""

from typing import Any

from pytorch_lightning import Trainer

from bioplausible.lightning_.module import BioLightningModule


class BioOptunaPruner:
    """
    Optuna HPO wrapper for bioplausible experiments.

    Prunes bad trials early (especially useful for 10-15x slower
    bio-plausible optimizers).
    """

    def __init__(
        self,
        model_name: str,
        optimizer_name: str,
        max_epochs: int = 10,
        metric: str = "val_acc",
        direction: str = "maximize",
        task_name: str | None = None,
    ):
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.max_epochs = max_epochs
        self.metric = metric
        self.direction = direction
        self.task_name = task_name

    def search(
        self,
        train_loader: Any,
        val_loader: Any,
        n_trials: int = 50,
        pruner_type: str = "median",
    ) -> dict[str, Any]:
        """
        Run Optuna search.

        Args:
            train_loader: Training data.
            val_loader: Validation data.
            n_trials: Number of Optuna trials.
            pruner_type: median or hyperband.

        Returns:
            Dictionary of best hyperparameters.
        """
        import optuna
        from optuna.integration import PyTorchLightningPruningCallback

        if pruner_type == "median":
            pruner = optuna.pruners.MedianPruner()
        elif pruner_type == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.MedianPruner()

        study = optuna.create_study(direction=self.direction, pruner=pruner)

        def objective(trial: optuna.trial.Trial) -> float:
            hparams = self._sample(trial)
            module = BioLightningModule(self.model_name, self.optimizer_name, **hparams)
            trainer = Trainer(
                max_epochs=self.max_epochs,
                callbacks=[PyTorchLightningPruningCallback(trial, monitor=self.metric)],
                enable_progress_bar=False,
                logger=False,
            )
            trainer.fit(module, train_loader, val_loader)
            return float(trainer.callback_metrics[self.metric].item())

        study.optimize(objective, n_trials=n_trials)
        return dict(study.best_trial.params)

    def _sample(self, trial) -> dict[str, Any]:
        """Sample hyperparameters using the hyperparameter metamodel."""
        from bioplausible.hyperopt.optuna_bridge import create_optuna_space

        config = create_optuna_space(
            trial=trial,
            model_name=self.model_name,
            task_name=self.task_name,
        )
        config["optimizer"] = self.optimizer_name
        return config


class BioRayTuneSearch:
    """
    Ray Tune (ASHA) distributed HPO for bioplausible experiments.
    """

    def __init__(
        self,
        model_name: str,
        optimizer_name: str,
        max_epochs: int = 10,
        grace_period: int = 1,
        reduction_factor: int = 2,
    ):
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.max_epochs = max_epochs
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor

    def search(
        self,
        train_loader: Any,
        val_loader: Any,
        num_samples: int = 50,
        gpus_per_trial: int = 1,
    ) -> dict[str, Any]:
        """
        Run Ray Tune ASHA search.

        Args:
            train_loader: Training data.
            val_loader: Validation data.
            num_samples: Number of hyperparameter configurations.
            gpus_per_trial: GPUs to allocate per trial.

        Returns:
            Best hyperparameter configuration.
        """
        from ray import tune
        from ray.tune.integration.pytorch_lightning import TuneReportCallback
        from ray.tune.schedulers import ASHAScheduler

        config = {
            "lr": tune.loguniform(1e-4, 1e-1),
            "hidden_dim": tune.choice([64, 128, 256, 512]),
            "batch_size": tune.choice([32, 64, 128, 256]),
        }

        scheduler = ASHAScheduler(
            max_t=self.max_epochs,
            grace_period=self.grace_period,
            reduction_factor=self.reduction_factor,
        )

        def train_func(cfg: dict[str, Any]) -> None:
            module = BioLightningModule(self.model_name, self.optimizer_name, **cfg)
            callback = TuneReportCallback({"val_acc": "val_acc"}, on="validation_end")
            trainer = Trainer(
                max_epochs=self.max_epochs,
                callbacks=[callback],
                enable_progress_bar=False,
                logger=False,
            )
            trainer.fit(module, train_loader, val_loader)

        analysis = tune.run(
            train_func,
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            resources_per_trial={"gpu": gpus_per_trial, "cpu": 2},
            metric="val_acc",
            mode="max",
        )

        return dict(analysis.best_config)
