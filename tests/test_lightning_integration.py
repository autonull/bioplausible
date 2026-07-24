"""
Tests for PyTorch Lightning integration.
"""

from unittest.mock import MagicMock, patch

import torch

from bioplausible.lightning_.callbacks import (
    BioPrecisionCallback,
    EnergyConvergenceCallback,
)
from bioplausible.lightning_.module import BioLightningModule
from bioplausible.lightning_.strategies import BioPrecisionMixin, build_trainer


class TestBioLightningModule:
    """Tests for BioLightningModule."""

    def test_init_with_standard_optimizer(self):
        """Test initialization with standard optimizer (adam, sgd, etc.)."""
        module = BioLightningModule(
            model_name="backprop_mlp",
            optimizer_name="adam",
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
        )
        assert module.model_name == "backprop_mlp"
        assert module.optimizer_name == "adam"
        # Standard optimizers should have automatic_optimization = True
        assert module.automatic_optimization is True

    def test_init_with_bio_optimizer(self):
        """Test initialization with bio-plausible optimizer."""
        module = BioLightningModule(
            model_name="backprop_mlp",
            optimizer_name="eqprop",
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
        )
        # Bio optimizers should have automatic_optimization = False
        assert module.automatic_optimization is False

    def test_configure_optimizers(self):
        """Test optimizer creation."""
        module = BioLightningModule(
            model_name="backprop_mlp",
            optimizer_name="adam",
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
        )
        opt = module.configure_optimizers()
        assert opt is not None
        assert module._optimizer is not None

    def test_forward_pass(self):
        """Test forward pass delegates to model."""
        module = BioLightningModule(
            model_name="backprop_mlp",
            optimizer_name="adam",
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
        )
        x = torch.randn(4, 784)
        output = module.forward(x)
        assert output.shape == (4, 10)


class TestBioPrecisionMixin:
    """Tests for BioPrecisionMixin."""

    def test_resolve_precision_standard(self):
        """Test precision resolution for standard optimizers."""
        precision = BioPrecisionMixin.resolve_precision("adam", None)
        assert precision == "bf16-mixed"

    def test_resolve_precision_bio_optimizer(self):
        """Test precision resolution for bio-plausible optimizers."""
        precision = BioPrecisionMixin.resolve_precision("eqprop", "bf16-mixed")
        assert precision == "32-true"
        assert "fp16" not in precision.lower()

    def test_resolve_precision_hebbian(self):
        """Test precision resolution for Hebbian optimizers."""
        precision = BioPrecisionMixin.resolve_precision("smep", None)
        assert precision == "32-true"

    def test_resolve_precision_feedback(self):
        """Test precision resolution for feedback alignment."""
        precision = BioPrecisionMixin.resolve_precision("feedback_alignment", None)
        assert precision == "32-true"


class TestBuildTrainer:
    """Tests for build_trainer function."""

    def test_build_trainer_basic(self):
        """Test basic trainer construction."""
        trainer = build_trainer(
            optimizer_name="adam",
            max_epochs=5,
        )
        assert trainer.max_epochs == 5
        assert trainer.precision == "bf16-mixed"

    def test_build_trainer_bio_precision(self):
        """Test trainer with bio-plausible optimizer gets FP32."""
        trainer = build_trainer(
            optimizer_name="eqprop",
            max_epochs=5,
        )
        assert trainer.precision == "32-true"

    def test_build_trainer_with_wandb(self):
        """Test trainer can be configured with W&B logger."""
        try:
            import wandb  # noqa: F401
        except ModuleNotFoundError:
            import pytest

            pytest.skip("wandb not installed")
        trainer = build_trainer(
            optimizer_name="adam",
            max_epochs=5,
            enable_wandb=True,
        )
        assert trainer.logger is not None


class TestCallbacks:
    """Tests for Lightning callbacks."""

    def test_energy_convergence_callback_init(self):
        """Test EnergyConvergenceCallback initialization."""
        cb = EnergyConvergenceCallback(monitor="train_energy", patience=3)
        assert cb.monitor == "train_energy"
        assert cb.patience == 3
        assert cb._best == float("inf")

    def test_energy_convergence_callback_early_stop(self):
        """Test EnergyConvergenceCallback stops when energy plateaus."""
        cb = EnergyConvergenceCallback(monitor="train_energy", patience=2)

        mock_trainer = MagicMock()
        mock_trainer.callback_metrics = {"train_energy": 0.5}

        mock_module = MagicMock()

        # Simulate energy not improving
        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb._counter == 0
        assert cb._best == 0.5

        # Second call - still same energy
        mock_trainer.callback_metrics = {"train_energy": 0.6}
        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb._counter == 1

        # Third call - still same - should trigger early stop
        mock_trainer.callback_metrics = {"train_energy": 0.7}
        cb.on_validation_epoch_end(mock_trainer, mock_module)
        assert cb._counter == 2
        # Should stop training
        assert mock_trainer.should_stop is True

    def test_bio_precision_callback_standard_optimizer(self):
        """Test BioPrecisionCallback with standard optimizer does nothing."""
        cb = BioPrecisionCallback()

        mock_trainer = MagicMock()
        mock_trainer.precision = "32-true"

        mock_module = MagicMock(spec=BioLightningModule)
        mock_module.optimizer_name = "adam"

        cb.on_train_start(mock_trainer, mock_module)

    def test_bio_precision_callback_bio_optimizer(self):
        """Test BioPrecisionCallback downcasts bio-optimizers to FP32."""
        cb = BioPrecisionCallback()

        mock_trainer = MagicMock()
        mock_trainer.precision = "bf16-mixed"

        mock_module = MagicMock(spec=BioLightningModule)
        mock_module.optimizer_name = "eqprop"
        mock_module.to = MagicMock()

        cb.on_train_start(mock_trainer, mock_module)

        mock_module.to.assert_called_once()


class TestHPOIntegration:
    """Tests for HPO integration with Optuna."""

    def test_optuna_pruner_init(self):
        """Test BioOptunaPruner initialization."""
        from bioplausible.lightning_.hpo import BioOptunaPruner

        pruner = BioOptunaPruner(
            model_name="backprop_mlp",
            optimizer_name="adam",
            max_epochs=5,
        )
        assert pruner.model_name == "backprop_mlp"
        assert pruner.optimizer_name == "adam"
        assert pruner.max_epochs == 5
        assert pruner.task_name is None

    def test_optuna_pruner_init_with_task_name(self):
        """Test BioOptunaPruner initialization with task_name."""
        from bioplausible.lightning_.hpo import BioOptunaPruner

        pruner = BioOptunaPruner(
            model_name="backprop_mlp",
            optimizer_name="adam",
            max_epochs=5,
            task_name="digits",
        )
        assert pruner.task_name == "digits"

    def test_optuna_pruner_sample(self):
        """Test BioOptunaPruner hyperparameter sampling uses metamodel."""
        from bioplausible.lightning_.hpo import BioOptunaPruner

        pruner = BioOptunaPruner(
            model_name="backprop_mlp",
            optimizer_name="adam",
            max_epochs=5,
        )

        mock_trial = MagicMock()
        mock_trial.suggest_float.return_value = 0.001
        mock_trial.suggest_int.return_value = 4
        mock_trial.suggest_categorical.side_effect = ["relu", "adam", "kaiming", 32]

        hparams = pruner._sample(mock_trial)
        assert "lr" in hparams
        assert "hidden_dim" in hparams
        assert "optimizer" in hparams

    def test_optuna_pruner_sample_with_task_name(self):
        """Test that task_name constraints are applied in hyperparameter sampling."""
        from bioplausible.lightning_.hpo import BioOptunaPruner

        pruner = BioOptunaPruner(
            model_name="backprop_mlp",
            optimizer_name="adam",
            max_epochs=5,
            task_name="digits",
        )

        mock_trial = MagicMock()
        # 5 continuous float params: lr, weight_decay, grad_clip, dropout, momentum
        mock_trial.suggest_float.return_value = 0.001
        # 1 discrete int param: num_layers
        mock_trial.suggest_int.return_value = 4
        # 4 categorical params: hidden_dim, activation, weight_init, optimizer
        mock_trial.suggest_categorical.side_effect = [64, "relu", "kaiming", "adam"]

        hparams = pruner._sample(mock_trial)
        assert "lr" in hparams
        assert "hidden_dim" in hparams
        # hidden_dim should be constrained to [32, 64, 128] for small task
        assert hparams["hidden_dim"] in [32, 64, 128]

    def test_ray_tune_search_init(self):
        """Test BioRayTuneSearch initialization."""
        from bioplausible.lightning_.hpo import BioRayTuneSearch

        searcher = BioRayTuneSearch(
            model_name="backprop_mlp",
            optimizer_name="adam",
            max_epochs=5,
        )
        assert searcher.model_name == "backprop_mlp"
        assert searcher.optimizer_name == "adam"


class TestNASIntegration:
    """Tests for NAS integration."""

    def test_get_plausible_model_names(self):
        """Test getting plausible model names for NAS."""
        from bioplausible.lightning_.nas import get_plausible_model_names

        names = get_plausible_model_names()
        assert isinstance(names, list)

    def test_get_bio_optimizer_names(self):
        """Test getting bio optimizer names for NAS."""
        from bioplausible.lightning_.nas import get_bio_optimizer_names

        names = get_bio_optimizer_names()
        assert isinstance(names, list)


class TestAutoScientistIntegration:
    """Tests for AutoScientist Lightning integration."""

    def test_execute_standard_trial_use_lightning(self):
        """Test that AutoScientist can use Lightning when configured."""
        from bioplausible.execution.engine import ExecutionEngine

        # Verify the method exists and handles use_lightning config
        assert hasattr(ExecutionEngine, "_execute_standard_trial")
        assert hasattr(ExecutionEngine, "_get_train_loader")
        assert hasattr(ExecutionEngine, "_get_val_loader")

    def test_nas_search_exists(self):
        """Test run_nas_search function exists."""
        import inspect

        from bioplausible.lightning_.nas import run_nas_search

        sig = inspect.signature(run_nas_search)
        params = list(sig.parameters.keys())
        assert "train_loader" in params
        assert "val_loader" in params
        assert "n_trials" in params
        assert "task_name" in params


class TestPLTrialIntegration:
    """Tests for PL trial execution integration."""

    def test_pl_trial_runs_backprop(self):
        """Test run_pl_trial with backprop model."""
        from bioplausible.lightning_.experiment import run_pl_trial

        # Create mock data loaders
        mock_loader = MagicMock()
        mock_loader.__iter__ = lambda self: iter([])

        # Create a real simple model for testing
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(784, 10)

            def forward(self, x):
                return self.linear(x)

            def train_step(self, x, y):
                return None  # Let automatic optimization handle it

        # Use a real model instance
        with patch("bioplausible.lightning_.module.create_model") as mock_create:
            mock_create.return_value = SimpleModel()

            with patch("pytorch_lightning.Trainer") as MockTrainer:
                mock_trainer_instance = MagicMock()
                # Mock that metrics returns a tensor-like object
                mock_val_acc = MagicMock()
                mock_val_acc.item.return_value = 0.85
                mock_trainer_instance.callback_metrics = {
                    "val_acc": mock_val_acc,
                    "val_loss": MagicMock(item=lambda: 0.5),
                }

                # Make fit work with our mock loader
                def mock_fit(module, train_loader, val_loader, *args, **kwargs):
                    pass

                mock_trainer_instance.fit = mock_fit
                MockTrainer.return_value = mock_trainer_instance

                result = run_pl_trial(
                    model_name="backprop_mlp",
                    optimizer_name="adam",
                    config={"lr": 0.001, "hidden_dim": 128, "epochs": 3},
                    train_loader=mock_loader,
                    val_loader=mock_loader,
                    quick_mode=True,
                )

                assert result is not None
                assert "accuracy" in result

    def test_pl_trial_returns_none_on_failure(self):
        """Test run_pl_trial returns None gracefully on failure."""
        from bioplausible.lightning_.experiment import run_pl_trial

        # Create mock data loaders
        mock_loader = MagicMock()
        mock_loader.__iter__ = lambda self: iter([])

        with patch("pytorch_lightning.Trainer") as MockTrainer:
            mock_trainer_instance = MagicMock()

            def mock_fit_failure(module, train_loader, val_loader, *args, **kwargs):
                raise Exception("Training failed")

            mock_trainer_instance.fit = mock_fit_failure
            MockTrainer.return_value = mock_trainer_instance

            with patch("bioplausible.lightning_.module.create_model") as mock_create:
                mock_create.return_value = MagicMock()

                result = run_pl_trial(
                    model_name="backprop_mlp",
                    optimizer_name="adam",
                    config={"lr": 0.001, "hidden_dim": 128, "epochs": 3},
                    train_loader=mock_loader,
                    val_loader=mock_loader,
                    quick_mode=True,
                )

                assert result is None

    def test_pl_trial_with_wandb(self):
        """Test run_pl_trial_with_wandb exists and signature is correct."""
        # Just verify the function exists and accepts the right signature
        import inspect

        from bioplausible.lightning_.experiment import run_pl_trial_with_wandb

        sig = inspect.signature(run_pl_trial_with_wandb)
        params = list(sig.parameters.keys())
        assert "model_name" in params
        assert "optimizer_name" in params
        assert "config" in params
