"""Tests for the CoreTrainer."""

import pytest
import torch
import torch.nn as nn

from bioplausible.core.trainer import CoreTrainer, TrainerConfig, TrainingMetrics, run_from_config
from bioplausible.core.registry import Registry, ComponentCategory


def test_trainer_config_defaults():
    """Test TrainerConfig default values."""
    config = TrainerConfig(model="test_model")
    assert config.epochs == 10
    assert config.batch_size == 64
    assert config.optimizer == "adam"
    assert config.task == "mnist"
    assert config.device == "auto"
    assert config.track_energy is True


def test_trainer_config_from_dict():
    """Test creating TrainerConfig from dict."""
    config = TrainerConfig.from_dict({
        "model": "test_model",
        "epochs": 20,
        "batch_size": 128,
        "optimizer": "sgd",
    })
    assert config.model == "test_model"
    assert config.epochs == 20
    assert config.batch_size == 128
    assert config.optimizer == "sgd"


def test_training_metrics():
    """Test TrainingMetrics creation and serialization."""
    metrics = TrainingMetrics(
        epoch=0,
        train_loss=0.5,
        train_accuracy=0.8,
        val_loss=0.4,
        val_accuracy=0.85,
        epoch_time=1.0,
    )
    assert metrics.train_loss == 0.5
    assert metrics.val_accuracy == 0.85
    
    d = metrics.to_dict()
    assert d["epoch"] == 0
    assert d["train_loss"] == 0.5
    assert d["val_accuracy"] == 0.85


def test_training_metrics_partial():
    """Test TrainingMetrics with only required fields."""
    metrics = TrainingMetrics(epoch=0, train_loss=0.5, train_accuracy=0.0)
    assert metrics.val_loss is None
    assert metrics.energy_proxy is None


def test_trainer_config_to_dict():
    """Test TrainerConfig serialization."""
    config = TrainerConfig(model="test_model")
    d = config.to_dict()
    assert d["model"] == "test_model"
    assert d["epochs"] == 10


def test_run_from_config_error():
    """Test run_from_config with invalid model (should raise)."""
    config = TrainerConfig(
        model="nonexistent_model",
        epochs=1,
        batches_per_epoch=2,
        task="mnist",
    )
    
    with pytest.raises((ValueError, KeyError, ImportError)):
        run_from_config(config)


def test_core_trainer_from_dict():
    """Test creating CoreTrainer from dict."""
    trainer = CoreTrainer.from_dict({
        "model": "test",
        "epochs": 1,
        "task": "mnist",
        "track_energy": False,
    })
    assert trainer.config.model == "test"
    assert trainer.config.epochs == 1


def test_core_trainer_initialization():
    """Test CoreTrainer initialization."""
    config = TrainerConfig(
        model="test",
        epochs=1,
        track_energy=False,
        task="mnist",
    )
    trainer = CoreTrainer(config)
    assert trainer is not None
    assert trainer.config.model == "test"
    assert trainer.config.epochs == 1