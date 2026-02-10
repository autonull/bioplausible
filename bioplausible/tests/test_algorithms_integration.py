import pytest
import torch
import torch.nn as nn

from bioplausible.core import EqPropTrainer
from bioplausible.models.base import ModelConfig
from bioplausible.models.simple_fa import StandardFA
from bioplausible.models.standard_eqprop import StandardEqProp


def test_eqprop_algorithm_integration():
    """
    Test that EqPropTrainer can train a StandardEqProp algorithm model.
    """
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 5

    # Create synthetic data
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Configure Algorithm
    config = ModelConfig(
        name="eqprop",
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        learning_rate=0.01,
        equilibrium_steps=5,  # Short for testing
        beta=0.1,
    )

    model = StandardEqProp(config)

    # Initialize Trainer
    trainer = EqPropTrainer(model, use_kernel=False, use_compile=False)

    # Train
    history = trainer.fit(loader, epochs=2)

    assert "train_loss" in history
    assert len(history["train_loss"]) == 2
    assert history["train_loss"][-1] > 0  # Just check it ran


def test_feedback_alignment_integration():
    """
    Test StandardFA with EqPropTrainer.
    Verifies that feedback weights are moved to device and training runs.
    """
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    batch_size = 4

    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    config = ModelConfig(
        name="feedback_alignment",
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        learning_rate=0.01,
    )

    model = StandardFA(config)

    # Test device movement (simulated if cpu, but structure check)
    if torch.cuda.is_available():
        device = "cuda"
        model.to(device)
        # Check feedback weights (stored as params/buffers in BioModel)
        # StandardFA stores them in self.feedback_weights which is a ParameterList
        assert model.feedback_weights[0].device.type == "cuda"

    # Trainer
    trainer = EqPropTrainer(
        model,
        use_kernel=False,
        use_compile=False,
        device=device if torch.cuda.is_available() else "cpu",
    )

    history = trainer.fit(loader, epochs=2)

    assert "train_loss" in history
    # Accuracy should exist
    assert "train_acc" in history
    assert len(history["train_acc"]) == 2


def test_eqprop_dynamics_shapes():
    """Verify shapes during dynamics."""
    config = ModelConfig(
        name="eqprop",
        input_dim=5,
        hidden_dims=[10, 8],
        output_dim=3,
        equilibrium_steps=2,
    )
    model = StandardEqProp(config)

    x = torch.randn(2, 5)  # Batch 2

    # Forward
    out = model(x)
    assert out.shape == (2, 3)

    # Check states
    activations = model._last_activations
    # Should have: Input, Hidden1, Hidden2, Output
    # Total 4 tensors
    assert len(activations) == 4
    assert activations[0].shape == (2, 5)
    assert activations[1].shape == (2, 10)
    assert activations[2].shape == (2, 8)
    assert activations[3].shape == (2, 3)
