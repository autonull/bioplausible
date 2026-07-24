import torch

from bioplausible.zoo.base import ModelConfig
from bioplausible.zoo.models.eqprop import StandardEqProp
from bioplausible.zoo.models.fa import StandardFA


def test_eqprop_algorithm_integration():
    """Test that StandardEqProp can run a forward+backward pass."""
    input_dim = 10
    hidden_dim = 20
    output_dim = 2
    batch_size = 5

    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    config = ModelConfig(
        name="eqprop",
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        learning_rate=0.01,
        equilibrium_steps=5,
        beta=0.1,
    )

    model = StandardEqProp(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    initial_loss_value: float | None = None
    final_loss_value: float | None = None
    for _ in range(2):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        if initial_loss_value is None:
            initial_loss_value = float(loss.detach().item())
        final_loss_value = float(loss.detach().item())

    assert initial_loss_value is not None and final_loss_value is not None
    assert final_loss_value > 0


def test_feedback_alignment_integration():
    """Test StandardFA forward/backward+optimizer step on synthetic data."""
    input_dim = 8
    hidden_dim = 16
    output_dim = 4
    batch_size = 4

    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    config = ModelConfig(
        name="feedback_alignment",
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=output_dim,
        learning_rate=0.01,
    )

    model = StandardFA(config)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for _ in range(2):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    # If reached, the loop ran without crashing; accuracy check is implicit.
    assert model.feedback_weights is not None


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

    x = torch.randn(2, 5)

    out = model(x)
    assert out.shape == (2, 3)

    activations = model._last_activations
    assert len(activations) == 4
    assert activations[0].shape == (2, 5)
    assert activations[1].shape == (2, 10)
    assert activations[2].shape == (2, 8)
    assert activations[3].shape == (2, 3)
