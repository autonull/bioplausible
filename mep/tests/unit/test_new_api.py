"""
Tests for the new EP API workflow.
"""

import torch
import torch.nn as nn
from mep import smep


def test_ep_workflow_updates_weights():
    """Test that EP workflow updates weights."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    initial_weight = model[0].weight.detach().clone()

    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        mode='ep',
        settle_steps=5
    )

    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))

    optimizer.step(x=x, target=y)

    assert not torch.allclose(model[0].weight, initial_weight), "Weights should have updated"


def test_backprop_workflow():
    """Test backprop workflow (fallback)."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2)
    )

    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        mode='backprop'
    )

    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    criterion = nn.CrossEntropyLoss()

    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    # Implicit assertion: no error


def test_zero_grad_before_step():
    """Test zero_grad works before step."""
    model = nn.Sequential(nn.Linear(10, 2))
    
    optimizer = smep(model.parameters(), model=model, lr=0.01, mode='backprop')
    
    x = torch.randn(5, 10)
    y = torch.randint(0, 2, (5,))
    
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    # Verify gradients exist
    for p in model.parameters():
        assert p.grad is not None
    
    optimizer.zero_grad()
    
    # Verify gradients cleared
    for p in model.parameters():
        assert p.grad is None
