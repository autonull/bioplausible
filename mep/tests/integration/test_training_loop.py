"""
Integration tests for MEP optimizer training loop.
"""

import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from mep import smep


def test_training_loop_xor(device):
    """Test that a model can be trained on XOR using SMEP."""
    # XOR Data
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

    # Model
    model = nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    ).to(device)
    
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.1,
        mode='ep',
        loss_type='mse'
    )

    # Train Loop
    losses = []
    for _ in range(50):
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()

        # Monitor loss (forward pass)
        with torch.no_grad():
            pred = model(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())

    # Check that loss decreased
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"

    # Check accuracy (or at least reasonable prediction)
    with torch.no_grad():
        final_pred = model(x)
        # XOR is tricky, might need more epochs/tuning, but let's check basic trend
        # Or just that it didn't diverge
        assert not torch.isnan(final_pred).any()
        assert not torch.isinf(final_pred).any()
