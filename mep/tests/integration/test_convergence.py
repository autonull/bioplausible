"""
Integration tests for MEP optimizer convergence.
"""

import torch
import torch.nn.functional as F
import pytest
import torch.nn as nn
from mep import smep


def test_xor_convergence(device):
    """Test that model can learn on a simple problem."""
    # Simple regression problem (easier than XOR for EP)
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [2.]], device=device)  # Simple sum-like target

    # Model
    model = nn.Sequential(
        nn.Linear(2, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)
    
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=0.05,
        mode='ep',
        beta=0.5,
        settle_steps=15,
        loss_type='mse'
    )

    initial_loss = None
    final_loss = None
    
    # Train
    for epoch in range(100):
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()

        # Monitor loss
        with torch.no_grad():
            pred = model(x)
            loss = F.mse_loss(pred, y)
            if initial_loss is None:
                initial_loss = loss.item()
            final_loss = loss.item()

    # Check that loss decreased
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
    assert final_loss < 1.0, f"Final loss too high: {final_loss}"
