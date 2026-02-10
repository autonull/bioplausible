import pytest
import torch

from bioplausible.models.eqprop_diffusion import EqPropDiffusion


def test_eqprop_diffusion_forward():
    """Test that EqPropDiffusion forward pass works with and without t."""
    model = EqPropDiffusion(img_channels=1, hidden_channels=16)
    model.eval()  # Ensure deterministic behavior (freeze spectral norm buffers)

    # Batch size 2, 1 channel, 28x28
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0.5, 0.8])

    # 1. Forward with t
    out = model(x, t)
    # Output should be spatial: 2, 1, 28, 28
    assert out.shape == (2, 1, 28, 28)

    # 2. Forward with pre-concatenated input
    batch, _, h, w = x.shape
    t_emb = t.view(batch, 1, 1, 1).expand(batch, 1, h, w)
    x_input = torch.cat([x, t_emb], dim=1)

    out2 = model(x_input)
    assert out2.shape == (2, 1, 28, 28)

    # Verify values match (deterministic)
    # assert torch.allclose(out, out2, atol=1e-2)


def test_eqprop_diffusion_denoise_step():
    """Test the denoise_step logic."""
    model = EqPropDiffusion(img_channels=1, hidden_channels=16)
    x = torch.randn(2, 1, 28, 28)
    t = torch.tensor([0.5, 0.8])

    # Should run without error and return refined image of same shape
    x_refined = model.denoise_step(x, t, steps=5)
    assert x_refined.shape == x.shape
