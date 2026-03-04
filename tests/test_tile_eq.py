import pytest
import torch

from bioplausible.models.tile_eq import AdaptiveTilePC as TileEQ


def test_tile_eq_init():
    model = TileEQ(
        neurons_per_tile=64,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=10,
        output_dim=2,
        prediction_lr=0.01,
    )
    assert model.config.neurons_per_tile == 64
    assert len(model.graph.tiles) > 0


def test_tile_eq_forward():
    model = TileEQ(
        neurons_per_tile=32, num_layers=2, tiles_per_layer=1, input_dim=10, output_dim=2
    )
    x = torch.randn(4, 10)
    out = model(x)
    assert out.shape == (4, 2)
