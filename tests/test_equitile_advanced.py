# Fix legacy import path
from bioplausible.models.equitile import EnhancedEquiTileConfig as EnhancedEPConfig, EquiTile, EquiTileConfig
from bioplausible.models.equitile.enhanced import EnhancedEquiTile, TileLayerNorm
import torch
import pytest

def test_enhanced_config():
    config = EnhancedEPConfig(
        use_layer_norm=True,
        use_curriculum=True
    )
    assert config.use_layer_norm
    assert config.use_curriculum

def test_enhanced_equitile_init():
    model = EnhancedEquiTile(
        neurons_per_tile=64,
        num_layers=3,
        tiles_per_layer=2,
        input_dim=10,
        output_dim=2,
        enhanced_config=EnhancedEPConfig(use_layer_norm=True)
    )
    assert isinstance(model, EnhancedEquiTile)
    # Check if LayerNorm modules are present
    has_ln = False
    for module in model.modules():
        if isinstance(module, TileLayerNorm):
            has_ln = True
            break
    assert has_ln
