# Fix legacy import path
# EdgeParams is internal/removed from top-level init, mocking for test structure if needed
# or assuming test meant to check internal state

from bioplausible.equitile import EquiTile


def test_equitile_structure():
    model = EquiTile(
        neurons_per_tile=64, num_layers=3, tiles_per_layer=2, input_dim=10, output_dim=2
    )
    assert len(model.graph.tiles) > 0
    # Check edges exist
    assert len(model.graph.edges) > 0
