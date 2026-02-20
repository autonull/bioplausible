"""Tests for the TileEQ model."""
import pytest
import torch
import torch.nn.functional as F

from bioplausible.models.tile_eq import (
    TileEQ,
    TileGraph,
    MemoryBlock,
    TileDescriptor,
    HeatScheduler,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def small_model(neurons=8, layers=3, **kw) -> TileEQ:
    return TileEQ(neurons_per_tile=neurons, num_layers=layers,
                  input_dim=neurons, output_dim=neurons // 2, **kw)


# -----------------------------------------------------------------------
# 1. Memory layout
# -----------------------------------------------------------------------

def test_memory_layout():
    """State slices for different tiles must be non-overlapping."""
    m = small_model(neurons=4, layers=3)
    assert len(m.graph.tiles) == 3  # input, hidden, output tile

    batch = 4
    states = torch.zeros(batch, m.graph.total_state_size)

    view0 = m.mem_block.state_view(states, 0)
    view0.fill_(1.0)

    view1 = m.mem_block.state_view(states, 1)
    view2 = m.mem_block.state_view(states, 2)
    assert view1.max().item() == 0.0, "tile 1 state was unexpectedly modified"
    assert view2.max().item() == 0.0, "tile 2 state was unexpectedly modified"

    # Weight view shape
    assert m.mem_block.weight_view(0, 1).shape == (4, 4)
    assert m.mem_block.weight_view(1, 2).shape == (4, 4)

    # Buffer size matches graph declaration
    assert m.memory.numel() == m.graph.total_buffer_size


# -----------------------------------------------------------------------
# 2. Heat components
# -----------------------------------------------------------------------

def test_heat_components():
    """Heat must be positive and bounded above by epoch_max_heat."""
    m = small_model(neurons=4, layers=3, heat_weights=(1.0, 0.5, 1.2, 0.3))
    tile = m.graph.tiles[0]
    tile.last_update_step = 5

    s_old = torch.zeros(2, 4)
    s_new = torch.tensor([[0.2, 0.1, 0.0, 0.1], [0.1, 0.3, 0.0, 0.0]])

    errors = torch.zeros(2, m.graph.total_state_size)
    err = m.mem_block.error_view(errors, 0)
    err[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    err[1] = torch.tensor([0.0, 2.0, 0.0, 0.0])

    m.scheduler.update(tile, s_old, s_new, err, step=10)

    assert tile.heat > 0.0
    assert m.scheduler.epoch_max_heat >= tile.heat


# -----------------------------------------------------------------------
# 3. Bucket → schedule ordering
# -----------------------------------------------------------------------

def test_bucket_assignment():
    """Hot tiles are scheduled first with more steps; cold tiles may be skipped."""
    m = small_model(neurons=4, layers=3, tau_max=1.0, tau_high=0.5, tau_low=0.05)

    # Manually bias tile heats
    m.graph.tiles[0].heat = 0.01   # < tau_low → cold
    m.graph.tiles[1].heat = 0.3    # warm
    m.graph.tiles[2].heat = 0.9    # > tau_high → hot

    schedule = m.scheduler.schedule(max_steps=16)

    assert schedule, "Schedule must not be empty"
    # Hottest tile (id=2) must come first
    assert schedule[0][0] == 2
    # And must receive the maximum steps
    assert schedule[0][1] == 16

    # Second entry must be tile 1 (warm)
    if len(schedule) >= 2:
        assert schedule[1][0] == 1
        assert schedule[1][1] < 16  # fewer than max

    # Cold tile (0) must either be absent or have steps ≤ warm tile
    cold_entries = [e for e in schedule if e[0] == 0]
    if cold_entries:
        assert cold_entries[0][1] <= (schedule[1][1] if len(schedule) >= 2 else 16)


# -----------------------------------------------------------------------
# 4. Free-phase convergence
# -----------------------------------------------------------------------

def test_free_phase_convergence():
    """Forward pass should converge within max_steps with small init weights."""
    m = TileEQ(
        neurons_per_tile=8,
        num_layers=3,
        input_dim=8,
        output_dim=4,
        max_steps=40,
        epsilon=5e-4,
    )
    x = torch.randn(2, 8)
    _, dyn = m.forward(x, return_dynamics=True)

    assert len(dyn["deltas"]) <= 40
    # After enough steps the dynamics should slow
    assert dyn["final_delta"] < 0.5


# -----------------------------------------------------------------------
# 5. EP update sign (contrastive Hebbian)
# -----------------------------------------------------------------------

def test_ep_update_sign():
    """ΔW_{ij} should be (free_corr − nudged_corr) so SGD advances the network."""
    m = TileEQ(neurons_per_tile=2, num_layers=2, input_dim=2, output_dim=2, beta=1.0)

    n_states = m.graph.total_state_size
    free   = torch.zeros(1, n_states)
    nudged = torch.zeros(1, n_states)

    # src tile=0: s=[1,0], dst tile=1: s_free=[0.5,0.5] s_nudged=[1.0,1.0]
    m.mem_block.state_view(free,   0)[:] = torch.tensor([[1.0, 0.0]])
    m.mem_block.state_view(free,   1)[:] = torch.tensor([[0.5, 0.5]])
    m.mem_block.state_view(nudged, 0)[:] = torch.tensor([[1.0, 0.0]])
    m.mem_block.state_view(nudged, 1)[:] = torch.tensor([[1.0, 1.0]])

    m.memory.grad = None
    m.compute_ep_updates(free, nudged, batch_size=1)

    grad = m.memory.grad
    assert grad is not None

    # Identify the weight block.  Buffer layout: biases (2+2=4) then W01 (4)
    n_biases = 2 * m.neurons_per_tile  # 2 tiles × 2 neurons
    w01_grad = grad[n_biases : n_biases + 4].view(2, 2)

    # prod_free[0,0]  = tanh^-1?  No — phi(s_dst), s_src=1:
    # prod_free  = s_src^T @ phi(s_dst_free)  = [[1],[0]] @ tanh([[0.5,0.5]])
    # prod_nudged = s_src^T @ phi(s_dst_nudged) = [[1],[0]] @ tanh([[1.0,1.0]])
    # dW[0,:] = prod_free[0] - prod_nudged[0] < 0  (tanh(0.5) < tanh(1.0))
    assert w01_grad[0, 0].item() < 0
    assert w01_grad[0, 1].item() < 0
    # Row 1 corresponds to s_src[1]=0 → contrib is zero
    assert abs(w01_grad[1, 0].item()) < 1e-6
    assert abs(w01_grad[1, 1].item()) < 1e-6


# -----------------------------------------------------------------------
# 6. Bias gradient sign
# -----------------------------------------------------------------------

def test_bias_update():
    """Bias gradient for a nudged tile should be (free − nudged) summed over batch."""
    m = TileEQ(neurons_per_tile=1, num_layers=2, input_dim=1, output_dim=1, beta=1.0)
    # 2 tiles: tile0 (input), tile1 (output)

    n_states = m.graph.total_state_size   # = 2
    free   = torch.zeros(1, n_states)
    nudged = torch.zeros(1, n_states)

    # Only output tile (id=1) has a different nudged state
    m.mem_block.state_view(free,   1)[:] = torch.tensor([[0.0]])
    m.mem_block.state_view(nudged, 1)[:] = torch.tensor([[1.0]])

    m.memory.grad = None
    m.compute_ep_updates(free, nudged, batch_size=1)

    grad = m.memory.grad
    assert grad is not None

    # Tile 0 bias offset=0, tile 1 bias offset=1
    # For tile 1: db = (s_free−s_nudged).sum(0) / beta = (0−1)/1 = −1.0
    assert abs(grad[1].item() - (-1.0)) < 1e-5


# -----------------------------------------------------------------------
# 7. Error diffusion
# -----------------------------------------------------------------------

def test_error_diffusion():
    """Error from tile 0 should spill to its fwd neighbor (tile 1) at diffusion_rate."""
    m = TileEQ(
        neurons_per_tile=2, num_layers=2, input_dim=2, output_dim=2, diffusion_rate=0.5
    )
    # 2 tiles, edge 0→1

    errors = torch.zeros(1, m.graph.total_state_size)
    m.mem_block.error_view(errors, 0)[:] = 10.0

    new_errors = m.diffuse_errors(errors)

    e0_new = m.mem_block.error_view(new_errors, 0)
    e1_new = m.mem_block.error_view(new_errors, 1)

    # Source decays by (1 − rate) = 0.5
    assert torch.allclose(e0_new, torch.tensor([[5.0, 5.0]]), atol=1e-5)

    # Only one fwd neighbor (tile 1), so fraction = rate * (norm/total_norm) ≈ rate
    # dest receives ≈ 0.5 * 10 = 5.0
    assert e1_new.min().item() > 0.0, "Tile 1 should receive some error"


# -----------------------------------------------------------------------
# 8. Full API Smoke Test
# -----------------------------------------------------------------------

def test_full_api_smoke():
    """Test full TileEQ API including instantiation, forward, dynamics, train_step, and stats."""
    m = TileEQ(
        neurons_per_tile=16, num_layers=3, input_dim=16, output_dim=4, max_steps=5
    )
    x = torch.randn(4, 16)
    y = torch.tensor([0, 1, 0, 1])

    # 1. Forward pass
    out = m(x)
    assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"

    # 2. Forward pass with dynamics
    logits, dyn = m.forward(x, return_dynamics=True)
    assert "trajectory" in dyn and "deltas" in dyn and "final_delta" in dyn

    # 3. Train step
    stats = m.train_step(x, y)
    assert "loss" in stats
    assert "accuracy" in stats
    assert 0.0 <= stats["accuracy"] <= 1.0
    assert stats["loss"] < 200.0  # sanity check (not infinity)

    # 4. Get Stats
    model_stats = m.get_stats()
    assert "heat_mean" in model_stats and "active_tiles" in model_stats
    assert "tau_max" in model_stats
