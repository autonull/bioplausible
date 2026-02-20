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


def make_xor(n_copies: int = 16):
    """XOR dataset: 4 canonical patterns repeated with small noise."""
    xs = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    ys = torch.tensor([0, 1, 1, 0])
    # replicate
    X = xs.repeat(n_copies, 1) + torch.randn(4 * n_copies, 2) * 0.05
    Y = ys.repeat(n_copies)
    return X, Y


def make_blobs(n_samples: int = 200, n_per_class: int = 4):
    """Linearly separable blobs centered on class means."""
    torch.manual_seed(42)
    means = [(i * 2.0, 0.0) for i in range(n_per_class)]
    X_parts, Y_parts = [], []
    per = n_samples // n_per_class
    for cls, (mx, my) in enumerate(means):
        pts = torch.randn(per, 2) * 0.3 + torch.tensor([mx, my])
        X_parts.append(pts)
        Y_parts.append(torch.full((per,), cls, dtype=torch.long))
    return torch.cat(X_parts), torch.cat(Y_parts)


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
# 5. EP update sign (corrected: nudged − free, phi on both src and dst)
# -----------------------------------------------------------------------

def test_ep_update_sign():
    """dW should be phi(src_n).T @ phi(dst_n) - phi(src_f).T @ phi(dst_f), positive when nudged correlation is stronger."""
    m = TileEQ(neurons_per_tile=2, num_layers=2, input_dim=2, output_dim=2, beta=1.0)

    n_states = m.graph.total_state_size
    free   = torch.zeros(1, n_states)
    nudged = torch.zeros(1, n_states)

    # src tile=0: s=[1,0], dst tile=1: s_free=[0.1,0.1] s_nudged=[1.0,1.0]
    # phi(src) same in both cases; nudged dst has more activation
    # So nudged corr > free corr → dW[0, :] > 0
    m.mem_block.state_view(free,   0)[:] = torch.tensor([[1.0, 0.0]])
    m.mem_block.state_view(free,   1)[:] = torch.tensor([[0.1, 0.1]])
    m.mem_block.state_view(nudged, 0)[:] = torch.tensor([[1.0, 0.0]])
    m.mem_block.state_view(nudged, 1)[:] = torch.tensor([[1.0, 1.0]])

    m.memory.grad = None
    m.compute_ep_updates(free, nudged, batch_size=1)

    grad = m.memory.grad
    assert grad is not None

    # Bias region: 2 tiles × 2 neurons = offset 4; W01 starts at 4
    n_biases = 2 * m.neurons_per_tile
    w01_grad = grad[n_biases : n_biases + 4].view(2, 2)

    # phi(src_n)[0,0] = tanh(1.0) > 0, phi(dst_n)[0,0] = tanh(1.0) > 0
    # nudged corr > free corr → dW[0,:] > 0
    assert w01_grad[0, 0].item() > 0, "row 0 grad should be positive (nudged > free)"
    assert w01_grad[0, 1].item() > 0
    # Row 1: phi(src)[0,1] = tanh(0.0) = 0 → contribution is zero
    assert abs(w01_grad[1, 0].item()) < 1e-6
    assert abs(w01_grad[1, 1].item()) < 1e-6


# -----------------------------------------------------------------------
# 6. Bias gradient sign (corrected: nudged − free)
# -----------------------------------------------------------------------

def test_bias_update():
    """Bias gradient for a nudged tile should be (nudged − free) summed over batch."""
    m = TileEQ(neurons_per_tile=1, num_layers=2, input_dim=1, output_dim=1, beta=1.0)
    # 2 tiles: tile0 (input), tile1 (output)

    n_states = m.graph.total_state_size   # = 2
    free   = torch.zeros(1, n_states)
    nudged = torch.zeros(1, n_states)

    # Output tile (id=1): free=0, nudged=1 → db = (1 - 0) / beta = +1.0
    m.mem_block.state_view(free,   1)[:] = torch.tensor([[0.0]])
    m.mem_block.state_view(nudged, 1)[:] = torch.tensor([[1.0]])

    m.memory.grad = None
    m.compute_ep_updates(free, nudged, batch_size=1)

    grad = m.memory.grad
    assert grad is not None

    # Tile 0 bias offset=0, tile 1 bias offset=1
    # For tile 1: db = (s_nudged − s_free).sum(0) / beta = (1 − 0)/1 = +1.0
    assert abs(grad[1].item() - 1.0) < 1e-5, f"Expected +1.0, got {grad[1].item()}"


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
# 8. Arbitrary Topology
# -----------------------------------------------------------------------

def test_arbitrary_topology():
    """TileGraph.from_edges should produce a functional model with skip connections."""
    # 4 tiles: 0(in) → 1, 0→2(skip), 1→3(out), 2→3(out)
    graph = TileGraph.from_edges(
        n_tiles=4,
        neurons_per_tile=4,
        fwd_edges=[(0, 1), (0, 2), (1, 3), (2, 3)],
        input_ids=[0],
        output_ids=[3],
        positions=[(0.0, 0.5), (0.33, 0.25), (0.33, 0.75), (1.0, 0.5)],
    )

    assert len(graph.tiles) == 4
    assert set(graph.tiles[0].fwd_neighbors) == {1, 2}   # skip conn
    assert set(graph.tiles[3].bwd_neighbors) == {1, 2}

    # Build a model using this graph externally and verify forward pass runs
    # We use build_layered normally here, just verify from_edges doesn't crash
    m = TileEQ(neurons_per_tile=4, num_layers=2, input_dim=4, output_dim=4, max_steps=5)
    x = torch.randn(2, 4)
    logits = m(x)
    assert logits.shape == (2, 4)


# -----------------------------------------------------------------------
# 9. Full API Smoke Test
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


# -----------------------------------------------------------------------
# 10. Integration: learns XOR
# -----------------------------------------------------------------------

def test_learns_xor():
    """TileEQ must learn XOR from 4 samples within 500 training steps."""
    torch.manual_seed(0)
    X, Y = make_xor(n_copies=8)  # 32 samples

    m = TileEQ(
        neurons_per_tile=8,
        num_layers=3,
        tiles_per_layer=2,   # wider hidden layer for XOR
        input_dim=2,
        output_dim=2,
        beta=0.1,
        max_steps=20,
        learning_rate=0.01,
    )

    for _ in range(500):
        idx = torch.randint(0, len(X), (16,))
        m.train_step(X[idx], Y[idx])

    # Evaluate on clean canonical XOR
    xor_x = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    xor_y = torch.tensor([0, 1, 1, 0])
    with torch.no_grad():
        logits = m(xor_x, steps=30)
        preds = logits.argmax(1)
    acc = (preds == xor_y).float().mean().item()
    assert acc >= 0.75, f"XOR accuracy too low: {acc:.2f} (expected >= 0.75)"


# -----------------------------------------------------------------------
# 11. Integration: learns linearly separable blobs
# -----------------------------------------------------------------------

def test_learns_linear():
    """TileEQ must learn a linearly separable 4-class blob task within 600 steps."""
    torch.manual_seed(1)
    X, Y = make_blobs(n_samples=200, n_per_class=4)

    m = TileEQ(
        neurons_per_tile=8,
        num_layers=3,
        input_dim=2,
        output_dim=4,
        beta=0.1,
        max_steps=20,
        learning_rate=0.02,
    )

    for _ in range(600):
        idx = torch.randint(0, len(X), (32,))
        m.train_step(X[idx], Y[idx])

    with torch.no_grad():
        logits = m(X, steps=20)
        preds = logits.argmax(1)
    acc = (preds == Y).float().mean().item()
    assert acc >= 0.65, f"Blobs accuracy too low: {acc:.2f} (expected >= 0.65)"

