"""
Research Tracks (42-44) for 2025 EqProp Research Landscape.

Validates new research directions:
- Holomorphic EP (Complex-valued)
- Directed EP (Asymmetric weights)
- Finite-Nudge EP (Large beta)
"""

import time

import torch
import torch.nn as nn

from ...models import DirectedEP, FiniteNudgeEP, HolomorphicEP
from ..notebook import TrackResult


def _get_synthetic_data(n=32, input_dim=64, output_dim=10):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, output_dim, (n,))
    return x, y


def track_42_holomorphic_ep(verifier) -> TrackResult:
    """Track 42: Holomorphic Equilibrium Propagation."""
    print("\n" + "=" * 60)
    print("TRACK 42: Holomorphic EP (Complex)")
    print("=" * 60)

    start = time.time()

    # 1. Setup
    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # 2. Model
    model = HolomorphicEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        eq_steps=10,
        learning_rate=0.01,
    )

    # 3. Training Loop
    print("\n[42a] Training HolomorphicEP...")
    initial_metrics = model.train_step(x[:32], y[:32])
    initial_loss = initial_metrics["loss"]
    print(f"  Initial Loss: {initial_loss:.4f}")

    losses = []
    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        epoch_loss = 0
        batches = 0
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            metrics = model.train_step(x[idx], y[idx])
            epoch_loss += metrics["loss"]
            batches += 1

        avg_loss = epoch_loss / batches
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss {avg_loss:.4f}")

    final_loss = losses[-1]

    # Check learning
    learned = final_loss < initial_loss * 0.95

    # Check complex weights
    is_complex = model.layers[0].weight.is_complex()

    score = 100 if learned and is_complex else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Holomorphic EP learns using complex-valued states and weights.

**Results**:
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Complex Weights: {"✅ Yes" if is_complex else "❌ No"}
- Learning: {"✅ Yes" if learned else "❌ No"}
"""

    return TrackResult(
        track_id=42,
        name="Holomorphic EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )


def track_43_directed_ep(verifier) -> TrackResult:
    """Track 43: Directed Equilibrium Propagation."""
    print("\n" + "=" * 60)
    print("TRACK 43: Directed EP (Asymmetric)")
    print("=" * 60)

    start = time.time()

    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    model = DirectedEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        eq_steps=10,
        learning_rate=0.01,
    )

    # Verify asymmetry
    w_fwd = model.forward_layers[0].weight
    w_bwd = model.feedback_layers[0].weight  # Corresponds to layer 0 connection?
    # In my implementation:
    # forward_layers[0] connects input -> h1 (dim 0 -> 1)
    # feedback_layers[0] connects h1 -> input (dim 1 -> 0)
    # Check shapes
    print(f"  Forward W shape: {w_fwd.shape}")
    print(f"  Feedback B shape: {w_bwd.shape}")

    # Check if tied (should NOT be tied/shared memory)
    is_tied = w_fwd.data_ptr() == w_bwd.data_ptr()
    print(f"  Weights Tied: {is_tied}")

    # Train
    print("\n[43a] Training DirectedEP...")
    metrics = model.train_step(x[:32], y[:32])
    initial_loss = metrics["loss"]
    print(f"  Initial Loss: {initial_loss:.4f}")

    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            model.train_step(x[idx], y[idx])

    metrics = model.train_step(x[:32], y[:32])
    final_loss = metrics["loss"]
    print(f"  Final Loss: {final_loss:.4f}")

    learned = final_loss < initial_loss * 0.95

    score = 100 if learned and not is_tied else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Directed EP learns with asymmetric forward/feedback weights.

**Results**:
- Asymmetric: {"✅ Yes" if not is_tied else "❌ No"}
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
"""

    return TrackResult(
        track_id=43,
        name="Directed EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )


def track_44_finite_nudge_ep(verifier) -> TrackResult:
    """Track 44: Finite-Nudge Equilibrium Propagation."""
    print("\n" + "=" * 60)
    print("TRACK 44: Finite-Nudge EP (Large Beta)")
    print("=" * 60)

    start = time.time()

    input_dim = 32
    hidden_dim = 64
    output_dim = 10

    x, y = _get_synthetic_data(
        n=verifier.n_samples if verifier.quick_mode else 1000,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # Use Beta = 1.0 (Very large compared to standard 0.1 or 0.5/sqrt(N))
    model = FiniteNudgeEP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        beta=1.0,
        eq_steps=10,
        learning_rate=0.01,
    )
    print(f"  Using Beta: {model.beta}")

    # Train
    print("\n[44a] Training FiniteNudgeEP...")
    metrics = model.train_step(x[:32], y[:32])
    initial_loss = metrics["loss"]
    print(f"  Initial Loss: {initial_loss:.4f}")

    epochs = 30 if verifier.quick_mode else 50
    batch_size = 32

    for epoch in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch_size):
            idx = perm[i : i + batch_size]
            model.train_step(x[idx], y[idx])

    metrics = model.train_step(x[:32], y[:32])
    final_loss = metrics["loss"]
    print(f"  Final Loss: {final_loss:.4f}")

    learned = final_loss < initial_loss * 0.95

    score = 100 if learned else 0
    status = "pass" if score == 100 else "fail"

    evidence = f"""
**Claim**: Finite-Nudge EP learns stably with large beta ({model.beta}).

**Results**:
- Initial Loss: {initial_loss:.4f}
- Final Loss: {final_loss:.4f}
- Stability: {"✅ Stable" if final_loss < 100 else "❌ Unstable"}
"""

    return TrackResult(
        track_id=44,
        name="Finite-Nudge EP",
        status=status,
        score=score,
        metrics={"initial_loss": initial_loss, "final_loss": final_loss},
        evidence=evidence,
        time_seconds=time.time() - start,
    )
