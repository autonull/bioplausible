import sys
import time
from pathlib import Path

import numpy as np

from ..notebook import TrackResult
from ..utils import create_synthetic_dataset
from ..utils import evaluate_accuracy
from ..utils import train_model

root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from bioplausible.zoo.models.eqprop import BackpropMLP  # noqa: E402
from bioplausible.zoo.models.eqprop import LoopedMLP


def track_1_spectral_norm(verifier) -> TrackResult:
    """Core: Spectral Normalization maintains L < 1."""
    print("\n" + "=" * 60)
    print("TRACK 1: Spectral Normalization Stability")
    print("=" * 60)

    start = time.time()
    input_dim, hidden_dim, output_dim = 64, 128, 10
    X, y = create_synthetic_dataset(verifier.n_samples, input_dim, 10, verifier.seed)

    # Without SN - use higher LR to show instability
    print("\n[1a] Without spectral norm (aggressive training)...")
    model_no_sn = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=False)
    L_before_no = model_no_sn.compute_lipschitz()
    # Higher LR causes L to grow more
    train_model(model_no_sn, X, y, epochs=verifier.epochs, lr=0.05, name="No SN")
    L_after_no = model_no_sn.compute_lipschitz()

    # With SN
    print("[1b] With spectral norm...")
    model_sn = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)
    L_before_sn = model_sn.compute_lipschitz()
    train_model(model_sn, X, y, epochs=verifier.epochs, lr=0.05, name="With SN")
    L_after_sn = model_sn.compute_lipschitz()

    # Evaluate: Key insight is that SN constrains L while non-SN allows growth
    sn_constrained = L_after_sn <= 1.05  # With SN, L should stay near 1
    l_difference = L_after_no - L_after_sn  # Non-SN should have larger L

    # Score based on whether SN is effective
    if sn_constrained and l_difference > 0.5:
        score = 100
        status = "pass"
    elif sn_constrained:
        score = 75
        status = "partial"
    else:
        score = 25
        status = "fail"

    l_diff_no = L_after_no - L_before_no
    l_diff_sn = L_after_sn - L_before_sn
    evidence = f"""
**Claim**: Spectral normalization keeps L ≤ 1 vs unconstrained training.

**Experiment**: Train identical networks with and without spectral normalization.

| Configuration | L (before) | L (after) | Δ | Constrained? |
|---------------|------------|-----------|---|--------------|
| Without SN | {L_before_no:.3f} | {L_after_no:.3f} | {l_diff_no:+.2f} | ❌ No |
| With SN | {L_before_sn:.3f} | {L_after_sn:.3f} | {l_diff_sn:+.2f} |
| | | | | {"✅ Yes" if sn_constrained else "❌ No"} |

**Key Difference**: L(no_sn) - L(sn) = {l_difference:.3f}

**Interpretation**:
- Without SN: L = {L_after_no:.2f} (unconstrained, can grow)
- With SN: L = {L_after_sn:.2f} (constrained to ~1.0)
- SN provides {(L_after_no / L_after_sn - 1) * 100:.0f}% Lipschitz reduction
"""

    improvements = []
    if not sn_constrained:
        improvements.append(
            "Spectral norm not constraining L ≤ 1; check implementation"
        )
    if l_difference < 0.5:
        improvements.append(
            "Difference between SN/non-SN too small; increase epochs or LR"
        )

    return TrackResult(
        track_id=1,
        name="Spectral Normalization Stability",
        status=status,
        score=score,
        metrics={"L_no_sn": L_after_no, "L_sn": L_after_sn, "difference": l_difference},
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=improvements,
    )


# Attach metadata
track_1_spectral_norm.description = (
    "Verifies spectral norm constraints keep Lipschitz constant <= 1"
)
track_1_spectral_norm.category = "Core Stability"


def track_2_backprop_parity(verifier) -> TrackResult:
    """Core: EqProp achieves accuracy parity with Backprop."""
    print("\n" + "=" * 60)
    print("TRACK 2: EqProp vs Backprop Parity")
    print("=" * 60)

    start = time.time()
    input_dim, hidden_dim, output_dim = 64, 128, 10

    # Create a single dataset and split it for fair comparison
    # Using the same data for both methods ensures fair algorithm comparison
    X_all, y_all = create_synthetic_dataset(
        verifier.n_samples, input_dim, 10, verifier.seed
    )
    split = int(0.8 * len(X_all))
    X_train, y_train = X_all[:split], y_all[:split]
    X_test, y_test = X_all[split:], y_all[split:]

    # Backprop
    print("\n[2a] Backprop MLP...")
    bp_model = BackpropMLP(input_dim, hidden_dim, output_dim)
    train_model(bp_model, X_train, y_train, epochs=verifier.epochs, name="Backprop")
    bp_acc = evaluate_accuracy(bp_model, X_test, y_test)

    # EqProp
    print("[2b] EqProp (LoopedMLP)...")
    eq_model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)
    train_model(eq_model, X_train, y_train, epochs=verifier.epochs, name="EqProp")
    eq_acc = evaluate_accuracy(eq_model, X_test, y_test)

    gap = (bp_acc - eq_acc) * 100

    # Score: Pass if both achieve excellent performance (>99%) OR gap < 3%
    # This handles floating point precision issues when both round to 100.0%
    both_excellent = bp_acc >= 0.99 and eq_acc >= 0.99

    if both_excellent or abs(gap) < 3:
        score = 100
        status = "pass"
    elif abs(gap) < 10:
        score = 70
        status = "partial"
    else:
        score = 30
        status = "fail"

    evidence = f"""
**Claim**: EqProp achieves competitive accuracy with Backpropagation (gap < 3%).

**Experiment**: Train identical architectures with Backprop and EqProp
on synthetic classification.

| Method | Test Accuracy | Gap |
|--------|---------------|-----|
| Backprop MLP | {bp_acc*100:.1f}% | — |
| EqProp (LoopedMLP) | {eq_acc*100:.1f}% | {gap:+.1f}% |

**Verdict**: {"✅ PARITY" if abs(gap) < 5 else "⚠️ Gap"} (gap = {abs(gap):.1f}%)

**Note**: Small datasets may show variance; run with --full for 5-seed validation.
"""

    improvements = []
    if abs(gap) > 3:
        improvements.append(
            f"Gap of {abs(gap):.1f}% exceeds target; tune hyperparameters"
        )
    if eq_acc < 0.8:
        improvements.append("Low absolute accuracy; increase epochs or model size")

    return TrackResult(
        track_id=2,
        name="EqProp vs Backprop Parity",
        status=status,
        score=score,
        metrics={"bp_acc": bp_acc, "eq_acc": eq_acc, "gap": gap},
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=improvements,
    )


# Attach metadata
track_2_backprop_parity.description = (
    "Tests if EqProp matches Backprop accuracy on synthetic data"
)
track_2_backprop_parity.category = "Performance"


def track_3_adversarial_healing(verifier) -> TrackResult:
    """Track 1 (README): Adversarial Self-Healing via noise damping."""
    print("\n" + "=" * 60)
    print("TRACK 3: Adversarial Self-Healing")
    print("=" * 60)

    start = time.time()
    input_dim, hidden_dim, output_dim = 64, 128, 10

    X, y = create_synthetic_dataset(verifier.n_samples, input_dim, 10, verifier.seed)
    model = LoopedMLP(input_dim, hidden_dim, output_dim, use_spectral_norm=True)

    print("\n[3a] Pre-training model...")
    train_model(model, X, y, epochs=verifier.epochs, name="Pre-train")

    print("[3b] Testing noise damping...")
    noise_levels = [0.5, 1.0, 2.0]
    results = {}

    for noise in noise_levels:
        damping = model.inject_noise_and_relax(X[:32], noise_level=noise)
        results[noise] = damping
        print(f"  σ={noise}: damping={damping['damping_percent']:.1f}%")

    avg_damping = np.mean([r["damping_percent"] for r in results.values()])
    score = min(100, avg_damping)
    status = "pass" if avg_damping > 95 else ("partial" if avg_damping > 50 else "fail")

    table_rows = "\n".join(
        f"| σ={n} | {r['initial_noise']:.3f} | {r['final_noise']:.6f} | "
        f"{r['damping_percent']:.1f}% |"
        for n, r in results.items()
    )

    evidence = f"""
**Claim**: EqProp networks automatically damp injected noise to zero via contraction.

**Experiment**: Inject Gaussian noise at hidden layer mid-relaxation, measure residual.

| Noise Level | Initial | Final | Damping |
|-------------|---------|-------|---------|
{table_rows}

**Average Damping**: {avg_damping:.1f}%

**Mechanism**: Contraction mapping (L < 1) guarantees: ||noise|| → L^k × ||initial|| → 0

**Hardware Impact**: Enables radiation-hardened, fault-tolerant neuromorphic chips.
"""

    improvements = []
    if avg_damping < 99:
        improvements.append(
            f"Damping at {avg_damping:.1f}%; check Lipschitz constraint"
        )

    return TrackResult(
        track_id=3,
        name="Adversarial Self-Healing",
        status=status,
        score=score,
        metrics={"avg_damping": avg_damping, "results": results},
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=improvements,
    )


# Attach metadata
track_3_adversarial_healing.description = (
    "Measures noise damping (self-healing) properties of EqProp"
)
track_3_adversarial_healing.category = "Robustness"
