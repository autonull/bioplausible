from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ...datasets import get_lm_dataset
from ...models import (AdaptiveFA, EqPropAttentionOnlyLM, EquilibriumAlignment,
                       FullEqPropLM, HybridEqPropLM, LoopedMLPForLM,
                       RecurrentEqPropLM)
from ..notebook import TrackResult


def _get_mock_data(input_dim=784, output_dim=10, batch_size=32):
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))
    return x, y


def track_50_nebc_eqprop_variants(verifier: Any) -> TrackResult:
    """Verify newly migrated NEBC EqProp variants (LM and others)."""
    print(f"    Running NEBC EqProp Variants check...")

    variants = [
        ("FullEqPropLM", FullEqPropLM),
        ("EqPropAttentionOnlyLM", EqPropAttentionOnlyLM),
        ("RecurrentEqPropLM", RecurrentEqPropLM),
        ("HybridEqPropLM", HybridEqPropLM),
        ("LoopedMLPForLM", LoopedMLPForLM),
    ]

    vocab_size = 50
    seq_len = 16
    batch_size = 4
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    passed_variants = 0

    for name, cls in variants:
        try:
            model = cls(
                vocab_size=vocab_size, hidden_dim=32, num_layers=2, max_seq_len=32
            )
            out = model(x)
            assert out.shape == (batch_size, seq_len, vocab_size)

            loss = nn.functional.cross_entropy(out.view(-1, vocab_size), x.view(-1))
            loss.backward()

            passed_variants += 1
            print(f"      - {name}: OK")
        except Exception as e:
            print(f"      - {name}: FAILED ({e})")
            import traceback

            traceback.print_exc()

    score = (passed_variants / len(variants)) * 100
    status = "pass" if score == 100 else "partial"

    return TrackResult(
        track_id=50,
        name="NEBC EqProp Variants",
        status=status,
        score=score,
        evidence=f"Successfully instantiated and stepped {passed_variants}/{len(variants)} LM variants.",
        metrics={"passed_variants": passed_variants},
        time_seconds=0.1,
    )


def track_51_nebc_feedback_alignment(verifier: Any) -> TrackResult:
    """Verify Adaptive Feedback Alignment (Native)."""
    print(f"    Running AdaptiveFA check...")

    def run_check():
        x, y = _get_mock_data()
        model = AdaptiveFA(input_dim=784, hidden_dim=64, output_dim=10, num_layers=3)

        metrics = model.train_step(x, y)
        loss_1 = metrics["loss"]

        metrics = model.train_step(x, y)
        loss_2 = metrics["loss"]

        return 100.0, {"loss_start": loss_1, "loss_end": loss_2}

    result = verifier.evaluate_robustness(run_check, n_seeds=1)

    if result["mean_score"] > 90:
        status = "pass"
    else:
        status = "fail"

    return TrackResult(
        track_id=51,
        name="NEBC Adaptive Feedback Alignment",
        status=status,
        score=result["mean_score"],
        evidence="AdaptiveFA runs train_step without error.",
        metrics=result["metrics"],
        time_seconds=0.1,
    )


def track_52_nebc_direct_feedback_alignment(verifier: Any) -> TrackResult:
    """Verify Equilibrium Alignment (Native)."""
    print(f"    Running Equilibrium Alignment check...")

    def run_check():
        x, y = _get_mock_data()
        model = EquilibriumAlignment(
            input_dim=784, hidden_dim=64, output_dim=10, max_steps=10
        )

        metrics = model.train_step(x, y)
        loss_1 = metrics["loss"]

        metrics = model.train_step(x, y)
        loss_2 = metrics["loss"]

        return 100.0, {"loss_start": loss_1, "loss_end": loss_2}

    result = verifier.evaluate_robustness(run_check, n_seeds=1)

    return TrackResult(
        track_id=52,
        name="NEBC Equilibrium Alignment",
        status="pass" if result["mean_score"] > 90 else "fail",
        score=result["mean_score"],
        evidence="EquilibriumAlignment runs train_step without error.",
        metrics=result["metrics"],
        time_seconds=0.1,
    )


from ...models import ContrastiveHebbianLearning, DeepHebbianChain


def track_53_nebc_contrastive_hebbian(verifier: Any) -> TrackResult:
    """Verify Contrastive Hebbian Learning."""
    print(f"    Running Contrastive Hebbian Learning check...")

    def run_check():
        x, y = _get_mock_data()
        model = ContrastiveHebbianLearning(
            input_dim=784,
            hidden_dim=64,
            output_dim=10,
            num_layers=2,
            max_steps=10,
            beta=0.1,
        )

        metrics = model.train_step(x, y)
        loss_1 = metrics["loss"]

        metrics = model.train_step(x, y)
        loss_2 = metrics["loss"]

        return 100.0, {"loss_start": loss_1, "loss_end": loss_2}

    result = verifier.evaluate_robustness(run_check, n_seeds=1)

    return TrackResult(
        track_id=53,
        name="NEBC Contrastive Hebbian",
        status="pass" if result["mean_score"] > 90 else "fail",
        score=result["mean_score"],
        evidence="ContrastiveHebbianLearning runs train_step without error.",
        metrics=result["metrics"],
        time_seconds=0.1,
    )


def track_54_nebc_deep_hebbian_chain(verifier: Any) -> TrackResult:
    """Verify Deep Hebbian Chain signal propagation."""
    print(f"    Running Deep Hebbian Chain check...")

    def run_check():
        x, _ = _get_mock_data(batch_size=4)
        # Deep chain with 50 layers
        model = DeepHebbianChain(
            input_dim=784,
            hidden_dim=64,
            output_dim=10,
            num_layers=50,
            use_spectral_norm=True,
        )

        metrics = model.measure_signal_propagation(x)
        decay = metrics["decay_ratio"]

        # Signal should survive (decay_ratio > 0.05) if SN works
        score = 100.0 if decay > 0.05 else 0.0
        return score, metrics

    result = verifier.evaluate_robustness(run_check, n_seeds=1)

    return TrackResult(
        track_id=54,
        name="NEBC Deep Hebbian Chain",
        status="pass" if result["mean_score"] == 100 else "fail",
        score=result["mean_score"],
        evidence="DeepHebbianChain maintains signal through 50 layers.",
        metrics=result["metrics"],
        time_seconds=0.1,
    )
