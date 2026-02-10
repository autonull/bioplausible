"""
Negative Result Tracks - Scientific Completeness

Documents what DOESN'T work to ensure honest claims and guide research.

Track 55: Pure Linear Chain Failure
  - Proves activations are essential for deep signal propagation
  - Pure linear layers vanish even with spectral normalization
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from ..notebook import TrackResult

root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))


class PureLinearChain(nn.Module):
    """
    Pure linear chain WITHOUT activations.

    This is intentionally broken to document the failure mode.
    """

    def __init__(self, dim=64, depth=100, use_spectral_norm=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_spectral_norm = use_spectral_norm

        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(dim, dim, bias=False)
            if use_spectral_norm:
                layer = nn.utils.parametrizations.spectral_norm(
                    layer, n_power_iterations=5
                )
            self.layers.append(layer)

    def forward(self, x, return_norms=False):
        norms = []
        h = x

        if return_norms:
            norms.append(h.norm(dim=1).mean().item())

        for layer in self.layers:
            h = layer(h)
            if return_norms:
                norms.append(h.norm(dim=1).mean().item())

        if return_norms:
            return h, norms
        return h


def track_55_negative_linear_chain(verifier) -> TrackResult:
    """
    Track 55: Pure Linear Chain Failure (Negative Result)

    Purpose: Document that pure linear layers WITHOUT activations
    cannot propagate signals through deep networks, even with SN.

    This is a NEGATIVE RESULT demonstrating architectural requirements.
    """
    print("\n" + "=" * 60)
    print("TRACK 55: NEGATIVE RESULT - Pure Linear Chain Failure")
    print("=" * 60)

    start = time.time()

    depths = [50, 100, 200] if verifier.quick_mode else [50, 100, 200, 500]
    dim = 64

    print(f"\n[55] Testing pure linear chains at depths: {depths}")
    print(f"     Purpose: Prove that activations are REQUIRED for depth")

    x = torch.randn(8, dim) * 0.1

    results = {}

    for depth in depths:
        print(f"\n[55a] Depth {depth}...")

        depth_results = {}
        for use_sn in [True, False]:
            label = "with_sn" if use_sn else "without_sn"

            model = PureLinearChain(dim=dim, depth=depth, use_spectral_norm=use_sn)

            with torch.no_grad():
                _, norms = model(x, return_norms=True)

            initial = norms[0]
            final = norms[-1]
            ratio = final / initial if initial > 0 else 0

            # Find where signal dies (< 1% of initial)
            death_layer = depth
            for i, n in enumerate(norms):
                if n < initial * 0.01:
                    death_layer = i
                    break

            depth_results[label] = {
                "initial_norm": initial,
                "final_norm": final,
                "ratio": ratio,
                "death_layer": death_layer,
            }

            print(
                f"    {label}: initial={initial:.4f}, final={final:.6f}, "
                f"ratio={ratio:.6f}, dies at layer {death_layer}"
            )

        # Key insight: BOTH fail
        both_vanish = (
            depth_results["with_sn"]["ratio"] < 0.01
            and depth_results["without_sn"]["ratio"] < 0.01
        )
        sn_no_better = (
            abs(
                depth_results["with_sn"]["ratio"] - depth_results["without_sn"]["ratio"]
            )
            < 0.01
        )

        depth_results["both_vanish"] = both_vanish
        depth_results["sn_no_better"] = sn_no_better
        results[depth] = depth_results

    # Score: This is a NEGATIVE result - we WANT both to fail
    all_vanish = all(r["both_vanish"] for r in results.values())
    sn_no_help = all(r["sn_no_better"] for r in results.values())

    if all_vanish and sn_no_help:
        score = 100  # Correctly demonstrates failure
        status = "pass"
        verdict = "CONFIRMED: Pure linear chains fail regardless of SN"
    else:
        score = 50
        status = "partial"
        verdict = "Unexpected: Some configurations survived?"

    # Build table
    table_rows = []
    for depth, r in results.items():
        table_rows.append(
            f"| {depth} | {r['with_sn']['ratio']:.6f} | "
            f"{r['without_sn']['ratio']:.6f} | "
            f"{r['with_sn']['death_layer']} | "
            f"{'✅' if r['both_vanish'] else '❌'} |"
        )

    evidence = f"""
**NEGATIVE RESULT**: Spectral normalization CANNOT save pure linear chains.

**Purpose**: Document architectural requirement for activations in deep networks.

| Depth | SN Ratio | No-SN Ratio | SN Death Layer | Both Vanish? |
|-------|----------|-------------|----------------|--------------|
{chr(10).join(table_rows)}

**Key Finding**: {verdict}

**Root Cause**: 
- Linear layers: h_n = W_n @ W_{n-1} @ ... @ W_1 @ x
- Even with ||W|| ≤ 1, product of 50+ matrices → exponential decay
- No activation = no signal regeneration = vanishing

**Implication**: 
- Deep EqProp REQUIRES activations (tanh, ReLU) between layers
- SN bounds ||W|| but cannot prevent cumulative decay in pure linear chains
- This is NOT a failure of SN - it's an architectural requirement

**Lesson**: Use `DeepHebbianChain` or `LoopedMLP` WITH activations.
"""

    return TrackResult(
        track_id=55,
        name="Negative Result: Linear Chain",
        status=status,
        score=score,
        metrics={
            "results": results,
            "all_vanish": all_vanish,
            "sn_no_help": sn_no_help,
        },
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=[],
    )
