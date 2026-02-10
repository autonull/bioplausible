#!/usr/bin/env python3
"""
Track 38: Adaptive Compute Analysis.

Hypothesis: Equilibrium settling time correlates with sequence complexity.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models import CausalTransformerEqProp


def measure_settling_time(model, x, max_steps=100, epsilon=1e-5):
    """
    Measure steps until convergence.
    NOTE: CausalTransformerEqProp forward() usually runs for fixed steps.
    We need to access the internal step-by-step logic or check partial results.

    The standard model.forward() doesn't return per-step states.
    However, for this experiment, we can fake it by calling forward with increasing steps
    and checking if the output changes. This is inefficient but works for a black box.
    """
    model.eval()

    prev_out = None

    # Optimization: check 5, 10, 15... instead of every single step to save time
    # or just 1..max_steps if max_steps is small (e.g. 20)

    steps_to_check = range(1, max_steps + 1)

    for step in steps_to_check:
        with torch.no_grad():
            # Run model with specific number of equilibrium steps
            # Ensure model accepts 'steps' argument
            out = model(x, steps=step)

            if prev_out is not None:
                # Calculate change in output
                delta = (out - prev_out).norm().item()
                if delta < epsilon:
                    return step  # Converged

            prev_out = out

    return max_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=65)  # Shakespeare approx
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TRACK 38: Adaptive Compute Analysis")
    print("=" * 60)

    # 1. Load Model (Untrained is fine to show dynamics, but trained is better)
    # Ideally we load a checkpoint from Track 37. If not found, init fresh.
    print("Initializing model...")
    model = CausalTransformerEqProp(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=4,
        max_seq_len=args.seq_len,
        eq_steps=30,  # Max steps
    ).to(device)

    # Try to load checkpoint if available
    ckpt_path = Path("results/track_37/best_model.pt")
    if ckpt_path.exists():
        print(f"Loading checkpoint from {ckpt_path}")
        try:
            model.load_state_dict(torch.load(ckpt_path))
        except:
            print("Failed to load checkpoint, using random weights.")
    else:
        print("No checkpoint found, using random weights (results may be noisy).")

    # 2. Generate/Load Data
    # For meaningful complexity correlation, we need real data.
    # If not available, we simulate "simple" vs "complex" sequences.

    results = []

    if args.quick:
        n_samples = 20
        print("⚠️ Quick mode: 20 samples")
    else:
        n_samples = args.num_samples

    print(f"Analyzing {n_samples} sequences...")

    # We will generate synthetic sequences of varying "entropy"
    for i in range(n_samples):
        # vary entropy by limiting vocab size for some samples
        current_vocab = np.random.randint(2, args.vocab_size + 1)

        # Generate sequence
        seq_data = torch.randint(0, current_vocab, (1, args.seq_len)).to(device)

        # Measure settling time
        settling_time = measure_settling_time(model, seq_data, max_steps=30)

        # Measure "complexity" (Perplexity of the model on this sequence)
        with torch.no_grad():
            logits = model(seq_data)
            # Shifts for LM loss (predict next token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = seq_data[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, args.vocab_size), shift_labels.view(-1)
            )
            complexity = torch.exp(loss).item()

        results.append(
            {
                "complexity": complexity,
                "settling_time": settling_time,
                "vocab_used": current_vocab,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{n_samples}...")

    # 3. Analyze Correlation
    complexities = [r["complexity"] for r in results]
    settling_times = [r["settling_time"] for r in results]

    if len(set(settling_times)) > 1:
        corr_r, p_val = pearsonr(complexities, settling_times)
    else:
        corr_r, p_val = 0.0, 1.0

    print("\nResults:")
    print(f"  Correlation (r): {corr_r:.3f}")
    print(f"  P-value: {p_val:.4f}")

    # Save
    save_dir = Path("results/track_38")
    save_dir.mkdir(parents=True, exist_ok=True)

    output = {"correlation": corr_r, "p_value": p_val, "data_points": results}

    with open(save_dir / "adaptive_compute_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {save_dir}")

    # Plot simple scatter
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(complexities, settling_times, alpha=0.6)
        plt.xlabel("Sequence Complexity (Perplexity)")
        plt.ylabel("Settling Time (Steps)")
        plt.title(f"Adaptive Compute: r={corr_r:.3f}")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / "complexity_vs_settling.png")
        print(f"Plot saved to {save_dir / 'complexity_vs_settling.png'}")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
