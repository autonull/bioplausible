#!/usr/bin/env python3
"""
Few-Shot Learning Study

Test EqProp's advantage on very small datasets (100-2000 samples).
This is where the scale study showed EqProp winning 3×.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import BackpropTransformerLM, get_eqprop_lm


def load_shakespeare(max_chars=None):
    """Load Shakespeare."""
    data_path = Path("data/shakespeare.txt")
    data_path.parent.mkdir(exist_ok=True)

    if not data_path.exists():
        import urllib.request

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r") as f:
        text = f.read()
        if max_chars:
            text = text[:max_chars]

    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}

    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))

    return data[:n], data[n:], vocab_size


def get_batch(data, seq_len, batch_size, device):
    """Sample batch."""
    if len(data) <= seq_len:
        raise ValueError(f"Data too short: {len(data)} <= {seq_len}")
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def train_and_eval(
    model, train_data, val_data, vocab_size, seq_len, epochs, lr, batch_size, device
):
    """Train and return final perplexity."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    batches_per_epoch = max(5, len(train_data) // (batch_size * seq_len))

    for epoch in range(epochs):
        model.train()
        for _ in range(batches_per_epoch):
            try:
                x, y = get_batch(train_data, seq_len, batch_size, device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            except ValueError:
                continue

    # Final eval
    model.eval()
    total_loss = 0
    n_batches = 20
    with torch.no_grad():
        for _ in range(n_batches):
            try:
                x, y = get_batch(val_data, seq_len, batch_size, device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                total_loss += loss.item()
            except ValueError:
                continue

    return torch.exp(torch.tensor(total_loss / n_batches)).item()


def run_few_shot_study(device, output_dir, n_seeds=10):
    """Run comprehensive few-shot study."""

    print("=" * 70)
    print("FEW-SHOT LEARNING STUDY")
    print("=" * 70)

    sample_sizes = [100, 200, 500, 1000, 2000, 5000]
    seq_len = 32  # Shorter for small data
    hidden = 64
    layers = 2
    epochs = 30
    batch_size = 16
    lr = 3e-4

    all_results = {}

    for size in sample_sizes:
        print(f"\n{'='*70}")
        print(f"Sample Size: {size}")
        print("=" * 70)

        bp_results = []
        eq_results = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_data, val_data, vocab_size = load_shakespeare(size)
            train_data = train_data.to(device)
            val_data = val_data.to(device)

            # Backprop
            bp_model = BackpropTransformerLM(
                vocab_size=vocab_size,
                hidden_dim=hidden,
                num_layers=layers,
                num_heads=4,
            ).to(device)

            bp_ppl = train_and_eval(
                bp_model,
                train_data,
                val_data,
                vocab_size,
                seq_len,
                epochs,
                lr,
                batch_size,
                device,
            )
            bp_results.append(bp_ppl)

            # EqProp (looped_mlp - best variant from scale study)
            torch.manual_seed(seed)
            eq_model = get_eqprop_lm(
                name="looped_mlp",
                vocab_size=vocab_size,
                hidden_dim=hidden,
                num_layers=layers,
                num_heads=4,
                max_eq_steps=10,
            ).to(device)

            eq_ppl = train_and_eval(
                eq_model,
                train_data,
                val_data,
                vocab_size,
                seq_len,
                epochs,
                lr,
                batch_size,
                device,
            )
            eq_results.append(eq_ppl)

            if (seed + 1) % 2 == 0:
                print(f"  Seed {seed+1}/{n_seeds}: BP={bp_ppl:.1f}, EQ={eq_ppl:.1f}")

        bp_mean = np.mean(bp_results)
        bp_std = np.std(bp_results)
        eq_mean = np.mean(eq_results)
        eq_std = np.std(eq_results)
        ratio = eq_mean / bp_mean

        print(f"\n  Backprop: {bp_mean:.2f} ± {bp_std:.2f}")
        print(f"  EqProp:   {eq_mean:.2f} ± {eq_std:.2f}")
        print(
            f"  Ratio:    {ratio:.2f}× {'✓ EqProp wins' if ratio < 1 else '✗ Backprop wins'}"
        )

        all_results[size] = {
            "backprop": {"mean": bp_mean, "std": bp_std, "values": bp_results},
            "eqprop": {"mean": eq_mean, "std": eq_std, "values": eq_results},
            "ratio": ratio,
        }

    # Save
    with open(output_dir / "few_shot_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\n| Size | Backprop | EqProp | Ratio | Winner |")
    print("|------|----------|--------|-------|--------|")

    for size in sample_sizes:
        r = all_results[size]
        winner = "EqProp" if r["ratio"] < 1 else "Backprop"
        print(
            f"| {size:,} | {r['backprop']['mean']:.1f}±{r['backprop']['std']:.1f} | "
            f"{r['eqprop']['mean']:.1f}±{r['eqprop']['std']:.1f} | {r['ratio']:.2f}× | {winner} |"
        )

    # Find crossover
    for i, size in enumerate(sample_sizes[:-1]):
        if (
            all_results[size]["ratio"] < 1
            and all_results[sample_sizes[i + 1]]["ratio"] >= 1
        ):
            print(f"\n📍 Crossover between {size} and {sample_sizes[i+1]} samples")
            break

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/few_shot_study"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    run_few_shot_study(args.device, args.output, args.seeds)

    print(f"\n✓ Complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()
