#!/usr/bin/env python3
"""
Comprehensive Language Modeling Scale Study

Multi-hour experiment to understand when and why EqProp outperforms Backprop
on language modeling tasks.

Research Questions:
1. At what dataset size does EqProp begin to outperform Backprop?
2. How do different EqProp variants compare?
3. How does sequence length affect the advantage?
4. What's the optimal equilibrium steps vs accuracy trade-off?

Expected runtime: 2-4 hours on GPU
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bioplausible.models import BackpropTransformerLM, get_eqprop_lm


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dataset_size: int  # Number of characters
    seq_len: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    epochs: int
    lr: float
    batch_size: int
    eq_steps: Optional[int] = None  # Only for EqProp

    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    model_type: str  # 'backprop' or eqprop variant name
    perplexity: float
    accuracy: float
    loss: float
    n_params: int
    train_time: float
    inference_time: float
    best_epoch: int

    def to_dict(self):
        result = {"model_type": self.model_type}
        result.update(self.config.to_dict())
        result.update(
            {
                "perplexity": self.perplexity,
                "accuracy": self.accuracy,
                "loss": self.loss,
                "n_params": self.n_params,
                "train_time": self.train_time,
                "inference_time": self.inference_time,
                "best_epoch": self.best_epoch,
            }
        )
        return result


def load_shakespeare(max_chars=None):
    """Load Shakespeare dataset."""
    data_path = Path("data/shakespeare.txt")
    data_path.parent.mkdir(exist_ok=True)

    if not data_path.exists():
        import urllib.request

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading Shakespeare...")
        urllib.request.urlretrieve(url, data_path)

    with open(data_path, "r") as f:
        text = f.read()
        if max_chars:
            text = text[:max_chars]

    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    n = int(0.9 * len(data))

    return data[:n], data[n:], vocab_size, char_to_idx, idx_to_char


def get_batch(data, seq_len, batch_size, device):
    """Sample a batch."""
    if len(data) <= seq_len:
        raise ValueError(f"Data length {len(data)} <= seq_len {seq_len}")

    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def train_and_eval(
    model, train_data, val_data, vocab_size, config, device, verbose=False
):
    """Train model and return detailed metrics."""
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    train_start = time.time()

    best_val_loss = float("inf")
    best_epoch = 0
    batches_per_epoch = max(10, len(train_data) // (config.batch_size * config.seq_len))

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0

        for _ in range(batches_per_epoch):
            try:
                x, y = get_batch(train_data, config.seq_len, config.batch_size, device)
            except ValueError:
                continue

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        n_val_batches = 20

        with torch.no_grad():
            for _ in range(n_val_batches):
                try:
                    x, y = get_batch(
                        val_data, config.seq_len, config.batch_size, device
                    )
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                    val_loss += loss.item()
                except ValueError:
                    continue

        avg_val_loss = val_loss / n_val_batches
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch

        if verbose and (epoch + 1) % max(1, config.epochs // 5) == 0:
            print(f"    Epoch {epoch+1}/{config.epochs}: val_loss={avg_val_loss:.4f}")

    train_time = time.time() - train_start

    # Final comprehensive evaluation
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    n_eval_batches = 50

    inference_start = time.time()
    with torch.no_grad():
        for _ in range(n_eval_batches):
            try:
                x, y = get_batch(val_data, config.seq_len, config.batch_size, device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                total_correct += (preds == y).sum().item()
                total_tokens += y.numel()
            except ValueError:
                continue

    inference_time = time.time() - inference_start

    avg_loss = total_loss / n_eval_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    n_params = sum(p.numel() for p in model.parameters())

    return ExperimentResult(
        config=config,
        model_type="",  # Set by caller
        perplexity=perplexity,
        accuracy=accuracy,
        loss=avg_loss,
        n_params=n_params,
        train_time=train_time,
        inference_time=inference_time,
        best_epoch=best_epoch,
    )


def run_experiment_suite(device, output_dir, quick=False):
    """Run comprehensive experiment suite."""

    print("=" * 80)
    print("LANGUAGE MODELING SCALE STUDY")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    all_results = []

    # Experiment 1: Dataset Size Scaling
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Dataset Size Scaling")
    print("=" * 80)

    if quick:
        dataset_sizes = [1000, 5000, 10000]
        base_config = ExperimentConfig(
            dataset_size=0,  # Set per iteration
            seq_len=32,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            epochs=15,
            lr=3e-4,
            batch_size=32,
        )
    else:
        dataset_sizes = [1000, 2000, 5000, 10000, 20000, 50000]
        base_config = ExperimentConfig(
            dataset_size=0,
            seq_len=64,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            epochs=30,
            lr=3e-4,
            batch_size=32,
        )

    eqprop_variants = ["looped_mlp", "recurrent_core", "full"]

    for size in dataset_sizes:
        print(f"\n{'='*80}")
        print(f"Dataset Size: {size:,} chars")
        print("=" * 80)

        # Load data
        train_data, val_data, vocab_size, _, _ = load_shakespeare(size)
        train_data = train_data.to(device)
        val_data = val_data.to(device)

        print(
            f"  Train: {len(train_data):,}, Val: {len(val_data):,}, Vocab: {vocab_size}"
        )

        config = ExperimentConfig(**{**base_config.to_dict(), "dataset_size": size})

        # Backprop baseline
        print(f"\n  [Backprop]...")
        try:
            bp_model = BackpropTransformerLM(
                vocab_size=vocab_size,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
            ).to(device)

            result = train_and_eval(
                bp_model, train_data, val_data, vocab_size, config, device, verbose=True
            )
            result.model_type = "backprop"
            all_results.append(result)

            print(
                f"    → PPL: {result.perplexity:.2f}, Acc: {result.accuracy:.3f}, "
                f"Params: {result.n_params:,}, Time: {result.train_time:.1f}s"
            )

            bp_ppl = result.perplexity
        except Exception as e:
            print(f"    ERROR: {e}")
            bp_ppl = None

        # EqProp variants
        for variant in eqprop_variants:
            eq_steps = 10 if quick else 15
            print(f"\n  [EqProp {variant}]...")

            try:
                eq_config = ExperimentConfig(
                    **{**config.to_dict(), "eq_steps": eq_steps}
                )
                eq_model = get_eqprop_lm(
                    name=variant,
                    vocab_size=vocab_size,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    num_heads=config.num_heads,
                    max_eq_steps=eq_steps,
                ).to(device)

                result = train_and_eval(
                    eq_model,
                    train_data,
                    val_data,
                    vocab_size,
                    eq_config,
                    device,
                    verbose=True,
                )
                result.model_type = f"eqprop_{variant}"
                all_results.append(result)

                if bp_ppl:
                    ratio = result.perplexity / bp_ppl
                    marker = "✓" if ratio < 1.0 else "✗"
                    print(
                        f"    → PPL: {result.perplexity:.2f} ({ratio:.2f}× BP), Acc: {result.accuracy:.3f}, "
                        f"Params: {result.n_params:,}, Time: {result.train_time:.1f}s {marker}"
                    )
                else:
                    print(
                        f"    → PPL: {result.perplexity:.2f}, Acc: {result.accuracy:.3f}, "
                        f"Params: {result.n_params:,}, Time: {result.train_time:.1f}s"
                    )

                # Save intermediate results
                save_results(all_results, output_dir / "results_intermediate.json")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback

                traceback.print_exc()

        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()

    # Experiment 2: Sequence Length Scaling (if not quick mode)
    if not quick:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Sequence Length Scaling")
        print("=" * 80)

        train_data, val_data, vocab_size, _, _ = load_shakespeare(10000)
        train_data = train_data.to(device)
        val_data = val_data.to(device)

        seq_lengths = [16, 32, 64, 128]

        for seq_len in seq_lengths:
            print(f"\n  Sequence Length: {seq_len}")

            config = ExperimentConfig(
                dataset_size=10000,
                seq_len=seq_len,
                hidden_dim=128,
                num_layers=2,
                num_heads=4,
                epochs=20,
                lr=3e-4,
                batch_size=32,
            )

            # Just test best EqProp variant vs Backprop
            for model_type in ["backprop", "eqprop_recurrent_core"]:
                print(f"    [{model_type}]...")

                try:
                    if model_type == "backprop":
                        model = BackpropTransformerLM(
                            vocab_size=vocab_size,
                            hidden_dim=config.hidden_dim,
                            num_layers=config.num_layers,
                            num_heads=config.num_heads,
                        ).to(device)
                    else:
                        eq_config = ExperimentConfig(
                            **{**config.to_dict(), "eq_steps": 15}
                        )
                        model = get_eqprop_lm(
                            name="recurrent_core",
                            vocab_size=vocab_size,
                            hidden_dim=config.hidden_dim,
                            num_layers=config.num_layers,
                            num_heads=config.num_heads,
                            max_eq_steps=15,
                        ).to(device)
                        config = eq_config

                    result = train_and_eval(
                        model, train_data, val_data, vocab_size, config, device
                    )
                    result.model_type = model_type
                    all_results.append(result)

                    print(f"      → PPL: {result.perplexity:.2f}")

                    save_results(all_results, output_dir / "results_intermediate.json")

                except Exception as e:
                    print(f"      ERROR: {e}")

            if device == "cuda":
                torch.cuda.empty_cache()

    # Save final results
    save_results(all_results, output_dir / "results_final.json")
    save_results_csv(all_results, output_dir / "results_final.csv")

    # Generate summary
    generate_summary(all_results, output_dir)

    return all_results


def save_results(results, path):
    """Save results as JSON."""
    with open(path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)


def save_results_csv(results, path):
    """Save results as CSV."""
    if not results:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].to_dict().keys())
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())


def generate_summary(results, output_dir):
    """Generate markdown summary of results."""

    summary = []
    summary.append("# Language Modeling Scale Study Results\n")
    summary.append(f"**Total experiments:** {len(results)}\n")
    summary.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # Group by dataset size
    by_size = {}
    for r in results:
        size = r.config.dataset_size
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(r)

    summary.append("## Dataset Size Scaling\n\n")
    summary.append("| Size | Model | Perplexity | Accuracy | Params | Time |\n")
    summary.append("|------|-------|------------|----------|--------|------|\n")

    for size in sorted(by_size.keys()):
        group = by_size[size]
        bp_ppl = next((r.perplexity for r in group if r.model_type == "backprop"), None)

        for r in group:
            ratio = (
                f"({r.perplexity/bp_ppl:.2f}×)"
                if bp_ppl and r.model_type != "backprop"
                else ""
            )
            summary.append(
                f"| {size:,} | {r.model_type} | {r.perplexity:.2f} {ratio} | "
                f"{r.accuracy:.3f} | {r.n_params:,} | {r.train_time:.0f}s |\n"
            )

    # Find crossover point
    summary.append("\n## Key Findings\n\n")

    crossover_found = False
    for size in sorted(by_size.keys()):
        group = by_size[size]
        bp = next((r for r in group if r.model_type == "backprop"), None)
        eq_best = min(
            (r for r in group if "eqprop" in r.model_type),
            key=lambda x: x.perplexity,
            default=None,
        )

        if (
            bp
            and eq_best
            and eq_best.perplexity < bp.perplexity
            and not crossover_found
        ):
            summary.append(f"**Crossover point:** Around {size:,} characters\n")
            summary.append(
                f"- EqProp ({eq_best.model_type}): {eq_best.perplexity:.2f} PPL\n"
            )
            summary.append(f"- Backprop: {bp.perplexity:.2f} PPL\n")
            summary.append(
                f"- Advantage: {bp.perplexity/eq_best.perplexity:.2f}× better\n\n"
            )
            crossover_found = True

    if not crossover_found:
        summary.append("**No crossover found** in tested range.\n\n")

    # Write summary
    with open(output_dir / "summary.md", "w") as f:
        f.writelines(summary)

    print("\n" + "=" * 80)
    print("".join(summary))
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive LM scale study")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=Path("results/lm_scale_study"))
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results = run_experiment_suite(args.device, args.output, quick=args.quick)

    print(f"\n✓ Complete! Results saved to {args.output}")
    print(f"  - JSON: {args.output / 'results_final.json'}")
    print(f"  - CSV: {args.output / 'results_final.csv'}")
    print(f"  - Summary: {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
