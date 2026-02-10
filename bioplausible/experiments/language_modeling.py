#!/usr/bin/env python3
"""
Track 37: Character-Level Language Modeling

Train CausalTransformerEqProp on Shakespeare or WikiText-2.
Target: Perplexity < 2.5 (Shakespeare), < 60 (WikiText-2)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models import CausalTransformerEqProp


def load_shakespeare():
    """Load Shakespeare dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    path = Path("data/shakespeare.txt")
    path.parent.mkdir(exist_ok=True)

    if not path.exists():
        import urllib.request

        print(f"Downloading Shakespeare dataset...")
        urllib.request.urlretrieve(url, path)

    with open(path, "r") as f:
        text = f.read()

    # Create character vocab
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Encode
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)

    return data, char_to_idx, idx_to_char


def get_batch(data, seq_len, batch_size, device):
    """Get a batch of sequences."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def train_epoch(model, data_train, optimizer, criterion, config, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = config["batches_per_epoch"]

    for _ in range(num_batches):
        x, y = get_batch(data_train, config["seq_len"], config["batch_size"], device)

        optimizer.zero_grad()
        logits = model(x)  # [batch, seq, vocab]
        logits = logits.reshape(-1, logits.size(-1))
        y = y.reshape(-1)

        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def evaluate(model, data_val, criterion, config, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    num_batches = config["val_batches"]

    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(data_val, config["seq_len"], config["batch_size"], device)

            logits = model(x)
            logits = logits.reshape(-1, logits.size(-1))
            y = y.reshape(-1)

            loss = criterion(logits, y)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="shakespeare", choices=["shakespeare"]
    )
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument("--eq-steps", type=int, default=20, help="Equilibrium steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    )

    print("=" * 60)
    print(f"TRACK 37: Language Modeling - {args.dataset}")
    print("=" * 60)
    print(f"Config: {vars(args)}")
    print(f"Device: {device}")
    print()

    # Load dataset
    print("Loading dataset...")
    if args.dataset == "shakespeare":
        data, char_to_idx, idx_to_char = load_shakespeare()
        vocab_size = len(char_to_idx)

        # Split train/val
        n = int(0.9 * len(data))
        data_train, data_val = data[:n], data[n:]

        print(f"  Vocab size: {vocab_size}")
        print(f"  Train size: {len(data_train)}")
        print(f"  Val size: {len(data_val)}")

    # Create model
    print("\nCreating model...")
    model = CausalTransformerEqProp(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        eq_steps=args.eq_steps,
    ).to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training config
    config = {
        "batches_per_epoch": 200,
        "val_batches": 50,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
    }

    # Training loop
    print("\nTraining...")
    best_perplexity = float("inf")
    results = []

    for epoch in range(args.epochs):
        start = time.time()

        train_loss = train_epoch(
            model, data_train, optimizer, criterion, config, device
        )
        val_loss, val_perplexity = evaluate(model, data_val, criterion, config, device)

        elapsed = time.time() - start

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_loss:.3f}, val_ppl={val_perplexity:.2f}, time={elapsed:.1f}s"
        )

        if val_perplexity < best_perplexity:
            best_perplexity = val_perplexity
            print(f"  ✅ New best perplexity: {best_perplexity:.2f}")

        results.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best perplexity: {best_perplexity:.2f}")
    print(f"Target: 2.5 (Shakespeare)")

    if best_perplexity < 2.5:
        print("✅ TARGET ACHIEVED")
    elif best_perplexity < 3.0:
        print("⚠️ Close to target")
    else:
        print("❌ Did not achieve target")

    # Save results
    save_dir = Path("results/track_37")
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / f"results_seed{args.seed}.json", "w") as f:
        json.dump(
            {
                "config": vars(args),
                "best_perplexity": best_perplexity,
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
