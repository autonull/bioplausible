#!/usr/bin/env python3
"""
Final Characterization of EqProp vs Backprop

Tests three critical regimes to validate the "Implicit Regularization" hypothesis:
1. High-Compute (Overfitting Risk): 1500 updates on 10K data.
2. Low-Compute (Speed Check): 120 updates on 10K data.
3. Few-Shot (Small Data): 5-200 samples.

Saves intermediate results after EVERY model training.
"""

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import BackpropTransformerLM, get_eqprop_lm


def load_shakespeare(max_chars=None):
    """Load Shakespeare dataset."""
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
    if len(data) <= seq_len:
        raise ValueError(f"Data length {len(data)} <= seq_len {seq_len}")
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix]).to(device)
    return x, y


def train_model(model, train_data, val_data, vocab_size, config, device, name):
    """Train single model and return stats."""
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()
    batches_per_epoch = config["batches_per_epoch"]
    history = []

    print(
        f"  Training {name} ({config['epochs']} eps, {batches_per_epoch} batch/ep)..."
    )

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0

        for _ in range(batches_per_epoch):
            try:
                x, y = get_batch(
                    train_data, config["seq_len"], config["batch_size"], device
                )
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            except ValueError:
                continue

        # Validation
        val_loss = 0
        n_val = 20
        model.eval()
        with torch.no_grad():
            for _ in range(n_val):
                try:
                    x, y = get_batch(
                        val_data, config["seq_len"], config["batch_size"], device
                    )
                    logits = model(x)
                    loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                    val_loss += loss.item()
                except ValueError:
                    continue

        avg_val_loss = val_loss / n_val
        ppl = torch.exp(torch.tensor(avg_val_loss)).item()
        history.append({"epoch": epoch + 1, "val_loss": avg_val_loss, "ppl": ppl})

        if (epoch + 1) % 5 == 0:
            print(f"    Ep {epoch+1}: PPL {ppl:.2f}")

    train_time = time.time() - start_time

    # Final metrics
    final_ppl = history[-1]["ppl"]

    return {
        "name": name,
        "config_type": config["type"],
        "dataset_size": config["dataset_size"],
        "updates_total": config["epochs"] * batches_per_epoch,
        "perplexity": final_ppl,
        "train_time": train_time,
        "history": history,
        "params": sum(p.numel() for p in model.parameters()),
    }


def save_result(result, output_dir):
    """Append result to JSON/CSV immediately."""
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    # 1. JSON (Read-Modify-Write)
    data = []
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except:
            pass
    data.append(result)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # 2. CSV (Append)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "config_type",
                "dataset_size",
                "updates_total",
                "perplexity",
                "train_time",
                "params",
            ],
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"  ✓ Saved result for {result['name']}")


def run_experiments(device, output_dir, seeds=3):
    """Run all regimes."""
    print(f"Running characterization on {device}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define Regimes
    regimes = [
        # REGIME A: Low Compute (Scale Study replication)
        {
            "type": "low_compute_10k",
            "dataset_size": 10000,
            "seq_len": 64,
            "hidden": 128,
            "layers": 3,
            "epochs": 30,
            "batch_size": 32,
            "batches_per_epoch": 4,  # 120 updates
            "lr": 3e-4,
        },
        # REGIME B: High Compute (Track 37 replication)
        {
            "type": "high_compute_10k",
            "dataset_size": 10000,
            "seq_len": 64,
            "hidden": 128,
            "layers": 3,
            "epochs": 30,
            "batch_size": 32,
            "batches_per_epoch": 50,  # 1500 updates
            "lr": 3e-4,
        },
        # REGIME C: Few-Shot (Small Data)
        {
            "type": "few_shot_1k",
            "dataset_size": 1000,
            "seq_len": 32,
            "hidden": 64,
            "layers": 2,
            "epochs": 30,
            "batch_size": 16,
            "batches_per_epoch": 10,  # 300 updates
            "lr": 3e-4,
        },
    ]

    for regime in regimes:
        print(f"\n{'='*60}")
        print(
            f"Regime: {regime['type']} (Size {regime['dataset_size']}, Updates {regime['epochs']*regime['batches_per_epoch']})"
        )
        print("=" * 60)

        for seed in range(seeds):
            print(f"\nSeed {seed+1}/{seeds}")
            torch.manual_seed(seed)

            # Load Data (per size)
            train_data, val_data, vocab_size = load_shakespeare(regime["dataset_size"])
            train_data = train_data.to(device)
            val_data = val_data.to(device)

            # 1. Backprop
            try:
                bp_model = BackpropTransformerLM(
                    vocab_size=vocab_size,
                    hidden_dim=regime["hidden"],
                    num_layers=regime["layers"],
                    num_heads=4,
                ).to(device)

                res = train_model(
                    bp_model,
                    train_data,
                    val_data,
                    vocab_size,
                    regime,
                    device,
                    f"Backprop_s{seed}",
                )
                save_result(res, output_dir)
            except Exception as e:
                print(f"Error Backprop: {e}")
                traceback.print_exc()

            # 2. EqProp (Recurrent Core - best balance)
            try:
                eq_model = get_eqprop_lm(
                    name="recurrent_core",
                    vocab_size=vocab_size,
                    hidden_dim=regime["hidden"],
                    num_layers=regime["layers"],
                    num_heads=4,
                    max_eq_steps=15,
                ).to(device)

                res = train_model(
                    eq_model,
                    train_data,
                    val_data,
                    vocab_size,
                    regime,
                    device,
                    f"EqProp_s{seed}",
                )
                save_result(res, output_dir)
            except Exception as e:
                print(f"Error EqProp: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, default=Path("results/final_char"))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    run_experiments(args.device, args.output, args.seeds)
