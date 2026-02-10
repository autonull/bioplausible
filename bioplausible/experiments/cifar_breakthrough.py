#!/usr/bin/env python3
"""
CIFAR-10 Breakthrough Training Script (Track 34)

Train ModernConvEqProp on CIFAR-10 with target of 75%+ accuracy.

Usage:
    python experiments/cifar_breakthrough.py --epochs 50 --seed 42
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Add project root to path so we can import bioplausible
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models import ModernConvEqProp, SimpleConvEqProp


def get_cifar10_loader(batch_size=128, num_workers=2, augment=True):
    """Create CIFAR-10 train and test loaders with data augmentation."""

    # Normalization stats for CIFAR-10
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with augmentation
    if augment:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    # Test transforms (no augmentation)
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    # Download and load datasets
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"  Batch {batch_idx+1}/{len(train_loader)}: "
                f"loss={loss.item():.3f}, acc={100*correct/total:.1f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="modern",
        choices=["modern", "simple"],
        help="Model architecture (modern or simple)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--eq-steps", type=int, default=15, help="Equilibrium steps")
    parser.add_argument(
        "--hidden-channels", type=int, default=64, help="Hidden channels"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--no-augment", action="store_true", help="Disable data augmentation"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto/cpu/cuda)"
    )
    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("=" * 60)
    print("CIFAR-10 BREAKTHROUGH TRAINING (Track 34)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"EQ steps: {args.eq_steps}")
    print(f"Hidden channels: {args.hidden_channels}")
    print(f"Seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Data augmentation: {not args.no_augment}")
    print("=" * 60)
    print()

    # Create data loaders
    print("Loading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loader(
        batch_size=args.batch_size, num_workers=2, augment=not args.no_augment
    )
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    print()

    # Create model
    print(f"Creating {args.model} model...")
    if args.model == "modern":
        model = ModernConvEqProp(
            eq_steps=args.eq_steps,
            hidden_channels=args.hidden_channels,
            gamma=0.5,
            use_spectral_norm=True,
        )
    else:
        model = SimpleConvEqProp(
            hidden_channels=args.hidden_channels,
            eq_steps=args.eq_steps,
            gamma=0.5,
            use_spectral_norm=True,
        )

    model = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Compute Lipschitz constant
    if hasattr(model, "compute_lipschitz"):
        L = model.compute_lipschitz()
        print(f"  Lipschitz constant: {L:.4f}")
    print()

    # Optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_test_acc = 0.0
    results = {
        "model": args.model,
        "config": vars(args),
        "epochs": [],
    }

    print("Starting training...")
    print()

    for epoch in range(args.epochs):
        start_time = time.time()

        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        epoch_time = time.time() - start_time

        print(f"  Train: loss={train_loss:.3f}, acc={train_acc:.2f}%")
        print(f"  Test:  loss={test_loss:.3f}, acc={test_acc:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Track best
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f"  ✅ New best: {best_test_acc:.2f}%")

            # Save checkpoint
            save_dir = Path(f"results/track_34")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "config": vars(args),
                },
                save_dir / f"{args.model}_seed{args.seed}_best.pt",
            )

        print()

        # Store results
        results["epochs"].append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "time": epoch_time,
            }
        )

    # Final summary
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best test accuracy: {best_test_acc:.2f}%")
    print(f"Target: 75%")

    if best_test_acc >= 75.0:
        print("✅ TARGET ACHIEVED!")
    elif best_test_acc >= 70.0:
        print("⚠️ Close to target (≥70%)")
    else:
        print("❌ Below target")

    # Save results
    save_dir = Path(f"results/track_34")
    results["best_test_acc"] = best_test_acc
    with open(save_dir / f"{args.model}_seed{args.seed}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {save_dir.absolute()}")


if __name__ == "__main__":
    main()
