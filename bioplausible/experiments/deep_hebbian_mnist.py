import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add root to path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from models.hebbian_chain import DeepHebbianChain


def train_linear_probe(features, targets, epochs=100, lr=0.01, device="cuda"):
    """Train a linear classifier on top of fixed features."""
    input_dim = features.size(1)
    num_classes = 10

    classifier = nn.Linear(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    batch_size = 128
    dataset = torch.utils.data.TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    classifier.train()
    for epoch in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            out = classifier(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    return classifier


def evaluate(model, classifier, loader, device):
    """Evaluate full model."""
    model.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x_flat = x.view(x.size(0), -1)

            # Get features
            features = model(x_flat)
            features = features.detach()  # Stop gradients

            # Classify
            out = classifier(features)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return correct / total


def run_deep_hebbian_mnist():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth", type=int, default=1000, help="Depth of Hebbian chain"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Hebbian training epochs")
    parser.add_argument(
        "--probe-epochs", type=int, default=20, help="Linear probe epochs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-sn", action="store_true", help="Disable spectral norm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Deep Hebbian Chain (Depth {args.depth}) on {device}")

    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create Model
    # Input 784 -> Hidden 128 -> Output 128 (features)
    model = DeepHebbianChain(
        input_dim=784,
        hidden_dim=128,
        output_dim=128,  # Output features for probe
        num_layers=args.depth,
        use_spectral_norm=not args.no_sn,
        hebbian_lr=0.01,
    ).to(device)

    print(f"Model created. Spectral Norm: {not args.no_sn}")

    # Phase 1: Hebbian Training
    print(f"\nPhase 1: Hebbian Feature Learning ({args.epochs} epochs)...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        batch_count = 0

        for x, _ in train_loader:
            x = x.view(x.size(0), -1).to(device)

            with torch.no_grad():
                # Forward through input projection
                h = torch.tanh(model.W_in(x))

                # Forward and update chain layer by layer
                # This simulates local learning
                for layer in model.chain:
                    h_in = h
                    if isinstance(layer, nn.Module):  # Handle SN wrapper
                        # Get underlying activation
                        h_out = layer(h_in)
                    else:
                        h_out = layer(h_in)

                    h_out = torch.tanh(h_out)

                    # Apply Hebbian update
                    if hasattr(layer, "hebbian_update"):
                        layer.hebbian_update(h_in, h_out)
                    elif hasattr(layer, "module") and hasattr(
                        layer.module, "hebbian_update"
                    ):
                        layer.module.hebbian_update(h_in, h_out)

                    h = h_out

            batch_count += 1
            if batch_count % 100 == 0:
                print(
                    f"  Epoch {epoch+1}/{args.epochs} [{batch_count}/{len(train_loader)}]"
                )

    hebbian_time = time.time() - start_time
    print(f"Hebbian training complete in {hebbian_time:.2f}s")

    # Phase 2: Feature Extraction & Linear Probe
    print(f"\nPhase 2: Training Linear Probe ({args.probe_epochs} epochs)...")

    # Extract training features
    print("  Extracting features...")
    features_list = []
    targets_list = []

    # Use a subset for probe training to be fast
    probe_samples = 10000
    curr_samples = 0

    with torch.no_grad():
        for x, y in train_loader:
            x = x.view(x.size(0), -1).to(device)
            feats = model(x)
            features_list.append(feats.cpu())
            targets_list.append(y)
            curr_samples += x.size(0)
            if curr_samples >= probe_samples:
                break

    train_features = torch.cat(features_list)
    train_targets = torch.cat(targets_list)

    # Train probe
    probe = train_linear_probe(
        train_features.to(device),
        train_targets.to(device),
        epochs=args.probe_epochs,
        device=device,
    )

    # Evaluate
    acc = evaluate(model, probe, test_loader, device)
    print(f"\nFinal Test Accuracy (Depth {args.depth}): {acc*100:.2f}%")

    # Sanity check: Signal propagation stats
    stats = model.measure_signal_propagation(x[:100].to(device))
    print(f"Signal Decay Ratio: {stats['decay_ratio']:.4f}")
    if stats["decay_ratio"] > 0.01:
        print("✅ Signal SURVIVED deep chain!")
    else:
        print("❌ Signal VANISHED.")


if __name__ == "__main__":
    run_deep_hebbian_mnist()
