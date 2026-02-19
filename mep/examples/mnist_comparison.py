#!/usr/bin/env python3
"""
MNIST Training: EP vs Backprop Comparison

This script compares Equilibrium Propagation (EP) with standard
backpropagation on MNIST classification.

Run: python examples/mnist_comparison.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

from mep import smep, muon_backprop


def load_mnist(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, test_loader


def create_model(device):
    """Create a simple MLP model."""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100 * correct / total


def train_epoch_ep(model, optimizer, train_loader, device):
    """Train one epoch with Equilibrium Propagation."""
    model.train()
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        
        optimizer.step(x=x, target=y)
        optimizer.zero_grad()
        
        # Track accuracy
        with torch.no_grad():
            output = model(x)
            pred = output.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return 100 * correct / total


def train_epoch_bp(model, optimizer, criterion, train_loader, device):
    """Train one epoch with backpropagation."""
    model.train()
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
        
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        pred = output.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    return 100 * correct / total


def run_experiment(mode, epochs, device, train_loader, test_loader):
    """Run training experiment with specified mode."""
    model = create_model(device)
    
    if mode == "ep":
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode="ep",
            beta=0.5,
            settle_steps=15,
            settle_lr=0.1,
            loss_type="cross_entropy",
        )
        train_fn = lambda m, o, t, d: train_epoch_ep(m, o, t, d)
    else:
        optimizer = muon_backprop(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
        )
        criterion = nn.CrossEntropyLoss()
        train_fn = lambda m, o, t, d: train_epoch_bp(m, o, criterion, t, d)
    
    results = []
    start_time = time.time()
    
    for epoch in range(epochs):
        train_acc = train_fn(model, optimizer, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        results.append({"train": train_acc, "test": test_acc})
        print(f"  Epoch {epoch+1}/{epochs}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    elapsed = time.time() - start_time
    final_acc = results[-1]["test"]
    
    return final_acc, elapsed, results


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    
    print("=" * 60)
    print("MNIST Classification: EP vs Backpropagation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print()
    
    # Load data
    print("Loading MNIST...")
    train_loader, test_loader = load_mnist(batch_size=128)
    print()
    
    # Train with Backpropagation
    print("-" * 60)
    print("Training with Backpropagation (Muon optimizer)")
    print("-" * 60)
    bp_acc, bp_time, _ = run_experiment("bp", EPOCHS, DEVICE, train_loader, test_loader)
    print(f"→ Final Accuracy: {bp_acc:.1f}%, Time: {bp_time:.1f}s")
    print()
    
    # Train with EP
    print("-" * 60)
    print("Training with Equilibrium Propagation (SMEP)")
    print("-" * 60)
    ep_acc, ep_time, _ = run_experiment("ep", EPOCHS, DEVICE, train_loader, test_loader)
    print(f"→ Final Accuracy: {ep_acc:.1f}%, Time: {ep_time:.1f}s")
    print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Backpropagation: {bp_acc:.1f}% accuracy in {bp_time:.1f}s")
    print(f"Equilibrium Prop: {ep_acc:.1f}% accuracy in {ep_time:.1f}s")
    print(f"Speed ratio: EP is {ep_time/bp_time:.1f}x slower")
    print(f"Accuracy gap: {bp_acc - ep_acc:.1f} percentage points")
    print()
    print("Notes:")
    print("- EP requires more settling steps for convergence")
    print("- EP uses only local learning rules (biologically plausible)")
    print("- EP has O(1) memory cost vs O(depth) for backprop")


if __name__ == "__main__":
    main()
