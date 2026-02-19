#!/usr/bin/env python3
"""
Quick Start: MNIST Classification with Equilibrium Propagation

This script demonstrates the simplest way to use MEP for training
a neural network without backpropagation.

Run: python examples/quickstart.py
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mep import smep


def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    EPOCHS = 5
    BATCH_SIZE = 128
    LR = 0.01
    
    print(f"Using device: {DEVICE}")
    print(f"Training with Equilibrium Propagation (SMEP)")
    print("=" * 50)

    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, transform=transform)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(DEVICE)

    # Create optimizer with EP
    optimizer = smep(
        model.parameters(),
        model=model,
        lr=LR,
        mode="ep",              # Equilibrium Propagation mode
        beta=0.5,               # Nudging strength
        settle_steps=15,        # Settling iterations
        settle_lr=0.1,          # Settling learning rate
        loss_type="cross_entropy",
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x = x.view(x.size(0), -1)  # Flatten
            
            # EP step - no backward() needed!
            optimizer.step(x=x, target=y)
            optimizer.zero_grad()
            
            # Track training accuracy
            with torch.no_grad():
                output = model(x)
                pred = output.argmax(dim=1)
                train_correct += (pred == y).sum().item()
                train_total += y.size(0)
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                x = x.view(x.size(0), -1)
                output = model(x)
                pred = output.argmax(dim=1)
                test_correct += (pred == y).sum().item()
                test_total += y.size(0)
        
        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Acc: {train_acc:.1f}% | "
              f"Test Acc: {test_acc:.1f}%")

    print("=" * 50)
    print(f"Final Test Accuracy: {test_acc:.1f}%")
    print("\nTo try backprop mode (faster, standard training):")
    print("  Change mode='ep' to mode='backprop' in the optimizer")


if __name__ == "__main__":
    main()
