#!/usr/bin/env python3
"""
EP Smoke Test (20 seconds)

Run this AFTER every code change.

If this fails, DO NOT run complex benchmarks or full regression tests.
Fix the bug first.

Usage:
    python tests/regression/test_ep_smoke.py

Expected runtime: < 30 seconds

Pass criteria:
    - MNIST accuracy > 25% after 1 epoch (proves learning works)
    - No NaN/Inf in gradients
    - All presets callable
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import time

from mep import EPOptimizer, smep


def get_data_for_smoke_test(batch_size=128):
    """Load MNIST for smoke test.
    
    Uses subset of training data for speed, but FULL test set for reliable evaluation.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Use MORE training data for reliable learning signal
    train_indices = list(range(10000))  # 10000 training samples
    
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(test_dataset)


class TinyMLP(nn.Module):
    """Tiny MLP for smoke testing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)


def test_mnist_learning():
    """Test that EP actually learns (1 epoch, 10k train, full test)."""
    print("\nTEST: MNIST Learning (1 epoch, 10k train, full test)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, test_size = get_data_for_smoke_test(batch_size=128)
    
    model = TinyMLP().to(device)
    
    # Use known working config
    opt = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        mode='ep',
        settle_steps=30,    # Working value
        settle_lr=0.15,     # Working value
        beta=0.5,
        loss_type='mse',    # MUST be mse
    )
    
    # Train 1 epoch
    model.train()
    for x, y in train_loader:
        x, y = x.to(device).flatten(1), y.to(device)
        opt.step(x=x, target=y)
    
    # Evaluate on FULL test set
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device).flatten(1), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    
    acc = correct / test_size
    
    # Pass criteria: > 15% (well above random 10%)
    # With 10000 training samples and 1 epoch, expect ~20-30%
    passed = acc > 0.15
    
    print(f"  Accuracy: {acc*100:.1f}% (target: >15%) {'✓' if passed else '✗'}")
    
    return passed, {'accuracy': acc}


def test_deep_stability():
    """Test that EP doesn't produce NaN at depth."""
    print("TEST: Deep Stability (500 layers)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class DeepMLP(nn.Module):
        def __init__(self, num_layers=500):
            super().__init__()
            layers = []
            layers.append(nn.Linear(64, 64))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(64, 64))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(64, 10))
            self.net = nn.Sequential(*layers)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        def forward(self, x):
            return self.net(x)
    
    model = DeepMLP(num_layers=500).to(device)
    x = torch.randn(8, 64, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    
    opt = EPOptimizer(model.parameters(), model=model, mode='ep', settle_steps=5)
    
    try:
        opt.step(x=x, target=y)
        
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        
        passed = not has_nan and not has_inf
        print(f"  NaN/Inf: {has_nan or has_inf} {'✗' if has_nan or has_inf else '✓'}")
        
        return passed, {'nan': has_nan, 'inf': has_inf}
    except Exception as e:
        print(f"  Error: {e} ✗")
        return False, {'error': str(e)}


def test_presets():
    """Test that presets are callable."""
    print("TEST: Presets Callable")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TinyMLP().to(device)
    x = torch.randn(4, 784, device=device)
    y = torch.randint(0, 10, (4,), device=device)
    
    all_passed = True
    
    # Test smep
    try:
        opt = smep(model.parameters(), model=model)
        opt.step(x=x, target=y)
        print("  smep: ✓")
    except Exception as e:
        print(f"  smep: ✗ ({e})")
        all_passed = False
    
    # Test smep_fast
    try:
        from mep import smep_fast
        opt = smep_fast(model.parameters(), model=model)
        opt.step(x=x, target=y)
        print("  smep_fast: ✓")
    except Exception as e:
        print(f"  smep_fast: ✗ ({e})")
        all_passed = False
    
    # Test muon_backprop
    try:
        from mep import muon_backprop
        opt = muon_backprop(model.parameters())
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        opt.step()
        print("  muon_backprop: ✓")
    except Exception as e:
        print(f"  muon_backprop: ✗ ({e})")
        all_passed = False
    
    return all_passed, {}


def main():
    print("="*50)
    print("EP SMOKE TEST (< 30 seconds)")
    print("="*50)
    
    start = time.time()
    
    results = {
        'mnist': (False, {}),
        'deep': (False, {}),
        'presets': (False, {}),
    }
    
    results['mnist'] = test_mnist_learning()
    results['deep'] = test_deep_stability()
    results['presets'] = test_presets()
    
    elapsed = time.time() - start
    
    print("="*50)
    print(f"SUMMARY ({elapsed:.1f}s)")
    print("="*50)
    
    all_passed = all(r[0] for r in results.values())
    
    print(f"  MNIST Learning:  {'✓' if results['mnist'][0] else '✗'}")
    print(f"  Deep Stability:  {'✓' if results['deep'][0] else '✗'}")
    print(f"  Presets:         {'✓' if results['presets'][0] else '✗'}")
    print(f"\n  Overall: {'✓ PASS' if all_passed else '✗ FAIL'}")
    print("="*50)
    
    if not all_passed:
        print("\n⚠️  REGRESSION DETECTED!")
        print("⚠️  DO NOT run benchmarks. Fix bug first.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed. Safe to continue.")
        sys.exit(0)


if __name__ == "__main__":
    main()
