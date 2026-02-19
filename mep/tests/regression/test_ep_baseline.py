#!/usr/bin/env python3
"""
EP Regression Test Suite

Run this AFTER every code change and BEFORE any complex benchmarks.

If this fails, DO NOT run complex benchmarks. Fix the bug first.

Usage:
    python tests/regression/test_ep_baseline.py

Expected output:
    - MNIST 3-epoch accuracy > 55%
    - 5000-layer training completes without NaN/Inf
    - Total runtime < 3 minutes

Pass criteria:
    - All tests pass
    - MNIST accuracy > 55% (3 epochs, small model)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import time

from mep import EPOptimizer, smep


def get_mnist_loaders(batch_size=128):
    """Load MNIST."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, len(test_dataset)


class SmallMLP(nn.Module):
    """Small MLP for quick testing."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        return self.net(x)


def test_mnist_accuracy():
    """Test that EP actually learns on MNIST."""
    print("\n" + "="*60)
    print("TEST 1: MNIST Learning (3 epochs)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, test_loader, test_size = get_mnist_loaders(batch_size=128)
    
    model = SmallMLP().to(device)
    
    # Use smep preset (known working configuration)
    opt = smep(
        model.parameters(),
        model=model,
        lr=0.01,
        mode='ep',
        settle_steps=30,
        settle_lr=0.15,
        beta=0.5,
        loss_type='mse',
    )
    
    start = time.time()
    
    for epoch in range(3):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device).flatten(1), y.to(device)
            opt.step(x=x, target=y)
        
        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device).flatten(1), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
        
        acc = correct / test_size
        print(f"  Epoch {epoch+1}/3: Accuracy = {acc*100:.1f}%")
    
    elapsed = time.time() - start
    
    # Pass criteria: > 55% after 3 epochs
    final_acc = acc
    passed = final_acc > 0.55
    
    print(f"\n  Result: {'✓ PASS' if passed else '✗ FAIL'}")
    print(f"  Accuracy: {final_acc*100:.1f}% (target: >55%)")
    print(f"  Time: {elapsed:.1f}s")
    
    return passed, {'accuracy': final_acc, 'time': elapsed}


def test_deep_stability():
    """Test that EP trains stably at depth."""
    print("\n" + "="*60)
    print("TEST 2: Deep Network Stability (1000 layers)")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create deep model
    class DeepMLP(nn.Module):
        def __init__(self, num_layers=1000):
            super().__init__()
            layers = []
            layers.append(nn.Linear(64, 64))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(64, 64))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(64, 10))
            self.net = nn.Sequential(*layers)
            
            # Init for stability
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            return self.net(x)
    
    model = DeepMLP(num_layers=1000).to(device)
    
    # Single training step
    x = torch.randn(16, 64, device=device)
    y = torch.randint(0, 10, (16,), device=device)
    
    opt = EPOptimizer(
        model.parameters(),
        model=model,
        mode='ep',
        settle_steps=10,
        settle_lr=0.2,
    )
    
    try:
        opt.step(x=x, target=y)
        
        # Check for NaN/Inf in gradients
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)
        
        passed = not has_nan and not has_inf
        
        print(f"  Result: {'✓ PASS' if passed else '✗ FAIL'}")
        print(f"  NaN gradients: {has_nan}")
        print(f"  Inf gradients: {has_inf}")
        
        return passed, {'nan': has_nan, 'inf': has_inf}
        
    except Exception as e:
        print(f"  Result: ✗ FAIL")
        print(f"  Error: {e}")
        return False, {'error': str(e)}


def test_backward_compatibility():
    """Test that preset functions still work."""
    print("\n" + "="*60)
    print("TEST 3: Backward Compatibility")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SmallMLP().to(device)
    x = torch.randn(8, 784, device=device)
    y = torch.randint(0, 10, (8,), device=device)
    
    tests_passed = 0
    tests_total = 0
    
    # Test smep
    tests_total += 1
    try:
        opt = smep(model.parameters(), model=model)
        opt.step(x=x, target=y)
        print("  ✓ smep works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ smep failed: {e}")
    
    # Test smep_fast
    tests_total += 1
    try:
        from mep import smep_fast
        opt = smep_fast(model.parameters(), model=model)
        opt.step(x=x, target=y)
        print("  ✓ smep_fast works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ smep_fast failed: {e}")
    
    # Test muon_backprop
    tests_total += 1
    try:
        from mep import muon_backprop
        opt = muon_backprop(model.parameters())
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
        loss.backward()
        opt.step()
        print("  ✓ muon_backprop works")
        tests_passed += 1
    except Exception as e:
        print(f"  ✗ muon_backprop failed: {e}")
    
    passed = tests_passed == tests_total
    print(f"\n  Result: {'✓ PASS' if passed else '✗ FAIL'} ({tests_passed}/{tests_total})")
    
    return passed, {'passed': tests_passed, 'total': tests_total}


def main():
    print("="*60)
    print("EP REGRESSION TEST SUITE")
    print("="*60)
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        'mnist': (False, {}),
        'deep': (False, {}),
        'compat': (False, {}),
    }
    
    # Run tests
    results['mnist'] = test_mnist_accuracy()
    results['deep'] = test_deep_stability()
    results['compat'] = test_backward_compatibility()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_passed = all(r[0] for r in results.values())
    
    print(f"  MNIST Learning:      {'✓ PASS' if results['mnist'][0] else '✗ FAIL'}")
    print(f"  Deep Stability:      {'✓ PASS' if results['deep'][0] else '✗ FAIL'}")
    print(f"  Backward Compat:     {'✓ PASS' if results['compat'][0] else '✗ FAIL'}")
    print(f"\n  Overall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("="*60)
    
    if not all_passed:
        print("\n⚠️  REGRESSION DETECTED!")
        print("⚠️  DO NOT run complex benchmarks.")
        print("⚠️  Fix the bug first, then re-run this test.")
        print(f"\nFailed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(1)
    else:
        print(f"\n✓ All tests passed. Safe to run complex benchmarks.")
        print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sys.exit(0)


if __name__ == "__main__":
    main()
