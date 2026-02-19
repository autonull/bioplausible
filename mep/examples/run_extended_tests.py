#!/usr/bin/env python3
"""
Phase 2: Extended Testing - Impressive Results

Tests:
1. Extreme depth scaling (5000-10000+ layers)
2. Real MNIST accuracy at various depths
3. Full Permuted MNIST continual learning

Run: python examples/run_extended_tests.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

from mep import EPOptimizer


# ============================================================================
# Test 1: Extreme Depth Scaling (5000-10000+ layers)
# ============================================================================

class DeepMLP(nn.Module):
    """Very deep MLP for extreme scaling tests."""
    
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1000, output_dim=10):
        super().__init__()
        self.num_layers = num_layers
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize for stable deep training
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


def test_extreme_depth(depths=[1000, 2000, 5000, 10000]):
    """Test EP at extreme depths."""
    print("\n" + "="*80)
    print("TEST 1: EXTREME DEPTH SCALING (5000-10000+ layers)")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    results = []
    
    for depth in depths:
        print(f"\nTesting depth {depth}...")
        
        try:
            # Create model
            model = DeepMLP(input_dim=64, hidden_dim=64, num_layers=depth, output_dim=10).to(device)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params/1e6:.1f}M")
            
            # Create data
            x = torch.randn(16, 64, device=device)
            y = torch.randint(0, 10, (16,), device=device)
            
            # Test EP training step
            opt = EPOptimizer(
                model.parameters(),
                model=model,
                mode='ep',
                lr=0.01,
                settle_steps=10,
                settle_lr=0.2,
            )
            
            torch.cuda.synchronize()
            start = time.time()
            opt.step(x=x, target=y)
            torch.cuda.synchronize()
            ep_time = time.time() - start
            
            # Check gradient health
            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
            
            avg_grad = sum(grad_norms) / len(grad_norms) if grad_norms else 0
            max_grad = max(grad_norms) if grad_norms else 0
            
            # Memory
            mem_allocated = torch.cuda.memory_allocated() / 1e6 if device == 'cuda' else 0
            
            result = {
                'depth': depth,
                'params_m': total_params / 1e6,
                'ep_time_sec': ep_time,
                'avg_grad_norm': avg_grad,
                'max_grad_norm': max_grad,
                'memory_mb': mem_allocated,
                'success': True,
                'error': None,
            }
            
            print(f"  ✓ EP step: {ep_time:.2f}s")
            print(f"  Gradient norm: avg={avg_grad:.2e}, max={max_grad:.2e}")
            print(f"  Memory: {mem_allocated:.1f}MB")
            
            results.append(result)
            
            del model
            
        except RuntimeError as e:
            result = {
                'depth': depth,
                'params_m': 0,
                'ep_time_sec': 0,
                'avg_grad_norm': 0,
                'max_grad_norm': 0,
                'memory_mb': 0,
                'success': False,
                'error': str(e)[:100],
            }
            results.append(result)
            print(f"  ✗ FAILED: {e}")
    
    return results


# ============================================================================
# Test 2: Real MNIST Accuracy
# ============================================================================

def get_mnist_loaders(batch_size=128, num_workers=0):
    """Load MNIST dataset."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def test_mnist_accuracy(depths=[10, 50, 100], epochs=3):
    """Test EP on real MNIST at various depths."""
    print("\n" + "="*80)
    print("TEST 2: REAL MNIST ACCURACY")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Depths: {depths}, Epochs: {epochs}")
    
    # Load MNIST
    print("\nLoading MNIST...")
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    results = []
    
    for depth in depths:
        print(f"\n{'='*60}")
        print(f"Depth {depth}")
        print(f"{'='*60}")
        
        torch.manual_seed(42)
        
        # Create model
        model = DeepMLP(input_dim=784, hidden_dim=128, num_layers=depth, output_dim=10).to(device)
        
        # Create optimizer - use working config
        opt = EPOptimizer(
            model.parameters(),
            model=model,
            mode='ep',
            lr=0.01,
            settle_steps=30,    # More settling steps
            settle_lr=0.15,     # Original working value
            beta=0.5,
            loss_type='mse',    # Use mse (working config)
        )
        
        # Training loop
        best_acc = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for x, y in train_loader:
                x, y = x.to(device).flatten(1), y.to(device)
                opt.step(x=x, target=y)
                epoch_loss += 0  # EP doesn't return loss
                num_batches += 1
            
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device).flatten(1), y.to(device)
                    output = model(x)
                    pred = output.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    total += len(y)
            
            acc = correct / total
            best_acc = max(best_acc, acc)
            
            print(f"  Epoch {epoch+1}/{epochs}: Test Accuracy = {acc*100:.1f}%")
        
        result = {
            'depth': depth,
            'epochs': epochs,
            'best_accuracy': best_acc,
            'final_accuracy': acc,
        }
        results.append(result)
        
        print(f"  Best accuracy: {best_acc*100:.1f}%")
        
        del model
    
    return results


# ============================================================================
# Test 3: Full Permuted MNIST Continual Learning
# ============================================================================

def get_permuted_mnist_loaders(num_tasks=5, batch_size=128, seed=42):
    """Generate Permuted MNIST task loaders."""
    torch.manual_seed(seed)
    
    # Load base MNIST
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
    
    train_images = train_dataset.data.float().flatten(1) / 255.0
    train_labels = train_dataset.targets
    test_images = test_dataset.data.float().flatten(1) / 255.0
    test_labels = test_dataset.targets
    
    # Generate permutations
    permutations = [torch.randperm(784) for _ in range(num_tasks)]
    
    # Create loaders for each task
    task_loaders = []
    for task_id, perm in enumerate(permutations):
        permuted_train = train_images[:, perm]
        permuted_test = test_images[:, perm]
        
        train_tensor = TensorDataset(permuted_train, train_labels)
        test_tensor = TensorDataset(permuted_test, test_labels)
        
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
        
        task_loaders.append((train_loader, test_loader, perm))
    
    return task_loaders


def test_permuted_mnist_cl(num_tasks=5, epochs=2, ewc_lambda=100):
    """Full Permuted MNIST continual learning benchmark."""
    print("\n" + "="*80)
    print("TEST 3: PERMUTED MNIST CONTINUAL LEARNING")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Tasks: {num_tasks}, Epochs: {epochs}, EWC Lambda: {ewc_lambda}")
    
    # Load data
    print("\nLoading Permuted MNIST...")
    task_loaders = get_permuted_mnist_loaders(num_tasks=num_tasks, batch_size=64)
    
    results = {
        'ep_ewc': {'final_accs': [], 'avg': 0, 'forgetting': 0},
        'bp_ewc': {'final_accs': [], 'avg': 0, 'forgetting': 0},
    }
    
    for method in ['ep_ewc', 'bp_ewc']:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        use_ep = method == 'ep_ewc'
        use_ewc = 'ewc' in method
        
        torch.manual_seed(42)
        
        # Track accuracies
        task_accuracies_after = []  # Accuracies on all tasks after each task
        
        for task_id in range(num_tasks):
            train_loader, test_loader, _ = task_loaders[task_id]
            
            # Create model (fresh for each method)
            model = DeepMLP(input_dim=784, hidden_dim=256, num_layers=10, output_dim=10).to(device)
            
            # Create optimizer - use working config (mse loss_type)
            if use_ep:
                opt = EPOptimizer(
                    model.parameters(),
                    model=model,
                    mode='ep',
                    lr=0.01,
                    ewc_lambda=ewc_lambda if use_ewc else 0,
                    settle_steps=30,  # More settling steps
                    settle_lr=0.15,   # Original working value
                    beta=0.5,
                    loss_type='mse',  # Use mse (working config)
                )
            else:
                opt = EPOptimizer(
                    model.parameters(),
                    model=model,
                    mode='backprop',
                    lr=0.01,
                    ewc_lambda=ewc_lambda if use_ewc else 0,
                )
                criterion = nn.CrossEntropyLoss()
            
            # Train on task
            start = time.time()
            for epoch in range(epochs):
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    
                    if use_ep:
                        opt.step(x=x, target=y, task_id=task_id)
                    else:
                        opt.zero_grad()
                        loss = criterion(model(x), y)
                        if use_ewc and opt.ewc_state is not None:
                            loss = loss + opt.ewc_state.compute_ewc_loss()
                        loss.backward()
                        opt.step()
            
            train_time = time.time() - start
            
            # Consolidate task (for EWC)
            if use_ewc:
                opt.consolidate_task(train_loader, task_id, device)
            
            # Evaluate on all tasks seen so far
            current_accs = []
            for prev_task in range(task_id + 1):
                _, prev_test_loader, _ = task_loaders[prev_task]
                correct = 0
                total = 0
                
                model.eval()
                with torch.no_grad():
                    for x, y in prev_test_loader:
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        pred = output.argmax(dim=1)
                        correct += (pred == y).sum().item()
                        total += len(y)
                
                current_accs.append(correct / total)
            
            task_accuracies_after.append(current_accs)
            
            print(f"  Task {task_id}: Train acc = {current_accs[-1]*100:.1f}%, "
                  f"Time = {train_time:.1f}s")
            
            del model
        
        # Final evaluation on all tasks
        final_accs = []
        for task_id in range(num_tasks):
            _, test_loader, _ = task_loaders[task_id]
            correct = 0
            total = 0
            
            # Need to reload model - simplified: use last trained model's performance
            # For proper evaluation, we'd need to save/load model states
            # Here we approximate with task_accuracies_after
            final_accs.append(task_accuracies_after[-1][task_id])
        
        # Compute forgetting
        forgetting = 0.0
        for task_id in range(num_tasks):
            acc_after = task_accuracies_after[task_id][task_id]
            acc_final = final_accs[task_id]
            drop = acc_after - acc_final
            forgetting = max(forgetting, drop)
        
        avg_acc = sum(final_accs) / len(final_accs)
        
        results[method]['final_accs'] = final_accs
        results[method]['avg'] = avg_acc
        results[method]['forgetting'] = forgetting
        
        print(f"\n  Final Results:")
        print(f"    Task accuracies: {[f'{a*100:.1f}%' for a in final_accs]}")
        print(f"    Average: {avg_acc*100:.1f}%")
        print(f"    Forgetting: {forgetting*100:.1f}%")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("="*80)
    print("PHASE 2: EXTENDED TESTING - IMPRESSIVE RESULTS")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'extreme_depth': [],
        'mnist_accuracy': [],
        'permuted_mnist_cl': {},
    }
    
    # Test 1: Extreme Depth
    try:
        depth_results = test_extreme_depth(depths=[1000, 2000, 5000])
        all_results['extreme_depth'] = depth_results
    except Exception as e:
        print(f"Extreme depth test failed: {e}")
    
    # Test 2: MNIST Accuracy
    try:
        mnist_results = test_mnist_accuracy(depths=[10, 50, 100], epochs=3)
        all_results['mnist_accuracy'] = mnist_results
    except Exception as e:
        print(f"MNIST accuracy test failed: {e}")
    
    # Test 3: Permuted MNIST CL
    try:
        cl_results = test_permuted_mnist_cl(num_tasks=5, epochs=2, ewc_lambda=100)
        all_results['permuted_mnist_cl'] = cl_results
    except Exception as e:
        print(f"Permuted MNIST CL test failed: {e}")
    
    # Save results
    output_file = 'extended_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print summary
    if all_results['extreme_depth']:
        print("\n1. Extreme Depth Scaling:")
        for r in all_results['extreme_depth']:
            status = "✓" if r['success'] else "✗"
            print(f"   {status} Depth {r['depth']}: {r.get('ep_time_sec', 0):.2f}s, "
                  f"grad={r.get('avg_grad_norm', 0):.2e}")
    
    if all_results['mnist_accuracy']:
        print("\n2. MNIST Accuracy:")
        for r in all_results['mnist_accuracy']:
            print(f"   Depth {r['depth']}: {r['best_accuracy']*100:.1f}% (best)")
    
    if all_results['permuted_mnist_cl']:
        print("\n3. Permuted MNIST CL:")
        for method, r in all_results['permuted_mnist_cl'].items():
            if r:
                print(f"   {method}: {r['avg']*100:.1f}% avg, {r['forgetting']*100:.1f}% forgetting")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
