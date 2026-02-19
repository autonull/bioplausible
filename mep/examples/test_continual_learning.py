#!/usr/bin/env python3
"""
Quick Continual Learning Test

Fast test of EP+EWC on a simplified continual learning scenario.
Uses synthetic data instead of MNIST for speed.

Run: python examples/test_continual_learning.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time

from mep import EPOptimizer


class MLP(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=64, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


def generate_task_data(task_id, num_samples=200, input_dim=50, num_classes=5, seed=None):
    """Generate synthetic task data with task-specific feature importance."""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate features
    X = torch.randn(num_samples, input_dim)
    
    # Task-specific: different features matter for different tasks
    # This creates genuine continual learning challenge
    feature_start = (task_id * 10) % input_dim
    feature_mask = torch.zeros(input_dim)
    feature_mask[feature_start:feature_start+10] = 1.0
    
    # Labels depend on task-relevant features
    relevant_features = X[:, feature_start:feature_start+10]
    label_scores = relevant_features.sum(dim=1)
    y = (label_scores - label_scores.mean()) / label_scores.std()
    y = ((y + 3) / 1.2).long().clamp(0, num_classes-1)
    
    return X, y


def compute_accuracy(model, X, y, device):
    model.eval()
    with torch.no_grad():
        output = model(X.to(device))
        pred = output.argmax(dim=1)
        return (pred == y.to(device)).float().mean().item()


def run_cl_benchmark(num_tasks=3, epochs=2, batch_size=32, ewc_lambda=100):
    """Run quick continual learning benchmark."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    input_dim = 50
    num_classes = 5
    
    results = {
        'ep_ewc': {'accuracies': [], 'forgetting': 0},
        'ep_no_ewc': {'accuracies': [], 'forgetting': 0},
        'bp_ewc': {'accuracies': [], 'forgetting': 0},
        'bp_no_ewc': {'accuracies': [], 'forgetting': 0},
    }
    
    methods = ['ep_ewc', 'ep_no_ewc', 'bp_ewc', 'bp_no_ewc']
    
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        torch.manual_seed(42)
        model = MLP(input_dim=input_dim, hidden_dim=64, output_dim=num_classes).to(device)
        
        use_ep = 'ep' in method
        use_ewc = 'ewc' in method
        
        # Track accuracy on all tasks after each task
        task_accuracies = []  # List of lists
        
        for task_id in range(num_tasks):
            # Generate task data
            X_train, y_train = generate_task_data(task_id, num_samples=200, seed=42+task_id)
            X_test, y_test = generate_task_data(task_id, num_samples=50, seed=142+task_id)
            
            train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
            
            # Create optimizer
            if use_ep:
                opt = EPOptimizer(
                    model.parameters(),
                    model=model,
                    mode='ep',
                    lr=0.01,
                    ewc_lambda=ewc_lambda if use_ewc else 0,
                    settle_steps=10,
                    settle_lr=0.2,
                )
            else:
                opt = EPOptimizer(
                    model.parameters(),
                    model=model,  # Need model for EWC
                    mode='backprop',
                    lr=0.01,
                    ewc_lambda=ewc_lambda if use_ewc else 0,
                )
                criterion = nn.CrossEntropyLoss()
            
            # Train on task
            start = time.time()
            for epoch in range(epochs):
                for X, y in train_loader:
                    X, y = X.to(device), y.to(device)
                    
                    if use_ep:
                        opt.step(x=X, target=y, task_id=task_id)
                    else:
                        opt.zero_grad()
                        loss = criterion(model(X), y)
                        if use_ewc and hasattr(opt, 'ewc_state') and opt.ewc_state is not None:
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
                X_prev, y_prev = generate_task_data(prev_task, num_samples=50, seed=142+prev_task)
                acc = compute_accuracy(model, X_prev, y_prev, device)
                current_accs.append(acc)
            
            task_accuracies.append(current_accs)
            
            print(f"  Task {task_id}: Train acc = {current_accs[-1]*100:.1f}%, "
                  f"Time = {train_time:.1f}s")
        
        # Final evaluation on all tasks
        final_accs = []
        for task_id in range(num_tasks):
            X_test, y_test = generate_task_data(task_id, num_samples=50, seed=142+task_id)
            acc = compute_accuracy(model, X_test, y_test, device)
            final_accs.append(acc)
        
        # Compute forgetting (max accuracy drop)
        forgetting = 0.0
        for task_id in range(num_tasks):
            acc_after = task_accuracies[task_id][task_id]
            acc_final = final_accs[task_id]
            drop = acc_after - acc_final
            forgetting = max(forgetting, drop)
        
        avg_acc = sum(final_accs) / len(final_accs)
        
        results[method]['accuracies'] = final_accs
        results[method]['average'] = avg_acc
        results[method]['forgetting'] = forgetting
        
        print(f"\n  Final Results:")
        print(f"    Task accuracies: {[f'{a*100:.1f}%' for a in final_accs]}")
        print(f"    Average: {avg_acc*100:.1f}%")
        print(f"    Forgetting: {forgetting*100:.1f}%")
    
    return results


def print_summary(results):
    """Print benchmark summary."""
    print("\n" + "="*80)
    print("CONTINUAL LEARNING BENCHMARK SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<15} {'Avg Accuracy':<15} {'Forgetting':<15}")
    print("-"*50)
    
    for method in ['ep_ewc', 'ep_no_ewc', 'bp_ewc', 'bp_no_ewc']:
        r = results[method]
        print(f"{method:<15} {r['average']*100:<14.1f}% {r['forgetting']*100:<14.1f}%")
    
    print("-"*50)
    
    # Compare EP+EWC vs EP without EWC
    ep_improvement = results['ep_ewc']['average'] - results['ep_no_ewc']['average']
    ep_forgetting_reduction = results['ep_no_ewc']['forgetting'] - results['ep_ewc']['forgetting']
    
    print(f"\nEP + EWC vs EP without EWC:")
    print(f"  Accuracy improvement: {ep_improvement*100:+.1f}%")
    print(f"  Forgetting reduction: {ep_forgetting_reduction*100:+.1f}%")
    
    # Compare to backprop
    bp_diff = results['ep_ewc']['average'] - results['bp_ewc']['average']
    bp_forget_diff = results['ep_ewc']['forgetting'] - results['bp_ewc']['forgetting']
    
    print(f"\nEP + EWC vs Backprop + EWC:")
    print(f"  Accuracy difference: {bp_diff*100:+.1f}%")
    print(f"  Forgetting difference: {bp_forget_diff*100:+.1f}%")
    
    print("="*80)


def main():
    print("="*80)
    print("PHASE 2: QUICK CONTINUAL LEARNING TEST")
    print("EP + EWC on Synthetic Task Sequence")
    print("="*80)
    
    results = run_cl_benchmark(
        num_tasks=3,
        epochs=2,
        batch_size=32,
        ewc_lambda=100,
    )
    
    print_summary(results)
    
    print("\n✅ Quick CL test complete!")
    print("\nNote: This is a fast synthetic benchmark.")
    print("For full Permuted MNIST results, run:")
    print("  python examples/benchmark_permuted_mnist.py")


if __name__ == "__main__":
    main()
