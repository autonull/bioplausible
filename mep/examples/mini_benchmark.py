import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import time
import sys

from mep.optimizers import SMEPOptimizer, SDMEPOptimizer, LocalEPMuon, NaturalEPMuon

# Configuration
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.05
BETA = 0.5
SETTLE_STEPS = 15
NS_STEPS = 5
Subset_Size = 2000  # Train on a small subset for speed

def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Download to local folder
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Subset for quick benchmark
    indices = torch.randperm(len(train_dataset))[:Subset_Size]
    subset = torch.utils.data.Subset(train_dataset, indices)

    loader = torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def create_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def train(optimizer_class, name, **kwargs):
    print(f"\n--- Benchmarking {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create model
    model = create_model().to(device)

    # Instantiate optimizer
    # Note: For EP modes, we pass model=model
    if 'mode' not in kwargs: kwargs['mode'] = 'ep'

    optimizer = optimizer_class(
        model.parameters(),
        model=model,
        lr=LR,
        beta=BETA,
        settle_steps=SETTLE_STEPS,
        ns_steps=NS_STEPS,
        **kwargs
    )

    loader = get_dataloader()

    model.train()
    start_time = time.time()

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # EP Training Step
            output = model(data)
            optimizer.step(target=target)

            # Compute loss/acc for reporting (using the free phase output)
            with torch.no_grad():
                 pred = output.argmax(dim=1, keepdim=True)
                 correct += pred.eq(target.view_as(pred)).sum().item()
                 total += target.size(0)
                 loss = F.cross_entropy(output, target)
                 total_loss += loss.item()

            if batch_idx % 10 == 0:
                sys.stdout.write(f"\rEpoch {epoch+1}/{EPOCHS} Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
                sys.stdout.flush()

    end_time = time.time()
    duration = end_time - start_time
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    print(f"\nResult: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, Time={duration:.2f}s")
    return avg_loss, accuracy, duration

if __name__ == "__main__":
    results = {}

    # 1. Standard SMEP (EP Mode)
    loss, acc, t = train(SMEPOptimizer, "SMEP (Standard EP)", mode='ep')
    results['SMEP'] = (loss, acc, t)

    # 2. SDMEP (Dion-Muon)
    loss, acc, t = train(SDMEPOptimizer, "SDMEP (Dion-Muon)", mode='ep', dion_thresh=1000)
    results['SDMEP'] = (loss, acc, t)

    # 3. LocalEPMuon
    loss, acc, t = train(LocalEPMuon, "LocalEPMuon (Bio-Plausible)", mode='ep')
    results['LocalEPMuon'] = (loss, acc, t)

    # 4. NaturalEPMuon
    loss, acc, t = train(NaturalEPMuon, "NaturalEPMuon (Fisher)", mode='ep', fisher_approx='empirical')
    results['NaturalEPMuon'] = (loss, acc, t)

    print("\n\n=== Final Comparison ===")
    print(f"{'Optimizer':<25} | {'Loss':<8} | {'Acc (%)':<8} | {'Time (s)':<8}")
    print("-" * 60)
    for name, (loss, acc, t) in results.items():
        print(f"{name:<25} | {loss:<8.4f} | {acc:<8.2f} | {t:<8.2f}")
