"""
Example: Training a CNN on MNIST/FashionMNIST using SDMEP and LocalEP.

Demonstrates support for convolutional layers, pooling, and normalization
within the Equilibrium Propagation pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time

from mep.presets import sdmep, local_ep

def get_model(num_classes: int = 10) -> nn.Module:
    """Create a standard LeNet-like CNN."""
    return nn.Sequential(
        # Conv 1: 1 -> 32
        nn.Conv2d(1, 32, kernel_size=3, padding=1), # 28x28
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2), # 14x14

        # Conv 2: 32 -> 64
        nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2), # 7x7

        # Flatten and Dense
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes)
    )

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if args.dataset == 'mnist':
        train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('./data', train=False, transform=transform)
    else:
        train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model().to(device)

    # Optimizer
    if args.optimizer == 'sdmep':
        print("Using SDMEP (Global EP + Dion/Muon)")
        optimizer = sdmep(
            model.parameters(),
            model=model,
            lr=args.lr,
            beta=args.beta,
            settle_steps=args.settle_steps,
            settle_lr=args.settle_lr,
            ns_steps=args.ns_steps,
            dion_thresh=args.dion_thresh,
        )
    elif args.optimizer == 'local_ep':
        print("Using LocalEP (Layer-local EP + Muon)")
        optimizer = local_ep(
            model.parameters(),
            model=model,
            lr=args.lr,
            beta=args.beta,
            settle_steps=args.settle_steps,
            settle_lr=args.settle_lr,
            ns_steps=args.ns_steps,
        )
    elif args.optimizer == 'sgd':
        print("Using SGD (Backprop)")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            if args.optimizer in ['sdmep', 'local_ep']:
                # EP Step
                optimizer.step(x=data, target=target)

                # Forward for metrics (EP step doesn't return loss)
                with torch.no_grad():
                    output = model(data)
                    loss = F.cross_entropy(output, target)
            else:
                # BP Step
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

            if batch_idx % args.log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
                if args.dry_run:
                    break

        epoch_time = time.time() - start_time
        avg_loss = total_loss / total
        accuracy = 100. * correct / total

        print(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")

        # Evaluate
        test(model, device, test_loader)

        if args.dry_run:
            break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MEP CNN Example')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs') # Default 1 for quick demo
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='EP nudging strength')
    parser.add_argument('--settle-steps', type=int, default=15, help='EP settling steps')
    parser.add_argument('--settle-lr', type=float, default=0.05, help='EP settling LR')
    parser.add_argument('--ns-steps', type=int, default=5, help='Newton-Schulz steps')
    parser.add_argument('--dion-thresh', type=int, default=100000, help='Dion threshold')
    parser.add_argument('--optimizer', type=str, default='sdmep', choices=['sdmep', 'local_ep', 'sgd'])
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion'])
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--log-interval', type=int, default=100, help='log interval')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single batch')

    args = parser.parse_args()
    train(args)
