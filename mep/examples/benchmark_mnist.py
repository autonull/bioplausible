import argparse
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add root directory to sys.path to allow imports from mep package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torchvision import datasets, transforms
from mep.optimizers import SMEPOptimizer, SDMEPOptimizer

def get_args():
    parser = argparse.ArgumentParser(description='SDMEP Benchmark')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'], help='Dataset to use')
    parser.add_argument('--model', type=str, default='MLP', choices=['MLP'], help='Model architecture')
    parser.add_argument('--optimizer', type=str, default='SDMEP', choices=['Backprop', 'SMEP', 'SDMEP', 'AdamW'], help='Optimizer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--results-file', type=str, default='results.json', help='Results file')
    return parser.parse_args()

def build_model(model_type, device):
    if model_type == "MLP":
        # 784 -> 1000 -> 10
        model = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        ).to(device)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    return model

def get_optimizer(args, model):
    if args.optimizer == "Backprop":
        # SGD
        return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    elif args.optimizer == "SMEP":
        # SMEP in EP mode
        return SMEPOptimizer(model.parameters(), lr=args.lr, mode='ep', ns_steps=4)
    elif args.optimizer == "SDMEP":
        # SDMEP in EP mode
        return SDMEPOptimizer(model.parameters(), lr=args.lr, gamma=0.95,
                                   rank_frac=0.1, error_beta=0.9, dion_thresh=500000, mode='ep')
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

def train(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    if args.dataset == 'MNIST':
        train_set = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'FashionMNIST':
        train_set = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)

    # Subset for speed
    train_subset_indices = torch.randperm(len(train_set))[:10000]
    train_set = torch.utils.data.Subset(train_set, train_subset_indices)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, device)
    optimizer = get_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "time": [],
        "l0_sigma": []
    }

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        correct = 0
        total = 0
        ep_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Flatten for MLP
            if args.model == "MLP":
                x = x.view(x.shape[0], -1)

            optimizer.zero_grad()

            if args.optimizer in ["Backprop", "AdamW"]:
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

                correct += (pred.argmax(1) == y).sum().item()
                ep_loss += loss.item()
                total += y.size(0)

            else:
                # EP Modes
                # Note: SMEP/SDMEP step handles EP gradient computation internally
                optimizer.step(x=x, target=y, model=model)

                # Inference for stats (extra forward pass)
                with torch.no_grad():
                    pred = model(x)
                    correct += (pred.argmax(1) == y).sum().item()
                    # Calculate MSE loss for EP monitoring
                    y_oh = F.one_hot(y, 10).float()
                    ep_loss += F.mse_loss(pred, y_oh).item()
                    total += y.size(0)

        train_acc = 100 * correct / total
        train_loss = ep_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        test_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if args.model == "MLP":
                    x = x.view(x.shape[0], -1)

                pred = model(x)
                test_correct += (pred.argmax(1) == y).sum().item()
                if args.optimizer in ["Backprop", "AdamW"]:
                    test_loss += criterion(pred, y).item()
                else:
                    y_oh = F.one_hot(y, 10).float()
                    test_loss += F.mse_loss(pred, y_oh).item()
                test_total += y.size(0)

        test_acc = 100 * test_correct / test_total
        test_loss = test_loss / len(test_loader)

        # Spectral Norm Check (Layer 0)
        # model is nn.Sequential. Layer 0 is Linear.
        l0_w = model[0].weight
        if l0_w.ndim > 2:
             w_mat = l0_w.view(l0_w.shape[0], -1)
        else:
             w_mat = l0_w

        # Power iteration to estimate sigma
        u = torch.randn(w_mat.shape[0], device=device)
        v = torch.randn(w_mat.shape[1], device=device)
        for _ in range(5):
            v = F.normalize(torch.mv(w_mat.t(), u), dim=0, eps=1e-8)
            u = F.normalize(torch.mv(w_mat, v), dim=0, eps=1e-8)
        sigma = torch.dot(u, torch.mv(w_mat, v)).item()

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | L0 Sigma: {sigma:.4f} | Time: {epoch_time:.2f}s")

        metrics["epoch"].append(epoch+1)
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["test_loss"].append(test_loss)
        metrics["test_acc"].append(test_acc)
        metrics["time"].append(epoch_time)
        metrics["l0_sigma"].append(sigma)

    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.2f}s")

    # Save results
    results = {
        "config": vars(args),
        "metrics": metrics,
        "total_time": total_time
    }

    try:
        with open(args.results_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(results)

    with open(args.results_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    args = get_args()
    train(args)
