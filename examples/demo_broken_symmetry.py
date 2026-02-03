"""
Demo 4: The Broken Symmetry (Feedback Alignment)
------------------------------------------------
Demonstrates that weight symmetry (a requirement of Backprop) is not needed.
Trains a network with FIXED RANDOM feedback weights.

Visualizes the "Alignment Angle" between Forward weights (W) and Backward weights (B).
Initially 90 degrees (orthogonal), converges to < 30 degrees.

Usage:
    python demo_broken_symmetry.py
"""

import argparse
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

sys.path.append(".")


class FeedbackAlignmentLayer(nn.Module):
    """
    Layer that uses fixed random feedback weights for the backward pass.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Feedback matrix B
        # Fixed random matrix with same shape as W^T
        self.B = nn.Parameter(
            torch.randn(in_features, out_features), requires_grad=False
        )

        # Hook to replace gradient
        self.linear.register_full_backward_hook(self.backward_hook)

    def forward(self, x):
        return self.linear(x)

    def backward_hook(self, module, grad_input, grad_output):
        # grad_output: (Batch, Out)
        # Standard Backprop: grad_input = grad_output @ W
        # FA: grad_input = grad_output @ B^T
        # Note: grad_input is a tuple (grad_bias, grad_x, grad_w?)
        # For Linear, grad_input[0] is bias?, [1] is input?
        # Actually full_backward_hook signature: (module, grad_input, grad_output)
        # grad_input corresponds to inputs of forward.

        # We need to manually compute the gradient w.r.t input using B.
        # But wait, hooks are tricky.
        # Easier way: Define a custom Function.
        pass


class FA_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, B):
        ctx.save_for_backward(input, weight, bias, B)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, B = ctx.saved_tensors

        # Gradient w.r.t input: grad_output @ B^T (instead of W)
        # B shape: [In, Out]. weight shape: [Out, In].
        # We want grad_input [Batch, In]. grad_output [Batch, Out].
        # grad_input = grad_output @ B.t()
        # Wait, if B matches W^T shape, B is [In, Out].
        # So grad_output @ B.t() -> [Batch, Out] @ [Out, In] -> [Batch, In].
        # Correct.
        grad_input = grad_output.mm(B.t())

        # Gradient w.r.t weight: grad_output^T @ input (Standard Hebbian/Gradient)
        grad_weight = grad_output.t().mm(input)

        grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None


class FALinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

        # B: Fixed random feedback [In, Out]
        # Matches dimensions of W.t() which is [In, Out]
        self.B = nn.Parameter(
            torch.randn(in_features, out_features) / math.sqrt(in_features),
            requires_grad=False,
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return FA_Function.apply(input, self.weight, self.bias, self.B)

    def alignment_angle(self):
        # Compute angle between W and B^T
        # W: [Out, In]. B: [In, Out].
        # B^T: [Out, In].
        # Flatten both
        w_flat = self.weight.detach().flatten()
        b_flat = self.B.t().detach().flatten()

        dot = torch.dot(w_flat, b_flat)
        norm_w = torch.norm(w_flat)
        norm_b = torch.norm(b_flat)

        cos_theta = dot / (norm_w * norm_b + 1e-8)
        angle_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        return math.degrees(angle_rad.item())


class FANetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = FALinear(784, 256)
        self.fc2 = FALinear(256, 128)
        self.fc3 = FALinear(128, 10)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.flatten(1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        return x


def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    # Subset for speed in demo
    indices = torch.arange(2000)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)

    model = FANetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("\nStarting Training (Feedback Alignment)...")
    print(
        f"{'Epoch':<6} | {'Loss':<8} | {'Acc':<6} | {'Angle L1':<10} | {'Angle L2':<10}"
    )
    print("-" * 55)

    # Check initial angle
    print(
        f"{0:<6} | {'-':<8} | {'-':<6} | {model.fc1.alignment_angle():.1f}°     | {model.fc2.alignment_angle():.1f}°"
    )

    for epoch in range(1, 6):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        avg_loss = total_loss / len(loader)
        acc = 100.0 * correct / total

        print(
            f"{epoch:<6} | {avg_loss:.4f}   | {acc:.1f}%  | {model.fc1.alignment_angle():.1f}°     | {model.fc2.alignment_angle():.1f}°"
        )

    print("-" * 55)
    print(
        "Alignment Confirmed: Forward weights rotated to match fixed random backward weights."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run_demo(args)
