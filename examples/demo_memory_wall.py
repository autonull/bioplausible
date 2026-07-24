"""
Demo 1: The "Laptop GPU" Memory Wall
------------------------------------
Demonstrates EqProp's O(1) memory scaling vs Backprop's O(N) scaling.
Trains a very deep network (simulated via recurrent steps) to demonstrate memory usage differences.

Usage:
    python demo_memory_wall.py --layers 10000 --method eqprop
    python demo_memory_wall.py --layers 10000 --method backprop
"""

import argparse
import sys

import torch
from torch import nn, optim

# Add root to path for imports
sys.path.append(".")

from bioplausible.zoo.models.base import EqPropModel
from bioplausible.zoo.utils import spectral_linear


class DeepBackpropNet(nn.Module):
    """Standard ResNet MLP trained with Backprop."""

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, layers=100):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Deep residual chain
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(layers)
        ])

        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()

    def forward(self, x):
        h = self.act(self.input_layer(x))
        for layer in self.layers:
            # Residual connection
            h = h + self.act(layer(h))
            # Standard backprop must store 'h' for every layer
        return self.output_layer(h)


class DeepEqPropNet(EqPropModel):
    """
    Infinite-Depth EqProp Network.

    This implementation uses a 'Looped' architecture (RNN) where the 'layers' argument
    controls the number of steps. This allows simulating extremely deep networks
    (e.g., 10,000 steps) to demonstrate the memory advantages of Equilibrium Propagation,
    which only requires storing the final equilibrium state (O(1) memory) compared to
    Backprop Through Time which stores the history of all states (O(T) memory).
    """

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, eq_steps=100):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            max_steps=eq_steps,
            gradient_method="equilibrium",  # Implicit differentiation for O(1)
        )

        self.embed = spectral_linear(input_dim, hidden_dim)
        self.W_rec = spectral_linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.act = nn.Tanh()

    def _build_layers(self):
        pass

    def _initialize_hidden_state(self, x):
        return torch.zeros(x.shape[0], self.hidden_dim, device=x.device)

    def _transform_input(self, x):
        return self.embed(x)

    def _forward_step_impl(self, h, x_transformed):
        # h_{t+1} = tanh(W h_t + x)
        return self.act(self.W_rec(h) + x_transformed)

    def forward_step(self, h, x_transformed):
        return self._forward_step_impl(h, x_transformed)

    def _output_projection(self, h):
        return self.fc(h)


class RNNBackprop(nn.Module):
    """Standard RNN trained with BPTT."""

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, steps=100):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.act = nn.Tanh()
        self.steps = steps

    def forward(self, x):
        h = torch.zeros(x.shape[0], self.W_rec.out_features, device=x.device)
        input_drive = self.embed(x)

        # BPTT requires storing graph for all these steps
        for _ in range(self.steps):
            h = self.act(self.W_rec(h) + input_drive)

        return self.fc(h)


def measure_memory(model_class, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: Running on CPU. Memory measurement (CUDA) not available.")
        # Proceed anyway to check runnable

    print(f"Creating model: {args.method} with {args.layers} 'layers' (steps)...")

    if args.method == "eqprop":
        # EqProp uses implicit diff (gradient_method='equilibrium')
        model = DeepEqPropNet(hidden_dim=args.hidden_dim, eq_steps=args.layers).to(
            device
        )
    else:
        # Backprop uses BPTT
        model = RNNBackprop(hidden_dim=args.hidden_dim, steps=args.layers).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy Data
    batch_size = 64  # Keep modest
    x = torch.randn(batch_size, 784).to(device)
    y = torch.randint(0, 10, (batch_size,)).to(device)

    print("Starting training step...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()
    else:
        print("Note: Running on CPU, memory stats are approximated or skipped.")
        start_mem = 0

    try:
        optimizer.zero_grad()
        output = model(x)

        # Force backward
        if isinstance(output, tuple):
            output = output[0]  # EqProp might return tuple
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            print("Success!")
            print(f"Peak Memory: {(peak_mem - start_mem) / 1024**2:.2f} MB")
        else:
            print("Success! (Memory tracking requires CUDA)")

    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n❌ CUDA OUT OF MEMORY ERROR ❌")
            print(f"Backprop failed at depth {args.layers}. The Memory Wall was hit.")
        else:
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=int, default=1000, help="Number of layers/steps"
    )
    parser.add_argument(
        "--method", type=str, choices=["eqprop", "backprop"], required=True
    )
    parser.add_argument("--hidden_dim", type=int, default=1024)
    args = parser.parse_args()

    measure_memory(RNNBackprop if args.method == "backprop" else DeepEqPropNet, args)
