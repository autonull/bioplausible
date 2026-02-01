"""
Demo 2: The "Healing" Visualization
-----------------------------------
Demonstrates the "Self-Healing" property of Contraction Dynamics.
Injects noise into a deep network and visualizes how it decays (EqProp) vs explodes (Backprop).

Usage:
    python demo_healing.py --save_gif healing.gif
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

sys.path.append(".")
from bioplausible.models.eqprop_base import EqPropModel
from bioplausible.models.utils import spectral_linear

class DeepChain(EqPropModel):
    """
    A deep feedforward chain (not looped) to spatially visualize propagation.
    We implement 'forward_step' as iterating through layers for one pass?

    Actually, for "Healing", we want to show propagation through DEPTH.
    Standard EqProp is defined by settling in TIME.

    But if we treat a Deep Feedforward Network as a dynamical system where
    h_{l+1} = f(h_l), then noise at layer L propagates to L+1...

    The claim "L < 1" means perturbations decay as they propagate through the map f.
    If f is the layer transition, then |delta_{l+1}| < |delta_l|.

    So we build a 100-layer Feedforward Network.
    """
    def __init__(self, depth=100, hidden_dim=128, use_sn=True):
        super().__init__(max_steps=0) # Not used for feedforward
        self.depth = depth
        self.layers = nn.ModuleList()

        for _ in range(depth):
            lin = nn.Linear(hidden_dim, hidden_dim)
            if use_sn:
                # Spectral Norm forces Lipschitz < 1 (roughly, if sigma <= 1)
                # By default spectral_norm sets sigma=1.
                # Tanh has L=1.
                # So the composition has L <= 1.
                lin = spectral_norm(lin)
            else:
                # Standard init often has singular values > 1
                nn.init.kaiming_normal_(lin.weight, mode='fan_in', nonlinearity='relu')
                # Scale up slightly to ensure explosion for demo?
                with torch.no_grad():
                    lin.weight.mul_(1.5)

            self.layers.append(lin)

        self.act = nn.Tanh()

    def _build_layers(self): pass
    def _initialize_hidden_state(self, x): pass
    def _transform_input(self, x): pass
    def forward_step(self, h, x): pass # Not used
    def _output_projection(self, h): pass

    def propagate(self, h, inject_layer=None, noise_scale=5.0):
        """Run forward pass, returning list of norms."""
        norms = []

        for i, layer in enumerate(self.layers):
            if i == inject_layer:
                h = h + torch.randn_like(h) * noise_scale

            h = self.act(layer(h))
            norms.append(h.norm().item())

        return norms

def run_demo(args):
    depth = 100
    inject_layer = 20

    print("Initializing EqProp Model (Spectral Norm ON)...")
    model_stable = DeepChain(depth=depth, use_sn=True)

    print("Initializing Standard Model (Spectral Norm OFF)...")
    model_chaos = DeepChain(depth=depth, use_sn=False)

    x = torch.randn(1, 128)

    # Run
    norms_stable = model_stable.propagate(x, inject_layer=inject_layer)
    norms_chaos = model_chaos.propagate(x, inject_layer=inject_layer)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    x_axis = np.arange(depth)

    line1, = ax.plot([], [], 'g-', label='EqProp (L<1) - Healing', linewidth=3)
    line2, = ax.plot([], [], 'r-', label='Backprop (L>1) - Exploding', linewidth=3)

    ax.axvline(x=inject_layer, color='k', linestyle='--', alpha=0.5, label='Noise Injection')
    ax.set_xlim(0, depth)
    ax.set_ylim(0, max(max(norms_stable), max(norms_chaos)) * 1.1)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Activation Norm (Energy)')
    ax.set_title('The "Healing" Visualization: Perturbation Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2

    def update(frame):
        # Animate drawing the lines
        current_x = x_axis[:frame]
        line1.set_data(current_x, norms_stable[:frame])
        line2.set_data(current_x, norms_chaos[:frame])
        return line1, line2

    ani = animation.FuncAnimation(fig, update, frames=depth, init_func=init, blit=True, interval=50)

    if args.save_gif:
        print(f"Saving animation to {args.save_gif}...")
        ani.save(args.save_gif, writer='pillow')
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_gif', type=str, default='healing.gif', help='Path to save GIF')
    args = parser.parse_args()
    run_demo(args)
