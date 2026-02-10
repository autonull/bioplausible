"""
Analysis Tools for Bio-Plausible Models

Provides utilities for inspecting model dynamics, convergence, and alignment.
Useful for research and "microscope" style analysis.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DynamicsAnalyzer:
    """
    Analyzer for inspecting the internal dynamics of Equilibrium Propagation models.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def get_convergence_data(
        self, x: torch.Tensor, steps: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run the model on input x and capture convergence dynamics.

        Args:
            x: Input tensor.
            steps: Number of equilibrium steps (overrides model default if provided).

        Returns:
            Dictionary containing:
            - 'trajectory': Array of hidden states [steps, batch, hidden_dim]
            - 'deltas': Array of state changes (L2 norm) per step [steps]
            - 'activities': Mean absolute activity per step [steps]
            - 'fixed_point': Final hidden state
        """
        self.model.eval()
        x = x.to(self.device)

        # Prepare input (similar to SupervisedTrainer logic)
        if hasattr(self.model, "has_embed") and self.model.has_embed:
            # Basic handling, assuming model has .embed
            h = self.model.embed(x).mean(dim=1)
        elif x.dim() > 2 and not any(
            k in self.model.__class__.__name__ for k in ["Conv", "Transformer"]
        ):
            h = x.reshape(x.size(0), -1)
        else:
            h = x

        with torch.no_grad():
            # Check if model supports return_trajectory
            kwargs = {"return_trajectory": True, "return_dynamics": True}
            if steps is not None:
                kwargs["steps"] = steps

            # Helper to check signature or try/except
            # We'll try passing kwargs.
            try:
                # Most EqProp models (LoopedMLP, etc) support this
                output = self.model(h, **kwargs)

                # output might be (out, trajectory) or (out, dynamics_dict)
                if isinstance(output, tuple):
                    if isinstance(output[1], dict):
                        dynamics = output[1]
                    else:
                        # Assume list of tensors
                        dynamics = {"trajectory": output[1]}
                else:
                    # Some models might not return tuple even if requested if not implemented
                    raise NotImplementedError(
                        "Model does not appear to return dynamics."
                    )

            except TypeError:
                # Fallback for models that might not accept return_dynamics
                warnings.warn(
                    "Model does not support 'return_dynamics'. Attempting generic hook-based analysis."
                )
                dynamics = self._hook_based_analysis(h, steps)

        # Process trajectory to numpy
        traj_tensors = dynamics.get("trajectory", [])
        if not traj_tensors:
            return {}

        traj_np = np.stack([t.cpu().numpy() for t in traj_tensors])

        # Compute deltas (L2 diff between steps)
        deltas = []
        activities = []
        for i in range(1, len(traj_np)):
            diff = np.linalg.norm(traj_np[i] - traj_np[i - 1])
            deltas.append(diff)
            activities.append(np.mean(np.abs(traj_np[i])))

        return {
            "trajectory": traj_np,
            "deltas": np.array(deltas),
            "activities": np.array(activities),
            "fixed_point": traj_np[-1],
        }

    def _hook_based_analysis(self, h, steps):
        """Fallback: Use hooks to capture hidden states if model doesn't support explicit return."""
        # This is hard to do generically without knowing layer names.
        # For now, return empty.
        return {}

    def plot_convergence(
        self,
        x: torch.Tensor,
        steps: Optional[int] = None,
        title: str = "Convergence Dynamics",
    ):
        """
        Plot convergence metrics using Matplotlib.

        Args:
            x: Input tensor.
            steps: Number of steps.
            title: Plot title.

        Returns:
            matplotlib.figure.Figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required for plotting. Please install it.")

        data = self.get_convergence_data(x, steps)
        if not data:
            raise ValueError("Could not extract convergence data from model.")

        deltas = data["deltas"]
        activities = data["activities"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot Deltas (Convergence Speed)
        ax1.plot(deltas, marker="o", color="tab:blue")
        ax1.set_title("Equilibrium Error (State Change)")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("|| h_t - h_{t-1} ||")
        ax1.set_yscale("log")
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # Plot Activity
        ax2.plot(activities, marker="s", color="tab:orange")
        ax2.set_title("Neural Activity")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Mean |h_t|")
        ax2.grid(True)

        fig.suptitle(title)
        plt.tight_layout()

        return fig

    def compute_gradient_alignment(
        self, x: torch.Tensor, y: torch.Tensor, criterion=nn.CrossEntropyLoss()
    ) -> float:
        """
        Compute the cosine similarity between the true gradient (via Backprop)
        and the update proposed by the bio-plausible learning rule (if accessible).

        Note: This is expensive as it requires running both BP and the custom rule.
        """
        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        # 1. Compute Bio-Plausible Update
        # We need to capture the gradients produced by the model's own train_step or backward
        # This is tricky because train_step usually applies updates or stores them.
        # We will assume the model accumulates grads in .grad attributes.

        self.model.zero_grad()

        # Prepare input
        if hasattr(self.model, "has_embed") and self.model.has_embed:
            h = self.model.embed(x).mean(dim=1)
        elif x.dim() > 2 and not any(
            k in self.model.__class__.__name__ for k in ["Conv", "Transformer"]
        ):
            h = x.reshape(x.size(0), -1)
        else:
            h = x

        # Run model's custom backward mechanism
        if hasattr(self.model, "train_step"):
            # If train_step applies updates, we can't inspect grads easily unless we mock the optimizer.
            # This is too complex for a generic analyzer without deep hooks.
            # Alternative: Check if model supports `accumulate_grad=True` in train_step (unlikely).
            pass
        else:
            # Standard EqProp with .backward()
            out = self.model(h)
            loss = criterion(out, y)
            loss.backward()

        # Capture Bio Gradients
        bio_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                bio_grads[name] = param.grad.clone()

        # 2. Compute True Backprop Gradients
        # We need a reference model or we need to clear grads and run standard BP
        # But wait, self.model *is* the model. If it's EqProp, .backward() MIGHT be overloaded
        # or it might be using autograd on the energy function (which IS EqProp).
        # True BP means BPTT through the settling phase.

        # If the model is LoopedMLP, it has `gradient_method`.
        if hasattr(self.model, "gradient_method"):
            original_method = self.model.gradient_method
            try:
                self.model.gradient_method = "bptt"  # Force BPTT
                self.model.zero_grad()
                out = self.model(h)
                loss = criterion(out, y)
                loss.backward()

                bp_grads = {}
                alignment_sum = 0
                count = 0

                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in bio_grads:
                        g_bio = bio_grads[name].flatten()
                        g_bp = param.grad.flatten()

                        # Cosine similarity
                        sim = torch.nn.functional.cosine_similarity(
                            g_bio.unsqueeze(0), g_bp.unsqueeze(0)
                        ).item()
                        alignment_sum += sim
                        count += 1

                return alignment_sum / count if count > 0 else 0.0
            finally:
                # Restore method
                self.model.gradient_method = original_method

        return float("nan")
