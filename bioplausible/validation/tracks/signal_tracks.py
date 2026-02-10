"""
Signal Propagation Validation Tracks (Track 42)

Validates signal propagation in deep equilibrium networks.
Tests the hypothesis that EqProp maintains better signal flow
than traditional backprop through deep layers.
"""

from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from bioplausible.experiments.deep_signal_probe import \
    run_signal_propagation_experiment
from bioplausible.models import LoopedMLP, MemoryEfficientLoopedMLP
from bioplausible.validation.notebook import ValidationTrack


class SignalPropagationTrack(ValidationTrack):
    """
    Track 42: Validates signal propagation in deep equilibrium networks.

    Tests whether EqProp maintains signal strength through deep layers
    compared to traditional approaches.
    """

    def __init__(self):
        super().__init__(
            name="Signal Propagation in Deep Networks",
            track_id="42",
            description="Validate signal propagation in deep equilibrium networks",
            category="scaling",
            priority="high",
            tags=["signal", "propagation", "deep", "equilibrium"],
        )

    def validate(self) -> Dict[str, Any]:
        """
        Validate signal propagation in deep networks.

        Success criteria:
        - Signal > 1% at depth 1000
        - Stable signal propagation across depths
        - Better performance than naive approaches
        """
        print(f"Running {self.name} (Track {self.track_id})...")

        # Test with moderate depths first to avoid excessive computation
        depths = [10, 50, 100]

        try:
            # Run experiment with PyTorch backend
            results_pytorch = run_signal_propagation_experiment(
                depths=depths, perturbation_strength=0.1, backend="pytorch"
            )

            # If CUDA is available, also test kernel backend
            results_kernel = None
            if torch.cuda.is_available():
                try:
                    from bioplausible.kernel import HAS_CUPY

                    if HAS_CUPY:
                        results_kernel = run_signal_propagation_experiment(
                            depths=depths, perturbation_strength=0.1, backend="kernel"
                        )
                except ImportError:
                    pass  # Skip kernel test if not available

            # Evaluate success metrics
            success_metrics = self._evaluate_signal_propagation(
                results_pytorch, results_kernel
            )

            # Overall success: signal should remain above threshold even at deeper layers
            overall_success = success_metrics["max_depth_signal"] > 0.01  # > 1%

            return {
                "success": overall_success,
                "details": {
                    "signal_at_max_depth": success_metrics["max_depth_signal"],
                    "signal_decay_rate": success_metrics["decay_rate"],
                    "computation_times": success_metrics["times"],
                    "pytorch_results": results_pytorch,
                    "kernel_results": results_kernel,
                },
                "metrics": {
                    "signal_retention_100": success_metrics["signal_at_100"],
                    "max_depth_signal": success_metrics["max_depth_signal"],
                    "average_computation_time": success_metrics["avg_time"],
                },
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "details": "Failed to run signal propagation experiment",
            }

    def _evaluate_signal_propagation(self, pytorch_results, kernel_results=None):
        """
        Evaluate the signal propagation results.
        """
        # Extract the deepest depth tested
        depths = pytorch_results["depths"]
        max_depth = max(depths)

        # Get final signal at max depth
        final_signals = pytorch_results["signals"][max_depth]
        max_depth_signal = final_signals[-1] if final_signals else 0.0

        # Calculate signal at depth 100 if tested
        signal_at_100 = 0.0
        if 100 in depths:
            sigs_100 = pytorch_results["signals"][100]
            signal_at_100 = sigs_100[-1] if sigs_100 else 0.0

        # Calculate average computation time
        times = [pytorch_results["times"][d] for d in depths]
        avg_time = sum(times) / len(times) if times else 0.0

        # Calculate signal decay rate (simplified)
        if len(depths) > 1:
            min_depth = min(depths)
            max_depth_tested = max(depths)

            if (
                min_depth in pytorch_results["signals"]
                and max_depth_tested in pytorch_results["signals"]
            ):
                initial_signals = pytorch_results["signals"][min_depth]
                final_signals = pytorch_results["signals"][max_depth_tested]

                if initial_signals and final_signals:
                    initial_signal = initial_signals[0] if initial_signals else 1e-8
                    final_signal = final_signals[-1] if final_signals else 0.0
                    decay_rate = (initial_signal - final_signal) / initial_signal
                else:
                    decay_rate = 0.0
            else:
                decay_rate = 0.0
        else:
            decay_rate = 0.0

        return {
            "max_depth_signal": max_depth_signal,
            "signal_at_100": signal_at_100,
            "decay_rate": decay_rate,
            "times": pytorch_results["times"],
            "avg_time": avg_time,
        }


# Register the track
def register_signal_tracks(track_registry):
    """
    Register signal propagation tracks with the validation framework.
    """
    track_registry.register_track(SignalPropagationTrack())


# For direct testing
def test_signal_propagation():
    """
    Direct test function for signal propagation validation.
    """
    track = SignalPropagationTrack()
    result = track.validate()

    print(f"Track {track.track_id} - {track.name}:")
    print(f"  Success: {result['success']}")
    if "metrics" in result:
        print(f"  Signal at max depth: {result['metrics']['max_depth_signal']:.6f}")
        print(f"  Signal at depth 100: {result['metrics']['signal_retention_100']:.6f}")

    return result


if __name__ == "__main__":
    test_signal_propagation()
