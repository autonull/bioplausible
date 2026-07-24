import os
import warnings

from omegaconf import OmegaConf

# Suppress harmless warnings for cleaner demo output
warnings.filterwarnings("ignore")

from bioplausible.config.schema import RunConfig
from bioplausible.core.trainer import run_from_runconfig as run_from_config


def load_yaml(path: str) -> RunConfig:
    conf = OmegaConf.load(path)
    schema = OmegaConf.structured(RunConfig)
    return OmegaConf.merge(schema, conf)


def run_demo():
    print("=" * 80)
    print(" Bioplausible Cross-Domain Demo: Phase 0 Validation")
    print("=" * 80)

    configs = [
        "configs/mep_mnist.yaml",
        "configs/forward_forward_mnist.yaml",
        "configs/backprop_mnist.yaml",
        "configs/eqprop_shakespeare.yaml",
        # "configs/rl_cartpole.yaml" # Requires gym, skip if not installed or minimal env
    ]

    # Check if configs exist
    for c in configs:
        if not os.path.exists(c):
            print(f"Config {c} not found. Skipping.")
            continue

        print(f"\n▶ Running {c}...")
        try:
            cfg = load_yaml(c)

            # Quick mode for demo
            cfg.trainer.epochs = 1
            cfg.trainer.batches_per_epoch = 10
            cfg.data.data_fraction = 0.1

            # If RL, adjust slightly
            if "rl" in c:
                cfg.trainer.epochs = 2

            res = run_from_config(cfg)

            final_acc = res.get("final_val_accuracy", 0.0)
            history = res.get("history", [])

            # Extract energy metrics from last epoch if available
            energy_proxy = "N/A"
            backward_flops = "N/A"
            requires_backward = "Unknown"

            if history and isinstance(history[-1], dict):
                last = history[-1]
                energy_proxy = last.get("energy_proxy", "N/A")
                backward_flops = last.get("backward_flops", "N/A")
                requires_backward = last.get("requires_backward", "N/A")

            print(f"  ✓ Success! Val Acc: {final_acc:.4f}")
            print(f"  ⚡ Energy Proxy: {energy_proxy}")
            print(f"  🔄 Backward FLOPs: {backward_flops}")
            print(f"  🔙 Requires Backward: {requires_backward}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    run_demo()
