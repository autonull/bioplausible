import json
import os
import traceback

import numpy as np

from bioplausible.config_schema import (
    RunConfig,
    RunConfigData,
    RunConfigModel,
    RunConfigOptimizer,
    RunConfigTrainer,
)
from bioplausible.runner import run_from_config


def main():
    algorithms = ["backprop", "forward_forward", "pepita", "diff_target_prop"]
    tasks = ["mnist", "cifar10"]
    results = []
    output_dir = "results/signals"
    os.makedirs(output_dir, exist_ok=True)
    for algo in algorithms:
        for task in tasks:
            print(f"Signal 1: Fast test {algo} on {task}")
            cfg = RunConfig(
                seed=42,
                device="auto",
                output_dir=f"{output_dir}/{algo}_{task}_signal_1",
                data=RunConfigData(task=task, batch_size=32),
                model=RunConfigModel(name=algo, hidden_dim=64),
                optimizer=RunConfigOptimizer(name="adam", lr=0.001),
                trainer=RunConfigTrainer(epochs=1, batches_per_epoch=2),
            )
            if algo in ["backprop_mlp", "looped_mlp", "memory_efficient_mlp"]:
                delattr(cfg.model, "num_layers")
            try:
                res = run_from_config(cfg)
                score = float(res.get("final_val_accuracy", 0.0))
                results.append(
                    {"model": algo, "task": task, "success": True, "final_score": score}
                )
            except Exception as e:
                results.append(
                    {"model": algo, "task": task, "success": False, "error": str(e)}
                )
    with open(f"{output_dir}/signal_1.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
