import json
import os

from bioplausible.config.schema import (
    RunConfig,
    RunConfigData,
    RunConfigModel,
    RunConfigOptimizer,
    RunConfigTrainer,
)
from bioplausible.core.trainer import run_from_runconfig as run_from_config


def main():
    algo = "equitile"
    tasks = ["mnist", "tiny_shakespeare", "cartpole"]
    results = []
    output_dir = "results/signals"
    os.makedirs(output_dir, exist_ok=True)
    for task in tasks:
        print(f"Signal 5: Fast test {algo} task={task}")
        cfg = RunConfig(
            seed=42,
            device="auto",
            output_dir=f"{output_dir}/{algo}_{task}_signal_5",
            data=RunConfigData(task=task, batch_size=32),
            model=RunConfigModel(name=algo, hidden_dim=64, num_layers=2),
            optimizer=RunConfigOptimizer(name="adam", lr=0.001),
            trainer=RunConfigTrainer(epochs=1, batches_per_epoch=2, track_energy=True),
        )
        if task == "cartpole":
            cfg.data.batch_size = 1
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
    with open(f"{output_dir}/signal_5.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
