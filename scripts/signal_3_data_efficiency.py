import json
import pathlib

from bioplausible.config.schema import (
    RunConfig,
    RunConfigData,
    RunConfigModel,
    RunConfigOptimizer,
    RunConfigTrainer,
)
from bioplausible.core.trainer import run_from_runconfig as run_from_config


def main():
    algorithms = ["backprop", "three_factor_hebbian", "pepita"]
    task = "mnist"
    data_fractions = [0.01, 1.0]
    results = []
    output_dir = "results/signals"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)
    for algo in algorithms:
        for frac in data_fractions:
            print(f"Signal 3: Fast test {algo} frac={frac}")
            cfg = RunConfig(
                seed=42,
                device="auto",
                output_dir=f"{output_dir}/{algo}_{task}_frac{frac}_signal_3",
                data=RunConfigData(task=task, batch_size=32, data_fraction=frac),
                model=RunConfigModel(name=algo, hidden_dim=64),
                optimizer=RunConfigOptimizer(name="adam", lr=0.001),
                trainer=RunConfigTrainer(
                    epochs=1, batches_per_epoch=2, track_energy=True
                ),
            )
            if algo in ["backprop_mlp", "looped_mlp", "memory_efficient_mlp"]:
                delattr(cfg.model, "num_layers")
            try:
                res = run_from_config(cfg)
                score = float(res.get("final_val_accuracy", 0.0))
                results.append({
                    "model": algo,
                    "task": task,
                    "data_fraction": frac,
                    "success": True,
                    "final_score": score,
                })
            except Exception as e:
                results.append({
                    "model": algo,
                    "task": task,
                    "data_fraction": frac,
                    "success": False,
                    "error": str(e),
                })
    with pathlib.Path(f"{output_dir}/signal_3.json").open("w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
