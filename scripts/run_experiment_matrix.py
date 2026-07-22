import json
import os
import traceback

from bioplausible.config_schema import (
    RunConfig,
    RunConfigData,
    RunConfigModel,
    RunConfigOptimizer,
    RunConfigTrainer,
)
from bioplausible.models.registry import get_model_spec
from bioplausible.runner import run_from_config


def main():
    algorithms = [
        "backprop",
        "eqprop_mlp",
        "forward_forward",
        "pepita",
        "diff_target_prop",
        "three_factor_hebbian",
    ]

    tasks = ["mnist", "cifar10", "tiny_shakespeare", "breast_cancer", "cora"]

    results = []
    output_dir_base = "results/matrix"
    os.makedirs(output_dir_base, exist_ok=True)

    for algo in algorithms:
        spec = get_model_spec(algo)
        for task in tasks:
            print("\n==================================================")
            print(f"Testing {algo} on {task}")
            print("==================================================")

            domain = "vision"
            if task == "tiny_shakespeare":
                domain = "lm"
            elif task == "cartpole":
                domain = "rl"
            elif task == "breast_cancer":
                domain = "tabular"
            elif task == "cora":
                domain = "graph"

            # Simple compatibility check
            is_compat = False
            compat_list = spec.task_compat or []
            if spec.model_type == "graph_eqprop" and domain == "graph":
                is_compat = True
            elif domain in compat_list or task in compat_list:
                is_compat = True
            elif algo == "modern_conv_eqprop" and task == "cifar10":
                is_compat = True
            elif (
                algo
                in [
                    "forward_forward",
                    "pepita",
                    "diff_target_prop",
                    "three_factor_hebbian",
                ]
                and domain == "vision"
            ):
                is_compat = True
            elif algo == "forward_forward" and domain == "tabular":
                is_compat = True

            if not is_compat:
                print(f"Skipping {algo} on {task} (incompatible)")
                continue

            cfg = RunConfig(
                seed=42,
                device="auto",
                output_dir=f"{output_dir_base}/{algo}_{task}",
                data=RunConfigData(task=task, batch_size=32),
                model=RunConfigModel(name=algo, hidden_dim=64, num_layers=2),
                optimizer=RunConfigOptimizer(name="adam", lr=0.001),
                trainer=RunConfigTrainer(
                    epochs=1, batches_per_epoch=20, track_energy=True
                ),
            )

            if algo in ["backprop", "eqprop_mlp"]:
                # The generic RunConfigModel includes num_layers, which these two don't accept
                # so we can't remove it directly if we do RunConfigModel(...)
                # But runner.py passes cfg.model.num_layers via `num_layers=...`
                # Let's just create an extra flag to ignore it or we need to modify runner.py.
                # Actually easiest is to rely on kwargs ignoring if possible, or modify runner.py
                pass

            if domain == "rl":
                cfg.data.batch_size = 1
                cfg.trainer.batches_per_epoch = 10

            try:
                run_res = run_from_config(cfg)
                res_dict = {
                    "model": algo,
                    "task": task,
                    "success": True,
                    "final_score": run_res.get("final_val_accuracy", 0.0),
                }
                print(f"SUCCESS: {res_dict}")
                results.append(res_dict)
            except Exception as e:
                print(f"FAILED {algo} on {task}")
                traceback.print_exc()
                results.append(
                    {"model": algo, "task": task, "success": False, "error": str(e)}
                )

    with open(f"{output_dir_base}/summary.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nExperiment Matrix Complete. Summary saved to results/matrix/summary.json")


if __name__ == "__main__":
    main()
