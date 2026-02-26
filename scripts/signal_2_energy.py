import json, os, traceback, numpy as np
from bioplausible.config_schema import RunConfig, RunConfigData, RunConfigModel, RunConfigOptimizer, RunConfigTrainer
from bioplausible.runner import run_from_config

def main():
    algorithms = ["backprop", "eqprop_mlp", "memory_efficient_mlp"]
    task = "mnist"
    lrs = [0.01, 0.001]
    epochs_list = [1]
    results = []
    output_dir = "results/signals"
    os.makedirs(output_dir, exist_ok=True)
    for algo in algorithms:
        for lr in lrs:
            for ep in epochs_list:
                print(f"Signal 2: Fast test {algo} on {task} lr={lr}")
                cfg = RunConfig(
                    seed=42, device="auto", output_dir=f"{output_dir}/{algo}_{task}_lr{lr}_ep{ep}_signal_2",
                    data=RunConfigData(task=task, batch_size=32),
                    model=RunConfigModel(name=algo, hidden_dim=64),
                    optimizer=RunConfigOptimizer(name="adam", lr=lr),
                    trainer=RunConfigTrainer(epochs=ep, batches_per_epoch=2, track_energy=True)
                )
                if algo in ["eqprop_mlp", "memory_efficient_mlp"]:
                    cfg.optimizer.mode = "ep"
                    cfg.optimizer.settle_steps = 10
                    cfg.optimizer.beta = 0.5
                if algo in ["backprop_mlp", "looped_mlp", "memory_efficient_mlp"]:
                    delattr(cfg.model, "num_layers")
                try:
                    res = run_from_config(cfg)
                    score = float(res.get("final_val_accuracy", 0.0))
                    energy = float(res.get("total_energy_proxy", 0.0))
                    results.append({"model": algo, "lr": lr, "epochs": ep, "success": True, "final_score": score, "energy": energy})
                except Exception as e:
                    results.append({"model": algo, "lr": lr, "epochs": ep, "success": False, "error": str(e)})
    with open(f"{output_dir}/signal_2.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
