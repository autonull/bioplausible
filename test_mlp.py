from bioplausible.config_schema import RunConfig, RunConfigData, RunConfigModel, RunConfigOptimizer, RunConfigTrainer
from bioplausible.runner import run_from_config
import traceback

for algo in ["backprop", "eqprop_mlp"]:
    print(f"\n==================================================")
    print(f"Testing {algo} on mnist")
    print(f"==================================================")
    cfg = RunConfig(
        seed=42,
        device="auto",
        output_dir=f"results/matrix/{algo}_mnist",
        data=RunConfigData(task="mnist", batch_size=32),
        model=RunConfigModel(name=algo, hidden_dim=64, num_layers=2),
        optimizer=RunConfigOptimizer(name="adam", lr=0.001),
        trainer=RunConfigTrainer(epochs=1, batches_per_epoch=2, track_energy=True)
    )
    try:
        run_res = run_from_config(cfg)
        print(f"SUCCESS: {algo}")
    except Exception as e:
        print(f"FAILED {algo}")
        traceback.print_exc()
