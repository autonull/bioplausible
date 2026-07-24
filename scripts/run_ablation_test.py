import os

from bioplausible.analysis.ablation import AblationStudy
from bioplausible.config.schema import (
    RunConfig,
    RunConfigData,
    RunConfigModel,
    RunConfigOptimizer,
    RunConfigTrainer,
)


def main():
    base_cfg = RunConfig(
        seed=42,
        device="cpu",  # Keep it simple for testing
        output_dir="results/ablation_test",
        data=RunConfigData(task="mnist", batch_size=64, data_fraction=0.01),
        model=RunConfigModel(name="looped_mlp", hidden_dim=64),
        optimizer=RunConfigOptimizer(name="adam", lr=0.001),
        trainer=RunConfigTrainer(epochs=1, batches_per_epoch=2),
    )

    dimensions = {"learning_rate": [0.001, 0.01], "hidden_dim": [32, 64]}

    print("Starting small AblationStudy...")
    study = AblationStudy(base_cfg, dimensions)

    df = study.run(parallel_workers=2)

    print("\nResults:")
    print(df.to_string())

    print("\nCritical Hyperparams:")
    print(study.identify_critical_hyperparams())

    os.makedirs("results/ablation_test", exist_ok=True)

    try:
        fig = study.plot_sensitivity_heatmap("learning_rate", "hidden_dim")
        fig.savefig("results/ablation_test/heatmap.png")
        print("\nSaved heatmap.png")
    except Exception as e:
        print(f"Failed to save heatmap: {e}")


if __name__ == "__main__":
    main()
