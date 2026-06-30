import os

from bioplausible.leaderboard.generator import (LeaderboardEntry,
                                                LeaderboardGenerator)


def main():
    print("Testing Phase 1.7: Leaderboard Generation...")

    # Mock data with biologically plausible models comparing against Backprop
    entries = [
        LeaderboardEntry(
            algorithm="backprop_mlp",
            optimizer="adam",
            task="mnist",
            val_accuracy=0.985,
            energy_proxy=1100.5,
            backward_flops=8500000,
            requires_backward=True,
            param_count=100000,
            wall_time_s=15.2,
            peak_memory_mb=120.5,
            mean_acc=0.983,
            std_acc=0.002,
            config_hash="h1",
        ),
        LeaderboardEntry(
            algorithm="eqprop_mlp",
            optimizer="ep",
            task="mnist",
            val_accuracy=0.978,
            energy_proxy=250.0,
            backward_flops=0,
            requires_backward=False,
            param_count=100000,
            wall_time_s=38.4,
            peak_memory_mb=85.0,
            mean_acc=0.975,
            std_acc=0.004,
            config_hash="h2",
        ),
        LeaderboardEntry(
            algorithm="pepita",
            optimizer="forward",
            task="mnist",
            val_accuracy=0.965,
            energy_proxy=150.2,
            backward_flops=0,
            requires_backward=False,
            param_count=100000,
            wall_time_s=12.1,
            peak_memory_mb=65.0,
            mean_acc=0.960,
            std_acc=0.005,
            config_hash="h3",
        ),
        LeaderboardEntry(
            algorithm="backprop_mlp",
            optimizer="adam",
            task="cifar10",
            val_accuracy=0.885,
            energy_proxy=5000.5,
            backward_flops=45500000,
            requires_backward=True,
            param_count=500000,
            wall_time_s=115.2,
            peak_memory_mb=420.5,
            mean_acc=0.880,
            std_acc=0.008,
            config_hash="h4",
        ),
        LeaderboardEntry(
            algorithm="spiking_stdp",
            optimizer="hebbian",
            task="cifar10",
            val_accuracy=0.720,
            energy_proxy=45.5,
            backward_flops=0,
            requires_backward=False,
            param_count=500000,
            wall_time_s=250.0,
            peak_memory_mb=55.0,
            mean_acc=0.700,
            std_acc=0.02,
            config_hash="h5",
        ),
    ]

    generator = LeaderboardGenerator(entries)

    out_path = "reports/leaderboard.md"
    generator.export_markdown(out_path)

    if os.path.exists(out_path):
        print(f"Successfully generated leaderboard to {out_path}!")
        with open(out_path, "r") as f:
            print("\nPreview:")
            lines = f.readlines()
            for line in lines[:25]:
                print(line.strip())
    else:
        print("Failed to generate leaderboard.")


if __name__ == "__main__":
    main()
