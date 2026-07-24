import hashlib
import json
import os
import pathlib

from bioplausible.leaderboard.generator import LeaderboardEntry, LeaderboardGenerator


def generate_leaderboard(results_dir="results/phase1_reduced"):
    if not pathlib.Path(results_dir).exists():
        print(f"Error: {results_dir} not found.")
        return

    runs_file = os.path.join(results_dir, "runs.jsonl")
    if not pathlib.Path(runs_file).exists():
        print(f"Error: {runs_file} not found.")
        return

    entries = []
    with pathlib.Path(runs_file).open("r") as f:
        content = f.read()
        lines = content.split("\n")
        if len(lines) == 1 and "\\n" in lines[0]:
            lines = lines[0].split("\\n")

        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    record = json.loads(line)
                    tags = record.get("tags", {})

                    alg_name = record.get("model", "unknown")
                    opt_name = tags.get("optimizer", "unknown")

                    # Create a deterministic config hash for this setting
                    config_str = f"{alg_name}_{opt_name}_{tags.get('hidden_dim', 0)}_{tags.get('num_layers', 0)}"
                    chash = hashlib.md5(config_str.encode()).hexdigest()[:8]

                    entry = LeaderboardEntry(
                        algorithm=alg_name,
                        optimizer=opt_name,
                        task=record.get("task", "unknown"),
                        val_accuracy=record.get("val_accuracy", 0.0),
                        energy_proxy=record.get("energy_proxy", 0.0),
                        backward_flops=record.get("backward_flops", 0),
                        requires_backward=record.get("requires_backward", True),
                        param_count=tags.get("hidden_dim", 256)
                        * tags.get("num_layers", 2)
                        * 1000,
                        wall_time_s=record.get("wall_time_ms", 0.0) / 1000.0,
                        peak_memory_mb=record.get("peak_memory_mb", 0.0),
                        mean_acc=record.get("val_accuracy", 0.0),
                        std_acc=0.0,
                        config_hash=chash,
                    )
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")

    if not entries:
        print("No entries parsed.")
        return

    gen = LeaderboardGenerator(entries)
    output_path = "reports/leaderboard.md"
    gen.export_markdown(output_path)
    print(f"Leaderboard exported to {output_path}")


if __name__ == "__main__":
    generate_leaderboard()
