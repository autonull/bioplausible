import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from bioplausible.analysis.scaling import (compute_compute_optimal,
                                           plot_scaling_curves)


def analyze_sweep(results_dir="results/phase1_reduced"):
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} not found.")
        return

    runs_file = os.path.join(results_dir, "runs.jsonl")
    if not os.path.exists(runs_file):
        print(f"Error: {runs_file} not found.")
        return

    # Read runs.jsonl
    data = []
    with open(runs_file, "r") as f:
        # Use simple line splitting since it might have extraneous text
        content = f.read()
        lines = content.split("\n")
        # Handle string containing escaped newlines \n instead of actual newlines
        if len(lines) == 1 and "\\n" in lines[0]:
            lines = lines[0].split("\\n")

        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    record = json.loads(line)
                    # Flatten tags
                    tags = record.get("tags", {})
                    if tags:
                        for k, v in tags.items():
                            record[f"tag_{k}"] = v

                    # Param count is proxy for size. In an ideal setup we extract exact param counts.
                    # Since we don't have it directly in the logs (it's in energy profile if enabled, let's check)

                    record["param_count"] = (
                        record.get("tag_hidden_dim", 256)
                        * record.get("tag_num_layers", 2)
                        * 1000
                    )  # Rough proxy

                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:50]}... -> {e}")

    df = pd.DataFrame(data)

    if df.empty:
        print("No data collected.")
        return

    # Fix Model column name for scaling tool
    if "model" not in df.columns and "tag_model" in df.columns:
        df["model"] = df["tag_model"]

    print(f"Loaded {len(df)} records.")

    # Run Scaling Analysis
    optimal_df = compute_compute_optimal(df)

    fig = plot_scaling_curves(optimal_df, metric="val_accuracy")
    fig.savefig(os.path.join(results_dir, "scaling_accuracy.png"))

    print(f"Saved scaling plot to {os.path.join(results_dir, 'scaling_accuracy.png')}")


if __name__ == "__main__":
    analyze_sweep()
