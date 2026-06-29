import os
from dataclasses import asdict, dataclass
from typing import List

import pandas as pd


@dataclass
class LeaderboardEntry:
    algorithm: str
    optimizer: str
    task: str
    val_accuracy: float
    energy_proxy: float
    backward_flops: int  # 0 for bio-plausible methods
    requires_backward: bool
    param_count: int
    wall_time_s: float
    peak_memory_mb: float
    mean_acc: float  # across seeds
    std_acc: float
    config_hash: str

    def to_dict(self):
        return asdict(self)


class LeaderboardGenerator:
    """
    Generates markdown leaderboards from LeaderboardEntry data.
    """

    def __init__(self, entries: List[LeaderboardEntry]):
        self.entries = entries
        self.df = pd.DataFrame([e.to_dict() for e in self.entries])

    def export_markdown(self, output_path: str = "reports/leaderboard.md"):
        """
        Exports the leaderboard to a GitHub-flavored Markdown file.
        Contains multiple views: sorted by Accuracy, by Energy Proxy, and Hardware.
        """
        if self.df.empty:
            print("No entries to export.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Bioplausible Leaderboard\n\n")
            f.write(
                "A quantitative comparison of alternative learning rules against standard backpropagation benchmarks.\n\n"
            )

            # --- 1. Highest Accuracy ---
            f.write("## 🏆 Highest Validation Accuracy\n\n")
            f.write("Sorted by descending validation accuracy.\n\n")
            acc_view = self.df.sort_values(by="val_accuracy", ascending=False)
            acc_cols = [
                "algorithm",
                "task",
                "val_accuracy",
                "requires_backward",
                "mean_acc",
                "std_acc",
            ]
            self._write_markdown_table(f, acc_view, acc_cols)

            # --- 2. Energy Efficiency ---
            f.write("## ⚡ Best Energy Efficiency\n\n")
            f.write(
                "Sorted by lowest energy proxy footprint against parameter cost.\n\n"
            )
            erg_view = self.df.sort_values(by="energy_proxy", ascending=True)
            erg_cols = [
                "algorithm",
                "task",
                "energy_proxy",
                "backward_flops",
                "param_count",
                "val_accuracy",
            ]
            self._write_markdown_table(f, erg_view, erg_cols)

            # --- 3. Hardware Footprint ---
            f.write("## 🖥️ Computing Hardware Footprint\n\n")
            f.write("Sorted by peak memory utilization during training runtime.\n\n")
            hw_view = self.df.sort_values(by="peak_memory_mb", ascending=True)
            hw_cols = [
                "algorithm",
                "task",
                "peak_memory_mb",
                "wall_time_s",
                "param_count",
            ]
            self._write_markdown_table(f, hw_view, hw_cols)

    def _write_markdown_table(
        self, file_handle, dataframe: pd.DataFrame, columns: List[str]
    ):
        """Helper to format pandas dataframe slice as markdown table"""
        subset = dataframe[columns].copy()

        # Format floats carefully for human consumption
        for col in subset.columns:
            if subset[col].dtype == "float64":
                subset[col] = subset[col].apply(lambda x: f"{x:.4f}")

        # Write Header
        header = [str(c).replace("_", " ").title() for c in subset.columns]
        file_handle.write(f"| {' | '.join(header)} |\n")

        # Write Separator
        file_handle.write(f"|{'|'.join(['---'] * len(header))}|\n")

        # Write Rows
        for _, row in subset.iterrows():
            row_str = [str(val) for val in row.values]
            file_handle.write(f"| {' | '.join(row_str)} |\n")
        file_handle.write("\n")
