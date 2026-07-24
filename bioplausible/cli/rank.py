"""
CLI Leaderboard Viewer
"""

import argparse

from tabulate import tabulate  # Assuming installed, or use simple formatter

from bioplausible.analysis.results import get_rankings
from bioplausible.analysis.results import load_trials


def view_rankings(args):
    db_path = args.db
    print(f"📊 Loading rankings from {db_path}...")

    trials = load_trials(db_path)
    if not trials:
        print("No trials found.")
        return

    rankings = get_rankings(trials)

    # Prepare table
    data = []
    headers = ["Rank", "Family", "Best Acc", "Gap", "Trials"]

    for r in rankings:
        if args.family and args.family.lower() not in r.family.lower():
            continue

        gap_str = f"{r.gap_to_baseline:+.1f}%" if r.gap_to_baseline != 0 else "Base"
        data.append(
            [f"#{r.rank}", r.family, f"{r.best_value*100:.2f}%", gap_str, r.n_trials]
        )

    print(tabulate(data, headers=headers, tablefmt="simple"))


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Leaderboard Viewer")
    parser.add_argument(
        "--db", default="examples/shallow_benchmark.db", help="Path to database"
    )
    parser.add_argument("--family", help="Filter by algorithm family")

    args = parser.parse_args()
    view_rankings(args)


if __name__ == "__main__":
    main()
