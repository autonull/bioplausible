"""
Bioplausible CLI Entry Point
Usage: python -m bioplausible.cli <command> [args]
"""

import argparse
import sys

from bioplausible.cli.lab import main as lab_main
from bioplausible.cli.rank import main as rank_main
from bioplausible.cli.run import main as run_main


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m bioplausible.cli <run|rank|lab> [args]")
        sys.exit(1)

    command = sys.argv[1]

    # Strip the sub-command from argv so argparse in modules sees their own args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "run":
        run_main()
    elif command == "rank":
        rank_main()
    elif command == "lab":
        lab_main()
    else:
        print(f"Unknown command: {command}")
        print("Available: run, rank, lab")
        sys.exit(1)


if __name__ == "__main__":
    main()
