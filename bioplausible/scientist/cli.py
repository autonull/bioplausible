"""
CLI Entry Points for AutoScientist.
"""

import argparse
from datetime import datetime
from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.reporting import ScientistReporter
# Dynamic import to allow game_rpg to be optional/external if needed,
# but here we assume it's in the python path
try:
    from game_rpg.interface import GameInterface
except ImportError:
    GameInterface = None

def main_scientist():
    """Entry point for the autonomous scientist."""
    print("Initializing AutoScientist...")
    scientist = AutoScientist()
    scientist.run()

def main_reporter():
    """Entry point for the reporter."""
    parser = argparse.ArgumentParser(description="Generate AutoScientist Report")
    parser.add_argument("--db", default="bioplausible.db", help="Path to database")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    if args.out is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        args.out = f"reports/{timestamp}"

    print(f"Generating report from {args.db} to {args.out}...")
    reporter = ScientistReporter(args.db)
    reporter.generate_report(args.out)
    print("Done.")

def main_game():
    """Entry point for the Research Game."""
    if GameInterface is None:
        print("Error: game_rpg module not found.")
        return

    interface = GameInterface()
    interface.run()
