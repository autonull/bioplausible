"""
CLI Entry Points for AutoScientist.
"""

import argparse
from datetime import datetime

from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.report.orchestrator import ReportOrchestrator


def main_scientist():
    """Entry point for the autonomous scientist."""
    parser = argparse.ArgumentParser(description="Run AutoScientist")
    parser.add_argument("--task", type=str, default=None, help="Filter tasks (e.g. 'vision', 'mnist')")
    parser.add_argument("--trials", type=int, default=None, help="Max trials (not strictly enforced yet)")
    args, unknown = parser.parse_known_args()

    print(f"Initializing AutoScientist (Task Filter: {args.task})...")
    scientist = AutoScientist(task_filter=args.task)
    scientist.run()


def main_reporter():
    """Entry point for the reporter."""
    parser = argparse.ArgumentParser(description="Generate AutoScientist Report")
    parser.add_argument("--db", default="bioplausible.db", help="Path to database")
    parser.add_argument("--out", default=None, help="Output directory")
    args = parser.parse_args()

    if args.out is None:
        args.out = "reports"

    print(f"Generating report from {args.db} to {args.out}...")
    orchestrator = ReportOrchestrator(args.db, args.out)
    orchestrator.generate_reports()
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        # Hack to remove 'report' from argv so argparse doesn't get confused
        sys.argv.pop(1)
        main_reporter()
    else:
        main_scientist()
