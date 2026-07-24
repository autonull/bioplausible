"""
CLI Entry Points for AutoScientist.

This module provides the command-line interface for the AutoScientist system.
It supports two main modes:
1. `run`: Start the autonomous discovery agent.
2. `report`: Generate scientific reports from the database.
"""

import argparse
import sys

from bioplausible.execution.engine import ExecutionEngine
from bioplausible.execution.report.orchestrator import ReportOrchestrator


def main() -> None:
    """
    Main entry point for the CLI.
    """
    parser = argparse.ArgumentParser(description="AutoScientist CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Subcommand: run (Default)
    run_parser = subparsers.add_parser("run", help="Start the autonomous scientist")
    run_parser.add_argument(
        "--task", type=str, default=None, help="Filter tasks (e.g. 'vision', 'mnist')"
    )
    run_parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Max trials (not strictly enforced yet)",
    )
    run_parser.add_argument(
        "--tier-limit",
        type=str,
        default=None,
        help="Limit maximum tier (smoke, shallow, standard, deep)",
    )
    run_parser.add_argument("--db", default="bioplausible.db", help="Path to database")
    run_parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (default: 1)"
    )

    # Subcommand: report
    report_parser = subparsers.add_parser("report", help="Generate scientific report")
    report_parser.add_argument(
        "--db", default="bioplausible.db", help="Path to database"
    )
    report_parser.add_argument(
        "--out", default="reports", help="Output directory for reports"
    )

    # Hack to support running without 'run' command for backward compatibility
    # If no command provided, default to 'run'
    if len(sys.argv) == 1:
        args = parser.parse_args(["run"])
    elif sys.argv[1] not in ["run", "report"] and not sys.argv[1].startswith("-"):
        # Could be 'bioplausible-scientist' assuming 'run'
        # But if user typed 'bioplausible-scientist --task mnist', sys.argv[1] is --task
        # We need to inject 'run' if it's missing
        args = parser.parse_args(["run"] + sys.argv[1:])
    elif sys.argv[1].startswith("-") and sys.argv[1] not in ["-h", "--help"]:
        args = parser.parse_args(["run"] + sys.argv[1:])
    else:
        args = parser.parse_args()

    if args.command == "report":
        _run_reporter(args)
    else:
        _run_scientist(args)


def _run_scientist(args: argparse.Namespace) -> None:
    """Execute the scientist runner."""
    print(
        f"Initializing AutoScientist (Task Filter: {args.task}, Workers: {args.workers})..."
    )
    engine = ExecutionEngine(
        db_path=args.db,
        task_filter=args.task,
        tier_limit=args.tier_limit,
        num_workers=args.workers,
    )
    engine.run()


def _run_reporter(args: argparse.Namespace) -> None:
    """Execute the report generator."""
    print(f"Generating report from {args.db} to {args.out}...")
    orchestrator = ReportOrchestrator(args.db, args.out)
    orchestrator.generate_reports()
    print("Done.")


if __name__ == "__main__":
    main()
