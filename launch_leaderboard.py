#!/usr/bin/env python3
"""
Quick launcher for the Bioplausible Leaderboard.

Usage:
    python launch_leaderboard.py [db_path]

If no db_path is provided, defaults to examples/shallow_benchmark.db
"""

from bioplausible_ui.leaderboard_window import main

if __name__ == "__main__":
    main()
