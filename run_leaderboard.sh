#!/bin/bash
# Leaderboard Launcher (PyQt6)

echo "🎯 Launching Bioplausible Leaderboard (PyQt6)"
echo "=============================================="
echo ""
echo "Database: ${1:-examples/shallow_benchmark.db}"
echo ""

python3 -m bioplausible_ui.leaderboard.leaderboard_window "$@"
