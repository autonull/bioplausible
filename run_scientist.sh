#!/bin/bash
# Runs the Autonomous Scientist
# It will continuously explore the search space until stopped (Ctrl+C).
# Decisions are logged to 'bioplausible.db'.

echo "Starting Auto-Scientist..."
python -m bioplausible.scientist.cli
