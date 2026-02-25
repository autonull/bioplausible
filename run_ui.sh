#!/bin/bash

# EquiTile UI Launcher
# Sets up environment and launches the visualization application

# Set PYTHONPATH to include the current directory (project root)
export PYTHONPATH=.

echo "Launching EquiTile UI..."
echo "Using PYTHONPATH=$PYTHONPATH"

# Run the UI application
python bioplausible_ui/apps/equitile_ui/main.py "$@"
