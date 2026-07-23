#!/bin/bash

# EquiTile UI Launcher (Unified Studio)
# Sets up environment and launches the visualization application

# Set PYTHONPATH to include the current directory (project root)
export PYTHONPATH=.

echo "Launching Bioplausible Studio (including EquiTile UI)..."
echo "Using PYTHONPATH=$PYTHONPATH"

# Run the Studio application
python bioplausible_ui/studio/studio.py "$@"
