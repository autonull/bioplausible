#!/bin/bash
# Run EquiTile UI
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 bioplausible_ui/apps/equitile_ui/main.py "$@"
