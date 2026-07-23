#!/usr/bin/env python3
"""
Run EquiTile UI Demo
=====================

Launch the EquiTile Live Training Demo.
"""

import os
import sys

# Ensure bioplausible is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bioplausible_ui.apps.equitile_ui.main import main

if __name__ == "__main__":
    main()
