#!/bin/bash
# Generates the Scientific Discovery Report from 'bioplausible.db'.
# Usage: ./generate_report.sh [--out <output_dir>]

echo "Generating Report..."
python -m bioplausible.scientist.cli report "$@"
