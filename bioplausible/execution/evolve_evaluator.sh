#!/bin/bash
# Wrapper script for ASI-Evolve's Engineer
# Engineer executes this script via subprocess.Popen(["bash", script_path], cwd=experiment_dir)
# The current working directory is the experiment_dir, which contains the candidate code in a file named "code"

# Execute the python evaluator passing the 'code' file in the current working directory
python3 $(dirname "$0")/evolve_evaluator.py ./code
