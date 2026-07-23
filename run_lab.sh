#!/bin/bash
# Launch the Lab Analysis Tool
export PYTHONPATH=$PYTHONPATH:.
python3 -m bioplausible_ui.lab.main "$@"
