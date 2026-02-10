#!/bin/bash
# Launch the main Training Dashboard
export PYTHONPATH=$PYTHONPATH:.
python3 -m bioplausible_ui.studio.studio "$@"
