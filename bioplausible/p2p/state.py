"""
P2P State Persistence.

Saves user contribution points and job counts.
"""

import json
from pathlib import Path

STATE_FILE = Path("results/p2p_state.json")


def load_state():
    if not STATE_FILE.exists():
        return {"points": 0, "jobs_done": 0}

    try:
        with Path(STATE_FILE).open("r") as f:
            return json.load(f)
    except Exception:
        return {"points": 0, "jobs_done": 0}


def save_state(points, jobs_done):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Path(STATE_FILE).open("w") as f:
            json.dump({"points": points, "jobs_done": jobs_done}, f)
    except Exception as e:
        print(f"Failed to save P2P state: {e}")
