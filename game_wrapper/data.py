import json
import sqlite3
import threading
import time
from dataclasses import dataclass

import numpy as np

# Try to import HyperoptStorage, or reimplement lightweight version if import fails
# to avoid pulling in heavy torch dependencies in the game loop main thread if possible,
# though we probably need them eventually.
try:
    from bioplausible.hyperopt.storage import HyperoptStorage
except ImportError:
    # Use a direct sqlite connection if path issues, assuming standard schema
    class HyperoptStorage:
        def __init__(self, db_path):
            self.db_path = db_path

        def get_all_trials(self):
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM hyperopt_logs")
            rows = c.fetchall()
            conn.close()
            return rows


@dataclass
class Star:
    id: int
    pos: np.array  # Normalized 3D position
    color: tuple
    size: float
    raw_data: dict


class DataManager:
    def __init__(self, db_path="results/hyperopt.db"):
        self.db_path = db_path
        self.stars = []
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop)
        self.last_count = 0

        # Ranges for normalization
        self.ranges = {"lr": [0.0001, 0.1], "hidden": [32, 1024], "steps": [5, 100]}

    def start(self):
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def _poll_loop(self):
        while self.running:
            try:
                self._refresh()
            except Exception as e:
                print(f"Data poll error: {e}")
            time.sleep(2.0)

    def _refresh(self):
        try:
            # Use WAL mode for better concurrency
            conn = sqlite3.connect(self.db_path, timeout=10.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hyperopt_logs")
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"DB Read Error: {e}")
            return

        new_stars = []
        model_colors = {
            "EqProp MLP": (50, 250, 250),
            "Backprop Baseline": (250, 50, 50),
            "Neural Cube": (50, 250, 50),
            "Holomorphic EqProp": (200, 50, 250),
        }

        for row in rows:
            try:
                # Integrity check
                if not row["config_json"]:
                    continue

                config = json.loads(row["config_json"])

                # Robust key extraction
                lr = float(config.get("learning_rate", config.get("lr", 0.001)))
                hidden = float(config.get("hidden_dim", config.get("hidden_size", 128)))
                steps = float(config.get("steps", config.get("num_layers", 20)))

                task = config.get("task", "vision")

                x = self._log_norm(steps, self.ranges["steps"]) * 20 - 10
                y = self._log_norm(hidden, self.ranges["hidden"]) * 20 - 10
                z = self._log_norm(lr, self.ranges["lr"]) * 20 - 10

                # Consistency Check: Use Trial ID for consistent noise
                np.random.seed(row["trial_id"] + 42)
                x += np.random.uniform(-0.2, 0.2)
                y += np.random.uniform(-0.2, 0.2)

                acc = row["accuracy"] if row["accuracy"] is not None else 0.0
                size = 2 + (acc * 8)

                color = model_colors.get(row["model_name"], (200, 200, 200))

                if task == "lm":
                    color = (color[0], color[1] // 2, color[2])
                elif task == "rl":
                    color = (color[0], min(255, color[1] + 50), color[2])

                if row["status"] == "running":
                    color = (255, 255, 255)
                elif row["status"] == "failed":
                    color = (100, 0, 0)

                new_stars.append(
                    Star(
                        id=row["trial_id"],
                        pos=np.array([x, y, z]),
                        color=color,
                        size=size,
                        raw_data=dict(row),
                    )
                )
            except Exception as e:
                # Log bad rows?
                continue

        # Only update if we have valid data (prevent flicker on empty read)
        with self.lock:
            if len(new_stars) > 0 or len(rows) == 0:
                self.stars = new_stars
                self.last_count = len(new_stars)

    def _log_norm(self, val, bounds):
        """Logarithmic normalization between 0 and 1"""
        start, end = np.log(bounds[0]), np.log(bounds[1])
        v = np.log(max(val, bounds[0]))
        return np.clip((v - start) / (end - start), 0, 1)

    def get_stars(self):
        with self.lock:
            return list(self.stars)
