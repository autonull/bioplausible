import sqlite3
import json
import numpy as np
import threading
import time
from dataclasses import dataclass
from pathlib import Path

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
        self.ranges = {
            'lr': [0.0001, 0.1],
            'hidden': [32, 1024],
            'steps': [5, 100]
        }

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
        # We can implement a more robust storage reader
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hyperopt_logs")
            rows = cursor.fetchall()
            conn.close()
        except Exception:
            return

        new_stars = []
        
        # Color map for models (fallback)
        model_colors = {
            'EqProp MLP': (50, 250, 250),
            'Backprop Baseline': (250, 50, 50),
            'Neural Cube': (50, 250, 50),
            'Holomorphic EqProp': (200, 50, 250)
        }
        
        for row in rows:
            try:
                config = json.loads(row['config_json'])
                
                # Extract dimensions
                # Try to find standard keys, or fallbacks
                lr = float(config.get('learning_rate', config.get('lr', 0.001)))
                hidden = float(config.get('hidden_dim', config.get('hidden_size', 128)))
                steps = float(config.get('steps', config.get('num_layers', 20))) # Use num_layers as proxy if steps missing?
                
                # Use a specific mapping based on task? 
                task = config.get('task', 'vision')
                
                # Normalize to -10..10 space
                x = self._log_norm(steps, self.ranges['steps']) * 20 - 10
                y = self._log_norm(hidden, self.ranges['hidden']) * 20 - 10
                z = self._log_norm(lr, self.ranges['lr']) * 20 - 10
                
                # Jitter slightly to avoid overlap
                x += np.random.uniform(-0.1, 0.1)
                
                # Size based on accuracy (0.1 to 1.0) -> size 1 to 5
                acc = row['accuracy'] if row['accuracy'] is not None else 0.0
                size = 2 + (acc * 8)
                
                # Color based on Model Name or Task?
                # Let's prioritize model spec color if we could access it, but hardcoded is faster
                color = model_colors.get(row['model_name'], (200, 200, 200))
                
                # Tint based on task?
                if task == 'lm':
                    color = (color[0], color[1]//2, color[2]) # Darker/Greener?
                elif task == 'rl':
                    color = (color[0], min(255, color[1]+50), color[2]) 
                
                # Status blink
                if row['status'] == 'running':
                    color = (255, 255, 255) # White for running
                elif row['status'] == 'failed':
                    color = (100, 0, 0)
                
                new_stars.append(Star(
                    id=row['trial_id'],
                    pos=np.array([x, y, z]),
                    color=color,
                    size=size,
                    raw_data=dict(row)
                ))
            except Exception as e:
                continue

        with self.lock:
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
