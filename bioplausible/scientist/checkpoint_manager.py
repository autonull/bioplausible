
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class CheckpointRecord:
    epoch: int
    step: int
    metrics: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)

class CheckpointManager:
    """
    Manages saving checkpoints to the database.
    Designed to be lightweight and synchronous for now (SQLite).
    """
    
    def __init__(self, db_path: str, trial_id: int):
        self.db_path = db_path
        self.trial_id = trial_id
        self.buffer = []
        self.buffer_size = 5 # Flush every 5 calls
        
    def log_metric(self, epoch: int, step: int, metrics: Dict[str, float]):
        """Buffer a metric record."""
        record = CheckpointRecord(epoch, step, metrics)
        self.buffer.append(record)
        
        if len(self.buffer) >= self.buffer_size:
            self.flush()
            
    def flush(self):
        """Write buffer to DB."""
        if not self.buffer:
            return
            
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            data = []
            for r in self.buffer:
                data.append((self.trial_id, r.epoch, r.step, json.dumps(r.metrics)))
                
            conn.executemany("""
                INSERT INTO checkpoints (trial_id, epoch, step, metrics)
                VALUES (?, ?, ?, ?)
            """, data)
            conn.commit()
            self.buffer = []
        except Exception as e:
            print(f"Warning: Failed to flush checkpoints: {e}")
        finally:
            conn.close()
            
    def close(self):
        self.flush()
