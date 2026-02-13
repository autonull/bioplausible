import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Dict


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
        self.buffer_size = 5  # Flush every 5 calls

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
                # Flatten metrics
                train_acc = r.metrics.get(
                    "training_accuracy", r.metrics.get("train_acc", 0.0)
                )
                val_acc = r.metrics.get("accuracy", r.metrics.get("val_acc", 0.0))
                train_loss = r.metrics.get("loss", r.metrics.get("train_loss", 0.0))
                val_loss = r.metrics.get("val_loss", 0.0)
                perplexity = r.metrics.get("perplexity", 0.0)
                samples_seen = r.metrics.get("samples_seen", 0)
                timestamp = r.metrics.get("timestamp", 0.0)

                # Check for table existence and create if needed (lazy init)
                # This ensures we don't crash if table missing
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS training_checkpoints (
                        trial_id INTEGER,
                        epoch INTEGER,
                        train_acc REAL,
                        val_acc REAL,
                        train_loss REAL,
                        val_loss REAL,
                        samples_seen INTEGER,
                        trajectory_id INTEGER,
                        perplexity REAL,
                        wall_time_seconds REAL,
                        PRIMARY KEY (trial_id, epoch)
                    )
                """)

                data.append(
                    (
                        self.trial_id,
                        r.epoch,
                        train_acc,
                        val_acc,
                        train_loss,
                        val_loss,
                        samples_seen,
                        perplexity,
                        timestamp,
                    )
                )

            conn.executemany(
                """
                INSERT OR REPLACE INTO training_checkpoints 
                (trial_id, trajectory_id, epoch, train_acc, val_acc, train_loss, val_loss, samples_seen, perplexity, wall_time_seconds)
                VALUES (?, -1, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                data,
            )
            conn.commit()
            self.buffer = []
        except Exception as e:
            print(f"Warning: Failed to flush checkpoints: {e}")
        finally:
            conn.close()

    def close(self):
        self.flush()
