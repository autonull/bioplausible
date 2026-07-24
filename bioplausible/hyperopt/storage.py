"""
SQLite Storage Backend

Persists trials, configurations, and results to a SQLite database.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .metrics import TrialMetrics


class HyperoptStorage:
    """Storage backend for hyperparameter optimization trials."""

    def __init__(self, db_path: str = "results/hyperopt.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable Write-Ahead Logging for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")

        cursor = self.conn.cursor()

        # Trials table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hyperopt_logs (
                trial_id INTEGER PRIMARY KEY,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                epochs_completed INTEGER DEFAULT 0,
                final_loss REAL,
                accuracy REAL,
                perplexity REAL,
                iteration_time REAL,
                param_count REAL,
                is_pareto INTEGER DEFAULT 0
            )
        """)

        # Epoch metrics table (for detailed logging)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS epoch_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                loss REAL,
                accuracy REAL,
                perplexity REAL,
                time REAL,
                FOREIGN KEY (trial_id) REFERENCES hyperopt_logs (trial_id)
            )
        """)

        self.conn.commit()

        # Training trajectories table (Scientist++ Phase 2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trial_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                task_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                convergence_epoch INTEGER,
                converged BOOLEAN,
                overfitting_detected BOOLEAN,
                unstable BOOLEAN,
                FOREIGN KEY (trial_id) REFERENCES hyperopt_logs(trial_id)
            )
        """)

        # Training checkpoints table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trajectory_id INTEGER NOT NULL,
                epoch INTEGER NOT NULL,
                train_acc REAL,
                val_acc REAL,
                test_acc REAL,
                train_loss REAL,
                val_loss REAL,
                grad_norm_mean REAL,
                grad_norm_std REAL,
                weight_norm REAL,
                learning_rate REAL,
                train_val_gap REAL,
                perplexity REAL,
                reward REAL,
                wall_time_seconds REAL,
                total_flops INTEGER,
                samples_seen INTEGER DEFAULT 0,
                FOREIGN KEY (trajectory_id) REFERENCES training_trajectories(id)
            )
        """)

        # Indices
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_trajectory"
            " ON training_checkpoints(trajectory_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch"
            " ON training_checkpoints(epoch);"
        )

        # Schema Migration: Add samples_seen if missing (for legacy DBs)
        cursor.execute("PRAGMA table_info(training_checkpoints)")
        columns = [row["name"] for row in cursor.fetchall()]
        if "samples_seen" not in columns:
            print("Migrating schema: Adding samples_seen column...")
            try:
                cursor.execute(
                    "ALTER TABLE training_checkpoints"
                    " ADD COLUMN samples_seen INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Already exists (race condition)

        self.conn.commit()

    def create_trial(
        self,
        model_name: str,
        config: Dict[str, Any],
        _legacy_force_id: Optional[int] = None,
    ) -> int:
        """
        Create a new trial log.

        Args:
            model_name: Name of the model
            config: Configuration dictionary
            _legacy_force_id: DEPRECATED. Do not use.
                              If provided, forces a specific Trial ID (dangerous).
        """
        cursor = self.conn.cursor()

        if _legacy_force_id is not None:
            cursor.execute(
                """
                INSERT INTO hyperopt_logs
                    (trial_id, model_name, config_json, status, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    _legacy_force_id,
                    model_name,
                    json.dumps(config),
                    "pending",
                    datetime.now().isoformat(),
                ),
            )
            self.conn.commit()
            return _legacy_force_id
        else:
            cursor.execute(
                """
                INSERT INTO hyperopt_logs (model_name, config_json, status, timestamp)
                VALUES (?, ?, ?, ?)
            """,
                (model_name, json.dumps(config), "pending", datetime.now().isoformat()),
            )
            self.conn.commit()
            return cursor.lastrowid

    def update_trial(
        self,
        trial_id: int,
        status: str = None,
        epochs_completed: int = None,
        final_loss: float = None,
        accuracy: float = None,
        perplexity: float = None,
        iteration_time: float = None,
        param_count: float = None,
    ):
        """Update trial with results."""
        updates = []
        values = []

        if status is not None:
            updates.append("status = ?")
            values.append(status)
        if epochs_completed is not None:
            updates.append("epochs_completed = ?")
            values.append(epochs_completed)
        if final_loss is not None:
            updates.append("final_loss = ?")
            values.append(final_loss)
        if accuracy is not None:
            updates.append("accuracy = ?")
            values.append(accuracy)
        if perplexity is not None:
            updates.append("perplexity = ?")
            values.append(perplexity)
        if iteration_time is not None:
            updates.append("iteration_time = ?")
            values.append(iteration_time)
        if param_count is not None:
            updates.append("param_count = ?")
            values.append(param_count)

        if updates:
            values.append(trial_id)
            query = f"UPDATE hyperopt_logs SET {', '.join(updates)} WHERE trial_id = ?"
            self.conn.execute(query, values)
            self.conn.commit()

    def log_epoch(
        self,
        trial_id: int,
        epoch: int,
        loss: float,
        accuracy: float,
        perplexity: float,
        time: float,
    ):
        """Log metrics for a specific epoch."""
        self.conn.execute(
            """
            INSERT INTO epoch_metrics
                (trial_id, epoch, loss, accuracy, perplexity, time)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (trial_id, epoch, loss, accuracy, perplexity, time),
        )
        self.conn.commit()

    def get_trial(self, trial_id: int) -> Optional[TrialMetrics]:
        """Retrieve a trial by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM hyperopt_logs WHERE trial_id = ?", (trial_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return TrialMetrics(
            trial_id=row["trial_id"],
            model_name=row["model_name"],
            config=json.loads(row["config_json"]),
            accuracy=row["accuracy"] or 0.0,
            perplexity=row["perplexity"] or 10.0,
            iteration_time=row["iteration_time"] or 1.0,
            param_count=row["param_count"] or 1.0,
            epochs_completed=row["epochs_completed"] or 0,
            final_loss=row["final_loss"] or 10.0,
            status=row["status"],
        )

    def get_all_trials(
        self, model_name: str = None, status: str = None
    ) -> List[TrialMetrics]:
        """Retrieve all trials, optionally filtered."""
        query = "SELECT * FROM hyperopt_logs WHERE 1=1"
        params = []

        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        if status is not None:
            query += " AND status = ?"
            params.append(status)

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        trials = []
        for row in rows:
            trials.append(
                TrialMetrics(
                    trial_id=row["trial_id"],
                    model_name=row["model_name"],
                    config=json.loads(row["config_json"]),
                    accuracy=row["accuracy"] or 0.0,
                    perplexity=row["perplexity"] or 10.0,
                    iteration_time=row["iteration_time"] or 1.0,
                    param_count=row["param_count"] or 1.0,
                    epochs_completed=row["epochs_completed"] or 0,
                    final_loss=row["final_loss"] or 10.0,
                    status=row["status"],
                )
            )

        return trials

    def mark_pareto_frontier(self, trial_ids: List[int]):
        """Mark trials as being on the Pareto frontier."""
        # Clear previous frontier
        self.conn.execute("UPDATE hyperopt_logs SET is_pareto = 0")

        # Mark new frontier
        if trial_ids:
            placeholders = ",".join("?" * len(trial_ids))
            self.conn.execute(
                f"UPDATE hyperopt_logs SET is_pareto = 1"
                f" WHERE trial_id IN ({placeholders})",
                trial_ids,
            )

        self.conn.commit()

    def clear_all_trials(self):
        """Clear all trials and associated epoch metrics from the database."""
        cursor = self.conn.cursor()

        # Clear epoch metrics first (due to foreign key constraint)
        cursor.execute("DELETE FROM epoch_metrics")

        # Clear trials
        cursor.execute("DELETE FROM hyperopt_logs")

        self.conn.commit()

    def save_trajectory(self, trajectory):
        """
        Save a full TrainingTrajectory and its checkpoints.

        Args:
            trajectory: TrainingTrajectory object
                (from bioplausible.scientist.training_dynamics)
        """
        try:
            cursor = self.conn.cursor()

            # Insert Trajectory
            cursor.execute(
                """
                INSERT INTO training_trajectories (
                    trial_id, model_name, task_name, config_json,
                    convergence_epoch, converged, overfitting_detected, unstable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trajectory.trial_id,
                    trajectory.model_name,
                    trajectory.task_name,
                    json.dumps(trajectory.config),
                    trajectory.convergence_epoch,
                    int(trajectory.converged),
                    int(trajectory.overfitting_detected),
                    int(trajectory.unstable),
                ),
            )

            trajectory_id = cursor.lastrowid

            # Bulk Insert Checkpoints
            checkpoints_data = []
            for ckpt in trajectory.checkpoints:
                checkpoints_data.append(
                    (
                        trajectory_id,
                        ckpt.epoch,
                        ckpt.train_acc,
                        ckpt.val_acc,
                        ckpt.test_acc,
                        ckpt.train_loss,
                        ckpt.val_loss,
                        ckpt.grad_norm_mean,
                        ckpt.grad_norm_std,
                        ckpt.weight_norm,
                        ckpt.learning_rate,
                        ckpt.train_val_gap,
                        ckpt.perplexity,
                        ckpt.reward,
                        ckpt.wall_time_seconds,
                        ckpt.total_flops,
                        ckpt.samples_seen,
                    )
                )

            cursor.executemany(
                """
                INSERT INTO training_checkpoints (
                    trajectory_id, epoch, train_acc, val_acc, test_acc,
                    train_loss, val_loss, grad_norm_mean, grad_norm_std,
                    weight_norm, learning_rate, train_val_gap, perplexity,
                    reward, wall_time_seconds, total_flops, samples_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                checkpoints_data,
            )

            self.conn.commit()

        except sqlite3.Error as e:
            print(f"Database error saving trajectory: {e}")
            # Don't crash training if logging fails, but maybe re-raise?
            # For now, just log error.

    def get_all_trajectories(self):
        """
        Retrieve all training trajectories with their checkpoints.
        Returns: List[TrainingTrajectory] (imported locally to avoid circular import)
        """
        from bioplausible.execution.training_dynamics import TrainingCheckpoint
        from bioplausible.execution.training_dynamics import TrainingTrajectory

        cursor = self.conn.cursor()

        # 1. Get all trajectories
        cursor.execute("""
            SELECT id, trial_id, model_name, task_name, config_json,
                   convergence_epoch, converged, overfitting_detected, unstable
            FROM training_trajectories
        """)
        rows = cursor.fetchall()

        trajectories = []
        for row in rows:
            traj_id = row["id"]

            # 2. Get checkpoints for this trajectory
            # Use SELECT * to handle varying schema (e.g. samples_seen)
            cursor.execute(
                """
                SELECT *
                FROM training_checkpoints
                WHERE trajectory_id = ?
                ORDER BY epoch ASC
            """,
                (traj_id,),
            )
            ckpt_rows = cursor.fetchall()

            checkpoints = []
            for cr in ckpt_rows:
                # Safely get samples_seen (defaults to 0 if column missing in legacy DB)
                # sqlite3.Row supports keys() or explicit check
                samples_seen_val = 0
                try:
                    samples_seen_val = cr["samples_seen"]
                except IndexError, KeyError:
                    pass

                checkpoints.append(
                    TrainingCheckpoint(
                        epoch=cr["epoch"],
                        train_acc=cr["train_acc"],
                        val_acc=cr["val_acc"],
                        test_acc=cr["test_acc"],
                        train_loss=cr["train_loss"],
                        val_loss=cr["val_loss"],
                        grad_norm_mean=cr["grad_norm_mean"],
                        grad_norm_std=cr["grad_norm_std"],
                        weight_norm=cr["weight_norm"],
                        learning_rate=cr["learning_rate"],
                        train_val_gap=cr["train_val_gap"],
                        perplexity=cr["perplexity"],
                        reward=cr["reward"],
                        wall_time_seconds=cr["wall_time_seconds"],
                        total_flops=cr["total_flops"],
                        samples_seen=samples_seen_val,
                    )
                )

            traj = TrainingTrajectory(
                trial_id=row["trial_id"],
                model_name=row["model_name"],
                task_name=row["task_name"],
                config=json.loads(row["config_json"]),
                checkpoints=checkpoints,
            )
            # Set computed/stored fields
            traj.convergence_epoch = row["convergence_epoch"]
            traj.converged = bool(row["converged"])
            traj.overfitting_detected = bool(row["overfitting_detected"])
            traj.unstable = bool(row["unstable"])

            trajectories.append(traj)

        return trajectories

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
