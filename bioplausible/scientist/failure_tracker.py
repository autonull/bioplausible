"""
Failure tracking and analysis system.

Records why trials fail and provides diagnostics to help the scientist
adapt its strategy (e.g., reducing learning rates if NaNs are detected).
"""

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    CONVERGENCE_FAILURE = "convergence_failure"
    GRADIENT_EXPLOSION = "gradient_explosion"
    SETTLING_DIVERGENCE = "settling_divergence"  # EP-specific: states don't converge
    SPECTRAL_INSTABILITY = "spectral_instability"  # σ(W) exceeds bound
    MEMORY_OOM = "memory_oom"
    TASK_INCOMPATIBILITY = "task_incompatibility"
    SLOW_CONVERGENCE = "slow_convergence"  # >3× baseline wall time
    NEGATIVE_TRANSFER = "negative_transfer"
    GOODNESS_COLLAPSE = "goodness_collapse"  # FF-specific: all goodness → 0
    SPIKE_SILENCING = "spike_silencing"  # STDP: all neurons go silent


@dataclass
class FailureRecord:
    """
    Records why a trial failed.

    Attributes:
        timestamp: ISO formatted timestamp.
        model_name: Name of the failing model.
        task_name: Task being attempted.
        tier: Experiment tier.
        trial_id: Optuna trial ID (if available).
        failure_type: Category (e.g., "grad_nan", "oom").
        failure_epoch: Epoch where failure occurred.
        failure_batch: Batch index where failure occurred.
        config: Hyperparameters used.
        last_metrics: Last recorded metrics before failure.
        stack_trace: Traceback string if an exception occurred.
    """

    timestamp: str
    model_name: str
    task_name: str
    tier: str
    trial_id: Optional[int]
    failure_type: str  # e.g. FailureCategory.GRADIENT_EXPLOSION.value
    failure_epoch: Optional[int]
    failure_batch: Optional[int]
    config: Dict[str, Any]
    last_metrics: Dict[str, Any]
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)


class FailureTracker:
    """
    Tracks and analyzes training failures.

    Persists failure data to SQLite and provides analysis methods to detect
    patterns (e.g., specific models prone to gradient explosions).
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the FailureTracker.

        Args:
            db_path (str): Path to the SQLite database.
        """
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a connection to the database."""
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        """Initialize failures table if it doesn't exist."""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    task_name TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    trial_id INTEGER,
                    failure_type TEXT NOT NULL,
                    failure_epoch INTEGER,
                    failure_batch INTEGER,
                    config TEXT NOT NULL,
                    last_metrics TEXT NOT NULL,
                    stack_trace TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_model ON failures(model_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(failure_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_failures_timestamp ON failures(timestamp)"
            )
            conn.commit()
        finally:
            conn.close()

    def log_failure(self, record: FailureRecord) -> None:
        """
        Record a failure to the database.

        Args:
            record (FailureRecord): The failure event details.
        """
        conn = self._get_connection()
        try:
            conn.execute(
                """
                INSERT INTO failures
                (timestamp, model_name, task_name, tier, trial_id,
                 failure_type, failure_epoch, failure_batch, config, last_metrics, stack_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    record.timestamp,
                    record.model_name,
                    record.task_name,
                    record.tier,
                    record.trial_id,
                    record.failure_type,
                    record.failure_epoch,
                    record.failure_batch,
                    json.dumps(record.config),
                    json.dumps(record.last_metrics),
                    record.stack_trace,
                ),
            )
            conn.commit()
            logger.info(f"Logged {record.failure_type} failure for {record.model_name}")
        finally:
            conn.close()

    def get_failure_stats(self, hours: Optional[int] = None) -> Dict[str, Any]:
        """
        Get aggregate failure statistics.

        Args:
            hours: If provided, only count failures in the last N hours.

        Returns:
            Dict[str, Any]: Statistics including counts by type, model, and task.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Build WHERE clause for time filtering
            where_clause = ""
            if hours:
                cutoff = datetime.now().isoformat()[:19]  # Truncate microseconds
                where_clause = (
                    f"WHERE timestamp >= datetime('{cutoff}', '-{hours} hours')"
                )

            # Failures by type
            cursor.execute(f"""
                SELECT failure_type, COUNT(*) as count
                FROM failures
                {where_clause}
                GROUP BY failure_type
                ORDER BY count DESC
            """)
            by_type = dict(cursor.fetchall())

            # Failures by model
            cursor.execute(f"""
                SELECT model_name, COUNT(*) as count
                FROM failures
                {where_clause}
                GROUP BY model_name
                ORDER BY count DESC
                LIMIT 10
            """)
            by_model = dict(cursor.fetchall())

            # Failures by task
            cursor.execute(f"""
                SELECT task_name, COUNT(*) as count
                FROM failures
                {where_clause}
                GROUP BY task_name
                ORDER BY count DESC
            """)
            by_task = dict(cursor.fetchall())

            # Total count
            cursor.execute(f"SELECT COUNT(*) FROM failures {where_clause}")
            total_failures = cursor.fetchone()[0]

            return {
                "total_failures": total_failures,
                "by_type": by_type,
                "by_model": by_model,
                "by_task": by_task,
                "time_window_hours": hours,
            }
        finally:
            conn.close()

    def get_recent_failures(self, limit: int = 50) -> List[FailureRecord]:
        """
        Get recent failure records.

        Args:
            limit (int): Max number of records to retrieve.

        Returns:
            List[FailureRecord]: List of recent failures.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT timestamp, model_name, task_name, tier, trial_id,
                       failure_type, failure_epoch, failure_batch, config, last_metrics, stack_trace
                FROM failures
                ORDER BY id DESC
                LIMIT ?
            """,
                (limit,),
            )

            records = []
            for row in cursor.fetchall():
                records.append(
                    FailureRecord(
                        timestamp=row[0],
                        model_name=row[1],
                        task_name=row[2],
                        tier=row[3],
                        trial_id=row[4],
                        failure_type=row[5],
                        failure_epoch=row[6],
                        failure_batch=row[7],
                        config=json.loads(row[8]),
                        last_metrics=json.loads(row[9]),
                        stack_trace=row[10],
                    )
                )
            return records
        finally:
            conn.close()

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns to suggest fixes.

        Includes advanced diagnostics:
        1. Divergence Detection (Early vs Late)
        2. Hyperparameter Correlation (vs Successful trials)
        3. Common failure signatures

        Returns:
            Dict[str, Any]: Analysis results and recommendations.
        """
        stats = self.get_failure_stats()
        recommendations: List[Dict[str, Any]] = []

        # 1. NaN/Inf Analysis
        nan_count = stats["by_type"].get("grad_nan", 0) + stats["by_type"].get(
            "loss_nan_or_inf", 0
        )
        pct_nan = (
            nan_count / stats["total_failures"] if stats["total_failures"] > 0 else 0
        )

        if pct_nan > 0.3:
            # Check if likely due to high LR
            high_lr_risk = self._check_hyperparam_correlation("lr", "grad_nan")
            msg = "Reduce learning rate ranges"
            if high_lr_risk:
                msg += f" (High LR detected in failures: mean={high_lr_risk:.2e})"

            recommendations.append(
                {
                    "issue": "High NaN failure rate",
                    "severity": "critical",
                    "suggestion": msg,
                    "affected_models": list(stats["by_model"].keys())[:3],
                }
            )

        # 2. OOM Analysis
        oom_count = stats["by_type"].get("oom", 0)
        if oom_count > 5:
            recommendations.append(
                {
                    "issue": "Out of memory errors",
                    "severity": "high",
                    "suggestion": "Reduce batch size or model size",
                    "count": oom_count,
                }
            )

        # 3. Timeout Analysis
        timeout_count = stats["by_type"].get("timeout", 0)
        if timeout_count > 3:
            recommendations.append(
                {
                    "issue": "Frequent timeouts",
                    "severity": "high",
                    "suggestion": "Reduce model size or iterations",
                    "count": timeout_count,
                    "affected_models": list(stats["by_model"].keys())[:3],
                }
            )

        # 4. Divergence Analysis
        divergence_recs = self._detect_divergence_signatures()
        recommendations.extend(divergence_recs)

        return {
            "stats": stats,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _check_hyperparam_correlation(
        self, param: str, failure_type: str
    ) -> Optional[float]:
        """
        Compare param value in failed vs successful trials.

        Args:
            param: Hyperparameter name.
            failure_type: Failure type to correlate with.

        Returns:
            Optional[float]: Average value of param in failed trials, or None.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config FROM failures WHERE failure_type=?", (failure_type,)
                )
                failed_vals = []
                for row in cursor.fetchall():
                    try:
                        cfg = json.loads(row[0])
                        if param in cfg:
                            failed_vals.append(float(cfg[param]))
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass

                if not failed_vals:
                    return None

                avg_fail = sum(failed_vals) / len(failed_vals)
                return avg_fail

        except Exception as e:
            logger.warning(f"Correlation check failed: {e}")
            return None

    def _detect_divergence_signatures(self) -> List[Dict[str, Any]]:
        """
        Identify if failures happen early (instability) or late (collapse).

        Returns:
            List[Dict]: List of diagnostic findings.
        """
        recs = []
        try:
            with self._get_connection() as conn:
                # Early failures (< epoch 2)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM failures WHERE failure_epoch < 2")
                res = cursor.fetchone()
                early_fails = res[0] if res else 0

                cursor.execute("SELECT COUNT(*) FROM failures")
                res = cursor.fetchone()
                total = res[0] if res else 0

                if total > 0 and (early_fails / total) > 0.5:
                    recs.append(
                        {
                            "issue": "Early Training Instability",
                            "severity": "high",
                            "suggestion": "Check initialization or reduce initial LR",
                            "details": f"{early_fails}/{total} failures occurred in first 2 epochs",
                        }
                    )

        except Exception as e:
            logger.warning(f"Divergence check failed: {e}")

        return recs
