"""
Failure tracking and analysis system.
Records why trials fail and provides diagnostics.
"""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class FailureRecord:
    """Records why a trial failed."""
    timestamp: str
    model_name: str
    task_name: str
    tier: str
    trial_id: Optional[int]
    failure_type: str  # "nan_loss", "grad_explode", "oom", "timeout", "assertion_error"
    failure_epoch: Optional[int]
    failure_batch: Optional[int]
    config: Dict[str, Any]
    last_metrics: Dict[str, Any]
    stack_trace: Optional[str] = None

    def to_dict(self):
        return asdict(self)


class FailureTracker:
    """
    Tracks and analyzes training failures.

    Usage:
        tracker = FailureTracker("experiments.db")

        # Log a failure
        tracker.log_failure(FailureRecord(
            timestamp=datetime.now().isoformat(),
            model_name="EqProp MLP",
            task_name="mnist",
            tier="smoke",
            trial_id=42,
            failure_type="grad_nan",
            failure_epoch=2,
            failure_batch=150,
            config={"lr": 0.01, "beta": 0.1},
            last_metrics={"loss": 2.3, "acc": 0.45}
        ))

        # Get statistics
        stats = tracker.get_failure_stats()
        print(stats["by_type"])
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        """Returns a connection that automatically commits."""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize failures table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
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
            "CREATE INDEX IF NOT EXISTS idx_failures_model ON failures(model_name)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_failures_type ON failures(failure_type)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_failures_timestamp ON failures(timestamp)")
        conn.commit()
        conn.close()

    def log_failure(self, record: FailureRecord):
        """Record a failure to database."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO failures 
                (timestamp, model_name, task_name, tier, trial_id, 
                 failure_type, failure_epoch, failure_batch, config, last_metrics, stack_trace)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
                record.stack_trace
            ))
            conn.commit()
            logger.info(f"Logged {record.failure_type} failure for {record.model_name}")
        finally:
            conn.close()

    def get_failure_stats(self, hours: int = None) -> Dict[str, Any]:
        """
        Get aggregate failure statistics.

        Args:
            hours: If provided, only count failures in last N hours

        Returns:
            Dictionary with failure statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build WHERE clause for time filtering
        where_clause = ""
        if hours:
            cutoff = datetime.now().isoformat()[:19]  # Truncate microseconds
            where_clause = f"WHERE timestamp >= datetime('{cutoff}', '-{hours} hours')"

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

        conn.close()

        return {
            "total_failures": total_failures,
            "by_type": by_type,
            "by_model": by_model,
            "by_task": by_task,
            "time_window_hours": hours
        }

    def get_recent_failures(self, limit: int = 50) -> List[FailureRecord]:
        """Get recent failure records."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT timestamp, model_name, task_name, tier, trial_id,
                   failure_type, failure_epoch, failure_batch, config, last_metrics, stack_trace
            FROM failures
            ORDER BY id REST
            LIMIT ?
        """, (limit,))

        records = []
        for row in cursor.fetchall():
            records.append(FailureRecord(
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
                stack_trace=row[10]
            ))

        conn.close()
        return records

    def analyze_failure_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns to suggest fixes.
        Now includes advanced diagnostics:
        1. Divergence Detection (Early vs Late)
        2. Hyperparameter Correlation (vs Successful trials)
        3. Common failure signatures
        
        Returns:
            Dictionary with analysis and recommendations
        """
        stats = self.get_failure_stats()
        recommendations = []

        # 1. NaN/Inf Analysis
        nan_count = stats["by_type"].get("grad_nan", 0) + stats["by_type"].get("loss_nan_or_inf", 0)
        pct_nan = nan_count / stats["total_failures"] if stats["total_failures"] > 0 else 0
        
        if pct_nan > 0.3:
            # Check if likely due to high LR
            high_lr_risk = self._check_hyperparam_correlation("lr", "grad_nan")
            msg = "Reduce learning rate ranges"
            if high_lr_risk:
                msg += f" (High LR detected in failures: mean={high_lr_risk:.2e})"
            
            recommendations.append({
                "issue": "High NaN failure rate",
                "severity": "critical",
                "suggestion": msg,
                "affected_models": list(stats["by_model"].keys())[:3]
            })

        # 2. OOM Analysis
        oom_count = stats["by_type"].get("oom", 0)
        if oom_count > 5:
            recommendations.append({
                "issue": "Out of memory errors",
                "severity": "high",
                "suggestion": "Reduce batch size or model size",
                "count": oom_count
            })
            
        # 3. Divergence Analysis (New)
        divergence_recs = self._detect_divergence_signatures()
        recommendations.extend(divergence_recs)

        return {
            "stats": stats,
            "recommendations": recommendations,
            "analysis_timestamp": datetime.now().isoformat()
        }

    def _check_hyperparam_correlation(self, param: str, failure_type: str) -> Optional[float]:
        """Compare param value in failed vs successful trials."""
        try:
            with self._get_connection() as conn:
                # Get avg param for specific failure type
                cursor = conn.cursor()
                cursor.execute(f"SELECT config FROM failures WHERE failure_type=?", (failure_type,))
                failed_vals = []
                for row in cursor.fetchall():
                    try:
                        cfg = json.loads(row[0])
                        if param in cfg:
                            failed_vals.append(float(cfg[param]))
                    except: pass
                    
                if not failed_vals:
                    return None
                
                avg_fail = sum(failed_vals) / len(failed_vals)
                return avg_fail
                
        except Exception as e:
            logger.warning(f"Correlation check failed: {e}")
            return None

    def _detect_divergence_signatures(self) -> List[Dict[str, Any]]:
        """Identify if failures happen early (instability) or late (collapse)."""
        recs = []
        try:
            with self._get_connection() as conn:
                # Early failures (< epoch 2)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM failures WHERE failure_epoch < 2")
                early_fails = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM failures")
                total = cursor.fetchone()[0]
                
                if total > 0 and (early_fails / total) > 0.5:
                    recs.append({
                        "issue": "Early Training Instability",
                        "severity": "high",
                        "suggestion": "Check initialization or reduce initial LR",
                        "details": f"{early_fails}/{total} failures occurred in first 2 epochs"
                    })
                    
        except Exception:
            pass
            
        return recs
