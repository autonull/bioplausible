import sqlite3
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger("DecisionLogger")

class DecisionLogger:
    """
    Logs high-level scientific decisions to a persistent database.
    This provides the "Auditable Decision Trail".
    """

    def __init__(self, db_path: str = "bioplausible.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the decisions table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS decision_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        event_type TEXT,
                        description TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init decision log DB: {e}")

    def log_decision(
        self, event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a decision.

        Args:
            event_type: e.g., "PROMOTION", "REFINEMENT", "FAILURE_ANALYSIS", "NEW_HYPOTHESIS"
            description: Human-readable explanation.
            metadata: Structured data (e.g., {"model": "EqProp", "accuracy": 0.95})
        """
        try:
            meta_json = json.dumps(metadata) if metadata else "{}"
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO decision_log (timestamp, event_type, description, metadata) VALUES (?, ?, ?, ?)",
                    (time.time(), event_type, description, meta_json),
                )
                conn.commit()
            logger.info(f"Decision Logged: [{event_type}] {description}")
        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    def get_log(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve the decision log."""
        entries = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM decision_log ORDER BY timestamp ASC LIMIT ?", (limit,)
                )
                rows = cursor.fetchall()

                for row in rows:
                    entries.append({
                        "id": row["id"],
                        "timestamp": row["timestamp"],
                        "date_str": datetime.fromtimestamp(row["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
                        "event_type": row["event_type"],
                        "description": row["description"],
                        "metadata": json.loads(row["metadata"])
                    })
        except Exception as e:
            logger.error(f"Failed to read decision log: {e}")

        return entries
