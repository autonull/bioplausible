"""
Decision Logger.

Logs high-level scientific decisions to a persistent database.
This provides the "Auditable Decision Trail" required for trustworthy
autonomous science.
"""

import json
import logging
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("DecisionLogger")


class DecisionLogger:
    """
    Logs high-level scientific decisions to a persistent database.

    Attributes:
        db_path (str): Path to the SQLite database.
    """

    def __init__(self, db_path: str = "bioplausible.db") -> None:
        """
        Initialize the Decision Logger.

        Args:
            db_path (str): Path to the database file.
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the decision log table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decision_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        event_type TEXT,
                        description TEXT,
                        metadata TEXT
                    )
                """
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to init decision log DB: {e}")

    def log_decision(
        self,
        event_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a decision in the database.

        Args:
            event_type (str): Type of event (e.g., "PROMOTION", "FAILURE_ANALYSIS").
            description (str): Human-readable explanation.
            metadata (Optional[Dict]): Structured data related to the decision.
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
        except sqlite3.Error as e:
            logger.error(f"Failed to log decision: {e}")
        except Exception as e:
            logger.error(f"Unexpected error logging decision: {e}", exc_info=True)

    def get_log(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieve the recent decision log entries.

        Args:
            limit (int): Maximum number of entries to retrieve.

        Returns:
            List[Dict]: List of decision log entries.
        """
        entries = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM decision_log ORDER BY timestamp ASC LIMIT ?",
                    (limit,),
                )
                rows = cursor.fetchall()

                for row in rows:
                    entries.append(
                        {
                            "id": row["id"],
                            "timestamp": row["timestamp"],
                            "date_str": datetime.fromtimestamp(row["timestamp"]).strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                            "event_type": row["event_type"],
                            "description": row["description"],
                            "metadata": json.loads(row["metadata"]),
                        }
                    )
        except sqlite3.Error as e:
            logger.error(f"Failed to read decision log: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading decision log: {e}", exc_info=True)

        return entries
