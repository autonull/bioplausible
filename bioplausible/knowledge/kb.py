"""
Upgraded KnowledgeBase with SQLite + Vector Store

Provides hybrid structured + embedding search for AutoScientist.
Integrates surrogate models, symbolic regression, causal discovery.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional dependencies for vector search
try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """A single knowledge entry with metadata and optional embedding."""

    id: str
    topic: str
    model_family: str
    finding: str
    details: str
    confidence: float
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    source: str = "manual"  # "manual", "experiment", "surrogate", "causal"
    experiment_id: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Don't store embedding in JSON
        d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in d.items() if k in cls.__annotations__})


class KnowledgeBase:
    """
    Upgraded KnowledgeBase with SQLite + Vector Store.

    Features:
    - SQLite for structured queries (tags, model_family, metrics, etc.)
    - FAISS/Vector store for semantic similarity search
    - Surrogate model integration for predicting experiment outcomes
    - Symbolic regression for extracting analytical formulas
    - Causal discovery for identifying causal factors
    """

    def __init__(
        self,
        db_path: str = "bioplausible_kb.db",
        vector_dim: int = 384,
        embedding_model: str = "all-MiniLM-L6-v2",
        auto_embed: bool = True,
    ):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.auto_embed = auto_embed

        # Initialize SQLite
        self._init_db()

        # Initialize vector index
        self._init_vector_index()

        # Initialize embedding model
        self.embedding_model = None
        if auto_embed and HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")

        # Load seed data if empty
        self._load_seed_if_empty()

    def _init_db(self) -> None:
        """Initialize SQLite database with tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    finding TEXT NOT NULL,
                    details TEXT,
                    confidence REAL NOT NULL,
                    tags TEXT,  -- JSON array
                    timestamp REAL NOT NULL,
                    source TEXT DEFAULT 'manual',
                    experiment_id TEXT,
                    metrics TEXT,  -- JSON
                    hyperparameters TEXT,  -- JSON
                    extra TEXT  -- JSON
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_topic ON knowledge(topic)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_family ON knowledge(model_family)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source ON knowledge(source)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment ON knowledge(experiment_id)
            """)

            # Table for experiment results
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    task TEXT NOT NULL,
                    config TEXT,  -- JSON
                    metrics TEXT,  -- JSON
                    status TEXT DEFAULT 'completed',
                    timestamp REAL NOT NULL,
                    artifacts TEXT  -- JSON
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_model ON experiments(model_family)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_task ON experiments(task)
            """)

            # Table for surrogate model predictions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surrogates (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    model_type TEXT NOT NULL,  -- 'gp', 'rf', 'nn', 'symbolic'
                    target_metric TEXT NOT NULL,
                    features TEXT,  -- JSON list of feature names
                    trained_at REAL NOT NULL,
                    performance TEXT,  -- JSON metrics
                    model_path TEXT
                )
            """)

            conn.commit()

    def _init_vector_index(self) -> None:
        """Initialize FAISS vector index."""
        if HAS_FAISS:
            self.vector_index = faiss.IndexFlatIP(
                self.vector_dim
            )  # Inner product for cosine similarity
            self.vector_ids = []  # Maps index position to knowledge entry ID
        else:
            self.vector_index = None
            self.vector_ids = []
            logger.warning(
                "FAISS not available. Vector search disabled. "
                "Install with: pip install faiss-cpu"
            )

    def _load_seed_if_empty(self) -> None:
        """Load seed knowledge if database is empty."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]

        if count == 0:
            self._load_seed_data()

    def _load_seed_data(self) -> None:
        """Load initial seed knowledge."""
        seed_entries = [
            KnowledgeEntry(
                id="KB-001",
                topic="Scaling",
                model_family="eqprop",
                finding="O(1) memory scaling for BPTT equivalent",
                details=(
                    "Equilibrium Propagation requires constant memory "
                    "regardless of trajectory length, unlike BPTT which scales O(T)."
                ),
                confidence=0.95,
                tags=["memory", "scaling", "eqprop"],
                source="literature",
            ),
            KnowledgeEntry(
                id="KB-002",
                topic="Architecture",
                model_family="tile_eq",
                finding="Optimal 2D grid layout improves locality",
                details=(
                    "TileEQ variants arranged in a 2D locally connected grid "
                    "demonstrate superior scaling on neuromorphic simulators "
                    "vs fully connected counterparts."
                ),
                confidence=0.85,
                tags=["architecture", "neuromorphic", "local-learning"],
                source="literature",
            ),
            KnowledgeEntry(
                id="KB-003",
                topic="Optimization",
                model_family="forward_forward",
                finding="Layer-local goodness thresholds",
                details=(
                    "A threshold of 2.0 provides stable contrastive separation "
                    "on MNIST-level tasks without causing early layer saturation."
                ),
                confidence=0.80,
                tags=["hyperparams", "forward-forward", "thresholds"],
                source="literature",
            ),
        ]

        for entry in seed_entries:
            self.add_entry(entry)

        logger.info(f"Loaded {len(seed_entries)} seed knowledge entries")

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text."""
        if self.embedding_model is None:
            return None
        try:
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None

    def add_entry(self, entry: KnowledgeEntry) -> str:
        """Add a knowledge entry to the database."""
        # Generate embedding if auto_embed enabled
        if self.auto_embed and entry.embedding is None:
            text = f"{entry.topic} {entry.finding} {entry.details}"
            embedding = self._embed_text(text)
            if embedding is not None:
                entry.embedding = embedding.tolist()

        # Store in SQLite
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge
                (id, topic, model_family, finding, details, confidence, tags,
                 timestamp, source, experiment_id, metrics, hyperparameters, extra)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.id,
                    entry.topic,
                    entry.model_family,
                    entry.finding,
                    entry.details,
                    entry.confidence,
                    json.dumps(entry.tags),
                    entry.timestamp,
                    entry.source,
                    entry.experiment_id,
                    json.dumps(entry.metrics),
                    json.dumps(entry.hyperparameters),
                    json.dumps(entry.extra),
                ),
            )
            conn.commit()

        # Add to vector index
        if self.vector_index is not None and entry.embedding is not None:
            embedding = np.array(entry.embedding, dtype=np.float32).reshape(1, -1)
            self.vector_index.add(embedding)
            self.vector_ids.append(entry.id)

        logger.debug(f"Added knowledge entry: {entry.id}")
        return entry.id

    def add_experiment(
        self,
        name: str,
        model_family: str,
        task: str,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        experiment_id: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
        status: str = "completed",
    ) -> str:
        """Add an experiment result to the knowledge base."""
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())[:8]

        entry = KnowledgeEntry(
            id=f"EXP-{experiment_id}",
            topic="Experiment",
            model_family=model_family,
            finding=f"{name} on {task}",
            details=f"Experiment {name} with {model_family} on {task}",
            confidence=1.0,
            tags=["experiment", model_family, task],
            source="experiment",
            experiment_id=experiment_id,
            metrics=metrics,
            hyperparameters=config,
            extra={"artifacts": artifacts or {}},
        )

        self.add_entry(entry)

        # Also store in experiments table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, name, model_family, task, config, metrics, status,
                 timestamp, artifacts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    name,
                    model_family,
                    task,
                    json.dumps(config),
                    json.dumps(metrics),
                    status,
                    time.time(),
                    json.dumps(artifacts or {}),
                ),
            )
            conn.commit()

        return experiment_id

    def query(
        self,
        tag: Optional[str] = None,
        model_family: Optional[str] = None,
        topic: Optional[str] = None,
        source: Optional[str] = None,
        min_confidence: Optional[float] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[KnowledgeEntry]:
        """Query knowledge base with structured filters."""
        conditions = []
        params = []

        if tag:
            conditions.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if topic:
            conditions.append("topic = ?")
            params.append(topic)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if min_confidence is not None:
            conditions.append("confidence >= ?")
            params.append(min_confidence)
        if experiment_id:
            conditions.append("experiment_id = ?")
            params.append(experiment_id)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT * FROM knowledge"
                f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])
            rows = cursor.fetchall()

        return [self._row_to_entry(row) for row in rows]

    def _row_to_entry(self, row: sqlite3.Row) -> KnowledgeEntry:
        """Convert SQLite row to KnowledgeEntry."""
        return KnowledgeEntry(
            id=row["id"],
            topic=row["topic"],
            model_family=row["model_family"],
            finding=row["finding"],
            details=row["details"] or "",
            confidence=row["confidence"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            timestamp=row["timestamp"],
            source=row["source"],
            experiment_id=row["experiment_id"],
            metrics=json.loads(row["metrics"]) if row["metrics"] else {},
            hyperparameters=(
                json.loads(row["hyperparameters"]) if row["hyperparameters"] else {}
            ),
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )

    def search(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """
        Semantic search using vector embeddings.

        Returns list of (entry, similarity_score) tuples.
        """
        if self.vector_index is None or self.embedding_model is None:
            logger.warning(
                "Vector search not available. Falling back to keyword search."
            )
            return self._keyword_search(query, k, filters)

        # Generate query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            return []

        query_embedding = query_embedding.reshape(1, -1)

        # Search vector index
        scores, indices = self.vector_index.search(
            query_embedding, min(k * 2, len(self.vector_ids))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.vector_ids):
                if score >= min_similarity:
                    entry_id = self.vector_ids[idx]
                    entry = self.get_by_id(entry_id)
                    if entry:
                        # Apply filters
                        if filters:
                            if not self._matches_filters(entry, filters):
                                continue
                        results.append((entry, float(score)))
                        if len(results) >= k:
                            break

        return results

    def _matches_filters(self, entry: KnowledgeEntry, filters: Dict[str, Any]) -> bool:
        """Check if entry matches filter criteria."""
        for key, value in filters.items():
            if key == "model_family" and entry.model_family != value:
                return False
            if key == "topic" and entry.topic != value:
                return False
            if key == "tags" and not all(tag in entry.tags for tag in value):
                return False
            if key == "min_confidence" and entry.confidence < value:
                return False
        return True

    def _keyword_search(
        self,
        query: str,
        k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[KnowledgeEntry, float]]:
        """Fallback keyword search."""
        query_lower = query.lower()
        results = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM knowledge ORDER BY timestamp DESC LIMIT 1000"
            )
            for row in cursor:
                entry = self._row_to_entry(row)

                # Simple keyword matching
                text = " ".join([
                    entry.topic, entry.finding, entry.details,
                    " ".join(entry.tags),
                ]).lower()
                score = sum(1 for word in query_lower.split() if word in text) / len(
                    query_lower.split()
                )

                if score > 0:
                    if filters and not self._matches_filters(entry, filters):
                        continue
                    results.append((entry, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def get_by_id(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM knowledge WHERE id = ?", (entry_id,))
            row = cursor.fetchone()

        if row:
            return self._row_to_entry(row)
        return None

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            )
            row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def list_experiments(
        self,
        model_family: Optional[str] = None,
        task: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filters."""
        conditions = []
        params = []

        if model_family:
            conditions.append("model_family = ?")
            params.append(model_family)
        if task:
            conditions.append("task = ?")
            params.append(task)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            sql = (
                "SELECT * FROM experiments"
                f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
            )
            cursor = conn.execute(sql, params + [limit])
            return [dict(row) for row in cursor]

    def get_surrogate(self, name: str) -> Optional[Dict[str, Any]]:
        """Get surrogate model by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates WHERE name = ?", (name,))
            row = cursor.fetchone()

        return dict(row) if row else None

    def register_surrogate(
        self,
        name: str,
        model_type: str,
        target_metric: str,
        features: List[str],
        performance: Dict[str, float],
        model_path: Optional[str] = None,
    ) -> str:
        """Register a surrogate model."""
        surrogate_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO surrogates
                (id, name, model_type, target_metric, features,
                 trained_at, performance, model_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    surrogate_id,
                    name,
                    model_type,
                    target_metric,
                    json.dumps(features),
                    time.time(),
                    json.dumps(performance),
                    model_path,
                ),
            )
            conn.commit()

        return surrogate_id

    def list_surrogates(self) -> List[Dict[str, Any]]:
        """List all registered surrogate models."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM surrogates ORDER BY trained_at DESC")
            return [dict(row) for row in cursor]

    def natural_language_query(self, question: str) -> str:
        """
        Answer natural language questions using the knowledge base.
        This is a simplified version - in production would use an LLM.
        """
        # Search for relevant entries
        results = self.search(question, k=5)

        if not results:
            return "No relevant knowledge found."

        # Build answer from top results
        answer_parts = [f"Found {len(results)} relevant entries:"]
        for entry, score in results[:3]:
            answer_parts.append(
                f"\n• [{entry.model_family}] {entry.finding} "
                f"(confidence: {entry.confidence:.0%})"
            )
            answer_parts.append(f"  Details: {entry.details[:200]}...")

        return "\n".join(answer_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
            by_source = dict(
                conn.execute(
                    "SELECT source, COUNT(*) FROM knowledge GROUP BY source"
                ).fetchall()
            )
            by_model = dict(
                conn.execute(
                    "SELECT model_family, COUNT(*) FROM knowledge GROUP BY model_family"
                ).fetchall()
            )
            by_topic = dict(
                conn.execute(
                    "SELECT topic, COUNT(*) FROM knowledge GROUP BY topic"
                ).fetchall()
            )

            exp_total = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            exp_by_model = dict(
                conn.execute(
                    "SELECT model_family, COUNT(*) FROM experiments "
                    "GROUP BY model_family"
                ).fetchall()
            )
            exp_by_task = dict(
                conn.execute(
                    "SELECT task, COUNT(*) FROM experiments GROUP BY task"
                ).fetchall()
            )

        return {
            "total_entries": total,
            "by_source": by_source,
            "by_model_family": by_model,
            "by_topic": by_topic,
            "total_experiments": exp_total,
            "experiments_by_model": exp_by_model,
            "experiments_by_task": exp_by_task,
            "vector_index_size": len(self.vector_ids) if self.vector_index else 0,
            "has_embeddings": self.embedding_model is not None,
        }

    def export_json(self, path: str) -> None:
        """Export knowledge base to JSON."""
        entries = self.query(limit=10000)
        with open(path, "w") as f:
            json.dump([e.to_dict() for e in entries], f, indent=2)

    def close(self) -> None:
        """Close connections."""
        pass

    # ------------------------------------------------------------------
    # Metamodel / Surrogate integration
    # ------------------------------------------------------------------

    def extract_symbolic_rules(
        self,
        target_metric: str = "outcome",
        focus_model: str = "eqprop_mlp",
    ) -> List[str]:
        """
        Extract human-readable symbolic rules from experiment data.

        Uses a decision tree surrogate to find interpretable decision boundaries
        that predict experiment success/failure.

        Args:
            target_metric: Column/field to predict.
            focus_model: Model family to analyze.

        Returns:
            List of human-readable rule strings.
        """
        try:
            from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.db_path)
            return mm.extract_symbolic_rules(
                target_metric=target_metric, focus_model=focus_model
            )
        except Exception as e:
            logger.warning(f"Symbolic rule extraction failed: {e}")
            return [f"Symbolic analysis unavailable: {e}"]

    def compute_algorithm_similarity(self) -> Dict[str, Dict[str, float]]:
        """
        Compute pairwise similarity between algorithms based on
        their hyperparameter sensitivity fingerprints.

        Returns:
            Dict of model_name -> {other_model: similarity_score}
        """
        try:
            from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel

            mm = KnowledgebaseMetamodel()
            mm.fit(self.db_path)
            sim_df = mm.compute_algorithm_similarity()
            if sim_df.empty:
                return {}
            return sim_df.to_dict()
        except Exception as e:
            logger.warning(f"Algorithm similarity failed: {e}")
            return {}

    def train_surrogate(
        self,
        target_metric: str = "val_accuracy",
        model_type: str = "rf",
        experiment_ids: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Train a surrogate model to predict experiment outcomes.

        Args:
            target_metric: Metric to predict.
            model_type: 'rf' (random forest), 'gp' (Gaussian process), or 'nn'.
            experiment_ids: Optional subset of experiments to use.

        Returns:
            Surrogate model ID if successful, None otherwise.
        """
        try:
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor

            exps = self.list_experiments(limit=500)
            if not exps or len(exps) < 10:
                logger.warning("Not enough experiments to train surrogate")
                return None

            records = []
            for exp in exps:
                config = json.loads(exp.get("config", "{}"))
                metrics = json.loads(exp.get("metrics", "{}"))
                record = {
                    "lr": config.get("lr", 0.001),
                    "batch_size": config.get("batch_size", 64),
                    "hidden_dim": config.get("hidden_dim", 256),
                    "num_layers": config.get("num_layers", 2),
                    "epochs": config.get("epochs", 10),
                }
                if target_metric in metrics:
                    record["target"] = metrics[target_metric]
                    records.append(record)

            if len(records) < 10:
                logger.warning(f"Not enough records with {target_metric}")
                return None

            df = pd.DataFrame(records)
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            y = df["target"].values

            model = RandomForestRegressor(n_estimators=100, max_depth=5)
            model.fit(X, y)

            score = model.score(X, y)
            surrogate_id = self.register_surrogate(
                name=f"surrogate_{target_metric}",
                model_type=model_type,
                target_metric=target_metric,
                features=feature_cols,
                performance={"r2": float(score), "n_samples": len(records)},
            )
            logger.info(f"Trained surrogate {surrogate_id} with R2={score:.4f}")
            return surrogate_id

        except Exception as e:
            logger.warning(f"Surrogate training failed: {e}")
            return None


# Factory function
def create_knowledge_base(
    db_path: str = "bioplausible_kb.db", **kwargs
) -> KnowledgeBase:
    """Create a KnowledgeBase instance."""
    return KnowledgeBase(db_path=db_path, **kwargs)


# Backward compatibility
class LegacyKnowledgeBase:
    """Backward compatible wrapper for old JSON-based KnowledgeBase."""

    def __init__(
        self, storage_path: str = "knowledgebase.json", load_seed: bool = False
    ):
        self.storage_path = storage_path
        self.kb = KnowledgeBase(db_path=storage_path.replace(".json", ".db"))
        if load_seed:
            self.kb._load_seed_data()

    @property
    def findings(self) -> List[Dict[str, Any]]:
        entries = self.kb.query(limit=10000)
        return [e.to_dict() for e in entries]

    def add_finding(
        self,
        topic: str,
        model_family: str,
        finding: str,
        details: str,
        confidence: float,
        tags: List[str],
    ) -> str:
        entry = KnowledgeEntry(
            id=f"KB-{len(self.findings) + 1:03d}",
            topic=topic,
            model_family=model_family,
            finding=finding,
            details=details,
            confidence=confidence,
            tags=tags,
        )
        return self.kb.add_entry(entry)

    def query(self, tag: str = None, model_family: str = None) -> List[Dict[str, Any]]:
        entries = self.kb.query(tag=tag, model_family=model_family)
        return [e.to_dict() for e in entries]


# Default instance
DEFAULT_KB = KnowledgeBase()


__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "LegacyKnowledgeBase",
    "create_knowledge_base",
    "DEFAULT_KB",
]
