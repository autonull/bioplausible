"""Core data structures used across the framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class Node:
    """
    One evolution node stored in the database.

    Attributes:
        name: Node name.
        created_at: ISO timestamp.
        parent: Parent node ids.
        motivation: Natural-language rationale for the candidate.
        code: Generated program.
        results: Structured evaluation results.
        analysis: Analyzer summary.
        meta_info: Auxiliary metadata.
        id: Unique identifier assigned by the database.
        visit_count: Number of times the node has been sampled.
        score: Scalar score used for ranking and sampling.
    """

    name: str = ""
    created_at: str = ""
    parent: List[int] = field(default_factory=list)
    motivation: str = ""
    code: str = ""
    results: Dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    meta_info: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None
    visit_count: int = 0
    score: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the node to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "parent": self.parent,
            "motivation": self.motivation,
            "code": self.code,
            "results": self.results,
            "analysis": self.analysis,
            "meta_info": self.meta_info,
            "visit_count": self.visit_count,
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create a node from a dictionary."""
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            created_at=data.get("created_at", ""),
            parent=data.get("parent", []),
            motivation=data.get("motivation", ""),
            code=data.get("code", ""),
            results=data.get("results", {}),
            analysis=data.get("analysis", ""),
            meta_info=data.get("meta_info", {}),
            visit_count=data.get("visit_count", 0),
            score=data.get("score", 0.0),
        )

    def get_context_text(self) -> str:
        """Return the text used for similarity search."""
        parts = [self.name, self.motivation, self.analysis]
        return " ".join(p for p in parts if p)


@dataclass
class CognitionItem:
    """
    One item stored in the cognition base.

    Attributes:
        id: Unique identifier.
        content: Stored text.
        source: Optional provenance metadata.
        metadata: Extra attributes.
    """

    content: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CognitionItem":
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExperimentConfig:
    """Resolved experiment configuration."""

    name: str
    experiment_dir: Path = None
    input_file: Optional[str] = None
    eval_script: Optional[str] = None
    run_script: Optional[str] = None

    def __post_init__(self):
        from pathlib import Path
        if self.experiment_dir is None:
            base_dir = Path(__file__).parent.parent / "experiments"
            self.experiment_dir = base_dir / self.name
        elif isinstance(self.experiment_dir, str):
            self.experiment_dir = Path(self.experiment_dir)


@dataclass
class LLMResponse:
    """Structured LLM response payload."""

    content: str
    raw_response: Any = None
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    call_time: float = 0.0
