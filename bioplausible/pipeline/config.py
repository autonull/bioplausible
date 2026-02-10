from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainingConfig:
    """Complete JSON-serializable training config."""

    task: str  # "vision", "lm", "rl", "diffusion"
    dataset: str
    model: str
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    gradient_method: str = "BPTT (Standard)"
    use_compile: bool = True
    use_kernel: bool = False
    monitor_dynamics: bool = False
    hyperparams: Dict[str, Any] = field(default_factory=dict)
