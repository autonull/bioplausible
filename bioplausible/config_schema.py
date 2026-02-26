from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from omegaconf import MISSING

@dataclass
class RunConfigData:
    task: str = MISSING                # "mnist", "cifar10", "shakespeare", "cartpole", "cora"
    batch_size: int = 64
    seq_len: int = 64                  # LM tasks
    augment: bool = False
    data_fraction: float = 1.0         # for data-efficiency curves

@dataclass
class RunConfigModel:
    name: str = MISSING                # registry key
    hidden_dim: int = 256
    num_layers: int = 3
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RunConfigOptimizer:
    name: str = "adam"                 # any key from OPTIMIZER_REGISTRY
    lr: float = 0.001
    weight_decay: float = 0.0
    # MEP-specific
    beta: float = 0.5
    settle_steps: int = 30
    mode: str = "ep"                   # "ep" | "backprop"

@dataclass
class RunConfigTrainer:
    epochs: int = 10
    batches_per_epoch: int = 100
    grad_clip: Optional[float] = None
    scheduler: Optional[str] = None
    use_compile: bool = True
    track_energy: bool = True

@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"                   # "auto" selects cuda if available
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"

    data: RunConfigData = field(default_factory=RunConfigData)
    model: RunConfigModel = field(default_factory=RunConfigModel)
    optimizer: RunConfigOptimizer = field(default_factory=RunConfigOptimizer)
    trainer: RunConfigTrainer = field(default_factory=RunConfigTrainer)
    ablation_tags: Dict[str, Any] = field(default_factory=dict)
