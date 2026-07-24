"""
OmegaConf-based configuration schemas for the Bioplausible platform.

Replaces the legacy config_schema.py and config_loader.py with
a unified, validated configuration system.
"""

from __future__ import annotations

# Register custom resolvers for date interpolation
import time
from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))
except Exception:
    pass  # Already registered


@dataclass
class ModelConfig:
    """Configuration for a model component."""

    name: str = "MLP"
    kwargs: dict[str, Any] = field(default_factory=dict)
    compile: bool = False
    compile_mode: str = "reduce-overhead"


@dataclass
class PropagatorConfig:
    """Configuration for a propagator/learning rule component."""

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Configuration for an optimizer component."""

    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class SparsityConfig:
    """Configuration for a sparsity component."""

    name: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for datasets."""

    name: str = "mnist"
    batch_size: int = 64
    val_batch_size: int | None = None
    num_workers: int = 4
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 10
    batches_per_epoch: int | None = None
    val_batches: int | None = None
    grad_clip: float | None = 1.0
    precision: str = "32-true"
    log_every_n_steps: int = 10
    log_dir: str = "logs"
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    save_best_only: bool = False
    early_stopping_patience: int | None = None
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"


@dataclass
class LightningConfig:
    """Configuration for PyTorch Lightning integration."""

    use_lightning: bool = False
    precision: str = "32-true"
    accelerator: str = "auto"
    devices: int = 1
    num_nodes: int = 1
    strategy: str = "auto"


@dataclass
class DomainConfig:
    """Configuration for domain-specific settings."""

    domain: str = "vision"
    task_specific: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientistConfig:
    """Configuration for the Scientist / AutoScientist."""

    mode: str = "autonomous"
    max_trials: int = 100
    task_filter: str | None = None
    tier_limit: str | None = None
    num_workers: int = 1
    report_interval: int = 50
    human_approval_gate: bool = False
    knowledge_base_path: str = "bioplausible_kb.db"
    llm_backend: str | None = None


@dataclass
class ExperimentConfig:
    """
    Top-level experiment configuration.

    Usage:
        config = ExperimentConfig(
            model=ModelConfig(name="equitile"),
            optimizer=OptimizerConfig(name="smep", lr=0.01),
            dataset=DatasetConfig(name="mnist", batch_size=128),
            trainer=TrainerConfig(epochs=20),
        )
        cfg = OmegaConf.structured(config)
        OmegaConf.save(cfg, "config.yaml")
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    propagator: PropagatorConfig = field(default_factory=PropagatorConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sparsity: SparsityConfig = field(default_factory=SparsityConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    trainer: TrainingConfig = field(default_factory=TrainingConfig)
    lightning: LightningConfig = field(default_factory=LightningConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    scientist: ScientistConfig = field(default_factory=ScientistConfig)
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"
    tags: dict[str, Any] = field(default_factory=dict)
    track_energy: bool = True
    track_flops: bool = True
    track_memory: bool = True
    use_wandb: bool = False
    wandb_project: str | None = None
    deterministic: bool = False


def get_default_config() -> ExperimentConfig:
    """Get the default experiment configuration."""
    return ExperimentConfig()


def validate_config(cfg: Any) -> ExperimentConfig:
    """
    Validate and convert a configuration to ExperimentConfig.

    Args:
        cfg: Dict, OmegaConf DictConfig, or ExperimentConfig.

    Returns:
        Validated ExperimentConfig.
    """
    if isinstance(cfg, ExperimentConfig):
        return cfg
    if isinstance(cfg, dict):
        return OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(ExperimentConfig),
                OmegaConf.create(cfg),
            )
        )
    # OmegaConf DictConfig
    return OmegaConf.to_object(
        OmegaConf.merge(
            OmegaConf.structured(ExperimentConfig),
            cfg,
        )
    )


# ──────────────────────────────────────────────
# Merged from config_schema.py (legacy RunConfig types)
# ──────────────────────────────────────────────

from dataclasses import dataclass, field

from omegaconf import MISSING, OmegaConf

try:
    OmegaConf.register_new_resolver("now", lambda fmt: time.strftime(fmt))
except Exception:
    pass


@dataclass
class RunConfigData:
    task: str = MISSING
    batch_size: int = 64
    seq_len: int = 64
    augment: bool = False
    data_fraction: float = 1.0


@dataclass
class RunConfigModel:
    name: str = MISSING
    hidden_dim: int = 256
    num_layers: int = 3
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfigOptimizer:
    name: str = "adam"
    lr: float = 0.001
    weight_decay: float = 0.0
    beta: float = 0.5
    settle_steps: int = 30
    mode: str = "ep"


@dataclass
class RunConfigTrainer:
    epochs: int = 10
    batches_per_epoch: int = 100
    grad_clip: float | None = None
    scheduler: str | None = None
    use_compile: bool = True
    track_energy: bool = True


@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"
    output_dir: str = "results/${now:%Y%m%d_%H%M%S}"
    data: RunConfigData = field(default_factory=RunConfigData)
    model: RunConfigModel = field(default_factory=RunConfigModel)
    optimizer: RunConfigOptimizer = field(default_factory=RunConfigOptimizer)
    trainer: RunConfigTrainer = field(default_factory=RunConfigTrainer)
    ablation_tags: dict[str, Any] = field(default_factory=dict)
