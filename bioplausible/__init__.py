"""
Bioplausible: Unified Platform for Bio-Plausible Learning Research

Minimal, clean API for training and experimentation.

Quick Start:
    from bioplausible import CoreTrainer, TrainerConfig

    config = TrainerConfig(
        model="equitile",
        model_kwargs={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        optimizer="smep",
        optimizer_kwargs={"lr": 0.01},
        task="mnist",
        epochs=10
    )
    trainer = CoreTrainer(config)
    history = trainer.fit()

Or from YAML:
    trainer = CoreTrainer.from_yaml("config.yaml")
    history = trainer.fit()
"""

# AutoScientist (LLM meta-reasoner)
from bioplausible.autoscientist import (
    AutoScientistBridge,
    AutoScientistCampaign,
    ExperimentProposal,
    ExperimentProposer,
    Hypothesis,
    HypothesisReasoner,
    LLMHypothesisGenerator,
)

# Config
from bioplausible.config import (
    DEFAULT_CONFIGS,
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    PropagatorConfig,
    ScientistConfig,
    SparsityConfig,
    TrainingConfig,
    get_default_config,
    validate_config,
)
from bioplausible.core.registry import (
    ComponentCategory,
    ComponentMetadata,
    ComputeProfile,
    Domain,
    LocalityLevel,
    Registry,
    register_callback,
    register_data_loader,
    register_domain,
    register_metric,
    register_model,
    register_optimizer,
    register_propagator,
    register_sparsity,
    register_task,
)
from bioplausible.core.trainer import (
    CoreTrainer,
    TrainerConfig,
    TrainingMetrics,
    run_from_config,
)

# Data
from bioplausible.datasets import (
    create_data_loaders,
    get_lm_dataset,
    get_vision_dataset,
)

# Domains
from bioplausible.domains import (
    Batch,
    DomainSpec,
    DomainTask,
    DomainType,
    GraphTask,
    LMTask,
    Metrics,
    RLTask,
    ScientificTask,
    TabularTask,
    TaskSplit,
    TimeSeriesTask,
    VisionTask,
    create_domain_task,
    list_domains,
)

# Evaluation
from bioplausible.evaluation import (
    BenchmarkRegistry,
    BenchmarkResult,
    BenchmarkSuiteConfig,
    BenchmarkSuiteResult,
    CrossDomainBenchmarkSuite,
    EvaluatorBase,
    MetricSuite,
    evaluate_model_on_task,
    get_benchmark,
    list_benchmarks,
    run_cross_domain_benchmark,
)

# Knowledge Base
from bioplausible.knowledge import (
    DEFAULT_KB,
    KnowledgeBase,
    KnowledgeEntry,
    create_knowledge_base,
)

# Leaderboard
from bioplausible.leaderboard.generator import LeaderboardEntry, LeaderboardGenerator

# Lightning Integration
from bioplausible.lightning_ import (
    BioLightningModule,
    BioOptunaPruner,
    BioPrecisionCallback,
    BioPrecisionMixin,
    BioPredictionWriter,
    BioRayTuneSearch,
    EnergyConvergenceCallback,
    build_trainer,
    run_nas_search,
    run_pl_trial,
    run_pl_trial_with_wandb,
)

# Models (legacy + new zoo)
from bioplausible.models import (
    BackpropMLP,
    ConvEqProp,
    LoopedMLP,
    MemoryEfficientLoopedMLP,
    TransformerEqProp,
    create_model,
    list_models,
)
from bioplausible.models.registry import list_model_specs

# Optimizers
from bioplausible.optimizers import (
    SGD,
    Adam,
    AdamW,
    DirectFA,
    EqProp,
    FeedbackAlignment,
    create_optimizer,
    list_optimizers,
    smep,
    smep_fast,
)

# Scientist (execution engine)
from bioplausible.scientist.core import AutoScientist, Scientist
from bioplausible.scientist.task import ExperimentTask

# Training (legacy)
from bioplausible.training.supervised import SupervisedTrainer

# Utilities
from bioplausible.utils import count_parameters

# Zoo
from bioplausible.zoo import models as zoo_models
from bioplausible.zoo import optimizers as zoo_optimizers
from bioplausible.zoo import propagators as zoo_propagators
from bioplausible.zoo import sparsity as zoo_sparsity

__version__ = "0.5.0"

__all__ = [
    # Core
    "CoreTrainer",
    "TrainerConfig",
    "TrainingMetrics",
    "run_from_config",
    # Registry
    "Registry",
    "ComponentCategory",
    "Domain",
    "LocalityLevel",
    "ComputeProfile",
    "ComponentMetadata",
    "register_model",
    "register_propagator",
    "register_optimizer",
    "register_sparsity",
    "register_metric",
    "register_data_loader",
    "register_task",
    "register_callback",
    "register_domain",
    # Domains
    "DomainTask",
    "DomainType",
    "DomainSpec",
    "TaskSplit",
    "Batch",
    "Metrics",
    "VisionTask",
    "LMTask",
    "RLTask",
    "GraphTask",
    "TabularTask",
    "TimeSeriesTask",
    "ScientificTask",
    "create_domain_task",
    "list_domains",
    # Evaluation
    "EvaluatorBase",
    "MetricSuite",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "evaluate_model_on_task",
    "get_benchmark",
    "list_benchmarks",
    "BenchmarkSuiteConfig",
    "BenchmarkSuiteResult",
    "CrossDomainBenchmarkSuite",
    "run_cross_domain_benchmark",
    # Models (Legacy)
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    "create_model",
    "list_models",
    "list_model_specs",
    # Zoo
    "zoo_models",
    "zoo_propagators",
    "zoo_optimizers",
    "zoo_sparsity",
    # Optimizers
    "FeedbackAlignment",
    "DirectFA",
    "EqProp",
    "smep",
    "smep_fast",
    "SGD",
    "Adam",
    "AdamW",
    "create_optimizer",
    "list_optimizers",
    # Training
    "SupervisedTrainer",
    # Knowledge Base
    "KnowledgeBase",
    "KnowledgeEntry",
    "create_knowledge_base",
    "DEFAULT_KB",
    # Data
    "get_vision_dataset",
    "get_lm_dataset",
    "create_data_loaders",
    # Lightning
    "BioLightningModule",
    "BioOptunaPruner",
    "BioRayTuneSearch",
    "BioPrecisionCallback",
    "EnergyConvergenceCallback",
    "BioPredictionWriter",
    "run_pl_trial",
    "run_pl_trial_with_wandb",
    "run_nas_search",
    "build_trainer",
    "BioPrecisionMixin",
    # AutoScientist (LLM meta-reasoner)
    "AutoScientistBridge",
    "AutoScientistCampaign",
    "ExperimentProposal",
    "ExperimentProposer",
    "Hypothesis",
    "HypothesisReasoner",
    "LLMHypothesisGenerator",
    # Scientist (execution engine)
    "Scientist",
    "AutoScientist",  # Alias for backward compatibility
    "ExperimentTask",
    # Config
    "DEFAULT_CONFIGS",
    "ExperimentConfig",
    "get_default_config",
    "validate_config",
    "ModelConfig",
    "OptimizerConfig",
    "PropagatorConfig",
    "TrainingConfig",
    "DatasetConfig",
    "SparsityConfig",
    "ScientistConfig",
    # Leaderboard
    "LeaderboardEntry",
    "LeaderboardGenerator",
    # Utilities
    "count_parameters",
]
