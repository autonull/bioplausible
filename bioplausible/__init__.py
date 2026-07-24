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
from bioplausible.autoscientist import AutoScientistBridge
from bioplausible.autoscientist import AutoScientistCampaign
from bioplausible.autoscientist import ExperimentProposal
from bioplausible.autoscientist import ExperimentProposer
from bioplausible.autoscientist import Hypothesis
from bioplausible.autoscientist import HypothesisReasoner
from bioplausible.autoscientist import LLMHypothesisGenerator

# Config
from bioplausible.config import DEFAULT_CONFIGS
from bioplausible.config import DatasetConfig
from bioplausible.config import ExperimentConfig
from bioplausible.config import ModelConfig
from bioplausible.config import OptimizerConfig
from bioplausible.config import PropagatorConfig
from bioplausible.config import ScientistConfig
from bioplausible.config import SparsityConfig
from bioplausible.config import TrainingConfig
from bioplausible.config import get_default_config
from bioplausible.config import validate_config
from bioplausible.core.registry import ComponentCategory
from bioplausible.core.registry import ComponentMetadata
from bioplausible.core.registry import ComputeProfile
from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import Registry
from bioplausible.core.registry import list_models
from bioplausible.core.registry import register_callback
from bioplausible.core.registry import register_data_loader
from bioplausible.core.registry import register_domain
from bioplausible.core.registry import register_metric
from bioplausible.core.registry import register_model
from bioplausible.core.registry import register_optimizer
from bioplausible.core.registry import register_propagator
from bioplausible.core.registry import register_sparsity
from bioplausible.core.registry import register_task
from bioplausible.core.trainer import CoreTrainer
from bioplausible.core.trainer import TrainerConfig
from bioplausible.core.trainer import TrainingMetrics
from bioplausible.core.trainer import run_from_config

# Data
from bioplausible.datasets import create_data_loaders
from bioplausible.datasets import get_lm_dataset
from bioplausible.datasets import get_vision_dataset

# Domains
from bioplausible.domains import Batch
from bioplausible.domains import DomainSpec
from bioplausible.domains import DomainTask
from bioplausible.domains import DomainType
from bioplausible.domains import GraphTask
from bioplausible.domains import LMTask
from bioplausible.domains import Metrics
from bioplausible.domains import RLTask
from bioplausible.domains import ScientificTask
from bioplausible.domains import TabularTask
from bioplausible.domains import TaskSplit
from bioplausible.domains import TimeSeriesTask
from bioplausible.domains import VisionTask
from bioplausible.domains import create_domain_task
from bioplausible.domains import list_domains

# Evaluation
from bioplausible.evaluation import BenchmarkRegistry
from bioplausible.evaluation import BenchmarkResult
from bioplausible.evaluation import BenchmarkSuiteConfig
from bioplausible.evaluation import BenchmarkSuiteResult
from bioplausible.evaluation import CrossDomainBenchmarkSuite
from bioplausible.evaluation import EvaluatorBase
from bioplausible.evaluation import MetricSuite
from bioplausible.evaluation import evaluate_model_on_task
from bioplausible.evaluation import get_benchmark
from bioplausible.evaluation import list_benchmarks
from bioplausible.evaluation import run_cross_domain_benchmark

# Scientist (execution engine) - now in execution
from bioplausible.execution.engine import ExecutionEngine
from bioplausible.execution.task import ExperimentTask

# EquiTile top-level package — importing registers all variants
from bioplausible.equitile import EquiTile as _EquiTile  # noqa: F401

# Knowledge Base
from bioplausible.knowledge import DEFAULT_KB
from bioplausible.knowledge import KnowledgeBase
from bioplausible.knowledge import KnowledgeEntry
from bioplausible.knowledge import create_knowledge_base

# Leaderboard
from bioplausible.leaderboard.generator import LeaderboardEntry
from bioplausible.leaderboard.generator import LeaderboardGenerator

# Lightning Integration
from bioplausible.lightning_ import BioLightningModule
from bioplausible.lightning_ import BioOptunaPruner
from bioplausible.lightning_ import BioPrecisionCallback
from bioplausible.lightning_ import BioPrecisionMixin
from bioplausible.lightning_ import BioPredictionWriter
from bioplausible.lightning_ import BioRayTuneSearch
from bioplausible.lightning_ import EnergyConvergenceCallback
from bioplausible.lightning_ import build_trainer
from bioplausible.lightning_ import run_nas_search
from bioplausible.lightning_ import run_pl_trial
from bioplausible.lightning_ import run_pl_trial_with_wandb

# Optimizers / Propagators
from bioplausible.zoo.mep.presets import muon_backprop
from bioplausible.zoo.mep.presets import smep
from bioplausible.zoo.mep.presets import smep_fast

# Models (legacy + new zoo)
from bioplausible.zoo.models.eqprop import BackpropMLP
from bioplausible.zoo.models.eqprop import ConvEqProp
from bioplausible.zoo.models.eqprop import LoopedMLP
from bioplausible.zoo.models.eqprop import MemoryEfficientLoopedMLP
from bioplausible.zoo.models.eqprop import TransformerEqProp
from bioplausible.zoo.propagators.eqprop import EqProp
from bioplausible.zoo.propagators.fa import DirectFA
from bioplausible.zoo.propagators.fa import FeedbackAlignment

# Utilities
from bioplausible.utils import count_parameters

# Zoo
from bioplausible.zoo import models as zoo_models
from bioplausible.zoo import optimizers as zoo_optimizers
from bioplausible.zoo import propagators as zoo_propagators
from bioplausible.zoo import sparsity as zoo_sparsity

__version__ = "1.0.0"

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
    "list_models",
    # Zoo
    "zoo_models",
    "zoo_propagators",
    "zoo_optimizers",
    "zoo_sparsity",
    # Optimizers / Propagators
    "FeedbackAlignment",
    "DirectFA",
    "EqProp",
    "smep",
    "smep_fast",
    "muon_backprop",
    # Execution Engine
    "ExecutionEngine",
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
