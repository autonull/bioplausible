"""
EquiTile: Scalable Local-Learning Architecture
==============================================

A production-ready, tile-based local learning framework featuring:
- Tile-based parallel architecture
- Local Hebbian weight updates (no global backprop)
- Multi-GPU support with NCCL
- Mixed precision training
- Dynamic tile growth/pruning
- Enhanced EP with LayerNorm and curriculum learning
- Async execution support
- Comprehensive profiling and benchmarking
- Research utilities for experiments

Quick Start
-----------
>>> from bioplausible.models.equitile import EquiTile
>>> model = EquiTile(
...     neurons_per_tile=64,
...     num_layers=4,
...     tiles_per_layer=4,
...     input_dim=784,
...     output_dim=10,
... )
>>> for X, y in dataloader:
...     stats = model.train_step(X, y)

Modules
-------
core : Core EquiTile implementation
config : Configuration classes
enhanced : Enhanced EP features
dynamics : Tile growth/pruning
async_execution : Async tile processing
multigpu : Multi-GPU training
distributed : Distributed training
profiler : Performance profiling
builder : Fluent builder API
research : Research utilities
vision : Vision (ConvEquiTile)
language : Language modeling (LMEquiTile)
rl : Reinforcement learning (RLEquiTile)
graph : Graph neural networks (GraphEquiTile)
timeseries : Time series modeling
deployment : Model export and optimization

Examples
--------
Basic usage:
>>> from bioplausible.models.equitile import EquiTile, create_production_config
>>> config = create_production_config()
>>> model = EquiTile(
...     neurons_per_tile=config.neurons_per_tile,
...     num_layers=config.num_layers,
...     tiles_per_layer=config.tiles_per_layer,
...     input_dim=784,
...     output_dim=10,
... )

Builder pattern:
>>> from bioplausible.models.equitile.builder import EquiTileBuilder
>>> model = (EquiTileBuilder.production(input_dim=784, output_dim=10)
...     .with_learning_rate(0.01)
...     .build())

Multi-GPU:
>>> from bioplausible.models.equitile import MultiGPUEquiTile, MultiGPUConfig
>>> multi_gpu = MultiGPUEquiTile(model, device_ids=[0, 1, 2, 3])

Async execution:
>>> from bioplausible.models.equitile import AsyncEquiTile, AsyncConfig
>>> async_model = AsyncEquiTile(model, config=AsyncConfig(n_workers=4))
>>> with async_model.async_context():
...     stats = async_model.train_step(X, y)

Profiling:
>>> from bioplausible.models.equitile import EquiTileProfiler
>>> profiler = EquiTileProfiler(model)
>>> with profiler.profile():
...     model.train_step(X, y)
>>> profiler.print_report()

Research utilities:
>>> from bioplausible.models.equitile.research import ExperimentTracker
>>> tracker = ExperimentTracker("my_experiment")
>>> tracker.log_params({"lr": 0.01})
>>> tracker.log_metrics({"loss": 0.5}, step=100)
"""

from .config import (
    EquiTileConfig,
    create_production_config,
    create_research_config,
    create_fast_config,
    create_enhanced_config,
    create_dynamic_config,
    # Distributed configs
    DistributedConfig,
    MultiGPUConfig,
    NCCLConfig,
    AsyncConfig,
    # Enhanced configs
    EnhancedEPConfig,
    CurriculumConfig,
    # Dynamics configs
    TileGrowthConfig,
    DynamicEquiTileConfig,
)

from .core import EquiTile, EquiTileEP
from .topology import TileGraph, TileState

from .enhanced import (
    TileLayerNorm,
    CurriculumConfig as EnhancedCurriculumConfig,
    CurriculumScheduler,
    EnhancedEPConfig as EnhancedEPConfigClass,
    EnhancedEquiTile,
    create_enhanced_model,
)

from .dynamics import (
    TileGrowthConfig as DynamicsTileGrowthConfig,
    TileMetrics,
    TileGrowthManager,
    DynamicEquiTileConfig as DynamicsConfig,
    DynamicEquiTile,
    create_dynamic_model,
)

# Async execution
from .async_execution import (
    TileTask,
    TileResult,
    TileProcessor,
    TileScheduler,
    AsyncConfig as AsyncExecutionConfig,
    AsyncEquiTile,
    create_async_model,
)

# Multi-GPU
from .multigpu import (
    NCCLConfig as NCCLConfigClass,
    NCCLCommunicator,
    MultiGPUConfig as MultiGPUConfigClass,
    MultiGPUEquiTile,
    AsyncTileExecutor,
    spawn_multi_gpu_worker,
    create_multigpu_model,
)

# Distributed
from .distributed import (
    DeviceAssignment,
    DistributedConfig as DistributedConfigClass,
    TileCommunicator,
    MixedPrecisionTrainer,
    TileGrowthConfig as DistributedGrowthConfig,
    DistributedEquiTile,
    create_distributed_model,
)

# Profiler
from .profiler import (
    TileStats,
    ProfileResult,
    EquiTileProfiler,
    LearningMonitor,
    MemoryProfiler,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkRunner,
    create_profiler,
    run_benchmark,
)

# Builder
from .builder import (
    EquiTileBuilder,
    EnhancedEquiTileBuilder,
    TrainingContext,
    InferenceContext,
    build_model,
    build_enhanced_model,
)

# Research utilities
from .research import (
    ExperimentConfig,
    ExperimentTracker,
    MetricEntry,
    MetricCollector,
    VisualizationHelper,
    AblationConfig,
    AblationStudy,
    create_tracker,
    create_metric_collector,
    create_visualization_helper,
    create_ablation_study,
)

# Domain-specific modules
from .vision import (
    ConvEquiTile,
    ConvEquiTileConfig,
    ConvFeatureExtractor,
    VisionAugmentation,
    create_vision_model,
    create_mnist_model,
    create_cifar_model,
    create_imagenet_model,
)

from .language import (
    LMEquiTile,
    LMEquiTileConfig,
    PositionalEncoding,
    TileAttention,
    TileFeedForward,
    EquiTileTransformerLayer,
    SimpleTokenizer,
    create_lm_model,
    create_small_lm,
    create_medium_lm,
    create_large_lm,
)

# Optimized Language Model
from .language_optimized import (
    OptimizedLMEquiTile,
    OptimizedTileAttention,
    OptimizedTileFeedForward,
    OptimizedEquiTileTransformerLayer,
    create_optimized_lm,
    create_optimized_small_lm,
)

from .rl import (
    RLEquiTile,
    RLEquiTileConfig,
    RecurrentRLEquiTile,
    RolloutBuffer,
    compute_gae,
    create_rl_model,
    create_recurrent_rl_model,
    create_atari_model,
    create_mujoco_model,
)

# Graph Neural Networks
from .graph import (
    GraphEquiTile,
    GraphEquiTileConfig,
    GraphAttentionLayer,
    GraphEquiTileLayer,
    aggregate_messages,
    scatter_mean,
    scatter_sum,
    scatter_max,
    create_graph_model,
    create_molecule_model,
    create_social_graph_model,
)

# Time Series
from .timeseries import (
    TimeSeriesEquiTile,
    TimeSeriesConfig,
    TemporalPositionalEncoding,
    TemporalAttentionLayer,
    TimeSeriesEquiTileLayer,
    create_forecasting_model,
    create_classification_model,
    create_anomaly_detection_model,
)

# Deployment
from .deployment import (
    EquiTileExporter,
    ExportConfig,
    ModelPruner,
    DeploymentChecker,
    export_model,
    quantize_model,
    prune_model,
    check_deployment,
)

__all__ = [
    # Core
    "EquiTile",
    "EquiTileEP",
    "TileGraph",
    "TileState",
    # "EdgeParams",  # Removed

    # Config
    "EquiTileConfig",
    "create_production_config",
    "create_research_config",
    "create_fast_config",
    "create_enhanced_config",
    "create_dynamic_config",

    # Distributed configs
    "DistributedConfig",
    "MultiGPUConfig",
    "NCCLConfig",
    "AsyncConfig",

    # Enhanced configs
    "EnhancedEPConfig",
    "CurriculumConfig",

    # Dynamics configs
    "TileGrowthConfig",
    "DynamicEquiTileConfig",

    # Enhanced
    "TileLayerNorm",
    "EnhancedCurriculumConfig",
    "CurriculumScheduler",
    "EnhancedEPConfigClass",
    "EnhancedEquiTile",
    "create_enhanced_model",

    # Dynamics
    "DynamicsTileGrowthConfig",
    "TileMetrics",
    "TileGrowthManager",
    "DynamicsConfig",
    "DynamicEquiTile",
    "create_dynamic_model",

    # Async execution
    "TileTask",
    "TileResult",
    "TileProcessor",
    "TileScheduler",
    "AsyncExecutionConfig",
    "AsyncEquiTile",
    "create_async_model",

    # Multi-GPU
    "NCCLConfigClass",
    "NCCLCommunicator",
    "MultiGPUConfigClass",
    "MultiGPUEquiTile",
    "AsyncTileExecutor",
    "spawn_multi_gpu_worker",
    "create_multigpu_model",

    # Distributed
    "DeviceAssignment",
    "DistributedConfigClass",
    "TileCommunicator",
    "MixedPrecisionTrainer",
    "DistributedGrowthConfig",
    "DistributedEquiTile",
    "create_distributed_model",

    # Profiler
    "TileStats",
    "ProfileResult",
    "EquiTileProfiler",
    "LearningMonitor",
    "MemoryProfiler",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRunner",
    "create_profiler",
    "run_benchmark",

    # Builder
    "EquiTileBuilder",
    "EnhancedEquiTileBuilder",
    "TrainingContext",
    "InferenceContext",
    "build_model",
    "build_enhanced_model",

    # Research utilities
    "ExperimentConfig",
    "ExperimentTracker",
    "MetricEntry",
    "MetricCollector",
    "VisualizationHelper",
    "AblationConfig",
    "AblationStudy",
    "create_tracker",
    "create_metric_collector",
    "create_visualization_helper",
    "create_ablation_study",

    # Domain-specific: Vision
    "ConvEquiTile",
    "ConvEquiTileConfig",
    "ConvFeatureExtractor",
    "VisionAugmentation",
    "create_vision_model",
    "create_mnist_model",
    "create_cifar_model",
    "create_imagenet_model",

    # Domain-specific: Language
    "LMEquiTile",
    "LMEquiTileConfig",
    "PositionalEncoding",
    "TileAttention",
    "TileFeedForward",
    "EquiTileTransformerLayer",
    "SimpleTokenizer",
    "create_lm_model",
    "create_small_lm",
    "create_medium_lm",
    "create_large_lm",

    # Optimized Language
    "OptimizedLMEquiTile",
    "OptimizedTileAttention",
    "OptimizedTileFeedForward",
    "OptimizedEquiTileTransformerLayer",
    "create_optimized_lm",
    "create_optimized_small_lm",

    # Domain-specific: RL
    "RLEquiTile",
    "RLEquiTileConfig",
    "RecurrentRLEquiTile",
    "RolloutBuffer",
    "compute_gae",
    "create_rl_model",
    "create_recurrent_rl_model",
    "create_atari_model",
    "create_mujoco_model",

    # Domain-specific: Graph
    "GraphEquiTile",
    "GraphEquiTileConfig",
    "GraphAttentionLayer",
    "GraphEquiTileLayer",
    "aggregate_messages",
    "scatter_mean",
    "scatter_sum",
    "scatter_max",
    "create_graph_model",
    "create_molecule_model",
    "create_social_graph_model",

    # Domain-specific: Time Series
    "TimeSeriesEquiTile",
    "TimeSeriesConfig",
    "TemporalPositionalEncoding",
    "TemporalAttentionLayer",
    "TimeSeriesEquiTileLayer",
    "create_forecasting_model",
    "create_classification_model",
    "create_anomaly_detection_model",

    # Deployment
    "EquiTileExporter",
    "ExportConfig",
    "ModelPruner",
    "DeploymentChecker",
    "export_model",
    "quantize_model",
    "prune_model",
    "check_deployment",
]

__version__ = "1.0.0"
