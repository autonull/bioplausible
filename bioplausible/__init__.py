"""
Bioplausible: Bio-Plausible Learning Algorithms

Minimal, clean API.

Quick Start:
    from bioplausible import create_model, create_optimizer

    model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = create_optimizer(model, 'smep')
"""

from bioplausible.core import EqPropTrainer

# Data
from bioplausible.datasets import (
    create_data_loaders,
    get_lm_dataset,
    get_vision_dataset,
)

# Models
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

# Training
from bioplausible.training.supervised import SupervisedTrainer

# Utilities
from bioplausible.utils import count_parameters

# PyTorch Lightning Integration
from bioplausible.lightning_ import (
    BioLightningModule,
    BioOptunaPruner,
    BioRayTuneSearch,
    BioPrecisionCallback,
    EnergyConvergenceCallback,
    BioPredictionWriter,
    run_pl_trial,
    run_pl_trial_with_wandb,
    run_nas_search,
    build_trainer,
    BioPrecisionMixin,
)

__version__ = "0.3.0"

__all__ = [
    # Simplest API
    "create_model",
    "create_optimizer",
    "list_models",
    "list_model_specs",
    "list_optimizers",
    # Models
    "LoopedMLP",
    "BackpropMLP",
    "ConvEqProp",
    "MemoryEfficientLoopedMLP",
    "TransformerEqProp",
    # Optimizers
    "FeedbackAlignment",
    "DirectFA",
    "EqProp",
    "smep",
    "smep_fast",
    "SGD",
    "Adam",
    "AdamW",
    # Training
    "SupervisedTrainer",
    "EqPropTrainer",
    # Data
    "get_vision_dataset",
    "get_lm_dataset",
    "create_data_loaders",
    # Utilities
    "count_parameters",
    # Lightning integration
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
]