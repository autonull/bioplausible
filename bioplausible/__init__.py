"""
Bioplausible: Bio-Plausible Learning Algorithms

Minimal, clean API.

Quick Start:
    from bioplausible import create_model, create_optimizer
    
    model = create_model('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = create_optimizer(model, 'smep')
"""

# Models
from bioplausible.models import (
    LoopedMLP,
    BackpropMLP,
    ConvEqProp,
    MemoryEfficientLoopedMLP,
    TransformerEqProp,
    create_model,
    list_models,
)
from bioplausible.models.registry import list_model_specs

# Optimizers
from bioplausible.optimizers import (
    FeedbackAlignment,
    DirectFA,
    EqProp,
    smep,
    smep_fast,
    SGD,
    Adam,
    AdamW,
    create_optimizer,
    list_optimizers,
)

# Training
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.core import EqPropTrainer

# Data
from bioplausible.datasets import get_vision_dataset, get_lm_dataset, create_data_loaders

# Utilities
from bioplausible.utils import count_parameters

__version__ = "0.3.0"

__all__ = [
    # Simplest API
    'create_model',
    'create_optimizer',
    'list_models',
    'list_model_specs',
    'list_optimizers',
    # Models
    'LoopedMLP',
    'BackpropMLP',
    'ConvEqProp',
    'MemoryEfficientLoopedMLP',
    'TransformerEqProp',
    # Optimizers
    'FeedbackAlignment',
    'DirectFA',
    'EqProp',
    'smep',
    'smep_fast',
    'SGD',
    'Adam',
    'AdamW',
    # Training
    'SupervisedTrainer',
    'EqPropTrainer',
    # Data
    'get_vision_dataset',
    'get_lm_dataset',
    'create_data_loaders',
    # Utilities
    'count_parameters',
]
