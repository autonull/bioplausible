"""
EquiTile Fast LM Demo
=====================

High-performance language modeling demo showcasing EquiTile's unique advantages:
- Mixture of Tiles (MoT) for conditional computation
- Tile-local attention for O(n) complexity
- Parameter efficiency (< 10M parameters)
- Fast training on commodity GPUs

Quick Start
-----------
>>> from bioplausible.equitile.lm_demo import FastLMEquiTile, FastLMConfig
>>> config = FastLMConfig(vocab_size=1000, embed_dim=192, num_layers=6)
>>> model = FastLMEquiTile(config)
>>> logits = model(input_ids)

Demo Usage
----------
$ python -m bioplausible.models.equitile.lm_demo.demo --task shakespeare --epochs 5
"""

from .data import CharacterTokenizer
from .data import LMDataset
from .data import Tokenizer
from .data import create_python_dataset
from .data import create_shakespeare_dataset
from .data import create_tinystories_dataset
from .data_advanced import BPETokenizer
from .data_advanced import WordPieceTokenizer
from .data_advanced import create_tokenizer
from .data_advanced import load_shakespeare_tokenizer
from .fast_lm import FastEquiTileLayer
from .fast_lm import FastLMConfig
from .fast_lm import FastLMEquiTile
from .fast_lm import MixtureOfTiles
from .fast_lm import SwiGLUFeedForward
from .fast_lm import TileLocalAttention
from .profiling import BandwidthAnalyzer
from .profiling import MemoryProfiler
from .profiling import MemorySnapshot
from .profiling import ProfileResult
from .profiling import profile_memory
from .training import LMTrainer
from .training import TrainingConfig
from .training import TrainingMetrics
from .training import train_model

__all__ = [
    # Model
    "FastLMEquiTile",
    "FastLMConfig",
    "MixtureOfTiles",
    "TileLocalAttention",
    "SwiGLUFeedForward",
    "FastEquiTileLayer",
    # Data
    "LMDataset",
    "create_shakespeare_dataset",
    "create_tinystories_dataset",
    "create_python_dataset",
    "Tokenizer",
    "CharacterTokenizer",
    # Advanced Tokenizers
    "BPETokenizer",
    "WordPieceTokenizer",
    "create_tokenizer",
    "load_shakespeare_tokenizer",
    # Training
    "LMTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "train_model",
    # Profiling
    "MemoryProfiler",
    "BandwidthAnalyzer",
    "profile_memory",
    "MemorySnapshot",
    "ProfileResult",
]
