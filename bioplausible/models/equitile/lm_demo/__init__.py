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
>>> from bioplausible.models.equitile.lm_demo import FastLMEquiTile, FastLMConfig
>>> config = FastLMConfig(vocab_size=1000, embed_dim=192, num_layers=6)
>>> model = FastLMEquiTile(config)
>>> logits = model(input_ids)

Demo Usage
----------
$ python -m bioplausible.models.equitile.lm_demo.demo --task shakespeare --epochs 5
"""

from .fast_lm import (
    FastLMEquiTile,
    FastLMConfig,
    MixtureOfTiles,
    TileLocalAttention,
    SwiGLUFeedForward,
    FastEquiTileLayer,
)

from .data import (
    LMDataset,
    create_shakespeare_dataset,
    create_tinystories_dataset,
    create_python_dataset,
    Tokenizer,
    CharacterTokenizer,
)

from .data_advanced import (
    BPETokenizer,
    WordPieceTokenizer,
    create_tokenizer,
    load_shakespeare_tokenizer,
)

from .training import (
    LMTrainer,
    TrainingConfig,
    TrainingMetrics,
    train_model,
)

from .profiling import (
    MemoryProfiler,
    BandwidthAnalyzer,
    profile_memory,
    MemorySnapshot,
    ProfileResult,
)

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
