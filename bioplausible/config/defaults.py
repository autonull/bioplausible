"""
Default experiment configurations for common scenarios.
"""

from typing import Dict

from omegaconf import OmegaConf

from bioplausible.config.schema import ExperimentConfig

DEFAULT_CONFIGS: Dict[str, ExperimentConfig] = {}


def _register_default(name: str, overrides: dict) -> None:
    """Register a default config by merging overrides into the base config."""
    base = OmegaConf.structured(ExperimentConfig)
    merged = OmegaConf.merge(base, OmegaConf.create(overrides))
    DEFAULT_CONFIGS[name] = OmegaConf.to_object(merged)


# ---- Vision benchmarks ----

_register_default(
    "vision_mlp",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "optimizer": {"name": "adam", "lr": 0.001},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_eqprop",
    {
        "model": {"name": "EqPropMLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_ff",
    {
        "model": {
            "name": "ForwardForwardNet",
            "kwargs": {"hidden_dim": 256, "num_layers": 3},
        },
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

_register_default(
    "vision_equitile",
    {
        "model": {"name": "equitile", "kwargs": {"hidden_dim": 256, "num_tiles": 4}},
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

# ---- MEP benchmarks ----

_register_default(
    "vision_mep_smep",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 256, "num_layers": 3}},
        "propagator": {"name": "smep", "kwargs": {"beta": 0.5}},
        "optimizer": {"name": "adam", "lr": 0.01},
        "dataset": {"name": "mnist", "batch_size": 64},
        "trainer": {"epochs": 10},
    },
)

# ---- LM benchmarks ----

_register_default(
    "lm_mlp",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 512, "num_layers": 4}},
        "optimizer": {"name": "adamw", "lr": 0.0003},
        "dataset": {"name": "tiny_shakespeare", "batch_size": 32},
        "domain": {"domain": "lm"},
        "trainer": {"epochs": 20},
    },
)

# ---- Ablation configs ----

_register_default(
    "ablation_quick",
    {
        "model": {"name": "MLP", "kwargs": {"hidden_dim": 128, "num_layers": 2}},
        "optimizer": {"name": "sgd", "lr": 0.01},
        "dataset": {"name": "digits", "batch_size": 32},
        "trainer": {"epochs": 3, "batches_per_epoch": 50},
        "track_energy": True,
    },
)
