"""
Search Space Definitions

Defines the hyperparameter search spaces for each model type in the registry.
"""

from typing import Any

import numpy as np

# Type aliases
NumberRange = tuple[
    float, float, str
]  # (min, max, scale) where scale in ['log', 'linear', 'int']
DiscreteChoice = list[int | float | str]


class SearchSpace:
    """
    Hyperparameter search space for a model.

    Note: This class now only stores parameter definitions.
    All sampling/mutation/crossover is handled by Optuna.
    Use optuna_bridge.create_optuna_space() for optimization.
    """

    def __init__(self, name: str, params: dict[str, NumberRange | DiscreteChoice]):
        self.name = name
        self.params = params

    def sample(self) -> dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}
        for name, space in self.params.items():
            if isinstance(space, list):
                # Discrete choice
                config[name] = np.random.choice(space)
                # Convert numpy types to python native
                if isinstance(config[name], (np.generic)):
                    config[name] = config[name].item()
            elif isinstance(space, tuple) and len(space) == 3:
                # Number range
                min_val, max_val, scale = space
                if scale == "int":
                    config[name] = int(np.random.randint(min_val, max_val + 1))
                elif scale == "log":
                    # Log uniform
                    log_min = np.log(min_val)
                    log_max = np.log(max_val)
                    config[name] = float(np.exp(np.random.uniform(log_min, log_max)))
                else:
                    # Linear
                    config[name] = float(np.random.uniform(min_val, max_val))
        return config

    def apply_constraints(self, constraints: dict[str, Any]) -> SearchSpace:
        """
        Return a new constrained search space based on constraints dictionary.
        Supports max_hidden, max_layers, max_steps.
        """
        import copy

        new_params = copy.deepcopy(self.params)

        mapping = {
            "max_hidden": "hidden_dim",
            "max_layers": "num_layers",
            "max_steps": "steps",
        }

        for const_key, limit in constraints.items():
            if const_key in mapping:
                param_key = mapping[const_key]
                if param_key in new_params:
                    space = new_params[param_key]
                    if isinstance(space, list):
                        new_params[param_key] = [v for v in space if v <= limit]
                    elif isinstance(space, tuple) and len(space) == 3:
                        min_val, max_val, scale = space
                        new_max = min(max_val, limit)
                        new_max = max(new_max, min_val)  # Safe fallback
                        new_params[param_key] = (min_val, new_max, scale)

        return SearchSpace(self.name + "_constrained", new_params)


# Define search spaces for all models
SEARCH_SPACES = {
    "backprop_mlp": SearchSpace(
        "backprop_mlp",
        {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [32, 64, 128, 256],
            "num_layers": [1, 2, 4],
        },
    ),
    "eqprop_mlp": SearchSpace(
        "eqprop_mlp",
        {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [32, 64, 128],
            "num_layers": [5, 10, 15],
        },
    ),
    # Research Models
    "Holomorphic EqProp": SearchSpace(
        "Holomorphic EqProp",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.01, 0.3, "linear"),
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Directed EqProp (Deep EP)": SearchSpace(
        "Directed EqProp (Deep EP)",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.1, 0.5, "linear"),
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Finite-Nudge EqProp": SearchSpace(
        "Finite-Nudge EqProp",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.5, 3.0, "linear"),  # Large beta
            "steps": (10, 40, "int"),
            "hidden_dim": [64, 128],
        },
    ),
    "Conv EqProp (CIFAR-10)": SearchSpace(
        "Conv EqProp (CIFAR-10)",
        {
            "lr": (1e-4, 1e-2, "log"),
            "steps": (10, 25, "int"),
            "hidden_dim": [128, 256],
        },
    ),
    # Hybrid & Experimental
    "Adaptive Feedback Alignment": SearchSpace(
        "Adaptive Feedback Alignment",
        {
            "lr": (1e-4, 1e-2, "log"),
            "fa_scale": (0.5, 1.5, "linear"),
            "adapt_rate": (0.001, 0.1, "log"),
            "hidden_dim": [64, 128, 256],
        },
    ),
    "Equilibrium Alignment": SearchSpace(
        "Equilibrium Alignment",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.1, 0.5, "linear"),
            "steps": (10, 30, "int"),
            "align_weight": (0.1, 1.0, "linear"),
        },
    ),
    # Add Missing Spaces
    "Layerwise Equilibrium FA": SearchSpace(
        "Layerwise Equilibrium FA",
        {"lr": (1e-4, 1e-2, "log"), "hidden_dim": [64, 128], "num_layers": [2, 4, 6]},
    ),
    "Energy Guided FA": SearchSpace(
        "Energy Guided FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "energy_scale": (0.1, 1.0, "linear"),
            "hidden_dim": [64, 128],
        },
    ),
    "Predictive Coding Hybrid": SearchSpace(
        "Predictive Coding Hybrid",
        {"lr": (1e-4, 1e-2, "log"), "steps": (10, 30, "int"), "hidden_dim": [64, 128]},
    ),
    "Sparse Equilibrium": SearchSpace(
        "Sparse Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "beta": (0.05, 0.3, "linear"),
            "sparsity": (0.1, 0.9, "linear"),
            "hidden_dim": [128, 256],
        },
    ),
    "Momentum Equilibrium": SearchSpace(
        "Momentum Equilibrium",
        {
            "lr": (1e-4, 1e-2, "log"),
            "momentum": (0.5, 0.95, "linear"),
            "steps": (10, 30, "int"),
        },
    ),
    "Stochastic FA": SearchSpace(
        "Stochastic FA",
        {
            "lr": (1e-4, 1e-2, "log"),
            "noise_scale": (0.01, 0.2, "log"),
            "hidden_dim": [64, 128],
        },
    ),
    "Energy Minimizing FA": SearchSpace(
        "Energy Minimizing FA", {"lr": (1e-4, 1e-2, "log"), "hidden_dim": [64, 128]}
    ),
    # Transformers
    "eqprop_transformer": SearchSpace(
        "eqprop_transformer",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 12, "int"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Full)": SearchSpace(
        "EqProp Transformer (Full)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Hybrid)": SearchSpace(
        "EqProp Transformer (Hybrid)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (5, 15, "int"),
            "hidden_dim": [128, 256],
            "num_layers": [2, 3],
        },
    ),
    "EqProp Transformer (Recurrent)": SearchSpace(
        "EqProp Transformer (Recurrent)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "steps": (10, 30, "int"),
            "hidden_dim": [128, 256],
            "num_layers": [1],  # Recurrent uses single block
        },
    ),
    "DFA (Direct Feedback Alignment)": SearchSpace(
        "DFA (Direct Feedback Alignment)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [10, 20, 30],
        },
    ),
    "CHL (Contrastive Hebbian)": SearchSpace(
        "CHL (Contrastive Hebbian)",
        {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.3, "linear"),
            "steps": (10, 30, "int"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [10, 20, 30],
        },
    ),
    "Deep Hebbian (Hundred-Layer)": SearchSpace(
        "Deep Hebbian (Hundred-Layer)",
        {
            "lr": (1e-5, 5e-3, "log"),
            "hidden_dim": [64, 128],
            "num_layers": [50, 100, 150],  # Test deep scaling
        },
    ),
    "equitile": SearchSpace(
        "equitile",
        {
            "lr": (1e-4, 1e-1, "log"),
            "inference_steps": (5, 30, "int"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [4, 8, 16],
            "num_layers": [3, 5, 8],
            "sparsity_threshold": (0.01, 0.2, "linear"),
        },
    ),
    "EquiTile EP": SearchSpace(
        "EquiTile EP",
        {
            "lr": (1e-4, 1e-1, "log"),
            "beta": (0.05, 0.5, "linear"),
            "inference_steps": (10, 50, "int"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [4, 8, 16],
            "num_layers": [3, 5, 8],
        },
    ),
    "LM EquiTile": SearchSpace(
        "LM EquiTile",
        {
            "lr": (1e-5, 1e-3, "log"),
            "neurons_per_tile": [64, 128],
            "tiles_per_layer": [4, 8],
            "num_layers": [4, 6],
            "embed_dim": [128, 256],
            "num_heads": [2, 4],
        },
    ),
    "RL EquiTile": SearchSpace(
        "RL EquiTile",
        {
            "lr": (1e-4, 1e-2, "log"),
            "neurons_per_tile": [32, 64],
            "tiles_per_layer": [2, 4, 8],
            "num_layers": [2, 3],
            "entropy_coef": (0.001, 0.05, "log"),
            "value_coef": (0.1, 1.0, "linear"),
        },
    ),
    "Conv EquiTile": SearchSpace(
        "Conv EquiTile",
        {
            "lr": (1e-4, 1e-2, "log"),
            "neurons_per_tile": [32, 64, 128],
            "tiles_per_layer": [2, 4, 8],
            "num_fc_layers": [1, 2, 3],
            "dropout": (0.0, 0.5, "linear"),
        },
    ),
}


def get_search_space(model_name: str) -> SearchSpace:
    """Get the search space for a model."""
    # 1. Try hardcoded spaces first (for customized ranges)
    if model_name in SEARCH_SPACES:
        return SEARCH_SPACES[model_name]

    # 2. Try to generate from registry
    # Check if exact name in registry
    spec = next((s for s in MODEL_REGISTRY if s.name == model_name), None)

    if spec:
        params = {
            "lr": (1e-5, 1e-2, "log"),
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 4, 6],
        }

        return SearchSpace(model_name, params)

    # 3. Canonicalize name using get_model_spec
    try:
        spec = get_model_spec(model_name)
        canonical_name = spec.name

        # Try again with canonical name
        if canonical_name in SEARCH_SPACES:
            return SEARCH_SPACES[canonical_name]

        # If registry spec exists but no explicit search space, infer one?
        if "EqProp" in canonical_name:
            params = {
                "lr": (1e-5, 1e-2, "log"),
                "beta": (0.05, 0.5, "linear"),
                "steps": (5, 20, "int"),
                "hidden_dim": [64, 128],
            }
            return SearchSpace(model_name, params)

        if "Backprop" in canonical_name:
            return SEARCH_SPACES["backprop_mlp"]

    except ValueError:
        pass  # Model unknown to registry

    # 4. Fallback inference (legacy)
    if "EqProp" in model_name:
        params = {
            "lr": (1e-5, 1e-2, "log"),
            "beta": (0.05, 0.5, "linear"),
            "steps": (5, 20, "int"),
            "hidden_dim": [64, 128],
        }
        return SearchSpace(model_name, params)

    raise ValueError(f"No search space defined for model: {model_name}")
