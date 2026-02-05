from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
import math

class HyperparamScope(Enum):
    """Defines which algorithms a hyperparameter applies to."""
    UNIVERSAL = "universal"        # All algorithms (lr, hidden_dim, etc.)
    GRADIENT_BASED = "gradient"    # Backprop, variants (optimizer, grad_clip)
    EQUILIBRIUM = "equilibrium"    # EqProp family (beta, steps, nudge_type)
    FEEDBACK_ALIGNMENT = "fa"      # FA variants (fa_scale, adapt_rate)
    HEBBIAN = "hebbian"            # CHL, etc. (contrastive_steps)
    TRANSFORMER = "transformer"    # Transformer-specific (num_heads, etc.)

@dataclass
class HyperparamSpec:
    """Specification for a single hyperparameter."""
    name: str
    scope: HyperparamScope
    param_type: str  # "continuous", "discrete", "categorical"
    
    # For continuous/discrete
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    scale: Optional[str] = None  # "log", "linear", "int"
    
    # For categorical
    choices: Optional[List[Any]] = None
    
    # Conditional dependencies
    requires: Optional[List[str]] = None  # Other hyperparams that must exist
    conflicts: Optional[List[str]] = None  # Hyperparams that cannot coexist
    
    # Metadata
    description: str = ""
    default: Any = None

# Universal hyperparameters (apply to ALL algorithms)
UNIVERSAL_HYPERPARAMS = [
    HyperparamSpec(
        name="lr",
        scope=HyperparamScope.UNIVERSAL,
        param_type="continuous",
        range_min=1e-5,
        range_max=1e-1,
        scale="log",
        description="Learning rate for weight updates",
        default=1e-3,
    ),
    HyperparamSpec(
        name="hidden_dim",
        scope=HyperparamScope.UNIVERSAL,
        param_type="discrete",
        choices=[32, 64, 128, 256, 512],
        description="Number of hidden units per layer",
        default=128,
    ),
    HyperparamSpec(
        name="num_layers",
        scope=HyperparamScope.UNIVERSAL,
        param_type="discrete",
        range_min=1,
        range_max=30,
        scale="int",
        description="Number of layers in the network",
        default=4,
    ),
    HyperparamSpec(
        name="activation",
        scope=HyperparamScope.UNIVERSAL,
        param_type="categorical",
        choices=["relu", "gelu", "silu", "tanh", "leaky_relu", "elu"],
        description="Activation function (NOTE: some algorithms constrain this)",
        default="silu",
    ),
    HyperparamSpec(
        name="weight_init",
        scope=HyperparamScope.UNIVERSAL,
        param_type="categorical",
        choices=["xavier", "kaiming", "orthogonal", "lecun"],
        description="Weight initialization scheme",
        default="kaiming",
    ),
]

# Gradient-based only (Backprop and gradient-based variants)
GRADIENT_HYPERPARAMS = [
    HyperparamSpec(
        name="optimizer",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="categorical",
        choices=["sgd", "adam", "adamw", "rmsprop"],
        description="Gradient descent optimizer (ONLY for Backprop)",
        default="adam",
    ),
    HyperparamSpec(
        name="weight_decay",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=1e-6,
        range_max=1e-2,
        scale="log",
        description="L2 regularization strength",
        default=1e-4,
    ),
    HyperparamSpec(
        name="grad_clip",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=10.0,
        scale="linear",
        description="Gradient clipping threshold",
        default=1.0,
    ),
    HyperparamSpec(
        name="dropout",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=0.5,
        scale="linear",
        description="Dropout probability",
        default=0.0,
    ),
    HyperparamSpec(
        name="momentum",
        scope=HyperparamScope.GRADIENT_BASED,
        param_type="continuous",
        range_min=0.0,
        range_max=0.99,
        scale="linear",
        description="SGD momentum (only if optimizer=sgd)",
        requires=["optimizer"],  # Conditional
        default=0.9,
    ),
]

# Equilibrium Propagation family
EQUILIBRIUM_HYPERPARAMS = [
    HyperparamSpec(
        name="beta",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="continuous",
        range_min=0.01,
        range_max=1.0,
        scale="linear",
        description="Nudge strength for clamping (EqProp only)",
        default=0.1,
    ),
    HyperparamSpec(
        name="steps",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="discrete",
        range_min=5,
        range_max=50,
        scale="int",
        description="Number of relaxation steps (EqProp only)",
        default=20,
    ),
    HyperparamSpec(
        name="nudge_type",
        scope=HyperparamScope.EQUILIBRIUM,
        param_type="categorical",
        choices=["output_clamping", "energy_based", "symmetric"],
        description="How to apply target nudging",
        default="output_clamping",
    ),
]

# Feedback Alignment family
FA_HYPERPARAMS = [
    HyperparamSpec(
        name="fa_scale",
        scope=HyperparamScope.FEEDBACK_ALIGNMENT,
        param_type="continuous",
        range_min=0.5,
        range_max=2.0,
        scale="linear",
        description="Scaling factor for feedback weights",
        default=1.0,
    ),
    HyperparamSpec(
        name="adapt_rate",
        scope=HyperparamScope.FEEDBACK_ALIGNMENT,
        param_type="continuous",
        range_min=1e-4,
        range_max=1e-1,
        scale="log",
        description="Adaptation rate for feedback weights",
        default=1e-2,
    ),
]

# Hebbian family
HEBBIAN_HYPERPARAMS = [
    HyperparamSpec(
        name="contrastive_steps",
        scope=HyperparamScope.HEBBIAN,
        param_type="discrete",
        range_min=5,
        range_max=30,
        scale="int",
        description="Steps in contrastive phase",
        default=10,
    ),
]

# Transformer-specific
TRANSFORMER_HYPERPARAMS = [
    HyperparamSpec(
        name="num_heads",
        scope=HyperparamScope.TRANSFORMER,
        param_type="categorical",
        choices=[2, 4, 8],
        description="Number of attention heads",
        default=4,
    ),
    HyperparamSpec(
        name="context_length",
        scope=HyperparamScope.TRANSFORMER,
        param_type="discrete",
        choices=[64, 128, 256, 512],
        description="Maximum sequence length",
        default=128,
    ),
]

class HyperparameterMetamodel:
    """
    Central registry that knows which hyperparameters apply to which algorithms.
    """
    
    def __init__(self):
        self.all_specs = (
            UNIVERSAL_HYPERPARAMS +
            GRADIENT_HYPERPARAMS +
            EQUILIBRIUM_HYPERPARAMS +
            FA_HYPERPARAMS +
            HEBBIAN_HYPERPARAMS +
            TRANSFORMER_HYPERPARAMS
        )
        self._spec_dict = {spec.name: spec for spec in self.all_specs}
    
    def get_search_space_for_model(self, model_spec: Any) -> Dict[str, HyperparamSpec]:
        """
        Return the appropriate hyperparameters for a given model.
        
        Uses the model's family to determine which scoped params apply.
        """
        applicable_scopes = {HyperparamScope.UNIVERSAL}
        
        # Map model families to hyperparameter scopes
        family = model_spec.family.lower()
        
        if family == "baseline":
            # Backprop uses gradient-based hyperparams
            applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
        
        elif family == "eqprop":
            # EqProp uses equilibrium hyperparams, NO optimizer
            applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
        
        elif family == "hybrid":
            # Hybrid models (e.g., Adaptive FA) might use both
            # Need to check model_type for specifics
            if "fa" in model_spec.model_type or "alignment" in model_spec.model_type:
                applicable_scopes.add(HyperparamScope.FEEDBACK_ALIGNMENT)
            if "equilibrium" in model_spec.model_type or "eq" in model_spec.model_type:
                applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
            if "hebbian" in model_spec.model_type:
                applicable_scopes.add(HyperparamScope.HEBBIAN)

            # Hybrids might also use gradient methods (often do)
            applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
        
        elif family == "hebbian":
            applicable_scopes.add(HyperparamScope.HEBBIAN)
        
        # Transformers get transformer-specific params
        if "transformer" in model_spec.model_type or "transformer" in model_spec.name.lower():
            applicable_scopes.add(HyperparamScope.TRANSFORMER)
            # Transformers also use gradient-based training (usually)
            if family != "eqprop":  # Unless it's EqProp Transformer
                applicable_scopes.add(HyperparamScope.GRADIENT_BASED)
            else:
                applicable_scopes.add(HyperparamScope.EQUILIBRIUM)
        
        # Filter specs by applicable scopes
        search_space = {}
        for spec in self.all_specs:
            if spec.scope in applicable_scopes:
                search_space[spec.name] = spec
        
        # Apply algorithm-specific activation constraints
        # Example: Holomorphic EqProp REQUIRES tanh (holomorphic)
        if "holomorphic" in model_spec.name.lower():
            # Create a copy to not modify the global spec
            act_spec = self._spec_dict["activation"]
            # We need a deep copy or just a new instance if we modify it
            import copy
            constrained_act = copy.deepcopy(act_spec)
            constrained_act.choices = ["tanh"]
            search_space["activation"] = constrained_act
        
        return search_space
    
    def validate_config(self, model_spec: Any, config: Dict[str, Any]) -> List[str]:
        """
        Validate that a config is compatible with a model.
        Returns list of error messages (empty if valid).
        """
        errors = []
        valid_space = self.get_search_space_for_model(model_spec)
        
        for key, value in config.items():
            if key not in valid_space:
                # Some infrastructure keys might be passed (e.g., 'epochs', 'device')
                # We should only flag if it matches a known hyperparam that ISN'T available
                if key in self._spec_dict: 
                     errors.append(
                        f"Hyperparameter '{key}' is not applicable to {model_spec.name} "
                        f"(family: {model_spec.family})"
                    )
        
        # Check for missing required params
        for key, spec in valid_space.items():
            if spec.requires:
                for req in spec.requires:
                    if req not in config:
                        # Only error if the dependent param is itself present/active
                        # This logic might need refinement depending on how config is built
                        pass
                        
        return errors

# Global instance
HYPERPARAM_METAMODEL = HyperparameterMetamodel()
