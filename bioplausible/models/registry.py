"""
Model Registry for Bio-Plausible Algorithms

Defines specifications for available models and algorithms, used by experiments and UI.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class ModelSpec:
    """Specification for a model."""

    name: str  # Display name
    description: str  # Short description
    model_type: str  # Internal type key (mapped to model class)
    variant: Optional[str] = None  # Variant for transformer models
    default_lr: float = 0.001
    color: str = "#888888"
    task_compat: Optional[List[str]] = (
        None  # ['vision', 'lm', 'rl'] or None for all applicable
    )

    # Model capabilities and family
    family: str = "experimental"
    custom_hyperparams: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[str] = None
    supports_dreaming: bool = False
    supports_dynamics: bool = False
    supports_oracle: bool = False
    supports_alignment: bool = False
    supports_cube_viz: bool = False
    supports_robustness: bool = False
    supports_p2p: bool = False
    supports_text_gen: bool = False
    supports_agent_watch: bool = False
    supports_diffusion_sample: bool = False

    # New fields for Phase 0
    credit_locality: str = "global"        # "global" | "local" | "layerwise" | "forward-only" | "equilibrium"
    domain_support: List[str] = field(default_factory=lambda: ["vision"])
    param_count_approx: Optional[int] = None
    training_paradigm: str = "supervised"  # "supervised" | "self-supervised" | "rl" | "hybrid"
    learning_rule_class: str = "gradient"  # "gradient" | "equilibrium" | "hebbian" | "target" | "forward-only" | "spiking"
    hardware_affinity: List[str] = field(default_factory=list)  # ["gpu", "neuromorphic", "optical", "memristor"]
    requires_backward: bool = True         # False for EP, FF, PEPITA, Hebbian, STDP
    memory_complexity: str = "O(N)"        # "O(1)" for MEP O(1) variants, "O(N)" standard


class ModelRegistry:
    """Registry for BioModels."""

    _models: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(model_cls: Type):
            cls._models[name] = model_cls
            model_cls.algorithm_name = name
            return model_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type:
        if name not in cls._models:
            raise ValueError(
                f"Unknown model: {name}. Available: {list(cls._models.keys())}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._models.keys())


# Convenience
register_model = ModelRegistry.register


# All models available - ordered by category
MODEL_REGISTRY = [
    # Baselines
    ModelSpec(
        name="Backprop Baseline",
        description="Standard Backprop baseline (Transformer/MLP)",
        model_type="backprop",
        default_lr=0.001,
        color="#ff6b6b",
        task_compat=["vision", "lm", "rl"],
        family="baseline",
        supports_agent_watch=True,
        citation=r"""@article{rumelhart1986learning,
  title={Learning representations by back-propagating errors},
  author={Rumelhart, David E and Hinton, Geoffrey E and Williams, Ronald J},
  journal={nature},
  volume={323},
  number={6088},
  pages={533--536},
  year={1986},
  publisher={Nature Publishing Group}
}""",
    ),
    # EqProp MLP
    ModelSpec(
        name="EqProp MLP",
        description="Looped MLP with Spectral Norm",
        model_type="eqprop_mlp",
        default_lr=0.001,
        color="#4ecdc4",
        task_compat=["vision", "rl"],
        family="eqprop",
        supports_dreaming=True,
        supports_dynamics=True,
        supports_oracle=True,
        supports_alignment=True,
        supports_robustness=True,
        supports_agent_watch=True,
        citation=r"""@article{scellier2017equilibrium,
  title={Equilibrium propagation: Bridging the gap between energy-based models and backpropagation},  # noqa: E501
  author={Scellier, Benjamin and Bengio, Yoshua},
  journal={Frontiers in computational neuroscience},
  volume={11},
  pages={24},
  year={2017},
  publisher={Frontiers}
}""",
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    # Advanced EqProp Variants
    ModelSpec(
        name="Holomorphic EqProp",
        description="Complex-valued Equilibrium Propagation",
        model_type="holomorphic_ep",
        default_lr=0.001,
        color="#a55eea",
        task_compat=["vision", "rl"],
        family="eqprop",
        supports_dreaming=True,
        supports_dynamics=True,
        supports_agent_watch=True,
        citation=r"""@inproceedings{laborieux2024holomorphic,
  title={Holomorphic Equilibrium Propagation},
  author={Laborieux, Axel and others},
  booktitle={NeurIPS},
  year={2024}
}""",
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Directed EqProp (Deep EP)",
        description="Asymmetric forward and feedback weights",
        model_type="directed_ep",
        default_lr=0.001,
        color="#fd9644",
        task_compat=["vision", "rl"],
        family="eqprop",
        supports_dreaming=True,
        supports_dynamics=True,
        supports_agent_watch=True,
        citation=r"""@inproceedings{deep2023directed,
  title={Directed Equilibrium Propagation},
  booktitle={ESANN},
  year={2023}
}""",
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Finite-Nudge EqProp",
        description="EqProp with large beta (finite difference)",
        model_type="finite_nudge_ep",
        default_lr=0.001,
        color="#fc5c65",
        task_compat=["vision", "rl"],
        family="eqprop",
        supports_dreaming=True,
        supports_dynamics=True,
        supports_agent_watch=True,
        citation=r"""@article{litman2025finite,
  title={Finite-Nudge Equilibrium Propagation},
  author={Litman, Roee},
  journal={ArXiv},
  year={2025}
}""",
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Conv EqProp (CIFAR-10)",
        description="Convolutional EqProp optimized for CIFAR-10",
        model_type="modern_conv_eqprop",
        default_lr=0.0005,
        color="#26de81",
        task_compat=["cifar10"],
        family="eqprop",
        supports_dreaming=True,
        supports_dynamics=True,
        citation=r"""@article{laborieux2021scaling,
  title={Scaling Equilibrium Propagation to Deep ConvNets by Drastically Reducing its Gradient Estimator Bias},
  author={Laborieux, Axel and Ernoult, Maxence and Scellier, Benjamin and Bengio, Yoshua and Grollier, Julie and Querlioz, Damien},
  journal={Frontiers in neuroscience},
  pages={129},
  year={2021},
  publisher={Frontiers}
}""",
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="EqProp Diffusion",
        description="Generative Diffusion via Equilibrium Propagation",
        model_type="eqprop_diffusion",
        default_lr=0.001,
        color="#fdcb6e",
        task_compat=["vision"],
        family="eqprop",
        supports_dynamics=True,
        supports_diffusion_sample=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Neural Cube",
        description="3D Lattice with local 26-neighbor connectivity",
        model_type="neural_cube",
        default_lr=0.002,
        color="#00b894",
        task_compat=["vision"],
        family="eqprop",
        supports_cube_viz=True,
        supports_dynamics=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    # Hybrid & Experimental Algorithms
    ModelSpec(
        name="Adaptive Feedback Alignment",
        description="FA with slowly adapting feedback weights",
        model_type="adaptive_feedback_alignment",
        default_lr=0.001,
        color="#4b7bec",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        citation=r"""@article{lillicrap2016random,
  title={Random synaptic feedback weights support error backpropagation for deep learning},
  author={Lillicrap, Timothy P and Cownden, Daniel and Tweed, Douglas B and Akerman, Colin J},
  journal={Nature communications},
  volume={7},
  number={1},
  pages={1--10},
  year={2016},
  publisher={Nature Publishing Group}
}""",
        learning_rule_class="gradient"
    ),
    ModelSpec(
        name="Equilibrium Alignment",
        description="EqProp dynamics + Feedback Alignment",
        model_type="eq_align",
        default_lr=0.001,
        color="#d1d8e0",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_dynamics=True,
        supports_agent_watch=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Layerwise Equilibrium FA",
        description="Layerwise training with EqProp/FA hybrid",
        model_type="layerwise_equilibrium_fa",
        default_lr=0.001,
        color="#a5b1c2",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Energy Guided FA",
        description="Feedback Alignment guided by Energy Function",
        model_type="energy_guided_fa",
        default_lr=0.001,
        color="#778ca3",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        learning_rule_class="gradient"
    ),
    ModelSpec(
        name="Predictive Coding Hybrid",
        description="Hybrid of Predictive Coding and EqProp",
        model_type="predictive_coding_hybrid",
        default_lr=0.001,
        color="#3867d6",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_dynamics=True,
        supports_agent_watch=True,
        citation=r"""@article{rao1999predictive,
  title={Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects},
  author={Rao, Rajesh PN and Ballard, Dana H},
  journal={Nature neuroscience},
  volume={2},
  number={1},
  pages={79--87},
  year={1999},
  publisher={Nature Publishing Group}
}""",
    ),
    ModelSpec(
        name="Sparse Equilibrium",
        description="EqProp with sparsity constraints",
        model_type="sparse_equilibrium",
        default_lr=0.001,
        color="#8854d0",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_dynamics=True,
        supports_agent_watch=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Momentum Equilibrium",
        description="EqProp with momentum dynamics",
        model_type="momentum_equilibrium",
        default_lr=0.001,
        color="#45aaf2",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_dynamics=True,
        supports_agent_watch=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="Stochastic FA",
        description="Feedback Alignment with stochastic weights",
        model_type="stochastic_fa",
        default_lr=0.001,
        color="#2bcbba",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        learning_rule_class="gradient"
    ),
    ModelSpec(
        name="Energy Minimizing FA",
        description="FA variant that minimizes local energy",
        model_type="energy_minimizing_fa",
        default_lr=0.001,
        color="#0fb9b1",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        learning_rule_class="gradient"
    ),
    # Other Bio-Plausible Algorithms
    ModelSpec(
        name="DFA (Direct Feedback Alignment)",
        description="Random feedback weights",
        model_type="dfa",
        default_lr=0.001,
        color="#45b7d1",
        task_compat=["vision", "rl"],
        family="hybrid",
        supports_agent_watch=True,
        citation=r"""@inproceedings{nokland2016direct,
  title={Direct feedback alignment provides learning in deep neural networks},
  author={N{\o}kland, Arild},
  booktitle={Advances in neural information processing systems},
  volume={29},
  year={2016}
}""",
        learning_rule_class="gradient"
    ),
    ModelSpec(
        name="CHL (Contrastive Hebbian)",
        description="Contrastive Hebbian Learning",
        model_type="chl",
        default_lr=0.001,
        color="#f9ca24",
        task_compat=["vision", "rl"],
        family="hebbian",
        supports_dynamics=True,
        supports_agent_watch=True,
        citation=r"""@inproceedings{movellan1991contrastive,
  title={Contrastive Hebbian learning in the continuous Hopfield model},
  author={Movellan, Javier R},
  booktitle={Connectionist models},
  pages={10--17},
  year={1991},
  organization={Morgan Kaufmann}
}""",
        credit_locality="local",
        learning_rule_class="hebbian",
        requires_backward=False,
        hardware_affinity=["neuromorphic"]
    ),
    ModelSpec(
        name="Deep Hebbian (Hundred-Layer)",
        description="100-layer Hebbian chain with SN",
        model_type="deep_hebbian",
        default_lr=0.0005,
        color="#6c5ce7",
        task_compat=["vision", "rl"],
        family="hebbian",
        supports_agent_watch=True,
        credit_locality="local",
        learning_rule_class="hebbian",
        requires_backward=False,
        hardware_affinity=["neuromorphic"]
    ),
    ModelSpec(
        name="EquiTile",
        description="Scalable Local-Learning Architecture with Tiled Substrates (PC)",
        model_type="equitile",
        default_lr=0.01,
        color="#fd79a8",
        task_compat=["vision", "rl", "classification"],
        family="equitile",
        supports_dynamics=True,
        supports_agent_watch=True,
        credit_locality="local",
        hardware_affinity=["neuromorphic", "distributed"]
    ),
    ModelSpec(
        name="EquiTile EP",
        description="EquiTile with Equilibrium Propagation Learning",
        model_type="equitile_ep",
        default_lr=0.01,
        color="#e84393",
        task_compat=["vision", "rl", "classification"],
        family="equitile",
        supports_dynamics=True,
        supports_agent_watch=True,
        credit_locality="local",
        hardware_affinity=["neuromorphic", "distributed"],
        learning_rule_class="equilibrium",
        requires_backward=False
    ),
    ModelSpec(
        name="LM EquiTile",
        description="Transformer-style EquiTile for Language Modeling",
        model_type="lm_equitile",
        default_lr=1e-4,
        color="#6c5ce7",
        task_compat=["lm"],
        family="equitile",
        supports_text_gen=True,
        credit_locality="local",
        hardware_affinity=["neuromorphic", "distributed"]
    ),
    ModelSpec(
        name="RL EquiTile",
        description="EquiTile Actor-Critic for Reinforcement Learning",
        model_type="rl_equitile",
        default_lr=3e-4,
        color="#a29bfe",
        task_compat=["rl"],
        family="equitile",
        supports_agent_watch=True,
        credit_locality="local",
        hardware_affinity=["neuromorphic", "distributed"]
    ),
    ModelSpec(
        name="Conv EquiTile",
        description="Convolutional EquiTile for Vision",
        model_type="conv_equitile",
        default_lr=0.01,
        color="#fdcb6e",
        task_compat=["vision"],
        family="equitile",
        supports_dynamics=True,
        credit_locality="local",
        hardware_affinity=["neuromorphic", "distributed"]
    ),
    # EqProp Transformers (From Track 37 results) - SLOW MODELS LAST
    ModelSpec(
        name="EqProp Transformer (Attention Only)",
        description="Best variant: EqProp in attention only",
        model_type="eqprop_transformer",
        variant="attention_only",
        default_lr=0.0003,
        color="#2ecc71",
        task_compat=["lm"],
        family="eqprop",
        supports_text_gen=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="EqProp Transformer (Full)",
        description="All layers use equilibrium",
        model_type="eqprop_transformer",
        variant="full",
        default_lr=0.0003,
        color="#27ae60",
        task_compat=["lm"],
        family="eqprop",
        supports_text_gen=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="EqProp Transformer (Hybrid)",
        description="Standard layers + EqProp final layer",
        model_type="eqprop_transformer",
        variant="hybrid",
        default_lr=0.0003,
        color="#1abc9c",
        task_compat=["lm"],
        family="eqprop",
        supports_text_gen=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    ModelSpec(
        name="EqProp Transformer (Recurrent)",
        description="Single recurrent block, parameter efficient",
        model_type="eqprop_transformer",
        variant="recurrent_core",
        default_lr=0.0003,
        color="#16a085",
        task_compat=["lm"],
        family="eqprop",
        supports_text_gen=True,
        credit_locality="equilibrium",
        learning_rule_class="equilibrium",
        requires_backward=False,
        hardware_affinity=["neuromorphic", "analog"]
    ),
    # Custom
    ModelSpec(
        name="Custom Stacked Model",
        description="User-defined stack of layers (Linear, Conv, EquiTile)",
        model_type="custom_stacked_model",
        default_lr=0.001,
        color="#55efc4",
        task_compat=["vision", "rl"],
        family="custom",
        supports_agent_watch=True,
    ),
    # New Phase 0 Families
    ModelSpec(
        name="Forward-Forward",
        description="Hinton's Forward-Forward layer-local goodness",
        model_type="forward_forward",
        default_lr=0.001,
        color="#e17055",
        task_compat=["vision", "tabular"],
        family="forward_only",
        credit_locality="layerwise",
        learning_rule_class="forward-only",
        requires_backward=False,
        citation="Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations."
    ),
    ModelSpec(
        name="PEPITA",
        description="Present the Error to Perturb the Input To modulate Activity",
        model_type="pepita",
        default_lr=0.001,
        color="#00cec9",
        task_compat=["vision"],
        family="forward_only",
        credit_locality="forward-only",
        learning_rule_class="forward-only",
        requires_backward=False
    ),
    ModelSpec(
        name="Difference Target Propagation",
        description="Propagates targets backward using approximate inverses",
        model_type="diff_target_prop",
        default_lr=0.001,
        color="#6c5ce7",
        task_compat=["vision"],
        family="target_prop",
        credit_locality="layerwise",
        learning_rule_class="target",
        requires_backward=True
    ),
    ModelSpec(
        name="Three-Factor Hebbian",
        description="Neuromodulated Hebbian Learning (pre x post x reward)",
        model_type="three_factor_hebbian",
        default_lr=0.005,
        color="#fdcb6e",
        task_compat=["vision", "rl"],
        family="hebbian",
        credit_locality="local",
        learning_rule_class="hebbian",
        requires_backward=False,
        hardware_affinity=["neuromorphic"]
    ),
    ModelSpec(
        name="Spiking STDP",
        description="Leaky Integrate-and-Fire with Spike-Timing-Dependent Plasticity",
        model_type="spiking_stdp",
        default_lr=0.001,
        color="#d63031",
        task_compat=["vision"],
        family="spiking",
        credit_locality="local",
        learning_rule_class="spiking",
        requires_backward=False,
        hardware_affinity=["neuromorphic"]
    ),
]


def get_model_spec(name: str) -> ModelSpec:
    """Get model spec by name."""
    target = name.lower().replace(" ", "").replace("_", "")
    for spec in MODEL_REGISTRY:
        # Check normalization of name
        spec_name_norm = spec.name.lower().replace(" ", "").replace("_", "")
        if spec_name_norm == target:
            return spec
        # Check normalization of model_type
        if spec.model_type.lower().replace("_", "") == target:
            return spec

    # Aliases
    if target == "backpropmlp":
        return get_model_spec("Backprop Baseline")
    if target == "loopedmlp":
        return get_model_spec("EqProp MLP")
    if target == "memoryefficientmlp":
        return get_model_spec("Sparse Equilibrium")

    # Fallback: check partial match if it's unambiguous?
    # For now, strict normalized match is safer.
    raise ValueError(
        f"Unknown model: {name}. Available: {[spec.name for spec in MODEL_REGISTRY]}"
    )


def list_model_names() -> List[str]:
    """List all available model names."""
    return [spec.name for spec in MODEL_REGISTRY]


def list_model_specs() -> List[ModelSpec]:
    """List all available model specifications."""
    return MODEL_REGISTRY
