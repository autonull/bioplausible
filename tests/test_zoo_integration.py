"""
Integration tests for Zoo component combinations.

Verifies that registered models, propagators, and optimizers can be
combined and used with CoreTrainer.
"""

import torch
import torch.nn as nn

# Import zoo modules to trigger registration
import bioplausible.zoo  # noqa: F401
from bioplausible.core.registry import (
    ComponentCategory,
    Domain,
    LocalityLevel,
    Registry,
)


def test_registry_has_models():
    """Verify that models are registered in the zoo."""
    models = Registry.list(ComponentCategory.MODEL)
    assert "model" in models
    assert len(models["model"]) > 0
    assert "MLP" in models["model"]
    assert "EquiTile" in models["model"]


def test_registry_has_propagators():
    """Verify that propagators are registered."""
    props = Registry.list(ComponentCategory.PROPAGATOR)
    assert "propagator" in props
    assert len(props["propagator"]) > 0
    assert "FeedbackAlignment" in props["propagator"]


def test_registry_has_optimizers():
    """Verify that optimizers are registered."""
    opts = Registry.list(ComponentCategory.OPTIMIZER)
    assert "optimizer" in opts
    assert len(opts["optimizer"]) > 0
    assert "adam" in opts["optimizer"]
    assert "sgd" in opts["optimizer"]


def test_registry_has_sparsity():
    """Verify that sparsity methods are registered."""
    sparsity = Registry.list(ComponentCategory.SPARSITY)
    assert "sparsity" in sparsity
    assert len(sparsity["sparsity"]) > 0


def test_query_by_domain_vision():
    """Test querying models by vision domain."""
    results = Registry.query(category=ComponentCategory.MODEL, domain=Domain.VISION)
    assert len(results) >= 1
    names = [r["name"] for r in results]
    assert "MLP" in names


def test_query_bio_plausible_models():
    """Test querying bio-plausible models (no backward pass)."""
    results = Registry.query(
        category=ComponentCategory.MODEL,
        requires_backward=False,
    )
    assert len(results) >= 1
    for r in results:
        assert r["metadata"].requires_backward is False


def test_query_local_learning():
    """Test querying local learning rules."""
    results = Registry.query(
        category=ComponentCategory.PROPAGATOR,
        locality=LocalityLevel.LOCAL,
    )
    names = [r["name"] for r in results]
    assert "ContrastiveHebbian" in names


def test_get_compatible():
    """Test getting compatible components for a model."""
    compat = Registry.get_compatible("MLP")
    assert ComponentCategory.PROPAGATOR in compat
    assert ComponentCategory.OPTIMIZER in compat


def test_metadata_on_registered_class():
    """Test that registered classes have metadata attached."""
    MLP_cls = Registry.get(ComponentCategory.MODEL, "MLP")
    assert hasattr(MLP_cls, "_registry_metadata")
    assert MLP_cls._registry_metadata.name == "MLP"
    assert MLP_cls._registry_metadata.bio_plausibility_score == 0.0


def test_mlp_instantiation():
    """Test instantiating a registered model."""
    MLP_cls = Registry.get(ComponentCategory.MODEL, "MLP")
    model = MLP_cls(input_dim=784, hidden_dim=64, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_equitile_instantiation():
    """Test instantiating EquiTile."""
    EqT_cls = Registry.get(ComponentCategory.MODEL, "EquiTile")
    model = EqT_cls(input_dim=784, hidden_dim=256, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_forward_forward_instantiation():
    """Test instantiating Forward-Forward network."""
    FF_cls = Registry.get(ComponentCategory.MODEL, "ForwardForwardNet")
    model = FF_cls(input_dim=784, hidden_dim=64, output_dim=10)
    assert model is not None

    x = torch.randn(4, 784)
    out = model(x)
    assert out.shape == (4, 10)


def test_optimizer_instantiation():
    """Test instantiating registered optimizers."""
    Adam_cls = Registry.get(ComponentCategory.OPTIMIZER, "adam")
    model = nn.Linear(10, 2)
    opt = Adam_cls(model.parameters(), lr=0.001)
    assert opt is not None


def test_cross_domain_query():
    """Test that models can be queried across multiple domains."""
    results = Registry.query(
        category=ComponentCategory.MODEL,
        domain=Domain.LM,
    )
    # EquiTile is registered for LM
    names = [r["name"] for r in results]
    assert "EquiTile" in names


def test_bio_score_query():
    """Test filtering by bio-plausibility score."""
    high_bio = Registry.query(
        category=ComponentCategory.MODEL,
        min_bio_score=0.8,
    )
    assert len(high_bio) >= 1
    for r in high_bio:
        assert r["metadata"].bio_plausibility_score >= 0.8


def test_export_yaml(tmp_path):
    """Test exporting registry to YAML."""
    yaml_path = tmp_path / "registry.yaml"
    Registry.export_yaml(str(yaml_path))
    assert yaml_path.exists()

    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    assert "model" in data
    assert "optimizer" in data
    assert "propagator" in data
