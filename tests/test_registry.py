"""Tests for the Registry system."""

import copy

import pytest

from bioplausible.core.registry import ComponentCategory
from bioplausible.core.registry import ComponentMetadata
from bioplausible.core.registry import Domain
from bioplausible.core.registry import LocalityLevel
from bioplausible.core.registry import Registry
from bioplausible.core.registry import register_model
from bioplausible.core.registry import register_optimizer


@pytest.fixture(autouse=True)
def _preserve_registry():
    """Save and restore registry state around each test to prevent cross-test pollution."""
    saved_components = copy.deepcopy(Registry._components)
    saved_name_map = dict(Registry._name_to_category)
    yield
    Registry._components.clear()
    Registry._components.update(copy.deepcopy(saved_components))
    Registry._name_to_category.clear()
    Registry._name_to_category.update(saved_name_map)


def test_registry_clear():
    """Test clearing the registry."""
    Registry.clear()
    assert Registry.list() == {}


def test_register_and_get():
    """Test registering and getting a component."""
    Registry.clear()

    @register_model(
        name="TestModel", domains=[Domain.VISION], description="A test model"
    )
    class TestModel:
        pass

    # Get by category and name
    cls = Registry.get(ComponentCategory.MODEL, "TestModel")
    assert cls == TestModel

    # Get metadata
    meta = Registry.get_metadata(ComponentCategory.MODEL, "TestModel")
    assert meta.name == "TestModel"
    assert Domain.VISION in meta.domains
    assert meta.description == "A test model"
    assert meta.category == ComponentCategory.MODEL


def test_register_duplicate_warning(caplog):
    """Test warning on duplicate registration."""
    Registry.clear()

    @register_model(name="DupModel")
    class ModelA:
        pass

    @register_model(name="DupModel")
    class ModelB:
        pass

    assert "Overwriting" in caplog.text


def test_get_unknown():
    """Test error on getting unknown component."""
    Registry.clear()

    # Register a model so category exists
    @register_model(name="DummyModel")
    class DummyModel:
        pass

    with pytest.raises(ValueError, match="Unknown model"):
        Registry.get(ComponentCategory.MODEL, "NonExistent")


def test_list_empty():
    """Test listing when registry is empty."""
    Registry.clear()
    assert Registry.list() == {}


def test_list_with_entries():
    """Test listing registered components."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_optimizer(name="OptA")
    class OptA:
        pass

    result = Registry.list()
    assert "model" in result
    assert "optimizer" in result
    assert result["model"] == ["ModelA"]
    assert result["optimizer"] == ["OptA"]


def test_list_by_category():
    """Test listing by category."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    result = Registry.list(ComponentCategory.MODEL)
    assert "model" in result
    assert result["model"] == ["ModelA"]
    assert "optimizer" not in result


def test_query_no_filters():
    """Test query without filters returns everything."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_model(name="ModelB")
    class ModelB:
        pass

    results = Registry.query()
    assert len(results) == 2
    assert {r["name"] for r in results} == {"ModelA", "ModelB"}


def test_query_by_domain():
    """Test query by domain."""
    Registry.clear()

    @register_model(name="VisionModel", domains=[Domain.VISION])
    class VisionModel:
        pass

    @register_model(name="LMModel", domains=[Domain.LM])
    class LMModel:
        pass

    vision_results = Registry.query(domain=Domain.VISION)
    assert len(vision_results) == 1
    assert vision_results[0]["name"] == "VisionModel"

    lm_results = Registry.query(domain=Domain.LM)
    assert len(lm_results) == 1
    assert lm_results[0]["name"] == "LMModel"


def test_query_by_locality():
    """Test query by locality level."""
    Registry.clear()

    @register_model(name="GlobalModel", locality_level=LocalityLevel.GLOBAL)
    class GlobalModel:
        pass

    @register_model(name="LocalModel", locality_level=LocalityLevel.LOCAL)
    class LocalModel:
        pass

    results = Registry.query(locality=LocalityLevel.LOCAL)
    assert len(results) == 1
    assert results[0]["name"] == "LocalModel"


def test_query_by_backward():
    """Test query by requires_backward."""
    Registry.clear()

    @register_model(name="GradModel", requires_backward=True)
    class GradModel:
        pass

    @register_model(name="BioModel", requires_backward=False)
    class BioModel:
        pass

    results = Registry.query(requires_backward=False)
    assert len(results) == 1
    assert results[0]["name"] == "BioModel"


def test_query_by_bio_score():
    """Test query by bio-plausibility score range."""
    Registry.clear()

    @register_model(name="LowBio", bio_plausibility_score=0.1)
    class LowBio:
        pass

    @register_model(name="HighBio", bio_plausibility_score=0.9)
    class HighBio:
        pass

    results = Registry.query(min_bio_score=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "HighBio"

    results = Registry.query(max_bio_score=0.5)
    assert len(results) == 1
    assert results[0]["name"] == "LowBio"


def test_query_tags():
    """Test query by tags."""
    Registry.clear()

    @register_model(name="TaggedModel", tags=["foo", "bar"])
    class TaggedModel:
        pass

    @register_model(name="OtherModel", tags=["baz"])
    class OtherModel:
        pass

    results = Registry.query(tags=["foo"])
    assert len(results) == 1
    assert results[0]["name"] == "TaggedModel"

    results = Registry.query(tags=["foo", "bar"])
    assert len(results) == 1

    results = Registry.query(tags=["foo", "nonexistent"])
    assert len(results) == 0


def test_query_category():
    """Test query by category."""
    Registry.clear()

    @register_model(name="ModelA")
    class ModelA:
        pass

    @register_optimizer(name="OptA")
    class OptA:
        pass

    results = Registry.query(category=ComponentCategory.MODEL)
    assert len(results) == 1
    assert results[0]["category"] == ComponentCategory.MODEL

    results = Registry.query(category=ComponentCategory.OPTIMIZER)
    assert len(results) == 1
    assert results[0]["category"] == ComponentCategory.OPTIMIZER


def test_component_metadata_defaults():
    """Test ComponentMetadata default values."""
    meta = ComponentMetadata(name="Test", category=ComponentCategory.MODEL)
    assert meta.bio_plausibility_score == 0.5
    assert meta.requires_backward is True
    assert meta.locality_level == LocalityLevel.GLOBAL
    assert Domain.VISION in meta.domains
    assert meta.memory_complexity == "O(N)"


def test_registry_metadata_on_class():
    """Test that metadata is attached to the registered class."""
    Registry.clear()

    @register_model(name="TestModel")
    class TestModel:
        pass

    assert hasattr(TestModel, "_registry_metadata")
    assert TestModel._registry_metadata.name == "TestModel"
    assert TestModel._registry_name == "TestModel"
    assert TestModel._registry_category == ComponentCategory.MODEL
