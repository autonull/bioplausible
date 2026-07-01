"""Tests for the KnowledgeBase system."""

import os
import tempfile
import time

import pytest

from bioplausible.knowledge import KnowledgeBase, KnowledgeEntry, create_knowledge_base


@pytest.fixture
def tmp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_kb.db")
        yield db_path


def test_knowledge_base_creation(tmp_db_path):
    """Test creating a KnowledgeBase."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    assert kb is not None
    assert os.path.exists(tmp_db_path)


def test_add_entry(tmp_db_path):
    """Test adding a knowledge entry."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    entry = KnowledgeEntry(
        id="TEST-001",
        topic="Test",
        model_family="test_model",
        finding="Test finding",
        details="Test details",
        confidence=0.9,
        tags=["test", "pytest"],
    )
    entry_id = kb.add_entry(entry)
    assert entry_id == "TEST-001"


def test_query_by_id(tmp_db_path):
    """Test querying by entry ID."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="TEST-002",
        topic="Test",
        model_family="test_model",
        finding="Test finding 2",
        details="Test details 2",
        confidence=0.8,
    )
    kb.add_entry(entry)

    retrieved = kb.get_by_id("TEST-002")
    assert retrieved is not None
    assert retrieved.finding == "Test finding 2"


def test_query_by_model_family(tmp_db_path):
    """Test querying by model family."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    for i in range(3):
        entry = KnowledgeEntry(
            id=f"MOD-{i:03d}",
            topic="Test",
            model_family=f"family_{i}",
            finding=f"Finding {i}",
            details=f"Details {i}",
            confidence=0.5 + i * 0.2,
        )
        kb.add_entry(entry)

    results = kb.query(model_family="family_1")
    assert len(results) == 1
    assert results[0].id == "MOD-001"


def test_query_by_tag(tmp_db_path):
    """Test querying by tag."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry1 = KnowledgeEntry(
        id="TAG-001",
        topic="A",
        model_family="m1",
        finding="f1",
        details="d1",
        confidence=0.5,
        tags=["alpha", "beta"],
    )
    entry2 = KnowledgeEntry(
        id="TAG-002",
        topic="B",
        model_family="m2",
        finding="f2",
        details="d2",
        confidence=0.5,
        tags=["alpha", "gamma"],
    )
    kb.add_entry(entry1)
    kb.add_entry(entry2)

    results = kb.query(tag="alpha")
    assert len(results) == 2

    results = kb.query(tag="beta")
    assert len(results) == 1
    assert results[0].id == "TAG-001"


def test_query_by_confidence(tmp_db_path):
    """Test querying by minimum confidence."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    for i in range(5):
        entry = KnowledgeEntry(
            id=f"CONF-{i:03d}",
            topic="Test",
            model_family="test_confidence",
            finding=f"f{i}",
            details="d",
            confidence=i * 0.25,
        )
        kb.add_entry(entry)

    results = kb.query(min_confidence=0.6, model_family="test_confidence")
    assert len(results) == 2  # 0.75 and 1.0


def test_add_experiment(tmp_db_path):
    """Test adding an experiment."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    exp_id = kb.add_experiment(
        name="test_exp",
        model_family="eqprop",
        task="mnist",
        config={"lr": 0.01, "epochs": 10},
        metrics={"accuracy": 0.95, "loss": 0.1},
    )

    assert exp_id is not None

    # Check it was added to experiments table
    exp = kb.get_experiment(exp_id)
    assert exp is not None
    assert exp["model_family"] == "eqprop"
    assert exp["task"] == "mnist"


def test_list_experiments(tmp_db_path):
    """Test listing experiments with filters."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    kb.add_experiment("exp1", "model_a", "mnist", {}, {"acc": 0.9})
    kb.add_experiment("exp2", "model_a", "cifar10", {}, {"acc": 0.8})
    kb.add_experiment("exp3", "model_b", "mnist", {}, {"acc": 0.7})

    results = kb.list_experiments(model_family="model_a")
    assert len(results) == 2

    results = kb.list_experiments(task="mnist")
    assert len(results) == 2


def test_get_stats(tmp_db_path):
    """Test getting knowledge base stats."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    stats = kb.get_stats()

    assert "total_entries" in stats
    assert "by_source" in stats
    assert "by_model_family" in stats
    assert "by_topic" in stats
    assert "total_experiments" in stats


def test_natural_language_query(tmp_db_path):
    """Test natural language query."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="NL-001",
        topic="Test",
        model_family="test_model",
        finding="The quick brown fox jumps over the lazy dog",
        details="This is a test sentence for natural language queries.",
        confidence=0.95,
        tags=["test", "nlq"],
    )
    kb.add_entry(entry)

    answer = kb.natural_language_query("What is the fox doing?")
    assert "brown fox" in answer or "test_model" in answer


def test_export_json(tmp_db_path):
    """Test exporting knowledge base to JSON."""
    kb = KnowledgeBase(db_path=tmp_db_path)

    entry = KnowledgeEntry(
        id="EXP-001",
        topic="Test",
        model_family="m",
        finding="test",
        details="test",
        confidence=0.5,
    )
    kb.add_entry(entry)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "export.json")
        kb.export_json(json_path)
        assert os.path.exists(json_path)

        import json

        with open(json_path) as f:
            data = json.load(f)
        assert len(data) >= 1


def test_create_knowledge_base_factory(tmp_db_path):
    """Test the factory function."""
    kb = create_knowledge_base(db_path=tmp_db_path)
    assert isinstance(kb, KnowledgeBase)


def test_seed_data_loading(tmp_db_path):
    """Test that seed data is loaded automatically."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    stats = kb.get_stats()
    assert stats["total_entries"] >= 3  # Seed data


def test_close(tmp_db_path):
    """Test close method."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    kb.close()  # Should not raise
