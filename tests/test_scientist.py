"""
Tests for the AutoScientist system.
"""

import pytest
import os
import shutil
import tempfile
import json
from bioplausible.scientist.core import ExperimentState, ScientistStrategy, PatientLevel
from bioplausible.scientist.reporting import ScientistReporter
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.models.registry import MODEL_REGISTRY

@pytest.fixture
def temp_db():
    """Create a temp DB path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_strategy_empty_db(temp_db):
    """Test strategy with an empty database."""
    state = ExperimentState(temp_db)
    strategy = ScientistStrategy(state)

    task = strategy.plan_next()

    assert task is not None
    assert task.tier == PatientLevel.SMOKE
    assert task.priority >= 80.0
    # Should pick one of the models
    assert any(m.name == task.model_name for m in MODEL_REGISTRY)

def test_strategy_verification_scheduling(temp_db):
    """Test that high performance in Standard tier triggers verification (repeats)."""
    state = ExperimentState(temp_db)
    storage = HyperoptStorage(temp_db)

    model = MODEL_REGISTRY[0].name
    task_name = "vision"

    # Create a high-performing STANDARD trial
    config = {
        "tier": "standard",
        "task": task_name,
        "model": model,
        "learning_rate": 0.01, # Specific param
        "beta": 0.5
    }

    # Add successful STANDARD trial
    storage.create_trial(model, config)
    trials = storage.get_all_trials()
    storage.update_trial(trials[0].trial_id, status="completed", accuracy=0.95) # Very high acc

    # We also need enough STANDARD trials to pass the count check?
    # No, verification check happens *before* count check in plan_next.
    # But get_stats needs to find it.

    strategy = ScientistStrategy(state)

    # Force strategy to look at this model/task
    # Since plan_next iterates all models, it should find this one.
    # However, other models might have higher priority (SMOKE tests).
    # To test logic specifically, we can call internal check or fill DB with smoke tests.

    # Let's call internal _check_verification_needed directly to verify logic
    progress = state.get_progress()
    stats = strategy._get_stats(progress, model, task_name, PatientLevel.STANDARD)

    task = strategy._check_verification_needed(stats, model, task_name, PatientLevel.STANDARD)

    assert task is not None
    assert task.fixed_config is not None
    assert task.fixed_config["learning_rate"] == 0.01
    assert task.priority > 90.0 # High priority

def test_reporter_smoke(temp_db):
    """Test reporter generation (does not crash)."""
    storage = HyperoptStorage(temp_db)
    storage.create_trial("TestModel", {"tier": "smoke", "task": "vision", "lr": 0.01})
    trials = storage.get_all_trials()
    storage.update_trial(trials[0].trial_id, status="completed", accuracy=0.8, param_count=1.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = ScientistReporter(temp_db)
        reporter.generate_report(tmpdir)

        assert os.path.exists(os.path.join(tmpdir, "index.md"))
        assert os.path.exists(os.path.join(tmpdir, "images"))
