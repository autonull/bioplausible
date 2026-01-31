"""
Tests for the AutoScientist system.
"""

import pytest
import os
import shutil
import tempfile
import json
import time
import sys
from unittest.mock import MagicMock, patch
from bioplausible.scientist.core import ExperimentState, ScientistStrategy, PatientLevel, AutoScientist, ResourceMonitor
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

    strategy = ScientistStrategy(state)

    # Check if verification logic catches it
    progress = state.get_progress()
    stats = strategy._get_stats(progress, model, task_name, PatientLevel.STANDARD)

    task = strategy._check_verification_needed(stats, model, task_name, PatientLevel.STANDARD)

    assert task is not None
    assert task.fixed_config is not None
    assert task.fixed_config["learning_rate"] == 0.01
    assert task.priority > 90.0 # High priority

def test_auto_scientist_robustness():
    """Test graceful handling of failures."""

    # Mock everything
    with patch('bioplausible.scientist.core.ExperimentState') as MockState, \
         patch('bioplausible.scientist.core.ScientistStrategy') as MockStrategy, \
         patch('bioplausible.scientist.core.run_single_trial_task') as mock_run, \
         patch('bioplausible.scientist.core.ResourceMonitor') as MockResource: # Need to mock resource too

        # Setup mocks
        mock_strategy = MockStrategy.return_value
        MockResource.return_value.should_pause.return_value = False

        task = MagicMock()
        task.model_name = "Test"
        task.task_name = "vision"
        task.tier = PatientLevel.SMOKE
        task.fixed_config = None
        task.study_name = "test"
        task.priority = 100.0 # Fix format error

        mock_strategy.plan_next.return_value = task

        # Simulate Exception
        mock_run.side_effect = Exception("Simulated Crash")

        # Override sleep to run fast
        with patch('time.sleep', return_value=None):

            # Let's subclass to break loop
            class TestScientist(AutoScientist):
                def __init__(self):
                    # We need to manually init because we patched classes used in __init__
                    # Actually patching classes patches them for the module.
                    # But we need to make sure we don't call real DB if not desired.
                    # With ExperimentState patched, DB_PATH is passed to mock.
                    super().__init__()

                def _signal_handler(self, sig, frame):
                    pass

            test_sci = TestScientist()
            test_sci.strategy = mock_strategy # Inject mock

            def stop_loop():
                test_sci.running = False
                return task

            mock_strategy.plan_next.side_effect = stop_loop

            # Run
            test_sci.run()

            # Check if consecutive failures increased
            assert test_sci.consecutive_failures == 1

def test_resource_monitor():
    """Test resource throttling logic."""
    monitor = ResourceMonitor(cpu_limit=50.0)

    # We must patch where it's USED or IMPORTED.
    # ResourceMonitor imports psutil.
    # So we patch bioplausible.scientist.core.psutil

    with patch('bioplausible.scientist.core.psutil') as mock_psutil:

        # Case 1: Low usage
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value.percent = 10.0
        assert not monitor.should_pause()

        # Case 2: High CPU
        mock_psutil.cpu_percent.return_value = 90.0
        assert monitor.should_pause()

def test_reporter_robustness(temp_db):
    """Test that reporter continues even if plotting fails."""
    storage = HyperoptStorage(temp_db)
    storage.create_trial("TestModel", {"tier": "smoke", "task": "vision", "lr": 0.01})
    trials = storage.get_all_trials()
    storage.update_trial(trials[0].trial_id, status="completed", accuracy=0.8, param_count=1.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        reporter = ScientistReporter(temp_db)

        # Mock plt to raise exception
        with patch('matplotlib.pyplot.savefig', side_effect=Exception("Plot Error")):
            reporter.generate_report(tmpdir)

        # Report should still exist
        assert os.path.exists(os.path.join(tmpdir, "index.md"))
