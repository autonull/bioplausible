"""
Tests for the AutoScientist system.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from bioplausible.hyperopt import PatientLevel
from bioplausible.hyperopt.parallel_runner import ParallelTrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.resources import ResourceMonitor
from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.strategy import ScientistStrategy
from bioplausible.scientist.task import ExperimentTask


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


def test_strategy_timeout_constraints(temp_db):
    """Test that frequent timeouts trigger model constraints."""
    state = ExperimentState(temp_db)
    strategy = ScientistStrategy(state)

    # Mock the failure analysis to simulate frequent timeouts
    mock_analysis = {
        "recommendations": [
            {"issue": "Frequent timeouts", "affected_models": [MODEL_REGISTRY[0].name]}
        ]
    }

    with patch.object(state, "get_failure_analysis", return_value=mock_analysis):
        # We also need to mock get_progress so we can enter generation loop
        # We can just return empty progress, so it generates Smoke tier tasks
        with patch.object(state, "get_progress", return_value={}):
            candidates = strategy.generate_candidates()

            # Find the candidate for the affected model
            affected_candidates = [
                c for c in candidates if c.model_name == MODEL_REGISTRY[0].name
            ]

            assert len(affected_candidates) > 0

            for task in affected_candidates:
                assert task.constraints is not None
                assert task.constraints.get("max_hidden_dim") == 256
                assert task.constraints.get("max_num_layers") == 6


def test_strategy_verification_scheduling(temp_db):
    """Test that high performance in Standard tier triggers verification (repeats)."""
    state = ExperimentState(temp_db)
    storage = HyperoptStorage(temp_db)

    model = MODEL_REGISTRY[0].name
    task_name = "mnist"

    # Create a high-performing STANDARD trial
    config = {
        "tier": "standard",
        "task": task_name,
        "model": model,
        "learning_rate": 0.01,  # Specific param
        "beta": 0.5,
    }

    # Add successful STANDARD trial
    storage.create_trial(model, config)
    trials = storage.get_all_trials()
    storage.update_trial(
        trials[0].trial_id, status="completed", accuracy=0.95
    )  # Very high acc

    strategy = ScientistStrategy(state)

    # Check if verification logic catches it
    progress = state.get_progress()
    stats = strategy._get_stats(progress, model, task_name, PatientLevel.STANDARD)

    task = strategy._check_verification_needed(
        stats, model, task_name, PatientLevel.STANDARD
    )

    assert task is not None
    assert task.fixed_config is not None
    assert task.fixed_config["learning_rate"] == 0.01
    assert task.priority > 90.0  # High priority


def test_auto_scientist_robustness():
    """Test graceful handling of failures."""

    # Mock everything
    with (
        patch("bioplausible.scientist.core.ExperimentState") as MockState,
        patch("bioplausible.scientist.core.ScientistStrategy") as MockStrategy,
        patch("bioplausible.scientist.core.run_single_trial_task") as mock_run,
        patch("bioplausible.scientist.core.ResourceMonitor") as MockResource,
    ):  # Need to mock resource too

        # Setup mocks
        mock_strategy = MockStrategy.return_value
        MockResource.return_value.should_pause.return_value = False

        task = MagicMock()
        task.model_name = "Test"
        task.task_name = "vision"
        task.tier = PatientLevel.SMOKE
        task.fixed_config = None
        task.study_name = "test"
        task.priority = 100.0  # Fix format error

        mock_strategy.plan_next.return_value = task

        # Simulate Exception
        mock_run.side_effect = Exception("Simulated Crash")

        # Override sleep to run fast
        with patch("time.sleep", return_value=None):

            # Subclass to break loop
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
            test_sci.strategy = mock_strategy  # Inject mock

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
    # So we patch bioplausible.scientist.resources.psutil

    with patch("bioplausible.scientist.resources.psutil") as mock_psutil:

        # Case 1: Low usage
        mock_psutil.cpu_percent.return_value = 10.0
        mock_psutil.virtual_memory.return_value.percent = 10.0
        assert not monitor.should_pause()

        # Case 2: High CPU
        mock_psutil.cpu_percent.return_value = 90.0
        assert monitor.should_pause()


def test_resource_monitor_multi_gpu():
    """Test multi-GPU resource throttling logic."""
    monitor = ResourceMonitor(gpu_limit=90.0)

    with (
        patch(
            "bioplausible.scientist.resources.torch.cuda.is_available",
            return_value=True,
        ),
        patch(
            "bioplausible.scientist.resources.torch.cuda.device_count", return_value=2
        ),
        patch(
            "bioplausible.scientist.resources.torch.cuda.mem_get_info"
        ) as mock_mem_get_info,
    ):

        # GPU 0: 10% used, GPU 1: 10% used (Low usage)
        def low_usage(device_id):
            return 90, 100

        mock_mem_get_info.side_effect = low_usage
        assert not monitor._check_gpu_overload()

        # GPU 0: 10% used, GPU 1: 95% used (Overload on GPU 1)
        def high_usage_gpu_1(device_id):
            if device_id == 0:
                return 90, 100
            else:
                return 5, 100

        mock_mem_get_info.side_effect = high_usage_gpu_1
        # Should NOT pause because GPU 0 is still free
        assert not monitor._check_gpu_overload()

        # GPU 0: 95% used, GPU 1: 95% used (Overload on all GPUs)
        def high_usage_all(device_id):
            return 5, 100

        mock_mem_get_info.side_effect = high_usage_all
        assert monitor._check_gpu_overload()


def test_parallel_trial_runner(temp_db):
    """Test the ParallelTrialRunner logic."""
    runner = ParallelTrialRunner(num_workers=2, db_path=temp_db)

    task1 = ExperimentTask(
        model_name="ModelA",
        task_name="vision",
        tier=PatientLevel.SMOKE,
        study_name="study1",
        priority=10.0,
    )
    task2 = ExperimentTask(
        model_name="ModelB",
        task_name="vision",
        tier=PatientLevel.SMOKE,
        study_name="study2",
        priority=20.0,
    )

    config1 = {"lr": 0.01}
    config2 = {"lr": 0.02}

    # multiprocessing mock is tricky, so we'll mock the multiprocessing.Pool itself
    with patch("multiprocessing.Pool") as MockPool:
        mock_pool_instance = MockPool.return_value.__enter__.return_value
        # Mock map to just apply the wrapped worker sequentially for testing
        mock_pool_instance.map.side_effect = lambda func, iterable: [
            func(item) for item in iterable
        ]

        with patch(
            "bioplausible.hyperopt.parallel_runner.run_single_trial_task"
        ) as mock_run:
            mock_run.side_effect = [{"accuracy": 0.9}, {"accuracy": 0.8}]

            results = runner.run_batch([task1, task2], [config1, config2])

            assert len(results) == 2
            assert results[0] == {"accuracy": 0.9}
            assert results[1] == {"accuracy": 0.8}

            assert mock_run.call_count == 2


def test_auto_scientist_safe_mode_diagnostic_failure(temp_db):
    """Test that AutoScientist terminates when the diagnostic task fails in Safe Mode."""
    with (
        patch("bioplausible.scientist.core.ExperimentState"),
        patch("bioplausible.scientist.core.ScientistStrategy"),
        patch("bioplausible.scientist.core.ResourceMonitor") as MockResource,
    ):
        MockResource.return_value.should_pause.return_value = False

        # Override to prevent infinite loops and sleep
        with patch("time.sleep", return_value=None):

            class TestScientist(AutoScientist):
                def __init__(self):
                    super().__init__(db_path=temp_db)

                def _signal_handler(self, sig, frame):
                    pass

            test_sci = TestScientist()
            # Force conditions for Safe Mode
            test_sci.consecutive_failures = test_sci.MAX_CONSECUTIVE_FAILURES

            # Mock the diagnostic to fail
            with patch.object(test_sci, "_run_diagnostic_task", return_value=False):
                # mock check_failures_pause which is called in the loop
                # Actually, check_failures_pause is what we want to test!

                # We can just call _check_failures_pause directly and assert its behavior
                should_pause = test_sci._check_failures_pause()

                # Diagnostic failed -> Terminate -> running should be False, and it returns True to "continue" the outer loop
                assert should_pause is True
                assert test_sci.running is False


def test_auto_scientist_safe_mode_diagnostic_success(temp_db):
    """Test that AutoScientist recovers when the diagnostic task succeeds."""
    with (
        patch("bioplausible.scientist.core.ExperimentState"),
        patch("bioplausible.scientist.core.ScientistStrategy"),
        patch("bioplausible.scientist.core.ResourceMonitor") as MockResource,
    ):
        MockResource.return_value.should_pause.return_value = False

        with patch("time.sleep", return_value=None):

            class TestScientist(AutoScientist):
                def __init__(self):
                    super().__init__(db_path=temp_db)

                def _signal_handler(self, sig, frame):
                    pass

            test_sci = TestScientist()
            test_sci.consecutive_failures = test_sci.MAX_CONSECUTIVE_FAILURES
            test_sci.running = True

            # Mock the diagnostic to succeed
            with patch.object(test_sci, "_run_diagnostic_task", return_value=True):
                should_pause = test_sci._check_failures_pause()

                # Diagnostic succeeded -> Recover -> failures reset to 0, running stays True, returns False
                assert should_pause is False
                assert test_sci.running is True
                assert test_sci.consecutive_failures == 0


def test_inject_tier_config():
    """Test that AutoScientist injects tier and metadata config correctly."""
    # We don't need a DB for this unit test if we just test the method directly
    # but AutoScientist needs db_path in init, so we pass a dummy string
    with (
        patch("bioplausible.scientist.core.ExperimentState"),
        patch("bioplausible.scientist.core.ScientistStrategy"),
    ):
        scientist = AutoScientist(db_path="dummy.db")

        # Basic task
        task_standard = ExperimentTask(
            model_name="ModelA",
            task_name="vision",
            tier=PatientLevel.STANDARD,
            study_name="study1",
            priority=10.0,
        )
        config1 = {}
        scientist._inject_tier_config(config1, task_standard)
        assert config1["tier"] == PatientLevel.STANDARD.value
        assert config1["model"] == "ModelA"
        assert config1["task"] == "vision"
        assert "epochs" in config1
        assert "batch_size" in config1
        assert not config1.get("is_verification")

        # Verification / Fixed config
        task_verify = ExperimentTask(
            model_name="ModelA",
            task_name="vision",
            tier=PatientLevel.DEEP,
            study_name="study1",
            priority=10.0,
            fixed_config={"learning_rate": 0.01},
            verification_of_trial_id=42,
        )
        config2 = {}
        scientist._inject_tier_config(config2, task_verify)
        assert config2["is_verification"] is True
        assert config2["verified_trial_id"] == 42

        # Ablation Task
        task_ablate = ExperimentTask(
            model_name="ModelA",
            task_name="vision",
            tier=PatientLevel.SHALLOW,
            study_name="study1",
            priority=10.0,
            is_ablation=True,
            ablation_param="dropout",
        )
        config3 = {}
        scientist._inject_tier_config(config3, task_ablate)
        assert config3["is_ablation"] is True
        assert config3["ablation_param"] == "dropout"
        assert config3["save_artifacts"] is True

def test_strategy_end_to_end_promotion(temp_db):
    """Test full multi-tier promotion without being constrained prematurely."""
    state = ExperimentState(temp_db)
    storage = HyperoptStorage(temp_db)

    model = "Backprop Baseline"
    task_name = "mnist"

    # Set up some SMOKE and SHALLOW tier successes
    # Backprop Baseline handles mnist, others might be restricted by curriculum
    for _ in range(5):
        config = {
            "tier": "smoke",
            "task": "digits",
            "model": model,
            "learning_rate": 0.01,
        }
        trial_id = storage.create_trial(model, config)
        storage.update_trial(trial_id, status="completed", accuracy=0.85)

    for _ in range(15):
        config = {
            "tier": "shallow",
            "task": "digits",
            "model": model,
            "learning_rate": 0.01,
        }
        trial_id = storage.create_trial(model, config)
        storage.update_trial(trial_id, status="completed", accuracy=0.92)

    strategy = ScientistStrategy(state)
    candidates = strategy.generate_candidates()

    # Model should be promoted to STANDARD
    model_candidates = [
        c for c in candidates if c.model_name == model and c.task_name == "digits"
    ]
    assert len(model_candidates) > 0

    standard_candidate = next((c for c in model_candidates if c.tier == PatientLevel.STANDARD), None)
    assert standard_candidate is not None
