"""
Tests for Continual Learning (Split-MNIST) Strategy.
"""

from unittest.mock import MagicMock

import pytest

from bioplausible.scientist.state import ExperimentState
from bioplausible.scientist.strategy import ScientistStrategy


@pytest.fixture
def mock_strategy():
    state = MagicMock(spec=ExperimentState)
    strategy = ScientistStrategy(state)
    return strategy


def create_mock_trial(trial_id, accuracy, config):
    trial = MagicMock()
    trial.trial_id = trial_id
    trial.accuracy = accuracy
    trial.config = config
    trial.status = "completed"
    return trial


def test_cl_no_start_if_mnist_poor(mock_strategy):
    """If base MNIST is poor (< 0.95), do not start CL."""
    progress = {
        "TestModel": {
            "mnist": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.80,
                    "trials": [create_mock_trial(1, 0.80, {"task": "mnist"})],
                }
            }
        }
    }

    task = mock_strategy._check_continual_learning_needed(
        progress["TestModel"]["mnist"]["standard"], progress, "TestModel", "mnist"
    )
    assert task is None


def test_cl_start_step0(mock_strategy):
    """If base MNIST is good, start Step 0 (mnist_01)."""
    best_config = {"task": "mnist", "lr": 0.01}
    progress = {
        "TestModel": {
            "mnist": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.96,
                    "trials": [create_mock_trial(1, 0.96, best_config)],
                }
            }
        }
    }

    task = mock_strategy._check_continual_learning_needed(
        progress["TestModel"]["mnist"]["standard"], progress, "TestModel", "mnist"
    )

    assert task is not None
    assert task.task_name == "mnist_01"
    assert task.is_continual is True
    assert task.continual_step == 0
    # Step 0 copies config but doesn't transfer weights (fresh start on subset)
    assert task.fixed_config["lr"] == 0.01


def test_cl_step1_transfer(mock_strategy):
    """If Step 0 is good, start Step 1 (mnist_23) with transfer."""
    mnist_config = {"task": "mnist", "lr": 0.01}
    step0_config = {"task": "mnist_01", "lr": 0.01, "is_continual": True}

    progress = {
        "TestModel": {
            "mnist": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.96,
                    "trials": [create_mock_trial(1, 0.96, mnist_config)],
                }
            },
            "mnist_01": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.90,  # Good enough (>0.80)
                    "trials": [create_mock_trial(2, 0.90, step0_config)],
                }
            },
        }
    }

    task = mock_strategy._check_continual_learning_needed(
        progress["TestModel"]["mnist"]["standard"], progress, "TestModel", "mnist"
    )

    assert task is not None
    assert task.task_name == "mnist_23"
    assert task.is_continual is True
    assert task.continual_step == 1
    assert task.transfer_from_trial == 2  # Should transfer from Step 0 trial
    assert task.fixed_config["transfer_from"] == 2
    assert task.fixed_config["freeze_layers"] is False


def test_cl_fail_previous_step(mock_strategy):
    """If Step 0 failed (<0.80), do not proceed to Step 1."""
    mnist_config = {"task": "mnist", "lr": 0.01}
    step0_config = {"task": "mnist_01", "lr": 0.01, "is_continual": True}

    progress = {
        "TestModel": {
            "mnist": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.96,
                    "trials": [create_mock_trial(1, 0.96, mnist_config)],
                }
            },
            "mnist_01": {
                "standard": {
                    "count": 1,
                    "best_acc": 0.50,  # Failed
                    "trials": [create_mock_trial(2, 0.50, step0_config)],
                }
            },
        }
    }

    task = mock_strategy._check_continual_learning_needed(
        progress["TestModel"]["mnist"]["standard"], progress, "TestModel", "mnist"
    )

    # Should return None because Step 0 failed, so we can't do Step 1.
    # It also shouldn't return Step 0 again because count > 0.
    assert task is None
