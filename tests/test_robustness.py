from unittest.mock import MagicMock, patch

import pytest
import torch

from bioplausible.scientist.robustness import RobustnessEvaluator


def test_robustness_evaluator_init():
    config = {"hidden_dim": 64}
    evaluator = RobustnessEvaluator("EqProp MLP", "mnist", config)
    assert evaluator.model_name == "EqProp MLP"
    assert evaluator.task_name == "mnist"
    assert evaluator.device in ["cpu", "cuda"]


@patch("bioplausible.scientist.robustness.create_task")
@patch("bioplausible.scientist.robustness.create_model")
def test_robustness_run_scratch(mock_create_model, mock_create_task):
    """Test running robustness check from scratch (no weights)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mock_task = MagicMock()
    mock_task.input_dim = 784
    mock_task.output_dim = 10
    mock_task.task_type = "vision"
    mock_task.get_batch.return_value = (
        torch.randn(32, 784, device=device),
        torch.randint(0, 10, (32,), device=device),
    )
    mock_create_task.return_value = mock_task

    mock_model = MagicMock()
    mock_model.train_step = None
    mock_model.return_value = torch.randn(32, 10, device=device)
    mock_model.inject_noise_and_relax.return_value = {"damping_percent": 85.0}
    mock_model.eval.return_value = None
    mock_create_model.return_value = mock_model

    config = {"hidden_dim": 64, "lr": 0.01}
    evaluator = RobustnessEvaluator("EqProp MLP", "mnist", config)

    score = evaluator.run()

    mock_task.create_trainer.assert_called_once()
    assert 0.0 <= score.get("robustness_score", 0.0) <= 1.0
