from unittest.mock import MagicMock, patch

import pytest

from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.events import CompletedEvent, ProgressEvent
from bioplausible.pipeline.session import SessionState, TrainingSession


def test_training_config():
    config = TrainingConfig(
        task="vision", dataset="mnist", model="EqProp MLP", epochs=5
    )
    assert config.task == "vision"
    assert config.dataset == "mnist"
    assert config.epochs == 5
    assert config.batch_size == 64


@patch("bioplausible.pipeline.session.create_task")
@patch("bioplausible.pipeline.session.create_model")
@patch("bioplausible.pipeline.session.get_model_spec")
def test_training_session_flow(mock_get_spec, mock_create_model, mock_create_task):
    # Mocks
    mock_task = MagicMock()
    mock_task.input_dim = 10
    mock_task.output_dim = 2
    mock_task.task_type = "vision"
    mock_create_task.return_value = mock_task

    mock_trainer = MagicMock()
    mock_trainer.train_epoch.return_value = {"loss": 0.5, "accuracy": 0.8}
    mock_task.create_trainer.return_value = mock_trainer

    mock_model = MagicMock()
    mock_create_model.return_value = mock_model

    mock_spec = MagicMock()
    mock_get_spec.return_value = mock_spec

    # Config
    config = TrainingConfig(
        task="vision", dataset="mnist", model="EqProp MLP", epochs=2
    )

    # Session
    session = TrainingSession(config)
    assert session.state == SessionState.IDLE

    events = list(session.start())

    assert session.state == SessionState.COMPLETED
    assert len(events) == 3  # 2 progress + 1 completed
    assert isinstance(events[0], ProgressEvent)
    assert events[0].epoch == 0
    assert isinstance(events[2], CompletedEvent)

    mock_create_task.assert_called_once()
    mock_task.setup.assert_called_once()
    mock_create_model.assert_called_once()
    mock_task.create_trainer.assert_called_once()
    assert mock_trainer.train_epoch.call_count == 2
