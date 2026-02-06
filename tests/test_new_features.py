
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
from bioplausible.hyperopt.tasks import VisionTask, create_task
from bioplausible.scientist.robustness import RobustnessEvaluator

# Mock dataset class
class MockDataset:
    def __init__(self, size=100, features=10, classes=2):
        self.data = torch.rand(size, 1, 5, 2) # N, C, H, W (10 pixels)
        self.targets = torch.randint(0, classes, (size,))
        # For bulk loading logic compatibility
        # data needs to be numpy or tensor
        # targets needs to be list or tensor

@pytest.fixture
def mock_vision_dataset():
    with patch("bioplausible.hyperopt.tasks.get_vision_dataset") as mock_get:
        # Train set
        train_ds = MockDataset(size=100)
        # Test set
        test_ds = MockDataset(size=20)

        # Configure mock side_effect or return_value
        def side_effect(name, train=True, **kwargs):
            return train_ds if train else test_ds

        mock_get.side_effect = side_effect
        yield mock_get

def test_vision_task_fold_splitting(mock_vision_dataset):
    # Test Fold 0 vs Fold 1
    task0 = VisionTask(name="mnist", quick_mode=True, fold=0, num_folds=5)
    task0.setup()

    task1 = VisionTask(name="mnist", quick_mode=True, fold=1, num_folds=5)
    task1.setup()

    # Train sets should be different (different 80% of data)
    assert len(task0.train_x) == 80 # 80% of 100
    assert len(task1.train_x) == 80

    # Check if they are different
    # Sum of first pixel of first sample
    # Not guaranteed to be different if data is random and duplicated, but likely distinct sets.
    # Check indices? But we don't expose indices.
    # Check overlap.

    # Val sets should be disjoint for 5 folds (20% each)
    # Val set 0 and Val set 1 should have 0 overlap if KFold works

    # Check intersection of tensors is hard without unique IDs.
    # But we mocked data with random float, likely unique.

    # Just check that val_x of task0 is not equal to val_x of task1
    assert not torch.equal(task0.val_x, task1.val_x)

def test_vision_task_data_fraction(mock_vision_dataset):
    # Test Data Fraction
    task = VisionTask(name="mnist", quick_mode=True, data_fraction=0.1)
    task.setup()

    # 10% of 100 is 10
    assert len(task.train_x) == 10

    task2 = VisionTask(name="mnist", quick_mode=True, data_fraction=0.5)
    task2.setup()
    assert len(task2.train_x) == 50

def test_robustness_evaluator_execution():
    # Mock model and task
    model = nn.Linear(10, 2)
    # Patch Task creation inside Evaluator
    with patch("bioplausible.scientist.robustness.create_task") as mock_create_task:
        mock_task = MagicMock()
        mock_task.input_dim = 10
        mock_task.output_dim = 2
        mock_task.task_type = "vision"
        mock_task.get_batch.return_value = (torch.randn(32, 10), torch.randint(0, 2, (32,)))

        mock_create_task.return_value = mock_task

        with patch("bioplausible.scientist.robustness.get_model_spec") as mock_spec:
            with patch("bioplausible.scientist.robustness.create_model") as mock_create_model:
                mock_create_model.return_value = model

                evaluator = RobustnessEvaluator("test_model", "mnist", {"hidden_dim": 10})

                # We need to mock _test_* methods if we don't want to run them fully,
                # but we want to verify they run.
                # Just verify run() completes and returns score.

                score = evaluator.run()
                assert 0.0 <= score <= 1.0
                assert isinstance(score, float)
