import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from bioplausible.hyperopt import PatientLevel
from bioplausible.scientist.strategy import ScientistStrategy


class TestStrategyTransfer(unittest.TestCase):
    def setUp(self):
        # Mock State and Curriculum
        self.mock_state = MagicMock()
        self.strategy = ScientistStrategy(self.mock_state)

        # We can use the real CurriculumManager (it's stateless logic)
        # But for the test we might want to ensure specific behavior.
        # Let's rely on the real one for integration logic since it was simple.

    def test_transfer_generation(self):
        # Setup: Model "mlp" has passed "mnist" with high accuracy
        model = "mlp"
        task = "mnist"

        # Create a mock trial with high accuracy
        mock_trial = SimpleNamespace(
            trial_id=101, accuracy=0.95, config={"hidden_dim": 64, "lr": 0.01}
        )

        # Mock progress data structure
        # progress[model][task][tier.value] -> dict with 'trials'
        progress = {
            model: {
                task: {
                    PatientLevel.STANDARD.value: {
                        "count": 5,
                        "best_acc": 0.95,
                        "trials": [mock_trial],
                    }
                },
                "fashion_mnist": {
                    PatientLevel.STANDARD.value: {"count": 0, "trials": []}
                },
            }
        }

        # Mock State.get_progress (not strictly needed if we pass progress directly to internal methods)
        # but _check_transfer_needed takes (stats, progress, model, task)

        stats = progress[model][task][PatientLevel.STANDARD.value]

        # Action
        transfer_task = self.strategy._check_transfer_needed(
            stats, progress, model, task
        )

        # Assertions
        self.assertIsNotNone(transfer_task)
        self.assertEqual(transfer_task.task_name, "fashion_mnist")
        self.assertTrue(transfer_task.is_transfer)
        self.assertEqual(transfer_task.transfer_from_trial, 101)
        self.assertEqual(transfer_task.fixed_config["freeze_layers"], True)

    def test_no_transfer_if_low_accuracy(self):
        model = "mlp"
        task = "mnist"
        mock_trial = SimpleNamespace(trial_id=102, accuracy=0.50, config={})

        progress = {
            model: {
                task: {
                    PatientLevel.STANDARD.value: {
                        "count": 5,
                        "best_acc": 0.50,
                        "trials": [mock_trial],
                    }
                }
            }
        }
        stats = progress[model][task][PatientLevel.STANDARD.value]

        transfer_task = self.strategy._check_transfer_needed(
            stats, progress, model, task
        )
        self.assertIsNone(transfer_task)

    def test_no_transfer_if_end_of_track(self):
        model = "mlp"
        task = "cifar100"  # Last in vision track
        mock_trial = SimpleNamespace(trial_id=103, accuracy=0.99, config={})

        progress = {
            model: {
                task: {
                    PatientLevel.STANDARD.value: {
                        "count": 5,
                        "best_acc": 0.99,
                        "trials": [mock_trial],
                    }
                }
            }
        }
        stats = progress[model][task][PatientLevel.STANDARD.value]

        transfer_task = self.strategy._check_transfer_needed(
            stats, progress, model, task
        )
        self.assertIsNone(transfer_task)


if __name__ == "__main__":
    unittest.main()
