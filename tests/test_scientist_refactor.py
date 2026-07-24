import unittest
from unittest.mock import MagicMock
from unittest.mock import patch


class MockModelSpec:
    def __init__(self, name, task_compat=None):
        self.name = name
        self.task_compat = task_compat


class MockExperimentTask:
    def __init__(self, model_name, task_name, tier, priority, **kwargs):
        self.model_name = model_name
        self.task_name = task_name
        self.tier = tier
        self.priority = priority
        for k, v in kwargs.items():
            setattr(self, k, v)


# Patching before importing core to handle dependencies
with (
    patch("bioplausible.execution.state.HyperoptStorage"),
    patch("bioplausible.execution.state.optuna"),
):
    from bioplausible.execution.strategy import ExecutionStrategy as ScientistStrategy
    from bioplausible.hyperopt import PatientLevel


class TestScientistRefactor(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock()
        self.strategy = ScientistStrategy(self.mock_state)

        # Mock MODEL_REGISTRY
        self.mock_registry = [
            MockModelSpec("ModelA", ["vision"]),
            MockModelSpec("ModelB", ["lm"]),
        ]

    def test_generate_candidates_smoke(self):
        """Test that smoke tests are generated when no progress exists."""
        self.mock_state.get_progress.return_value = {}

        with patch(
            "bioplausible.execution.strategy._MODEL_SPECS", self.mock_registry
        ):
            candidates = self.strategy.generate_candidates()

        # Expect Smoke tests for ModelA (digits) and ModelB (char_ngram)
        # Curriculum: vision -> digits (first)
        # Curriculum: lm -> char_ngram (first)

        tasks = [(c.model_name, c.task_name, c.tier) for c in candidates]

        # Verify initial vision task
        self.assertIn(("ModelA", "digits", PatientLevel.SMOKE), tasks)
        # Verify subsequent vision tasks are blocked by curriculum
        self.assertNotIn(("ModelA", "mnist", PatientLevel.SMOKE), tasks)
        self.assertNotIn(("ModelA", "cifar10", PatientLevel.SMOKE), tasks)

        # Verify initial LM task
        self.assertIn(("ModelB", "char_ngram", PatientLevel.SMOKE), tasks)
        # Verify subsequent LM tasks are blocked
        self.assertNotIn(("ModelB", "tiny_shakespeare", PatientLevel.SMOKE), tasks)

    def test_plan_next_selects_best(self):
        """Test that plan_next selects the highest priority candidate."""
        # Mock generate_candidates
        task_low = MockExperimentTask("M1", "T1", PatientLevel.SMOKE, 10.0)
        task_high = MockExperimentTask("M2", "T2", PatientLevel.SMOKE, 100.0)

        with patch.object(
            self.strategy, "generate_candidates", return_value=[task_low, task_high]
        ):
            selected = self.strategy.plan_next()

        self.assertEqual(selected, task_high)

    def test_plan_next_no_candidates(self):
        """Test plan_next with no candidates."""
        with patch.object(self.strategy, "generate_candidates", return_value=[]):
            selected = self.strategy.plan_next()

        self.assertIsNone(selected)


if __name__ == "__main__":
    unittest.main()
