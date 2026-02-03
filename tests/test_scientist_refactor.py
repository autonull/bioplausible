
import unittest
from unittest.mock import MagicMock, patch
from enum import Enum

# Define minimal mocks/classes to simulate the environment
class PatientLevel(Enum):
    SMOKE = 1
    SHALLOW = 2
    STANDARD = 3
    CROSS_VAL = 4
    DEEP = 5

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
with patch("bioplausible.scientist.core.HyperoptStorage"), \
     patch("bioplausible.scientist.core.optuna"):
    from bioplausible.scientist.core import ScientistStrategy, ExperimentTask, PatientLevel

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

        with patch("bioplausible.scientist.core.MODEL_REGISTRY", self.mock_registry):
            candidates = self.strategy.generate_candidates()

        # Expect Smoke tests for ModelA (mnist, cifar10) and ModelB (tiny_shakespeare)
        # However, due to curriculum, only mnist and tiny_shakespeare (maybe)
        # Checking logic:
        # ModelA -> vision -> mnist, cifar10.
        # mnist prerequisites: None.
        # cifar10 prerequisites: mnist Standard.
        # So only ModelA-mnist-SMOKE should be generated.
        # ModelB -> lm -> tiny_shakespeare. Prereq: None.

        tasks = [(c.model_name, c.task_name, c.tier) for c in candidates]
        self.assertIn(("ModelA", "mnist", PatientLevel.SMOKE), tasks)
        self.assertIn(("ModelB", "tiny_shakespeare", PatientLevel.SMOKE), tasks)

        # Should NOT have cifar10 yet
        self.assertNotIn(("ModelA", "cifar10", PatientLevel.SMOKE), tasks)

    def test_plan_next_selects_best(self):
        """Test that plan_next selects the highest priority candidate."""
        # Mock generate_candidates
        task_low = ExperimentTask("M1", "T1", PatientLevel.SMOKE, "s1", 10.0)
        task_high = ExperimentTask("M2", "T2", PatientLevel.SMOKE, "s2", 100.0)

        with patch.object(self.strategy, 'generate_candidates', return_value=[task_low, task_high]):
            selected = self.strategy.plan_next()

        self.assertEqual(selected, task_high)

    def test_plan_next_no_candidates(self):
        """Test plan_next with no candidates."""
        with patch.object(self.strategy, 'generate_candidates', return_value=[]):
            selected = self.strategy.plan_next()

        self.assertIsNone(selected)

if __name__ == "__main__":
    unittest.main()
