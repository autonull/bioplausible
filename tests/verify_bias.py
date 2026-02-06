import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict
import random

from bioplausible.scientist.strategy import ScientistStrategy, PatientLevel
from bioplausible.scientist.task import ExperimentTask

# Mock Classes
class MockExperimentState:
    def __init__(self, progress=None):
        self.progress = progress or {}
    def get_progress(self):
        return self.progress

class MockModelSpec:
    def __init__(self, name, task_compat=None):
        self.name = name
        self.task_compat = task_compat

class TestBias(unittest.TestCase):
    def setUp(self):
        self.mock_state = MockExperimentState()
        self.strategy = ScientistStrategy(self.mock_state)

        self.mock_registry = [
            MockModelSpec("ModelA", ["vision"]),
            MockModelSpec("ModelB", ["lm"]),
            MockModelSpec("ModelC", ["rl"]),
        ]

    def test_curriculum_lookahead_bias(self):
        """
        Verify that a low-value task (TaskA) gets a priority boost if it leads
        to a high-value task (TaskB).
        """
        # Define a custom curriculum and weights for testing
        mock_tracks = {
            "test_track": ["task_a", "task_gap", "task_b"] # Increased gap to test lookahead
        }

        # task_a = 0.1, task_gap = 0.1, task_b = 0.5 (High value)
        # task_c = 0.15 (Dead end, but initially higher than task_a)
        mock_weights = {
            "task_a": 0.1,
            "task_gap": 0.1,
            "task_b": 0.5,
            "task_c": 0.15
        }

        # Mock ModelD supports task_a and task_c
        mock_reg = [MockModelSpec("ModelD", ["task_a", "task_c"])]

        with patch.object(self.strategy.curriculum, "TRACKS", mock_tracks), \
             patch("bioplausible.scientist.strategy.ScientistStrategy.TASK_WEIGHTS", mock_weights), \
             patch("bioplausible.scientist.strategy.MODEL_REGISTRY", mock_reg):

            candidates = self.strategy.generate_candidates()

        cand_a = next((c for c in candidates if c.task_name == "task_a"), None)
        cand_c = next((c for c in candidates if c.task_name == "task_c"), None)

        self.assertIsNotNone(cand_a)
        self.assertIsNotNone(cand_c)

        print(f"Task A Priority: {cand_a.priority}")
        print(f"Task C Priority: {cand_c.priority}")

        # Task A: Base 0.1. Leads to Task B (0.5) at dist 2.
        # Boost = (0.5 - 0.1) * (0.9 ** 2) = 0.4 * 0.81 = 0.324
        # Effective A = 0.424

        # Task C: Base 0.15. Dead end.
        # Effective C = 0.15

        # A should beat C, even though C starts higher.
        self.assertGreater(cand_a.priority, cand_c.priority)

if __name__ == "__main__":
    unittest.main()
