import unittest
from unittest.mock import MagicMock, patch
from collections import defaultdict
import random

from bioplausible.scientist.strategy import ScientistStrategy, PatientLevel
from bioplausible.scientist.task import ExperimentTask

# Mock Classes
class MockExperimentState:
    def __init__(self, progress=None, recent_tasks=None):
        self.progress = progress or {}
        self.recent_tasks = recent_tasks or []

    def get_progress(self):
        return self.progress

    def get_recent_tasks(self, limit=10):
        return self.recent_tasks

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

    def test_diversity_mechanism(self):
        """
        Verify that a task which has run recently gets penalized.
        """
        # Set recent tasks to be all "task_a"
        self.mock_state.recent_tasks = ["task_a"] * 5

        # Define weights where Task A is normally much higher than Task B
        mock_weights = {
            "task_a": 0.5,
            "task_b": 0.2
        }

        mock_tracks = {
            "track_a": ["task_a"],
            "track_b": ["task_b"]
        }

        mock_reg = [MockModelSpec("ModelX", ["task_a", "task_b"])]

        with patch.object(self.strategy.curriculum, "TRACKS", mock_tracks), \
             patch("bioplausible.scientist.strategy.ScientistStrategy.TASK_WEIGHTS", mock_weights), \
             patch("bioplausible.scientist.strategy.MODEL_REGISTRY", mock_reg):

            candidates = self.strategy.generate_candidates()

        cand_a = next((c for c in candidates if c.task_name == "task_a"), None)
        cand_b = next((c for c in candidates if c.task_name == "task_b"), None)

        self.assertIsNotNone(cand_a)
        self.assertIsNotNone(cand_b)

        print(f"Task A Priority (Penalized): {cand_a.priority}")
        print(f"Task B Priority: {cand_b.priority}")

        # Base A = 0.5 * 5.0 * 100 = 250
        # Penalty A = 0.9^5 = 0.59
        # Final A = 147.6

        # Base B = 0.2 * 5.0 * 100 = 100

        # Task A might still be higher if gap is huge, but let's check relative reduction.
        # Without penalty, A would be 2.5x B.
        # With penalty, it should be closer.

        ratio = cand_a.priority / cand_b.priority
        self.assertLess(ratio, 2.0) # Should have dropped significantly from 2.5

        # Let's try 10 runs
        self.mock_state.recent_tasks = ["task_a"] * 10
        with patch.object(self.strategy.curriculum, "TRACKS", mock_tracks), \
             patch("bioplausible.scientist.strategy.ScientistStrategy.TASK_WEIGHTS", mock_weights), \
             patch("bioplausible.scientist.strategy.MODEL_REGISTRY", mock_reg):
            candidates_heavy = self.strategy.generate_candidates()

        cand_a_heavy = next((c for c in candidates_heavy if c.task_name == "task_a"), None)
        cand_b_heavy = next((c for c in candidates_heavy if c.task_name == "task_b"), None)

        # 0.9^10 = 0.34
        # A = 250 * 0.34 = 85
        # B = 100
        # Now B should win!

        print(f"Task A (10 runs): {cand_a_heavy.priority}")
        print(f"Task B (0 runs): {cand_b_heavy.priority}")

        self.assertGreater(cand_b_heavy.priority, cand_a_heavy.priority)

if __name__ == "__main__":
    unittest.main()
