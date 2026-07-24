import unittest
from unittest.mock import patch


class MockExperimentState:
    def __init__(self, progress=None, recent_tasks=None):
        self.progress = progress or {}
        self.recent_tasks = recent_tasks or []

    def get_progress(self):
        return self.progress

    def get_recent_tasks(self, limit=10):
        return self.recent_tasks

    def get_recent_models(self, limit=10):
        return []


class MockModelSpec:
    def __init__(self, name, task_compat=None):
        self.name = name
        self.task_compat = task_compat


class TestBias(unittest.TestCase):
    def setUp(self):
        from bioplausible.execution.strategy import ExecutionStrategy

        self.mock_state = MockExperimentState()
        self.strategy = ExecutionStrategy(self.mock_state)

        self.mock_registry = [
            MockModelSpec("ModelA", ["vision"]),
            MockModelSpec("ModelB", ["lm"]),
            MockModelSpec("ModelC", ["rl"]),
        ]

    def test_diversity_mechanism(self):
        """Verify that a task which has run recently gets penalized."""
        self.mock_state.recent_tasks = ["task_a"] * 5
        mock_weights = {"task_a": 0.5, "task_b": 0.2}
        mock_tracks = {"track_a": ["task_a"], "track_b": ["task_b"]}
        mock_reg = [MockModelSpec("ModelX", ["task_a", "task_b"])]

        with (
            patch.object(self.strategy.curriculum, "TRACKS", mock_tracks),
            patch.object(
                self.strategy, "TASK_WEIGHTS", mock_weights
            ),
            patch(
                "bioplausible.execution.strategy._MODEL_SPECS", mock_reg
            ),
        ):
            candidates = self.strategy.generate_candidates()

        cand_a = next((c for c in candidates if c.task_name == "task_a"), None)
        cand_b = next((c for c in candidates if c.task_name == "task_b"), None)

        self.assertIsNotNone(cand_a)
        self.assertIsNotNone(cand_b)

        if cand_a is not None and cand_b is not None and cand_b.priority > 0:
            ratio = cand_a.priority / cand_b.priority
            print(f"Task A Priority (Penalized): {cand_a.priority}")
            print(f"Task B Priority: {cand_b.priority}")
            self.assertLess(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
