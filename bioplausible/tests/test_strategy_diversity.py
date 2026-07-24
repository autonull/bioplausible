import unittest
from unittest.mock import MagicMock

from bioplausible.execution.task import ExperimentTask
from bioplausible.hyperopt import PatientLevel


class TestStrategyDiversity(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock()
        self.strategy = ScientistStrategy(self.mock_state)

    def test_model_diversity_penalty(self):
        # Setup: Mock recent models
        # "model_A" has been run 3 times recently
        # "model_B" has been run 0 times recently
        self.mock_state.get_recent_models.return_value = [
            "model_A",
            "model_A",
            "model_A",
        ]
        self.mock_state.get_recent_tasks.return_value = []  # No task penalty for simplicity

        # Create candidates
        candidate_A = ExperimentTask(
            model_name="model_A",
            task_name="task_X",
            tier=PatientLevel.SMOKE,
            study_name="study_A",
            priority=100.0,
        )

        candidate_B = ExperimentTask(
            model_name="model_B",
            task_name="task_X",
            tier=PatientLevel.SMOKE,
            study_name="study_B",
            priority=100.0,
        )

        candidates = [candidate_A, candidate_B]

        # Apply prioritization
        self.strategy._apply_prioritization(candidates)

        # Assertions
        # candidate_A should have been penalized
        # Penalty is 0.8 ** count. count=3. 0.8^3 = 0.512.
        # However, _apply_prioritization also applies task weights.
        # task_X weight is likely default 0.10 if not in TASK_WEIGHTS.
        # Let's check TASK_WEIGHTS for "task_X" (it's not there, so 0.10).
        # And future boost (likely 0.0).
        # So effective_weight = 0.10.
        # base priority multiplier = effective_weight * 5.0 = 0.5.

        # So expected priority for B (no model penalty):
        # 100.0 * 0.5 = 50.0

        # Expected priority for A (model penalty applied):
        # 100.0 * 0.5 * (0.8 ** 3) = 100 * 0.5 * 0.512 = 25.6

        # Let's just assert A < B since exact math depends on TASK_WEIGHTS implementation details
        self.assertLess(candidate_A.priority, candidate_B.priority)

        # Verify the penalty is roughly what we expect (allowing for floating point and weights)
        ratio = candidate_A.priority / candidate_B.priority
        self.assertAlmostEqual(ratio, 0.8**3, places=2)

    def test_task_diversity_penalty(self):
        # Verify task penalty still works
        self.mock_state.get_recent_models.return_value = []
        self.mock_state.get_recent_tasks.return_value = [
            "task_Y",
            "task_Y",
        ]  # Run twice

        candidate_X = ExperimentTask(
            model_name="model_C",
            task_name="task_X",
            tier=PatientLevel.SMOKE,
            study_name="study_X",
            priority=100.0,
        )

        candidate_Y = ExperimentTask(
            model_name="model_C",
            task_name="task_Y",
            tier=PatientLevel.SMOKE,
            study_name="study_Y",
            priority=100.0,
        )

        # Assume task_X and task_Y have same weight for fair comparison
        # We can mock TASK_WEIGHTS on the instance or choose tasks with same weight
        # Let's choose tasks not in TASK_WEIGHTS, so they get default 0.10

        candidates = [candidate_X, candidate_Y]
        self.strategy._apply_prioritization(candidates)

        self.assertLess(candidate_Y.priority, candidate_X.priority)

        ratio = candidate_Y.priority / candidate_X.priority
        self.assertAlmostEqual(ratio, 0.9**2, places=2)


if __name__ == "__main__":
    unittest.main()
