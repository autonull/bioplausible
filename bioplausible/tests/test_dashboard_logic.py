import unittest
from unittest.mock import patch

from bioplausible.execution.dashboard import Dashboard


class TestDashboardLogic(unittest.TestCase):
    def setUp(self):
        # We need to mock rich console/live to prevent actual output
        with (
            patch("bioplausible.scientist.dashboard.Console"),
            patch("bioplausible.scientist.dashboard.Live"),
        ):
            self.dashboard = Dashboard()

    def test_complete_trial_stores_metrics(self):
        # Setup a current trial
        self.dashboard.set_trial("123", "mlp", "digits", "standard", {})

        # Complete it with metrics
        metrics = {"accuracy": 0.95, "loss": 0.1, "robustness_score": 0.85}
        self.dashboard.complete_trial("completed", metrics)

        # Verify
        last_trial = self.dashboard.recent_trials[-1]
        self.assertEqual(last_trial["id"], "123")
        self.assertEqual(last_trial["accuracy"], 0.95)
        self.assertEqual(last_trial["metrics"], metrics)
        self.assertEqual(last_trial["status"], "completed")

    def test_update_handles_robustness_metrics(self):
        # Verify update() runs without error when robustness metrics are present

        # Add a trial with robustness
        self.dashboard.recent_trials.append(
            {
                "id": "123",
                "model": "mlp",
                "task": "digits",
                "accuracy": 0.95,
                "status": "completed",
                "metrics": {"robustness_score": 0.88},
            }
        )

        # Add a trial without robustness
        self.dashboard.recent_trials.append(
            {
                "id": "124",
                "model": "mlp",
                "task": "digits",
                "accuracy": 0.96,
                "status": "completed",
                "metrics": {"accuracy": 0.96},  # No robustness
            }
        )

        try:
            self.dashboard.update()
        except Exception as e:
            self.fail(f"Dashboard.update() raised exception: {e}")


if __name__ == "__main__":
    unittest.main()
