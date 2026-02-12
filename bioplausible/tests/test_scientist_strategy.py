import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List

from bioplausible.scientist.strategy import ScientistStrategy
from bioplausible.scientist.state import ExperimentState
from bioplausible.hyperopt import PatientLevel
from bioplausible.scientist.task import ExperimentTask
from bioplausible.hyperopt.storage import TrialMetrics
from bioplausible.models.registry import MODEL_REGISTRY

class TestScientistStrategy(unittest.TestCase):
    def setUp(self):
        self.mock_state = MagicMock(spec=ExperimentState)
        self.mock_state.get_progress.return_value = {}
        self.mock_state.get_recent_tasks.return_value = []
        self.mock_state.get_failure_analysis.return_value = {}

        self.strategy = ScientistStrategy(self.mock_state)

        self.valid_model = None
        for spec in MODEL_REGISTRY:
             tasks = self.strategy._resolve_tasks(spec.task_compat, spec.name)
             if "digits" in tasks:
                 self.valid_model = spec.name
                 break

        if not self.valid_model:
            self.valid_model = "mlp"

    def _create_mock_trial(self, config, accuracy=0.85, trial_id=1):
        trial = MagicMock(spec=TrialMetrics)
        trial.config = config
        trial.accuracy = accuracy
        trial.trial_id = trial_id
        trial.final_loss = 0.1
        return trial

    def _setup_progress(self, model, task, tier, count=0, best_acc=0.0, trials=None):
        """Helper to setup the progress dictionary."""
        progress = self.mock_state.get_progress.return_value
        if model not in progress:
            progress[model] = {}
        if task not in progress[model]:
            progress[model][task] = {}

        progress[model][task][tier.value] = {
            "count": count,
            "best_acc": best_acc,
            "trials": trials if trials else []
        }
        self.mock_state.get_progress.return_value = progress

    def test_smoke_generation(self):
        task_name = "digits"
        candidates = self.strategy.generate_candidates()
        my_candidates = [c for c in candidates if c.model_name == self.valid_model and c.task_name == task_name]
        self.assertTrue(len(my_candidates) > 0, f"No candidates for {self.valid_model} on {task_name}")
        self.assertEqual(my_candidates[0].tier, PatientLevel.SMOKE)

    def test_shallow_promotion(self):
        task_name = "digits"
        # Set SMOKE as passed but NOT saturated (< 0.98)
        self._setup_progress(self.valid_model, task_name, PatientLevel.SMOKE, count=5, best_acc=0.85)

        candidates = self.strategy.generate_candidates()
        my_candidates = [c for c in candidates if c.model_name == self.valid_model and c.task_name == task_name]

        shallow = [c for c in my_candidates if c.tier == PatientLevel.SHALLOW]
        self.assertTrue(len(shallow) > 0, f"Shallow candidate not generated. Candidates: {my_candidates}")

    def test_ablation_generation(self):
        task_name = "digits"
        config = {"lr": 0.01, "beta": 0.1, "use_top_down": True}
        trial = self._create_mock_trial(config, accuracy=0.85, trial_id=100)

        self._setup_progress(self.valid_model, task_name, PatientLevel.SMOKE, count=5, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.SHALLOW, count=10, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.STANDARD, count=25, best_acc=0.85, trials=[trial])

        candidates = self.strategy.generate_candidates()

        ablations = [c for c in candidates if c.is_ablation and c.model_name == self.valid_model]
        self.assertTrue(len(ablations) > 0, "Ablation candidate not generated")
        # We expect beta=0.0 and use_top_down=False (2 ablations)
        self.assertTrue(len(ablations) >= 2, f"Expected multiple ablations, got {len(ablations)}")
        self.assertEqual(ablations[0].tier, PatientLevel.STANDARD)
        self.assertTrue(ablations[0].is_ablation)

    def test_verification_needed(self):
        task_name = "digits"
        config = {"lr": 0.01}
        trial = self._create_mock_trial(config, accuracy=0.85, trial_id=100)

        self._setup_progress(self.valid_model, task_name, PatientLevel.SMOKE, count=5, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.SHALLOW, count=10, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.STANDARD, count=25, best_acc=0.85, trials=[trial])

        candidates = self.strategy.generate_candidates()

        verifications = [c for c in candidates if c.model_name == self.valid_model and c.verification_of_trial_id == 100]
        self.assertTrue(len(verifications) > 0, "Verification candidate not generated")
        # Need 3 repeats total. Have 1. Need 2 more.
        self.assertEqual(len(verifications), 2, f"Expected 2 verification tasks, got {len(verifications)}")

    def test_cv_needed(self):
        task_name = "digits"
        config = {"lr": 0.01}
        t1 = self._create_mock_trial(config, accuracy=0.85, trial_id=100)
        t2 = self._create_mock_trial(config, accuracy=0.84, trial_id=101)
        t3 = self._create_mock_trial(config, accuracy=0.86, trial_id=102)

        self._setup_progress(self.valid_model, task_name, PatientLevel.SMOKE, count=5, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.SHALLOW, count=10, best_acc=0.85)
        self._setup_progress(self.valid_model, task_name, PatientLevel.STANDARD, count=25, best_acc=0.85, trials=[t1, t2, t3])

        candidates = self.strategy.generate_candidates()

        cv_tasks = [c for c in candidates if c.model_name == self.valid_model and c.tier == PatientLevel.CROSS_VAL]
        self.assertTrue(len(cv_tasks) > 0, "CV candidate not generated")
        # 5 folds needed. None done. Expect 5 tasks.
        self.assertEqual(len(cv_tasks), 5, f"Expected 5 CV tasks, got {len(cv_tasks)}")

if __name__ == "__main__":
    unittest.main()
