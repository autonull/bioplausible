import logging
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from bioplausible.execution.task import ExperimentTask
from bioplausible.hyperopt import PatientLevel


class TestRobustnessIntegration(unittest.TestCase):
    def setUp(self):
        # Create temp directories
        self.test_dir = tempfile.mkdtemp()
        self.artifacts_dir = Path("artifacts")
        self.artifacts_dir.mkdir(exist_ok=True)
        self.db_path = str(Path(self.test_dir) / "test.db")

        # Silence logger
        logging.getLogger("AutoScientist").setLevel(logging.CRITICAL)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("bioplausible.scientist.core.run_robustness_check")
    def test_robustness_uses_pretrained_weights_zip(self, mock_run_robustness):
        """Test that robustness check correctly extracts and uses weights from zip artifact."""

        # Define side effect to check file existence during call
        def side_effect(*args, **kwargs):
            weights_path = kwargs.get("weights_path")
            if weights_path and Path(weights_path).exists():
                # Verify content
                with Path(weights_path).open("rb") as f:
                    content = f.read()
                if content == b"dummy pytorch weights zip":
                    return {"robustness_score": 0.85, "noise_score": 0.9}
            return {"robustness_score": 0.0}

        mock_run_robustness.side_effect = side_effect

        trial_id = 99999
        model_name = "test_model_zip"

        # Create a dummy zip artifact
        artifact_path = self.artifacts_dir / f"trial_{trial_id}_{model_name}.zip"
        dummy_weights_content = b"dummy pytorch weights zip"

        try:
            with zipfile.ZipFile(artifact_path, "w") as zf:
                zf.writestr("model.pt", dummy_weights_content)

            # Instantiate AutoScientist
            scientist = AutoScientist(db_path=self.db_path)

            # Create Task
            task = ExperimentTask(
                model_name=model_name,
                task_name="digits",
                tier=PatientLevel.DEEP,
                study_name="test_study",
                priority=1.0,
                verification_of_trial_id=trial_id,
                is_robustness_check=True,
            )

            # Execute
            metrics = scientist._execute_robustness_check(task, {})

            # Verify
            self.assertTrue(mock_run_robustness.called)
            args, kwargs = mock_run_robustness.call_args
            weights_path = kwargs.get("weights_path")
            output_dir = kwargs.get("output_dir")

            self.assertIsNotNone(weights_path)
            self.assertEqual(output_dir, f"artifacts/trial_{trial_id}/interpretability")

            self.assertEqual(
                metrics["robustness_score"],
                0.85,
                "Robustness check failed to find/verify weights file",
            )
            self.assertEqual(metrics["noise_score"], 0.9)

        finally:
            # Clean up artifact
            if artifact_path.exists():
                Path(artifact_path).unlink()

    @patch("bioplausible.scientist.core.run_robustness_check")
    def test_robustness_uses_pretrained_weights_dir(self, mock_run_robustness):
        """Test that robustness check correctly uses weights from directory artifact."""
        mock_run_robustness.return_value = {"robustness_score": 0.85, "ood_score": 0.7}

        trial_id = 88888
        model_name = "test_model_dir"

        # Create a dummy dir artifact
        artifact_path = self.artifacts_dir / f"trial_{trial_id}_{model_name}"
        artifact_path.mkdir(exist_ok=True)
        weights_file = artifact_path / "model.pt"
        dummy_weights_content = b"dummy dir weights"

        try:
            with Path(weights_file).open("wb") as f:
                f.write(dummy_weights_content)

            # Instantiate AutoScientist
            scientist = AutoScientist(db_path=self.db_path)

            # Create Task
            task = ExperimentTask(
                model_name=model_name,
                task_name="digits",
                tier=PatientLevel.DEEP,
                study_name="test_study",
                priority=1.0,
                verification_of_trial_id=trial_id,
                is_robustness_check=True,
            )

            # Execute
            metrics = scientist._execute_robustness_check(task, {})

            # Verify
            self.assertTrue(mock_run_robustness.called)
            args, kwargs = mock_run_robustness.call_args
            weights_path = kwargs.get("weights_path")
            output_dir = kwargs.get("output_dir")

            self.assertIsNotNone(weights_path)
            self.assertEqual(weights_path, str(weights_file))
            self.assertEqual(output_dir, f"artifacts/trial_{trial_id}/interpretability")
            self.assertTrue(Path(weights_path).exists())
            self.assertEqual(metrics["robustness_score"], 0.85)
            self.assertEqual(metrics["ood_score"], 0.7)

        finally:
            # Clean up artifact
            if artifact_path.exists():
                shutil.rmtree(artifact_path)


if __name__ == "__main__":
    unittest.main()
