import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile
import shutil

from bioplausible.scientist.reporting import ScientistReporter


class MockTrial:
    def __init__(self, trial_id, model_name, accuracy, loss, config):
        self.trial_id = trial_id
        self.model_name = model_name
        self.accuracy = accuracy
        self.final_loss = loss
        self.config = config
        self.status = "completed"
        self.param_count = 1000


class TestReportingEnhancements(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("bioplausible.scientist.reporting.HyperoptStorage")
    @patch("bioplausible.scientist.reporting.ResultVisualizer")
    @patch("bioplausible.scientist.reporting.StatisticalAnalyzer")
    @patch("bioplausible.scientist.reporting.shutil.which")
    def test_generate_report(
        self, mock_which, mock_analyzer, mock_visualizer, mock_storage
    ):
        # Mock trials
        trials = [
            MockTrial(
                1,
                "Model A",
                0.85,
                0.5,
                {"learning_rate": 0.01, "tier": "standard", "task": "mnist"},
            ),
            MockTrial(
                2,
                "Model B",
                0.90,
                0.4,
                {"learning_rate": 0.005, "tier": "standard", "task": "mnist"},
            ),
            MockTrial(
                3,
                "Model A",
                0.86,
                0.48,
                {"learning_rate": 0.01, "tier": "standard", "task": "mnist"},
            ),  # Verify repeating
        ]

        # Setup mocks
        mock_storage_instance = MagicMock()
        mock_storage_instance.get_all_trials.return_value = trials
        mock_storage.return_value = mock_storage_instance

        # Mock pdflatex and bibtex existence
        mock_which.return_value = "/usr/bin/pdflatex"

        # Initialize reporter
        reporter = ScientistReporter(self.db_path)

        # Run report generation
        output_dir = os.path.join(self.temp_dir, "report")
        reporter.generate_report(output_dir)

        # Verify best_config.json
        config_path = os.path.join(output_dir, "best_config.json")
        self.assertTrue(
            os.path.exists(config_path), "best_config.json should be created"
        )

        with open(config_path, "r") as f:
            config = json.load(f)
            self.assertEqual(config["model"], "Model B", "Best model should be Model B")
            self.assertEqual(config["accuracy"], 0.90, "Best accuracy should be 0.90")

        # Verify report.tex
        tex_path = os.path.join(output_dir, "report.tex")
        self.assertTrue(os.path.exists(tex_path), "report.tex should be created")

        with open(tex_path, "r") as f:
            content = f.read()
            self.assertIn(
                r"\section{Machine Learning Analysis}",
                content,
                "Should contain ML Analysis section",
            )
            self.assertIn(r"\appendix", content, "Should contain Appendix")
            self.assertIn(
                r"\section{Best Configuration}",
                content,
                "Should contain Best Config section",
            )
            self.assertIn("90.00", content, "Should contain best accuracy")

        # Verify compile script
        compile_script = os.path.join(output_dir, "compile_report.sh")
        self.assertTrue(
            os.path.exists(compile_script), "compile_report.sh should be created"
        )
        with open(compile_script, "r") as f:
            script_content = f.read()
            self.assertIn(
                "pdflatex report.tex",
                script_content,
                "Should contain compilation commands",
            )

    @patch("bioplausible.scientist.reporting.HyperoptStorage")
    @patch("bioplausible.scientist.reporting.ResultVisualizer")
    @patch("bioplausible.scientist.reporting.StatisticalAnalyzer")
    @patch("bioplausible.scientist.reporting.shutil.which")
    def test_generate_report_no_latex(
        self, mock_which, mock_analyzer, mock_visualizer, mock_storage
    ):
        # Mock trials
        trials = [
            MockTrial(
                1,
                "Model A",
                0.85,
                0.5,
                {"learning_rate": 0.01, "tier": "standard", "task": "mnist"},
            ),
        ]

        mock_storage_instance = MagicMock()
        mock_storage_instance.get_all_trials.return_value = trials
        mock_storage.return_value = mock_storage_instance

        # Mock NO pdflatex
        mock_which.return_value = None

        reporter = ScientistReporter(self.db_path)
        output_dir = os.path.join(self.temp_dir, "report_no_latex")
        reporter.generate_report(output_dir)

        # Verify compile script warns
        compile_script = os.path.join(output_dir, "compile_report.sh")
        with open(compile_script, "r") as f:
            script_content = f.read()
            self.assertIn(
                "echo 'pdflatex or bibtex not found",
                script_content,
                "Should contain warning echo",
            )


if __name__ == "__main__":
    unittest.main()
