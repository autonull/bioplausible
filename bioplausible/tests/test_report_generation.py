import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bioplausible.scientist.report.composer import ReportComposer


class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_scientist.db")
        self.output_dir = os.path.join(self.temp_dir, "report_output")

        # Initialize a real SQLite DB for testing
        self.conn = sqlite3.connect(self.db_path)
        self.create_dummy_data()

    def tearDown(self):
        if self.conn:
            self.conn.close()
        shutil.rmtree(self.temp_dir)

    def create_dummy_data(self):
        """Populate the database with dummy trial data."""
        cursor = self.conn.cursor()

        # Tables needed by ReportComposer._get_trials_df
        cursor.execute("""
            CREATE TABLE trials (
                trial_id INTEGER PRIMARY KEY,
                study_id INTEGER,
                state VARCHAR(255)
            )
        """)
        cursor.execute("""
            CREATE TABLE studies (
                study_id INTEGER PRIMARY KEY,
                study_name VARCHAR(255)
            )
        """)
        cursor.execute("""
            CREATE TABLE trial_values (
                trial_id INTEGER,
                value REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE trial_user_attributes (
                trial_id INTEGER,
                key VARCHAR(255),
                value_json TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE hyperopt_logs (
                trial_id INTEGER,
                param_count INTEGER,
                iteration_time REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE trial_params (
                trial_id INTEGER,
                param_name VARCHAR(255),
                param_value REAL
            )
        """)

        # Insert Data
        # Trial 1: Good model
        cursor.execute("INSERT INTO studies VALUES (1, 'vision_mnist')")
        cursor.execute("INSERT INTO trials VALUES (1, 1, 'COMPLETE')")
        cursor.execute("INSERT INTO trial_values VALUES (1, 0.95)")
        cursor.execute(
            "INSERT INTO trial_user_attributes VALUES (1, 'model_name', '\"TestModel\"')"
        )
        cursor.execute(
            "INSERT INTO trial_user_attributes VALUES (1, 'task_name', '\"mnist\"')"
        )
        cursor.execute(
            "INSERT INTO trial_user_attributes VALUES (1, 'tier',"
            ' "\'standard\'")'
        )
        cursor.execute("INSERT INTO hyperopt_logs VALUES (1, 10000, 0.5)")

        # Trial 2: Another model
        cursor.execute("INSERT INTO trials VALUES (2, 1, 'COMPLETE')")
        cursor.execute("INSERT INTO trial_values VALUES (2, 0.85)")
        cursor.execute(
            "INSERT INTO trial_user_attributes VALUES (2, 'model_name', '\"Baseline\"')"
        )
        cursor.execute(
            "INSERT INTO trial_user_attributes VALUES (2, 'task_name', '\"mnist\"')"
        )
        cursor.execute("INSERT INTO hyperopt_logs VALUES (2, 5000, 0.2)")

        # Tables for convergence data
        cursor.execute("""
            CREATE TABLE training_trajectories (
                id INTEGER PRIMARY KEY,
                trial_id INTEGER,
                model_name TEXT,
                task_name TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE training_checkpoints (
                id INTEGER PRIMARY KEY,
                trajectory_id INTEGER,
                epoch INTEGER,
                val_acc REAL,
                samples_seen INTEGER
            )
        """)

        # Trajectory data
        cursor.execute(
            "INSERT INTO training_trajectories VALUES (1, 1, 'TestModel', 'mnist')"
        )
        cursor.execute("INSERT INTO training_checkpoints VALUES (1, 1, 1, 0.50, 1000)")
        cursor.execute("INSERT INTO training_checkpoints VALUES (2, 1, 2, 0.95, 2000)")

        # Decision Logs
        cursor.execute("""
            CREATE TABLE decision_logs (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                event_type TEXT,
                description TEXT
            )
        """)
        cursor.execute(
            "INSERT INTO decision_logs VALUES (1, '2023-01-01', 'START', 'Started')"
        )

        self.conn.commit()

    @patch("bioplausible.scientist.report.composer.ResultVisualizer")
    @patch("bioplausible.scientist.report.composer.MLAnalyzer")
    @patch("bioplausible.scientist.report.composer.BayesianRanker")
    @patch("bioplausible.scientist.report.composer.LatexGenerator")
    def test_report_generation_flow(self, mock_latex, mock_ranker, mock_ml, mock_viz):
        """Test the end-to-end report generation flow."""

        # Setup mocks to return dummy paths/data
        mock_viz_instance = mock_viz.return_value
        mock_viz_instance.plot_pareto_frontier.return_value = "pareto.png"
        mock_viz_instance.plot_tier_progress.return_value = "progress.png"
        mock_viz_instance.plot_leaderboard.return_value = "leaderboard.png"

        mock_ml_instance = mock_ml.return_value
        mock_ml_instance.run_analysis.return_value = ("Insights", "Robustness")

        mock_ranker_instance = mock_ranker.return_value
        mock_ranker_instance.rank_models.return_value = "| Model | Rank |"

        # Run Composer
        with ReportComposer(self.db_path, self.output_dir) as composer:
            composer.generate_report()

        # Check Output Files
        summary_path = Path(self.output_dir) / "01_summary.md"
        leaderboard_path = Path(self.output_dir) / "03_leaderboards.md"
        full_report_path = Path(self.output_dir) / "FULL_REPORT.md"
        manifest_path = Path(self.output_dir) / "manifest.json"

        self.assertTrue(summary_path.exists(), "Summary file not created")
        self.assertTrue(leaderboard_path.exists(), "Leaderboard file not created")
        self.assertTrue(full_report_path.exists(), "Full report not created")
        self.assertTrue(manifest_path.exists(), "Manifest not created")

        # Verify Content
        with open(summary_path, "r") as f:
            content = f.read()
            self.assertIn("TestModel", content)  # Should show best model
            self.assertIn("95.00%", content)

        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            self.assertIn("title", manifest)
            # Check visuals were registered
            images = [img["path"] for img in manifest["images"]]
            self.assertIn("pareto.png", images)


if __name__ == "__main__":
    unittest.main()
