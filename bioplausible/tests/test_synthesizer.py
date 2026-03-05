import unittest
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from bioplausible.scientist.synthesizer import ResearchSynthesizer
from bioplausible.scientist.training_dynamics import (
    TrainingCheckpoint,
    TrainingTrajectory,
)


# Helper to create mock trajectories
def create_mock_traj(
    model_name: str,
    task_name: str,
    final_acc: float,
    convergence_epoch: int,
    config: Dict = None,
) -> TrainingTrajectory:

    if config is None:
        config = {"activation": "relu"}

    checkpoints = []
    # Add a final checkpoint
    ckpt = TrainingCheckpoint(
        epoch=convergence_epoch + 5,
        train_acc=final_acc + 0.05,
        val_acc=final_acc,
        train_loss=0.5,
        val_loss=0.6,
        grad_norm_mean=0.1,
        grad_norm_std=0.01,
        weight_norm=1.0,
        learning_rate=0.01,
        train_val_gap=0.05,
        wall_time_seconds=10.0,
    )
    checkpoints.append(ckpt)

    traj = TrainingTrajectory(
        trial_id=1,
        model_name=model_name,
        task_name=task_name,
        config=config,
        checkpoints=checkpoints,
    )
    # Mock computed metrics
    traj.compute_convergence_speed = lambda: convergence_epoch
    traj.compute_sample_efficiency = lambda: final_acc * 100  # Dummy metric

    return traj


class TestResearchSynthesizer(unittest.TestCase):

    def setUp(self):
        import sqlite3

        self.db_path = ":memory:"
        self.synth = ResearchSynthesizer(self.db_path)

        # Setup mock db connection and populate
        self.conn = sqlite3.connect(self.db_path)

        # Create minimal schemas
        self.conn.execute("""
            CREATE TABLE studies (
                study_id INTEGER PRIMARY KEY,
                study_name TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE trials (
                trial_id INTEGER PRIMARY KEY,
                study_id INTEGER,
                state TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE hyperopt_logs (
                trial_id INTEGER PRIMARY KEY,
                param_count INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE trial_params (
                trial_id INTEGER,
                param_name TEXT,
                param_value TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE trial_values (
                trial_id INTEGER,
                value REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE trial_user_attributes (
                trial_id INTEGER,
                key TEXT,
                value_json TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE failures (
                id INTEGER PRIMARY KEY,
                trial_id INTEGER,
                model_name TEXT,
                task_name TEXT,
                failure_type TEXT
            )
        """)

        # Populate trials
        trials = [
            (1, 0.95, "Baseline Backprop", "mnist", "standard"),
            (2, 0.85, "Baseline Backprop", "cifar10", "standard"),
            (3, 0.92, "EqProp MLP", "mnist", "standard"),
            (4, 0.80, "EqProp Conv", "cifar10", "standard"),
            (5, 0.98, "GELU Model", "mnist", "standard"),
            (6, 0.94, "ReLU Model", "mnist", "standard"),
        ]

        for tid, acc, model, task, tier in trials:
            self.conn.execute(
                "INSERT INTO trials (trial_id, study_id, state) VALUES (?, 1, 'COMPLETE')",
                (tid,),
            )
            self.conn.execute(
                "INSERT INTO trial_values (trial_id, value) VALUES (?, ?)", (tid, acc)
            )
            self.conn.execute(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, 'model_name', ?)",
                (tid, f'"{model}"'),
            )
            self.conn.execute(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, 'task_name', ?)",
                (tid, f'"{task}"'),
            )
            self.conn.execute(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, 'tier', ?)",
                (tid, f'"{tier}"'),
            )
            self.conn.execute(
                "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES (?, 'param_count', '100000')",
                (tid,),
            )

        # Populate failures
        failures = [
            (1, "EqProp Conv", "cifar10", "nan"),
            (2, "EqProp Conv", "cifar10", "nan"),
            (3, "EqProp Conv", "cifar10", "nan"),
            (4, "EqProp Conv", "cifar10", "nan"),
            (5, "EqProp Conv", "cifar10", "nan"),
            (6, "EqProp Conv", "cifar10", "nan"),
        ]

        for fid, model, task, ftype in failures:
            self.conn.execute(
                "INSERT INTO failures (id, trial_id, model_name, task_name, failure_type) VALUES (?, NULL, ?, ?, ?)",
                (fid, model, task, ftype),
            )

        self.conn.commit()

        # Override synth's get_trials_df to use our memory connection
        import pandas as pd

        def get_trials_df_mock(conn):
            return self.synth.__class__._get_trials_df(self.synth, self.conn)

        self.synth._get_trials_df = get_trials_df_mock

        # Replace find_quick_wins to use the in-memory db connection
        def find_quick_wins_mock():
            import pandas as pd

            trials = self.synth._get_trials_df(self.conn)
            failures = pd.read_sql("SELECT * FROM failures", self.conn)
            return self.synth._find_quick_wins(trials, failures)

        self.synth.find_quick_wins = find_quick_wins_mock

    def tearDown(self):
        self.conn.close()

    def test_cross_algorithm_insights(self):
        """Test that insights are generated correctly."""
        insights = self.synth.generate_cross_algorithm_insights()
        self.assertIn("rankings", insights)
        rankings = insights["rankings"]

        best = rankings[0]
        self.assertEqual(best["model"], "GELU Model")
        self.assertAlmostEqual(best["best_accuracy"], 0.98)

    def test_architecture_recommendations(self):
        """Test hybrid recommendation generation."""
        recs = self.synth.generate_architecture_recommendations()
        self.assertGreater(len(recs), 0)

    def test_quick_wins(self):
        """Test detection of activation function wins."""
        wins = self.synth.find_quick_wins()

        # Expecting NaN failure advice
        nan_win = next(
            (w for w in wins if "nan" in w.lower() or "failure rate" in w.lower()), None
        )
        self.assertIsNotNone(
            nan_win, f"Expected NaN or failure rate advice, but got: {wins}"
        )

        # Expecting underexplored advice
        underexplored = next((w for w in wins if "Underexplored" in w), None)
        self.assertIsNotNone(underexplored)

    def test_research_gaps(self):
        """Test gap detection."""
        gaps = self.synth.identify_research_gaps()
        self.assertTrue(any("cartpole" in g for g in gaps))
        self.assertTrue(any("Graph Neural Network" in g for g in gaps))


if __name__ == "__main__":
    unittest.main()
