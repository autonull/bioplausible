import shutil
import tempfile
import unittest
from pathlib import Path

from bioplausible.config import GLOBAL_CONFIG
from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage

# from bioplausible.core.registry import Registry # Imported implicitly


class TestPhase2Integration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "verify.db"

        # Configure global settings
        GLOBAL_CONFIG.quick_mode = True
        # Save original epochs
        self.original_epochs = GLOBAL_CONFIG.epochs
        GLOBAL_CONFIG.epochs = 5

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        GLOBAL_CONFIG.epochs = self.original_epochs

    def test_end_to_end_trajectory_storage(self):
        """Verify that running a trial stores trajectory and checkpoints in DB."""

        storage = HyperoptStorage(str(self.db_path))

        # Create Trial
        model_name = "Backprop Baseline"
        config = {
            "lr": 0.01,
            "hidden_dim": 32,
            "num_layers": 2,
            "save_artifacts": False,
        }
        trial_id = storage.create_trial(model_name, config)

        # Run Trial via Runner
        runner = TrialRunner(
            storage=storage, device="cpu", task="mnist", quick_mode=True
        )

        # Override runner epochs to match config
        runner.epochs = 5

        success = runner.run_trial(trial_id)
        self.assertTrue(success, "Trial failed to run successfully")

        # Verify DB Content
        cursor = storage.conn.cursor()

        # Check Trajectory
        cursor.execute(
            "SELECT * FROM training_trajectories WHERE trial_id=?", (trial_id,)
        )
        traj = cursor.fetchone()

        self.assertIsNotNone(traj, "No training_trajectory found")
        self.assertTrue(traj["converged"] in [0, 1])

        # Check Checkpoints
        cursor.execute(
            "SELECT * FROM training_checkpoints WHERE trajectory_id=? ORDER BY epoch",
            (traj["id"],),
        )
        checkpoints = cursor.fetchall()

        self.assertTrue(len(checkpoints) > 0, "No checkpoints found")

        # Expect checkpoints at least at 1, 2, 5 (or valid subset)
        found_epochs = [c["epoch"] for c in checkpoints]
        self.assertIn(5, found_epochs)

        # Check for non-null metrics
        last_ckpt = checkpoints[-1]
        self.assertIsNotNone(last_ckpt["val_acc"])
        self.assertIsNotNone(last_ckpt["train_loss"])

        storage.close()


if __name__ == "__main__":
    unittest.main()
