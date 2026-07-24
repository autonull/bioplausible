import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from bioplausible.execution.training_dynamics import ContinuousTrainingSchedule
from bioplausible.execution.training_dynamics import TrainingCheckpoint
from bioplausible.execution.training_dynamics import TrainingTrajectory
from bioplausible.hyperopt.storage import HyperoptStorage


# Mock classes
@dataclass
class MockMetrics:
    loss: float
    accuracy: float
    time: float = 0.1


class MockTrainer:
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list
        self.idx = 0

    def train_epoch(self):
        m = self.metrics_list[self.idx % len(self.metrics_list)]
        self.idx += 1
        return {
            "loss": m.loss,
            "accuracy": m.accuracy,
            "time": m.time,
            "train_loss": m.loss,
            "train_acc": m.accuracy + 0.1,  # Fake train gap
        }


class TestContinuousTraining(unittest.TestCase):

    def test_convergence_detection(self):
        """Test detection of convergence when accuracy plateaus."""
        # Create a trajectory with plateauing accuracy
        checkpoints = [
            TrainingCheckpoint(
                epoch=1, val_acc=0.5, train_acc=0.6, train_loss=1.0, val_loss=1.0
            ),
            TrainingCheckpoint(
                epoch=2, val_acc=0.6, train_acc=0.7, train_loss=0.8, val_loss=0.8
            ),
            TrainingCheckpoint(
                epoch=5, val_acc=0.7, train_acc=0.8, train_loss=0.6, val_loss=0.6
            ),
            TrainingCheckpoint(
                epoch=10, val_acc=0.705, train_acc=0.8, train_loss=0.5, val_loss=0.6
            ),  # < 1% improvement
            TrainingCheckpoint(
                epoch=20, val_acc=0.708, train_acc=0.8, train_loss=0.4, val_loss=0.6
            ),
        ]

        traj = TrainingTrajectory(
            trial_id=1,
            model_name="test",
            task_name="test",
            config={},
            checkpoints=checkpoints,
        )

        schedule = ContinuousTrainingSchedule()
        conv_epoch = schedule._find_convergence(traj)

        # Should detect convergence around epoch 5 (starting point of window with low improvement)
        # Window size 3: [5, 10, 20]. Improv 20 vs 5: 0.708 - 0.7 = 0.008 < 0.01
        self.assertEqual(conv_epoch, 5)

    def test_storage_integration(self):
        """Test saving and retrieving trajectories."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test.db"

        try:
            storage = HyperoptStorage(str(db_path))

            # Create dummy trial log first (FK constraint)
            config = {"lr": 0.01}
            trial_id = storage.create_trial("test_model", config)

            # Create trajectory
            traj = TrainingTrajectory(
                trial_id=trial_id,
                model_name="test_model",
                task_name="test_task",
                config=config,
                checkpoints=[
                    TrainingCheckpoint(
                        epoch=1,
                        val_acc=0.5,
                        train_acc=0.6,
                        train_loss=1.0,
                        val_loss=1.1,
                    ),
                    TrainingCheckpoint(
                        epoch=5,
                        val_acc=0.7,
                        train_acc=0.8,
                        train_loss=0.5,
                        val_loss=0.6,
                    ),
                ],
            )
            traj.converged = True

            # Save
            storage.save_trajectory(traj)

            # Verify via SQL
            cursor = storage.conn.cursor()
            cursor.execute(
                "SELECT * FROM training_trajectories WHERE trial_id=?", (trial_id,)
            )
            row_traj = cursor.fetchone()
            self.assertIsNotNone(row_traj)
            self.assertEqual(row_traj["task_name"], "test_task")

            cursor.execute(
                "SELECT * FROM training_checkpoints WHERE trajectory_id=?",
                (row_traj["id"],),
            )
            rows_ckpts = cursor.fetchall()
            self.assertEqual(len(rows_ckpts), 2)
            self.assertEqual(rows_ckpts[1]["epoch"], 5)
            self.assertAlmostEqual(rows_ckpts[0]["val_acc"], 0.5)

            storage.close()

        finally:
            shutil.rmtree(temp_dir)

    def test_schedule_execution_logic(self):
        """Test that train_with_checkpoints calls trainer correctly."""
        # Mock trainer returning increasing accuracy
        metrics_sequence = [
            MockMetrics(loss=0.9, accuracy=0.1),  # epoch 1
            MockMetrics(loss=0.8, accuracy=0.2),  # epoch 2
            MockMetrics(loss=0.7, accuracy=0.3),  # epoch 3
            MockMetrics(loss=0.6, accuracy=0.4),  # epoch 4
            MockMetrics(loss=0.5, accuracy=0.5),  # epoch 5
        ]
        trainer = MockTrainer(metrics_sequence)

        schedule = ContinuousTrainingSchedule(max_epochs=5)  # Checkpoints: [1, 2, 5]

        # Test callbacks
        callback_epochs = []

        def on_epoch_end(epoch, m):
            callback_epochs.append(epoch)

        traj = schedule.train_with_checkpoints(
            trainer,
            trial_id=999,
            model_name="m",
            task_name="t",
            config={},
            on_epoch_end=on_epoch_end,
        )

        self.assertEqual(len(traj.checkpoints), 3)  # 1, 2, 5
        self.assertEqual(traj.checkpoints[0].epoch, 1)
        self.assertEqual(traj.checkpoints[2].epoch, 5)

        # Check standard checkpoints logic (last epoch metrics used)
        self.assertAlmostEqual(traj.checkpoints[2].val_acc, 0.5)  # epoch 5 metrics

        # Check callbacks were called for every epoch
        self.assertEqual(callback_epochs, [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
