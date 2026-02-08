import unittest
import torch
from bioplausible.hyperopt.tasks import create_task, VisionTask
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec

class TestVisionTask(unittest.TestCase):
    def test_digits_task_setup_and_train(self):
        """Regression test for digits task permutation bug."""
        task = create_task("digits", device="cpu", quick_mode=True)
        self.assertIsInstance(task, VisionTask)

        task.setup()

        # Verify shape (N, 1, 8, 8) or similar valid NCHW
        # Batch size 4
        x, y = task.get_batch(split="train", batch_size=4)
        self.assertEqual(x.dim(), 4, "Expected 4D input (NCHW)")
        self.assertEqual(x.shape[1], 1, "Expected 1 channel")
        self.assertEqual(x.shape[2], 8, "Expected height 8")
        self.assertEqual(x.shape[3], 8, "Expected width 8")

        # Create a compatible model (MLP for flattened input or Conv)
        # Using Backprop Baseline (MLP) which flattens internally or via trainer adapter
        # But wait, BackpropMLP expects flattened input dim.
        # VisionTask provides input_dim=64.
        # SupervisedTrainer handles flattening if model expects it?
        # Let's check SupervisedTrainer._prepare_input: returns x.view(x.size(0), -1)

        spec = get_model_spec("Backprop Baseline")
        model = create_model(
            spec,
            input_dim=task.input_dim,
            output_dim=task.output_dim,
            hidden_dim=32,
            num_layers=1,
            task_type="vision" # Critical!
        )

        trainer = task.create_trainer(model)

        # Run one epoch
        metrics = trainer.train_epoch()
        self.assertIn("loss", metrics)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["loss"])))

if __name__ == "__main__":
    unittest.main()
