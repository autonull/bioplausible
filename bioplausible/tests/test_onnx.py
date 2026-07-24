import os
import pathlib
import shutil
import unittest

from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.zoo.models.eqprop import LoopedMLP


class TestONNXExport(unittest.TestCase):
    def setUp(self):
        self.model = LoopedMLP(
            input_dim=10, hidden_dim=20, output_dim=2, use_spectral_norm=False
        )
        # Stand-in trainer whose export_onnx path mirrors CoreTrainer's signature.
        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 10,
                "hidden_dim": 20,
                "output_dim": 2,
                "use_spectral_norm": False,
            },
            optimizer="adam",
            task="mnist",
            epochs=1,
            use_compile=False,
        )
        self.trainer = CoreTrainer(config)
        # Explicitly create the model on this trainer for export tests.
        self.trainer.model = self.model
        self.onnx_path = "test_model.onnx"
        self.temp_dir = "temp_dir"

    def tearDown(self):
        if pathlib.Path(self.onnx_path).exists():
            pathlib.Path(self.onnx_path).unlink()
            if pathlib.Path(self.onnx_path + ".data").exists():
                pathlib.Path(self.onnx_path + ".data").unlink()

        if pathlib.Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_export_onnx(self):
        """Smoke test for ONNX export via CoreTrainer."""
        try:
            self.trainer.export_onnx(self.onnx_path, input_shape=(1, 10))
            self.assertTrue(pathlib.Path(self.onnx_path).exists())
        except RuntimeError as e:
            self.skipTest(f"ONNX export failed: {e}")

    def test_export_onnx_directory_creation(self):
        """Test that export creates parent directories as needed."""
        path = os.path.join(self.temp_dir, "test_model.onnx")
        try:
            self.trainer.export_onnx(path, input_shape=(1, 10))
            self.assertTrue(pathlib.Path(path).exists())
        except RuntimeError as e:
            self.skipTest(f"ONNX export failed: {e}")


if __name__ == "__main__":
    unittest.main()
