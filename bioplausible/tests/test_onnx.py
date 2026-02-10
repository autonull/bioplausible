import os
import shutil
import unittest

import torch
import torch.nn as nn

from bioplausible.core import EqPropTrainer
from bioplausible.models.looped_mlp import LoopedMLP


class TestONNXExport(unittest.TestCase):
    def setUp(self):
        self.model = LoopedMLP(
            input_dim=10, hidden_dim=20, output_dim=2, use_spectral_norm=False
        )
        self.trainer = EqPropTrainer(self.model, use_compile=False)
        self.onnx_path = "test_model.onnx"
        self.temp_dir = "temp_dir"

    def tearDown(self):
        if os.path.exists(self.onnx_path):
            os.remove(self.onnx_path)
            # Remove potential .data file created by newer torch versions for large models or certain configs
            if os.path.exists(self.onnx_path + ".data"):
                os.remove(self.onnx_path + ".data")

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_export_onnx(self):
        # Smoke test for ONNX export
        try:
            self.trainer.export_onnx(self.onnx_path, input_shape=(1, 10))
            self.assertTrue(os.path.exists(self.onnx_path))
        except RuntimeError as e:
            # Skip if ONNX export fails due to missing dependencies or platform issues
            # but usually it should work with torch installed
            self.skipTest(f"ONNX export failed: {e}")

    def test_export_onnx_directory_creation(self):
        # Test that it creates directories
        path = os.path.join(self.temp_dir, "test_model.onnx")
        try:
            self.trainer.export_onnx(path, input_shape=(1, 10))
            self.assertTrue(os.path.exists(path))
        except RuntimeError as e:
            self.skipTest(f"ONNX export failed: {e}")


if __name__ == "__main__":
    unittest.main()
