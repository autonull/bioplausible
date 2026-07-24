import unittest

import torch

from bioplausible.config import GLOBAL_CONFIG
from bioplausible.core.registry import ComponentCategory
from bioplausible.core.registry import Registry
from bioplausible.core.trainer import CoreTrainer
from bioplausible.core.trainer import TrainerConfig
from bioplausible.hyperopt.tasks import BaseTask


# Mock Task for testing
class MockVisionTask(BaseTask):
    def __init__(self):
        super().__init__("mock_vision", "cpu", True)
        self._input_dim = 32
        self._output_dim = 10

    @property
    def task_type(self):
        return "vision"

    def setup(self):
        pass

    def get_batch(self, split="train", batch_size=32):
        return torch.randn(batch_size, 32), torch.randint(0, 10, (batch_size,))

    def create_trainer(self, model, **kwargs):
        # Stub for testing
        return model


class TestModelRegistryInstantiation(unittest.TestCase):
    def setUp(self):
        GLOBAL_CONFIG.quick_mode = True
        self.task = MockVisionTask()

    def _test_model(self, model_name, input_dim=32, input_shape=None):
        print(f"\nTesting {model_name}...")
        model_cls = Registry.get(ComponentCategory.MODEL, model_name)

        if input_shape:
            self.task._input_dim = (
                input_shape[1:] if len(input_shape) == 4 else input_shape
            )
        else:
            self.task._input_dim = input_dim

        model = model_cls(
            output_dim=10,
            input_dim=(input_dim if not input_shape else input_shape[1]),
            hidden_dim=16 if not input_shape else 64,
            num_layers=2,
        )

        if input_shape:
            x = torch.randn(*input_shape)
        else:
            x = torch.randn(4, input_dim)

        y = torch.randint(0, 10, (4,))

        out = model(x)
        self.assertIsNotNone(out)
        print(f"  Passed: {model_name} Output shape={out.shape}")

    def test_holomorphic_ep_instantiation(self):
        self._test_model("Holomorphic EqProp")

    def test_directed_ep_instantiation(self):
        self._test_model("Directed EqProp (Deep EP)")

    def test_finite_nudge_ep_instantiation(self):
        self._test_model("Finite-Nudge EqProp")

    def test_modern_conv_eqprop_instantiation(self):
        # Conv input shape
        # Pass input_dim=3 to indicate 3 channels
        self._test_model(
            "Conv EqProp (CIFAR-10)", input_dim=3, input_shape=(4, 3, 32, 32)
        )

    def test_hybrid_models(self):
        models_to_test = [
            "Adaptive Feedback Alignment",
            "Equilibrium Alignment",
            "Layerwise Equilibrium FA",
            "Energy Guided FA",
            "Predictive Coding Hybrid",
            "Sparse Equilibrium",
            "Momentum Equilibrium",
            "Stochastic FA",
            "Energy Minimizing FA",
        ]

        for model_name in models_to_test:
            try:
                self._test_model(model_name)
            except Exception as e:
                self.fail(f"Failed {model_name}: {e}")


if __name__ == "__main__":
    unittest.main()
