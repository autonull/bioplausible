import unittest

import torch
import torch.nn as nn

from bioplausible.core import EqPropTrainer
from bioplausible.models import ConvEqProp, LoopedMLP, TransformerEqProp
from bioplausible.models.base import ModelRegistry
from bioplausible.models.registry import MODEL_REGISTRY


class TestRefactor(unittest.TestCase):
    def test_imports_and_models(self):
        """Test that core models can be instantiated."""
        mlp = LoopedMLP(10, 20, 5)
        self.assertIsInstance(mlp, nn.Module)

        # ConvEqProp inputs: input_channels, hidden_channels, output_dim
        conv = ConvEqProp(1, 16, 5)
        self.assertIsInstance(conv, nn.Module)

        # TransformerEqProp inputs: vocab_size, hidden_dim, output_dim
        trans = TransformerEqProp(100, 32, 5)
        self.assertIsInstance(trans, nn.Module)

    def test_trainer_init(self):
        """Test that trainer can be initialized."""
        mlp = LoopedMLP(10, 20, 5)
        trainer = EqPropTrainer(
            mlp, use_compile=False
        )  # Disable compile to avoid overhead/issues in test env
        self.assertIsInstance(trainer, EqPropTrainer)

    def test_models_registry(self):
        """Test that models are registered."""
        # Use ModelRegistry for algorithms (StandardEqProp, etc.)
        # 'eqprop' should be registered by StandardEqProp
        try:
            eqprop_cls = ModelRegistry.get("eqprop")
            self.assertIsNotNone(eqprop_cls)
        except ValueError:
            # Depending on import order, it might not be registered yet unless we import it
            from bioplausible.models.standard_eqprop import StandardEqProp

            eqprop_cls = ModelRegistry.get("eqprop")
            self.assertIsNotNone(eqprop_cls)

        # Instantiate
        model = eqprop_cls(input_dim=10, hidden_dim=20, output_dim=5)
        self.assertIsInstance(model, nn.Module)

    def test_ui_registry_specs(self):
        """Test that UI/Experiment registry specs are valid."""
        self.assertTrue(len(MODEL_REGISTRY) > 0)
        # Check that 'eqprop_mlp' is in there
        names = [spec.model_type for spec in MODEL_REGISTRY]
        self.assertIn("eqprop_mlp", names)


if __name__ == "__main__":
    unittest.main()
