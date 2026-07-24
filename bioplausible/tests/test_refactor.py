import unittest

from torch import nn

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.zoo.models.eqprop import ConvEqProp, LoopedMLP, TransformerEqProp


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
        """Test that CoreTrainer can be initialized with a config."""
        from bioplausible.core.trainer import CoreTrainer, TrainerConfig

        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 784,
                "hidden_dim": 20,
                "output_dim": 10,
            },
            optimizer="adam",
            task="mnist",
            epochs=1,
            use_compile=False,
        )
        trainer = CoreTrainer(config)
        self.assertIsInstance(trainer, CoreTrainer)

    def test_models_registry(self):
        """Test that models are registered with the unified Registry."""
        # 'eqprop' (StandardEqProp) is registered by zoo/models/eqprop.py.
        names = Registry.list(ComponentCategory.MODEL).get("model", [])
        self.assertIn("eqprop", names)

        eqprop_cls = Registry.get(ComponentCategory.MODEL, "eqprop")
        self.assertIsNotNone(eqprop_cls)

        # And we can instantiate by name
        # NB: the registry class for "eqprop" is StandardEqProp which needs
        # a ModelConfig (or kwargs matching its signature). Smoke check:
        from bioplausible.zoo.base import ModelConfig

        spec = ModelConfig(
            name="eqprop",
            input_dim=10,
            hidden_dims=[20],
            output_dim=5,
        )
        model = eqprop_cls(spec)
        self.assertIsInstance(model, nn.Module)


if __name__ == "__main__":
    unittest.main()
