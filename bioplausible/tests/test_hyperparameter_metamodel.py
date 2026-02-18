import unittest
from dataclasses import dataclass

from bioplausible.hyperopt.hyperparameter_metamodel import (
    HYPERPARAM_METAMODEL,
    HyperparameterMetamodel,
    HyperparamScope,
)


@dataclass
class MockModelSpec:
    name: str
    family: str
    model_type: str = "default"


class TestHyperparameterMetamodel(unittest.TestCase):

    def test_universal_scope(self):
        """Test that universal params appear for all models."""
        specs = [
            MockModelSpec("Backprop", "baseline"),
            MockModelSpec("EqProp", "eqprop"),
            MockModelSpec("FA", "hybrid", "fa_variant"),
        ]

        for spec in specs:
            space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)
            self.assertIn("lr", space)
            self.assertIn("hidden_dim", space)
            self.assertIn("activation", space)

    def test_backprop_scope(self):
        """Test gradient-based params apply only to backprop/gradient models."""
        spec = MockModelSpec("Backprop Baseline", "baseline")
        space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)

        self.assertIn("optimizer", space)
        self.assertIn("dropout", space)

        # Should NOT have equilibrium params
        self.assertNotIn("beta", space)
        self.assertNotIn("nudge_type", space)

    def test_eqprop_scope(self):
        """Test equilibrium params apply only to EqProp models."""
        spec = MockModelSpec("EqProp MLP", "eqprop")
        space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)

        self.assertIn("beta", space)
        self.assertIn("steps", space)
        self.assertIn("nudge_type", space)

        # Should NOT have optimizer (EqProp uses energy dynamics)
        self.assertNotIn("optimizer", space)

    def test_hybrid_scope(self):
        """Test hybrid models get params from multiple scopes."""
        # FA + Equilibrium hybrid
        spec = MockModelSpec("Eq-FA Hybrid", "hybrid", "equilibrium_fa")
        space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)

        # Universal
        self.assertIn("lr", space)
        # Equilibrium
        self.assertIn("beta", space)
        # FA
        self.assertIn("fa_scale", space)
        # Gradient (defaults for hybrid)
        self.assertIn("optimizer", space)

    def test_transformer_scope(self):
        """Test transformer-specific params."""
        # Standard Transformer (Backprop)
        spec = MockModelSpec("Transformer", "baseline", "transformer")
        space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)

        self.assertIn("num_heads", space)
        self.assertIn("optimizer", space)

        # EqProp Transformer
        spec_eq = MockModelSpec("EqProp Transformer", "eqprop", "eqprop_transformer")
        space_eq = HYPERPARAM_METAMODEL.get_search_space_for_model(spec_eq)

        self.assertIn("num_heads", space_eq)
        self.assertIn("beta", space_eq)
        # EqProp transformer might NOT use optimizer if purely local
        # (Assuming current logic: EqProp family excludes gradient scope unless explicitly added)
        self.assertNotIn("optimizer", space_eq)

    def test_activation_constraint(self):
        """Test specific constraints (e.g. Holomorphic EqProp needs tanh)."""
        spec = MockModelSpec("Holomorphic EqProp", "eqprop")
        space = HYPERPARAM_METAMODEL.get_search_space_for_model(spec)

        self.assertEqual(space["activation"].choices, ["tanh"])

        # Verify standard model isn't affected
        spec_std = MockModelSpec("Standard EqProp", "eqprop")
        space_std = HYPERPARAM_METAMODEL.get_search_space_for_model(spec_std)
        self.assertNotEqual(space_std["activation"].choices, ["tanh"])

    def test_validate_config(self):
        """Test configuration validation logic."""
        spec = MockModelSpec("Backprop", "baseline")

        # Valid config
        valid_config = {"lr": 0.01, "optimizer": "adam"}
        errors = HYPERPARAM_METAMODEL.validate_config(spec, valid_config)
        self.assertEqual(len(errors), 0)

        # Invalid: passing EqProp param to Backprop
        invalid_config = {"lr": 0.01, "beta": 0.5}
        errors = HYPERPARAM_METAMODEL.validate_config(spec, invalid_config)
        self.assertTrue(len(errors) > 0)
        self.assertIn("beta", errors[0])


if __name__ == "__main__":
    unittest.main()
