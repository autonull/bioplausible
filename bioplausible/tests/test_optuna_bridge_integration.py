import unittest

import optuna

from bioplausible.hyperopt.hyperparameter_metamodel import HYPERPARAM_METAMODEL
from bioplausible.hyperopt.optuna_bridge import (create_optuna_space,
                                                 create_study)


class TestOptunaBridgeIntegration(unittest.TestCase):

    def test_create_optuna_space_eqprop(self):
        """Test that EqProp model gets EqProp params via the bridge."""
        study = optuna.create_study()
        trial = study.ask()

        # Should use Metamodel under the hood now
        config = create_optuna_space(trial, "EqProp MLP")

        self.assertIn("beta", config)
        self.assertIn("steps", config)
        self.assertIn("lr", config)
        self.assertNotIn("optimizer", config)

    def test_create_optuna_space_backprop(self):
        """Test that Backprop model gets Backprop params via the bridge."""
        study = optuna.create_study()
        trial = study.ask()

        config = create_optuna_space(trial, "Backprop Baseline")

        self.assertIn("optimizer", config)
        self.assertIn("lr", config)
        self.assertNotIn("beta", config)

    def test_constraints_propagation(self):
        """Test that constraints passed to bridge are respected by metamodel filtering."""
        study = optuna.create_study()
        trial = study.ask()

        constraints = {"max_hidden": 32}
        config = create_optuna_space(
            trial, "Backprop Baseline", constraints=constraints
        )

        # Depending on how the new logic samples, we expect hidden_dim to be <= 32
        # The metamodel spec has [32, 64, 128...]
        # If constraint works, it should pick 32 (or filtered set)
        self.assertLessEqual(config["hidden_dim"], 32)


if __name__ == "__main__":
    unittest.main()
