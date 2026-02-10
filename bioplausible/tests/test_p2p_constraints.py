import unittest
from unittest.mock import MagicMock, patch

from bioplausible.hyperopt.search_space import SearchSpace
from bioplausible.p2p.evolution import P2PEvolution


class TestP2PConstraints(unittest.TestCase):

    def test_search_space_constraints(self):
        # Create a search space
        space = SearchSpace(
            "test",
            {
                "hidden_dim": [64, 128, 256, 512],
                "num_layers": (2, 10, "int"),
                "steps": (5, 50, "int"),
            },
        )

        # Apply constraints
        constraints = {"max_hidden": 128, "max_layers": 4, "max_steps": 20}

        constrained_space = space.apply_constraints(constraints)

        # Verify
        # 1. Discrete Choice
        self.assertEqual(constrained_space.params["hidden_dim"], [64, 128])

        # 2. Range
        self.assertEqual(constrained_space.params["num_layers"], (2, 4, "int"))
        self.assertEqual(constrained_space.params["steps"], (5, 20, "int"))

        # Sample to double check
        sample = constrained_space.sample()
        self.assertLessEqual(sample["hidden_dim"], 128)
        self.assertLessEqual(sample["num_layers"], 4)
        self.assertLessEqual(sample["steps"], 20)

    @patch("bioplausible.p2p.evolution.DHTNode")
    @patch("bioplausible.p2p.evolution.Worker")
    def test_p2p_evolution_settings(self, mock_worker, mock_dht):
        # Test Quick Mode
        evo = P2PEvolution(discovery_mode="quick")

        # Mock dependencies to allow instantiation
        evo.dht = MagicMock()

        # We can't easily test internal loop variables without running it,
        # but we can verify the object state.
        self.assertEqual(evo.discovery_mode, "quick")

        # Test Constraints passed
        constraints = {"max_hidden": 32}
        evo_constr = P2PEvolution(constraints=constraints)
        self.assertEqual(evo_constr.constraints, constraints)


if __name__ == "__main__":
    unittest.main()
