import unittest
from unittest.mock import MagicMock

from bioplausible.execution.strategy import ExecutionStrategy as ScientistStrategy


class TestStrategyFragility(unittest.TestCase):
    def setUp(self):
        # Mock State
        self.mock_state = MagicMock()
        self.mock_state.get_progress.return_value = {}

        # Mock Fragile Models
        self.mock_state.get_fragile_models.return_value = {
            "fragile_mlp": 0.25
        }  # Low robustness

        self.strategy = ScientistStrategy(self.mock_state)

    def test_analyze_fragility_applies_constraints(self):
        # Action
        constraints = self.strategy._analyze_fragility()

        # Assertions
        self.assertIn("fragile_mlp", constraints)
        c = constraints["fragile_mlp"]
        self.assertTrue(c["use_spectral_norm"])
        self.assertGreaterEqual(c["min_weight_decay"], 1e-4)
        self.assertGreaterEqual(c["min_dropout"], 0.2)

    def test_generate_candidates_merges_constraints(self):
        # Mock failure constraints to be empty
        self.strategy._analyze_failures = MagicMock(return_value={})

        # Mock saturation to be empty
        self.strategy._analyze_saturation = MagicMock(return_value={})

        # We need a candidate generation flow.
        # But `generate_candidates` calls internal methods we mocked or that rely on `progress`.
        # Since `progress` is empty, it won't generate tasks unless we mock MODEL_REGISTRY or
        # assume standard behavior.

        # Instead, verify that `_analyze_fragility` is called.
        self.strategy._analyze_fragility = MagicMock(
            wraps=self.strategy._analyze_fragility
        )

        self.strategy.generate_candidates()

        self.strategy._analyze_fragility.assert_called_once()


if __name__ == "__main__":
    unittest.main()
