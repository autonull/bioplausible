import shutil
import unittest
from pathlib import Path

import torch

from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage
from bioplausible.models.factory import create_model
from bioplausible.models.registry import MODEL_REGISTRY, get_model_spec


class TestHyperoptIntegration(unittest.TestCase):
    def setUp(self):
        self.test_db = "test_hyperopt_integration.db"
        self.storage = HyperoptStorage(self.test_db)
        self.storage.clear_all_trials()

    def tearDown(self):
        self.storage.close()
        if Path(self.test_db).exists():
            Path(self.test_db).unlink()

    def test_instantiate_all_models_lm(self):
        """Test instantiation of all model types for LM task using factory."""
        vocab_size = 65
        for spec in MODEL_REGISTRY:
            try:
                model = create_model(
                    spec,
                    input_dim=None,
                    output_dim=vocab_size,
                    hidden_dim=32,
                    num_layers=2,
                    device="cpu",
                    task_type="lm",
                )
                self.assertIsNotNone(model)

                # Check embedding logic which was previously in ExperimentAlgorithm
                # Factory now attaches it.
                if spec.model_type in ["eqprop_mlp", "dfa", "chl", "deep_hebbian"]:
                    self.assertTrue(getattr(model, "has_embed", False))
                    self.assertIsNotNone(getattr(model, "embed", None))

            except Exception as e:
                self.fail(f"Failed to instantiate {spec.name} for LM: {e}")

    def test_instantiate_vision_models(self):
        """Test instantiation for Vision tasks (vector input)."""
        input_dim = 784
        output_dim = 10
        # Only test models compatible with vector input (usually MLPs)
        mlp_specs = [
            s
            for s in MODEL_REGISTRY
            if s.model_type in ["backprop", "eqprop_mlp", "dfa", "chl", "deep_hebbian"]
        ]

        for spec in mlp_specs:
            try:
                model = create_model(
                    spec,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=32,
                    num_layers=2,
                    device="cpu",
                    task_type="vision",
                )
                self.assertIsNotNone(model)
                self.assertFalse(getattr(model, "has_embed", False))
            except Exception as e:
                self.fail(f"Failed to instantiate {spec.name} for Vision: {e}")

    def test_rl_runner_step(self):
        """Test RL runner execution (integration with RLTrainer)."""
        # Create a runner for CartPole
        runner = TrialRunner(
            storage=self.storage, device="cpu", task="cartpole", quick_mode=True
        )

        # Override epochs to 1 for speed
        runner.epochs = 1

        # Pick a compatible model (e.g., EqProp MLP)
        spec = [m for m in MODEL_REGISTRY if m.model_type == "eqprop_mlp"][0]

        config = {"hidden_dim": 32, "num_layers": 1, "lr": 0.01, "steps": 5}

        trial_id = self.storage.create_trial(spec.name, config)

        # Run
        success = runner.run_trial(trial_id)
        self.assertTrue(success)

        # Check results
        trial = self.storage.get_trial(trial_id)
        self.assertEqual(trial.status, "completed")
        self.assertIsNotNone(trial.accuracy)  # Reward


if __name__ == "__main__":
    unittest.main()
