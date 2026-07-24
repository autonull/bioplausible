import unittest
from pathlib import Path

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.hyperopt.experiment import TrialRunner
from bioplausible.hyperopt.storage import HyperoptStorage


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
        """Test instantiation of all registered models for an LM task."""
        vocab_size = 65
        names = Registry.list(ComponentCategory.MODEL).get("model", [])
        # LM-compatible names: filter by Domain.LM in metadata.
        lm_models = []
        for name in names:
            meta = Registry.get_metadata(ComponentCategory.MODEL, name)
            if any(d.value == "lm" for d in meta.domains):
                lm_models.append(name)

        tested_count = 0
        for name in lm_models:
            model_cls = Registry.get(ComponentCategory.MODEL, name)
            try:
                # Call via standard kwargs (most LM models accept these).
                model = model_cls(
                    input_dim=vocab_size,
                    output_dim=vocab_size,
                    hidden_dim=32,
                    num_layers=2,
                )
            except TypeError:
                # Model has its own constructor signature.
                continue
            self.assertIsNotNone(model)
            tested_count += 1

        self.assertGreater(tested_count, 0, "No LM models instantiated")

    def test_instantiate_vision_models(self):
        """Test instantiation for vision tasks (vector input).

        Models that don't accept standard (input_dim/hidden_dim/output_dim)
        kwargs (e.g. transformer LMs) are skipped rather than failing - they
        have their own instantiation paths.
        """
        input_dim = 784
        output_dim = 10
        names = Registry.list(ComponentCategory.MODEL).get("model", [])
        vision_models = []
        for name in names:
            meta = Registry.get_metadata(ComponentCategory.MODEL, name)
            if any(d.value == "vision" for d in meta.domains):
                vision_models.append(name)

        tested_count = 0
        for name in vision_models[:8]:
            model_cls = Registry.get(ComponentCategory.MODEL, name)
            try:
                model = model_cls(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=32,
                    num_layers=2,
                )
            except TypeError:
                # Model has its own constructor signature; not vector-MLP shaped.
                continue
            self.assertIsNotNone(model)
            self.assertFalse(getattr(model, "has_embed", False))
            tested_count += 1
        self.assertGreater(tested_count, 0, "No vision models instantiated")

    def test_rl_runner_step(self):
        """Test RL runner execution (integration with RLTrainer)."""
        runner = TrialRunner(
            storage=self.storage, device="cpu", task="cartpole", quick_mode=True
        )
        runner.epochs = 1

        # Pick a model that's registered.
        names = Registry.list(ComponentCategory.MODEL).get("model", [])
        target = "eqprop_mlp"
        if target not in names:
            self.skipTest(f"'{target}' not registered")

        config = {"hidden_dim": 32, "num_layers": 1, "lr": 0.01, "steps": 5}

        trial_id = self.storage.create_trial(target, config)

        success = runner.run_trial(trial_id)
        self.assertTrue(success)

        trial = self.storage.get_trial(trial_id)
        self.assertEqual(trial.status, "completed")
        self.assertIsNotNone(trial.accuracy)


if __name__ == "__main__":
    unittest.main()
