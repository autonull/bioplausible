import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.core.trainer import CoreTrainer
from bioplausible.zoo.models.backprop import BackpropTransformerLM
from bioplausible.zoo.models.eqprop import (
    BackpropMLP,
    CausalTransformerEqProp,
    ConvEqProp,
    DirectedEP,
    EqPropAttentionOnlyLM,
    FiniteNudgeEP,
    FullEqPropLM,
    HolomorphicEP,
    HomeostaticEqProp,
    HybridEqPropLM,
    LazyEqProp,
    LoopedMLP,
    LoopedMLPForLM,
    ModernConvEqProp,
    RecurrentEqPropLM,
    SimpleConvEqProp,
    TemporalResonanceEqProp,
    TernaryEqProp,
    TransformerEqProp,
)
from bioplausible.zoo.models.fa import (
    DirectFeedbackAlignmentEqProp,
    EquilibriumAlignment,
    FeedbackAlignmentEqProp,
)
from bioplausible.zoo.models.hebbian import DeepHebbianChain
from bioplausible.zoo.propagators.fa import AdaptiveFA
from bioplausible.zoo.propagators.hebbian import ContrastiveHebbianLearning


class TestSmokeTraining(unittest.TestCase):
    """
    Comprehensive smoke tests to verify that all models can perform a full training step.
    This ensures models are valid for comparison, hyperparameter search, and training loops.
    """

    def setUp(self):
        self.device = "cpu"  # Use CPU for smoke tests to be environment agnostic
        self.batch_size = 4

    def _run_training_step(self, model, x, y, criterion=None, optimizer=None):
        """Helper to run a standard BPTT training step."""
        if criterion is None:
            if x.dtype == torch.long or y.dtype == torch.long:
                criterion = nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        optimizer.zero_grad()

        output = model(x)

        # Reshape output/target for sequence models if needed
        if output.ndim == 3 and y.ndim == 2:
            # (B, T, V) -> (B*T, V)
            output = output.view(-1, output.size(-1))
            y = y.view(-1)
        elif output.ndim == 3 and y.ndim == 3:
            # (B, T, D) vs (B, T, D)
            pass

        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        return loss.item()

    def _run_custom_training_step(self, model, x, y):
        """Helper to run a custom train_step if available."""
        if not hasattr(model, "train_step"):
            raise ValueError(f"Model {type(model)} does not have train_step method")

        metrics = model.train_step(x, y)
        return metrics

    def test_looped_mlp_bptt(self):
        # Hyperparameter search simulation: non-default params
        model = LoopedMLP(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            max_steps=10,
            gradient_method="bptt",
        ).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_looped_mlp_equilibrium(self):
        # Test implicit differentiation path
        model = LoopedMLP(
            input_dim=10,
            hidden_dim=32,
            output_dim=5,
            max_steps=10,
            gradient_method="equilibrium",
        ).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_backprop_mlp(self):
        model = BackpropMLP(input_dim=10, hidden_dim=32, output_dim=5).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_conv_eqprop(self):
        model = ConvEqProp(
            input_channels=3, hidden_channels=8, output_dim=10, max_steps=5
        ).to(self.device)
        x = torch.randn(self.batch_size, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_modern_conv_eqprop(self):
        # ModernConvEqProp is hardcoded for CIFAR-like (3 in, 10 out)
        model = ModernConvEqProp(hidden_channels=16, eq_steps=5).to(self.device)
        x = torch.randn(self.batch_size, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_simple_conv_eqprop(self):
        # SimpleConvEqProp is hardcoded for CIFAR-like (3 in, 10 out)
        model = SimpleConvEqProp(hidden_channels=8).to(self.device)
        x = torch.randn(self.batch_size, 3, 32, 32).to(self.device)
        y = torch.randint(0, 10, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_transformer_eqprop(self):
        vocab_size = 50
        model = TransformerEqProp(
            vocab_size=vocab_size,
            hidden_dim=32,
            output_dim=10,
            num_layers=2,
            max_seq_len=20,
        ).to(self.device)
        x = torch.randint(0, vocab_size, (self.batch_size, 10)).to(self.device)
        # Assuming classification on the sequence
        y = torch.randint(0, 10, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_causal_transformer_eqprop(self):
        vocab_size = 50
        model = CausalTransformerEqProp(
            vocab_size=vocab_size, hidden_dim=32, num_layers=2, max_seq_len=20
        ).to(self.device)
        x = torch.randint(0, vocab_size, (self.batch_size, 10)).to(self.device)
        y = torch.randint(0, vocab_size, (self.batch_size, 10)).to(
            self.device
        )  # Target is sequence

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_backprop_transformer_lm(self):
        vocab_size = 50
        model = BackpropTransformerLM(
            vocab_size=vocab_size, hidden_dim=32, num_layers=2, max_seq_len=20
        ).to(self.device)
        x = torch.randint(0, vocab_size, (self.batch_size, 10)).to(self.device)
        y = torch.randint(0, vocab_size, (self.batch_size, 10)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_feedback_alignment(self):
        model = FeedbackAlignmentEqProp(input_dim=10, hidden_dim=32, output_dim=5).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # FA models usually support BPTT interface but modify gradients internally
        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_dfa_eqprop(self):
        model = DirectFeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=32, output_dim=5
        ).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_adaptive_fa(self):
        from bioplausible.zoo.models.fa import DirectFeedbackAlignmentEqProp

        model = DirectFeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=32, output_dim=5
        ).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # AdaptiveFA is an optimizer for FA networks
        optimizer = AdaptiveFA(model.parameters(), model=model)

        optimizer.step(x=x, target=y)
        loss = torch.nn.functional.cross_entropy(model(x), y).item()
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_chl(self):
        # ContrastiveHebbianLearning is now a propagator (operates on a model).
        # Use a small LoopedMLP model and exercise the propagator step instead.
        from bioplausible.zoo.models.eqprop import LoopedMLP

        model = LoopedMLP(
            input_dim=10, hidden_dim=32, output_dim=5
        ).to(self.device)
        params = list(model.parameters())
        chl = ContrastiveHebbianLearning(
            params, model=model, lr=0.01
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # CHL has its own custom step (x, target).
        chl.step(x=x, target=y)
        # No exception means it ran. Light sanity: assert the optimizer ptr.
        self.assertTrue(hasattr(chl, "clamp_strength"))

    def test_equilibrium_alignment(self):
        model = EquilibriumAlignment(input_dim=10, hidden_dim=32, output_dim=5).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        if hasattr(model, "train_step"):
            metrics = self._run_custom_training_step(model, x, y)
            self.assertIn("loss", metrics)
        else:
            loss = self._run_training_step(model, x, y)
            self.assertGreater(loss, 0)

    def test_temporal_resonance(self):
        model = TemporalResonanceEqProp(input_dim=10, hidden_dim=32, output_dim=5).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_ternary_eqprop(self):
        model = TernaryEqProp(input_dim=10, hidden_dim=32, output_dim=5).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_lazy_eqprop(self):
        model = LazyEqProp(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        loss = self._run_training_step(model, x, y)
        self.assertGreater(loss, 0)

    def test_hebbian_chain(self):
        model = DeepHebbianChain(input_dim=10, hidden_dim=32, output_dim=5).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        if hasattr(model, "train_step"):
            self._run_custom_training_step(model, x, y)
        else:
            loss = self._run_training_step(model, x, y)
            self.assertGreater(loss, 0)

    def test_homeostatic(self):
        model = HomeostaticEqProp(input_dim=10, hidden_dim=32, output_dim=5).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # Override _run_training_step to pass apply_homeostasis=False
        # Homeostasis modifies weights in-place which can conflict with autograd in smoke tests
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()
        optimizer.zero_grad()
        output = model(x, apply_homeostasis=False)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        self.assertGreater(loss.item(), 0)

    def test_holomorphic_ep(self):
        model = HolomorphicEP(input_dim=10, hidden_dim=32, output_dim=5).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        if hasattr(model, "train_step"):
            metrics = self._run_custom_training_step(model, x, y)
            self.assertIn("loss", metrics)

    def test_directed_ep(self):
        model = DirectedEP(input_dim=10, hidden_dim=32, output_dim=5).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        if hasattr(model, "train_step"):
            metrics = self._run_custom_training_step(model, x, y)
            self.assertIn("loss", metrics)

    def test_finite_nudge_ep(self):
        model = FiniteNudgeEP(input_dim=10, hidden_dim=32, output_dim=5, beta=1.0).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        if hasattr(model, "train_step"):
            metrics = self._run_custom_training_step(model, x, y)
            self.assertIn("loss", metrics)

    def test_lm_variants(self):
        vocab_size = 20
        seq_len = 10
        classes = [
            FullEqPropLM,
            EqPropAttentionOnlyLM,
            RecurrentEqPropLM,
            HybridEqPropLM,
            LoopedMLPForLM,
        ]

        for cls in classes:
            with self.subTest(model_class=cls.__name__):
                model = cls(
                    vocab_size=vocab_size, hidden_dim=32, num_layers=2, max_seq_len=20
                ).to(self.device)
                x = torch.randint(0, vocab_size, (self.batch_size, seq_len)).to(
                    self.device
                )
                y = torch.randint(0, vocab_size, (self.batch_size, seq_len)).to(
                    self.device
                )

                loss = self._run_training_step(model, x, y)
                self.assertGreater(loss, 0)

    def test_eqprop_trainer_integration(self):
        """Test the CoreTrainer (canonical unified trainer) with a tiny synthetic dataset
        by setting custom DataLoaders after instantiation."""
        from bioplausible.core.trainer import TrainerConfig

        dataset = TensorDataset(torch.randn(10, 784), torch.randint(0, 10, (10,)))
        loader = DataLoader(dataset, batch_size=2)
        val_loader = DataLoader(dataset, batch_size=2)

        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 784,
                "hidden_dim": 32,
                "output_dim": 10,
            },
            optimizer="adam",
            task="mnist",
            epochs=1,
            batches_per_epoch=2,
            val_batches=2,
            use_compile=False,
            device=self.device,
        )
        trainer = CoreTrainer(config)
        trainer.setup()
        # Override with our small custom loaders.
        trainer.train_loader = loader
        trainer.val_loader = val_loader
        history = trainer.fit()
        self.assertGreater(len(history), 0)
        self.assertIn("train_loss", history[0].to_dict())

    def test_real_data_digits(self):
        """Test training on real Digits dataset (sklearn)."""
        try:
            from bioplausible.datasets import get_vision_dataset
        except ImportError as e:
            if "scikit-learn" in str(e):
                self.skipTest("scikit-learn not installed")
            raise

        from bioplausible.core.trainer import TrainerConfig

        # Digits: 64 input features (8x8 flattened), 10 classes
        train_dataset = get_vision_dataset("digits", train=True, flatten=True)

        loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=16, shuffle=False)

        config = TrainerConfig(
            model="eqprop_mlp",
            model_kwargs={
                "input_dim": 64,
                "hidden_dim": 32,
                "output_dim": 10,
            },
            optimizer="adam",
            task="mnist",
            epochs=2,
            batches_per_epoch=2,
            val_batches=2,
            use_compile=False,
            device=self.device,
        )
        trainer = CoreTrainer(config)
        trainer.setup()
        trainer.train_loader = loader
        trainer.val_loader = val_loader
        history = trainer.fit()
        self.assertGreater(len(history), 0)


if __name__ == "__main__":
    unittest.main()
