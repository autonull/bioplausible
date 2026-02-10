import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.core import EqPropTrainer
from bioplausible.models.adaptive_fa import AdaptiveFA
from bioplausible.models.backprop_transformer_lm import BackpropTransformerLM
from bioplausible.models.causal_transformer_eqprop import \
    CausalTransformerEqProp
from bioplausible.models.chl import ContrastiveHebbianLearning
from bioplausible.models.conv_eqprop import ConvEqProp
from bioplausible.models.deep_ep import DirectedEP
from bioplausible.models.dfa_eqprop import DirectFeedbackAlignmentEqProp
from bioplausible.models.eq_align import EquilibriumAlignment
from bioplausible.models.eqprop_lm_variants import (EqPropAttentionOnlyLM,
                                                    FullEqPropLM,
                                                    HybridEqPropLM,
                                                    LoopedMLPForLM,
                                                    RecurrentEqPropLM)
from bioplausible.models.feedback_alignment import FeedbackAlignmentEqProp
from bioplausible.models.finite_nudge_ep import FiniteNudgeEP
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.models.holomorphic_ep import HolomorphicEP
from bioplausible.models.homeostatic import HomeostaticEqProp
from bioplausible.models.lazy_eqprop import LazyEqProp
from bioplausible.models.looped_mlp import BackpropMLP, LoopedMLP
from bioplausible.models.modern_conv_eqprop import (ModernConvEqProp,
                                                    SimpleConvEqProp)
from bioplausible.models.temporal_resonance import TemporalResonanceEqProp
from bioplausible.models.ternary import TernaryEqProp
from bioplausible.models.transformer_eqprop import TransformerEqProp


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
        model = AdaptiveFA(input_dim=10, hidden_dim=32, output_dim=5, num_layers=2).to(
            self.device
        )
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # AdaptiveFA has a custom train_step
        metrics = self._run_custom_training_step(model, x, y)
        self.assertIn("loss", metrics)

    def test_chl(self):
        model = ContrastiveHebbianLearning(
            input_dim=10, hidden_dim=32, output_dim=5
        ).to(self.device)
        x = torch.randn(self.batch_size, 10).to(self.device)
        y = torch.randint(0, 5, (self.batch_size,)).to(self.device)

        # CHL often has custom train_step
        if hasattr(model, "train_step"):
            metrics = self._run_custom_training_step(model, x, y)
            self.assertIn("loss", metrics)
        else:
            loss = self._run_training_step(model, x, y)
            self.assertGreater(loss, 0)

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
            metrics = self._run_custom_training_step(model, x, y)
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
        """Test the EqPropTrainer with a simple model to ensure the high-level API works."""
        model = LoopedMLP(input_dim=10, hidden_dim=32, output_dim=5).to(self.device)
        dataset = TensorDataset(torch.randn(10, 10), torch.randint(0, 5, (10,)))
        loader = DataLoader(dataset, batch_size=2)

        trainer = EqPropTrainer(
            model, use_compile=False
        )  # Disable compile for smoke test speed
        history = trainer.fit(loader, epochs=1)

        self.assertIn("train_loss", history)
        self.assertGreater(len(history["train_loss"]), 0)

    def test_real_data_digits(self):
        """Test training on real Digits dataset (sklearn)."""
        try:
            from bioplausible.datasets import get_vision_dataset

            # Digits: 64 input features (8x8 flattened), 10 classes
            train_dataset = get_vision_dataset("digits", train=True, flatten=True)

            model = LoopedMLP(
                input_dim=64, hidden_dim=32, output_dim=10, max_steps=10
            ).to(self.device)

            loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            trainer = EqPropTrainer(model, use_compile=False)
            history = trainer.fit(loader, epochs=2)

            self.assertIn("train_loss", history)
            # Check loss went down (basic sanity check)
            self.assertLess(
                history["train_loss"][-1], history["train_loss"][0] * 1.5
            )  # allow some fluctuation but not explosion

        except ImportError as e:
            if "scikit-learn" in str(e):
                self.skipTest("scikit-learn not installed")
            else:
                raise e


if __name__ == "__main__":
    unittest.main()
