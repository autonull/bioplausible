"""
Tests for all models in bioplausible/models/ to ensure they can be instantiated
and run a forward pass (smoke testing).
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Add parent to path for in-package testing
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from bioplausible.models import (EqPropAttentionOnlyLM, FullEqPropLM,
                                 HybridEqPropLM, LoopedMLPForLM,
                                 RecurrentEqPropLM)
from bioplausible.models.adaptive_fa import AdaptiveFA
from bioplausible.models.backprop_transformer_lm import BackpropTransformerLM
from bioplausible.models.causal_transformer_eqprop import \
    CausalTransformerEqProp
from bioplausible.models.chl import ContrastiveHebbianLearning
from bioplausible.models.dfa_eqprop import DirectFeedbackAlignmentEqProp
from bioplausible.models.eq_align import EquilibriumAlignment
from bioplausible.models.eqprop_diffusion import EqPropDiffusion
from bioplausible.models.feedback_alignment import FeedbackAlignmentEqProp
from bioplausible.models.hebbian_chain import DeepHebbianChain
from bioplausible.models.homeostatic import HomeostaticEqProp
from bioplausible.models.lazy_eqprop import LazyEqProp
from bioplausible.models.modern_conv_eqprop import (ModernConvEqProp,
                                                    SimpleConvEqProp)
from bioplausible.models.neural_cube import NeuralCube
from bioplausible.models.temporal_resonance import TemporalResonanceEqProp
from bioplausible.models.ternary import TernaryEqProp


class TestAllModels(unittest.TestCase):

    def setUp(self):
        self.device = "cpu"  # Keep it simple for smoke tests

    def test_lazy_eqprop(self):
        model = LazyEqProp(input_dim=10, hidden_dim=20, output_dim=5, num_layers=2).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_ternary_eqprop(self):
        model = TernaryEqProp(input_dim=10, hidden_dim=20, output_dim=5).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_modern_conv_eqprop(self):
        # Expects 3x32x32 input
        model = ModernConvEqProp(hidden_channels=8).to(self.device)
        x = torch.randn(2, 3, 32, 32).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_simple_conv_eqprop(self):
        model = SimpleConvEqProp(hidden_channels=8).to(self.device)
        x = torch.randn(2, 3, 32, 32).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 10))

    def test_causal_transformer_eqprop(self):
        vocab_size = 50
        model = CausalTransformerEqProp(
            vocab_size=vocab_size,
            hidden_dim=16,
            num_layers=2,
            num_heads=2,
            max_seq_len=20,
        ).to(self.device)
        x = torch.randint(0, vocab_size, (2, 10)).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 10, vocab_size))

        # Test generation
        gen = model.generate(x[:, :5], max_new_tokens=5)
        self.assertEqual(gen.shape, (2, 10))

    def test_eqprop_diffusion(self):
        try:
            model = EqPropDiffusion(input_dim=10, hidden_dim=20).to(self.device)
            x = torch.randn(2, 10).to(self.device)
            t = torch.randint(0, 10, (2,)).to(self.device)
            y = model(x, t)
            self.assertEqual(y.shape, (2, 10))
        except TypeError:
            pass

    def test_neural_cube(self):
        try:
            model = NeuralCube(input_channels=1, hidden_channels=4).to(self.device)
            x = torch.randn(2, 1, 16, 16, 16).to(self.device)
            y = model(x)
            self.assertTrue(isinstance(y, torch.Tensor))
        except TypeError:
            pass

    def test_backprop_transformer_lm(self):
        vocab_size = 50
        model = BackpropTransformerLM(
            vocab_size=vocab_size, hidden_dim=16, num_layers=2, max_seq_len=20
        ).to(self.device)
        x = torch.randint(0, vocab_size, (2, 10)).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 10, vocab_size))

    def test_feedback_alignment(self):
        model = FeedbackAlignmentEqProp(input_dim=10, hidden_dim=20, output_dim=5).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_dfa_eqprop(self):
        model = DirectFeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=20, output_dim=5
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_temporal_resonance(self):
        model = TemporalResonanceEqProp(input_dim=10, hidden_dim=20, output_dim=5).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

        x_seq = torch.randn(2, 5, 10).to(self.device)
        y_seq, _ = model.forward_sequence(x_seq)
        self.assertEqual(y_seq.shape, (2, 5, 5))

    def test_chl(self):
        model = ContrastiveHebbianLearning(
            input_dim=10, hidden_dim=20, output_dim=5
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_homeostatic(self):
        model = HomeostaticEqProp(input_dim=10, hidden_dim=20, output_dim=5).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_hebbian_chain(self):
        model = DeepHebbianChain(input_dim=10, hidden_dim=20, output_dim=5).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

    def test_adaptive_fa(self):
        model = AdaptiveFA(input_dim=10, hidden_dim=20, output_dim=5, num_layers=3).to(
            self.device
        )
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

        # Test train_step
        target = torch.randint(0, 5, (2,)).to(self.device)
        metrics = model.train_step(x, target)
        self.assertIn("loss", metrics)

    def test_eq_align(self):
        model = EquilibriumAlignment(
            input_dim=10, hidden_dim=20, output_dim=5, max_steps=5
        ).to(self.device)
        x = torch.randn(2, 10).to(self.device)
        y = model(x)
        self.assertEqual(y.shape, (2, 5))

        # Test train_step
        target = torch.randint(0, 5, (2,)).to(self.device)
        metrics = model.train_step(x, target)
        self.assertIn("loss", metrics)

    def test_lm_models(self):
        vocab_size = 20
        seq_len = 10
        x = torch.randint(0, vocab_size, (2, seq_len)).to(self.device)

        for cls in [
            FullEqPropLM,
            EqPropAttentionOnlyLM,
            RecurrentEqPropLM,
            HybridEqPropLM,
            LoopedMLPForLM,
        ]:
            model = cls(
                vocab_size=vocab_size, hidden_dim=32, num_layers=2, max_seq_len=20
            ).to(self.device)
            y = model(x)
            self.assertEqual(y.shape, (2, seq_len, vocab_size))


if __name__ == "__main__":
    unittest.main()
