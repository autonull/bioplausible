import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from bioplausible.execution.interpretability import FeatureAttribution
from bioplausible.execution.robustness import RobustnessEvaluator


class SimpleModel(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)
        self.input_dim = input_dim

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.linear(x)


class TestInterpretability(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.fa = FeatureAttribution(self.model)

    def test_saliency_shape(self):
        x = torch.randn(1, 10)
        y = torch.tensor([0])
        saliency = self.fa.compute_saliency(x, y)
        self.assertEqual(saliency.shape, x.shape)
        self.assertFalse(saliency.requires_grad)

    def test_integrated_gradients_shape(self):
        x = torch.randn(1, 10)
        y = torch.tensor([0])
        ig = self.fa.compute_integrated_gradients(x, y, steps=5)
        self.assertEqual(ig.shape, x.shape)

    def test_forward_pass_flattening(self):
        # Model expects 10 features. Pass (1, 1, 5, 2) -> 10 features
        x = torch.randn(1, 1, 5, 2)
        # Check if _forward_pass handles it using model.input_dim
        out = self.fa._forward_pass(x)
        self.assertEqual(out.shape, (1, 2))


class TestRobustness(unittest.TestCase):
    def test_pgd_attack_execution(self):
        # Mocking
        mock_task = MagicMock()
        mock_task.get_batch.return_value = (torch.randn(10, 10), torch.zeros(10).long())

        mock_model = SimpleModel(10)

        evaluator = RobustnessEvaluator("test_model", "test_task", {})
        evaluator.device = "cpu"

        # We test the _test_pgd_attack method directly
        score = evaluator._test_pgd_attack(mock_model, mock_task, steps=2)

        self.assertTrue(
            0.0 <= score <= 1.5
        )  # Score can be slightly > 1 if adv accidentally improves accuracy?
        # Unlikely but possible in edge cases or if clean acc is low.
        # Usually <= 1.0.
        # Actually logic is acc_adv / acc_clean. If acc_clean is 0, returns 0.

        # Ensure it ran without error


if __name__ == "__main__":
    unittest.main()
