import unittest

import torch
from bioplausible.models.dfa_eqprop import DirectFeedbackAlignmentEqProp
from bioplausible.optimizers.learning_rules import AdaptiveFA


class TestAdaptiveFA(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DirectFeedbackAlignmentEqProp(
            input_dim=10, hidden_dim=20, output_dim=5
        ).to(self.device)
        self.optimizer = AdaptiveFA(self.model.parameters(), model=self.model)

    def test_forward(self):
        x = torch.randn(4, 10, device=self.device)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 5))

    def test_training_step(self):
        x = torch.randn(4, 10, device=self.device)
        y = torch.randint(0, 5, (4,), device=self.device)

        # Capture weights before
        w_before = self.model.layers[0].weight.data.clone()
        fb_before = self.optimizer.feedback_weights[1].data.clone()

        self.optimizer.zero_grad()
        self.optimizer.step(x=x, target=y)

        # Check if weights updated
        w_after = self.model.layers[0].weight.data
        self.assertFalse(torch.allclose(w_before, w_after))

        # Check if feedback weights updated (Alignment)
        fb_after = self.optimizer.feedback_weights[1].data
        self.assertFalse(torch.allclose(fb_before, fb_after))


if __name__ == "__main__":
    unittest.main()
