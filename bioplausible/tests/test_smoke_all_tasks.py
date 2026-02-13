import unittest

import torch
import torch.nn as nn

from bioplausible.hyperopt.tasks import create_task


class TestSmokeAllTasks(unittest.TestCase):
    def _test_task(self, task_name, task_type):
        print(f"\n>>> Smoke Testing Task: {task_name} ({task_type})")
        try:
            task = create_task(task_name, device="cpu", quick_mode=True)
            # Mock download/loading if possible?
            # For now, we rely on the environment having access or fast download.
            task.setup()
        except Exception as e:
            self.fail(f"{task_name} setup failed: {e}")

        # Basic Check
        if task_type == "vision":
            x, y = task.get_batch(split="train", batch_size=2)
            self.assertEqual(x.shape[0], 2)
        elif task_type == "lm":
            x, y = task.get_batch(split="train", batch_size=2)
            self.assertEqual(x.shape[0], 2)

        # Create minimal model
        input_dim = task.input_dim
        output_dim = task.output_dim

        # Simple MLP
        if task_type == "vision":
            # Flatten input if needed
            input_flat = input_dim
            model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_flat, 16),
                nn.ReLU(),
                nn.Linear(16, output_dim),
            )
        elif task_type == "rl":
            # RL expects (B, Obs) -> (B, Actions) logits
            model = nn.Sequential(
                nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, output_dim)
            )
        else:  # LM
            # LMTask input_dim is None (uses embeddings).
            # output_dim is vocab size.
            # Model expects (B, T).
            class SimpleLM(nn.Module):
                def __init__(self, vocab_size):
                    super().__init__()
                    self.emb = nn.Embedding(vocab_size, 16)
                    self.head = nn.Linear(16, vocab_size)

                def forward(self, x):
                    return self.head(self.emb(x))  # (B, T, V)

            model = SimpleLM(output_dim)

        # Trainer
        try:
            trainer = task.create_trainer(model)
            # Run one epoch (or episode)
            # For RL, episodes_per_epoch=1 to be fast
            if task_type == "rl":
                trainer.episodes_per_epoch = 1

            # Monkey patch episodes if needed
            metrics = trainer.train_epoch()
            self.assertIn("loss", metrics)
            print(f"    ✓ {task_name} passed with loss {metrics['loss']:.4f}")
        except Exception as e:
            self.fail(f"{task_name} training failed: {e}")

    # Vision
    def test_vision_digits(self):
        self._test_task("digits", "vision")

    def test_vision_usps(self):
        self._test_task("usps", "vision")

    def test_vision_kmnist(self):
        self._test_task("kmnist", "vision")

    def test_vision_mnist(self):
        self._test_task("mnist", "vision")

    def test_vision_fashion(self):
        self._test_task("fashion_mnist", "vision")

    # def test_vision_svhn(self): self._test_task("svhn", "vision")
    def test_vision_cifar10(self):
        self._test_task("cifar10", "vision")

    # def test_vision_cifar100(self): self._test_task("cifar100", "vision")

    # LM
    def test_lm_tiny_shakespeare(self):
        self._test_task("tiny_shakespeare", "lm")

    def test_lm_char_ngram(self):
        self._test_task("char_ngram", "lm")

    # RL
    def test_rl_cartpole(self):
        self._test_task("cartpole", "rl")

    def test_rl_acrobot(self):
        self._test_task("acrobot", "rl")

    # Pendulum is now fixed (Continuous RLTrainer)
    def test_rl_pendulum(self):
        self._test_task("pendulum", "rl")


if __name__ == "__main__":
    unittest.main()
