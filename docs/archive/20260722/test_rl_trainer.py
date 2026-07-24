import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from bioplausible.equitile.rl import RolloutBuffer
from bioplausible.training.rl import RLTrainer  # noqa: E402


class TestRolloutBuffer(unittest.TestCase):
    def test_gae_computation(self):
        """Test GAE computation in RolloutBuffer."""
        buffer = RolloutBuffer(obs_dim=4, action_dim=2)

        # Add dummy data
        obs = torch.randn(4)
        action = torch.tensor(0)
        reward = torch.tensor(1.0)
        done = torch.tensor(0.0)
        value = torch.tensor(0.5)
        log_prob = torch.tensor(-0.1)

        # Add 2 steps
        buffer.add(obs, action, reward, done, value, log_prob)
        buffer.add(obs, action, reward, done, value, log_prob)

        # Get with GAE
        obs_batch, act_batch, adv_batch, ret_batch, log_prob_batch = buffer.get(
            gamma=0.99, lam=0.95
        )

        self.assertEqual(obs_batch.shape, (2, 4))
        self.assertEqual(adv_batch.shape, (2,))
        self.assertEqual(ret_batch.shape, (2,))


class TestRLTrainerRefactor(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"

        # Mock model for discrete (output 2)
        self.model_discrete = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2)
        )

        # Mock model for continuous (output 1)
        self.model_continuous = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    @patch("bioplausible.training.rl.gym.make")
    def test_train_episode_discrete(self, mock_make):
        """Test train_episode with discrete action space."""
        mock_env = MagicMock()
        # Mock action space (default MagicMock is not Box)
        mock_env.action_space = MagicMock()
        mock_env.action_space.n = 2

        mock_env.reset.return_value = (np.random.randn(4), {})

        # Step returns (obs, reward, terminated, truncated, info)
        # Return terminated=True on second call to exit loop fast
        mock_env.step.side_effect = [
            (np.random.randn(4), 1.0, False, False, {}),
            (np.random.randn(4), 1.0, True, False, {}),
        ]

        mock_make.return_value = mock_env

        trainer = RLTrainer(
            model=self.model_discrete, env_name="MockEnv-v0", device=self.device
        )

        self.assertFalse(trainer.is_continuous)

        metrics = trainer.train_episode()

        self.assertIn("reward", metrics)
        self.assertEqual(metrics["steps"], 2)

    @patch("bioplausible.training.rl.gym.make")
    def test_train_episode_continuous(self, mock_make):
        """Test train_episode with continuous action space."""
        mock_env = MagicMock()
        # Explicit Box for continuous detection
        mock_env.action_space = Box(
            low=np.array([-1.0]), high=np.array([1.0]), shape=(1,)
        )

        mock_env.reset.return_value = (np.random.randn(4), {})

        mock_env.step.side_effect = [
            (np.random.randn(4), 1.0, False, False, {}),
            (np.random.randn(4), 1.0, True, False, {}),
        ]

        mock_make.return_value = mock_env

        trainer = RLTrainer(
            model=self.model_continuous,
            env_name="MockEnvContinuous-v0",
            device=self.device,
        )

        self.assertTrue(trainer.is_continuous)

        metrics = trainer.train_episode()
        self.assertIn("reward", metrics)
        self.assertEqual(metrics["steps"], 2)


if __name__ == "__main__":
    unittest.main()
