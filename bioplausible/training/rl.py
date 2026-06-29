import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box

from bioplausible.tracking import ExperimentTracker
from bioplausible.training.base import BaseTrainer

# Constants
MAX_STEPS = 1000


class RLTrainer(BaseTrainer):
    """
    Reinforcement Learning Trainer for EqProp and Backprop models.
    Implements standard REINFORCE (Policy Gradient).
    """

    def __init__(
        self,
        model: nn.Module,
        env_name: str,
        device: str = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.99,
        seed: int = 42,
        episodes_per_epoch: int = 10,
        tracker: Optional[ExperimentTracker] = None,
        **kwargs: Any,
    ):
        super().__init__(model, device)
        self.model = self.model.to(device)
        self.gamma = gamma
        self.episodes_per_epoch = episodes_per_epoch
        self.tracker = tracker

        # Initialize Environment
        self.env = gym.make(env_name)

        # Detect Action Space
        self.is_continuous = isinstance(self.env.action_space, Box)
        self._setup_action_space(device, lr)

        # Seeding
        self._seed_environment(seed)

        # History
        self.reward_history: List[float] = []
        self.loss_history: List[float] = []

    def _setup_action_space(self, device: str, lr: float) -> None:
        """Initialize action space specific parameters."""
        if self.is_continuous:
            assert isinstance(self.env.action_space, Box)
            high = self.env.action_space.high
            low = self.env.action_space.low

            scale = (high - low) / 2.0
            bias = (high + low) / 2.0

            self.action_scale = torch.tensor(scale, device=device).float()
            self.action_bias = torch.tensor(bias, device=device).float()

            # Initialize log_std parameter for continuous policy
            self.log_std = nn.Parameter(
                torch.zeros(self.env.action_space.shape[0], device=device)
            )
            # Add log_std to optimizer
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + [self.log_std], lr=lr
            )
        else:
            self.action_scale = None
            self.action_bias = None
            self.log_std = None
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def _seed_environment(self, seed: int) -> None:
        """Seed the environment and random generators."""
        try:
            self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            self.env.reset()

        torch.manual_seed(seed)
        np.random.seed(seed)

    def _reset_env(self) -> np.ndarray:
        """Reset environment and return observation."""
        if hasattr(self.env, "reset"):
            obs_info = self.env.reset()
            if isinstance(obs_info, tuple):
                return obs_info[0]
            return obs_info
        else:
            # Fallback for older gym
            return self.env.reset()

    def _step_env(self, action: Any) -> Tuple[np.ndarray, float, bool, bool]:
        """Step environment and return (obs, reward, terminated, truncated)."""
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, _ = step_result
            return obs, float(reward), terminated, truncated
        else:
            obs, reward, terminated, _ = step_result
            return obs, float(reward), terminated, False

    def _get_action(self, logits: torch.Tensor) -> Tuple[Any, torch.Tensor]:
        """Sample action from logits and return (env_action, log_prob)."""
        if self.is_continuous:
            # Continuous Action Space (Gaussian Policy)
            mean = torch.tanh(logits) * self.action_scale + self.action_bias
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            # For environment step, convert to numpy
            env_action = action.cpu().detach().numpy()[0]
        else:
            # Discrete Action Space (Categorical Policy)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            env_action = action.item()

        return env_action, log_prob

    def _collect_trajectory(self) -> Tuple[List[torch.Tensor], List[float]]:
        """Run one episode and collect log_probs and rewards."""
        obs = self._reset_env()

        log_probs = []
        rewards = []

        # Run up to MAX_STEPS for safety
        for _ in range(MAX_STEPS):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Forward pass
            logits = self.model(obs_tensor)

            # Get action
            env_action, log_prob = self._get_action(logits)

            # Step environment
            obs, reward, terminated, truncated = self._step_env(env_action)

            log_probs.append(log_prob)
            rewards.append(reward)

            if terminated or truncated:
                break

        return log_probs, rewards

    def _compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns_tensor = torch.tensor(returns, device=self.device)

        # Normalize returns for stability
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (
                returns_tensor.std() + 1e-9
            )

        return returns_tensor

    def _update_policy(
        self, log_probs: List[torch.Tensor], returns: torch.Tensor
    ) -> float:
        """Update policy using REINFORCE."""
        loss_list = []
        for log_prob, R in zip(log_probs, returns):
            loss_list.append(-log_prob * R)

        loss = torch.stack(loss_list).sum()

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        return loss.item()

    def train_episode(self) -> Dict[str, float]:
        """Run one episode and update policy."""
        self.model.train()

        # 1. Collect Trajectory
        log_probs, rewards = self._collect_trajectory()

        # 2. Compute Returns
        returns = self._compute_returns(rewards)

        # 3. Policy Gradient Update
        loss_value = self._update_policy(log_probs, returns)

        total_reward = sum(rewards)
        self.reward_history.append(total_reward)
        self.loss_history.append(loss_value)

        return {"reward": total_reward, "loss": loss_value, "steps": len(rewards)}

    def train_epoch(self) -> Dict[str, float]:
        """Run multiple episodes as an 'epoch'."""
        t0 = time.time()

        epoch_reward_sum = 0.0
        epoch_loss_sum = 0.0

        for _ in range(self.episodes_per_epoch):
            metrics = self.train_episode()
            epoch_reward_sum += metrics["reward"]
            epoch_loss_sum += metrics["loss"]

        avg_reward = epoch_reward_sum / self.episodes_per_epoch
        avg_loss = epoch_loss_sum / self.episodes_per_epoch
        epoch_time = time.time() - t0

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_reward,  # Raw reward as accuracy proxy
            "reward": avg_reward,
            "perplexity": 0.0,
            "time": epoch_time,
            "iteration_time": epoch_time / self.episodes_per_epoch,
        }

        if self.tracker:
            self.tracker.log_metrics(metrics)

        return metrics

    def evaluate(self, episodes: int = 5) -> float:
        """Evaluate without updating."""
        self.model.eval()
        total_rewards = []

        for _ in range(episodes):
            obs = self._reset_env()

            ep_reward = 0.0
            done = False
            steps = 0

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(obs_tensor)

                    if self.is_continuous:
                        # Deterministic policy for eval (mean)
                        assert self.action_scale is not None
                        assert self.action_bias is not None
                        action_tensor = (
                            torch.tanh(logits) * self.action_scale + self.action_bias
                        )
                        action = action_tensor.cpu().numpy()[0]
                    else:
                        action = logits.argmax(dim=-1).item()

                obs, reward, terminated, truncated = self._step_env(action)
                done = terminated or truncated

                ep_reward += reward
                steps += 1
                if steps >= MAX_STEPS:
                    break

            total_rewards.append(ep_reward)

        return float(np.mean(total_rewards))
