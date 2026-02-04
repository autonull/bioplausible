import time
from collections import deque
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from bioplausible.training.base import BaseTrainer
from bioplausible.tracking import ExperimentTracker


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
        **kwargs,
    ):
        super().__init__(model, device)
        self.model = self.model.to(device)
        self.gamma = gamma
        self.episodes_per_epoch = episodes_per_epoch
        self.tracker = tracker

        # Initialize Environment
        self.env = gym.make(env_name)

        # Seeding (gymnasium vs gym API differences)
        try:
            self.env.reset(seed=seed)
        except TypeError:
            self.env.seed(seed)
            self.env.reset()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # History
        self.reward_history = []
        self.loss_history = []

    def train_episode(self) -> Dict[str, float]:
        """Run one episode and update policy."""
        self.model.train()

        obs, _ = (
            self.env.reset() if hasattr(self.env, "reset") else (self.env.reset(), {})
        )
        if isinstance(obs, tuple):  # Handle some gym versions
            obs = obs[0]

        log_probs = []
        rewards = []

        terminated = False
        truncated = False

        # 1. Collect Trajectory
        while not (terminated or truncated):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Forward pass
            # EqPropModel will use equilibrium iterations here if configured
            logits = self.model(obs_tensor)
            probs = torch.softmax(logits, dim=-1)

            # Sample action
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            step_result = self.env.step(action.item())

            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
            else:
                obs, reward, terminated, _ = step_result
                truncated = False

            log_probs.append(log_prob)
            rewards.append(reward)

            if len(rewards) > 1000:  # Safety break
                truncated = True

        # 2. Compute Returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device)
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 3. Policy Gradient Update
        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)

        loss = torch.stack(loss).sum()

        self.optimizer.zero_grad()
        loss.backward()  # This triggers BPTT or Equilibrium Backward depending on model config

        # Gradient clipping usually helps RL
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        total_reward = sum(rewards)
        self.reward_history.append(total_reward)
        self.loss_history.append(loss.item())

        return {"reward": total_reward, "loss": loss.item(), "steps": len(rewards)}

    def train_epoch(self) -> Dict[str, float]:
        """Run multiple episodes as an 'epoch'."""
        t0 = time.time()

        epoch_reward_sum = 0
        epoch_loss_sum = 0

        for _ in range(self.episodes_per_epoch):
            metrics = self.train_episode()
            epoch_reward_sum += metrics["reward"]
            epoch_loss_sum += metrics["loss"]

        avg_reward = epoch_reward_sum / self.episodes_per_epoch
        avg_loss = epoch_loss_sum / self.episodes_per_epoch
        epoch_time = time.time() - t0

        # Normalize reward to "accuracy" range [0, 1] for visualization if possible,
        # otherwise keep as is but aware it's raw reward.
        # For CartPole, max reward is 500 (v1) or 200 (v0).
        # We'll just pass raw reward as "accuracy" but we should rename the field in future.
        # For now, to avoid 6000% accuracy in logs, we divide by expected max if known,
        # or just cap it / leave it.
        # Better yet, let's log reward explicitly.

        metrics = {
            "loss": avg_loss,
            "accuracy": avg_reward / 500.0,  # Normalize for CartPole-v1
            "reward": avg_reward,
            "perplexity": 0.0,
            "time": epoch_time,
            "iteration_time": epoch_time / self.episodes_per_epoch,
        }

        if self.tracker:
            self.tracker.log_metrics(metrics)

        return metrics

    def evaluate(self, episodes=5) -> float:
        """Evaluate without updating."""
        self.model.eval()
        total_rewards = []

        for _ in range(episodes):
            obs, _ = (
                self.env.reset()
                if hasattr(self.env, "reset")
                else (self.env.reset(), {})
            )
            if isinstance(obs, tuple):
                obs = obs[0]

            ep_reward = 0
            done = False
            steps = 0

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(obs_tensor)
                    action = logits.argmax(dim=-1).item()

                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, _ = step_result
                    done = terminated or truncated
                else:
                    obs, reward, terminated, _ = step_result
                    done = terminated

                ep_reward += reward
                steps += 1
                if steps > 1000:
                    break

            total_rewards.append(ep_reward)

        return np.mean(total_rewards)
