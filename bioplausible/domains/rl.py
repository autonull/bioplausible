"""
Reinforcement Learning Domain Tasks

Standard RL environments (CartPole, Pendulum, etc.)
"""

from typing import Optional

import torch
import torch.nn as nn

from bioplausible.domains.base import (
    DomainTask,
    DomainType,
    DomainSpec,
    TaskSplit,
    Metrics,
)


class RLTask(DomainTask):
    """Reinforcement learning domain tasks."""

    def __init__(
        self,
        name: str = "cartpole",
        env_id: str = "CartPole-v1",
        max_steps: int = 1000,
        gamma: float = 0.99,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.env_id = env_id
        self.max_steps = max_steps
        self.gamma = gamma
        self._env = None

    @property
    def domain_type(self) -> DomainType:
        return DomainType.RL

    @property
    def spec(self) -> DomainSpec:
        return DomainSpec(
            name=self.name,
            domain_type=DomainType.RL,
            description=f"RL task: {self.env_id}",
            default_metrics=["reward", "episode_length"],
            supported_tasks=["reinforcement_learning"],
            default_batch_size=1,  # Episodes
            default_lr=3e-4,
            tags=["rl", "control"],
        )

    def setup(self) -> None:
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "gymnasium required for RL tasks. Install with: pip install gymnasium"
            )

        self._env = gym.make(self.env_id)
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        self._input_dim = (
            obs_space.shape[0] if hasattr(obs_space, "shape") else obs_space.n
        )
        if hasattr(act_space, "n"):
            self._output_dim = act_space.n
        else:
            self._output_dim = act_space.shape[0]

        self._setup_done = True

    def get_dataloader(self, split: TaskSplit) -> None:
        # RL doesn't use traditional dataloaders
        return None

    def evaluate(
        self,
        model: nn.Module,
        split: TaskSplit = TaskSplit.VAL,
        max_batches: Optional[int] = None,
        n_episodes: int = 10,
    ) -> Metrics:
        if self._env is None:
            self.setup()

        model.eval()
        total_rewards = []
        total_lengths = []

        for _ in range(n_episodes):
            obs, _ = self._env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done or episode_length < self.max_steps:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action_logits = model(obs_tensor)
                    action = action_logits.argmax(1).item()

                obs, reward, terminated, truncated, _ = self._env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        import numpy as np

        return Metrics(
            loss=-np.mean(total_rewards),  # Negative reward as loss
            custom={
                "mean_reward": np.mean(total_rewards),
                "std_reward": np.std(total_rewards),
                "mean_length": np.mean(total_lengths),
            },
        )
