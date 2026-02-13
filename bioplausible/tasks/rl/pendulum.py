"""
Pendulum-v1 Task (Standard RL Benchmark).
"""

from typing import Tuple

import gymnasium as gym
import torch
import torch.nn as nn

from bioplausible.hyperopt.tasks import BaseTask


class PendulumTask(BaseTask):
    """
    Pendulum-v1: Continuous control task.
    Goal: Swing up and balance the pendulum.
    Input: [cos(theta), sin(theta), theta_dot] (3 dim)
    Output: Torque [-2, 2] (1 dim continuous)
    """

    def __init__(self, name="pendulum", device="cpu", quick_mode=False):
        super().__init__(name, device, quick_mode)
        self.env_name = "Pendulum-v1"
        self.env = None

    @property
    def task_type(self) -> str:
        return "rl"

    def setup(self):
        try:
            self.env = gym.make(self.env_name)
            self._input_dim = self.env.observation_space.shape[0]  # 3
            self._output_dim = self.env.action_space.shape[0]  # 1
        except Exception as e:
            print(f"Failed to load {self.env_name}: {e}")
            raise

    def get_batch(
        self, split="train", batch_size=32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Use RLTrainer")

    def create_trainer(self, model: nn.Module, **kwargs):
        from bioplausible.training.rl import RLTrainer

        return RLTrainer(model, self.env_name, device=self.device, **kwargs)
