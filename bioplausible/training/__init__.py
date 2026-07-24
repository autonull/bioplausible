"""Trainers.

CoreTrainer (config-based supervised learning) is the canonical trainer and
lives in ``bioplausible.core.trainer``. RLTrainer is a standalone trainer for
reinforcement learning trajectories, decoupled because the RL flow has a
different shape (no fixed DataLoader; samples come from an environment).
"""

from bioplausible.core.trainer import CoreTrainer as BaseTrainer

from .rl import RLTrainer

__all__ = ["BaseTrainer", "RLTrainer"]
