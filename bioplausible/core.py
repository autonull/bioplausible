"""
EqProp-Torch Core Trainer (Legacy)

This module is deprecated. Use `bioplausible.training.supervised.SupervisedTrainer` instead.  # noqa: E501
"""

from bioplausible.training.supervised import (
    SupervisedTrainer as EqPropTrainer,
)

__all__ = ["EqPropTrainer"]
