"""
Backward-compatible re-exports for the data package.

All original functions are now maintained in bioplausible.data.
This module re-exports them so existing imports continue to work.
"""

from bioplausible.data.lm import get_lm_dataset
from bioplausible.data.vision import (
    CharDataset,
    create_data_loaders,
    get_vision_dataset,
)

__all__ = [
    "CharDataset",
    "create_data_loaders",
    "get_lm_dataset",
    "get_vision_dataset",
]
