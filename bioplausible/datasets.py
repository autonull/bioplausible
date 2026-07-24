"""
Backward-compatible re-exports for the data package.

All original functions are now maintained in bioplausible.data.
This module re-exports them so existing imports continue to work.
"""

from bioplausible.data.lm import get_lm_dataset
from bioplausible.data.vision import CharDataset
from bioplausible.data.vision import create_data_loaders
from bioplausible.data.vision import get_vision_dataset

__all__ = [
    "get_vision_dataset",
    "get_lm_dataset",
    "CharDataset",
    "create_data_loaders",
]
