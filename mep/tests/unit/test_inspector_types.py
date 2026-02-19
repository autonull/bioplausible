"""
Tests for ModelInspector type identification.
"""

import torch.nn as nn
from mep.optimizers.inspector import ModelInspector

def test_inspector_types():
    inspector = ModelInspector()

    assert inspector._get_module_type(nn.Linear(1, 1)) == "layer"
    assert inspector._get_module_type(nn.Flatten()) == "flatten"
    assert inspector._get_module_type(nn.Dropout()) == "dropout"
    assert inspector._get_module_type(nn.ReLU()) == "act"
    assert inspector._get_module_type(nn.Conv2d(1, 1, 1)) == "layer"
