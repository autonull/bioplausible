"""
Model structure inspection utilities.

Extracts layer structure from PyTorch models for EP state tracking.
"""

import torch.nn as nn
from typing import List, Dict, Any, Optional


class ModelInspector:
    """
    Extracts sequence of layers and activations from a model.
    
    Caches structure to avoid repeated introspection.
    Uses recursive inspection to handle nested modules correctly,
    treating complex modules (like MultiheadAttention) as atomic units
    if they are explicitly handled, preventing duplicate structure entries
    for their submodules.
    """
    
    def __init__(self) -> None:
        self._cache: Dict[int, List[Dict[str, Any]]] = {}
    
    def inspect(self, model: nn.Module) -> List[Dict[str, Any]]:
        """
        Extract model structure.
        
        Args:
            model: Neural network to inspect.
        
        Returns:
            List of structure items with 'type' and 'module' keys.
        """
        model_id = id(model)
        if model_id in self._cache:
            return self._cache[model_id]
        
        structure: List[Dict[str, Any]] = []
        self._inspect_recursive(model, structure)
        
        self._cache[model_id] = structure
        return structure

    def _inspect_recursive(self, module: nn.Module, structure: List[Dict[str, Any]]) -> None:
        """
        Recursively inspect module structure.

        If a module is recognized as a specific type (layer, attention, etc.),
        it is added to structure and its children are NOT inspected.
        Otherwise, we recurse into children.
        """
        # Check if the module itself matches any known type
        item_type = self._get_module_type(module)

        if item_type:
            # Atomic module found (e.g. Linear, Conv, Attention, Act)
            structure.append({"type": item_type, "module": module})
            # Do not recurse into children of atomic modules
            # (e.g. don't find Linear inside MultiheadAttention)
            return

        # If not atomic, recurse into children
        # This handles Sequential, ModuleList, and custom containers
        for child in module.children():
            self._inspect_recursive(child, structure)

    def _get_module_type(self, m: nn.Module) -> Optional[str]:
        """Determine module type string, or None if not atomic."""
        # Convolutional and linear layers
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return "layer"

        # Transformer attention
        elif isinstance(m, nn.MultiheadAttention):
            return "attention"

        # Normalization layers
        elif isinstance(m, (
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d
        )):
            return "norm"

        # Activations
        elif isinstance(m, (
            nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.Softmax,
            nn.GELU, nn.SiLU, nn.ELU,
            nn.CELU, nn.GLU, nn.Hardswish, nn.Mish
        )):
            return "act"

        elif isinstance(m, nn.Flatten):
            return "flatten"

        elif isinstance(m, nn.Dropout):
            return "dropout"

        # Pooling layers
        elif isinstance(m, (
            nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
            nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d
        )):
            return "pool"

        return None
    
    def clear_cache(self) -> None:
        """Clear the structure cache."""
        self._cache.clear()
    
    def get_layers(
        self,
        structure: List[Dict[str, Any]]
    ) -> List[nn.Module]:
        """Extract only layer modules from structure."""
        return [item["module"] for item in structure if item["type"] == "layer"]
