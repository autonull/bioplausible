from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
from bioplausible.models.registry import register_model

# Layer Factory (Internal)
def create_layer(config: Dict[str, Any], in_features: int) -> Tuple[nn.Module, int]:
    """
    Create a layer from config.
    Returns (layer_module, out_features).
    """
    layer_type = config.get("type", "linear").lower()
    out_features = config.get("size", 64)

    layer = None

    if layer_type == "linear":
        layer = nn.Linear(in_features, out_features)

    elif layer_type == "conv2d":
        # Simplified Conv2d builder
        # Assume square input for now or handle flattening later?
        # Conv2d takes in_channels, out_channels
        # If previous layer was linear, we might need to reshape?
        # Or just treat 'in_features' as channels?
        # Usually Conv -> Linear. Linear -> Conv is rare without explicit reshape.

        # Let's assume this stack is mostly MLP-like, but maybe 1st layer is Conv.
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 1)

        # If in_features is large (e.g. 784), it's flattened image.
        # We need to reshape in forward.
        # Here we just create the layer.
        # But Conv needs in_channels.
        # Heuristic: if in_features > 100, assume 1 channel (grayscale) or 3 (RGB).
        # Or user specifies input_shape?

        # For simplicity in this demo:
        # If in_features is small (<10), assume channels.
        # If large, assume flattened 1-channel image (sqrt(in_features)).

        in_channels = in_features if in_features < 10 else 1

        layer = nn.Conv2d(in_channels, out_features, kernel_size, stride, padding)

        # Output size is tricky without input spatial dim.
        # We'll return 'out_features' as 'channels'.
        # Subsequent layers must handle this.

    elif layer_type == "equitile":
        # Use EquiTile logic (simplified)
        # We might need to import EquiTile or create a mock if avoiding circular deps?
        # Or just use Linear with special initialization/handling?
        # The prompt says "EquiTile is just one of many".
        # Let's try to import the real EquiTile layer if possible, or just use Linear with a tag.

        # For now, let's use Linear but tag it, or use the actual EquiTile class if modular.
        # The actual EquiTile is a full model.
        # But we can make a 'EquiTileLayer'.

        # Let's just use Linear for now but name it EquiTile for the UI to treat differently?
        # Or better: Create a custom layer class here.

        layer = nn.Linear(in_features, out_features)
        layer.is_equitile = True
        # Initialize with tile importance
        layer.tile_importance = nn.Parameter(torch.zeros(out_features)) # 1 importance per unit?
        # Actually EquiTile has tiles.
        # If size=64, tiles=16 -> 4 neurons/tile.
        # Let's keep it simple: 1 neuron = 1 tile for this custom builder.

    elif layer_type == "activation":
        act_name = config.get("act", "relu").lower()
        if act_name == "relu": layer = nn.ReLU()
        elif act_name == "tanh": layer = nn.Tanh()
        elif act_name == "sigmoid": layer = nn.Sigmoid()
        elif act_name == "gelu": layer = nn.GELU()
        else: layer = nn.ReLU()

        return layer, in_features # Size doesn't change

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return layer, out_features


@register_model("custom_stacked_model")
class CustomStackedModel(nn.Module):
    """
    A model built from a user-defined stack of layers.
    Allows mixing Linear, Conv, EquiTile-like layers.
    """

    def __init__(self, input_dim: int, output_dim: int, layers_config: List[Dict[str, Any]]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config

        self.layers = nn.ModuleList()
        self.layer_sizes = [] # For visualization

        current_dim = input_dim

        # Build layers
        for cfg in layers_config:
            # If activation, add it
            if cfg.get("type") == "activation":
                layer, _ = create_layer(cfg, current_dim)
                self.layers.append(layer)
                continue

            # If standard layer
            layer, out_dim = create_layer(cfg, current_dim)
            self.layers.append(layer)
            self.layer_sizes.append(out_dim)
            current_dim = out_dim

            # Add implicit activation if not specified?
            # Let's assume explicit activation layers in config, or default ReLU.
            # But user might want to mix.
            # Let's add ReLU by default if not specified in config to NOT add it.
            if cfg.get("activation", True):
                self.layers.append(nn.ReLU())

        # Final output layer
        # If the last layer size != output_dim, add a projection
        if current_dim != output_dim:
            self.output_layer = nn.Linear(current_dim, output_dim)
            self.layers.append(self.output_layer)
            self.layer_sizes.append(output_dim)

        # Store for generic wrapper
        # The generic wrapper looks for 'layers' or named modules.
        # self.layers is ModuleList, so it works.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten input if first layer is Linear but input is image
        if x.dim() > 2 and isinstance(self.layers[0], nn.Linear):
            x = x.view(x.size(0), -1)

        # Reshape input if first layer is Conv but input is flat
        if x.dim() == 2 and isinstance(self.layers[0], nn.Conv2d):
            # Assume square image
            side = int(x.size(1)**0.5)
            x = x.view(x.size(0), 1, side, side) # 1 channel assumption

        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and out.dim() > 2:
                # Flatten before Linear
                out = out.view(out.size(0), -1)

            out = layer(out)

        return out

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs):
        """
        Factory build method.
        Expects 'layers_config' in kwargs.
        If not present, builds a default stack based on num_layers/hidden_dim.
        """
        layers_config = kwargs.get("layers_config")

        if not layers_config:
            # Create default config
            layers_config = []
            for _ in range(num_layers):
                layers_config.append({"type": "linear", "size": hidden_dim})

        return cls(input_dim, output_dim, layers_config).to(device)
