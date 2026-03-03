import torch
import torch.nn as nn
from typing import Dict, Any
from .registry import register_model

@register_model("pepita")
class PEPITA(nn.Module):
    """
    PEPITA: Present the Error to Perturb the Input To modulate Activity.
    Two forward passes; error-modulated input; no backward pass through network.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if isinstance(input_dim, tuple):
            import math
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.out_layer = nn.Linear(hidden_dim, output_dim)
        
        self.relu = nn.ReLU()
        self.feedback_matrix = nn.Parameter(torch.randn(input_dim, output_dim) / input_dim**0.5)
        self.lr = 0.01

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers=2, device="cpu", task_type="vision", **kwargs):
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)

    def forward(self, x, return_activations=False):
        activations = []
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
            activations.append(h)
        out = self.out_layer(h)
        if return_activations:
            return out, activations
        return out

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        y_onehot = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
        
        with torch.no_grad():
            out_s, act_s = self.forward(x, return_activations=True)
            error = out_s - y_onehot
            
            x_mod = x + torch.mm(error, self.feedback_matrix.T)
            
            out_m, act_m = self.forward(x_mod, return_activations=True)
            
            inputs = [x] + act_s[:-1]
            for layer, a_s, a_m, inp in zip(self.layers, act_s, act_m, inputs):
                delta_a = a_m - a_s
                layer.weight.data -= self.lr * torch.mm(delta_a.T, inp) / x.shape[0]
                if layer.bias is not None:
                    layer.bias.data -= self.lr * delta_a.mean(0)
            
            self.out_layer.weight.data -= self.lr * torch.mm(error.T, act_s[-1]) / x.shape[0]
            if self.out_layer.bias is not None:
                self.out_layer.bias.data -= self.lr * error.mean(0)
                
        loss = (error**2).sum(1).mean().item()
        acc = (out_s.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}
