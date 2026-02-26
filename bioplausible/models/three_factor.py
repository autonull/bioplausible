import torch
import torch.nn as nn
from typing import Dict, Any
from .registry import register_model

@register_model("three_factor_hebbian")
class ThreeFactorHebbian(nn.Module):
    """
    Three-Factor Learning: Δw = η · M · pre · post
    where M is a neuromodulatory signal (dopamine-like global reward).
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim, bias=False)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        self.out_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        self.relu = nn.ReLU()
        self.lr = 0.005
        
    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers=2, device="cpu", task_type="vision", **kwargs):
        model = cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        model = model.to(device)
        return model

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
        return self.out_layer(h)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        hs = [x]
        h = x
        for layer in self.layers:
            h = self.relu(layer(h))
            hs.append(h)
        out = self.out_layer(h)
        
        preds = out.argmax(1)
        correct = (preds == y).float()
        M = correct * 2 - 1
        M = M.to(x.device)
        
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                pre = hs[i]
                post = hs[i+1]
                post_mod = post * M.unsqueeze(1)
                layer.weight.data += self.lr * torch.mm(post_mod.T, pre) / x.shape[0]
                
            y_onehot = torch.zeros_like(out, device=out.device)
            y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
            error = y_onehot - out
            self.out_layer.weight.data += self.lr * torch.mm(error.T, hs[-1]) / x.shape[0]
            
        loss = nn.functional.cross_entropy(out, y).item()
        return {"loss": loss, "accuracy": correct.mean().item()}
