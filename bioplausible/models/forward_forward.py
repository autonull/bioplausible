import torch
import torch.nn as nn
from typing import Dict, Any
from .registry import register_model

class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.03)

    def forward(self, x):
        x_dir = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.relu(torch.mm(x_dir, self.weight.T) + self.bias.unsqueeze(0))

@register_model("forward_forward")
class ForwardForwardNet(nn.Module):
    """
    Hinton's Forward-Forward (2022).
    Two forward passes (positive/negative), layer-local goodness objective.
    No backward pass. requires_backward = False.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, threshold: float = 2.0, num_layers: int = 2):
        super().__init__()
        if isinstance(input_dim, tuple):
            import math
            input_dim = math.prod(input_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.threshold = threshold
        
        self.layers = nn.ModuleList([FFLayer(input_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(FFLayer(hidden_dim, hidden_dim))
            
        self.classifier = nn.Linear(hidden_dim * num_layers, output_dim)
        self.classifier_opt = torch.optim.Adam(self.classifier.parameters(), lr=0.01)

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers=2, device="cpu", task_type="vision", **kwargs):
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)

    def predict(self, x):
        h = x
        hidden_states = []
        for layer in self.layers:
            h = layer(h)
            hidden_states.append(h)
        h_all = torch.cat(hidden_states, dim=1)
        return self.classifier(h_all)
        
    def forward(self, x):
        return self.predict(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        x_pos = x.clone()
        x_neg = x.clone()
        
        batch_size = x.shape[0]
        y_neg = torch.randint(0, self.output_dim, (batch_size,), device=x.device)
        for i in range(batch_size):
            while y_neg[i] == y[i]:
                y_neg[i] = torch.randint(0, self.output_dim, (1,)).item()
                
        x_pos[:, :self.output_dim] = 0.0
        x_neg[:, :self.output_dim] = 0.0
        x_pos[range(batch_size), y] = x.max()
        x_neg[range(batch_size), y_neg] = x.max()

        total_loss = 0.0
        h_pos, h_neg = x_pos, x_neg
        
        for layer in self.layers:
            h_pos = layer(h_pos)
            g_pos = (h_pos ** 2).mean(dim=1)
            
            h_neg = layer(h_neg)
            g_neg = (h_neg ** 2).mean(dim=1)
            
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold, 
                g_neg - self.threshold
            ]))).mean()
            
            layer.opt.zero_grad()
            loss.backward()
            layer.opt.step()
            
            total_loss += loss.item()
            
            h_pos = h_pos.detach()
            h_neg = h_neg.detach()
            
        h = x
        hidden_states = []
        with torch.no_grad():
            for layer in self.layers:
                h = layer(h)
                hidden_states.append(h)
        h_all = torch.cat(hidden_states, dim=1).detach()
        
        logits = self.classifier(h_all)
        cls_loss = nn.functional.cross_entropy(logits, y)
        
        self.classifier_opt.zero_grad()
        cls_loss.backward()
        self.classifier_opt.step()
        
        acc = (logits.argmax(1) == y).float().mean().item()
        
        return {"loss": total_loss / len(self.layers), "accuracy": acc, "cls_loss": cls_loss.item()}
