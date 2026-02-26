import torch
import torch.nn as nn
from typing import Dict, Any
from .eqprop_base import EqPropModel
from .registry import register_model

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

@register_model("graph_eqprop")
class GraphEqProp(EqPropModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_steps: int = 30):
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, max_steps=max_steps)

    def _build_layers(self):
        if GCNConv is None:
            self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
            self.conv = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.W_out = nn.Linear(self.hidden_dim, self.output_dim)
            return
            
        self.W_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.conv = GCNConv(self.hidden_dim, self.hidden_dim)
        self.W_out = nn.Linear(self.hidden_dim, self.output_dim)
        
    def _initialize_hidden_state(self, x: Any) -> torch.Tensor:
        if hasattr(x, "x"):
            num_nodes = x.x.size(0)
            return torch.zeros((num_nodes, self.hidden_dim), device=x.x.device, dtype=x.x.dtype)
        else:
            return torch.zeros((x.size(0), self.hidden_dim), device=x.device, dtype=x.dtype)
        
    def _transform_input(self, x: Any) -> Any:
        if hasattr(x, "x"):
            u = self.W_in(x.x)
            return (u, x.edge_index)
        else:
            return self.W_in(x)

    def forward_step(self, h: torch.Tensor, x_transformed: Any) -> torch.Tensor:
        if isinstance(x_transformed, tuple):
            u, edge_index = x_transformed
            if GCNConv is not None:
                return torch.tanh(u + self.conv(h, edge_index))
        return torch.tanh(x_transformed + self.conv(h))

    def _output_projection(self, h: torch.Tensor) -> torch.Tensor:
        return self.W_out(h)

    def train_step(self, x: Any, y: torch.Tensor) -> Dict[str, float]:
        if not hasattr(x, "train_mask"):
            return super().train_step(x, y)
            
        with torch.no_grad():
            h_star, _ = self.solve_equilibrium(x)
        logits = self._output_projection(h_star)
        
        mask = x.train_mask
        if not mask.any():
            mask = torch.ones_like(y, dtype=torch.bool)
            
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits[mask], y[mask])
        
        beta = 0.5
        v = torch.zeros_like(logits)
        logits_masked = logits[mask].clone().detach().requires_grad_(True)
        L_masked = loss_fn(logits_masked, y[mask])
        grad_out = torch.autograd.grad(L_masked, logits_masked)[0]
        v[mask] = grad_out
        
        x_trans = self._transform_input(x)
        h_nudged = h_star.clone()
        with torch.no_grad():
            for _ in range(5):
                h_nudged = self.forward_step(h_nudged, x_trans) - beta * torch.mm(v, self.W_out.weight)
                
        self.zero_grad()
        
        h_star.requires_grad = False
        if isinstance(x_trans, tuple):
            u, edge_index = x_trans
            pre_act_star = u + self.conv(h_star, edge_index)
        else:
            pre_act_star = x_trans + self.conv(h_star)
        E_free = torch.sum(0.5 * h_star**2 - h_star * pre_act_star)
        E_free.backward()
        free_grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.parameters()]
        
        self.zero_grad()
        h_nudged.requires_grad = False
        if isinstance(x_trans, tuple):
            u, edge_index = x_trans
            pre_act_nudged = u + self.conv(h_nudged, edge_index)
        else:
            pre_act_nudged = x_trans + self.conv(h_nudged)
        E_nudged = torch.sum(0.5 * h_nudged**2 - h_nudged * pre_act_nudged)
        E_nudged.backward()
        
        lr = 0.001
        with torch.no_grad():
            for p, gf in zip(self.parameters(), free_grads):
                gn = p.grad
                if gn is not None:
                    p.data -= lr * (gf - gn) / beta
                    
            self.W_out.weight.data -= lr * torch.mm(v[mask].T, h_star[mask])
            self.W_out.bias.data -= lr * v[mask].sum(0)
        
        acc = (logits[mask].argmax(1) == y[mask]).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc}

    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers=2, device="cpu", task_type="graph", **kwargs):
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
