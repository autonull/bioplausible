import torch
import torch.nn as nn
from typing import Dict, Any
from .registry import register_model

try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNN = True
except ImportError:
    HAS_SNN = False

@register_model("spiking_stdp")
class SpikingSTDP(nn.Module):
    """
    Leaky Integrate-and-Fire neurons with Spike-Timing-Dependent Plasticity.
    Uses snnTorch for LIF dynamics; custom STDP learning rule overlaid.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_steps: int = 10):
        super().__init__()
        self.num_steps = num_steps
        if not HAS_SNN:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            return
            
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=False)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)
        self.lr = 0.001
        
    @classmethod
    def build(cls, spec, input_dim, output_dim, hidden_dim, num_layers=2, device="cpu", task_type="vision", **kwargs):
        return cls(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)

    def forward(self, x):
        if not HAS_SNN:
            return self.fc2(torch.relu(self.fc1(x)))
            
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        for step in range(self.num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            
        return torch.stack(spk2_rec, dim=0).sum(0)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        if not HAS_SNN:
            return {"loss": 0.0, "accuracy": 0.0}
            
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        pre_trace = torch.zeros_like(x)
        post_trace1 = torch.zeros(x.shape[0], self.fc1.out_features, device=x.device)
        
        spk2_rec = []
        
        with torch.no_grad():
            for step in range(self.num_steps):
                cur1 = self.fc1(x)
                spk1, mem1 = self.lif1(cur1, mem1)
                
                pre_trace = 0.9 * pre_trace + x
                post_trace1 = 0.9 * post_trace1 + spk1
                dw1 = self.lr * (torch.mm(spk1.T, pre_trace) - torch.mm(post_trace1.T, x))
                self.fc1.weight.data += dw1 / x.shape[0]
                
                cur2 = self.fc2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                spk2_rec.append(spk2)
                
            out = torch.stack(spk2_rec, dim=0).sum(0)
            
            y_onehot = torch.zeros_like(out)
            y_onehot.scatter_(1, y.unsqueeze(1), self.num_steps * 0.8)
            error = y_onehot - out
            self.fc2.weight.data += self.lr * torch.mm(error.T, post_trace1) / x.shape[0]
            
        loss = (error**2).mean().item()
        acc = (out.argmax(1) == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}
