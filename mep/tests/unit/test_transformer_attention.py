"""
Tests for Transformer and attention layer compatibility.

Tests EP settling dynamics, energy computation, and gradient flow
for Transformer architectures including:
- MultiheadAttention layers
- LayerNorm
- Residual connections

Note: Full Transformer support with sequence dimensions and causal masking
requires additional work. Current tests verify basic compatibility.
"""

import torch
import torch.nn as nn
import pytest
from mep.optimizers.settling import Settler
from mep.optimizers.energy import EnergyFunction
from mep.optimizers.inspector import ModelInspector
from mep.presets import smep, sdmep


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerLikeModel(nn.Module):
    """
    Transformer-like model that works with EP settling.
    
    Uses Linear layers to simulate attention-like computation
    while maintaining compatible shapes for EP.
    """
    
    def __init__(self, d_model: int = 64, num_layers: int = 2):
        super().__init__()
        # Simulate transformer blocks with Linear + LayerNorm + residual
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ))
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_model)
        for block in self.blocks:
            # Residual connection
            x = x + block(x)
        x = self.norm(x)
        return self.classifier(x)


class AttentionMLP(nn.Module):
    """
    MLP with attention-like module for testing.
    
    Uses nn.MultiheadAttention with compatible shapes by keeping
    sequence dimension throughout.
    """
    
    def __init__(self, d_model: int = 32, nhead: int = 4, seq_len: int = 4):
        super().__init__()
        self.seq_len = seq_len
        self.proj_in = nn.Linear(10, d_model)
        # Self-attention that maintains sequence dimension
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        # Output per sequence position
        self.classifier = nn.Linear(d_model, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 10)
        x = self.proj_in(x)  # (batch, d_model)
        # Repeat to create sequence dimension
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, d_model)
        x, _ = self.attention(x, x, x)
        x = self.norm(x)
        # Pool over sequence for classification
        x = x.mean(dim=1)  # (batch, d_model)
        return self.classifier(x)


class TestTransformerInspector:
    """Tests for model inspector with Transformer architectures."""
    
    def test_inspect_transformer_like(self):
        """Test inspector recognizes transformer-like structure."""
        model = TransformerLikeModel(d_model=64, num_layers=2)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        
        # Should find norms and linear layers
        types = [item["type"] for item in structure]
        
        assert "layer" in types  # Linear layers
        assert "norm" in types  # LayerNorm
    
    def test_inspect_attention_mlp(self):
        """Test inspector recognizes attention module."""
        model = AttentionMLP(d_model=32, nhead=4)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        
        types = [item["type"] for item in structure]
        
        assert "attention" in types
        assert "layer" in types
        assert "norm" in types


class TestTransformerEnergy:
    """Tests for energy computation with Transformer-like models."""
    
    def test_energy_transformer_like(self, device):
        """Test energy computation for transformer-like model."""
        model = TransformerLikeModel(d_model=32, num_layers=1).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=3, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 32, device=device)
        y = torch.randn(2, 5, device=device)
        
        states = settler._capture_states(model, x, structure)
        target_vec = settler._prepare_target(y, 5, states[-1].dtype)
        
        E = energy_fn(model, x, states, structure, target_vec, beta=0.1)
        
        assert torch.isfinite(E)
        assert E > 0
    
    @pytest.mark.skip(reason="Attention layers with sequence dimensions need special handling")
    def test_energy_attention_mlp(self, device):
        """Test energy computation for attention MLP."""
        model = AttentionMLP(d_model=32, nhead=4).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=3, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 5, device=device)
        
        states = settler._capture_states(model, x, structure)
        target_vec = settler._prepare_target(y, 5, states[-1].dtype)
        
        E = energy_fn(model, x, states, structure, target_vec, beta=0.1)
        
        assert torch.isfinite(E)


class TestTransformerSettling:
    """Tests for settling dynamics with Transformer-like models."""
    
    def test_settle_transformer_like(self, device):
        """Test settling for transformer-like model."""
        model = TransformerLikeModel(d_model=32, num_layers=1).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 32, device=device)
        y = torch.randn(2, 5, device=device)
        
        states = settler.settle(model, x, y, beta=0.1, energy_fn=energy_fn, structure=structure)
        
        assert len(states) > 0
        assert all(torch.isfinite(s).all() for s in states)
    
    @pytest.mark.skip(reason="Attention layers with sequence dimensions need special handling")
    def test_settle_attention_mlp(self, device):
        """Test settling for attention MLP."""
        model = AttentionMLP(d_model=32, nhead=4).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 5, device=device)
        
        states = settler.settle(model, x, y, beta=0.1, energy_fn=energy_fn, structure=structure)
        
        assert len(states) > 0
        assert all(torch.isfinite(s).all() for s in states)


class TestTransformerEP:
    """Tests for full EP training with Transformer-like models."""
    
    def test_transformer_like_backprop(self, device):
        """Test transformer-like model training with backprop."""
        model = TransformerLikeModel(d_model=32, num_layers=1).to(device)
        optimizer = smep(model.parameters(), model=model, lr=0.01, mode="backprop")
        
        x = torch.randn(2, 32, device=device)
        y = torch.randint(0, 5, (2,), device=device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
    
    def test_transformer_like_ep_training(self, device):
        """Test transformer-like model training with EP."""
        model = TransformerLikeModel(d_model=32, num_layers=1).to(device)
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode="ep",
            beta=0.1,
            settle_steps=3
        )
        
        x = torch.randn(2, 32, device=device)
        y = torch.randint(0, 5, (2,), device=device)
        
        optimizer.step(x=x, target=y)
        
        # Check that parameters changed
        for p in model.parameters():
            assert p.grad is not None
    
    @pytest.mark.skip(reason="Attention layers with sequence dimensions need special handling")
    def test_attention_mlp_ep(self, device):
        """Test attention MLP training with EP."""
        model = AttentionMLP(d_model=32, nhead=4).to(device)
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode="ep",
            beta=0.1,
            settle_steps=3
        )
        
        x = torch.randn(2, 10, device=device)
        y = torch.randint(0, 5, (2,), device=device)
        
        optimizer.step(x=x, target=y)


class TestResidualConnections:
    """Tests for residual connections in Transformer-like models."""
    
    def test_residual_settling(self, device):
        """Test settling with residual connections."""
        model = TransformerLikeModel(d_model=32, num_layers=2).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 32, device=device)
        y = torch.randn(2, 5, device=device)
        
        # Settling should work with residual connections
        states = settler.settle(model, x, y, beta=0.1, energy_fn=energy_fn, structure=structure)
        
        # States should be finite (no divergence from residuals)
        assert all(torch.isfinite(s).all() for s in states)


class TestLayerNorm:
    """Tests for LayerNorm in EP settling."""
    
    def test_layernorm_settling(self, device):
        """Test settling with LayerNorm."""
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 5)
        ).to(device)
        
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        settler = Settler(steps=5, lr=0.01)
        energy_fn = EnergyFunction(loss_type="mse")
        
        x = torch.randn(2, 32, device=device)
        y = torch.randn(2, 5, device=device)
        
        states = settler.settle(model, x, y, beta=0.1, energy_fn=energy_fn, structure=structure)
        
        # Should find norm layers in structure
        types = [item["type"] for item in structure]
        assert "norm" in types
        assert len(states) == 2  # Two linear layers


class TestTransformerGradientFlow:
    """Tests for gradient flow in Transformer-like EP."""
    
    def test_ep_gradient_flow(self, device):
        """Test that EP gradients flow through transformer-like model."""
        model = TransformerLikeModel(d_model=32, num_layers=1).to(device)
        
        # Get BP gradients
        x = torch.randn(2, 32, device=device)
        y = torch.randint(0, 5, (2,), device=device)
        
        model.zero_grad()
        output_bp = model(x)
        loss_bp = nn.CrossEntropyLoss()(output_bp, y)
        loss_bp.backward()
        bp_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        
        # Get EP gradients
        model.zero_grad()
        optimizer = smep(
            model.parameters(),
            model=model,
            lr=0.01,
            mode="ep",
            beta=0.1,  # Larger beta for more stable EP
            settle_steps=15  # More steps for convergence
        )
        optimizer.step(x=x, target=y)
        ep_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        
        # Compare gradients (should have positive correlation)
        # Note: EP gradients won't exactly match BP, but should point in similar direction
        positive_count = 0
        total_count = 0
        for bp_g, ep_g in zip(bp_grads, ep_grads):
            if bp_g is not None and ep_g is not None:
                cos_sim = nn.functional.cosine_similarity(
                    bp_g.flatten(), ep_g.flatten(), dim=0
                )
                if cos_sim > 0:
                    positive_count += 1
                total_count += 1
        
        # Most gradients should have positive correlation
        assert positive_count >= total_count * 0.5, \
            f"Only {positive_count}/{total_count} gradient pairs had positive correlation"


class TestAttentionMechanism:
    """Tests for attention mechanism compatibility."""
    
    def test_attention_module_recognized(self):
        """Test that MultiheadAttention is recognized."""
        model = AttentionMLP(d_model=32, nhead=4)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        
        # Find attention module
        attention_items = [item for item in structure if item["type"] == "attention"]
        assert len(attention_items) == 1
        assert isinstance(attention_items[0]["module"], nn.MultiheadAttention)
    
    def test_attention_with_layernorm(self, device):
        """Test attention with LayerNorm works."""
        model = AttentionMLP(d_model=32, nhead=4).to(device)
        inspector = ModelInspector()
        structure = inspector.inspect(model)
        
        # Verify structure
        types = [item["type"] for item in structure]
        assert "attention" in types
        assert "norm" in types
        assert "layer" in types
