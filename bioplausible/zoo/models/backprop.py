"""
Combined Backprop Models
=========================

Aggregates all standard backprop-based models into a single module for the model zoo.
"""

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from bioplausible.core.registry import register_model

# ============================================================================
# backprop_transformer_lm.py - BackpropTransformerLM
# ============================================================================


class BackpropCausalAttention(nn.Module):
    """Standard causal self-attention (no equilibrium settling)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, causal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = h.shape

        Q = (
            self
            .W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self
            .W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self
            .W_v(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if causal_mask is not None:
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        return self.W_o(
            out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        )


class BackpropTransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attention = BackpropCausalAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, causal_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


@register_model("backprop_transformer_lm")
class BackpropTransformerLM(nn.Module):
    """
    Standard Causal Transformer LM (Backprop baseline).

    Matches CausalTransformerEqProp architecture for fair comparison.
    Key difference: No equilibrium settling - single forward pass.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        ffn_mult: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BackpropTransformerBlock(hidden_dim, num_heads, ffn_mult, dropout)
            for _ in range(num_layers)
        ])

        self.norm_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        self.register_buffer("causal_mask", None)
        self._create_causal_mask(max_seq_len)

        self._init_weights()

    def _create_causal_mask(self, seq_len: int):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = x.shape

        if x.dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
            x = x.long()

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)

        causal_mask = (
            self.causal_mask[:seq_len, :seq_len]
            if self.causal_mask is not None
            else None
        )

        for block in self.blocks:
            h = block(h, causal_mask)

        h = self.norm_f(h)
        logits = self.lm_head(h)

        return logits

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0
    ):
        self.eval()
        generated = prompt.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(generated)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

                if generated.size(1) >= self.max_seq_len:
                    break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        return cls(
            vocab_size=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            max_seq_len=256,
        ).to(device)


def create_scaled_model(
    base_hidden: int,
    base_layers: int,
    scale: float = 1.0,
    vocab_size: int = 65,
    model_type: str = "backprop",
) -> nn.Module:
    scaled_hidden = int(base_hidden * math.sqrt(scale))
    scaled_hidden = max(32, (scaled_hidden // 4) * 4)

    if model_type == "backprop":
        return BackpropTransformerLM(
            vocab_size=vocab_size,
            hidden_dim=scaled_hidden,
            num_layers=base_layers,
            num_heads=4,
            max_seq_len=256,
        )
    elif model_type == "eqprop":
        from ..eqprop import CausalTransformerEqProp

        return CausalTransformerEqProp(
            vocab_size=vocab_size,
            hidden_dim=scaled_hidden,
            num_layers=base_layers,
            num_heads=4,
            max_seq_len=256,
            eq_steps=15,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ============================================================================
# custom_stack.py - CustomStackedModel
# ============================================================================


def create_layer(config: dict[str, Any], in_features: int) -> tuple[nn.Module, int]:
    layer_type = config.get("type", "linear").lower()
    out_features = config.get("size", 64)

    layer = None

    if isinstance(in_features, tuple):
        in_features = math.prod(in_features)

    if layer_type == "linear":
        layer = nn.Linear(in_features, out_features)

    elif layer_type == "conv2d":
        kernel_size = config.get("kernel_size", 3)
        stride = config.get("stride", 1)
        padding = config.get("padding", 1)

        in_channels = in_features if in_features < 10 else 1

        layer = nn.Conv2d(in_channels, out_features, kernel_size, stride, padding)

    elif layer_type == "equitile":
        layer = nn.Linear(in_features, out_features)
        layer.is_equitile = True
        layer.tile_importance = nn.Parameter(torch.zeros(out_features))

    elif layer_type == "activation":
        act_name = config.get("act", "relu").lower()
        if act_name == "relu":
            layer = nn.ReLU()
        elif act_name == "tanh":
            layer = nn.Tanh()
        elif act_name == "sigmoid":
            layer = nn.Sigmoid()
        elif act_name == "gelu":
            layer = nn.GELU()
        else:
            layer = nn.ReLU()

        return layer, in_features

    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return layer, out_features


@register_model("custom_stacked_model")
class CustomStackedModel(nn.Module):
    """
    A model built from a user-defined stack of layers.
    Allows mixing Linear, Conv, EquiTile-like layers.
    """

    def __init__(
        self, input_dim: int, output_dim: int, layers_config: list[dict[str, Any]]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers_config = layers_config

        self.layers = nn.ModuleList()
        self.layer_sizes = []

        current_dim = input_dim

        for cfg in layers_config:
            if cfg.get("type") == "activation":
                layer, _ = create_layer(cfg, current_dim)
                self.layers.append(layer)
                continue

            layer, out_dim = create_layer(cfg, current_dim)
            self.layers.append(layer)
            self.layer_sizes.append(out_dim)
            current_dim = out_dim

            if cfg.get("activation", True):
                self.layers.append(nn.ReLU())

        if current_dim != output_dim:
            self.output_layer = nn.Linear(current_dim, output_dim)
            self.layers.append(self.output_layer)
            self.layer_sizes.append(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2 and isinstance(self.layers[0], nn.Linear):
            x = x.view(x.size(0), -1)

        if x.dim() == 2 and isinstance(self.layers[0], nn.Conv2d):
            side = int(x.size(1) ** 0.5)
            x = x.view(x.size(0), 1, side, side)

        out = x
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and out.dim() > 2:
                out = out.view(out.size(0), -1)

            out = layer(out)

        return out

    @classmethod
    def build(
        cls,
        spec,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        device,
        task_type,
        **kwargs,
    ):
        layers_config = kwargs.get("layers_config")

        if not layers_config:
            layers_config = []
            for _ in range(num_layers):
                layers_config.append({"type": "linear", "size": hidden_dim})

        return cls(input_dim, output_dim, layers_config).to(device)
