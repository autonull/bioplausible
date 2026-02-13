"""
Backprop Transformer LM - Standard Transformer baseline for fair comparison

This provides a standard causal transformer (no equilibrium dynamics) with
identical architecture to CausalTransformerEqProp for fair comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.registry import register_model


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
            self.W_q(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.W_k(h)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_v(h)
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
        # Pre-norm attention
        x = x + self.attention(self.norm1(x), causal_mask)
        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


@register_model("backprop_transformer_lm")
class BackpropTransformerLM(nn.Module):
    """
    Standard Causal Transformer LM (Backprop baseline).

    Matches CausalTransformerEqProp architecture for fair comparison:
    - Same embedding dimensions
    - Same number of layers and heads
    - Same FFN structure

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
        ffn_mult: int = 2,  # Match EqProp's 2x FFN
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                BackpropTransformerBlock(hidden_dim, num_heads, ffn_mult, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output
        self.norm_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

        # Causal mask (register as buffer)
        self.register_buffer("causal_mask", None)
        self._create_causal_mask(max_seq_len)

        self._init_weights()

    def _create_causal_mask(self, seq_len: int):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def _init_weights(self):
        """Initialize weights like GPT-2."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass (single, no equilibrium).

        Args:
            x: Input token IDs [batch, seq_len]
            **kwargs: Ignored (for API compatibility with EqProp models)

        Returns:
            Logits [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)
        h = self.dropout(h)

        # Get causal mask for this sequence length
        causal_mask = (
            self.causal_mask[:seq_len, :seq_len]
            if self.causal_mask is not None
            else None
        )

        # Transformer blocks
        for block in self.blocks:
            h = block(h, causal_mask)

        # Output
        h = self.norm_f(h)
        logits = self.lm_head(h)

        return logits

    def generate(
        self, prompt: torch.Tensor, max_new_tokens: int = 100, temperature: float = 1.0
    ):
        """
        Autoregressive generation (matches EqProp API).
        """
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
        """Count trainable parameters."""
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
    """
    Factory function to create scaled model variants.

    Args:
        base_hidden: Base hidden dimension (will be scaled)
        base_layers: Base number of layers (kept constant)
        scale: Parameter scale factor (1.0 = 100%, 0.9 = 90%, 0.75 = 75%)
        vocab_size: Vocabulary size
        model_type: 'backprop' or 'eqprop'

    Returns:
        Scaled model instance
    """
    # Scale hidden dimension (primarily affects parameter count)
    # params ≈ O(hidden_dim²), so scale sqrt for linear param scaling
    scaled_hidden = int(base_hidden * math.sqrt(scale))
    # Round to nearest multiple of num_heads (usually 4)
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
        from .causal_transformer_eqprop import CausalTransformerEqProp

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
