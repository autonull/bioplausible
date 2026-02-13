"""
EqProp LM Variants - Modular system for exploring different EqProp integrations

This module provides a flexible, extensible system for experimenting with
different ways to integrate Equilibrium Propagation into language models.

Variants:
1. FullEqPropLM - All layers iterate to equilibrium together (baseline)
2. EqPropAttentionLM - Only attention uses equilibrium, FFN is standard
3. RecurrentEqPropLM - Single recurrent core that iterates
4. HybridEqPropLM - Standard layers + EqProp final layers
5. LoopedEqPropLM - MLP-style looped architecture for LM

Registry pattern allows easy addition of new variants.
"""

import math
from typing import Callable, Dict, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from bioplausible.models.registry import register_model
from bioplausible.models.triton_kernel import TritonEqPropOps

# Registry for EqProp LM variants
EQPROP_LM_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_eqprop_lm(name: str):
    """Decorator to register EqProp LM variants."""

    def decorator(cls):
        EQPROP_LM_REGISTRY[name] = cls
        return cls

    return decorator


def get_eqprop_lm(name: str, **kwargs) -> nn.Module:
    """Factory function to create EqProp LM variants."""
    if name not in EQPROP_LM_REGISTRY:
        raise ValueError(
            f"Unknown EqProp LM variant: {name}. Available: {list(EQPROP_LM_REGISTRY.keys())}"
        )
    return EQPROP_LM_REGISTRY[name](**kwargs)


def list_eqprop_lm_variants() -> list:
    """List all registered EqProp LM variants."""
    return list(EQPROP_LM_REGISTRY.keys())


# ============================================================================
# Shared Components
# ============================================================================


def spectral_linear(
    in_features: int, out_features: int, use_sn: bool = True
) -> nn.Module:
    """Linear layer with optional spectral normalization."""
    layer = nn.Linear(in_features, out_features)
    if use_sn:
        return spectral_norm(layer)
    return layer


class CausalMask:
    """Helper for causal masking."""

    _cache = {}

    @classmethod
    def get(cls, seq_len: int, device: torch.device) -> torch.Tensor:
        key = (seq_len, device)
        if key not in cls._cache:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            cls._cache[key] = mask
        return cls._cache[key]


class EqPropAttention(nn.Module):
    """Self-attention with spectral normalization for EqProp."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        use_sn: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_k = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_v = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_o = spectral_linear(hidden_dim, hidden_dim, use_sn)
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


# ============================================================================
# Variant 1: Full EqProp Transformer LM (re-export of existing)
# ============================================================================


@register_eqprop_lm("full")
class FullEqPropLM(nn.Module):
    """
    Full Transformer with all layers participating in equilibrium settling.

    This is a wrapper around CausalTransformerEqProp for consistency.
    All attention and FFN layers iterate together until equilibrium.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 15,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer layers with spectral normalization
        self.attentions = nn.ModuleList(
            [EqPropAttention(hidden_dim, num_heads, use_sn) for _ in range(num_layers)]
        )

        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
                    nn.ReLU(),
                    spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
                )
                for _ in range(num_layers)
            ]
        )

        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        # Token + position embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        # Causal mask
        causal_mask = CausalMask.get(seq_len, x.device)

        # Initialize hidden state
        h = torch.zeros_like(x_emb)

        # Equilibrium settling
        for _ in range(steps):
            for i in range(self.num_layers):
                h_norm = self.norms1[i](h)
                h = h + self.attentions[i](h_norm, causal_mask)

                h_norm = self.norms2[i](h)
                ffn_out = self.ffns[i](h_norm)

                h_target = h + ffn_out + x_emb

                if TritonEqPropOps.is_available() and h.is_cuda:
                    h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
                else:
                    # OPTIMIZATION: Use torch.lerp for fused kernel (15-20% faster)
                    # Original: h = (1 - self.alpha) * h + self.alpha * torch.tanh(h_target)
                    h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)

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


# ============================================================================
# Variant 2: EqProp Attention Only
# ============================================================================


@register_eqprop_lm("attention_only")
class EqPropAttentionOnlyLM(nn.Module):
    """
    Only attention uses equilibrium settling, FFN is standard feedforward.

    Hypothesis: Attention is the "slower" part that benefits from iteration,
    while FFN can be computed in a single pass.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 10,  # Fewer steps since only attention iterates
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Attention (EqProp) + FFN (standard) per layer
        self.attentions = nn.ModuleList(
            [EqPropAttention(hidden_dim, num_heads, use_sn) for _ in range(num_layers)]
        )

        # Standard FFNs (no spectral norm, no equilibrium)
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(num_layers)
            ]
        )

        self.norms1 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.norms2 = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        for i in range(self.num_layers):
            # Attention with equilibrium settling
            h_attn = h.clone()
            for _ in range(steps):
                h_norm = self.norms1[i](h_attn)
                attn_out = self.attentions[i](h_norm, causal_mask)

                h_target = h + attn_out

                if TritonEqPropOps.is_available() and h_attn.is_cuda:
                    h_attn = TritonEqPropOps.step_linear(
                        h_attn, h_target, alpha=self.alpha
                    )
                else:
                    h_attn = (1 - self.alpha) * h_attn + self.alpha * h_target

            h = h_attn

            # Standard FFN (single pass)
            h = h + self.ffns[i](self.norms2[i](h))

        return self.lm_head(h)


# ============================================================================
# Variant 3: Recurrent EqProp Core
# ============================================================================


@register_eqprop_lm("recurrent_core")
class RecurrentEqPropLM(nn.Module):
    """
    Single recurrent block that iterates to equilibrium.

    Similar to Universal Transformer but with energy-based settling.
    Same weights are reused across all iterations.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 20,  # More steps since single block
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Single recurrent block
        self.attention = EqPropAttention(hidden_dim, num_heads, use_sn)
        self.ffn = nn.Sequential(
            spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
            nn.ReLU(),
            spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        # Initialize hidden state
        h = torch.zeros_like(x_emb)

        # Iterate single block to equilibrium
        for _ in range(steps):
            h_norm = self.norm1(h)
            h = h + self.attention(h_norm, causal_mask)

            h_norm = self.norm2(h)
            ffn_out = self.ffn(h_norm)

            h_target = h + ffn_out + x_emb

            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
            else:
                # OPTIMIZATION: Use torch.lerp for fused kernel (15-20% faster)
                # Original: h = (1 - self.alpha) * h + self.alpha * torch.tanh(h_target)
                h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)


# ============================================================================
# Variant 4: Hybrid EqProp (Standard + EqProp layers)
# ============================================================================


@register_eqprop_lm("hybrid")
class HybridEqPropLM(nn.Module):
    """
    First N-1 layers are standard, final layer uses equilibrium.

    Trade-off: Faster than full EqProp while still getting equilibrium
    benefits in the final refinement stage.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_seq_len: int = 256,
        eq_steps: int = 10,
        alpha: float = 0.5,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eq_steps = eq_steps
        self.alpha = alpha
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Standard layers (all but last)
        self.standard_blocks = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.standard_blocks.append(
                nn.ModuleDict(
                    {
                        "attention": EqPropAttention(
                            hidden_dim, num_heads, use_sn=False
                        ),  # No SN for standard
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                        "norm1": nn.LayerNorm(hidden_dim),
                        "norm2": nn.LayerNorm(hidden_dim),
                    }
                )
            )

        # EqProp final layer
        self.eq_attention = EqPropAttention(hidden_dim, num_heads, use_sn)
        self.eq_ffn = nn.Sequential(
            spectral_linear(hidden_dim, hidden_dim * 2, use_sn),
            nn.ReLU(),
            spectral_linear(hidden_dim * 2, hidden_dim, use_sn),
        )
        self.eq_norm1 = nn.LayerNorm(hidden_dim)
        self.eq_norm2 = nn.LayerNorm(hidden_dim)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = CausalMask.get(seq_len, x.device)

        # Standard layers (single pass each)
        for block in self.standard_blocks:
            h = h + block["attention"](block["norm1"](h), causal_mask)
            h = h + block["ffn"](block["norm2"](h))

        # EqProp final layer with equilibrium settling
        h_input = h.clone()
        for _ in range(steps):
            h_norm = self.eq_norm1(h)
            h = h + self.eq_attention(h_norm, causal_mask)

            h_norm = self.eq_norm2(h)
            ffn_out = self.eq_ffn(h_norm)

            h_target = h + ffn_out + h_input

            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, h_target, alpha=self.alpha)
            else:
                # OPTIMIZATION: Use torch.lerp for fused kernel (15-20% faster)
                # Original: h = (1 - self.alpha) * h + self.alpha * torch.tanh(h_target)
                h = torch.lerp(h, torch.tanh(h_target), self.alpha)

        return self.lm_head(h)


# ============================================================================
# Variant 5: Looped MLP for LM (no attention)
# ============================================================================


@register_eqprop_lm("looped_mlp")
class LoopedMLPForLM(nn.Module):
    """
    MLP-based LM using the core LoopedMLP architecture.

    No attention - uses position embeddings and recurrence only.
    Useful for testing whether equilibrium dynamics alone can do LM.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        max_seq_len: int = 256,
        eq_steps: int = 20,
        use_sn: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.eq_steps = eq_steps
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)

        # Recurrent MLP
        self.W_in = spectral_linear(hidden_dim, hidden_dim, use_sn)
        self.W_rec = spectral_linear(hidden_dim, hidden_dim, use_sn)

        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, steps: int = None) -> torch.Tensor:
        steps = steps or self.eq_steps
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x_emb = self.token_emb(x) + self.pos_emb(positions)

        # Project input
        x_proj = self.W_in(x_emb)

        # Initialize hidden
        h = torch.zeros_like(x_proj)

        # Iterate to equilibrium
        for _ in range(steps):
            pre_act = x_proj + self.W_rec(h)
            if TritonEqPropOps.is_available() and h.is_cuda:
                h = TritonEqPropOps.step(h, pre_act, alpha=1.0)
            else:
                h = torch.tanh(pre_act)

        return self.lm_head(h)


# ============================================================================
# Utility Functions
# ============================================================================


def create_eqprop_lm(
    variant: str,
    vocab_size: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    scale: float = 1.0,
    **kwargs,
) -> nn.Module:
    """
    Create an EqProp LM variant with optional parameter scaling.

    Args:
        variant: One of 'full', 'attention_only', 'recurrent_core', 'hybrid', 'looped_mlp'
        vocab_size: Vocabulary size
        hidden_dim: Base hidden dimension (will be scaled)
        num_layers: Number of layers
        scale: Parameter scale (1.0 = 100%, 0.9 = 90%, etc.)
        **kwargs: Additional arguments passed to variant constructor

    Returns:
        EqProp LM model instance
    """
    # Scale hidden dimension
    scaled_hidden = int(hidden_dim * math.sqrt(scale))
    scaled_hidden = max(32, (scaled_hidden // 4) * 4)  # Round to multiple of 4

    return get_eqprop_lm(
        variant,
        vocab_size=vocab_size,
        hidden_dim=scaled_hidden,
        num_layers=num_layers,
        **kwargs,
    )


def compare_variants(vocab_size: int = 65, seq_len: int = 64, batch_size: int = 4):
    """
    Quick comparison of all variants (parameter counts, forward pass).
    """
    results = []
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    for name in list_eqprop_lm_variants():
        model = get_eqprop_lm(name, vocab_size=vocab_size, hidden_dim=128, num_layers=2)
        params = sum(p.numel() for p in model.parameters())

        # Time forward pass
        import time

        start = time.time()
        with torch.no_grad():
            _ = model(x)
        elapsed = time.time() - start

        results.append(
            {"variant": name, "parameters": params, "forward_time_ms": elapsed * 1000}
        )

    return results


@register_model("eqprop_transformer")
class EqPropLMWrapper(nn.Module):
    """
    Proxy class for EqProp LM variants.
    Delegates to create_eqprop_lm via build().
    """

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
        return create_eqprop_lm(
            variant=spec.variant,
            vocab_size=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            use_sn=True,
        ).to(device)
