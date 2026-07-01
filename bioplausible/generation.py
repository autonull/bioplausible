"""
Universal Text Generation for Any Model

Provides autoregressive generation for any model that outputs logits,
including bioplausible research algorithms.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def generate_text(
    model: torch.nn.Module,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    device: str = "cpu",
) -> str:
    """
    Universal autoregressive text generation for any model.

    Works with:
    - EqProp LM variants (have built-in generate)
    - Research algorithms (need this wrapper)
    - Any model that outputs logits

    Args:
        model: Any torch.nn.Module that outputs logits
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        prompt: Starting text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k tokens
        device: Device to run on

    Returns:
        Generated text string
    """
    model.eval()

    # Convert prompt to indices
    if prompt:
        indices = [char_to_idx.get(c, 0) for c in prompt]
    else:
        indices = [0]  # Start with first char

    generated = list(indices)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Prepare input
            if hasattr(model, "config"):
                # Research algorithm - needs one-hot encoding
                vocab_size = len(char_to_idx)
                x = torch.tensor([indices[-1]], device=device)
                x = F.one_hot(x, num_classes=vocab_size).float()
            else:
                # LM variant or standard model
                x = torch.tensor([indices], device=device)

            # Forward pass
            try:
                logits = model(x)

                # Handle different output shapes
                if logits.dim() == 3:
                    # [batch, seq, vocab] - take last token
                    logits = logits[0, -1, :]
                elif logits.dim() == 2:
                    # [batch, vocab] or [seq, vocab]
                    if logits.shape[0] == 1:
                        logits = logits[0]
                    else:
                        logits = logits[-1]

            except Exception:
                # Fallback: try with different input format
                try:
                    x = torch.tensor([[indices[-1]]], device=device)
                    logits = model(x)
                    if logits.dim() == 3:
                        logits = logits[0, -1, :]
                    elif logits.dim() == 2:
                        logits = logits[0]
                except (RuntimeError, ValueError, IndexError):
                    break

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("Inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1).item()

            # Add to generated sequence
            generated.append(next_idx)
            indices.append(next_idx)

    # Decode to text
    text = "".join([idx_to_char.get(idx, "?") for idx in generated])
    return text


def generate_from_dataset(
    model: torch.nn.Module,
    dataset: Any,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> str:
    """
    Convenience wrapper that extracts vocab from dataset.

    Args:
        model: Model to generate with
        dataset: CharDataset with vocab_size, char_to_idx, idx_to_char
        prompt: Starting text
        max_new_tokens: Tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering

    Returns:
        Generated text string
    """
    # Extract vocab from dataset
    char_to_idx = dataset.char_to_idx
    idx_to_char = dataset.idx_to_char

    # Determine device
    device = next(model.parameters()).device if list(model.parameters()) else "cpu"

    return generate_text(
        model=model,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device,
    )
