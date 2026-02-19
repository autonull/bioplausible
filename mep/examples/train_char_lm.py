#!/usr/bin/env python3
"""
Character-level Language Model with Equilibrium Propagation

This example demonstrates EP training on a simple character-level LM task.
Unlike classification, LM training shows different dynamics:
- Sequential prediction rather than single-label classification
- Energy-based formulation may offer different convergence properties
- Local learning rules could affect how context is learned

Note: EP works with continuous states, so we use an MLP architecture
that operates on embedded representations directly.

Run: python examples/train_char_lm.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Tuple

from mep import smep, muon_backprop


class CharLM(nn.Module):
    """Simple character-level language model with MLP architecture for EP."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 256, seq_len: int = 32, use_embedding: bool = True):
        super().__init__()
        self.use_embedding = use_embedding
        
        if use_embedding:
            self.embed = nn.Embedding(vocab_size, embed_dim)
        else:
            self.embed = None
            
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # MLP architecture compatible with EP
        # Input: flattened embedding sequence (seq_len * embed_dim)
        self.network = nn.Sequential(
            nn.Linear(seq_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size * seq_len),
        )
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len) for Long input
        # or (batch, seq_len * embed_dim) for Float input
        
        if self.use_embedding and x.dtype == torch.long:
            # Embed and flatten
            embed = self.embed(x)  # (batch, seq_len, embed_dim)
            x = embed.view(x.size(0), -1)  # (batch, seq_len * embed_dim)
        elif x.dim() == 3:
            # (batch, seq_len, embed_dim) - flatten it
            x = x.view(x.size(0), -1)
        # else: already flattened (batch, seq_len * embed_dim)
        
        # Network forward pass
        output = self.network(x)  # (batch, vocab_size * seq_len)
        
        # Reshape to (batch * seq_len, vocab_size) for cross-entropy
        output = output.view(-1, self.vocab_size)
        return output


def load_shakespeare() -> Tuple[str, dict, dict]:
    """Load Shakespeare text and create vocab."""
    # Simple Shakespeare corpus (subset)
    text = """
    ROMEO: But, soft! what light through yonder window breaks?
    It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon,
    Who is already sick and pale with grief,
    That thou her maid art far more fair than she.
    
    JULIET: O Romeo, Romeo! wherefore art thou Romeo?
    Deny thy father and refuse thy name;
    Or, if thou wilt not, be but sworn my love,
    And I'll no longer be a Capulet.
    
    ROMEO: I take thee at thy word.
    Call me but love, and I'll be new baptized;
    Henceforth I never will be Romeo.
    
    JULIET: What man art thou that thus bescreened in night
    So stumblest on my counsel?
    
    ROMEO: By a name
    I know not how to tell thee who I am.
    """ * 10  # Repeat for more data
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    return text, char_to_idx, idx_to_char


def create_batches(text: str, char_to_idx: dict, seq_len: int, batch_size: int):
    """Create training batches."""
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    
    # Create sequences
    sequences = []
    targets = []
    for i in range(0, len(data) - seq_len, batch_size):
        seq = data[i:i+seq_len]
        tgt = data[i+1:i+seq_len+1]
        if len(seq) == seq_len:
            sequences.append(seq)
            targets.append(tgt)
    
    return torch.stack(sequences), torch.stack(targets)


def train_epoch_ep(model, optimizer, sequences, targets, device, embed_layer):
    """Train one epoch with EP."""
    model.train()
    total_loss = 0

    for seq, tgt in zip(sequences, targets):
        seq, tgt = seq.to(device), tgt.to(device)
        
        # Add batch dimension: (seq_len,) -> (1, seq_len)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)
        
        # Pre-embed the input for EP (EP needs float states)
        with torch.no_grad():
            seq_embedded = embed_layer(seq)  # (1, seq_len, embed_dim)
            seq_embedded = seq_embedded.view(seq.size(0), -1)  # (1, seq_len * embed_dim)

        # EP step - predict next character using embedded input
        optimizer.step(x=seq_embedded, target=tgt)

    # Compute loss for reporting
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for seq, tgt in zip(sequences, targets):
            seq, tgt = seq.to(device), tgt.to(device)
            
            # Add batch dimension
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            if tgt.dim() == 1:
                tgt = tgt.unsqueeze(0)
            
            # For EP model (no embedding), embed first
            seq_embedded = embed_layer(seq)
            seq_embedded = seq_embedded.view(seq.size(0), -1)
            
            # Use network directly (model without embedding)
            output = model.network(seq_embedded)  # (1, vocab_size * seq_len)
            # Reshape to (batch * seq_len, vocab_size)
            logits = output.view(-1, model.vocab_size)
            
            # Reshape target: (1, seq_len) -> (seq_len,)
            tgt_flat = tgt.view(-1)
            loss = nn.functional.cross_entropy(logits, tgt_flat)
            total_loss += loss.item()

    return total_loss / len(sequences)


def train_epoch_bp(model, optimizer, sequences, targets, device):
    """Train one epoch with backprop."""
    model.train()
    total_loss = 0

    for seq, tgt in zip(sequences, targets):
        seq, tgt = seq.to(device), tgt.to(device)
        
        # Add batch dimension: (seq_len,) -> (1, seq_len)
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(0)

        optimizer.zero_grad()
        logits = model(seq)
        # Reshape target to match logits: (batch * seq_len,)
        tgt_flat = tgt.view(-1)
        loss = nn.functional.cross_entropy(logits, tgt_flat)
        loss.backward()
        optimizer.step()

    return total_loss / len(sequences)


def generate_text(model, char_to_idx, idx_to_char, seed_text: str, max_len: int = 100, device='cpu'):
    """Generate text from model (with embedding layer)."""
    model.eval()

    # Encode seed
    chars = list(seed_text)

    with torch.no_grad():
        for _ in range(max_len):
            # Use last SEQ_LEN characters (pad if needed)
            context = chars[-model.seq_len:] if len(chars) >= model.seq_len else chars
            # Pad to seq_len
            while len(context) < model.seq_len:
                context = [' '] + context
            
            seq = torch.tensor([[char_to_idx.get(ch, 0) for ch in context]], device=device)
            logits = model(seq)
            # Get prediction for last position
            next_char_idx = logits[0, -1].argmax().item()
            chars.append(idx_to_char.get(next_char_idx, ' '))

    return ''.join(chars)


def generate_text_ep(model, embed_layer, char_to_idx, idx_to_char, seed_text: str, max_len: int = 100, device='cpu'):
    """Generate text from EP model (separate embedding layer)."""
    model.eval()
    embed_layer.eval()

    # Encode seed
    chars = list(seed_text)

    with torch.no_grad():
        for _ in range(max_len):
            # Use last SEQ_LEN characters
            context = chars[-model.seq_len:] if len(chars) >= model.seq_len else chars
            # Pad to seq_len
            while len(context) < model.seq_len:
                context = [' '] + context
            
            # Embed
            seq_indices = torch.tensor([[char_to_idx.get(ch, 0) for ch in context]], device=device)
            seq_embedded = embed_layer(seq_indices)
            seq_embedded = seq_embedded.view(seq_embedded.size(0), -1)
            
            # Forward through network
            output = model.network(seq_embedded)
            next_char_idx = output[0, -1].argmax().item()
            chars.append(idx_to_char.get(next_char_idx, ' '))

    return ''.join(chars)


def main():
    print("=" * 60)
    print("Character-level Language Model: EP vs Backprop")
    print("=" * 60)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 3
    SEQ_LEN = 32
    BATCH_SIZE = 16
    
    # Load data
    print("\nLoading Shakespeare corpus...")
    text, char_to_idx, idx_to_char = load_shakespeare()
    vocab_size = len(char_to_idx)
    print(f"Vocabulary size: {vocab_size} characters")
    print(f"Corpus length: {len(text)} characters")
    
    # Create batches
    sequences, targets = create_batches(text, char_to_idx, SEQ_LEN, BATCH_SIZE)
    print(f"Training sequences: {len(sequences)}")
    
    # Train with Backprop
    print("\n" + "-" * 60)
    print("Training with Backpropagation")
    print("-" * 60)

    model_bp = CharLM(vocab_size, seq_len=SEQ_LEN, use_embedding=True).to(DEVICE)
    opt_bp = muon_backprop(model_bp.parameters(), lr=0.01)

    start = time.time()
    for epoch in range(EPOCHS):
        loss = train_epoch_bp(model_bp, opt_bp, sequences, targets, DEVICE)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss:.3f}, Time={elapsed:.1f}s")

    print("\nGenerated text (backprop):")
    print(generate_text(model_bp, char_to_idx, idx_to_char, "ROMEO: ", device=DEVICE))

    # Train with EP
    print("\n" + "-" * 60)
    print("Training with Equilibrium Propagation")
    print("-" * 60)

    # For EP, we use a model without embedding layer (EP works with float inputs)
    model_ep = CharLM(vocab_size, seq_len=SEQ_LEN, use_embedding=False).to(DEVICE)
    # Create a separate embedding layer for preprocessing
    embed_layer = nn.Embedding(vocab_size, 64).to(DEVICE)
    
    opt_ep = smep(
        list(model_ep.parameters()) + list(embed_layer.parameters()),
        model=model_ep,
        lr=0.005,  # Lower LR for EP
        mode='ep',
        settle_steps=10,
        loss_type='cross_entropy',
    )

    start = time.time()
    for epoch in range(EPOCHS):
        loss = train_epoch_ep(model_ep, opt_ep, sequences, targets, DEVICE, embed_layer)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS}: Loss={loss:.3f}, Time={elapsed:.1f}s")

    print("\nGenerated text (EP):")
    # For generation with EP model, we need to use embed_layer + model_ep.network
    print(generate_text_ep(model_ep, embed_layer, char_to_idx, idx_to_char, "ROMEO: ", device=DEVICE))
    
    print("\n" + "=" * 60)
    print("Notes:")
    print("- EP trains without backpropagation through time")
    print("- Energy-based formulation may affect learning dynamics")
    print("- Generated text quality depends on training duration")
    print("=" * 60)


if __name__ == "__main__":
    main()
