#!/usr/bin/env python3
"""
Perplexity Investigation: EquiTile vs NanoGPT
==============================================

Systematic ablation study to understand the perplexity gap.

Key hypotheses to test:
1. MoT sparsity loses information
2. Output scaling affects gradient quality
3. Initialization too conservative
4. Architecture differences (pre-norm vs post-norm)
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from bioplausible.models.equitile.benchmarks.compare_nanoGPT import (
    NanoGPTConfig,
    NanoGPTModel,
)
from bioplausible.models.equitile.lm_demo import FastLMConfig, FastLMEquiTile

# =============================================================================
# Test Harness
# =============================================================================


@dataclass
class AblationResult:
    """Results from an ablation test."""

    name: str
    config: Dict
    initial_loss: float
    final_loss: float
    initial_ppl: float
    final_ppl: float
    grad_norm: float
    param_count: int


def run_training_ablation(
    model: nn.Module,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    val_data: Tuple[torch.Tensor, torch.Tensor],
    epochs: int = 3,
    learning_rate: float = 3e-4,
    name: str = "model",
) -> AblationResult:
    """Run training ablation and track metrics."""
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95)
    )

    train_input, train_target = train_data
    val_input, val_target = val_data

    # Initial evaluation
    model.eval()
    with torch.no_grad():
        output = model(val_input)
        # Handle NanoGPT which returns (logits, loss) tuple
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        val_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), val_target.view(-1)
        )
        initial_loss = val_loss.item()
        initial_ppl = torch.exp(val_loss).item()

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_input)
        if isinstance(output, tuple):
            logits = output[0]
            loss = (
                output[1]
                if output[1] is not None
                else F.cross_entropy(
                    logits.view(-1, logits.size(-1)), train_target.view(-1)
                )
            )
        else:
            logits = output
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), train_target.view(-1)
            )
        loss.backward()

        # Track gradient norm
        grad_norm = torch.sqrt(
            sum(
                p.grad.data.norm(2) ** 2
                for p in model.parameters()
                if p.grad is not None
            )
        )

        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        output = model(val_input)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        val_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), val_target.view(-1)
        )
        final_loss = val_loss.item()
        final_ppl = torch.exp(val_loss).item()

    param_count = sum(p.numel() for p in model.parameters())

    return AblationResult(
        name=name,
        config={},
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_ppl=initial_ppl,
        final_ppl=final_ppl,
        grad_norm=grad_norm.item(),
        param_count=param_count,
    )


# =============================================================================
# Ablation Studies
# =============================================================================


def ablation_mot_sparsity(vocab_size: int = 1000, seq_len: int = 64):
    """Test impact of MoT sparsity (k value)."""
    print("\n" + "=" * 70)
    print("ABLATION 1: MoT Sparsity (k value)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data
    train_input = torch.randint(0, vocab_size, (32, seq_len), device=device)
    train_target = train_input.clone()
    val_input = torch.randint(0, vocab_size, (8, seq_len), device=device)
    val_target = val_input.clone()

    results = []

    # Test different k values
    for mot_k in [1, 2, 4, -1]:  # -1 = use all tiles (no sparsity)
        k_name = "all" if mot_k == -1 else f"k={mot_k}"
        print(f"\nTesting MoT {k_name}...")

        tiles_per_layer = 4
        actual_k = tiles_per_layer if mot_k == -1 else mot_k

        config = FastLMConfig(
            vocab_size=vocab_size,
            embed_dim=128,
            num_layers=4,
            neurons_per_tile=32,
            tiles_per_layer=tiles_per_layer,
            mot_k=actual_k,
            num_heads=4,
            num_kv_heads=2,
            use_gradient_checkpointing=False,
            use_compile=False,
        )
        model = FastLMEquiTile(config).to(device)

        result = run_training_ablation(
            model,
            (train_input, train_target),
            (val_input, val_target),
            epochs=3,
            learning_rate=3e-4,
            name=f"MoT {k_name}",
        )
        result.config = {"mot_k": k_name}
        results.append(result)

        print(
            f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
        )
        print(f"  Grad norm: {result.grad_norm:.4f}")

    # Summary
    print("\n" + "-" * 70)
    print("Summary: MoT Sparsity")
    print("-" * 70)
    print(f"{'k value':<15} {'Initial PPL':>15} {'Final PPL':>15} {'Grad Norm':>15}")
    for r in results:
        k = r.config["mot_k"]
        print(
            f"{k:<15} {r.initial_ppl:>15.2f} {r.final_ppl:>15.2f} {r.grad_norm:>15.4f}"
        )

    return results


def ablation_initialization(vocab_size: int = 1000, seq_len: int = 64):
    """Test impact of initialization schemes."""
    print("\n" + "=" * 70)
    print("ABLATION 2: Initialization Schemes")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data
    train_input = torch.randint(0, vocab_size, (32, seq_len), device=device)
    train_target = train_input.clone()
    val_input = torch.randint(0, vocab_size, (8, seq_len), device=device)
    val_target = val_input.clone()

    results = []

    # Test different initialization std values
    for init_std in [0.01, 0.02, 0.05, 0.1]:
        print(f"\nTesting init_std={init_std}...")

        config = FastLMConfig(
            vocab_size=vocab_size,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
        )
        model = FastLMEquiTile(config).to(device)

        # Override initialization
        with torch.no_grad():
            nn.init.normal_(model.token_embedding.weight, mean=0, std=init_std)
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=init_std)

        result = run_training_ablation(
            model,
            (train_input, train_target),
            (val_input, val_target),
            epochs=3,
            learning_rate=3e-4,
            name=f"init_std={init_std}",
        )
        result.config = {"init_std": init_std}
        results.append(result)

        print(
            f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
        )

    # Summary
    print("\n" + "-" * 70)
    print("Summary: Initialization")
    print("-" * 70)
    print(f"{'Init Std':<15} {'Initial PPL':>15} {'Final PPL':>15} {'Grad Norm':>15}")
    for r in results:
        std = r.config["init_std"]
        print(
            f"{std:<15.3f} {r.initial_ppl:>15.2f} {r.final_ppl:>15.2f} {r.grad_norm:>15.4f}"
        )

    return results


def ablation_output_scale(vocab_size: int = 1000, seq_len: int = 64):
    """Test impact of output scaling."""
    print("\n" + "=" * 70)
    print("ABLATION 3: Output Scaling")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data
    train_input = torch.randint(0, vocab_size, (32, seq_len), device=device)
    train_target = train_input.clone()
    val_input = torch.randint(0, vocab_size, (8, seq_len), device=device)
    val_target = val_input.clone()

    results = []

    # Test different output scale values
    for scale_init in [0.1, 0.5, 1.0, 2.0]:
        print(f"\nTesting output_scale={scale_init}...")

        config = FastLMConfig(
            vocab_size=vocab_size,
            embed_dim=128,
            num_layers=4,
            num_heads=4,
            num_kv_heads=2,
        )
        model = FastLMEquiTile(config).to(device)

        # Override output scale
        with torch.no_grad():
            model.output_scale.fill_(scale_init)

        result = run_training_ablation(
            model,
            (train_input, train_target),
            (val_input, val_target),
            epochs=3,
            learning_rate=3e-4,
            name=f"scale={scale_init}",
        )
        result.config = {"output_scale": scale_init}
        results.append(result)

        print(
            f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
        )

    # Summary
    print("\n" + "-" * 70)
    print("Summary: Output Scaling")
    print("-" * 70)
    print(f"{'Scale':<15} {'Initial PPL':>15} {'Final PPL':>15} {'Grad Norm':>15}")
    for r in results:
        scale = r.config["output_scale"]
        print(
            f"{scale:<15.2f} {r.initial_ppl:>15.2f} {r.final_ppl:>15.2f} {r.grad_norm:>15.4f}"
        )

    return results


def ablation_architecture(vocab_size: int = 1000, seq_len: int = 64):
    """Test architecture differences (EquiTile vs NanoGPT style)."""
    print("\n" + "=" * 70)
    print("ABLATION 4: Architecture Comparison")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data
    train_input = torch.randint(0, vocab_size, (32, seq_len), device=device)
    train_target = train_input.clone()
    val_input = torch.randint(0, vocab_size, (8, seq_len), device=device)
    val_target = val_input.clone()

    results = []

    # NanoGPT baseline
    print("\nTesting NanoGPT baseline...")
    nanogpt_config = NanoGPTConfig(
        vocab_size=vocab_size,
        block_size=seq_len,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    nanogpt = NanoGPTModel(nanogpt_config).to(device)

    result = run_training_ablation(
        nanogpt,
        (train_input, train_target),
        (val_input, val_target),
        epochs=3,
        learning_rate=3e-4,
        name="NanoGPT",
    )
    results.append(result)
    print(
        f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
    )

    # EquiTile default
    print("\nTesting EquiTile default...")
    equitile_config = FastLMConfig(
        vocab_size=vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        mot_k=2,
    )
    equitile = FastLMEquiTile(equitile_config).to(device)

    result = run_training_ablation(
        equitile,
        (train_input, train_target),
        (val_input, val_target),
        epochs=3,
        learning_rate=3e-4,
        name="EquiTile",
    )
    results.append(result)
    print(
        f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
    )

    # EquiTile with NanoGPT-like settings
    print("\nTesting EquiTile (NanoGPT-like: k=all, init=0.02)...")
    equitile_config2 = FastLMConfig(
        vocab_size=vocab_size,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        mot_k=4,  # Use all tiles
    )
    equitile2 = FastLMEquiTile(equitile_config2).to(device)

    # Override initialization to match NanoGPT
    with torch.no_grad():
        nn.init.normal_(equitile2.token_embedding.weight, mean=0, std=0.02)
        for module in equitile2.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
        equitile2.output_scale.fill_(1.0)

    result = run_training_ablation(
        equitile2,
        (train_input, train_target),
        (val_input, val_target),
        epochs=3,
        learning_rate=3e-4,
        name="EquiTile*",
    )
    results.append(result)
    print(
        f"  Initial PPL: {result.initial_ppl:.2f} → Final PPL: {result.final_ppl:.2f}"
    )

    # Summary
    print("\n" + "-" * 70)
    print("Summary: Architecture")
    print("-" * 70)
    print(f"{'Model':<20} {'Params':>10} {'Initial PPL':>15} {'Final PPL':>15}")
    for r in results:
        print(
            f"{r.name:<20} {r.param_count:>10,} {r.initial_ppl:>15.2f} {r.final_ppl:>15.2f}"
        )

    return results


# =============================================================================
# Main
# =============================================================================


def run_all_ablations():
    """Run all ablation studies."""
    print("=" * 70)
    print("PERPLEXITY INVESTIGATION: EquiTile vs NanoGPT")
    print("=" * 70)

    all_results = {
        "mot_sparsity": ablation_mot_sparsity(),
        "initialization": ablation_initialization(),
        "output_scale": ablation_output_scale(),
        "architecture": ablation_architecture(),
    }

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)

    # Find best configuration from each ablation
    best_mot = min(all_results["mot_sparsity"], key=lambda x: x.final_ppl)
    best_init = min(all_results["initialization"], key=lambda x: x.final_ppl)
    best_scale = min(all_results["output_scale"], key=lambda x: x.final_ppl)
    best_arch = min(all_results["architecture"], key=lambda x: x.final_ppl)

    print(f"""
Based on the ablation studies:

1. MoT Sparsity:
   - Best: {best_mot.config['mot_k']} (PPL: {best_mot.final_ppl:.2f})
   - Recommendation: {'Use all tiles (no sparsity)' if best_mot.config['mot_k'] == 'all' else f'Use k={best_mot.config["mot_k"]}'}

2. Initialization:
   - Best: init_std={best_init.config['init_std']:.3f} (PPL: {best_init.final_ppl:.2f})
   - Recommendation: Use {best_init.config['init_std']:.3f} for embedding and linear layers

3. Output Scale:
   - Best: scale={best_scale.config['output_scale']:.2f} (PPL: {best_scale.final_ppl:.2f})
   - Recommendation: Initialize output scale to {best_scale.config['output_scale']:.2f}

4. Architecture:
   - Best: {best_arch.name} (PPL: {best_arch.final_ppl:.2f})
   - Recommendation: {'EquiTile matches NanoGPT' if best_arch.final_ppl < 10 else 'Further tuning needed'}

Next Steps:
- Apply best configurations to main model
- Re-run NanoGPT comparison benchmark
- Validate on larger dataset (TinyStories)
""")

    return all_results


if __name__ == "__main__":
    run_all_ablations()
