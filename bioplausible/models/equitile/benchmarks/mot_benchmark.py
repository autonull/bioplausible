"""
Optimized Mixture of Tiles (MoT) Kernels
=========================================

High-performance implementations for MoT operations.

Note: The vectorized PyTorch implementation in fast_lm.py is already
highly optimized. This module provides documentation and utilities.

Usage
-----
>>> from bioplausible.models.equitile.lm_demo.fast_lm import MixtureOfTiles
>>> mot = MixtureOfTiles(embed_dim=192, neurons_per_tile=48, tiles_per_layer=4, mot_k=2)
>>> output, tile_importance = mot(x)
"""

import torch

# =============================================================================
# Benchmark
# =============================================================================


def benchmark_mot(
    batch_size: int = 32,
    seq_len: int = 128,
    embed_dim: int = 192,
    tiles: int = 4,
    tile_dim: int = 48,
    k: int = 2,
    warmup: int = 10,
    repeat: int = 50,
) -> dict:
    """Benchmark MoT implementation.

    Returns
    -------
    dict
        Timing results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)

    results = {}

    # Standard (from fast_lm.py)
    from bioplausible.models.equitile.lm_demo.fast_lm import MixtureOfTiles

    mot_standard = MixtureOfTiles(
        embed_dim=embed_dim,
        neurons_per_tile=tile_dim,
        tiles_per_layer=tiles,
        mot_k=k,
    ).to(device)
    mot_standard.eval()

    # Warmup
    for _ in range(warmup):
        _ = mot_standard(x)

    # Measure
    import time

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(repeat):
        _ = mot_standard(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - start
    results["time_ms"] = elapsed / repeat * 1000
    results["throughput_tok_s"] = (batch_size * seq_len) / (elapsed / repeat)

    return results


if __name__ == "__main__":
    print("MoT Kernel Benchmark")
    print("=" * 50)

    results = benchmark_mot(
        batch_size=32,
        seq_len=128,
        embed_dim=192,
        tiles=4,
        tile_dim=48,
        k=2,
        warmup=10,
        repeat=50,
    )

    print(f"Time:       {results['time_ms']:.2f} ms")
    print(f"Throughput: {results['throughput_tok_s']:,.0f} tok/s")
