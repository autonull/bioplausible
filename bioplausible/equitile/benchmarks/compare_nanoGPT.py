"""
NanoGPT Comparison Benchmark
=============================

Head-to-head comparison between EquiTile and NanoGPT:
- Same parameter count comparison
- Same FLOPs budget comparison
- Training speed comparison
- Memory efficiency comparison

NanoGPT Reference: https://github.com/karpathy/nanoGPT

Example
-------
>>> from bioplausible.equitile.benchmarks import compare_nanoGPT
>>> results = compare_nanoGPT(
...     task="shakespeare",
...     epochs=5,
...     batch_size=32,
... )
>>> print(f"EquiTile advantage: {results['equitile_speedup']:.2f}x")
"""

import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

# Use new torch.amp API (2.0+) or fallback
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast


# =============================================================================
# NanoGPT Implementation (for comparison)
# =============================================================================


@dataclass
class NanoGPTConfig:
    """NanoGPT configuration."""

    vocab_size: int = 1000
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.1
    bias: bool = True  # True for dense, False for MoE
    use_compile: bool = False
    compile_mode: str = "max-autotune"


class NanoGPTModel(nn.Module):
    """NanoGPT-style transformer for comparison.

    Standard decoder-only transformer architecture
    as implemented in Karpathy's nanoGPT.

    Parameters
    ----------
    config : NanoGPTConfig
        Model configuration
    """

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.block_size, config.n_embd) * 0.02
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output
        self.ln_f = nn.LayerNorm(config.n_embd)
        # Weight tying
        self.output_proj = None

        self.dropout = nn.Dropout(config.dropout)
        self.block_size = config.block_size

        self._init_weights()

        # Compile if requested
        if config.use_compile and hasattr(torch, "compile"):
            try:
                print(f"Compiling NanoGPT model (mode={config.compile_mode})...")
                self.forward = torch.compile(self.forward, mode=config.compile_mode)
            except Exception as e:
                print(f"Compilation failed: {e}")

    def _init_weights(self) -> None:
        """Initialize weights."""
        with torch.no_grad():
            nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
            nn.init.normal_(self.position_embedding, mean=0, std=0.02)

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input token IDs (batch, seq_len)
        targets : torch.Tensor, optional
            Target token IDs

        Returns
        -------
        tuple
            (logits, loss) if targets provided, else (logits, None)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Position embeddings
        pos_embs = self.position_embedding[:, :seq_len, :]

        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x + pos_embs)

        # Create causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final norm
        x = self.ln_f(x)

        # Output projection (weight tying)
        logits = F.linear(x, self.token_embedding.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            # Crop sequence if too long
            input_cond = input_ids[:, -self.block_size :]

            # Forward pass
            logits, _ = self.forward(input_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids

    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


class Block(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    """Causal self-attention."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # QKV projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len, _ = x.shape

        # QKV
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, -1).transpose(1, 2)

        # Attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout.p if self.training else 0,
            is_causal=True,
        )

        # Output
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        return self.c_proj(y)


class MLP(nn.Module):
    """Feed-forward MLP with GELU."""

    def __init__(self, config: NanoGPTConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.c_proj(x)


# =============================================================================
# Benchmark Comparison
# =============================================================================


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    model_name: str
    parameter_count: int
    train_loss: float
    val_loss: float
    train_ppl: float
    val_ppl: float
    tokens_per_sec: float
    memory_mb: float
    training_time_sec: float


def benchmark_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    gradient_accumulation_steps: int = 1,
) -> BenchmarkResult:
    """Benchmark a model's training performance.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    epochs : int
        Number of epochs
    learning_rate : float
        Learning rate
    device : str
        Device to use
    gradient_accumulation_steps : int
        Gradient accumulation steps

    Returns
    -------
    BenchmarkResult
        Benchmark results
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.95),
    )

    # Warmup
    total_steps = len(train_loader) * epochs // gradient_accumulation_steps
    warmup_steps = total_steps // 10

    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return (
            learning_rate
            * 0.5
            * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    # Training
    start_time = time.time()
    total_tokens = 0
    step = 0
    final_train_loss = 0.0

    scaler = GradScaler() if device == "cuda" else None

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (input_ids, targets) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            if scaler:
                with autocast(device_type="cuda"):
                    if hasattr(model, "forward"):
                        if model.__class__.__name__ == "NanoGPTModel":
                            logits, loss = model(input_ids, targets)
                        else:
                            logits = model(input_ids)
                            loss = model.compute_loss(logits, targets)
                    else:
                        logits = model(input_ids)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), targets.view(-1)
                        )

                scaler.scale(loss / gradient_accumulation_steps).backward()
            else:
                if (
                    hasattr(model, "forward")
                    and model.__class__.__name__ == "NanoGPTModel"
                ):
                    logits, loss = model(input_ids, targets)
                else:
                    logits = model(input_ids)
                    loss = (
                        model.compute_loss(logits, targets)
                        if hasattr(model, "compute_loss")
                        else F.cross_entropy(
                            logits.view(-1, logits.size(-1)), targets.view(-1)
                        )
                    )
                (loss / gradient_accumulation_steps).backward()

            # Optimizer step
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                step += 1

            epoch_loss += loss.item() * gradient_accumulation_steps
            n_batches += 1
            total_tokens += input_ids.numel()

        final_train_loss = epoch_loss / n_batches
        print(f"  Epoch {epoch + 1}/{epochs}: Loss = {final_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if hasattr(model, "forward") and model.__class__.__name__ == "NanoGPTModel":
                logits, loss = model(input_ids, targets)
            else:
                logits = model(input_ids)
                loss = (
                    model.compute_loss(logits, targets)
                    if hasattr(model, "compute_loss")
                    else F.cross_entropy(
                        logits.view(-1, logits.size(-1)), targets.view(-1)
                    )
                )

            val_loss += loss.item()
            n_val_batches += 1

    val_loss /= max(1, n_val_batches)

    # Calculate metrics
    training_time = time.time() - start_time
    tokens_per_sec = total_tokens / training_time

    # Memory usage
    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        memory_mb = 0.0

    return BenchmarkResult(
        model_name=model.__class__.__name__,
        parameter_count=(
            model.get_parameter_count()
            if hasattr(model, "get_parameter_count")
            else sum(p.numel() for p in model.parameters())
        ),
        train_loss=final_train_loss,
        val_loss=val_loss,
        train_ppl=torch.exp(torch.tensor(final_train_loss)).item(),
        val_ppl=torch.exp(torch.tensor(val_loss)).item(),
        tokens_per_sec=tokens_per_sec,
        memory_mb=memory_mb,
        training_time_sec=training_time,
    )


def compare_nanoGPT(
    task: str = "shakespeare",
    epochs: int = 5,
    batch_size: int = 32,
    seq_length: int = 256,
    device: str = "auto",
) -> dict[str, Any]:
    """Compare EquiTile vs NanoGPT.

    Parameters
    ----------
    task : str
        Dataset task
    epochs : int
        Number of epochs
    batch_size : int
        Batch size
    seq_length : int
        Sequence length
    device : str
        Device to use

    Returns
    -------
    dict
        Comparison results
    """
    from bioplausible.equitile.lm_demo.data import create_shakespeare_dataset
    from bioplausible.equitile.lm_demo.fast_lm import FastLMConfig, FastLMEquiTile

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("EquiTile vs NanoGPT Comparison Benchmark")
    print("=" * 60)

    # Load dataset
    print(f"\nLoading {task} dataset...")
    train_loader, val_loader, tokenizer = create_shakespeare_dataset(
        batch_size=batch_size,
        seq_length=seq_length,
        num_workers=2,
    )
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary: {vocab_size}")

    # Create models with matched parameters
    print("\nCreating models with matched parameters...")

    # NanoGPT config
    nanogpt_config = NanoGPTConfig(
        vocab_size=vocab_size,
        block_size=seq_length,
        n_layer=6,
        n_head=6,
        n_embd=192,
        dropout=0.1,
    )
    nanogpt = NanoGPTModel(nanogpt_config)
    nanogpt_params = nanogpt.get_parameter_count()
    print(f"NanoGPT parameters: {nanogpt_params:,}")

    # EquiTile config (matched)
    equitile_config = FastLMConfig(
        vocab_size=vocab_size,
        embed_dim=192,
        num_layers=6,
        hidden_dim=512,
        neurons_per_tile=48,
        tiles_per_layer=4,
        mot_k=2,
        num_heads=6,
        num_kv_heads=2,
        dropout=0.1,
        max_seq_len=seq_length,
    )
    equitile = FastLMEquiTile(equitile_config)
    equitile._init_weights()
    equitile_params = equitile.get_parameter_count()
    print(f"EquiTile parameters: {equitile_params:,}")

    # Benchmark NanoGPT
    print("\n" + "-" * 40)
    print("Benchmarking NanoGPT...")
    print("-" * 40)
    nanogpt_result = benchmark_model(
        nanogpt,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
    )

    # Benchmark EquiTile
    print("\n" + "-" * 40)
    print("Benchmarking EquiTile...")
    print("-" * 40)
    equitile_result = benchmark_model(
        equitile,
        train_loader,
        val_loader,
        epochs=epochs,
        device=device,
    )

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    results = {
        "nanogpt": vars(nanogpt_result),
        "equitile": vars(equitile_result),
        "equitile_speedup": nanogpt_result.training_time_sec
        / max(0.001, equitile_result.training_time_sec),
        "equitile_throughput_gain": equitile_result.tokens_per_sec
        / max(0.001, nanogpt_result.tokens_per_sec),
        "equitile_ppl_improvement": nanogpt_result.val_ppl
        / max(0.001, equitile_result.val_ppl),
        "parameter_efficiency": (nanogpt_result.val_ppl / equitile_result.val_ppl)
        / (nanogpt_params / max(1, equitile_params)),
    }

    print("\nParameter Count:")
    print(f"  NanoGPT:  {nanogpt_params:,}")
    print(f"  EquiTile: {equitile_params:,}")

    print("\nValidation Perplexity (lower is better):")
    print(f"  NanoGPT:  {nanogpt_result.val_ppl:.2f}")
    print(f"  EquiTile: {equitile_result.val_ppl:.2f}")

    print("\nTraining Throughput (tokens/sec):")
    print(f"  NanoGPT:  {nanogpt_result.tokens_per_sec:.0f}")
    print(f"  EquiTile: {equitile_result.tokens_per_sec:.0f}")

    print("\nTraining Time:")
    print(f"  NanoGPT:  {nanogpt_result.training_time_sec:.1f}s")
    print(f"  EquiTile: {equitile_result.training_time_sec:.1f}s")

    print("\nEquiTile Advantages:")
    print(f"  Speedup:              {results['equitile_speedup']:.2f}x")
    print(f"  Throughput Gain:      {results['equitile_throughput_gain']:.2f}x")
    print(f"  PPL Improvement:      {results['equitile_ppl_improvement']:.2f}x")
    print(f"  Parameter Efficiency: {results['parameter_efficiency']:.2f}x")

    return results


def run_benchmark_comparison(
    model_configs: list[dict[str, Any]],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    device: str = "cuda",
) -> list[BenchmarkResult]:
    """Run benchmark comparison across multiple model configurations.

    Parameters
    ----------
    model_configs : list
        List of model configurations to compare
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    epochs : int
        Number of epochs
    device : str
        Device to use

    Returns
    -------
    list
        List of benchmark results
    """
    results = []

    for config in model_configs:
        model_type = config.get("type", "nanogpt")

        if model_type == "nanogpt":
            nanogpt_config = NanoGPTConfig(
                vocab_size=config.get("vocab_size", 1000),
                block_size=config.get("block_size", 256),
                n_layer=config.get("n_layer", 6),
                n_head=config.get("n_head", 6),
                n_embd=config.get("n_embd", 192),
            )
            model = NanoGPTModel(nanogpt_config)
        elif model_type == "equitile":
            from bioplausible.equitile.lm_demo.fast_lm import (
                FastLMConfig,
                FastLMEquiTile,
            )

            equitile_config = FastLMConfig(
                vocab_size=config.get("vocab_size", 1000),
                embed_dim=config.get("embed_dim", 192),
                num_layers=config.get("num_layers", 6),
                neurons_per_tile=config.get("neurons_per_tile", 48),
                tiles_per_layer=config.get("tiles_per_layer", 4),
                mot_k=config.get("mot_k", 2),
            )
            model = FastLMEquiTile(equitile_config)
            model._init_weights()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        print(f"\nBenchmarking {config.get('name', model_type)}...")
        result = benchmark_model(
            model, train_loader, val_loader, epochs=epochs, device=device
        )
        results.append(result)

    return results
