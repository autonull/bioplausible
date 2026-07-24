"""
Efficiency Analysis for EquiTile LM
====================================

Analyzes parameter and FLOP efficiency:
- Parameter efficiency: quality per parameter
- FLOP efficiency: quality per FLOP
- Memory efficiency: tokens/sec/GB
- Compute utilization analysis

Example
-------
>>> from bioplausible.equitile.benchmarks import analyze_parameter_efficiency
>>> result = analyze_parameter_efficiency(model, train_loader, val_loader)
>>> print(f"Parameter efficiency score: {result.efficiency_score:.2f}")
"""

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn


@dataclass
class ParameterEfficiencyResult:
    """Results from parameter efficiency analysis."""

    model_name: str
    parameter_count: int
    val_perplexity: float
    efficiency_score: float  # Lower PPL per million params is better
    params_per_layer: List[int]
    embedding_params: int
    attention_params: int
    mlp_params: int


@dataclass
class FLOPEfficiencyResult:
    """Results from FLOP efficiency analysis."""

    model_name: str
    flops_per_token: int
    val_perplexity: float
    efficiency_score: float  # Lower PPL per GFLOP is better
    theoretical_flops: int
    measured_flops: int


@dataclass
class MemoryEfficiencyResult:
    """Results from memory efficiency analysis."""

    model_name: str
    memory_mb: float
    tokens_per_sec: float
    tokens_per_sec_per_gb: float
    peak_memory_mb: float


class EfficiencyAnalyzer:
    """Analyzes model efficiency metrics.

    Provides comprehensive analysis of:
    - Parameter distribution
    - FLOP computation
    - Memory usage
    - Throughput efficiency
    """

    def __init__(self, model: nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = torch.device(
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component.

        Returns
        -------
        dict
            Parameter counts by component
        """
        params = {
            "total": 0,
            "embedding": 0,
            "attention": 0,
            "mlp": 0,
            "normalization": 0,
            "other": 0,
        }

        for name, module in self.model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                param_count = param.numel()
                params["total"] += param_count

                if isinstance(module, nn.Embedding):
                    params["embedding"] += param_count
                elif isinstance(module, (nn.Linear,)) and any(
                    k in name.lower() for k in ["q", "k", "v", "attn", "attention"]
                ):
                    params["attention"] += param_count
                elif isinstance(module, (nn.Linear,)) and any(
                    k in name.lower() for k in ["fc", "mlp", "feedforward", "proj"]
                ):
                    params["mlp"] += param_count
                elif isinstance(module, nn.LayerNorm):
                    params["normalization"] += param_count
                else:
                    params["other"] += param_count

        return params

    def estimate_flops(self, seq_length: int, batch_size: int = 1) -> int:
        """Estimate FLOPs for forward pass.

        Parameters
        ----------
        seq_length : int
            Sequence length
        batch_size : int
            Batch size

        Returns
        -------
        int
            Estimated FLOPs
        """
        # Get model config
        if hasattr(self.model, "config"):
            config = self.model.config
            embed_dim = getattr(config, "embed_dim", getattr(config, "n_embd", 256))
            num_layers = getattr(config, "num_layers", getattr(config, "n_layer", 6))
            num_heads = getattr(config, "num_heads", getattr(config, "n_head", 8))
            vocab_size = getattr(config, "vocab_size", 1000)
        else:
            embed_dim = 256
            num_layers = 6
            num_heads = 8
            vocab_size = 1000

        # Embedding lookup
        embedding_flops = batch_size * seq_length * embed_dim

        # Attention per layer
        # QKV projection: 3 * (batch * seq * embed^2)
        # Attention scores: batch * heads * seq^2 * (embed/heads)
        # Output projection: batch * seq * embed^2
        head_dim = embed_dim // num_heads
        attention_flops_per_layer = (
            3 * batch_size * seq_length * embed_dim * embed_dim  # QKV
            + 2
            * batch_size
            * num_heads
            * seq_length
            * seq_length
            * head_dim  # Attention
            + batch_size * seq_length * embed_dim * embed_dim  # Output
        )

        # MLP per layer (with SwiGLU or GELU)
        hidden_dim = 4 * embed_dim
        mlp_flops_per_layer = (
            2 * batch_size * seq_length * embed_dim * hidden_dim  # FC layers
            + batch_size * seq_length * hidden_dim  # Activation
        )

        # Total
        total_flops = (
            embedding_flops
            + num_layers * (attention_flops_per_layer + mlp_flops_per_layer)
            + batch_size * seq_length * embed_dim * vocab_size  # Output projection
        )

        return int(total_flops)

    def measure_throughput(
        self,
        seq_length: int = 256,
        batch_size: int = 32,
        warmup_steps: int = 10,
        measure_steps: int = 50,
    ) -> float:
        """Measure inference throughput.

        Parameters
        ----------
        seq_length : int
            Sequence length
        batch_size : int
            Batch size
        warmup_steps : int
            Warmup steps
        measure_steps : int
            Steps to measure

        Returns
        -------
        float
            Tokens per second
        """
        import time

        self.model.eval()
        self.model = self.model.to(self.device)

        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_steps):
                if hasattr(self.model, "forward"):
                    self.model(input_ids)

        # Measure
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.time()

        with torch.no_grad():
            for _ in range(measure_steps):
                if hasattr(self.model, "forward"):
                    self.model(input_ids)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.time() - start_time
        total_tokens = batch_size * seq_length * measure_steps

        return total_tokens / elapsed

    def measure_memory(self, batch_size: int = 32, seq_length: int = 256) -> float:
        """Measure peak memory usage.

        Parameters
        ----------
        batch_size : int
            Batch size
        seq_length : int
            Sequence length

        Returns
        -------
        float
            Peak memory in MB
        """
        if self.device.type != "cuda":
            return 0.0

        self.model.train()
        self.model = self.model.to(self.device)

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats(self.device)

        # Create input and run forward/backward
        input_ids = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)
        targets = torch.randint(0, 1000, (batch_size, seq_length), device=self.device)

        if hasattr(self.model, "train_step"):
            self.model.train_step(input_ids, targets)
        else:
            output = self.model(input_ids)
            loss = nn.functional.cross_entropy(
                output.view(-1, output.size(-1)), targets.view(-1)
            )
            loss.backward()

        peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024

        return peak_memory


def analyze_parameter_efficiency(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = "auto",
) -> ParameterEfficiencyResult:
    """Analyze parameter efficiency.

    Parameters
    ----------
    model : nn.Module
        Model to analyze
    val_loader : DataLoader
        Validation data
    device : str
        Device to use

    Returns
    -------
    ParameterEfficiencyResult
        Analysis results
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Count parameters
    analyzer = EfficiencyAnalyzer(model, device)
    param_counts = analyzer.count_parameters()

    # Evaluate perplexity
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if hasattr(model, "forward") and hasattr(model, "compute_loss"):
                logits = model(input_ids)
                loss = model.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            total_loss += loss.item()
            n_batches += 1

    val_loss = total_loss / max(1, n_batches)
    val_ppl = torch.exp(torch.tensor(val_loss)).item()

    # Efficiency score: PPL per million parameters
    params_millions = param_counts["total"] / 1e6
    efficiency_score = val_ppl / max(0.001, params_millions)

    return ParameterEfficiencyResult(
        model_name=model.__class__.__name__,
        parameter_count=param_counts["total"],
        val_perplexity=val_ppl,
        efficiency_score=efficiency_score,
        params_per_layer=[param_counts["total"] // 10],  # Approximate
        embedding_params=param_counts["embedding"],
        attention_params=param_counts["attention"],
        mlp_params=param_counts["mlp"],
    )


def analyze_flop_efficiency(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    seq_length: int = 256,
    device: str = "auto",
) -> FLOPEfficiencyResult:
    """Analyze FLOP efficiency.

    Parameters
    ----------
    model : nn.Module
        Model to analyze
    val_loader : DataLoader
        Validation data
    seq_length : int
        Sequence length
    device : str
        Device to use

    Returns
    -------
    FLOPEfficiencyResult
        Analysis results
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Estimate FLOPs
    analyzer = EfficiencyAnalyzer(model, device)
    flops_per_token = analyzer.estimate_flops(seq_length, batch_size=1) // seq_length

    # Evaluate perplexity
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for input_ids, targets in val_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            if hasattr(model, "forward") and hasattr(model, "compute_loss"):
                logits = model(input_ids)
                loss = model.compute_loss(logits, targets)
            else:
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )

            total_loss += loss.item()
            n_batches += 1

    val_loss = total_loss / max(1, n_batches)
    val_ppl = torch.exp(torch.tensor(val_loss)).item()

    # Efficiency score: PPL per GFLOP
    gflops = flops_per_token / 1e9
    efficiency_score = val_ppl / max(0.001, gflops)

    return FLOPEfficiencyResult(
        model_name=model.__class__.__name__,
        flops_per_token=flops_per_token,
        val_perplexity=val_ppl,
        efficiency_score=efficiency_score,
        theoretical_flops=flops_per_token,
        measured_flops=flops_per_token,
    )


def analyze_memory_efficiency(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    batch_size: int = 32,
    seq_length: int = 256,
    device: str = "auto",
) -> MemoryEfficiencyResult:
    """Analyze memory efficiency.

    Parameters
    ----------
    model : nn.Module
        Model to analyze
    train_loader : DataLoader
        Training data
    batch_size : int
        Batch size
    seq_length : int
        Sequence length
    device : str
        Device to use

    Returns
    -------
    MemoryEfficiencyResult
        Analysis results
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Measure throughput
    analyzer = EfficiencyAnalyzer(model, device)
    tokens_per_sec = analyzer.measure_throughput(seq_length, batch_size)
    memory_mb = analyzer.measure_memory(batch_size, seq_length)

    # Calculate efficiency
    tokens_per_sec_per_gb = (tokens_per_sec / 1000) / max(0.001, memory_mb / 1024)

    return MemoryEfficiencyResult(
        model_name=model.__class__.__name__,
        memory_mb=memory_mb,
        tokens_per_sec=tokens_per_sec,
        tokens_per_sec_per_gb=tokens_per_sec_per_gb,
        peak_memory_mb=memory_mb,
    )


def compare_efficiency(
    models: List[Tuple[str, nn.Module]],
    val_loader: torch.utils.data.DataLoader,
    device: str = "auto",
) -> Dict[str, Any]:
    """Compare efficiency across multiple models.

    Parameters
    ----------
    models : list
        List of (name, model) tuples
    val_loader : DataLoader
        Validation data
    device : str
        Device to use

    Returns
    -------
    dict
        Comparison results
    """
    results = {
        "parameter_efficiency": [],
        "flop_efficiency": [],
        "memory_efficiency": [],
    }

    for name, model in models:
        print(f"\nAnalyzing {name}...")

        # Parameter efficiency
        param_result = analyze_parameter_efficiency(model, val_loader, device)
        results["parameter_efficiency"].append(
            {
                "name": name,
                "params": param_result.parameter_count,
                "val_ppl": param_result.val_perplexity,
                "efficiency_score": param_result.efficiency_score,
            }
        )

        # FLOP efficiency
        flop_result = analyze_flop_efficiency(model, val_loader, device=device)
        results["flop_efficiency"].append(
            {
                "name": name,
                "flops_per_token": flop_result.flops_per_token,
                "val_ppl": flop_result.val_perplexity,
                "efficiency_score": flop_result.efficiency_score,
            }
        )

    return results
