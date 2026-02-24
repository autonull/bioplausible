"""
Rigorous Benchmark Suite for EquiTile
======================================

Scientific-grade benchmarking with:
- Statistical significance testing
- Multiple runs for variance analysis
- Proper controls and baselines
- Comprehensive metrics
- Reproducible configurations

Example
-------
>>> from bioplausible.models.equitile.benchmarks.rigorous import RigorousBenchmark
>>> benchmark = RigorousBenchmark(num_runs=5, confidence=0.95)
>>> results = benchmark.run_comparison()
>>> benchmark.report(results)
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from bioplausible.models.equitile.benchmarks.compare_nanoGPT import (
    NanoGPTConfig,
    NanoGPTModel,
    benchmark_model,
)
from bioplausible.models.equitile.lm_demo.fast_lm import FastLMEquiTile, FastLMConfig
from bioplausible.models.equitile.lm_demo.data import create_shakespeare_dataset


# =============================================================================
# Reproducibility Framework
# =============================================================================

def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_system_info() -> Dict[str, str]:
    """Get system information for reproducibility."""
    return {
        "python_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# Statistical Analysis
# =============================================================================

@dataclass
class StatisticalMetrics:
    """Statistical metrics for benchmark results."""
    mean: float
    std: float
    std_error: float
    confidence_interval_95: Tuple[float, float]
    min: float
    max: float
    median: float
    n_runs: int
    
    @classmethod
    def from_samples(cls, samples: List[float], confidence: float = 0.95) -> 'StatisticalMetrics':
        """Compute statistical metrics from samples."""
        n = len(samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1) if n > 1 else 0.0
        std_error = std / math.sqrt(n) if n > 0 else 0.0
        
        # t-distribution for confidence interval
        if n > 1:
            t_value = 1.96 if n >= 30 else 2.571 if n >= 5 else 4.303  # Approximate t-values
            margin = t_value * std_error
            ci = (mean - margin, mean + margin)
        else:
            ci = (mean, mean)
        
        return cls(
            mean=mean,
            std=std,
            std_error=std_error,
            confidence_interval_95=ci,
            min=float(np.min(samples)),
            max=float(np.max(samples)),
            median=float(np.median(samples)),
            n_runs=n,
        )


def compute_speedup_with_uncertainty(
    baseline_metrics: StatisticalMetrics,
    experimental_metrics: StatisticalMetrics,
) -> Tuple[float, float, float]:
    """Compute speedup ratio with uncertainty propagation.
    
    Calculates Experimental / Baseline ratio.

    Returns
    -------
    tuple
        (speedup, lower_bound, upper_bound)
    """
    # Calculate speedup as Experimental / Baseline
    speedup = experimental_metrics.mean / baseline_metrics.mean
    
    # Error propagation for ratio
    relative_error_baseline = baseline_metrics.std_error / baseline_metrics.mean if baseline_metrics.mean > 0 else 0
    relative_error_experimental = experimental_metrics.std_error / experimental_metrics.mean if experimental_metrics.mean > 0 else 0
    
    combined_relative_error = math.sqrt(relative_error_baseline**2 + relative_error_experimental**2)
    absolute_error = speedup * combined_relative_error
    
    return speedup, speedup - 1.96 * absolute_error, speedup + 1.96 * absolute_error


# =============================================================================
# Benchmark Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for rigorous benchmarking."""
    # Reproducibility
    seed: int = 42
    num_runs: int = 5  # Number of runs for statistical significance
    
    # Dataset
    task: str = "shakespeare"
    seq_length: int = 128
    batch_size: int = 32
    
    # Training
    epochs: int = 3
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    
    # Model
    embed_dim: int = 192
    num_layers: int = 6
    num_heads: int = 6
    num_kv_heads: int = 2
    
    # Optimization
    attention_type: str = "auto"
    sliding_window: int = 0
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    use_gradient_checkpointing: bool = True
    use_amp: bool = True
    
    # Hardware
    device: str = "auto"
    
    # Statistical
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


# =============================================================================
# Rigorous Benchmark Runner
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a rigorous benchmark run."""
    model_name: str
    config: Dict[str, Any]
    
    # Performance metrics (with statistics)
    throughput_stats: StatisticalMetrics
    time_per_epoch_stats: StatisticalMetrics
    memory_mb: float
    
    # Quality metrics
    final_train_loss: float
    val_loss: float
    val_ppl: float
    
    # System info
    system_info: Dict[str, str]
    
    # Raw data
    raw_throughput_samples: List[float] = field(default_factory=list)
    raw_time_samples: List[float] = field(default_factory=list)


class RigorousBenchmark:
    """Rigorous benchmark runner with statistical analysis.
    
    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark configuration
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None) -> None:
        self.config = config or BenchmarkConfig()
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def run_single_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> BenchmarkResult:
        """Run benchmark for a single model with multiple runs."""
        device = torch.device(self.config.device if self.config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model = model.to(device)
        
        throughput_samples = []
        time_samples = []
        
        for run in range(self.config.num_runs):
            # Set seed for this run
            set_all_seeds(self.config.seed + run)
            
            # Create fresh optimizer for each run
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                betas=(0.9, 0.95),
                weight_decay=0.1,
            )
            
            # Warmup
            model.train()
            for _ in range(3):
                for batch in train_loader:
                    input_ids, targets = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    output = model(input_ids)
                    if isinstance(output, tuple):
                        loss = output[1] if output[1] is not None else torch.nn.functional.cross_entropy(
                            output[0].view(-1, output[0].size(-1)), targets.view(-1)
                        )
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        )
                    loss.backward()
                    optimizer.step()
                    break
            
            # Measure
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            epoch_start = time.time()
            total_tokens = 0
            
            model.train()
            for epoch in range(self.config.epochs):
                for batch in train_loader:
                    input_ids, targets = batch[0].to(device), batch[1].to(device)
                    optimizer.zero_grad()
                    
                    output = model(input_ids)
                    if isinstance(output, tuple):
                        loss = output[1] if output[1] is not None else torch.nn.functional.cross_entropy(
                            output[0].view(-1, output[0].size(-1)), targets.view(-1)
                        )
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            output.view(-1, output.size(-1)), targets.view(-1)
                        )
                    loss.backward()
                    optimizer.step()
                    
                    total_tokens += input_ids.numel()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            epoch_elapsed = time.time() - epoch_start
            throughput = total_tokens / epoch_elapsed
            
            throughput_samples.append(throughput)
            time_samples.append(epoch_elapsed)
        
        # Compute statistics
        throughput_stats = StatisticalMetrics.from_samples(
            throughput_samples, self.config.confidence_level
        )
        time_stats = StatisticalMetrics.from_samples(
            time_samples, self.config.confidence_level
        )
        
        # Final evaluation
        model.eval()
        val_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, targets = batch[0].to(device), batch[1].to(device)
                output = model(input_ids)
                if isinstance(output, tuple):
                    loss = output[1] if output[1] is not None else torch.nn.functional.cross_entropy(
                        output[0].view(-1, output[0].size(-1)), targets.view(-1)
                    )
                else:
                    loss = torch.nn.functional.cross_entropy(
                        output.view(-1, output.size(-1)), targets.view(-1)
                    )
                val_loss += loss.item()
                n_batches += 1
        
        val_loss /= max(1, n_batches)
        val_ppl = math.exp(val_loss)
        
        # Memory
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0
        
        return BenchmarkResult(
            model_name=model_name,
            config=self.config.to_dict(),
            throughput_stats=throughput_stats,
            time_per_epoch_stats=time_stats,
            memory_mb=memory_mb,
            final_train_loss=loss.item(),
            val_loss=val_loss,
            val_ppl=val_ppl,
            system_info=get_system_info(),
            raw_throughput_samples=throughput_samples,
            raw_time_samples=time_samples,
        )
    
    def run_comparison(self) -> Dict[str, BenchmarkResult]:
        """Run comparison between EquiTile and NanoGPT."""
        print("=" * 70)
        print("Rigorous Benchmark: EquiTile vs NanoGPT")
        print("=" * 70)
        print(f"Number of runs: {self.config.num_runs}")
        print(f"Confidence level: {self.config.confidence_level * 100:.0f}%")
        print(f"Device: {self.config.device}")
        print()
        
        # Create dataset (same for both models)
        print("Loading dataset...")
        train_loader, val_loader, tokenizer = create_shakespeare_dataset(
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            num_workers=0,
        )
        vocab_size = tokenizer.vocab_size
        print(f"Vocabulary size: {vocab_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print()
        
        results = {}
        
        # NanoGPT
        print("-" * 70)
        print("Benchmarking NanoGPT...")
        print("-" * 70)
        nanogpt_config = NanoGPTConfig(
            vocab_size=vocab_size,
            block_size=self.config.seq_length,
            n_layer=self.config.num_layers,
            n_head=self.config.num_heads,
            n_embd=self.config.embed_dim,
        )
        nanogpt = NanoGPTModel(nanogpt_config)
        nanogpt_params = sum(p.numel() for p in nanogpt.parameters())
        print(f"Parameters: {nanogpt_params:,}")
        
        results['nanogpt'] = self.run_single_model(
            nanogpt, "NanoGPT", train_loader, val_loader
        )
        print(f"Throughput: {results['nanogpt'].throughput_stats.mean:,.0f} ± {results['nanogpt'].throughput_stats.std:.0f} tok/s")
        print()
        
        # EquiTile
        print("-" * 70)
        print("Benchmarking EquiTile...")
        print("-" * 70)
        equitile_config = FastLMConfig(
            vocab_size=vocab_size,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            num_kv_heads=self.config.num_kv_heads,
            attention_type=self.config.attention_type,
            sliding_window=self.config.sliding_window,
            use_compile=self.config.use_compile,
            compile_mode=self.config.compile_mode,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
        )
        equitile = FastLMEquiTile(equitile_config)
        equitile_params = sum(p.numel() for p in equitile.parameters())
        print(f"Parameters: {equitile_params:,}")
        
        results['equitile'] = self.run_single_model(
            equitile, "EquiTile", train_loader, val_loader
        )
        print(f"Throughput: {results['equitile'].throughput_stats.mean:,.0f} ± {results['equitile'].throughput_stats.std:.0f} tok/s")
        print()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: Dict[str, BenchmarkResult]) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.results_dir / f"benchmark_{timestamp}.json"
        
        data = {
            "config": self.config.to_dict(),
            "system_info": get_system_info(),
            "results": {
                name: {
                    "model_name": r.model_name,
                    "throughput_stats": {
                        "mean": r.throughput_stats.mean,
                        "std": r.throughput_stats.std,
                        "std_error": r.throughput_stats.std_error,
                        "ci_95": r.throughput_stats.confidence_interval_95,
                        "min": r.throughput_stats.min,
                        "max": r.throughput_stats.max,
                        "median": r.throughput_stats.median,
                        "n_runs": r.throughput_stats.n_runs,
                    },
                    "time_stats": {
                        "mean": r.time_per_epoch_stats.mean,
                        "std": r.time_per_epoch_stats.std,
                        "ci_95": r.time_per_epoch_stats.confidence_interval_95,
                    },
                    "memory_mb": r.memory_mb,
                    "val_loss": r.val_loss,
                    "val_ppl": r.val_ppl,
                    "raw_throughput": r.raw_throughput_samples,
                    "raw_time": r.raw_time_samples,
                }
                for name, r in results.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to {filepath}")
    
    def report(self, results: Dict[str, BenchmarkResult]) -> str:
        """Generate comprehensive report."""
        nanogpt = results['nanogpt']
        equitile = results['equitile']
        
        # Compute speedup with uncertainty
        speedup, ci_lower, ci_upper = compute_speedup_with_uncertainty(
            nanogpt.throughput_stats,
            equitile.throughput_stats,
        )
        
        # Statistical significance test
        t_stat = (nanogpt.throughput_stats.mean - equitile.throughput_stats.mean) / math.sqrt(
            nanogpt.throughput_stats.std_error**2 + equitile.throughput_stats.std_error**2
        )
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
        
        lines = [
            "=" * 70,
            "RIGOROUS BENCHMARK REPORT",
            "=" * 70,
            "",
            "CONFIGURATION",
            "-" * 70,
            f"Number of runs: {self.config.num_runs}",
            f"Confidence level: {self.config.confidence_level * 100:.0f}%",
            f"Sequence length: {self.config.seq_length}",
            f"Batch size: {self.config.batch_size}",
            "",
            "THROUGHPUT RESULTS",
            "-" * 70,
            f"NanoGPT:  {nanogpt.throughput_stats.mean:,.0f} ± {nanogpt.throughput_stats.std:.0f} tok/s",
            f"          95% CI: [{nanogpt.throughput_stats.confidence_interval_95[0]:,.0f}, {nanogpt.throughput_stats.confidence_interval_95[1]:,.0f}]",
            f"EquiTile: {equitile.throughput_stats.mean:,.0f} ± {equitile.throughput_stats.std:.0f} tok/s",
            f"          95% CI: [{equitile.throughput_stats.confidence_interval_95[0]:,.0f}, {equitile.throughput_stats.confidence_interval_95[1]:,.0f}]",
            "",
            "SPEEDUP ANALYSIS",
            "-" * 70,
            f"Speedup: {speedup:.2f}x (95% CI: [{ci_lower:.2f}x, {ci_upper:.2f}x])",
            f"t-statistic: {t_stat:.2f}",
            f"p-value: {p_value:.6f}",
            f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)",
            "",
            "QUALITY METRICS",
            "-" * 70,
            f"NanoGPT Val PPL: {nanogpt.val_ppl:.2f}",
            f"EquiTile Val PPL: {equitile.val_ppl:.2f}",
            f"PPL Ratio: {nanogpt.val_ppl / equitile.val_ppl:.2f}x",
            "",
            "MEMORY EFFICIENCY",
            "-" * 70,
            f"NanoGPT: {nanogpt.memory_mb:.0f} MB",
            f"EquiTile: {equitile.memory_mb:.0f} MB",
            f"Tokens/sec/GB - NanoGPT: {nanogpt.throughput_stats.mean / max(1, nanogpt.memory_mb/1024):,.0f}",
            f"Tokens/sec/GB - EquiTile: {equitile.throughput_stats.mean / max(1, equitile.memory_mb/1024):,.0f}",
            "",
            "CONCLUSION",
            "-" * 70,
        ]
        
        if p_value < 0.05:
            if speedup > 1.0:
                lines.append(f"✓ EquiTile is STATISTICALLY SIGNIFICANTLY faster than NanoGPT")
                lines.append(f"  Speedup: {speedup:.2f}x (p < 0.05)")
            else:
                lines.append(f"✗ EquiTile is SLOWER than NanoGPT")
                lines.append(f"  Speedup: {speedup:.2f}x (NanoGPT is {1/speedup:.2f}x faster) (p < 0.05)")
        else:
            lines.append("~ Difference is NOT statistically significant")
            lines.append(f"  Speedup: {speedup:.2f}x (p = {p_value:.4f})")
        
        if equitile.val_ppl <= nanogpt.val_ppl * 1.1:
            lines.append("✓ EquiTile achieves COMPARABLE quality (within 10%)")
        
        lines.append("")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        print(report)
        
        # Save report
        report_path = self.results_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report


# =============================================================================
# CLI Interface
# =============================================================================

def run_rigorous_benchmark(
    num_runs: int = 5,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 32,
    seq_length: int = 128,
    device: str = "auto",
) -> Dict[str, BenchmarkResult]:
    """Run rigorous benchmark with specified parameters.
    
    Parameters
    ----------
    num_runs : int
        Number of runs for statistical significance
    seed : int
        Random seed
    epochs : int
        Training epochs
    batch_size : int
        Batch size
    seq_length : int
        Sequence length
    device : str
        Device to use
    
    Returns
    -------
    dict
        Benchmark results
    """
    config = BenchmarkConfig(
        num_runs=num_runs,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        seq_length=seq_length,
        device=device,
    )
    
    benchmark = RigorousBenchmark(config)
    results = benchmark.run_comparison()
    benchmark.report(results)
    
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rigorous benchmarks for EquiTile vs NanoGPT.")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for statistical significance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")

    args = parser.parse_args()

    run_rigorous_benchmark(
        num_runs=args.num_runs,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        device=args.device,
    )
