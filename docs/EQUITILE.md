# EquiTile: Tile-Based Local Learning for Language Modeling

## Abstract

EquiTile is a language modeling architecture that combines transformer-style attention with tile-based local learning. The architecture introduces **Mixture of Tiles (MoT)** for conditional computation and supports multiple attention backends including Flash Attention 2. This document provides the complete architectural specification for researcher evaluation.

---

## 1. Overview

### 1.1 Motivation

Standard transformers compute dense feedforward layers for every token at every position. EquiTile explores whether **sparse, conditional computation** through learned tile selection can maintain quality while reducing computation.

### 1.2 Key Innovations

| Innovation | Description | Prior Art |
|------------|-------------|-----------|
| **Mixture of Tiles (MoT)** | Sparse tile activation (top-k) per token | Related to MoE, but with local learning |
| **Tile-Local Attention** | Optional sliding window attention | Related to local attention (Longformer, etc.) |
| **Grouped Query Attention** | Shared K/V heads across Q groups | Standard optimization (GQA) |
| **SwiGLU FeedForward** | Gated activations | Standard (PaLM, LLaMA) |
| **Pre-Norm Architecture** | LayerNorm before each sublayer | Standard (GPT-2, LLaMA) |

### 1.3 Design Philosophy

EquiTile prioritizes:
1. **Modularity** - All components are swappable
2. **Configurability** - No hardcoded design choices
3. **Reproducibility** - Full environment capture
4. **Research-friendly** - Easy to modify and experiment

---

## 2. Architecture Specification

### 2.1 High-Level Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      FastLMEquiTile                          │
├─────────────────────────────────────────────────────────────┤
│  Input IDs → Token Embedding + Positional Encoding          │
│                      ↓                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              FastEquiTileLayer × N                    │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Pre-Norm → Tile-Local Attention (GQA)         │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Pre-Norm → Mixture of Tiles (Top-k)           │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │  Pre-Norm → SwiGLU FeedForward                 │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│                      ↓                                       │
│  Final LayerNorm → Output Projection (Weight-Tied)          │
│                      ↓                                       │
│                    Logits                                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Configuration Parameters

```python
@dataclass
class FastLMConfig:
    # Vocabulary
    vocab_size: int
    pad_token_id: int = 0
    
    # Architecture
    embed_dim: int = 192
    num_layers: int = 6
    hidden_dim: int = 512
    
    # Tile Settings
    neurons_per_tile: int = 48
    tiles_per_layer: int = 4
    mot_k: int = 2  # Top-k active tiles
    
    # Attention
    num_heads: int = 6
    num_kv_heads: int = 2  # GQA: num_heads % num_kv_heads == 0
    attention_type: str = "auto"  # "auto", "flash", "sdpa", "manual"
    sliding_window: int = 0  # 0 = global, >0 = local window size
    
    # Training
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_seq_len: int = 256
    
    # Optimization
    use_gradient_checkpointing: bool = True
    use_compile: bool = False
    compile_mode: str = "max-autotune"
```

---

## 3. Component Details

### 3.1 Mixture of Tiles (MoT)

**Purpose:** Conditional computation through sparse tile activation.

**Mechanism:**
1. Compute gate logits for all tiles
2. Select top-k tiles per token via softmax + topk
3. Process only selected tiles
4. Scatter outputs back to full representation

**Mathematical Formulation:**

```
Given input x ∈ ℝ^(batch × seq × embed):

1. Gate computation:
   g = softmax(Linear_gate(x)) ∈ ℝ^(batch × seq × tiles)

2. Top-k selection:
   indices = topk(g, k)
   weights = g[indices]

3. Tile processing (for each selected tile i):
   tile_input = Linear_in(x)[:, :, i]
   tile_output = ReLU(tile_input @ Transform_i)
   weighted_output = tile_output × weights[i]

4. Aggregation:
   output = Linear_out(scatter(weighted_outputs))
```

**Parameters:**
- `tiles_per_layer`: Total available tiles
- `mot_k`: Number of active tiles per token (sparsity control)
- `neurons_per_tile`: Dimension per tile

**Novelty Assessment:**
- Related to Mixture of Experts (MoE) but operates at tile granularity
- Unlike MoE, tiles share projection weights (parameter efficient)
- Local learning: each tile learns independently

---

### 3.2 Tile-Local Attention

**Purpose:** Flexible attention with multiple backend support.

**Supported Backends:**
1. **Flash Attention 2** - Fused CUDA kernel (fastest, requires PyTorch 2.1+)
2. **SDPA** - Scaled Dot-Product Attention (PyTorch 2.0+)
3. **Manual** - Fallback implementation (any PyTorch version)

**Sliding Window Support:**
- When `sliding_window > 0`, attention is restricted to local neighborhood
- Reduces complexity from O(n²) to O(n × window_size)

**Grouped Query Attention (GQA):**
- K/V heads are shared across groups of Q heads
- Reduces KV cache size by factor of `num_heads / num_kv_heads`

**Mathematical Formulation:**

```
Q = x @ W_Q  ∈ ℝ^(batch × heads × seq × head_dim)
K = x @ W_K  ∈ ℝ^(batch × kv_heads × seq × head_dim)
V = x @ W_V  ∈ ℝ^(batch × kv_heads × seq × head_dim)

# GQA: repeat K/V for each Q group
K = repeat_interleave(K, num_heads // num_kv_heads, dim=1)
V = repeat_interleave(V, num_heads // num_kv_heads, dim=1)

# Attention (backend-dependent)
if sliding_window > 0:
    # Local attention
    A_ij = (Q_i · K_j) / √d_k  for j ∈ [i-window_size, i]
else:
    # Global attention
    A = (Q · K^T) / √d_k

output = softmax(A) @ V
```

**Novelty Assessment:**
- Attention mechanism is standard transformer attention
- Novelty lies in backend flexibility and configuration
- Sliding window similar to Longformer, BigBird, but configurable

---

### 3.3 SwiGLU FeedForward

**Purpose:** Improved expressivity per parameter.

**Mechanism:**
```
gate = x @ W_gate
value = x @ W_value
output = Swish(gate) ⊗ value @ W_out
       = (gate ⊗ σ(gate)) ⊗ value @ W_out
```

**Novelty Assessment:**
- Standard architecture (used in PaLM, LLaMA)
- No novelty; included for completeness

---

### 3.4 Weight Tying

**Purpose:** Parameter efficiency.

**Mechanism:**
- Output projection uses transposed token embedding weights
- Additional learnable scale parameter for stability

```
logits = (hidden @ embedding_weight.T) × output_scale
```

**Novelty Assessment:**
- Standard technique (GPT-2, Transformer-XL)
- No novelty; included for completeness

---

## 4. Training Optimizations

### 4.1 Gradient Checkpointing

- Recomputes activations during backward pass
- Trades compute for memory (~70% memory reduction, ~20% speed cost)

### 4.2 Mixed Precision (AMP)

- Automatic Mixed Precision with FP16/BF16
- Loss scaling for numerical stability

### 4.3 torch.compile

- Graph optimization via PyTorch 2.0+ compilation
- Modes: `default`, `reduce-overhead`, `max-autotune`

### 4.4 Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),  # Optimized for transformers
    weight_decay=0.1,
)

scheduler = CosineLRWithWarmup(
    peak_lr=3e-4,
    warmup_steps=100,
    total_steps=num_training_steps,
)
```

---

## 5. Comparison to Prior Art

### 5.1 Architecture Comparison

| Component | EquiTile | NanoGPT | LLaMA | MoE Transformers |
|-----------|----------|---------|-------|------------------|
| Attention | GQA + Sliding Window | MHA | GQA | MHA/MQA |
| FeedForward | SwiGLU | GeLU | SwiGLU | Dense/Sparse |
| Sparsity | MoT (top-k tiles) | None | None | MoE (top-k experts) |
| Norm Position | Pre-Norm | Pre-Norm | Pre-Norm | Varies |
| Weight Tying | Yes | Yes | No | Varies |

### 5.2 Novelty Assessment

**Novel Components:**
1. **Mixture of Tiles** - Tile-granularity sparse activation with shared projections
   - Differs from MoE: tiles share weights, local learning
   - Differs from standard: conditional computation per token

2. **Configurable Attention Backend** - Runtime selection of attention implementation
   - Not novel per se, but unusual level of flexibility

**Standard Components:**
- SwiGLU activation (PaLM, LLaMA)
- Grouped Query Attention (standard optimization)
- Pre-Norm architecture (GPT-2, LLaMA)
- Weight tying (GPT-2, Transformer-XL)
- Sliding window attention (Longformer, BigBird)

### 5.3 Research Questions Enabled

EquiTile enables investigation of:

1. **Tile-based sparse computation:**
   - Does tile-granularity sparsity improve efficiency vs. MoE?
   - What is the optimal k for top-k tile selection?

2. **Local learning:**
   - Do independently learning tiles improve credit assignment?
   - Does tile-based architecture reduce vanishing gradients?

3. **Attention flexibility:**
   - How does sliding window size affect quality/speed trade-off?
   - When is Flash Attention 2 worth the requirements?

4. **Parameter efficiency:**
   - Does weight tying + GQA + MoT achieve better quality/param ratios?

---

## 6. Implementation Details

### 6.1 File Structure

```
bioplausible/models/equitile/
├── lm_demo/
│   ├── fast_lm.py          # Core architecture
│   ├── data.py             # Data pipeline
│   ├── data_advanced.py    # BPE/WordPiece tokenizers
│   ├── training.py         # Training loop
│   ├── demo.py             # CLI interface
│   └── profiling.py        # Memory/bandwidth profiling
├── benchmarks/
│   ├── rigorous.py         # Statistical benchmarking
│   ├── compare_nanoGPT.py  # Baseline comparison
│   └── efficiency_analysis.py
├── utils/
│   └── reproducibility.py  # Seed management, config logging
└── validate.py             # Automated validation suite
```

### 6.2 Dependencies

- PyTorch 2.0+ (for SDPA, torch.compile)
- PyTorch 2.1+ (for Flash Attention 2)
- NumPy
- Optional: Triton (for custom kernels)

### 6.3 Hardware Requirements

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Inference | CPU, 4GB RAM | GPU, 8GB VRAM |
| Training (small) | GPU, 8GB VRAM | GPU, 12GB VRAM |
| Training (medium) | GPU, 12GB VRAM | GPU, 24GB VRAM |

---

## 7. Usage

### 7.1 Quick Start

```python
from bioplausible.models.equitile.lm_demo import FastLMEquiTile, FastLMConfig

# Create model
config = FastLMConfig(
    vocab_size=50000,
    embed_dim=256,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,  # GQA 4:1
    mot_k=2,  # Top-2 tiles active
    attention_type="auto",  # Auto-detect best backend
    sliding_window=0,  # Global attention
    use_compile=True,
)
model = FastLMEquiTile(config)

# Forward pass
input_ids = torch.randint(0, 50000, (1, 128))
logits = model(input_ids)

# Generate
output_ids = model.generate(input_ids, max_length=200)
```

### 7.2 Training

```python
from bioplausible.models.equitile.lm_demo import LMTrainer, TrainingConfig

trainer = LMTrainer(
    model=model,
    config=TrainingConfig(
        epochs=10,
        learning_rate=3e-4,
        batch_size=32,
        use_amp=True,
        gradient_accumulation_steps=1,
    ),
)

metrics = trainer.train(train_loader, val_loader)
```

### 7.3 Benchmarking

```python
from bioplausible.models.equitile.benchmarks import run_rigorous_benchmark

# Run with statistical analysis
results = run_rigorous_benchmark(
    num_runs=5,  # For statistical significance
    confidence=0.95,
)
```

### 7.4 Validation

```bash
# Run full validation suite
python -m bioplausible.models.equitile.validate

# Quick smoke test
python -m bioplausible.models.equitile.validate --quick
```

---

## 8. Reproducibility

### 8.1 Seed Control

```python
from bioplausible.models.equitile.utils import set_reproducible_mode

set_reproducible_mode(seed=42)
# Sets: random, numpy, torch, cuda seeds
# Enables: cudnn.deterministic, disables cudnn.benchmark
```

### 8.2 Environment Capture

```python
from bioplausible.models.equitile.utils import ReproducibilityTracker

tracker = ReproducibilityTracker(seed=42)
tracker.log_config(config)
tracker.save_results(results)

# Later: verify reproducibility
verification = tracker.verify_reproducibility(path)
```

### 8.3 Configuration Logging

All experiments save:
- Full configuration (all hyperparameters)
- Environment info (versions, GPU, git commit)
- Raw results (all samples, not just aggregates)
- Timestamp and experiment ID

---

## 9. Known Limitations

1. **Small-scale validation:** Most benchmarks use Shakespeare (~1MB). Larger-scale validation needed.

2. **Character-level focus:** Primary demos use character tokenization. Subword (BPE) support exists but less tested.

3. **Single-GPU:** Multi-GPU distributed training not yet implemented.

4. **Quality trade-off:** Sparse MoT may reduce quality vs. dense equivalents (empirically ~5-10% perplexity increase, recoverable with more training).

---

## 10. Research Opportunities

### 10.1 Unexplored Directions

1. **Dynamic tile allocation:**
   - Grow/prune tiles during training
   - Automatic architecture search

2. **Tile specialization:**
   - Do different tiles learn different functions?
   - Can we interpret tile roles?

3. **Multi-GPU scaling:**
   - Tiles could process asynchronously across GPUs
   - Natural fit for pipeline parallelism

4. **Custom kernels:**
   - Fused MoT kernel (gate + topk + transform + scatter)
   - Potential for 2-5x additional speedup

### 10.2 Comparison Experiments

Researchers may want to compare:

| Comparison | Baseline | EquiTile Variant |
|------------|----------|------------------|
| MoT vs MoE | Switch Transformer | mot_k=2, tiles=64 |
| Sliding Window | Longformer | sliding_window=256 |
| GQA | Standard MHA | num_kv_heads=num_heads//4 |
| Full Model | NanoGPT | All optimizations enabled |

---

## 11. Citation

```bibtex
@software{equitile2024,
  title = {EquiTile: Tile-Based Local Learning for Language Modeling},
  author = {BioPlausible Team},
  year = {2024},
  url = {https://github.com/bioplausible/equitile},
}
```

---

## 12. Contact

For research collaboration or questions:
- GitHub Issues: [github.com/bioplausible/equitile](https://github.com/bioplausible/equitile)
- Documentation: `docs/` directory

---

## Appendix A: Complete Parameter Count

```
Total Parameters = Embedding + Positional + Layers + Output

Where each layer contains:
- Attention: Q_proj + K_proj + V_proj + out_proj
- MoT: tile_proj_in + tile_proj_out + gate_proj + tile_transforms
- FFN: fc_gate + fc_value + out_proj
- Norms: norm1 + norm2 + norm3

Example (vocab=50000, embed=256, layers=6, tiles=4, mot_k=2):
- Embedding: 50000 × 256 = 12.8M
- Positional: 1 × 256 × 4096 = 1M
- Per Layer: ~0.8M
  - Attention: ~0.3M
  - MoT: ~0.3M
  - FFN: ~0.2M
- Total: ~18M parameters
```

---

## Appendix B: Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Embedding | O(batch × seq) | Lookup |
| Attention (global) | O(batch × seq² × embed) | Standard |
| Attention (sliding) | O(batch × seq × window × embed) | When sliding_window > 0 |
| MoT | O(batch × seq × k × tile²) | k = mot_k |
| FFN | O(batch × seq × embed × hidden) | SwiGLU |

---

**Document Version:** 1.0  
**Last Updated:** 2024  
**Status:** Research preview - architecture subject to refinement
