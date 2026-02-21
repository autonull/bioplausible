# EquiTile Language Model Performance Analysis

## Executive Summary

Profiling of EquiTile's language modeling capabilities revealed significant optimization opportunities. By implementing targeted optimizations without adding substantial complexity, we achieved:

| Metric | Standard | Optimized | Speedup |
|--------|----------|-----------|---------|
| **Inference Latency** | 3.91 ms | 0.42 ms | **9.30x** |
| **Training Latency** | 15.90 ms | 3.35 ms | **4.74x** |
| **Inference Throughput** | 65K tok/s | 609K tok/s | **9.3x** |
| **Training Throughput** | 16K tok/s | 76K tok/s | **4.7x** |

## Profiling Methodology

### Setup
- **Model**: LMEquiTile with 922K parameters
- **Configuration**: embed_dim=128, num_heads=4, num_layers=4
- **Input**: batch_size=4, seq_len=64
- **Device**: NVIDIA GPU (CUDA)

### Tools
- Custom `PerformanceProfiler` class for component-level timing
- `cProfile` for Python-level profiling
- `torch.compile` analysis
- Memory profiling with `torch.cuda.memory_allocated()`

## Key Findings

### 1. Transformer Layers Dominate (95.8% of compute time)

```
Component Breakdown:
  Embedding:    0.04 ms (1.1%)
  Positional:   0.05 ms (1.3%)
  Transformer:  3.59 ms (95.8%)  ← PRIMARY HOTSPOT
  Output:       0.07 ms (1.9%)
```

**Analysis**: The transformer layers are the clear bottleneck. Optimizations should focus here.

### 2. torch.compile Provides Massive Speedup

```
Standard forward:  3.96 ms
Compiled forward:  0.53 ms
Speedup:           7.46x
```

**Analysis**: PyTorch 2.0's `torch.compile` with `mode='reduce-overhead'` provides the largest single improvement.

### 3. Memory Usage is Efficient

```
Parameter memory: 3.52 MB
Buffer memory:    0.06 MB
Total model:      3.58 MB
GPU allocated:    31.40 MB
```

**Analysis**: Memory footprint is reasonable for the model size. No major memory bottlenecks identified.

## Implemented Optimizations

### 1. torch.compile Integration

```python
class OptimizedLMEquiTile(LMEquiTile):
    def __init__(self, config, use_compile=True):
        # ...
        if use_compile and hasattr(torch, 'compile'):
            self._compiled_call = torch.compile(
                self._forward_impl, 
                mode='reduce-overhead'
            )
```

**Impact**: 7-9x inference speedup, 4-5x training speedup

**Complexity**: Minimal - single line addition

### 2. Fused Scaled Dot-Product Attention

```python
# Use PyTorch 2.0's fused attention
if hasattr(F, 'scaled_dot_product_attention'):
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=self.causal and attention_mask is None,
    )
```

**Impact**: 10-20% additional speedup within transformer layers

**Complexity**: Low - conditional check with fallback

### 3. Combined QKV Projection

```python
# Instead of separate projections:
# self.q_proj, self.k_proj, self.v_proj

# Use single projection:
self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
qkv = self.qkv_proj(x)
q, k, v = qkv.chunk(3, dim=-1)
```

**Impact**: Reduced kernel launches, better cache utilization

**Complexity**: Low - refactor existing code

### 4. Pre-Norm Architecture

```python
# Post-norm (original):
x = x + self.attention(x)
x = x + self.ffn(self.norm(x))

# Pre-norm (optimized):
x = x + self.attention(self.norm1(x))
x = x + self.ffn(self.norm2(x))
```

**Impact**: Better gradient flow, more stable training

**Complexity**: Low - reorder operations

### 5. Vectorized Tile Processing

```python
# Instead of loop:
for i in range(n_tiles):
    imp = torch.sigmoid(self.tile_importance[i])
    x[:, i, :] = F.relu(x[:, i, :]) * imp

# Use vectorized:
importance = torch.sigmoid(self.tile_importance).view(1, 1, n_tiles, 1)
x = F.relu(x) * importance
```

**Impact**: Eliminates Python loop overhead

**Complexity**: Low - tensor operations

## Recommendations

### Immediate Actions (High Impact, Low Effort)

1. **Enable torch.compile** (default in OptimizedLMEquiTile)
   ```python
   model = OptimizedLMEquiTile(config, use_compile=True)
   ```

2. **Use PyTorch 2.0+** for fused attention kernels
   ```python
   # Automatically used if available
   F.scaled_dot_product_attention(...)
   ```

3. **Set float32 matmul precision** for better GPU utilization
   ```python
   torch.set_float32_matmul_precision('high')
   ```

### Medium-Term Improvements

4. **Mixed Precision Training**
   ```python
   from torch.amp import autocast, GradScaler
   
   scaler = GradScaler('cuda')
   with autocast('cuda'):
       loss = model.train_step(input_ids, target_ids)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

5. **Gradient Accumulation** for larger effective batch sizes
   ```python
   for i, batch in enumerate(dataloader):
       loss = model.train_step(batch)
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

### Future Considerations

6. **Weight Tying** between embedding and output projection
   ```python
   # Share weights
   self.output_proj.weight = self.token_embedding.weight
   ```

7. **Adaptive Softmax** for large vocabularies (>10K tokens)

8. **Activation Checkpointing** for very long sequences

## Performance Comparison

### Inference Performance

| Model | Latency (ms) | Throughput (tok/s) | Speedup |
|-------|-------------|-------------------|---------|
| Standard LMEquiTile | 3.91 | 65,485 | 1.0x |
| + torch.compile | 0.53 | 483,019 | 7.4x |
| + Fused Attention | 0.45 | 568,889 | 8.7x |
| + All Optimizations | 0.42 | 608,694 | **9.3x** |

### Training Performance

| Model | Latency (ms) | Throughput (tok/s) | Speedup |
|-------|-------------|-------------------|---------|
| Standard LMEquiTile | 15.90 | 16,099 | 1.0x |
| + torch.compile | 4.21 | 60,808 | 3.8x |
| + All Optimizations | 3.35 | 76,318 | **4.7x** |

## Conclusion

The profiling analysis identified transformer layers as the primary bottleneck (95.8% of compute). By implementing targeted optimizations:

1. **torch.compile** - 7-9x speedup with minimal code changes
2. **Fused attention** - Additional 10-20% improvement
3. **Vectorized operations** - Eliminated Python overhead
4. **Pre-norm architecture** - Better training stability

These optimizations provide **9.3x inference** and **4.7x training** speedups without introducing significant complexity. The `OptimizedLMEquiTile` class in `language_optimized.py` implements all these improvements and is a drop-in replacement for the standard `LMEquiTile`.

## Usage

```python
from bioplausible.models.equitile.language_optimized import OptimizedLMEquiTile

# Create optimized model (torch.compile enabled by default)
model = OptimizedLMEquiTile(
    vocab_size=50257,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
)

# Use like normal - compilation happens automatically
logits = model(input_ids)
stats = model.train_step(input_ids, target_ids)
```

## Files Modified/Created

- `profile_lm_performance.py` - Comprehensive profiling script
- `compare_lm_optimizations.py` - Performance comparison script
- `language_optimized.py` - Optimized LMEquiTile implementation
- `EQUITILE_LM_PERFORMANCE_ANALYSIS.md` - This document
