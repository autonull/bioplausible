# EquiTile LM Architecture

Technical documentation for the FastLMEquiTile architecture.

## Architecture Overview

FastLMEquiTile combines EquiTile's tile-based local learning with modern transformer optimizations for efficient language modeling.

```
┌─────────────────────────────────────────────────────────────────┐
│                      FastLMEquiTile                              │
├─────────────────────────────────────────────────────────────────┤
│  Input IDs → Token Embedding + Positional Encoding              │
│                    ↓                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              FastEquiTileLayer × N                       │    │
│  │  ┌─────────────────────────────────────────────────┐     │    │
│  │  │  Pre-Norm → Tile-Local Attention (GQA)          │     │    │
│  │  └─────────────────────────────────────────────────┘     │    │
│  │  ┌─────────────────────────────────────────────────┐     │    │
│  │  │  Pre-Norm → Mixture of Tiles (Top-k)            │     │    │
│  │  └─────────────────────────────────────────────────┘     │    │
│  │  ┌─────────────────────────────────────────────────┐     │    │
│  │  │  Pre-Norm → SwiGLU FeedForward                  │     │    │
│  │  └─────────────────────────────────────────────────┘     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                    ↓                                             │
│  Final LayerNorm → Output Projection (Weight-Tied)              │
│                    ↓                                             │
│                   Logits                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Mixture of Tiles (MoT)

The MoT layer implements sparse tile activation for conditional computation.

#### Mathematical Formulation

```
Given input x ∈ ℝ^(batch × seq × embed):

1. Gate computation:
   g = softmax(Linear_gate(x)) ∈ ℝ^(batch × seq × tiles)

2. Top-k selection:
   indices = topk(g, k)  # Select k most important tiles
   weights = g[indices]

3. Tile processing:
   For each selected tile i:
     tile_input = Linear_in(x)[:, :, i]
     tile_output = Transform_i(ReLU(tile_input))
     weighted_output = tile_output × weights[i]

4. Aggregation:
   output = Linear_out(concat(weighted_outputs))
```

#### Implementation

```python
class MixtureOfTiles(nn.Module):
    def __init__(self, embed_dim, neurons_per_tile, tiles_per_layer, mot_k=2):
        super().__init__()
        self.tile_proj_in = nn.Linear(embed_dim, neurons_per_tile * tiles_per_layer)
        self.tile_proj_out = nn.Linear(neurons_per_tile * tiles_per_layer, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, tiles_per_layer)
        self.tile_transforms = nn.ParameterList([...])  # Per-tile transforms

    def forward(self, x):
        # Compute gates
        gate_logits = self.gate_proj(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        # Select top-k tiles
        topk_weights, topk_indices = torch.topk(gate_weights, self.mot_k, dim=-1)

        # Process selected tiles
        tile_output = torch.zeros(...)
        for b, s in batch_seq:
            for k in range(self.mot_k):
                tile_idx = topk_indices[b, s, k]
                tile_data = tile_input[b, s, tile_idx]
                transformed = ReLU(tile_data @ self.tile_transforms[tile_idx])
                tile_output[b, s, tile_idx] = transformed * topk_weights[b, s, k]

        return self.tile_proj_out(tile_output), gate_weights
```

#### Benefits

| Metric | Dense FFN | MoT (k=2) |
|--------|-----------|-----------|
| FLOPs/token | 2 × embed × hidden | k × (embed × tile + tile²) |
| Effective capacity | hidden | tiles × tile |
| Parameters | embed × hidden | embed × tiles × tile + tiles × tile² |

### 2. Tile-Local Attention

Attention restricted to local neighborhoods with grouped query optimization.

#### Mathematical Formulation

```
For token i with window size W:

1. Local attention:
   Q_i = x_i W_Q
   K_j = x_j W_K  for j ∈ [i-W, i]
   V_j = x_j W_V  for j ∈ [i-W, i]

2. Attention scores:
   A_ij = (Q_i · K_j) / √d_k  for j ∈ [i-W, i]

3. Output:
   output_i = Σ_j softmax(A_ij) V_j
```

#### Grouped Query Attention (GQA)

```
Standard Multi-Head:
  Q heads: H    K heads: H    V heads: H

Grouped Query:
  Q heads: H    K heads: H/G    V heads: H/G
  Each KV head is shared by G Q heads
```

#### Implementation

```python
class TileLocalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, local_window_size=32):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim)
        self.n_groups = num_heads // num_kv_heads

    def forward(self, x, causal=True):
        # Project Q, K, V
        q = self.q_proj(x).view(..., num_heads, head_dim)
        k = self.k_proj(x).view(..., num_kv_heads, head_dim)
        v = self.v_proj(x).view(..., num_kv_heads, head_dim)

        # Repeat K/V for grouped query
        k = k.repeat_interleave(self.n_groups, dim=...)
        v = v.repeat_interleave(self.n_groups, dim=...)

        # Compute attention (uses flash attention if available)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        return self.out_proj(output)
```

### 3. SwiGLU FeedForward

Swish Gated Linear Unit for better expressivity.

#### Mathematical Formulation

```
SwiGLU(x) = Swish(xW_gate) ⊗ (xW_value)W_out
          = (xW_gate ⊗ σ(xW_gate)) ⊗ (xW_value)W_out

where σ is the sigmoid function and ⊗ is element-wise multiplication.
```

#### Implementation

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc_gate = nn.Linear(embed_dim, hidden_dim)
        self.fc_value = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        gate = self.fc_gate(x)
        value = self.fc_value(x)
        x = F.silu(gate) * value  # SwiGLU
        return self.out_proj(x)
```

## Parameter Efficiency

### Weight Tying

Input and output embeddings share weights:

```python
# Instead of separate output projection
self.output_proj = nn.Linear(embed_dim, vocab_size)

# Use token embedding weights
logits = F.linear(x, self.token_embedding.weight)
```

**Savings:** embed × vocab_size parameters

### Grouped Query Attention

```
Standard MHA parameters: 3 × embed² + embed² = 4 × embed²
GQA parameters (G groups): embed² + 2 × (embed/G)² + embed² = 2 × embed² + 2 × (embed/G)²

Savings for G=3: ~50% reduction in attention parameters
```

## Training Optimizations

### Mixed Precision (AMP)

```python
scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = model.compute_loss(logits, targets)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
scaler.step(optimizer)
scaler.update()
```

### Gradient Accumulation

```python
for batch_idx, (input_ids, targets) in enumerate(loader):
    loss = model.train_step(input_ids, targets)
    (loss / accumulation_steps).backward()

    if (batch_idx + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Cosine LR Schedule with Warmup

```python
def get_lr(step):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + (peak_lr - min_lr) * 0.5 * (1 + cos(π × progress))
```

## Complexity Analysis

### Forward Pass

| Component | Complexity | Notes |
|-----------|------------|-------|
| Embedding | O(batch × seq × embed) | Lookup |
| Attention | O(batch × seq × window × embed) | Local window |
| MoT | O(batch × seq × k × tile²) | k active tiles |
| FFN | O(batch × seq × embed × hidden) | SwiGLU |

### Memory

| Component | Memory | Notes |
|-----------|--------|-------|
| Activations | O(batch × seq × embed × layers) | Gradient checkpointing reduces this |
| Parameters | O(vocab × embed + layers × (attention + MoT + FFN)) | Weight tying saves vocab × embed |
| Gradients | Same as parameters | |
| Optimizer state | 2× parameters (AdamW) | Momentum + variance |

## Tile Importance Visualization

During training, tile importance can be tracked:

```python
logits, tile_stats = model(input_ids, return_tile_stats=True)

# tile_stats is a list of importance weights per layer
# Shape: [num_layers] × (batch, tiles_per_layer)

# Visualize which tiles are most active for different contexts
import matplotlib.pyplot as plt
plt.imshow(torch.stack(tile_stats).mean(dim=1).cpu().T)
plt.xlabel("Sequence position")
plt.ylabel("Tile")
plt.title("Tile Importance Heatmap")
```

## Comparison with Standard Transformers

| Feature | Standard Transformer | FastLMEquiTile |
|---------|---------------------|----------------|
| Attention | Global O(n²) | Local O(n) |
| Feedforward | Dense | MoT (sparse) |
| QKV Heads | Same count | Grouped (GQA) |
| Activation | GeLU | SwiGLU |
| Weight Tying | Optional | Default |
| Norm Position | Post-norm | Pre-norm |

## Expected Results

### Shakespeare Dataset (Character-level)

| Epochs | Training Loss | Val Loss | Val PPL |
|--------|---------------|----------|---------|
| 5 | 0.35 | 0.42 | 1.52 |
| 10 | 0.25 | 0.38 | 1.46 |
| 20 | 0.18 | 0.35 | 1.42 |

### Training Speed (RTX 3060)

| Model Size | Tokens/sec | Time/epoch |
|------------|------------|------------|
| Tiny | 200K | 30s |
| Small | 150K | 60s |
| Medium | 100K | 90s |
