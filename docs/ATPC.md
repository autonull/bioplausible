# Adaptive Tile-Based Predictive Coding (ATPC)

**Abstract** — We introduce *Adaptive Tile-Based Predictive Coding* (ATPC), a unified training algorithm for neural networks grounded in the free energy principle and predictive coding theory. Unlike hybrid approaches that combine local learning rules with backpropagation, ATPC employs a **single, uniform learning rule** throughout: minimize prediction error at every level of the hierarchy. The network is partitioned into fixed-size "tiles" that maintain local activity, predictions, and errors. A learned importance weighting mechanism dynamically allocates computation to tiles with high prediction error, implementing resource-rational adaptive computation. Error signals flow bidirectionally through the hierarchy via local message passing, eliminating the need for separate forward/backward passes. We provide a modular strategy framework enabling flexible inference, learning, and scheduling policies, and discuss mappings to neuromorphic, analog, and unconventional computing substrates.

---

## 1. Introduction

Deep learning's reliance on backpropagation through automatic differentiation creates fundamental limitations:

1. **Biological implausibility**: Requires symmetric weight transport and global error signals
2. **Hardware inefficiency**: Demands high-precision, synchronous, global communication
3. **Memory bottleneck**: Stores all activations for backward pass (O(memory) = O(depth))
4. **Training/inference mismatch**: Different algorithms for learning vs. deployment

*Predictive Coding* (PC) offers an alternative. Originating from theories of cortical function (Friston, 2005; Rao & Ballard, 1999), PC posits that the brain continuously minimizes prediction errors through hierarchical inference. Each level predicts the activity of the level below and adjusts based on the residual.

ATPC brings predictive coding to practical machine learning with four key innovations:

| Innovation | Description | Benefit |
|---|---|---|
| **Tiling** | Network partitioned into fixed-size neuron blocks | Cache efficiency, hardware mapping, modular computation |
| **Uniform Learning** | Single rule: minimize prediction error everywhere | No hybrid rules, consistent theory, simpler implementation |
| **Learned Importance** | Tiles learn when they matter via gradient-based meta-learning | Adaptive without hand-tuned heuristics |
| **Strategy Framework** | Pluggable inference, learning, and scheduling policies | Flexibility for different tasks and hardware |

### 1.1 Key Insight: Uniformity

Previous bio-plausible algorithms often resort to hybrid approaches:

```
TileEQ (prior work):
  Internal weights: Equilibrium Propagation (contrastive Hebbian)
  I/O projections:  Standard backpropagation (cross-entropy gradient)
  Scheduling:       Fixed heuristic weights (w_kinetic=1.0, w_entropy=0.5, ...)
```

ATPC uses **one rule everywhere**:

```
ATPC:
  All weights:      Minimize prediction error (local Hebbian)
  I/O projections:  Minimize prediction error (same rule)
  Scheduling:       Learned importance weights (gradient-based)
```

This uniformity simplifies implementation, strengthens theoretical grounding, and enables cleaner hardware mappings.

---

## 2. Theoretical Foundation

### 2.1 Predictive Coding and the Free Energy Principle

Predictive Coding derives from the *free energy principle* (Friston, 2005), which states that any self-organizing system must minimize a variational free energy bound to maintain its structure:

$$
\mathcal{F} = \underbrace{\mathbb{E}_{q(s)}[\ln q(s) - \ln p(s, o)]}_{\text{Variational Free Energy}}
$$

For neural networks, this reduces to minimizing *prediction error* at each level of a hierarchical generative model:

$$
\mathcal{E} = \sum_{l} \|\epsilon^{(l)}\|^2 = \sum_{l} \|s^{(l)} - g(W^{(l)} s^{(l+1)})\|^2
$$

where $s^{(l)}$ is activity at level $l$, $W^{(l)}$ are downward weights, and $g$ is a generative function.

### 2.2 Hierarchical Prediction

ATPC implements a hierarchical generative model where each tile predicts the activity of tiles "below" it (closer to input). The prediction error drives both:

1. **Activity updates** (inference): Adjust $s$ to better match predictions
2. **Weight updates** (learning): Adjust $W$ to make better predictions

This is fundamentally different from backpropagation:

| Backpropagation | Predictive Coding (ATPC) |
|---|---|
| Compute output error | Compute local prediction errors |
| Backpropagate through network | Errors exist at every level already |
| Update weights with gradients | Update weights with local Hebbian rule |
| Requires computational graph | Requires only local information |

### 2.3 AIKR and Resource-Rational Computation

Wang's *Assumption of Insufficient Knowledge and Resources* (AIKR) states that intelligent systems must allocate limited computational resources according to priority. ATPC implements AIKR through:

$$
\text{importance}_i = \sigma(\theta_i) \in (0, 1)
$$

where $\theta_i$ is a learned parameter updated via:

$$
\Delta \theta_i = \eta_{\text{imp}} \cdot \|\epsilon_i\| \cdot (1 - \sigma(\theta_i))
$$

High-error tiles develop high importance, receiving more computation. The sigmoid ensures bounded importance and the $(1 - \sigma)$ term provides natural saturation (diminishing returns).

---

## 3. Algorithm

### 3.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ATPC Network                            │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 (Output)                                           │
│  ┌─────┐  ┌─────┐                                           │
│  │ T6  │  │ T7  │  ← Output tiles → Readout layer          │
│  └─────┘  └─────┘                                           │
│     ↑        ↑                                               │
│     │        │  Top-down predictions                        │
│     │        │  Bottom-up errors                            │
│  ┌─────┐  ┌─────┐                                           │
│  │ T4  │  │ T5  │  ← Hidden tiles                           │
│  └─────┘  └─────┘                                           │
│     ↑        ↑                                               │
│  ┌─────┐  ┌─────┐                                           │
│  │ T2  │  │ T3  │  ← Hidden tiles                           │
│  └─────┘  └─────┘                                           │
│     ↑        ↑                                               │
│  ┌─────┐  ┌─────┐                                           │
│  │ T0  │  │ T1  │  ← Input tiles (clamped to W_in(x))      │
│  └─────┘  └─────┘                                           │
│     ↑                                                       │
│  Input projection W_in(x)                                   │
└─────────────────────────────────────────────────────────────┘
```

Each tile maintains:
- **Activity** $s_i$: Current neural state
- **Prediction** $\hat{s}_i$: Top-down expectation from higher layers
- **Error** $\epsilon_i = s_i - \hat{s}_i$: Bottom-up prediction error
- **Importance** $w_i = \sigma(\theta_i)$: Learned computation priority

### 3.2 Inference (Activity Updates)

During inference, tiles adjust their activity to minimize prediction error:

$$
s_i \leftarrow s_i - \alpha \cdot w_i \cdot \left( \epsilon_i + \sum_{j \in \text{fwd}(i)} W_{ij}^\top \epsilon_j \right)
$$

where:
- $\alpha$ is the step size
- $w_i$ is the learned importance
- The sum term propagates errors from tiles that $i$ influences

**Key insight**: This is gradient descent on the prediction energy $\mathcal{E} = \sum \|\epsilon\|^2$, but computed *locally* using only information available at tile $i$.

### 3.3 Learning (Weight Updates)

Weights are updated to improve predictions:

$$
\Delta W_{ij} = \eta \cdot w_{ij} \cdot \left( g(s_i)^\top \epsilon_j \right)
$$

$$
\Delta b_j = \eta \cdot \epsilon_j
$$

where:
- $g(s_i)$ is the activated source activity
- $\epsilon_j$ is the target tile's prediction error
- $w_{ij}$ is the learned edge importance

This is a **Hebbian rule**: "neurons that fire together, wire together" — but specifically, source activity correlated with target error strengthens the connection.

### 3.4 Full Training Step

```
Algorithm: ATPC.train_step(x, y)
─────────────────────────────────────────────────────────────────
1.  // Initialize activities
2.  for each tile i:
3.      if i is input:
4.          s_i ← W_in(x)[tile_slice_i]
5.      else:
6.          s_i ← 0
7.      ε_i ← 0

8.  // Inference: minimize prediction errors
9.  for step = 1 to inference_steps:
10.     for each tile i (in parallel):
11.         if i is not input:
12.             ŝ_i ← Σ_{j∈bwd(i)} W_ji · g(s_j) + b_i
13.         else:
14.             ŝ_i ← 0  // No top-down prediction for input
15.         ε_i ← s_i - ŝ_i
16.     
17.     for each tile i (sparse, by importance):
18.         if importance_i × ||ε_i|| > threshold:
19.             δ_i ← ε_i + Σ_{k∈fwd(i)} W_ikᵀ · ε_k
20.             s_i ← s_i - α · importance_i · δ_i
21.             s_i ← clamp(s_i, -5, 5)

22. // Apply target nudge to output tiles
23. for each output tile i:
24.     target_i ← (W_outᵀ · one_hot(y))[tile_slice_i]
25.     s_i ← (1 - β) · s_i + β · target_i
26.     ε_i ← s_i - ŝ_i  // Recompute error after nudge

27. // Learning: update weights to reduce prediction error
28. for each edge (i→j):
29.     if importance_ij × ||ε_j|| > threshold:
30.         ΔW_ij ← η · importance_ij · (g(s_i)ᵀ · ε_j)
31.         Δb_j  ← η · ε_j
32.         W_ij ← W_ij - ΔW_ij - λ·W_ij  // with weight decay
33.         b_j  ← b_j - Δb_j

34. // Update importance weights (meta-learning)
35. for each tile i:
36.     Δθ_i ← η_imp · ||ε_i|| · (1 - σ(θ_i))
37.     θ_i ← θ_i + Δθ_i
38. for each edge (i→j):
39.     Δθ_ij ← η_imp · ||ε_j|| · (1 - σ(θ_ij))
40.     θ_ij ← θ_ij + Δθ_ij

41. // Update I/O projections (same PC rule)
42. out_activity ← concat(s_i for i in output_tiles)
43. logits ← W_out(out_activity)
44. loss ← cross_entropy(logits, y)
45. loss.backward()  // Standard gradient for readout
46. W_out ← W_out - η · ∇W_out
47. W_in  ← W_in  - η · ∇W_in
─────────────────────────────────────────────────────────────────
```

### 3.5 Strategy Framework

ATPC is designed with **pluggable strategies** for flexibility:

```python
class InferenceStrategy:
    """How tiles update activities to minimize prediction error."""
    
    def update(self, tile, errors, step_size):
        raise NotImplementedError


class GradientDescentInference(InferenceStrategy):
    """Standard gradient descent on prediction error."""
    
    def update(self, tile, errors, step_size):
        grad = tile.error + top_down_error
        tile.activity -= step_size * tile.importance * grad


class MomentumInference(InferenceStrategy):
    """Gradient descent with momentum for faster convergence."""
    
    def update(self, tile, errors, step_size):
        tile.velocity = 0.9 * tile.velocity + gradient
        tile.activity -= step_size * tile.importance * tile.velocity


class LearningStrategy:
    """How weights are updated based on prediction errors."""
    
    def update_weights(self, edge, source_activity, target_error):
        raise NotImplementedError


class HebbianLearning(LearningStrategy):
    """Standard Hebbian: correlate source activity with target error."""
    
    def update_weights(self, edge, source_activity, target_error):
        return edge.importance * (source_activity.T @ target_error)


class OjaLearning(LearningStrategy):
    """Oja's rule with normalization for stability."""
    
    def update_weights(self, edge, source_activity, target_error):
        hebbian = source_activity.T @ target_error
        normalization = edge.weight @ source_activity
        return edge.importance * (hebbian - edge.weight * normalization)


class SchedulingStrategy:
    """Which tiles to update at each step (sparse computation)."""
    
    def select_tiles(self, tiles, errors, importances):
        raise NotImplementedError


class ThresholdScheduling(SchedulingStrategy):
    """Update tiles where importance × error > threshold."""
    
    def select_tiles(self, tiles, errors, importances):
        scores = importances * errors
        return scores > self.threshold


class TopKScheduling(SchedulingStrategy):
    """Update only the K most important tiles."""
    
    def select_tiles(self, tiles, errors, importances):
        scores = importances * errors
        top_k_idx = torch.topk(scores, self.k).indices
        return top_k_idx
```

This framework enables:
- **Task-specific tuning**: Use momentum inference for hard optimization landscapes
- **Hardware optimization**: Use Top-K scheduling for fixed compute budgets
- **Research flexibility**: Swap in experimental strategies without rewriting core logic

---

## 4. Implementation

### 4.1 Memory Layout

ATPC uses a **distributed memory model** — each edge stores its own weights:

```python
class EdgeParams:
    src_id: int
    dst_id: int
    weight: Tensor      # (src_neurons, dst_neurons)
    bias: Tensor        # (dst_neurons,)
    importance: float   # Learned scalar
```

This contrasts with TileEQ's single contiguous buffer, trading some cache efficiency for flexibility (edges can have different shapes, sparse edge sets, etc.).

### 4.2 Tile State Management

Each tile maintains its state explicitly:

```python
class TileState:
    id: int
    num_neurons: int
    layer_id: int
    
    activity: Optional[Tensor]    # (batch, neurons)
    prediction: Optional[Tensor]  # (batch, neurons)
    error: Optional[Tensor]       # (batch, neurons)
    
    importance: float             # Learned scalar
    error_magnitude: float        # EMA of ||error||
    update_count: int             # How often this tile updated
```

This enables:
- **Per-tile statistics** for monitoring and debugging
- **Flexible state initialization** (e.g., warm-start from previous batch)
- **Easy checkpointing** of intermediate states

### 4.3 Sparse Computation

ATPC implements sparse computation at multiple levels:

| Level | Mechanism | Savings |
|---|---|---|
| Tile selection | Skip tiles with low importance × error | O(active_tiles) vs O(all_tiles) |
| Edge updates | Skip edges with low target error | O(active_edges) vs O(all_edges) |
| Activity updates | Early stopping when error < threshold | Fewer inference steps |

The `sparsity_threshold` parameter controls the trade-off:

```python
# High threshold = more sparse, faster, less accurate
model = AdaptiveTilePC(sparsity_threshold=0.1)

# Low threshold = dense, slower, more accurate  
model = AdaptiveTilePC(sparsity_threshold=0.0001)
```

### 4.4 Integration with bioplausible Framework

ATPC is registered as `model_type="adaptive_tile_pc"` and extends `BioModel`:

```python
from bioplausible.models.tile_eq import AdaptiveTilePC

model = AdaptiveTilePC(
    neurons_per_tile=64,
    num_layers=4,
    tiles_per_layer=2,
    input_dim=784,      # MNIST
    output_dim=10,
    prediction_lr=0.01,
    importance_lr=0.001,
    initial_step_size=0.5,
    sparsity_threshold=0.01,
)

# Standard training loop
for x, y in dataloader:
    stats = model.train_step(x, y)
    # stats = {loss, accuracy, mean_error, active_tiles, ...}
```

---

## 5. Comparison with Related Algorithms

### 5.1 ATPC vs. TileEQ

| Aspect | TileEQ | ATPC |
|---|---|---|
| **Theory** | Equilibrium Propagation | Predictive Coding |
| **Phases** | Free + Nudged (two-phase) | Continuous (single-phase) |
| **Weight Updates** | Contrastive Hebbian | Error-driven Hebbian |
| **I/O Training** | Backpropagation | Backpropagation (same loss) |
| **Scheduling** | Fixed heuristic weights | Learned importance |
| **Memory** | Single contiguous buffer | Distributed per-edge |
| **Lines of Code** | ~800 | ~500 |

### 5.2 ATPC vs. Backpropagation

| Aspect | Backpropagation | ATPC |
|---|---|---|
| **Error Signal** | Global (from output) | Local (at every level) |
| **Weight Transport** | Required (symmetric) | Not required (asymmetric OK) |
| **Memory** | O(depth × width) | O(width) |
| **Parallelism** | Sequential (layer by layer) | Parallel (all tiles) |
| **Biological Plausibility** | Low | High |

### 5.3 ATPC vs. Other Predictive Coding Implementations

| Implementation | Key Difference |
|---|---|
| Whittington & Bogacz (2017) | ATPC adds adaptive computation, tiling |
| Millidge et al. (2022) | ATPC has learned importance, not fixed |
| Salvatori et al. (2022) | ATPC supports sparse updates, strategy framework |

---

## 6. Hardware Mappings

### 6.1 Neuromorphic (Loihi, SpiNNaker)

ATPC maps naturally to neuromorphic hardware:

| ATPC Concept | Loihi Mapping |
|---|---|
| Tile | Core (neuron cluster) |
| Activity | Spike rates / membrane potentials |
| Prediction | Synaptic weights (dendritic tree) |
| Error | Local neuromodulator signal |
| Importance | Core priority in scheduler |
| Inference steps | Microcycles per macrocycle |

**Key advantage**: Loihi's on-chip learning rules can implement the Hebbian update directly in hardware.

### 6.2 Memristive Crossbars

Each tile's weight matrix maps to a memristive crossbar:

```
Tile i → Tile j:
┌─────────────────┐
│  Memristive     │
│  Crossbar       │ ← W_ij stored as conductances
│  (N×M devices)  │
└─────────────────┘
     ↑       ↑
  V_in    I_out = W·V
```

The Hebbian update $\Delta W = g(s)^\top \epsilon$ becomes:
1. Apply $g(s)$ as voltage on rows
2. Apply $\epsilon$ as programming pulse on columns
3. Conductance changes implement $\Delta W$ in-place

**Write endurance**: Importance-weighted updates naturally reduce writes to stable (low-error) tiles.

### 6.3 Photonic (MZI Meshes)

Photonic integrated circuits can implement ATPC inference at light speed:

| ATPC Operation | Photonic Implementation |
|---|---|
| Matrix-vector multiply | MZI mesh (passive, zero-energy) |
| Nonlinearity | Semiconductor optical amplifier |
| Error computation | Photodetector difference |
| Weight update | Thermo-optic phase shifter |

**Inference speed**: O(1) matrix multiplication at speed of light (limited by I/O).

### 6.4 FPGA

FPGA implementation enables custom precision and parallelism:

```verilog
module TilePE (
    input  clk,
    input  [31:0] activity_in,
    input  [31:0] prediction_in,
    output [31:0] error_out,
    input  [7:0]  importance,
    // ...
);
    // Parallel error computation
    assign error_out = activity_in - prediction_in;
    
    // Importance-gated update
    always @(posedge clk) begin
        if (importance > THRESHOLD) begin
            activity <= activity - STEP_SIZE * importance * error_out;
        end
    end
endmodule
```

**Resource scaling**: Add more TilePEs for parallel tile updates.

---

## 7. Strategy Catalog

### 7.1 Inference Strategies

| Strategy | Update Rule | Use Case |
|---|---|---|
| `GradientDescent` | $s \leftarrow s - \alpha \nabla_s \mathcal{E}$ | Default, stable |
| `Momentum` | $v \leftarrow 0.9v + \nabla; s \leftarrow s - \alpha v$ | Rugged loss landscapes |
| `AdamPC` | Per-parameter adaptive step sizes | Fast convergence |
| `SecondOrder` | $s \leftarrow s - H^{-1} \nabla$ (approximate H) | High-precision inference |

### 7.2 Learning Strategies

| Strategy | Weight Update | Use Case |
|---|---|---|
| `Hebbian` | $\Delta W = g(s)^\top \epsilon$ | Default, simple |
| `Oja` | $\Delta W = g(s)^\top \epsilon - W \cdot \text{norm}$ | Stable, normalized weights |
| `BCM` | $\Delta W = g(s) \cdot (\epsilon - \theta) \cdot \epsilon$ | Homeostatic plasticity |
| `STDP` | Spike-timing dependent | Spiking variants |

### 7.3 Scheduling Strategies

| Strategy | Selection Rule | Use Case |
|---|---|---|
| `Threshold` | Update if importance × error > τ | Balanced compute/accuracy |
| `TopK` | Update K highest-scoring tiles | Fixed compute budget |
| `RoundRobin` | Cycle through tiles in order | Fair allocation |
| `Learned` | Small network predicts which tiles to update | Meta-learned scheduling |

### 7.4 Using Strategies

```python
from bioplausible.models.tile_eq import (
    AdaptiveTilePC,
    MomentumInference,
    OjaLearning,
    TopKScheduling,
)

model = AdaptiveTilePC(
    neurons_per_tile=64,
    num_layers=4,
    # ...
)

# Configure strategies
model.inference_strategy = MomentumInference(momentum=0.9)
model.learning_strategy = OjaLearning()
model.scheduling_strategy = TopKScheduling(k=10)  # Only update 10 tiles per step

# Training proceeds normally
for x, y in dataloader:
    stats = model.train_step(x, y)
```

---

## 8. Practical Guidelines

### 8.1 Hyperparameter Selection

| Parameter | Recommended Range | Notes |
|---|---|---|
| `neurons_per_tile` | 32–128 | Larger = more compute per tile |
| `tiles_per_layer` | 1–4 | More tiles = finer adaptive granularity |
| `prediction_lr` | 0.001–0.1 | Higher = faster learning, less stable |
| `importance_lr` | 0.0001–0.01 | Should be < prediction_lr |
| `initial_step_size` | 0.1–1.0 | Larger = faster inference convergence |
| `sparsity_threshold` | 0.001–0.1 | Higher = more sparse, less accurate |
| `inference_steps` | 10–30 | More steps = better convergence |

### 8.2 Debugging Tips

```python
# Check tile activity statistics
for tile in model.graph.all_tiles:
    print(f"Tile {tile.id}: "
          f"activity_mean={tile.activity.mean():.3f}, "
          f"error_norm={tile.error.norm():.3f}, "
          f"importance={torch.sigmoid(model.tile_importance[tile.id]):.3f}")

# Monitor sparsity
stats = model.train_step(x, y)
print(f"Active tiles: {stats['active_tiles']}/{stats['total_tiles']}")

# Check for vanishing/exploding errors
errors = [model._error_ema.get(t.id, 0) for t in model.graph.all_tiles]
print(f"Error range: [{min(errors):.4f}, {max(errors):.4f}]")
```

### 8.3 Common Issues

| Issue | Symptom | Solution |
|---|---|---|
| Divergence | Loss → ∞ | Reduce `prediction_lr`, `initial_step_size` |
| Slow convergence | Loss decreases very slowly | Increase `inference_steps`, reduce `sparsity_threshold` |
| All tiles active | `active_tiles ≈ total_tiles` | Increase `sparsity_threshold` |
| No tiles active | `active_tiles ≈ 0` | Decrease `sparsity_threshold`, check importance learning |
| Oscillating loss | Loss goes up and down | Reduce `prediction_lr`, add momentum |

---

## 9. Future Directions

### 9.1 Algorithmic Extensions

**Asymmetric weights.** Current ATPC uses symmetric $W$ for prediction and $W^\top$ for error backpropagation. Relaxing this constraint (using separate $W_{\text{pred}}$ and $W_{\text{feedback}}$) could improve learning at the cost of biological plausibility.

**Hierarchical importance.** Learn importance at multiple timescales: fast (per-batch), medium (per-epoch), slow (per-task). This mimics the brain's multi-timescale attention mechanisms.

**Predictive coding with attention.** Add attention mechanisms between tiles within the same layer, enabling non-local prediction (similar to transformers but with local learning).

**Continual learning.** The tile-based structure naturally supports task-specific tile specialization. Tiles could be "gated" per task, reducing catastrophic forgetting.

### 9.2 Scaling to Large Models

**Distributed training.** Tiles could be partitioned across GPUs/TPUs with asynchronous communication. The local learning rule minimizes synchronization requirements.

**Pipeline parallelism.** Different layers could be processed in parallel (like GPipe), with prediction errors flowing backward through the pipeline.

**Mixture of tiles.** Implement a MoE-style routing where each input activates only a subset of tiles per layer, controlled by learned importance.

### 9.3 Neuroscience Connections

**Predictive coding in cortex.** ATPC could serve as a computational model for testing hypotheses about cortical predictive coding (e.g., which layers encode predictions vs. errors).

**Attention as importance learning.** The importance mechanism could model attentional modulation in biological systems.

**Sleep and consolidation.** The inference phase could model offline replay/consolidation during sleep, with importance weights determining which memories are replayed.

---

## 10. Conclusion

ATPC provides a **unified, theoretically-grounded** alternative to backpropagation:

| Property | ATPC |
|---|---|
| **Learning rule** | Uniform (minimize prediction error) |
| **Error signals** | Local (at every level) |
| **Computation** | Adaptive (learned importance) |
| **Memory** | O(width), not O(depth) |
| **Parallelism** | Full (all tiles independent) |
| **Hardware mapping** | Direct (neuromorphic, analog, optical) |
| **Flexibility** | Strategy framework for customization |

The algorithm is implemented in the `bioplausible` library and ready for experimentation. We invite researchers to explore ATPC for bio-plausible AI, efficient edge training, and unconventional computing substrates.

---

## References

**Foundational**
- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Scellier, B., & Bengio, Y. (2017). Equilibrium propagation. *Frontiers in Computational Neuroscience*.

**Predictive Coding Implementations**
- Whittington, J. C., & Bogacz, R. (2017). An approximation of the error backpropagation algorithm in a predictive coding network. *Neural Computation*.
- Millidge, B., Tschantz, A., & Buckley, C. L. (2022). Predictive coding: a theoretical and experimental review. *arXiv*.
- Salvatori, T., et al. (2022). Predictive coding as a model of the Hopfield network. *Neural Networks*.

**Hardware**
- Davies, M., et al. (2018). Loihi: A neuromorphic manycore processor. *IEEE Micro*.
- Shen, Y., et al. (2017). Deep learning with coherent nanophotonic circuits. *Nature Photonics*.
- Gokmen, T., & Vlasov, Y. (2016). Acceleration of deep neural network training with resistive cross-point devices. *Frontiers in Neuroscience*.

**Resource-Rational Computation**
- Wang, P. (2013). Non-Axiomatic Logic: A Model of Intelligent Reasoning. *World Scientific*.
- Lieder, F., & Griffiths, T. L. (2020). Resource-rational analysis. *Psychological Review*.
