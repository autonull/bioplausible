# Adaptive Tile-Based Predictive Coding (ATPC)

## Overview

**ATPC** is a general-purpose, bio-plausible deep learning algorithm that combines:

1. **Predictive Coding**: Hierarchical error minimization through local prediction
2. **Adaptive Computation**: Learned tile importance for sparse, efficient updates
3. **Classification-Driven Learning**: Task-optimized internal representations
4. **Strategy Framework**: Pluggable inference, learning, and scheduling policies

### Key Features

| Feature | Description |
|---------|-------------|
| **Task Support** | Classification, Regression, Custom objectives |
| **Topologies** | Layered MLP, Custom graphs with skip connections |
| **Adaptive** | Learned tile importance, sparse computation |
| **Bio-Plausible** | Local learning rules, no global backprop |
| **Flexible** | Strategy framework for experimentation |

---

## Quick Start

### Installation

```python
from bioplausible.models.tile_eq import AdaptiveTilePC
```

### Basic Classification

```python
model = AdaptiveTilePC(
    neurons_per_tile=64,  # Neurons per tile
    num_layers=4,  # Total layers (input + hidden + output)
    tiles_per_layer=4,  # Tiles per hidden layer
    input_dim=784,  # Input features (e.g., MNIST)
    output_dim=10,  # Output classes
    task_type="classification",
    prediction_lr=0.02,
    inference_steps=15,
)

# Training
for epoch in range(10):
    for X_batch, y_batch in dataloader:
        stats = model.train_step(X_batch, y_batch)
        print(f"Loss: {stats['loss']:.3f}, Acc: {stats['accuracy']:.3f}")
```

### Regression Task

```python
model = AdaptiveTilePC(
    neurons_per_tile=32,
    num_layers=3,
    tiles_per_layer=4,
    input_dim=20,
    output_dim=1,
    task_type="regression",  # MSE loss, R² metric
)

for X_batch, y_batch in dataloader:
    stats = model.train_step(X_batch, y_batch)
    print(f"MSE: {stats['loss']:.3f}, R²: {stats['accuracy']:.3f}")
```

### Custom Topology

```python
# Skip connection topology: 0 -> 1 -> 3, 0 -> 2 -> 3
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]

model = AdaptiveTilePC(
    neurons_per_tile=32,
    num_layers=4,
    tiles_per_layer=1,
    input_dim=64,
    output_dim=4,
    topology="custom",
    custom_edges=edges,
)
```

---

## Architecture

### Tile Structure

```
┌─────────────────────────────────────────────────────────┐
│                    ATPC Network                          │
├─────────────────────────────────────────────────────────┤
│  Output Layer                                            │
│  ┌─────┐  ┌─────┐  ← Output tiles → Readout            │
│  │ T6  │  │ T7  │                                       │
│  └─────┘  └─────┘                                       │
│     ↑        ↑                                           │
│     │        │  Top-down predictions                    │
│     │        │  Bottom-up errors                        │
│  ┌─────┐  ┌─────┐                                       │
│  │ T4  │  │ T5  │  ← Hidden tiles                       │
│  └─────┘  └─────┘                                       │
│     ↑        ↑                                           │
│  ┌─────┐  ┌─────┐                                       │
│  │ T2  │  │ T3  │  ← Hidden tiles                       │
│  └─────┘  └─────┘                                       │
│     ↑        ↑                                           │
│  ┌─────┐  ┌─────┐                                       │
│  │ T0  │  │ T1  │  ← Input tiles (clamped)             │
│  └─────┘  └─────┘                                       │
│     ↑                                                   │
│  Input projection W_in(x)                               │
└─────────────────────────────────────────────────────────┘
```

### Each Tile Maintains

| State | Description |
|-------|-------------|
| **Activity** | Current neural state s_i |
| **Prediction** | Top-down expectation ŝ_i |
| **Error** | Bottom-up prediction error ε_i = s_i - ŝ_i |
| **Importance** | Learned computation priority w_i |

---

## Algorithm

### Forward Pass (Inference)

Tiles iteratively minimize prediction error:

```python
for step in range(inference_steps):
    # 1. Compute predictions from lower tiles
    for tile in tiles:
        tile.prediction = sum(W_ji · activation(s_j) for j in tile.bwd_neighbors)
    
    # 2. Compute errors
    for tile in tiles:
        tile.error = tile.activity - tile.prediction
    
    # 3. Update activities (sparse, importance-weighted)
    for tile in active_tiles:
        gradient = tile.error + top_down_error
        tile.activity -= step_size × importance × gradient
```

### Learning (Classification-Driven)

**Key Innovation**: Internal weights learn to support classification directly.

```python
# 1. Compute output and loss
logits = W_out(output_activities)
loss = cross_entropy(logits, y)

# 2. Backpropagate error through output layer
output_delta = (softmax(logits) - one_hot(y)) @ W_out.T

# 3. Propagate error through tile hierarchy
for tile in reverse_layer_order:
    tile.error = sum(fwd_tile.error @ W.T for fwd_tile in tile.fwd_neighbors)

# 4. Update internal weights (Hebbian)
for edge in edges:
    ΔW = activation(src).T @ dst.error
    W -= lr × ΔW
```

---

## Configuration

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `neurons_per_tile` | int | Neurons in each tile |
| `num_layers` | int | Total layers (input + hidden + output) |
| `tiles_per_layer` | int | Tiles per hidden layer |
| `input_dim` | int | Input feature dimension |
| `output_dim` | int | Output dimension (classes or continuous) |

### Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_type` | "classification" | "classification" or "regression" |
| `prediction_lr` | 0.01 | Learning rate for internal weights |
| `importance_lr` | 0.001 | Learning rate for importance weights |
| `initial_step_size` | 0.5 | Inference step size |
| `inference_steps` | 20 | Inference iterations |
| `sparsity_threshold` | 0.01 | Skip tiles below this error |
| `activation` | "gelu" | "tanh", "relu", or "gelu" |
| `topology` | "layered" | "layered" or "custom" |
| `custom_edges` | None | List of (src, dst) for custom topology |

---

## Strategy Framework

### Inference Strategies

```python
from bioplausible.models.tile_eq import MomentumInference

# Standard gradient descent (default)
model.inference_strategy = GradientDescentInference()

# With momentum for faster convergence
model.inference_strategy = MomentumInference(momentum=0.9)
```

### Learning Strategies

```python
from bioplausible.models.tile_eq import OjaLearning

# Standard Hebbian (default)
model.learning_strategy = HebbianLearning()

# Oja's rule with normalization
model.learning_strategy = OjaLearning()
```

### Scheduling Strategies

```python
from bioplausible.models.tile_eq import TopKScheduling

# Threshold-based (default)
model.scheduling_strategy = ThresholdScheduling(threshold=0.01)

# Top-K sparse scheduling
model.scheduling_strategy = TopKScheduling(k=5, min_fraction=0.2)

# All tiles (no sparsity)
model.scheduling_strategy = AllTilesScheduling()
```

---

## Model Utilities

### Save/Load

```python
# Save checkpoint
model.save_checkpoint("model.pt")

# Load checkpoint
model.load_checkpoint("model.pt", device=torch.device("cuda"))
```

### Inspection

```python
# Model summary
print(model.summarize())

# Weight statistics
stats = model.get_weight_statistics()
print(f"Mean weight: {stats['mean_weight']:.4f}")

# Tile activity stats
activity = model.get_tile_activity_stats()
print(f"Active tiles: {activity['active_tiles']}")

# Topology info (for visualization)
topo = model.get_topology_info()
```

### Training Metrics

```python
stats = model.train_step(X, y)
print(f"Loss: {stats['loss']:.3f}")
print(f"Accuracy: {stats['accuracy']:.3f}")
print(f"Mean error: {stats['mean_error']:.4f}")
print(f"Active tiles: {stats['active_tiles']}/{stats['total_tiles']}")
```

---

## Performance Guidelines

### Hyperparameter Recommendations

| Task | neurons_per_tile | tiles_per_layer | prediction_lr | inference_steps |
|------|-----------------|-----------------|---------------|-----------------|
| Small (≤10 classes) | 16-32 | 2-4 | 0.02-0.05 | 10-15 |
| Medium (10-100) | 32-64 | 4-8 | 0.01-0.02 | 15-20 |
| Large (100+) | 64-128 | 8-16 | 0.005-0.01 | 20-30 |
| Regression | 32-64 | 4-8 | 0.01-0.02 | 15-20 |

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Model collapse | All same prediction | Ensure classification-driven learning is enabled |
| Slow convergence | Loss decreases slowly | Increase `prediction_lr`, reduce `inference_steps` |
| Too sparse | Few tiles active | Reduce `sparsity_threshold` |
| Unstable training | Loss oscillates | Reduce `prediction_lr`, increase `inference_steps` |
| Overfitting | Train >> Test accuracy | Increase `weight_decay`, reduce model size |

---

## Theoretical Foundation

### Predictive Coding

ATPC implements the free energy principle (Friston, 2005):

$$\mathcal{F} = \sum_l \|\epsilon^{(l)}\|^2 = \sum_l \|s^{(l)} - g(W^{(l)} s^{(l+1)})\|^2$$

Each tile minimizes local prediction error through iterative inference.

### Classification-Driven Learning

Unlike pure predictive coding, ATPC backpropagates task error through the hierarchy:

$$\Delta W_{ij} = \eta \cdot \text{activation}(s_i)^T \cdot \epsilon_j^{\text{task}}$$

This ensures internal representations become task-discriminative.

### Adaptive Computation

Tile importance is learned via gradient descent:

$$\Delta \theta_i = \eta_{\text{imp}} \cdot \|\epsilon_i\| \cdot (1 - \sigma(\theta_i))$$

High-error tiles develop high importance, receiving more computation.

---

## References

### Foundational

- Friston, K. (2005). A theory of cortical responses. *Philosophical Transactions of the Royal Society B*.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex. *Nature Neuroscience*.
- Scellier, B., & Bengio, Y. (2017). Equilibrium propagation. *Frontiers in Computational Neuroscience*.

### Related Work

- Whittington, J. C., & Bogacz, R. (2017). An approximation of error backpropagation. *Neural Computation*.
- Millidge, B., et al. (2022). Predictive coding: A theoretical review. *arXiv*.
- Salvatori, T., et al. (2022). Predictive coding as Hopfield network. *Neural Networks*.

---

## API Reference

### AdaptiveTilePC

```python
class AdaptiveTilePC(BioModel):
    """Adaptive Tile-Based Predictive Coding."""
    
    def __init__(
        self,
        neurons_per_tile: int,
        num_layers: int,
        tiles_per_layer: int,
        input_dim: int,
        output_dim: int,
        task_type: str = "classification",
        prediction_lr: float = 0.01,
        importance_lr: float = 0.001,
        initial_step_size: float = 0.5,
        inference_steps: int = 20,
        sparsity_threshold: float = 0.01,
        activation: str = "gelu",
        topology: str = "layered",
        custom_edges: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    )
    
    def train_step(self, x: Tensor, y: Tensor) -> Dict[str, float]
    def forward(self, x: Tensor, steps: int = None) -> Tensor
    def save_checkpoint(self, path: str) -> None
    def load_checkpoint(self, path: str, device: Device = None) -> None
    def summarize(self) -> str
    def get_stats(self) -> Dict[str, float]
    def get_topology_info(self) -> Dict
```

---

## License

Part of the bioplausible library. See main repository for license details.
