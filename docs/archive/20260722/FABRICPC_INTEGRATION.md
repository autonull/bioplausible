# FabricPC Integration

## Overview

This integration brings FabricPC's **node-graph topology abstraction** and **predictive coding training mode** into Bioplausible's unified framework for biologically plausible learning.

- **Graph API:** `Linear`, `ReLU`, `Tanh` nodes connected via `Edge` objects into a validated `GraphStructure`, with `InferenceSGD` for energy-minimization settling.
- **Training:`train_backprop` and `train_pcn`** — both accept the same graph and parameters, enabling fair comparison.
- **BioModel wrapper:** `FabricPCGraphPCN` integrates via `@register_model("fabricpc_graph_pcn")` for use with Bioplausible's factory, trainer, and demo infrastructure.

## What We Adopted from FabricPC

| FabricPC Concept | Bioplausible Location |
|---|---|
| Node-graph topology (`Slot`, `NodeBase`, `Edge`, `GraphStructure`) | `bioplausible/graph/nodes.py`, `topology.py` |
| PC energy function `E = Σ ||a_child - f_parent(a_parent, θ)||²` | `bioplausible/graph/inference.py` (`InferenceSGD`) |
| `train_pcn` (predictive coding weight updates) | `bioplausible/graph/training.py` |
| `train_backprop` on the same graph | `bioplausible/graph/training.py` |
| Graph parameter initialization | `bioplausible/graph/initialization.py` |

## What Bioplausible Adds

- **Unified factory:** The `FabricPCGraphPCN` model plugs into `create_model()`, `SupervisedTrainer`, and all existing infrastructure without adapters.
- **Six-way comparison:** Backprop, PC, EqProp, CHL, Hebbian, Forward-Forward share the same MLP architecture via the existing `ModelSpec` registry (see `examples/fabricpc_six_way_comparison.py`).
- **Local autodiff:** Uses `torch.func.grad` for local per-node gradients (PC), with Hebbian-like fallback.
- **Python 3.9+ compatible**, no JAX dependency.

## Architecture Mapping

```
┌─────────────────────────────┐      ┌──────────────────────────────┐
│      FabricPC (JAX)         │      │   Bioplausible (PyTorch)     │
│                             │      │                              │
│  Linear() / ReLU()          │ ──── │  graph.nodes.Linear/ReLU     │
│  Slot / Edge                │ ──── │  graph.topology.Slot/Edge    │
│  GraphStructure             │ ──── │  graph.topology.GraphStructure│
│  graph() helper             │ ──── │  graph.topology.graph()      │
│  InferenceSGD               │ ──── │  graph.inference.InferenceSGD│
│  train_pcn                  │ ──── │  graph.training.train_pcn    │
│  train_backprop             │ ──── │  graph.training.train_backprop│
└─────────────────────────────┘      └──────────────────────────────┘
```

## Roadmap (Tier 3 — Post-Submission)

1. **`ConvNode`, `MaxPool`** — convolutional graph nodes for vision tasks
2. **`torch.compile`** acceleration on the inference settle loop
3. **Multi-GPU** via Lightning DDP/FSDP for PC training
4. **Depth-scaling experiments** (FabricPC `examples/scaling/`)
5. **AutoScientist hyperparameter search**
6. **TransformerBlock, Hopfield** attention-based nodes
7. **JAX backend** as optional accelerator (interop)

## Quick Start

```python
from bioplausible.graph import (
    Linear,
    ReLU,
    Edge,
    TaskMap,
    graph,
    initialize_params,
    train_backprop,
    train_pcn,
)

# Define graph once
inp = Linear(shape=(784, 256), name="input")
act = ReLU(name="hidden")
out = Linear(shape=(256, 10), name="output")
g = graph(
    nodes=[inp, act, out],
    edges=[Edge(inp, act.slot("input")), Edge(act, out.slot("input"))],
    task_map=TaskMap(x=inp, y=out),
)

# Train with two modes
params = initialize_params(g)
bp = train_backprop(g, params, train_loader, epochs=5)
pc = train_pcn(g, params, train_loader, epochs=5)
```

## Credits

The graph abstraction, PC training mode, and local autodiff pattern are adapted from [FabricPC](https://github.com/trueagi-io/FabricPC) (MIT License), created by Dr. Matthew Behrend and maintained by SingularityNET as part of the Artificial Superintelligence Alliance. We gratefully acknowledge their contribution to the predictive coding research community.

## License

This integration is distributed under the MIT License. See `LICENSE` for details.
