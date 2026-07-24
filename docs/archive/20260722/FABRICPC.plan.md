# FABRICPC Integration — Phased Development Plan

**Status:** VERIFIED against actual Bioplausible codebase  
**Target:** BGI Open Build submission (SingularityNET / AGI Society)  
**Deadline:** July 24, 2026  
**Working Time:** ~48 hours effective

---

## Executive Summary

Integrate FabricPC's **node-graph topology abstraction** and **predictive coding training mode** into Bioplausible. The deliverable is a **working Tier 1 demo** showing `train_pcn` vs `train_backprop` on the same `GraphStructure` definition, plus documentation and submission materials. Tier 2 uses Bioplausible's **existing unified `build()` contract** to run all six learning rules on the same MLP architecture via the model factory. Tier 3 is post-submission roadmap.

---

## Critical Verified Facts (Incorporated Below)

| Assumption in Original Prompt | Reality in Codebase | Plan Adjustment |
|------------------------------|---------------------|-----------------|
| Separate `LearningRule` class exists | **No.** Zoo = `BioModel` subclasses with `train_step()` | New PC model = `BioModel` subclass `@register_model("fabricpc_graph_pcn")` |
| Six rules need adapter for same graph | **No.** All models share `build(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs)` | Tier 2: use existing factory on MLP (784→256→10) — no adapter needed |
| `predictive_coding_hybrid` is only PC model | **Also** `EquiTile` (PC mode), `GraphEqProp`, `pc_hybrid` | New model type = `"fabricpc_graph_pcn"` — distinct |
| `bioplausible/graph/` free | **Yes.** Free (EquiTile has `topology.py` + `graph.py` for tile-substrate, not node-graph) | Proceed with new module |
| Python 3.10+ | **3.9+** per pyproject.toml | Use `from __future__ import annotations` for PEP 585 types |
| Linting = black/isort/flake8 | **ruff + black + isort + flake8** (E501 ignored) | Run all four |
| Tests use pytest | **Yes**, `tests/conftest.py` at root | Place tests in `tests/graph/` |

---

## Tiering (Non-Negotiable)

### TIER 1 — MUST HAVE (Submission-Blocking)

| Item | Description | Done? |
|------|-------------|-------|
| **Graph abstraction** | `Linear` node, `Edge`, `Slot`, `TaskMap`, `graph()`, validation | ☐ |
| **Parameter init** | `initialize_params()` per node | ☐ |
| **train_backprop** | Standard autograd on graph (topological feedforward) | ☐ |
| **InferenceSGD** | Energy-minimization settling with local AD (`torch.func.grad`) | ☐ |
| **train_pcn** | PC weight updates from settled activities | ☐ |
| **FabricPCGraphPCN model** | `BioModel` subclass wrapping graph API, `@register_model("fabricpc_graph_pcn")` | ☐ |
| **MNIST demo** | 784→256→10 graph, backprop vs PC comparison table | ☐ |
| **FABRICPC_INTEGRATION.md** | Architecture mapping, credits, roadmap | ☐ |
| **Submission abstract** | `docs/open_build_submission.md` | ☐ |
| **README update** | Integration section with code snippet | ☐ |

### TIER 2 — STRONGLY DESIRED (Compelling Demo)

| Item | Description | Done? |
|------|-------------|-------|
| **Six-way comparison** | Use existing factory: `create_model(spec, 784, 10, 256, 2, device)` for Backprop, PC (new), EqProp, CHL, Hebbian, Forward-Forward | ☐ |
| **Comparison table** | Side-by-side accuracy/time on MNIST (1 epoch + subset for speed) | ☐ |
| **Unit tests** | Graph assembly, training, `torch.func` verification | ☐ |

### TIER 3 — STRETCH (Post-Submission Roadmap)

| Item | Description |
|------|-------------|
| `ConvNode`, `MaxPool`, `Hopfield`, `TransformerBlock` nodes | |
| `torch.compile` acceleration on inference loop | |
| Multi-GPU via Lightning DDP/FSDP for PC training | |
| Depth-scaling experiments (FabricPC `examples/scaling/`) | |
| Hyperparameter search with AutoScientist | |
| JAX backend as optional accelerator (interop) | |

---

## Phase 0: Codebase Orientation (30 min) — DO NOT SKIP

**Before writing any code, read and internalize:**

| File/Directory | Purpose | Key Takeaway |
|----------------|---------|--------------|
| `bioplausible/models/base.py` | `BioModel`, `ModelConfig`, `train_step` contract | All algorithms = `BioModel` subclass with `forward()` + `train_step(x,y)->Dict` |
| `bioplausible/models/registry.py` | `@register_model`, `ModelSpec`, `MODEL_REGISTRY` | Register new model + add to `MODEL_REGISTRY` list |
| `bioplausible/models/factory.py` | `create_model(spec, input_dim, output_dim, hidden_dim, num_layers, device, task_type, **kwargs)` | All 6 rules built via this uniform contract |
| `bioplausible/models/forward_forward.py` | `ForwardForwardNet` reference implementation | `train_step` returns `{"loss", "accuracy"}` |
| `bioplausible/models/pc_hybrid.py` | Existing PC hybrid model | Avoid naming collision; use `fabricpc_graph_pcn` |
| `bioplausible/training/supervised.py` | `SupervisedTrainer` — runs any `BioModel` | Your new model plugs in automatically |
| `tests/conftest.py` | Pytest fixtures, torch mocking | Tests in `tests/graph/` follow same pattern |
| `examples/equitile_mode_comparison.py` | Demo pattern: side-by-side comparison | Replicate structure for MNIST benchmark |

**Do not proceed until you can write a minimal `BioModel` subclass from memory.**

---

## Phase 1: Graph Abstraction — Nodes (2 hours)

### Files
- `bioplausible/graph/nodes.py`
- `bioplausible/graph/__init__.py` (exports)

### Requirements
1. **`Slot` class** — named input port with `name: str` and `owner: "NodeBase"`
2. **`NodeBase` abstract class** with contract:
   - `forward(**slot_inputs) -> torch.Tensor` — **PURE FUNCTION** (no in-place mutation, no side effects)
   - `get_slots() -> dict[str, Slot]`
   - `slot(name: str) -> Slot`
   - `initialize_params(rng_key: torch.Generator) -> dict[str, torch.Tensor]`
   - `name: str`
3. **`Linear(shape: tuple[int, int], name: str)`** — slots `{"in"}`, params `weight`, `bias`, forward = `F.linear`
4. **`ReLU(name: str)`, `Tanh(name: str)`** — slots `{"in"}`, no params, forward = `F.relu` / `F.tanh`

### Python 3.9 Compatibility
```python
from __future__ import annotations
from typing import Dict, List, Tuple  # not dict[str, X] inline
```

### Critical Verification Test (Write First — Must Pass Before Proceeding)
```python
# tests/graph/test_torch_func_verification.py
import torch
from bioplausible.graph.nodes import Linear


def test_torch_func_grad_on_linear_node():
    node = Linear(shape=(784, 256), name="test")
    params = node.initialize_params(torch.Generator().manual_seed(0))
    x = torch.randn(32, 784)

    def energy(p):
        out = node.forward(x, p["weight"], p["bias"])
        return (out**2).sum()

    grads = torch.func.grad(energy)(params)
    assert grads["weight"].shape == params["weight"].shape
    assert grads["bias"].shape == params["bias"].shape
```

**If this test fails, STOP. Fix the node implementation.** This is the single most important technical verification.

### File Header (Every New File in `bioplausible/graph/`)
```python
# Adapted from FabricPC (https://github.com/trueagi-io/FabricPC)
# Original authors: Dr. Matthew Behrend et al., SingularityNET
# MIT License. See FABRICPC_INTEGRATION.md for details.
```

---

## Phase 2: Graph Abstraction — Topology (1.5 hours)

### File
- `bioplausible/graph/topology.py`

### Classes
- **`Edge(source: NodeBase, target: Slot)`** — directed connection
- **`TaskMap(x: NodeBase, y: NodeBase)`** — input/output nodes
- **`GraphStructure`**:
  - `nodes: List[NodeBase]`
  - `edges: List[Edge]`
  - `task_map: TaskMap`
  - `inference: InferenceSGD`
  - `topological_order() -> List[NodeBase]` — Kahn's algorithm for feedforward
  - `get_predecessors(node: NodeBase) -> List[Tuple[NodeBase, Slot]]`
  - `validate() -> None` — check: all slots have exactly one incoming edge, no dangling edges, task_map nodes in graph. **Cycles permitted.**
- **`graph(nodes, edges, task_map, inference) -> GraphStructure`** — assemble and validate

### Tests
- `tests/graph/test_topology.py`: feedforward, cyclic, self-recurrent, skip connections, validation errors

---

## Phase 3: Initialization (30 min)

### File
- `bioplausible/graph/initialization.py`

### Function
```python
def initialize_params(
    structure: GraphStructure,
    rng_key: torch.Generator | int = 0,
) -> Dict[str, Dict[str, torch.Tensor]]:
    # Returns {node_name: {param_name: tensor}}
    # Calls each node's initialize_params()
```

---

## Phase 4: InferenceSGD — Energy Minimization (3 hours)

### File
- `bioplausible/graph/inference.py`

### PC Energy Function (Explicit Specification)
For graph with nodes and edges defining parent→child relationships:

```
E = Σ_{(parent, child) ∈ edges} || a_child - f_parent(a_parent, θ_parent) ||²
```

Where:
- `a_child` = current activity of child node
- `f_parent` = parent node's `forward()` function
- `θ_parent` = parent node's parameters
- `|| · ||²` = squared error (prediction error)

**Inference (settling) updates ACTIVITIES, not weights:**
```
a_child ← a_child - η_infer * ∂E/∂a_child
```

This gradient is **LOCAL** — depends only on prediction error at that node and its children. No full-graph backprop.

### Implementation via `torch.func.grad`
```python
def node_energy(a_child, a_parent, params_parent):
    prediction = parent_node.forward(a_parent, **params_parent)
    return ((a_child - prediction) ** 2).sum()


grad_a = torch.func.grad(node_energy, argnums=0)(a_child, a_parent, params_parent)
a_child = a_child - eta_infer * grad_a
```

### `InferenceSGD` Class
```python
class InferenceSGD:
    eta_infer: float = 0.05
    infer_steps: int = 20
    
    def settle(
        self,
        structure: GraphStructure,
        params: Dict[str, Dict[str, torch.Tensor]],
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:  # node_name -> settled activity
        """
        1. Initialize activities (zeros or feedforward pass)
        2. For each step:
           a. For each node in topological order (or any order for cyclic):
              - Get parent activities via edges
              - Compute prediction = parent.forward(parent_activity, **params)
              - Compute energy gradient w.r.t. this node's activity
              - Update activity
           b. If y provided: clamp output node activity toward target
        3. Return settled activities
        """
```

### Fallback if `torch.func.grad` fails:
- Use `torch.autograd.grad` with explicit inputs/outputs
- Reduce `eta_infer` to 0.01, increase `infer_steps` to 50
- Verify energy decreases monotonically

### Tests
- `tests/graph/test_inference.py`: energy decreases, supervised/unsupervised modes, 1-step degenerate case

---

## Phase 5: Training — train_pcn & train_backprop (3 hours)

### File
- `bioplausible/graph/training.py`

### `train_backprop` (Standard Autograd)
```python
def train_backprop(
    structure: GraphStructure,
    params: Dict[str, Dict[str, torch.Tensor]],
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> Dict[str, float]:
    """
    1. Feedforward through graph in topological_order()
    2. Compute loss (cross-entropy)
    3. Standard torch.autograd backward
    4. Update params with SGD/Adam
    5. Return metrics dict
    """
```

### `train_pcn` (Predictive Coding)
```python
def train_pcn(
    structure: GraphStructure,
    params: Dict[str, Dict[str, torch.Tensor]],
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> Dict[str, float]:
    """
    Per batch:
    1. Run inference.settle() to get settled activities
    2. Compute prediction errors at each node
    3. For each node, weight gradient = local gradient from its prediction error
       Use torch.func.grad on node.forward() independently
    4. Update weights
    5. Return metrics dict
    """
```

**Critical Requirement:** Both functions accept **exact same** `GraphStructure` and `params`. User defines graph once, switches training mode.

### Tests
- `tests/graph/test_training.py`: both reduce loss on synthetic data (100 samples, linearly separable), backprop >90%, PC >80% after 3 epochs

---

## Phase 6: FabricPCGraphPCN BioModel Wrapper (1.5 hours)

### File
- `bioplausible/models/fabricpc_graph_pcn.py`

### Requirements
- Subclass `BioModel` (from `bioplausible.models.base`)
- `@register_model("fabricpc_graph_pcn")`
- `train_step(x, y)` internally:
  - Builds/uses fixed `GraphStructure` (784→256→10) from `config.hidden_dims`
  - Calls `train_pcn(structure, params, ...)` for PC mode
  - Calls `train_backprop(structure, params, ...)` for backprop mode (configurable via `config.extra`)
- Returns `{"loss": ..., "accuracy": ...}`

This wrapper makes the new graph API **instantly usable** via the existing factory/trainer/demo infrastructure without any adapter work.

---

## Phase 7: MNIST Benchmark — Tier 1 Demo (2 hours)

### File
- `examples/fabricpc_mnist_bridge.py`

### Requirements
1. **Define graph once:**
   ```python
   input_node = Linear(shape=(784,), name="input")
   hidden_node = Linear(shape=(256,), name="hidden")
   output_node = Linear(shape=(10,), name="output")
   
   structure = graph(
       nodes=[input_node, hidden_node, output_node],
       edges=[
           Edge(source=input_node, target=hidden_node.slot("in")),
           Edge(source=hidden_node, target=output_node.slot("in")),
       ],
       task_map=TaskMap(x=input_node, y=output_node),
       inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
   )
   ```

2. **Train same graph with two modes:**
   ```python
   params = initialize_params(structure)
   results_bp = train_backprop(structure, params, train_loader, epochs=5)
   results_pc = train_pcn(structure, params, train_loader, epochs=5)
   print(f"Backprop: {results_bp['test_acc']:.1%}  |  PC: {results_pc['test_acc']:.1%}")
   ```

3. **Comparison table output:**
   ```
   Learning Rule     | Test Accuracy | Train Time (s)
   ------------------|---------------|----------------
   Backpropagation   | 98.2%         | 12.3
   Predictive Coding | 97.8%         | 45.1
   ```

4. **Header crediting FabricPC:**
   ```
   # Bioplausible × FabricPC Integration Demo
   # Graph API and PC training mode inspired by FabricPC (Behrend et al.)
   # https://github.com/trueagi-io/FabricPC
   # Running on PyTorch/Lightning — no JAX dependency.
   ```

5. **Use Lightning** where applicable (`SupervisedTrainer`, callbacks, logging)

---

## Phase 8: Six-Way Comparison — Tier 2 (1.5 hours)

### File
- `examples/fabricpc_six_way_comparison.py`

### Approach
**Use the existing model factory — no adapters needed.** All six rules share the same MLP architecture via `create_model(spec, 784, 10, 256, 2, device)`:

| Rule | Model Spec (from `MODEL_REGISTRY`) | Model Type |
|------|-----------------------------------|------------|
| Backprop | "Backprop Baseline" | `"backprop"` (task_type="vision" → MLP) |
| Predictive Coding | (new) | `"fabricpc_graph_pcn"` |
| Equilibrium Propagation | "EqProp MLP" | `"eqprop_mlp"` |
| Contrastive Hebbian | "CHL (Contrastive Hebbian)" | `"chl"` |
| Hebbian Learning | "Deep Hebbian (Hundred-Layer)" | `"deep_hebbian"` |
| Forward-Forward | "Forward-Forward" | `"forward_forward"` |

### Code Pattern
```python
from bioplausible.models.registry import get_model_spec, MODEL_REGISTRY
from bioplausible.models.factory import create_model
from bioplausible.training.supervised import SupervisedTrainer

specs = [
    ("Backprop", get_model_spec("Backprop Baseline")),
    ("Predictive Coding", get_model_spec("Predictive Coding Hybrid")),  # or new one
    ("EqProp", get_model_spec("EqProp MLP")),
    ("CHL", get_model_spec("CHL (Contrastive Hebbian)")),
    ("Hebbian", get_model_spec("Deep Hebbian (Hundred-Layer)")),
    ("Forward-Forward", get_model_spec("Forward-Forward")),
]

for name, spec in specs:
    model = create_model(spec, 784, 10, 256, 2, device, task_type="vision")
    trainer = SupervisedTrainer(model, device=device, lr=spec.default_lr, ...)
    # train + eval
    # collect metrics
```

This produces the **six-row comparison table** with zero custom integration work because Bioplausible already unified the build/train contract.

---

## Phase 9: Documentation (1.5 hours)

### `FABRICPC_INTEGRATION.md` (Repo Root)
- Overview, what adopted from FabricPC, what Bioplausible adds
- Architecture mapping table (FabricPC concept → Bioplausible location)
- Roadmap (Tier 3 items)
- Credits section

### `README.md` Update
- Add integration section after existing content
- Code snippet showing graph definition + two training modes
- Link to `examples/fabricpc_mnist_bridge.py`
- Link to `FABRICPC_INTEGRATION.md`

### Docstrings
Every public class/function in `bioplausible/graph/`:
- What it does
- FabricPC equivalent note
- Pure-function requirement for `forward()`

---

## Phase 10: Submission Materials (1 hour)

### `docs/open_build_submission.md`
- **Project Title:** `Bioplausible: A Unified Framework for Biologically Plausible Learning, Integrating Predictive Coding`
- **Abstract (≤2000 chars):** PC as one paradigm among several; FabricPC graph API adopted; six learning rules on one graph architecture (via existing factory); invitation to collaborate
- **Project Link:** `https://github.com/autonull/bioplausible`
- **Demo Link:** `https://github.com/autonull/bioplausible/blob/main/examples/fabricpc_mnist_bridge.py`
- **Demo Video Script (2-3 min):**
  1. Show graph definition (10 lines)
  2. Run `train_pcn` and `train_backprop` on same graph
  3. Show six-way comparison table
  4. Closing: *"Predictive coding is biologically plausible. Now it lives alongside every other biologically plausible learning rule, in one framework, on one graph."*

---

## Phase 11: Test Suite & CI (1.5 hours)

### Required Tests (All Pass Before Submission)
| Test File | Coverage |
|-----------|----------|
| `tests/graph/test_torch_func_verification.py` | **Critical** — torch.func.grad on Linear node |
| `tests/graph/test_nodes.py` | All node types: init, forward shape, torch.func.grad |
| `tests/graph/test_topology.py` | Feedforward, cyclic, self-recurrent, skip, validation errors |
| `tests/graph/test_inference.py` | Energy decreases, supervised/unsupervised, 1-step |
| `tests/graph/test_training.py` | Both train on synthetic data, loss decreases |
| `tests/graph/test_mnist_smoke.py` | `@pytest.mark.slow` — 1 epoch, subset, all rules complete |

### Commands
```bash
# Fast tests (exclude slow)
pytest tests/graph/ -v -m "not slow"

# Full suite
pytest tests/graph/ -v

# Linting (all four)
ruff check .
black --check .
isort --check-only .
flake8 .
```

---

## Dependencies & Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 (pyproject.toml) |
| PyTorch | ≥ 2.0 (stable `torch.func` since 2.1) |
| Lightning | ≥ 2.0 (already in deps) |
| JAX | **FORBIDDEN** — no import anywhere |

---

## Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `torch.func.grad` fails on node ops | Medium | High | Fallback to `torch.autograd.grad`; test early (Phase 1) |
| InferenceSGD energy doesn't decrease | Medium | High | Reduce η, increase steps; verify energy formula; debug with 2-node graph |
| PC training diverges / low accuracy | Medium | Medium | Tune `eta_infer`, `infer_steps`, `lr`; compare to FabricPC defaults |
| `fabricpc_graph_pcn` BioModel doesn't integrate with trainer | Low | High | Follow `pc_hybrid.py` pattern exactly; test with `SupervisedTrainer` early |
| Six-way comparison too slow for CI | Low | Low | Use MNIST subset (1000/200) + `@pytest.mark.slow` |

---

## Execution Order Checklist

```
[ ] Phase 0: Codebase orientation (30 min)
[ ] Phase 1: Nodes + torch.func verification test (2 hr)
[ ] Phase 2: Topology (1.5 hr)
[ ] Phase 3: Initialization (30 min)
[ ] Phase 4: InferenceSGD (3 hr)
[ ] Phase 5: train_pcn + train_backprop (3 hr)
[ ] Phase 6: FabricPCGraphPCN BioModel wrapper (1.5 hr)
[ ] Phase 7: MNIST Tier 1 demo (2 hr)
[ ] Phase 8: Six-way comparison (1.5 hr) — OPTIONAL but high value
[ ] Phase 9: Documentation (1.5 hr)
[ ] Phase 10: Submission materials (1 hr)
[ ] Phase 11: Test suite + CI (1.5 hr)
```

**Total: ~18.5 hours** (with buffer for debugging)

---

## Submission Readiness Gate

**Do not submit until:**
- [ ] `examples/fabricpc_mnist_bridge.py` runs without error
- [ ] Comparison table prints (backprop vs PC)
- [ ] `FABRICPC_INTEGRATION.md` exists with credits
- [ ] `docs/open_build_submission.md` exists
- [ ] Fast test suite passes (`pytest tests/graph/ -m "not slow"`)
- [ ] All linters clean: `ruff check . && black --check . && isort --check-only . && flake8 .`

---

## Post-Submission (Tier 3) — Roadmap

See `FABRICPC_INTEGRATION.md` for full roadmap. Priority order:
1. `ConvNode` + `MaxPool` + conv MNIST demo
2. `torch.compile` on inference loop
3. Multi-GPU via Lightning
4. Depth-scaling experiments (FabricPC `examples/scaling/`)
5. AutoScientist hyperparameter search
6. TransformerBlock, Hopfield nodes
7. JAX backend as optional accelerator (interop)

---

## Credits (Reiterated in Every New File)

> The graph abstraction, PC training mode, and local autodiff pattern are adapted from [FabricPC](https://github.com/trueagi-io/FabricPC) (MIT License), created by Dr. Matthew Behrend and maintained by SingularityNET as part of the Artificial Superintelligence Alliance. We gratefully acknowledge their contribution to the predictive coding research community.