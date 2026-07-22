# FABRICPC Integration — Phased Development Plan

**Status:** DRAFT — Incorporates all feedback from evaluation  
**Target:** BGI Open Build submission (SingularityNET / AGI Society)  
**Deadline:** July 24, 2026  
**Working Time:** ~48 hours effective

---

## Executive Summary

Integrate FabricPC's graph-based topology abstraction and predictive coding training mode into Bioplausible. The deliverable is a **working Tier 1 demo** showing `train_pcn` vs `train_backprop` on the same graph definition, plus documentation and submission materials. Tier 2 adds Equilibrium Propagation on the same graph. Tier 3 is post-submission roadmap.

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
| **MNIST demo** | 784→256→10 graph, backprop vs PC comparison table | ☐ |
| **FABRICPC_INTEGRATION.md** | Architecture mapping, credits, roadmap | ☐ |
| **Submission abstract** | `docs/open_build_submission.md` | ☐ |
| **README update** | Integration section with code snippet | ☐ |

### TIER 2 — STRONGLY DESIRED (Compelling Demo)

| Item | Description | Done? |
|------|-------------|-------|
| **EqProp adapter** | Graph → EquiTile EP mode (two-phase settling) | ☐ |
| **Three-way table** | Backprop, PC, EqProp on same graph | ☐ |
| **Unit tests** | Graph assembly, training, torch.func verification | ☐ |

### TIER 3 — STRETCH (Post-Submission Roadmap)

| Item | Description |
|------|-------------|
| `ConvNode`, `MaxPool`, `Hopfield`, `TransformerBlock` | |
| CHL, Hebbian, Forward-Forward on graph | |
| Full six-way comparison table | |
| Conv MNIST demo (`examples/fabricpc_conv_demo.py`) | |
| `torch.compile` acceleration | |
| Multi-GPU via Lightning DDP/FSDP | |
| Depth-scaling experiments | |
| Hyperparameter search with AutoScientist | |

---

## Phase 0: Codebase Orientation (30 min)

**Before writing any code, read and internalize:**

| File/Directory | Purpose |
|----------------|---------|
| `bioplausible/training/` | Zoo learning rules — find `LearningRule` interface, registration |
| `bioplausible/models/` | Architecture definitions — find `EquiTile`, `EquilibriumTile` |
| `bioplausible/tiles/` | Tile implementations — PC/EP modes, settling loops |
| `tests/` | Testing patterns, fixtures, `conftest.py` |
| `examples/` | Demo structure |
| `pyproject.toml` / `setup.py` | Dependencies, Python version, Lightning version |

**Key questions to answer:**
- What is the `LearningRule` interface? (method signatures, config dataclass, registration decorator)
- How does `EquiTile` run EP mode? (two phases, β parameter, weight update formula)
- How are existing demos structured? (LightningModule? raw training loop?)
- What test runner? (pytest? fixtures?)

**Do not proceed until you can answer these.**

---

## Phase 1: Graph Abstraction — Nodes (2 hours)

### Files
- `bioplausible/graph/nodes.py`
- `bioplausible/graph/__init__.py` (exports)

### Requirements
1. **`Slot` class** — named input port with `name` and `owner: NodeBase`
2. **`NodeBase` abstract class** with contract:
   - `forward(**slot_inputs) -> torch.Tensor` — **PURE FUNCTION** (no in-place mutation, no side effects)
   - `get_slots() -> dict[str, Slot]`
   - `slot(name) -> Slot`
   - `initialize_params(rng_key: torch.Generator) -> dict[str, torch.Tensor]`
   - `name: str`
3. **`Linear(shape, name)`** — slots `{"in"}`, params `weight`, `bias`, forward = `F.linear`
4. **`ReLU(name)`, `Tanh(name)`** — slots `{"in"}`, no params, forward = `F.relu` / `F.tanh`

### Critical Verification Test (Write First)
```python
# tests/graph/test_torch_func_verification.py
def test_torch_func_grad_on_linear_node():
    node = Linear(shape=(784, 256), name="test")
    params = node.initialize_params(torch.Generator().manual_seed(0))
    x = torch.randn(32, 784)
    
    def energy(p):
        out = node.forward(x, p["weight"], p["bias"])
        return (out ** 2).sum()
    
    grads = torch.func.grad(energy)(params)
    assert grads["weight"].shape == params["weight"].shape
    assert grads["bias"].shape == params["bias"].shape
```

**If this test fails, STOP. Fix the node implementation.** This is the single most important technical verification.

### File Header (Every New File)
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
  - `nodes: list[NodeBase]`
  - `edges: list[Edge]`
  - `task_map: TaskMap`
  - `inference: InferenceSGD`
  - `topological_order() -> list[NodeBase]` — Kahn's algorithm for feedforward
  - `get_predecessors(node) -> list[tuple[NodeBase, Slot]]`
  - `validate()` — check: all slots have exactly one incoming edge, no dangling edges, task_map nodes in graph. **Cycles permitted.**
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
) -> dict[str, dict[str, torch.Tensor]]:
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
        params: dict[str, dict[str, torch.Tensor]],
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:  # node_name -> settled activity
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
    params: dict,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> dict:
    """
    1. Feedforward through graph in topological_order()
    2. Compute loss (cross-entropy)
    3. Standard torch.autograd backward
    4. Update params with SGD/Adam
    5. Return metrics
    """
```

### `train_pcn` (Predictive Coding)
```python
def train_pcn(
    structure: GraphStructure,
    params: dict,
    dataloader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> dict:
    """
    Per batch:
    1. Run inference.settle() to get settled activities
    2. Compute prediction errors at each node
    3. For each node, weight gradient = local gradient from its prediction error
       Use torch.func.grad on node.forward() independently
    4. Update weights
    5. Return metrics
    """
```

**Critical Requirement:** Both functions accept **exact same** `GraphStructure` and `params`. User defines graph once, switches training mode.

### Tests
- `tests/graph/test_training.py`: both reduce loss on synthetic data (100 samples, linearly separable), backprop >90%, PC >80% after 3 epochs

---

## Phase 6: MNIST Benchmark — Tier 1 Demo (2 hours)

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

5. **Use Lightning** where applicable (Trainer, callbacks, logging)

---

## Phase 7: EqProp Adapter — Tier 2 (2 hours)

### Concept
EqProp does **NOT** use `InferenceSGD`. It uses `EquiTile`'s existing EP settling loop (two phases, β parameter). The graph provides **topology**; `EquiTile` provides **dynamics**.

### Adapter Contract
```python
# In GraphStructure or separate adapter
def to_equitile_layers(self) -> list[nn.Module]:
    """Flatten feedforward graph into sequential layers for EquiTile."""
    
def get_node_activities(self) -> dict[str, Tensor]:
    """Return settled activities for all nodes. Used by EP two-phase."""
    
def get_prediction_errors(self) -> dict[str, Tensor]:
    """Return per-node prediction errors. Used by PC weight updates."""
```

### Integration Steps
1. Extract layer sequence from graph (feedforward only)
2. Pass to `EquiTile` in EP mode
3. Run free phase (settle), nudged phase (clamp output + β, re-settle)
4. Weight update: `ΔW ∝ (a_nudged - a_free) / β`
5. Wrap as `PredictiveCoding` learning rule in zoo (or separate `EquilibriumPropagationGraph` rule)

### Fallback
If adapter cannot be built cleanly in time → **drop EqProp from demo**. Show backprop vs PC only. Note in documentation that EqProp integration is in progress.

---

## Phase 8: Documentation (1.5 hours)

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

## Phase 9: Submission Materials (1 hour)

### `docs/open_build_submission.md`
- **Project Title:** `Bioplausible: A Unified Framework for Biologically Plausible Learning, Integrating Predictive Coding`
- **Abstract (≤2000 chars):** PC as one paradigm among several; FabricPC graph API adopted; six learning rules on one graph (aspirational); invitation to collaborate
- **Project Link:** `https://github.com/autonull/bioplausible`
- **Demo Link:** `https://github.com/autonull/bioplausible/blob/main/examples/fabricpc_mnist_bridge.py`
- **Demo Video Script (2-3 min):**
  1. Show graph definition (10 lines)
  2. Run `train_pcn` and `train_backprop` on same graph
  3. Show comparison table
  4. Closing: *"Predictive coding is biologically plausible. Now it lives alongside every other biologically plausible learning rule, in one framework, on one graph."*

---

## Phase 10: Test Suite & CI (1.5 hours)

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
```

---

## Dependencies & Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.1 (stable `torch.func`) |
| Lightning | ≥ 2.0 |
| JAX | **FORBIDDEN** — no import anywhere |

---

## Code Style (Enforced)

- **Format:** `black .`
- **Imports:** `isort .`
- **Lint:** `flake8` — zero errors
- **Type hints:** All public APIs
- **Docstrings:** Google-style on all public classes/functions
- **Line length:** 88 (Black default), up to 100 for scientific expressions

---

## Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `torch.func.grad` fails on node ops | Medium | High | Fallback to `torch.autograd.grad`; test early (Phase 1) |
| InferenceSGD energy doesn't decrease | Medium | High | Reduce η, increase steps; verify energy formula; debug with 2-node graph |
| EqProp adapter too complex | High | Medium (Tier 2) | **Explicit fallback:** drop EqProp, demo backprop vs PC only |
| Graph validation too strict for cycles | Low | Medium | Permit cycles in `validate()`; topological_order only for feedforward |
| Zoo LearningRule interface mismatch | Medium | Medium | Read existing zoo FIRST (Phase 0); conform exactly |
| MNIST demo too slow for submission | Low | High | Use small subset (1000 train / 200 test) for smoke test; full run optional |

---

## Execution Order Checklist

```
[ ] Phase 0: Codebase orientation (30 min)
[ ] Phase 1: Nodes + torch.func verification test (2 hr)
[ ] Phase 2: Topology (1.5 hr)
[ ] Phase 3: Initialization (30 min)
[ ] Phase 4: InferenceSGD (3 hr)
[ ] Phase 5: train_pcn + train_backprop (3 hr)
[ ] Phase 6: MNIST Tier 1 demo (2 hr)
[ ] Phase 7: EqProp adapter (2 hr) — OPTIONAL, fallback ready
[ ] Phase 8: Documentation (1.5 hr)
[ ] Phase 9: Submission materials (1 hr)
[ ] Phase 10: Test suite + CI (1.5 hr)
```

**Total: ~17.5 hours** (with buffer for debugging)

---

## Submission Readiness Gate

**Do not submit until:**
- [ ] `examples/fabricpc_mnist_bridge.py` runs without error
- [ ] Comparison table prints (backprop vs PC)
- [ ] `FABRICPC_INTEGRATION.md` exists with credits
- [ ] `docs/open_build_submission.md` exists
- [ ] Fast test suite passes (`pytest tests/graph/ -m "not slow"`)
- [ ] No `flake8` errors, code formatted with `black` + `isort`

---

## Post-Submission (Tier 3) — Roadmap

See `FABRICPC_INTEGRATION.md` for full roadmap. Priority order:
1. ConvNode + MaxPool + conv MNIST demo
2. CHL, Hebbian, Forward-Forward on graph
3. Six-way comparison table
4. `torch.compile` on inference loop
5. Multi-GPU via Lightning
6. Depth-scaling experiments
7. AutoScientist hyperparameter search
8. TransformerBlock, Hopfield nodes

---

## Credits (Reiterated in Every File)

> The graph abstraction, PC training mode, and local autodiff pattern are adapted from [FabricPC](https://github.com/trueagi-io/FabricPC) (MIT License), created by Dr. Matthew Behrend and maintained by SingularityNET as part of the Artificial Superintelligence Alliance. We gratefully acknowledge their contribution to the predictive coding research community.