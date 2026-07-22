# Bioplausible Open Build Submission

## Project Title

**Bioplausible: A Unified Framework for Biologically Plausible Learning, Integrating Predictive Coding**

## Abstract

Bioplausible is an open-source framework that unifies a diverse spectrum of biologically plausible learning rules — backpropagation, predictive coding, equilibrium propagation, contrastive Hebbian learning, deep Hebbian, forward-forward, feedback alignment, and more — under a single, extensible architecture. All models share a uniform `build()` contract and `train_step()` interface, enabling drop-in comparison of learning rules on identical tasks.

This submission integrates FabricPC's node-graph topology abstraction into Bioplausible, adding predictive coding (PC) as a first-class learning rule alongside the existing six others. The graph API (`GraphStructure`, `Edge`, `Slot`, `InferenceSGD`) allows users to define a network topology once and train it with either backpropagation or predictive coding, with the same code path. A `BioModel` wrapper (`FabricPCGraphPCN`) plugs the graph into Bioplausible's existing model factory, trainer, and benchmarking infrastructure without any adapter work.

The result is a single framework where six learning rules can be compared on the same MLP architecture at the same task, enabling fair, reproducible research. The full six-way comparison (backprop, PC, EqProp, CHL, Hebbian, Forward-Forward) runs via `create_model()`, requiring no custom integration.

## Project Link

[https://github.com/autonull/bioplausible](https://github.com/autonull/bioplausible)

## Demo Link

[https://github.com/autonull/bioplausible/blob/main/examples/fabricpc_mnist_bridge.py](https://github.com/autonull/bioplausible/blob/main/examples/fabricpc_mnist_bridge.py)

## Demo Video Script (2-3 minutes)

1. **Graph Definition (10 lines):** Show the `graph()` call — three nodes, two edges, one `TaskMap`. Emphasize "define once, train twice."
2. **Backprop vs PC:** Run `train_backprop()` and `train_pcn()` on the same graph, print comparison table side by side.
3. **Six-Way Comparison:** Run `fabricpc_six_way_comparison.py` showing all six rules on the same architecture.
4. **Closing:** "Predictive coding is biologically plausible. Now it lives alongside every other biologically plausible learning rule, in one framework, on one graph."

## Technical Details

- **Language:** Python 3.9+
- **Framework:** PyTorch 2.0+ (no JAX dependency)
- **Key Files:**
  - `bioplausible/graph/nodes.py` — Slot, NodeBase, Linear, ReLU, Tanh
  - `bioplausible/graph/topology.py` — Edge, GraphStructure, TaskMap, graph()
  - `bioplausible/graph/inference.py` — InferenceSGD (PC settling)
  - `bioplausible/graph/training.py` — train_backprop, train_pcn
  - `bioplausible/graph/initialization.py` — parameter initialization
  - `bioplausible/models/fabricpc_graph_pcn.py` — BioModel wrapper
- **Tests:** 55 unit tests in `tests/graph/`, including torch.func.grad verification
