# Equilibrium Propagation Research Archive

This directory contains research implementations of strict Equilibrium Propagation (EP) for EquiTile.

## Status: **Research/Experimental**

The EP implementation is preserved here for:
- Neuroscience research applications
- Algorithm comparison studies
- Future optimization work

## What Is Equilibrium Propagation?

Equilibrium Propagation (Scellier & Bengio, 2017) is a learning algorithm that:
- Uses two-phase relaxation (free + nudged)
- Computes weight updates via contrastive Hebbian learning
- Requires no error backpropagation through the computational graph
- Is strictly local (each synapse only needs pre/post activities)

## Why Archived?

EP mode is archived (not removed) because:

| Metric | PC Mode | EP Mode |
|--------|---------|---------|
| **Accuracy** | 97%+ | ~23% |
| **Speed** | 0.5s/epoch | 1.2s/epoch |
| **Stability** | Stable | Requires tuning |
| **Use Case** | Production | Research |

**Conclusion:** For scalable learning systems, PC mode is the practical choice. EP mode remains valuable for research into bio-plausible credit assignment.

## Files

- `equitile_ep_demo.py` - Demonstration of EP mode
- `ep_vs_pc_comparison.py` - Detailed comparison study
- `ep_tuning_guide.md` - Tips for improving EP performance

## Using EP Mode

EP mode is still available in the main EquiTile class:

```python
from bioplausible.models import EquiTile

model = EquiTile(
    mode='ep',           # Enable EP mode
    beta=0.1,            # Nudge strength
    beta_anneal=0.99,    # Beta decay per epoch
    inference_steps_free=15,
    inference_steps_nudged=15,
    ...
)
```

Or use the convenience class:

```python
from bioplausible.models import EquiTileEP

model = EquiTileEP(
    beta=0.1,
    ...
)
```

## Research Directions

If you're working on improving EP, consider:

1. **Better initialization**: Smaller weights, layer-wise pre-training
2. **Adaptive beta**: Start high, decay during training
3. **More inference steps**: EP may need 30-50 steps
4. **Normalization**: LayerNorm within tiles
5. **Curriculum learning**: Start with easy examples

## References

- Scellier, B., & Bengio, Y. (2017). Equilibrium Propagation: Bridging the Gap Between Energy-Based Models and Backpropagation. *Frontiers in Computational Neuroscience*.
- Laborieux, A., et al. (2021). Scaling Equilibrium Propagation to Deep ConvNets. *ICLR*.
- Ernoult, M., et al. (2020). Equilibrium Propagation with Continual-Time Recurrent Neural Networks. *NeurIPS*.

---

**Note:** For production use, see the main EquiTile documentation which focuses on PC mode (Predictive Coding + Local Hebbian Learning).
