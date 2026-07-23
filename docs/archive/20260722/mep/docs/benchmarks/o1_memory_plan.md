# O(1) Memory Implementation Plan

## Objective

Achieve O(1) activation memory for EP by avoiding PyTorch functionality that triggers unnecessary activation storage.

## Motivation

EP's theoretical advantage over backpropagation is O(1) activation memory—independent of network depth. However, our current implementation shows EP using MORE memory than backprop+checkpointing because:

1. PyTorch autograd tracks operations during settling
2. Intermediate activations are stored unnecessarily
3. Standard PyTorch layers trigger gradient history

## Technical Approach

### 1. Avoid Autograd During Settling

**Current (problematic):**
```python
with torch.enable_grad():
    E = energy_fn(model, x, states, structure, target_vec, beta)
    # This triggers full autograd graph construction!
```

**Proposed (O(1)):**
```python
# Manual energy computation without autograd
with torch.no_grad():
    E = manual_energy_compute(model, x, states, structure, target_vec, beta)
    
# Then compute gradients manually only where needed
```

### 2. Manual Settling Loop

**Current:** Uses standard PyTorch operations that store history.

**Proposed:** Custom settling that:
- Operates in `no_grad()` mode
- Manually computes state updates
- Only stores current state, not history

```python
def settle_manual(model, x, states, beta, steps=30, lr=0.15):
    """Settle without autograd overhead."""
    momentum_buffers = [torch.zeros_like(s) for s in states]
    
    for step in range(steps):
        # Manual energy computation (no autograd)
        E = compute_energy_no_grad(model, x, states, beta)
        
        # Manual gradient computation
        grads = manual_state_gradients(E, states)
        
        # Manual momentum update
        for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
            buf.mul_(0.5).add_(g)
            state.sub_(buf, alpha=lr)
    
    return states
```

### 3. Custom Layer Forward (No Activation Storage)

**Current:** Standard `nn.Linear` stores activation history.

**Proposed:** Manual forward that doesn't trigger autograd:

```python
def linear_forward_no_grad(x, weight, bias):
    """Linear forward without activation storage."""
    with torch.no_grad():
        return x @ weight.t() + bias
```

### 4. Gradient Checkpointing for EP

For the contrast step (where we DO need gradients):

```python
# Store only boundary states (free and nudged)
# Recompute intermediate states during contrast
# Trade compute for memory (favorable for EP)
```

### 5. Custom CUDA Kernel

The existing `fused_settle_step_inplace` kernel should:
- Operate entirely in GPU memory
- Avoid PyTorch dispatch overhead
- Use manual gradient computation

## Implementation Plan

### Week 1-2: Profiling
- [ ] Profile current memory usage by component
- [ ] Identify which PyTorch operations trigger activation storage
- [ ] Establish baseline memory vs depth curve

### Week 3-4: Manual Settling
- [ ] Implement `settle_manual()` without autograd
- [ ] Implement `compute_energy_no_grad()`
- [ ] Test correctness vs current implementation
- [ ] Measure memory savings

### Week 5-6: Custom Layer Operations
- [ ] Implement manual forward passes for Linear, Conv2d
- [ ] Ensure no activation history stored
- [ ] Test at depth (100, 500, 1000 layers)

### Week 7-8: Integration & Testing
- [ ] Integrate all components
- [ ] Test at extreme depth (2000, 5000, 10000 layers)
- [ ] Compare vs backprop+checkpointing
- [ ] Document results

## Success Criteria

| Depth | Current EP Memory | Target EP Memory | Backprop+Checkpoint |
|-------|------------------|------------------|---------------------|
| 100   | ~32 MB | ~20 MB | ~20 MB |
| 500   | ~85 MB | ~20 MB | ~26 MB |
| 1000  | ~151 MB | ~20 MB | ~35 MB |
| 2000  | ~285 MB | ~20 MB | ~51 MB |
| 5000  | ~700 MB (est) | ~20 MB | ~100 MB |

**Goal:** EP memory flat at ~20 MB regardless of depth.

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Manual settling diverges | High | Validate against current implementation |
| Gradients incorrect | High | Numerical gradient verification |
| Too slow | Medium | CUDA kernel optimization |
| PyTorch internals still store activations | Medium | Profile and identify specific triggers |

## Files to Modify

- `mep/optimizers/settling.py` - Manual settling implementation
- `mep/optimizers/energy.py` - No-grad energy computation
- `mep/optimizers/strategies/gradient.py` - Manual gradient computation
- `mep/cuda/kernels.py` - Enhanced settling kernel

## References

- Bioplausible Track 35 methodology
- `examples/validate_memory_scaling.py` - Memory measurement script
- Current memory results: `docs/benchmarks/VALIDATION_RESULTS.md`

---

*Created: 2026-02-18*
*Status: Planning phase*
