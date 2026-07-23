# Phase 2: Week 1-2 Action Plan

**Focus:** Memory profiling and baseline establishment
**Duration:** 2 weeks (10 working days)
**Deliverable:** Complete memory profile identifying optimization targets

---

## Week 1: Memory Profiling

### Day 1-2: Setup and Baseline

**Tasks:**
1. Create memory profiling script
2. Run baseline measurements at multiple depths
3. Document current memory usage

**Script to create:**
```python
# examples/profile_memory_detailed.py
"""
Detailed memory profiling for EP vs backprop.

Measures:
- Activation memory (excluding weights)
- Memory by component (settling, energy, contrast)
- Memory vs depth scaling
"""

import torch
import torch.nn as nn
from mep import smep

def measure_activation_memory(model, x, y, method='ep'):
    """Measure activation memory only (exclude weights)."""
    torch.cuda.reset_peak_memory_stats()
    
    # Get weight memory
    weight_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Training step
    if method == 'ep':
        optimizer.step(x=x, target=y)
    else:
        loss.backward()
    
    # Get peak memory
    peak_mem = torch.cuda.max_memory_allocated()
    activation_mem = peak_mem - weight_mem
    
    return {
        'total_mb': peak_mem / 1e6,
        'weight_mb': weight_mem / 1e6,
        'activation_mb': activation_mem / 1e6,
    }

# Test at multiple depths
depths = [10, 50, 100, 200, 500, 1000]
for depth in depths:
    # Create model and data
    model = make_deep_mlp(num_layers=depth)
    x = torch.randn(32, 784).cuda()
    y = torch.randint(0, 10, (32,)).cuda()
    
    # Measure
    ep_mem = measure_activation_memory(model, x, y, method='ep')
    bp_mem = measure_activation_memory(model, x, y, method='backprop')
    
    print(f"Depth {depth}: EP={ep_mem['activation_mb']:.1f}MB, BP={bp_mem['activation_mb']:.1f}MB")
```

**Expected output:**
```
Depth 10: EP=19.6MB, BP=18.3MB
Depth 100: EP=31.6MB, BP=19.8MB
Depth 500: EP=84.9MB, BP=26.3MB
...
```

**Deliverable:** Baseline memory vs depth curve

---

### Day 3-4: Component Profiling

**Tasks:**
1. Profile memory by EP component
2. Identify which operations trigger activation storage
3. Quantify overhead from each component

**Components to profile:**
- Settling loop (free phase)
- Settling loop (nudged phase)
- Energy computation
- Contrast step (gradient computation)
- Parameter update

**Script additions:**
```python
def profile_by_component():
    """Profile memory usage by EP component."""
    
    # Profile settling
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(...) as prof:
        states_free = settler.settle(model, x, None, beta=0.0, ...)
    settling_mem = torch.cuda.max_memory_allocated()
    
    # Profile energy computation
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(...) as prof:
        E_free = energy_fn(model, x, states_free, ...)
    energy_mem = torch.cuda.max_memory_allocated()
    
    # Profile contrast
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(...) as prof:
        contrast_loss = (E_nudged - E_free) / beta
        grads = torch.autograd.grad(contrast_loss, params)
    contrast_mem = torch.cuda.max_memory_allocated()
    
    return {
        'settling_mb': settling_mem / 1e6,
        'energy_mb': energy_mem / 1e6,
        'contrast_mb': contrast_mem / 1e6,
    }
```

**Expected findings:**
- Settling loop: Major contributor (stores intermediate states)
- Energy computation: Moderate (autograd graph construction)
- Contrast step: Major (full graph for parameter gradients)

**Deliverable:** Component breakdown showing optimization targets

---

### Day 5: PyTorch Operation Analysis

**Tasks:**
1. Identify which PyTorch operations trigger activation storage
2. Test `no_grad()` impact on each component
3. Document findings

**Tests to run:**
```python
# Test 1: Standard forward (with autograd)
with torch.enable_grad():
    h = layer(x)  # Stores activation for backward
print(f"With grad: {torch.cuda.memory_allocated() / 1e6:.1f}MB")

# Test 2: no_grad forward
with torch.no_grad():
    h = layer(x)  # No activation storage
print(f"No grad: {torch.cuda.memory_allocated() / 1e6:.1f}MB")

# Test 3: Manual forward
with torch.no_grad():
    h = x @ weight.t() + bias  # Direct computation
print(f"Manual: {torch.cuda.memory_allocated() / 1e6:.1f}MB")
```

**Expected findings:**
- `nn.Module` forward with `enable_grad()` stores activations
- `no_grad()` prevents storage
- Manual operations (direct matmul) have minimal overhead

**Deliverable:** List of operations to replace with manual versions

---

## Week 2: O(1) Prototype

### Day 6-7: Manual Settling Implementation

**Tasks:**
1. Implement `settle_manual()` without autograd
2. Test correctness vs current settling
3. Measure memory savings

**Implementation:**
```python
def settle_manual(model, x, target, beta, energy_fn, structure,
                  steps=30, lr=0.15):
    """Manual settling without autograd overhead."""
    
    # Capture initial states (no_grad)
    with torch.no_grad():
        states = capture_states_no_grad(model, x, structure)
    
    momentum_buffers = [torch.zeros_like(s) for s in states]
    
    for step in range(steps):
        # Manual energy computation (no_grad)
        with torch.no_grad():
            E = manual_energy_compute(model, x, states, structure, target, beta)
        
        # Manual gradient computation
        # Option: Use torch.autograd.grad on detached states
        states_detached = [s.detach().requires_grad_(True) for s in states]
        E_recompute = manual_energy_compute(model, x, states_detached, ...)
        grads = torch.autograd.grad(E_recompute, states_detached)
        
        # Update states (no_grad)
        with torch.no_grad():
            for i, (state, buf, g) in enumerate(zip(states, momentum_buffers, grads)):
                buf.mul_(0.5).add_(g)
                state.sub_(buf, alpha=lr)
    
    return states
```

**Testing:**
```python
# Compare manual vs current settling
states_current = settler.settle(model, x, target, beta, ...)
states_manual = settle_manual(model, x, target, beta, ...)

# Check state similarity
for s1, s2 in zip(states_current, states_manual):
    diff = (s1 - s2).abs().mean()
    print(f"State diff: {diff:.6f}")  # Should be < 1e-5
```

**Deliverable:** Working manual settling with verified correctness

---

### Day 8-9: No-Grad Energy Computation

**Tasks:**
1. Implement `manual_energy_compute()` without autograd
2. Replace `nn.Linear` forward with manual matmul
3. Test energy correctness

**Implementation:**
```python
def manual_energy_compute(model, x, states, structure, target_vec, beta):
    """Compute EP energy without autograd overhead."""
    E = torch.tensor(0.0, device=x.device, dtype=torch.float32)
    batch_size = x.shape[0]
    prev = x
    state_idx = 0
    
    for item in structure:
        if item["type"] == "layer":
            module = item["module"]
            state = states[state_idx]
            
            # Manual forward pass (no autograd)
            with torch.no_grad():
                h = prev @ module.weight.t() + module.bias
                h = torch.relu(h)
            
            # Manual MSE (no graph)
            mse = ((h.float() - state.float()) ** 2).sum() / batch_size
            E = E + 0.5 * mse
            
            prev = state
            state_idx += 1
        
        elif item["type"] in ["norm", "pool", "flatten", "dropout"]:
            # Skip or apply manually
            pass
    
    # Nudge term
    if target_vec is not None and beta > 0:
        if loss_type == "cross_entropy":
            nudge = beta * F.cross_entropy(prev, target_vec, reduction="sum") / batch_size
        else:
            nudge = beta * ((prev - target_vec) ** 2).sum() / batch_size
        E = E + nudge
    
    return E
```

**Testing:**
```python
# Compare energy values
E_current = energy_fn(model, x, states, structure, target, beta)
E_manual = manual_energy_compute(model, x, states, structure, target, beta)

print(f"Energy diff: {(E_current - E_manual).abs().item():.8f}")  # Should be ~0
```

**Deliverable:** Manual energy computation with verified correctness

---

### Day 10: Integration and Initial Results

**Tasks:**
1. Integrate manual settling + manual energy
2. Run memory comparison vs current implementation
3. Document initial savings

**Test script:**
```python
# Compare memory usage
depths = [10, 50, 100, 200, 500]

print("Depth | Current EP | Manual EP | Savings")
print("-" * 50)

for depth in depths:
    model = make_deep_mlp(num_layers=depth).cuda()
    x = torch.randn(32, 784).cuda()
    y = torch.randint(0, 10, (32,)).cuda()
    
    # Current implementation
    torch.cuda.reset_peak_memory_stats()
    states_current = settler.settle(model, x, y, beta=0.3, ...)
    current_mem = torch.cuda.max_memory_allocated() / 1e6
    
    # Manual implementation
    torch.cuda.reset_peak_memory_stats()
    states_manual = settle_manual(model, x, y, beta=0.3, ...)
    manual_mem = torch.cuda.max_memory_allocated() / 1e6
    
    savings = (1 - manual_mem / current_mem) * 100
    print(f"{depth:5d} | {current_mem:10.1f}MB | {manual_mem:9.1f}MB | {savings:6.1f}%")
```

**Expected results:**
```
Depth | Current EP | Manual EP | Savings
--------------------------------------------------
   10 |       19.6MB |      18.2MB |   7.1%
  100 |       31.6MB |      22.4MB |  29.1%
  500 |       84.9MB |      25.8MB |  69.6%
```

**Deliverable:** Initial O(1) memory results showing progress

---

## Success Criteria for Week 1-2

| Metric | Target | Status |
|--------|--------|--------|
| Baseline memory curve | ✅ Complete | |
| Component breakdown | ✅ Complete | |
| PyTorch operation analysis | ✅ Complete | |
| Manual settling working | ✅ Correctness verified | |
| Manual energy working | ✅ Correctness verified | |
| Initial memory savings | ✅ 50%+ at depth 500 | |

---

## Blockers and Escalation

| Blocker | Action |
|---------|--------|
| Manual settling diverges | Debug gradient computation; compare to current |
| Energy values don't match | Check numerical precision; verify formulas |
| Memory not reduced | Profile again; identify remaining autograd usage |
| Too slow | Profile timing; optimize hot paths |

---

## Next Steps (Week 3-4)

After Week 1-2 success:
1. Extend manual approach to contrast step
2. Implement gradient checkpointing for EP
3. Test at 1000+ layer depth
4. Document full O(1) results

---

*Created: 2026-02-18*
*Status: Ready to execute*
