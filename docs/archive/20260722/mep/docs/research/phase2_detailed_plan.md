# Phase 2: Technical Excellence — Detailed Plan

**Timeline:** Q2-Q3 2026 (approximately 6 months)
**Goal:** Achieve compelling technical advantages that differentiate EP from backpropagation
**Philosophy:** Results before outreach—build undeniable advantages first

---

## Executive Summary

Phase 2 focuses on four technical priorities that, if successful, will establish EP as a genuinely different alternative to backpropagation—not just biologically plausible, but technically superior in specific domains.

| Priority | Goal | Timeline | Impact |
|----------|------|----------|--------|
| 1. O(1) Memory | Flat memory vs depth | Months 1-3 | Very High |
| 2. Deep Scaling | Train 10000+ layers | Months 3-4 | High |
| 3. Continual Learning | EP+EWC <15% forgetting | Months 2-4 | Medium-High |
| 4. Speed | Reduce 2× → 1.5× | Months 4-5 | Medium |

**Success =** EP can train networks that are impractical for backprop (due to memory) while maintaining accuracy parity.

---

## Priority 1: O(1) Memory Implementation

### Technical Challenge

Current EP memory scales linearly with depth (O(depth)), same as backprop. This contradicts EP's theoretical advantage. The culprit: PyTorch's autograd system stores intermediate activations even when we don't need them.

### Root Cause Analysis

**What triggers activation storage:**

1. **`torch.enable_grad()` context** — PyTorch builds computation graph
2. **Standard layer forward passes** — `nn.Linear`, `nn.Conv2d` store inputs for backward
3. **In-place operation detection** — PyTorch saves pre-operation state
4. **Function dispatch overhead** — Each `nn.Module` call may trigger saves

**What we actually need:**

- Only the **final settled states** (free and nudged phases)
- No intermediate activation history
- Gradients computed from state contrast, not backprop

### Implementation Strategy

#### Step 1: Manual Settling Without Autograd

```python
# Current approach (stores activations)
with torch.enable_grad():
    for step in range(settle_steps):
        E = energy_fn(model, x, states, ...)  # Triggers graph build
        grads = torch.autograd.grad(E, states)  # More graph overhead
        # Update states...

# O(1) approach (no activation storage)
with torch.no_grad():
    for step in range(settle_steps):
        E = manual_energy_compute(model, x, states, ...)  # No graph
        grads = manual_state_gradients(E, states)  # Direct computation
        # Update states...
```

**Key insight:** We don't need autograd during settling. We need autograd only for the final contrast step (computing parameter gradients from state differences).

#### Step 2: Manual Energy Computation

```python
def manual_energy_compute(model, x, states, structure, target_vec, beta):
    """Compute EP energy without autograd overhead."""
    E = 0.0
    prev = x
    
    for item in structure:
        if item["type"] == "layer":
            module = item["module"]
            state = states[state_idx]
            
            # Manual forward pass (no autograd)
            with torch.no_grad():
                h = manual_linear_forward(prev, module.weight, module.bias)
                h = F.relu(h)  # Activation is element-wise, no storage needed
            
            # Manual MSE (no graph)
            mse = ((h - state) ** 2).sum() / batch_size
            E = E + 0.5 * mse
            
            prev = state
            state_idx += 1
    
    # Nudge term
    if target_vec is not None and beta > 0:
        E = E + beta * manual_nudge_term(prev, target_vec)
    
    return E
```

#### Step 3: Manual State Gradients

```python
def manual_state_gradients(E, states):
    """Compute gradients w.r.t. states manually."""
    # This is the tricky part—we need dE/dstate for each layer
    # Option 1: Finite differences (slow but simple)
    # Option 2: Analytic gradients (fast but requires derivation)
    # Option 3: Selective autograd (only on states, not full graph)
    
    # Recommended: Option 3 with detached states
    grads = []
    for state in states:
        state.requires_grad_(True)
        # Recompute E with this state requiring grad
        E_recompute = recompute_energy_for_state(state, ...)
        grad = torch.autograd.grad(E_recompute, state, retain_graph=False)[0]
        grads.append(grad)
        state.requires_grad_(False)
    
    return grads
```

#### Step 4: Minimal Autograd for Contrast

```python
# After settling, we have states_free and states_nudged
# Now we need parameter gradients from the contrast

with torch.enable_grad():
    # Re-run forward pass with autograd, but ONLY for parameter gradients
    # States are already computed, we just need dL/dW
    
    E_free = energy_fn(model, x, states_free, ..., target_vec=None, beta=0.0)
    E_nudged = energy_fn(model, x, states_nudged, ..., target_vec=target, beta=beta)
    
    contrast_loss = (E_nudged - E_free) / beta
    
    # This builds a graph, but it's shallow—no settling iterations stored
    params = list(model.parameters())
    grads = torch.autograd.grad(contrast_loss, params)
    
    # Apply gradients
    for p, g in zip(params, grads):
        p.grad = g.detach()
```

### Milestones

| Week | Deliverable | Success Metric |
|------|-------------|----------------|
| 1-2 | Memory profiling complete | Baseline curve established |
| 3-4 | Manual settling prototype | Matches current accuracy |
| 5-6 | No-grad energy computation | Memory reduced 50%+ |
| 7-8 | Full O(1) integration | Memory flat vs depth |
| 9-10 | Deep network tests | 5000+ layers trained |

### Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Memory @ 100 layers | ~32 MB | ~20 MB |
| Memory @ 1000 layers | ~151 MB | ~20 MB |
| Memory @ 5000 layers | ~700 MB (est) | ~20 MB |
| Scaling | O(depth) | **O(1)** |
| Accuracy impact | - | <1% change |

---

## Priority 2: Deep Network Scaling

### Objective

Demonstrate EP can train networks that are impractical for backprop due to memory constraints.

### Test Architecture

```python
def make_deep_mlp(input_dim=784, hidden_dim=128, num_layers=1000, output_dim=10):
    """Create very deep MLP for scaling tests."""
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    
    for _ in range(num_layers - 2):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ])
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)
```

### Test Plan

| Depth | Purpose | Success Criteria |
|-------|---------|------------------|
| 100 | Baseline | Train to >90% MNIST |
| 500 | Scaling test | Train to >85% MNIST |
| 1000 | Memory wall for BP | Train to >80% MNIST |
| 2000 | Extreme depth | Train to >75% MNIST |
| 5000 | Beyond BP limits | Any convergence |
| 10000 | Proof of concept | Training stable |

### Measurements

For each depth:
- Peak memory (activation only, exclude weights)
- Training time per epoch
- Final accuracy
- Convergence behavior (settling steps needed)
- Gradient norms (check for vanishing/exploding)

### Potential Failure Modes

| Issue | Symptoms | Mitigation |
|-------|----------|------------|
| Vanishing gradients | Gradient norms → 0 | Muon orthogonalization, skip connections |
| Exploding gradients | Gradient norms → ∞ | Spectral constraints, gradient clipping |
| Settling divergence | Energy increases | Reduce settle_lr, increase steps |
| Memory still grows | O(1) not achieved | Profile and identify leaks |

---

## Priority 3: Continual Learning (EP+EWC)

### Objective

Integrate EWC with EP and demonstrate competitive continual learning performance.

### Current Status

- EP+ErrorFeedback: 32% forgetting (better than no EF at 48%, worse than EWC at 5-15%)
- EWC not yet integrated with EP

### Implementation Plan

```python
class EWCForEP:
    """EWC regularization for EP training."""
    
    def __init__(self, model, fisher_damping=1e-3):
        self.model = model
        self.fisher_damping = fisher_damping
        self.fisher_estimates = {}
        self.optimal_params = {}
    
    def compute_fisher(self, data_loader, task_id):
        """Compute Fisher information diagonal after task completion."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for x, y in data_loader:
            # Forward pass
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)
            
            # Accumulate squared gradients (Fisher diagonal)
            for (n, p), g in zip(self.model.named_parameters(), grads):
                fisher[n] += g.pow(2) * x.size(0)
        
        # Normalize and store
        for n in fisher:
            fisher[n] /= len(data_loader.dataset)
            fisher[n] += self.fisher_damping
        
        self.fisher_estimates[task_id] = fisher
        self.optimal_params[task_id] = {
            n: p.data.clone() for n, p in self.model.named_parameters()
        }
    
    def ewc_loss(self):
        """Compute EWC regularization term."""
        ewc_loss = torch.tensor(0.0)
        
        for task_id in self.fisher_estimates:
            fisher = self.fisher_estimates[task_id]
            optimal = self.optimal_params[task_id]
            
            for n, p in self.model.named_parameters():
                ewc_loss += (fisher[n] * (p - optimal[n]).pow(2)).sum()
        
        return ewc_loss
```

### Training Loop

```python
for task_id, (train_loader, test_loader) in enumerate(task_sequence):
    # Train on current task with EP
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.step(x=x, target=y)  # EP step
        
        # Add EWC loss to regularize (optional: integrate into EP energy)
        if task_id > 0:
            ewc_loss = ewc.ewc_loss()
            # Option: add to settling energy or as post-update regularization
    
    # After task completion: compute Fisher
    ewc.compute_fisher(train_loader, task_id)
    
    # Evaluate on all previous tasks
    for prev_task_id in range(task_id + 1):
        acc = evaluate(model, task_loaders[prev_task_id])
        # Track forgetting...
```

### Success Criteria

| Method | Forgetting | Notes |
|--------|------------|-------|
| Backprop (no regularization) | 40-50% | Baseline forgetting |
| EP (no regularization) | 40-50% | Expected similar |
| EP + Error Feedback | 30-40% | Current best |
| **EP + EWC (target)** | **<15%** | Goal |
| Backprop + EWC | 5-15% | Reference |

---

## Priority 4: Speed Optimization

### Objective

Reduce EP training time from 2-3× backprop to 1.5× backprop.

### Profiling Results Needed

Before optimization, profile to identify bottlenecks:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    optimizer.step(x=x, target=y)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### Optimization Opportunities

| Component | Current | Target | Approach |
|-----------|---------|--------|----------|
| Settling iterations | 30 steps | 15-20 steps | Adaptive early stopping |
| Per-iteration cost | Python loop | Fused kernel | CUDA optimization |
| Energy computation | Python | CUDA | Custom kernel |
| Contrast step | Full graph | Minimal graph | Selective autograd |

### Adaptive Settling

```python
def settle_adaptive(model, x, target, beta, energy_fn, structure,
                    max_steps=50, tol=1e-4, patience=5):
    """Settle with early stopping when converged."""
    prev_energy = None
    patience_counter = 0
    
    for step in range(max_steps):
        E = energy_fn(model, x, states, structure, target, beta)
        
        if prev_energy is not None:
            delta = abs(E.item() - prev_energy)
            if delta < tol:
                patience_counter += 1
                if patience_counter >= patience:
                    break  # Converged!
        
        prev_energy = E.item()
        # ... update states ...
    
    return states  # Settled in fewer steps!
```

### Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Settling steps | 30 | 15-20 |
| Time per epoch | 4-5s | 2-3s |
| Speed vs backprop | 2-3× | 1.5× |
| Accuracy impact | - | <1% change |

---

## Integration & Milestones

### Month 1-2: O(1) Memory Foundation
- [ ] Memory profiling complete
- [ ] Manual settling prototype working
- [ ] No-grad energy computation implemented
- [ ] Initial memory savings measured (50%+ reduction)

### Month 3: O(1) Memory Complete
- [ ] Full O(1) integration
- [ ] Memory flat vs depth confirmed
- [ ] 1000-layer network trained successfully
- [ ] Technical report drafted

### Month 4: Deep Scaling + CL
- [ ] 5000-layer network trained
- [ ] EP+EWC implemented
- [ ] CL benchmarks run (Permuted MNIST)
- [ ] Forgetting <20% achieved

### Month 5: Speed + Polish
- [ ] Adaptive settling implemented
- [ ] CUDA kernels optimized
- [ ] Speed improved to 1.5× backprop
- [ ] All results documented

### Month 6: Synthesis
- [ ] Comprehensive technical report
- [ ] All code cleaned and documented
- [ ] Ready for Phase 3 (outreach)

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| O(1) memory not achievable | Medium | High | Document findings; may be PyTorch limitation |
| Deep networks don't converge | Medium | High | Investigate gradient flow; add skip connections |
| EP+EWC doesn't improve CL | Low | Medium | Publish negative result; still valuable |
| Speed optimizations insufficient | Low | Low | 2× slowdown is acceptable for EP's advantages |

---

## Success Definition

Phase 2 is successful if we can demonstrate:

1. **O(1) memory** — EP memory flat at ~20 MB from 100 to 5000+ layers
2. **Deep scaling** — EP trains 5000+ layer networks that backprop cannot (OOM)
3. **Competitive CL** — EP+EWC forgetting <15% (comparable to backprop+EWC)
4. **Maintained accuracy** — MNIST 90%+ at all depths

If all four are achieved, EP has **genuine technical advantages** beyond biological plausibility.

---

*Created: 2026-02-18*
*Status: Ready for execution*
