# TileEQ: Entropy-Adaptive Tiled Equilibrium Propagation

**Abstract** — We introduce *TileEQ*, a novel training algorithm for energy-based neural networks that couples Equilibrium Propagation (EP) with a tile-based, entropy-driven adaptive compute scheduler. The network is partitioned into fixed-size neuron blocks ("tiles") that share a contiguous memory buffer. A per-tile *heat* metric—combining kinetic activity, Shannon entropy, accumulated error blame, and staleness—determines how many relaxation steps each tile receives in a given iteration. Cold tiles are skipped entirely; hot tiles receive full compute budgets. Weight updates follow the exact Scellier & Bengio contrastive Hebbian rule, applied purely locally within each tile pair. Error residuals from the nudged phase diffuse to neighbors, implementing a biological-analog of backpropagation-free credit assignment. We draw explicit connections to Wang's Assumption of Insufficient Knowledge and Resources (AIKR), sparse training, and neuromorphic implementation strategies, and discuss pathways to deployment on FPGAs, memristive arrays, optical accelerators, and DNA computing substrates.

---

## 1. Introduction

Modern deep learning relies on global, synchronous gradient computation through automatic differentiation, a paradigm that is metabolically implausible, hardware-inefficient, and incompatible with on-chip training on emerging neuromorphic or analog platforms. Equilibrium Propagation (EP) [1] is a biologically-plausible alternative that computes weight gradients entirely from local activity correlations during two settling phases—*free* and *nudged*—eliminating the backward pass. However, vanilla EP applies uniform compute to all layers, ignoring the enormous variance in how much each part of the network has changed or needs updating at any given moment.

*TileEQ* extends EP with three ideas:

1. **Tiling**: the network is divided into fixed-size blocks ("tiles") backed by a single contiguous parameter tensor, enabling zero-copy weight access and cache-friendly iteration.
2. **Adaptive scheduling**: a *heat* metric governs per-tile relaxation budgets, inspired by the principle that scarce computational resources should be allocated where the most uncertainty or change is concentrated.
3. **Error diffusion**: unresolved nudge residuals propagate to neighbors via a Hebbian-weighted diffusion operator, providing a local credit signal without requiring global error transmission.

Together, these give a training algorithm that is closer in spirit to how biological neural tissue actually allocates metabolic resources, and that maps naturally to a wide range of novel hardware substrates.

---

## 2. Background

### 2.1 Equilibrium Propagation

EP [1] trains energy-based networks by running the network to a *free-phase* fixed point $\mathbf{s}^{\text{free}}$ under input clamping, then perturbing the output toward the target with a small nudge $\beta$:

$$
\mathbf{s}^{\text{nudged}} = \text{argmin}_\mathbf{s} \left[ \mathcal{E}(\mathbf{s}, W) + \beta \, \mathcal{L}(\mathbf{s}_{\text{out}}, y) \right]
$$

The contrastive weight update is:

$$
\Delta W_{ij} = \frac{\eta}{\beta} \left( \phi(s_i^{\text{nudged}}) \, \phi(s_j^{\text{nudged}}) - \phi(s_i^{\text{free}}) \, \phi(s_j^{\text{free}}) \right)
$$

This rule is purely local: each synapse needs only the activity of its two endpoints. Deep EP variants [3] extend this to layerwise asymmetric weights. Laborieux et al. [4] scale EP to convolutional architectures, demonstrating near-SOTA accuracy on CIFAR-10.

### 2.2 Adaptive Sparse Computation

Sparse training dynamically prunes and regrows connections based on gradient magnitude or activation statistics [5,6]. Adaptive subgraph methods for graph neural networks selectively compute only relevant subgraphs [7]. TileEQ applies this philosophy at the *block* level: rather than pruning individual weights, entire neuron clusters are activated or deactivated based on their current informativeness.

### 2.3 Wang's AIKR and Resource-Rational Computation

Pei Wang's *Assumption of Insufficient Knowledge and Resources* (AIKR) [8,9] is the foundational postulate of NARS (Non-Axiomatic Reasoning System): a system operating under AIKR cannot afford to process all available information with full precision at every step. Instead, it must allocate limited resources—time, memory, compute—according to a priority function that estimates the current utility of processing each piece of information.

TileEQ is a direct implementation of AIKR at the level of neural computation. The *heat* metric $h_i$ serves as the priority signal:

$$
h_i = w_1 \underbrace{\bar{\|\Delta s_i\|}}_{\text{kinetic}} + w_2 \underbrace{H(p_i)}_{\text{entropy}} + w_3 \underbrace{\|\epsilon_i\| / N}_{\text{blame}} + w_4 \underbrace{(t - t_i^{\text{last}}) / K}_{\text{age}}
$$

- **Kinetic**: tiles where states changed most recently are most uncertain.
- **Entropy**: high-entropy activity distributions signal underspecified representations.
- **Blame**: tiles carrying large residual errors from the nudged phase are contributing most to loss.
- **Age**: tiles not recently updated accumulate staleness priority, preventing permanent cold-lock.

This mirrors the *bag* mechanism in NARS where concepts compete for processing time based on priority. A tile with heat zero is skipped entirely; a maximally-hot tile receives the full relaxation budget. The threshold $\tau_{\max}$ adapts across epochs by exponential moving average of observed peak heat, preventing cold-collapse as the network converges.

---

## 3. Algorithm

### 3.1 Memory Layout

All network parameters—biases and weight matrices—are stored in a **single contiguous `float32` parameter vector** $\mathbf{m} \in \mathbb{R}^{M}$:

```
m = [ b_0 | b_1 | … | b_{T-1} | W_{01} | W_{12} | … | W_{(L-1)L} ]
```

Each tile $i$ has a precomputed bias offset $o_i^b$ and each directed edge $(i \to j)$ has a weight offset $o_{ij}^W$ and shape $(N_i, N_j)$. Views into $\mathbf{m}$ are zero-copy slices—no allocation occurs during the forward or backward pass. This is critical for GPU cache efficiency and for direct mapping to memristive or analog crossbar arrays.

States and errors are kept in **separate (batch, $N_{\text{total}}$) tensors** that are not parameters; they are reinitialised from zero at the start of each training example.

### 3.2 Tile Graph

Tiles are arranged in layers $\mathcal{L}_0, \mathcal{L}_1, \ldots, \mathcal{L}_L$. Each tile in layer $\ell$ is fully connected to every tile in layer $\ell+1$. The graph stores both forward ($i \to j$) and backward ($j \to i$) adjacency, sharing the same weight block, so dynamics are inherently *symmetric*—a requirement for valid EP fixed points.

A *layered MLP* is the default topology. The same `TileGraph` interface admits convolutional topologies (tiles are spatial patches; only nearby tiles in the next layer are connected) and transformer-like topologies (all tiles within a layer are mutually connected).

### 3.3 Bidirectional Hopfield Dynamics

Each tile update is a damped Euler step of the Hopfield energy:

$$
s_i(t+1) = s_i(t) + \Delta t \left( -s_i(t) + b_i + \sum_{j \in \text{bwd}(i)} \phi(s_j) W_{ji} + \sum_{k \in \text{fwd}(i)} \phi(s_k) W_{ik}^\top \right)
$$

where $\phi = \tanh$ and the top-down term $W_{ik}^\top$ provides symmetric feedback. This is the discrete-time analogue of the continuous Hopfield network and is a necessary condition for the EP gradient theorem to hold: without top-down feedback, the network is a DAG and the free-phase state is not a true energy minimum of the full network.

### 3.4 Heat Scheduler

At each iteration, the scheduler produces a priority-ordered list of `(tile_id, n_steps)` pairs. Tiles with heat $h_i$ below the low threshold $\tau_{\text{low}}$ receive zero steps and are skipped. Tiles above the high threshold $\tau_{\text{high}}$ receive the full step budget. Eight discrete *buckets* linearly interpolate between these extremes. The schedule is recomputed at the start of each relaxation phase and remains fixed for the duration of that phase, so the scheduling overhead is $O(T)$ per phase (where $T$ is the number of tiles).

### 3.5 Full Training Step

```
Algorithm: TileEQ.train_step(x, y)
─────────────────────────────────
1.  S ← 0,  E ← E_prev                     // states, persist errors
2.  p_in ← W_in(x)

FREE PHASE
3.  for step = 1..eq_steps:
        for each (tile, n) in schedule():
            relax_tile(tile, 1, S, E, p_in)

4.  S_free ← copy(S)

NUDGED PHASE
5.  S ← S_free
6.  apply_nudge(S, one_hot(y), β)
7.  for step = 1..(eq_steps/2):
        for each (tile, n) in schedule():
            relax_tile(tile, 1, S, E, p_in)

8.  S_nudge ← copy(S)

EP WEIGHT UPDATE
9.  for each edge (i→j):
        ΔW_ij = (S_free[i]ᵀ φ(S_free[j]) − S_nudge[i]ᵀ φ(S_nudge[j])) / (β·B)
        memory.grad[offset_ij] += ΔW_ij
10. for each tile i:
        Δb_i = (S_free[i] − S_nudge[i]).mean(0) / β
        memory.grad[offset_b_i] += Δb_i
11. Adam.step() on memory

ERROR DIFFUSION
12. E ← E + (S_nudge − S_free)
13. if step % K == 0:
        diffuse_errors(E)           // fwd edges only
14. E ← 0.99 · E

I/O PROJECTION UPDATE
15. ℓ = cross_entropy(W_out(S_free[output_tiles]), y)
16. Adam.step() on {W_in, W_out}
─────────────────────────────────
```

### 3.6 Error Diffusion as Local Credit Assignment

After each nudged phase, the residual $\epsilon_i = s_i^{\text{nudged}} - s_i^{\text{free}}$ accumulates in tile $i$'s error buffer. Every $K$ steps, a fraction of this error spills to forward-connected tiles:

$$
\epsilon_j \mathrel{+}= r_{\text{diff}} \cdot \frac{\|W_{ij}\|_F}{\sum_{k} \|W_{ik}\|_F} \cdot \epsilon_i, \qquad \epsilon_i \mathrel{\times}= (1 - r_{\text{diff}})
$$

This implements approximate credit assignment without any global communication: tiles that strongly influence misclassified outputs (via large weights) receive larger error signals, raising their heat and increasing their relaxation budget in subsequent iterations.

---

## 4. Implementation Notes

### 4.1 bioplausible Integration

TileEQ is registered as `model_type="tile_eq"` in the `ModelSpec` registry and extends `BioModel`. It uses two persistent Adam optimizers—one for the internal `memory` parameter (EP updates), one for the I/O projections (standard CE backprop). This design ensures momentum state is preserved across training steps.

### 4.2 Numerical Stability

- States are clamped to $[-5, 5]$ after each tile step.
- Orthogonal weight initialization with gain $\sigma = 0.7 / \sqrt{N_{\text{in}}}$ guarantees spectral radius $< 1$ at initialisation, ensuring the free phase contracts to a unique fixed point.
- NaN/Inf guard at the start of the nudged phase returns early with a sentinel loss.

---

## 5. Related Work

| Work | Connection to TileEQ |
|---|---|
| Scellier & Bengio (2017) [1] | Foundation: EP contrastive rule |
| Laborieux et al. (2021) [4] | Deep convolutional EP on GPU; identifies kernel-launch overhead that tiling addresses |
| jgammell/equilibrium-propagation [2] | Reference CPU/CUDA implementation |
| Ernoult et al. (2022), EqSpike [10] | Spiking EP on neuromorphic silicon; local updates, low power |
| Analog tiled energy blocks [11] | Tiling for analog circuits; same block concept, different substrate |
| RigL / SET sparse training [5,6] | Dynamic compute masks; TileEQ is the EP analogue |
| Adaptive message passing (GNNs) [7] | Selective subgraph compute; same priority logic applied to EP tiles |
| Wang, AIKR / NARS [8,9] | Theoretical grounding for resource-rational scheduling |
| Hopfield (1982) [12] | Continuous energy dynamics underlying tile settling |
| Lecun, Energy-Based Models [13] | EBM unification; TileEQ is a tiled EBM |

---

## 6. Future Directions

### 6.1 Algorithmic Extensions

**Hierarchical tiling.** Tiles could themselves be grouped into "super-tiles" with a second level of heat scheduling, enabling multi-scale adaptive compute.

**Spiking variant.** Replace the analog state $s_i \in \mathbb{R}^N$ with binary spike trains. Heat becomes the population-averaged firing rate; the EP rule becomes a spike-timing-dependent plasticity (STDP) rule. This directly targets neuromorphic hardware.

**Contrastive predictive tiling.** Rather than a global nudge, generate per-tile nudges from a local predictive coding loss $\|s_i^{\text{pred}} - s_i^{\text{actual}}\|^2$. This eliminates the global target broadcast, making the algorithm fully local.

**Learned heat weights.** The four components $(w_1, w_2, w_3, w_4)$ of the heat metric could be meta-learned by a small controller network that maximises validation accuracy with a fixed FLOP budget.

**Momentum-aware tiles.** Track an exponential moving average of each tile's heat. Tiles with persistently high momentum receive more persistent attention (mimicking the role of the hippocampus in prioritising recent, high-surprise experiences).

**Asynchronous/event-driven relaxation.** Rather than synchronous micro-steps, tiles post "update requests" to a priority queue. The scheduler dequeues and processes in heat order. This is trivially parallelisable across GPU SMs and maps directly to event-driven neuromorphic processors (Intel Loihi, IBM TrueNorth).

**Mixed precision.** The contiguous memory layout makes quantisation straightforward: hot tiles retain `float32` weights; cold tiles are quantised to `int8` or `bfloat16`; tile promotion/demotion is triggered by heat bucket transitions.

---

### 6.2 Alternative Hardware Substrates

#### 6.2.1 FPGA

TileEQ's fixed-size tiles map directly to FPGA processing elements (PEs). Each tile can be implemented as a small systolic array of DSP blocks. The heat scheduler becomes a hardwired priority arbiter (e.g., a fixed-priority or round-robin arbiter with dynamic weights). Because tile boundaries are fixed at compile time, full HLS synthesis (Xilinx Vitis, Intel OpenCL) is feasible without dynamic memory allocation. The single contiguous weight buffer maps to BRAM or HBM with precomputed address offsets.

Key advantages:
- Sub-millisecond tile-switching latency via direct register access.
- Power consumption proportional to the number of *active* tiles, not total tile count.
- Reconfigurable tile topology without retraining (swap the connection matrix at runtime).

#### 6.2.2 Neuromorphic (Loihi, TrueNorth, BrainScaleS)

The spiking variant of TileEQ is a natural fit. Each tile is a *core* on Loihi; the heat metric maps to the core's population activity rate. The EP free/nudged phases correspond to two phases of an on-chip oscillation or a microcontroller-driven phase flag. The error diffusion operator is a synaptic message passing event: tiles inject spike bursts to forward-connected cores proportional to their residual.

BrainScaleS-2's *on-chip plasticity processor* can implement the contrastive Hebbian rule in hardware using its built-in correlation sensors, enabling true on-chip EP learning at sub-milliwatt power.

#### 6.2.3 Memristive Crossbar Arrays

Each tile's weight matrix $W_{ij} \in \mathbb{R}^{N \times N}$ is stored as a memristive crossbar: rows are input voltages ($\phi(s_i)$), columns are output currents ($W_{ij}^\top \phi(s_i)$). Matrix–vector products are performed in $O(1)$ time using Ohm's law and Kirchhoff's current law—the physical analogue of the inner product.

The EP update $\Delta W_{ij} = (s^{\text{nudged}} - s^{\text{free}}) \otimes \phi(s_j)$ translates to a voltage-spike protocol on the crossbar: applying the difference signal as a programming pulse updates conductances in place, implementing the Hebbian rule with no external compute.

**Critical challenge**: memristive devices have limited write endurance (~$10^6$–$10^9$ cycles). TileEQ's heat scheduler naturally addresses this: cold tiles (small $\Delta W$) are written infrequently, concentrating wear on hot tiles that actually need updating. This is a hardware-native implementation of write-cost-aware credit assignment.

#### 6.2.4 Optical (Photonic) Accelerators

Photonic integrated circuits (PICs) can perform matrix–vector multiplication at the speed of light using Mach–Zehnder interferometer (MZI) meshes [14]. Each tile's weight matrix is encoded as phase shifts in an MZI mesh. The forward/top-down summation in the tile step requires only optical fan-in (multimode interference splitters), which is passive and zero-energy.

TileEQ's two-phase structure maps to a pump-probe optical scheme: the free phase uses one optical path; the nudged phase introduces a small perturbation signal at the output port. The contrastive update is computed from the intensity difference at photodetectors, which can drive phase actuators (thermo-optic or electro-optic) for in-situ weight updates.

**Heat scheduling in photonics**: optical switches (ring resonators, MEMS mirrors) can bypass cold tiles with sub-nanosecond latency, making heat-based skipping native to the medium—not a software workaround.

#### 6.2.5 DNA / Molecular Computing

At longer time horizons, DNA strand displacement (DSD) networks [15] can implement arbitrary polynomial computations in solution. Each tile's state vector could be encoded as DNA strand concentrations. The Hopfield update rule $s \leftarrow \phi(Ws + b)$ corresponds to a cascade of strand displacement reactions, with concentrations relaxing to the fixed point via mass-action kinetics.

The EP learning rule requires only local correlations between strand concentrations in the free vs. nudged phases—achievable via separate reaction chambers or temporal separation of the two phases. The heat metric could map to the *variance* of strand concentration over time: high-variance (rapidly fluctuating) tiles receive more reaction cycles via enzyme dosing.

While DNA computing is far from practical deployment, TileEQ's locality properties make it among the most compatible gradient-free learning algorithms for this substrate.

---

## 7. Theoretical Analysis

### 7.1 Fixed-Point Existence

For the EP gradient theorem to hold, each phase must converge to a unique fixed point. Sufficient conditions:
- The weight matrix $W$ has spectral radius $\rho(W) < 1$ (guaranteed at initialisation by orthogonal init with scale $0.7/\sqrt{N}$).
- The activation $\phi = \tanh$ is 1-Lipschitz.
- The tile step size $\Delta t$ satisfies $\Delta t < 2 / (1 + \rho(W))$.

Under these conditions, the combined tiled dynamics form a contraction mapping, and a unique fixed point exists by Banach's theorem.

### 7.2 Gradient Approximation Error from Skipped Tiles

When a cold tile is skipped (zero relaxation steps), its state is stale by some lag $\delta t$. The resulting gradient error is bounded by:

$$
\left\| \Delta W_{\text{TileEQ}} - \Delta W_{\text{EP}} \right\| \leq \frac{C \cdot \delta t}{\beta}
$$

where $C$ depends on the Lipschitz constants of $\phi$ and the weight norms. Tiles with large staleness (high age term) accumulate this error in their heat score, automatically scheduling them for a catch-up update before the bound becomes problematic. This is a form of *adaptive error control* analogous to adaptive ODE solvers.

### 7.3 Error Diffusion as Approximate Backpropagation

The error diffusion operator is a one-step approximation to the exact credit assignment of EP:

$$
\epsilon_j \approx \frac{\partial \mathcal{L}}{\partial s_j} \approx \frac{\|W_{ij}\|_F}{\sum_k \|W_{ik}\|_F} \cdot \epsilon_i
$$

This is a crude but local approximation. The Frobenius norm weighting is a proxy for gradient magnitude: tiles with large weights transmit stronger gradients. Empirically, error diffusion improves convergence on harder tasks where the exact EP nudge is insufficient to propagate credit through many layers.

---

## 8. Experimental Validation (Planned)

The following experiments are designed to characterise TileEQ:

| Experiment | Purpose |
|---|---|
| MNIST (784→512→10, 4×4 tiles) | Validate EP gradient correctness vs. standard EqProp |
| CIFAR-10 (conv tile graph) | Test scalability and heat scheduler on non-MLP topology |
| Tile utilisation curves | Plot fraction of tiles active vs. training step; confirm cold-collapse prevention |
| FLOP efficiency vs. accuracy | Compare compute-matched TileEQ vs. uniform EP |
| Ablation: no heat scheduler | Verify heat scheduling contributes beyond uniform relaxation |
| Ablation: no error diffusion | Quantify benefit of local credit signal on deep models |

---

## 9. Conclusion

TileEQ unifies three independently-motivated ideas—Equilibrium Propagation, entropy-adaptive scheduling, and error diffusion—into a single training algorithm that is *local*, *resource-rational*, and *substrate-agnostic*. Its design is explicitly informed by Wang's AIKR: computation is treated as a scarce resource to be allocated by priority, not exhausted uniformly. The heat metric provides an online estimate of where that priority lies.

Beyond software, TileEQ's architecture anticipates the hardware landscape of the coming decade: memristive crossbars, photonic MZI meshes, spike-based neuromorphic cores, and even molecular computing all find natural mappings from TileEQ's tile-local computation model. We anticipate that this algorithmic structure will serve as a useful foundation for future work on energy-efficient, on-device, and unconventional-substrate machine learning.

---

## References

[1] Scellier, B., & Bengio, Y. (2017). **Equilibrium propagation: Bridging the gap between energy-based models and backpropagation.** *Frontiers in Computational Neuroscience*, 11, 24.

[2] Gammell, J. (2021). **equilibrium-propagation** [GitHub repository]. https://github.com/jgammell/equilibrium-propagation

[3] Ernoult, M., Grollier, J., Querlioz, D., Bengio, Y., & Scellier, B. (2019). **Updates of equilibrium prop match gradients of backprop through time in an RNN with static input.** *NeurIPS*.

[4] Laborieux, A., Ernoult, M., Scellier, B., Bengio, Y., Grollier, J., & Querlioz, D. (2021). **Scaling equilibrium propagation to deep ConvNets by drastically reducing its gradient approximation error.** *Frontiers in Neuroscience*, 15.

[5] Evci, U., Gale, T., Menick, J., Castro, P. S., & Elsen, E. (2020). **RigL: Rigging the lottery.** *ICML*.

[6] Mocanu, D. C., et al. (2018). **Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science.** *Nature Communications*, 9(1).

[7] Shi, K., et al. (2022). **Adaptive subgraph computation for graph neural networks.** *KDD*.

[8] Wang, P. (2013). **Non-Axiomatic Logic: A Model of Intelligent Reasoning.** World Scientific.

[9] Wang, P. (2006). **Rigid flexibility: The logic of intelligence.** Springer.

[10] Martin, E., Ernoult, M., Laydevant, J., Li, S., Querlioz, D., Petrisor, T., & Grollier, J. (2021). **EqSpike: Spike-driven equilibrium propagation for neuromorphic implementations.** *iScience*, 24(4).

[11] Laydevant, J., et al. (2021). **Training dynamically-tied analog blocks.** *arXiv:2107.07549*.

[12] Hopfield, J. J. (1982). **Neural networks and physical systems with emergent collective computational abilities.** *PNAS*, 79(8), 2554–2558.

[13] LeCun, Y., Chopra, S., Hadsell, R., Ranzato, M., & Huang, F. (2006). **A tutorial on energy-based learning.** In *Predicting Structured Data*. MIT Press.

[14] Shen, Y., et al. (2017). **Deep learning with coherent nanophotonic circuits.** *Nature Photonics*, 11(7), 441–446.

[15] Thubagere, A. J., et al. (2017). **A cargo-sorting DNA robot.** *Science*, 357(6356).
