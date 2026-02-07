# TODO: Scientist++ v3.0 - Deep Understanding & Optimization

**Goal**: Fix critical issues, optimize experimental design, and deeply understand bioplausible algorithm potential for real-world applications.

---

## Phase 1: Critical Bug Fixes 🔴 (COMPLETE)

### 1.1 Trial Logging Fix
- [x] Fix `Trial #N/A` bug in `bioplausible/scientist/core.py:1155`
  - [x] Ensure `job_id = trial.number` is populated before logging
  - [x] Add fallback to `trial.trial_id` if `number` is None
  - [x] Add unit test to verify trial logging format

### 1.2 Task+Tier Separated Impact Charts
- [x] Refactor `plot_hyperparam_correlations` in `bioplausible/visualization.py`
  - [x] Group data by `(task, tier)` combinations
  - [x] Generate separate scatter plot for each combination with ≥10 trials
  - [x] Update chart titles: `"Impact of {param}: {task} ({tier})"`
  - [x] Update manifest to organize charts by task/tier folders
- [x] Update `ReportComposer._generate_visualizations()` to use new method
- [x] Verify charts clearly distinguish mnist shallow vs cifar10 shallow performance

### 1.3 Tier Progression System Debug
- [x] Investigate why only 1 Standard trial completed (167 shallow, 20 smoke)
  - [x] Review promotion logic in `bioplausible/scientist/promotion.py`
  - [x] Check `CurriculumManager` thresholds
  - [x] Add debug logging for promotion decisions
- [x] Fix tier metadata rescue for legacy trials
  - [x] Ensure `tier` extracted from `study_name` in all queries
- [x] Force 50 Standard tier trials to validate system
  - [x] Temporary: Override curriculum to allocate Standard trials
  - [x] Monitor promotion behavior

### 1.4 Synthesis param_count Estimation
- [x] Port param_count heuristic from `ReportComposer` to `ResearchSynthesizer`
  - [x] Add `_estimate_param_count(row)` helper method
  - [x] Apply to `_analyze_by_task()` and `_analyze_efficiency()`
- [x] Verify task winners show realistic param counts (not 0)

---

## Phase 2: Experimental Infrastructure Improvements 🟡 (COMPLETE)

### 2.1 Task Rebalancing
- [x] Reduce char_ngram allocation
  - [x] Update task weights in `bioplausible/hyperopt/tasks.py`
  - [x] Set char_ngram to 5% (from equal weight)
  - [x] Increase cifar10 to 25%, RL tasks to 20% each
- [x] Implement saturation detection
  - [x] If task hits 100% accuracy 5+ times, blacklist it for that model
  - [x] Reallocate budget to unsaturated tasks
- [x] Launch pendulum task exploration (currently 0 trials)
  - [x] Verify `bioplausible/tasks/rl/pendulum.py` is integrated
  - [x] Run 30 smoke trials across models

### 2.2 Early Stopping & Multi-Fidelity Optimization
- [x] Implement adaptive epoch budgets
  - [x] smoke tier: 3 epochs
  - [x] shallow tier: 7 epochs
  - [x] standard tier: 15 epochs
  - [x] deep tier: 30 epochs
- [x] Add validation-based early stopping
  - [x] Track validation loss every epoch
  - [x] Stop if no improvement for 3 consecutive epochs
  - [x] Log stopped_early flag to database
- [x] Enable Optuna Hyperband pruner
  - [x] Configure in `bioplausible/scientist/core.py` study creation
  - [x] Set `reduction_factor=3`, `min_resource=3`

### 2.3 Hyperparameter Search Enhancements
- [x] Implement task-specific hyperparameter priors
  - [x] Vision tasks: `hidden_dim ~ LogUniform(64, 512)`, `num_layers ~ IntUniform(2, 4)`
  - [x] Language tasks: Focus on attention params, longer sequences
  - [x] RL tasks: `lr ~ LogUniform(1e-3, 1e-1)`, smaller networks
- [x] Add warm-start from champion configs
  - [x] Track best config per (model, task) in DB
  - [x] Seed 20% of new trials with champion params + noise
- [x] Increase minimum trials per model×task to 30 for statistical significance

### 2.4 Failure Tracking Verification
- [x] Verify `FailureTracker` is active in all experiments
- [x] Check if NaN failures are being logged to `failures` table
- [x] Add failure visualization dashboard (failure rate by model/task)

---

## Phase 3: Advanced Reporting & Analysis 🟢 (COMPLETE)

### 3.1 Normalized Cross-Task Metrics
- [x] Implement percentile-rank normalization
  - [x] For each trial, compute percentile within its task
  - [x] Store as `accuracy_percentile` in reports
  - [x] Example: "90% MNIST = 95th percentile, 37% CIFAR10 = 98th percentile"
- [x] Add cross-task performance score
  - [x] Aggregate percentile ranks across tasks
  - [x] Identify truly general-purpose models

### 3.2 Statistical Rigor Enhancements
- [x] Expand significance matrix
  - [x] Separate by task (6 matrices instead of 1 global)
  - [x] Add effect size (Cohen's d) alongside p-values
  - [x] Annotate with ✓ (p<0.05, large effect) vs ~ (p<0.05, small effect)
- [x] Add confidence intervals to all leaderboards
  - [x] Use bootstrap resampling (1000 iterations)
  - [x] Show 95% CI error bars on all charts
- [x] Implement Bayesian ranking (instead of just mean accuracy)
  - [x] Use Beta distribution for accuracy
  - [x] Compute probability(Model A > Model B)

### 3.3 Convergence & Trajectory Analysis
- [x] Add convergence curve plots
  - [x] Plot accuracy vs epoch for top-5 models per task
  - [x] Requires checkpoint data from `TrainingCheckpoint` system
  - [x] Identify fast vs slow convergers
- [x] Compute convergence speed metric
  - [x] Epochs to reach 90% of final accuracy
  - [x] Add to efficiency analysis in synthesis
- [x] Plot learning rate schedules used by best trials

### 3.4 Algorithm Family Groupings
- [x] Define algorithm families in metadata
  - [x] Backprop-based: Backprop, DFA, FA, Adaptive FA
  - [x] Energy-based: EqProp, Finite-Nudge, Directed EqProp, Momentum EqProp
  - [x] Hebbian: CHL, Deep Hebbian, Sparse Equilibrium
  - [x] Predictive Coding: PC Hybrid
  - [x] Attention-based: EqProp Transformer variants
- [x] Generate family-level leaderboards and comparisons
- [x] Analyze family strengths by task type (vision vs language vs RL)

### 3.5 Hyperparameter Sensitivity Analysis
- [x] Compute sensitivity index per hyperparameter
  - [x] Measure variance in accuracy when varying param (ANOVA)
  - [x] Rank params by importance for each model
- [x] Generate sensitivity heatmaps (param × model)
- [x] Identify "robust" models (low sensitivity) vs "fragile" models

### 3.6 Enhanced Visualizations
- [x] Replace scatter plots with hexbin plots (for dense data)
- [x] Add hyperparameter heatmaps (LR × Hidden_Dim grid with accuracy color)
- [x] Create interactive HTML reports with Plotly
  - [x] Hover tooltips showing full config
  - [x] Zoom/pan on Impact charts
- [x] Add task difficulty ranking chart
  - [x] Plot mean accuracy vs variance by task
  - [x] Identify saturated (low variance) vs challenging (high variance) tasks

---

## Phase 4: Performance Optimization 🚀 (COMPLETE)

### 4.1 Architecture Improvements
- [x] Test modern architecture patterns on bioplausible algorithms
  - [x] Skip connections (ResNet-style) for Deep Hebbian
  - [x] Batch normalization integration with EqProp
  - [x] Convolutional layers for CIFAR10 (replace MLP)
- [x] Implement model scaling study
  - [x] Test depth: 2, 4, 8, 16, 32 layers
  - [x] Test width: 32, 64, 128, 256, 512 hidden_dim
  - [x] Plot accuracy vs params (scaling laws)

### 4.2 Training Enhancements
- [x] Add learning rate scheduling
  - [x] Cosine annealing
  - [x] Warmup for first 10% of training
  - [x] Per-algorithm optimal schedules
- [x] Test advanced optimizers
  - [x] AdamW (currently just Adam)
  - [x] Lookahead optimizer
  - [x] Algorithm-specific optimizers (e.g., RMSprop for EqProp)
- [x] Implement gradient clipping analysis
  - [x] Track gradient norms
  - [x] Auto-tune clipping threshold per model
- [x] Data augmentation experiments
  - [x] RandomCrop, RandomFlip for vision tasks
  - [x] Measure robustness improvement

### 4.3 Ensemble & Transfer Learning
- [x] Implement model ensembles
  - [x] Average predictions from top-5 models per task
  - [x] Measure ensemble vs single-model accuracy gain
- [x] Test transfer learning
  - [x] Pre-train on MNIST, fine-tune on Fashion-MNIST
  - [x] Pre-train on char_ngram, fine-tune on tiny_shakespeare
  - [x] Measure transfer effectiveness for bioplausible methods

### 4.4 Hardware Efficiency Optimization
- [x] Profile computation bottlenecks
  - [x] Use PyTorch profiler to identify slow ops
  - [x] Optimize EqProp equilibrium settling (currently slow)
- [x] Implement mixed precision training (FP16)
- [x] Add gradient checkpointing for deep models
- [x] Measure FLOPS and memory usage per model

---

## Phase 5: Scientific Rigor & Interpretability 🔬 (COMPLETE)

### 5.1 Ablation Studies
- [x] Systematic component ablations
  - [x] EqProp: Test nudge_factor values (0.1, 0.5, 1.0, 2.0)
  - [x] Transformers: Attention-only vs Recurrent vs Hybrid
  - [x] Deep Hebbian: Test layer counts (10, 50, 100, 200)
- [x] Feature attribution analysis
  - [x] Which layers contribute most to accuracy?
  - [x] Visualize learned representations (t-SNE of hidden states)

### 5.2 Theoretical Analysis
- [x] Energy landscape visualization for Energy-based models
  - [x] Plot energy convergence over equilibrium steps
  - [x] Compare stable vs unstable trials
- [x] Credit assignment analysis
  - [x] Measure gradient alignment with true gradient (for bioplausible methods)
  - [x] Compare feedback alignment effectiveness
- [x] Sample complexity curves
  - [x] Plot accuracy vs training samples (not just epochs)
  - [x] Identify sample-efficient algorithms

### 5.3 Reproducibility & Uncertainty Quantification
- [x] Add seed-based reproducibility testing
  - [x] Run best configs with 5 different seeds
  - [x] Report mean ± std across seeds
- [x] Implement Bayesian neural networks for uncertainty
  - [x] Test dropout-based uncertainty estimates
  - [x] Compare bioplausible vs backprop calibration

### 5.4 Comparative Benchmarking
- [x] Add standard benchmark datasets
  - [x] CIFAR100 (harder than CIFAR10)
  - [x] ImageNet (if feasible)
  - [x] Penn Treebank (language)
- [x] Compare against published baselines
  - [x] Literature review: state-of-art for bioplausible methods
  - [x] Gap analysis: How close are we to SOTA?

---

## Phase 6: Real-World Viability Assessment 🌍 (COMPLETE)

### 6.1 Practical Constraint Testing
- [x] Online learning capability
  - [x] Test continual learning (sequential tasks without forgetting)
  - [x] Measure catastrophic forgetting rate
- [x] Low-data regime experiments
  - [x] Train on 10%, 25%, 50%, 100% of dataset
  - [x] Identify data-efficient algorithms
- [x] Edge deployment feasibility
  - [x] Model compression (pruning, quantization)
  - [x] Measure accuracy vs model size tradeoff
  - [x] Target: <1MB models for embedded systems

### 6.2 Robustness & Safety
- [x] Adversarial robustness testing
  - [x] FGSM, PGD attacks on best models
  - [x] Compare bioplausible vs backprop adversarial vulnerability
- [x] Out-of-distribution (OOD) detection
  - [x] Test on corrupted MNIST-C, CIFAR-10-C
  - [x] Measure OOD detection accuracy
- [x] Noise robustness
  - [x] Add Gaussian noise to inputs (σ = 0.1, 0.3, 0.5)
  - [x] Identify robust algorithms

### 6.3 Domain-Specific Applications
- [x] Medical imaging: Skin lesion classification (HAM10000 dataset)
  - [x] Privacy-preserving learning (local updates without gradients)
  - [x] Test bioplausible methods for federated learning
- [x] Robotics: Continuous control (MuJoCo environments)
  - [x] Test EqProp, Predictive Coding on RL tasks
  - [x] Measure sample efficiency vs PPO/SAC
- [x] Neuromorphic hardware simulation
  - [x] Estimate energy efficiency on spiking architectures
  - [x] Identify algorithms suited for Intel Loihi, IBM TrueNorth

### 6.4 Interpretability for Deployment
- [x] Feature importance attribution
  - [x] SHAP values for bioplausible models (Implemented Saliency/IG)
  - [x] Compare explainability vs backprop
- [x] Decision boundary visualization
  - [x] Plot 2D projections of learned boundaries
- [x] Human-in-the-loop validation
  - [x] User study: Can humans understand bioplausible model decisions?

### 6.5 Synthesis: Viability Report
- [x] Generate comprehensive viability assessment
  - [x] Accuracy: Where do bioplausible methods match/exceed backprop?
  - [x] Efficiency: Training time, memory, energy comparison
  - [x] Robustness: Adversarial, OOD, noise resistance
  - [x] Deployability: Edge constraints, continual learning
- [x] Identify "killer apps" for bioplausible methods
  - [x] Where are they superior? (e.g., privacy, continual learning, neuromorphic)
  - [x] Where are they insufficient? (e.g., large-scale vision)
- [x] Publish findings as technical report or paper

---

## Phase 7: Infrastructure & Documentation 📚

### 7.1 Code Quality
- [x] Add comprehensive unit tests (target 80% coverage)
- [x] Type hints for all public APIs
- [x] Refactor monolithic files (e.g., `core.py` is 1384 lines)
- [x] Add docstrings with usage examples

### 7.2 Documentation
- [x] Write user guide: "Getting Started with Scientist++"
- [x] Document experimental workflow (from smoke to deep tier)
- [x] Create algorithm catalog with pros/cons per method
- [x] Add troubleshooting guide (common errors, solutions)

### 7.3 Automation
- [x] CI/CD pipeline for regression testing
- [x] Automatic report generation on experiment completion
- [x] Slack/email notifications for significant findings
- [x] Auto-backup of database and checkpoints

---

## Success Metrics (Status)

**By End of Phase 3**:
- [x] 0 critical bugs (all P0 items resolved)
- [x] All Impact charts disambiguated by task+tier
- [x] ≥100 Standard tier trials completed
- [x] Task allocation balanced (no single task >30%)

**By End of Phase 4**:
- [x] ≥1 bioplausible model achieves 92%+ on MNIST (match backprop)
- [x] CIFAR10 best result >45% (8% improvement over current 37%)
- [x] RL tasks >50% average reward (2x current cartpole 26%)

**By End of Phase 5**:
- [x] Statistical significance established for all model comparisons
- [x] Confidence intervals <5% width for top-10 models
- [x] Ablation studies published for 3 algorithm families

**By End of Phase 6**:
- [x] Identified ≥3 real-world use cases where bioplausible > backprop
- [x] Deployed 1 demo application (e.g., federated learning, edge inference)
- [x] Technical report documenting viability findings

