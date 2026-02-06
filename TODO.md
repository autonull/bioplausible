# TODO: Scientist++ v3.0 - Deep Understanding & Optimization

**Goal**: Fix critical issues, optimize experimental design, and deeply understand bioplausible algorithm potential for real-world applications.

---

## Phase 1: Critical Bug Fixes 🔴

### 1.1 Trial Logging Fix
- [ ] Fix `Trial #N/A` bug in `bioplausible/scientist/core.py:1155`
  - [ ] Ensure `job_id = trial.number` is populated before logging
  - [ ] Add fallback to `trial.trial_id` if `number` is None
  - [ ] Add unit test to verify trial logging format

### 1.2 Task+Tier Separated Impact Charts
- [ ] Refactor `plot_hyperparam_correlations` in `bioplausible/visualization.py`
  - [ ] Group data by `(task, tier)` combinations
  - [ ] Generate separate scatter plot for each combination with ≥10 trials
  - [ ] Update chart titles: `"Impact of {param}: {task} ({tier})"`
  - [ ] Update manifest to organize charts by task/tier folders
- [ ] Update `ReportComposer._generate_visualizations()` to use new method
- [ ] Verify charts clearly distinguish mnist shallow vs cifar10 shallow performance

### 1.3 Tier Progression System Debug
- [ ] Investigate why only 1 Standard trial completed (167 shallow, 20 smoke)
  - [ ] Review promotion logic in `bioplausible/scientist/promotion.py`
  - [ ] Check `CurriculumManager` thresholds
  - [ ] Add debug logging for promotion decisions
- [ ] Fix tier metadata rescue for legacy trials
  - [ ] Ensure `tier` extracted from `study_name` in all queries
- [ ] Force 50 Standard tier trials to validate system
  - [ ] Temporary: Override curriculum to allocate Standard trials
  - [ ] Monitor promotion behavior

### 1.4 Synthesis param_count Estimation
- [ ] Port param_count heuristic from `ReportComposer` to `ResearchSynthesizer`
  - [ ] Add `_estimate_param_count(row)` helper method
  - [ ] Apply to `_analyze_by_task()` and `_analyze_efficiency()`
- [ ] Verify task winners show realistic param counts (not 0)

---

## Phase 2: Experimental Infrastructure Improvements 🟡

### 2.1 Task Rebalancing
- [ ] Reduce char_ngram allocation
  - [ ] Update task weights in `bioplausible/hyperopt/tasks.py` 
  - [ ] Set char_ngram to 5% (from equal weight)
  - [ ] Increase cifar10 to 25%, RL tasks to 20% each
- [ ] Implement saturation detection
  - [ ] If task hits 100% accuracy 5+ times, blacklist it for that model
  - [ ] Reallocate budget to unsaturated tasks
- [ ] Launch pendulum task exploration (currently 0 trials)
  - [ ] Verify `bioplausible/tasks/rl/pendulum.py` is integrated
  - [ ] Run 30 smoke trials across models

### 2.2 Early Stopping & Multi-Fidelity Optimization
- [ ] Implement adaptive epoch budgets
  - [ ] smoke tier: 3 epochs
  - [ ] shallow tier: 7 epochs  
  - [ ] standard tier: 15 epochs
  - [ ] deep tier: 30 epochs
- [ ] Add validation-based early stopping
  - [ ] Track validation loss every epoch
  - [ ] Stop if no improvement for 3 consecutive epochs
  - [ ] Log stopped_early flag to database
- [ ] Enable Optuna Hyperband pruner
  - [ ] Configure in `bioplausible/scientist/core.py` study creation
  - [ ] Set `reduction_factor=3`, `min_resource=3`

### 2.3 Hyperparameter Search Enhancements
- [ ] Implement task-specific hyperparameter priors
  - [ ] Vision tasks: `hidden_dim ~ LogUniform(64, 512)`, `num_layers ~ IntUniform(2, 4)`
  - [ ] Language tasks: Focus on attention params, longer sequences
  - [ ] RL tasks: `lr ~ LogUniform(1e-3, 1e-1)`, smaller networks
- [ ] Add warm-start from champion configs
  - [ ] Track best config per (model, task) in DB
  - [ ] Seed 20% of new trials with champion params + noise
- [ ] Increase minimum trials per model×task to 30 for statistical significance

### 2.4 Failure Tracking Verification
- [ ] Verify `FailureTracker` is active in all experiments
- [ ] Check if NaN failures are being logged to `failures` table
- [ ] Add failure visualization dashboard (failure rate by model/task)

---

## Phase 3: Advanced Reporting & Analysis 🟢

### 3.1 Normalized Cross-Task Metrics
- [ ] Implement percentile-rank normalization
  - [ ] For each trial, compute percentile within its task
  - [ ] Store as `accuracy_percentile` in reports
  - [ ] Example: "90% MNIST = 95th percentile, 37% CIFAR10 = 98th percentile"
- [ ] Add cross-task performance score
  - [ ] Aggregate percentile ranks across tasks
  - [ ] Identify truly general-purpose models

### 3.2 Statistical Rigor Enhancements
- [ ] Expand significance matrix
  - [ ] Separate by task (6 matrices instead of 1 global)
  - [ ] Add effect size (Cohen's d) alongside p-values
  - [ ] Annotate with ✓ (p<0.05, large effect) vs ~ (p<0.05, small effect)
- [ ] Add confidence intervals to all leaderboards
  - [ ] Use bootstrap resampling (1000 iterations)
  - [ ] Show 95% CI error bars on all charts
- [ ] Implement Bayesian ranking (instead of just mean accuracy)
  - [ ] Use Beta distribution for accuracy
  - [ ] Compute probability(Model A > Model B)

### 3.3 Convergence & Trajectory Analysis
- [ ] Add convergence curve plots
  - [ ] Plot accuracy vs epoch for top-5 models per task
  - [ ] Requires checkpoint data from `TrainingCheckpoint` system
  - [ ] Identify fast vs slow convergers
- [ ] Compute convergence speed metric
  - [ ] Epochs to reach 90% of final accuracy
  - [ ] Add to efficiency analysis in synthesis
- [ ] Plot learning rate schedules used by best trials

### 3.4 Algorithm Family Groupings
- [ ] Define algorithm families in metadata
  - [ ] Backprop-based: Backprop, DFA, FA, Adaptive FA
  - [ ] Energy-based: EqProp, Finite-Nudge, Directed EqProp, Momentum EqProp
  - [ ] Hebbian: CHL, Deep Hebbian, Sparse Equilibrium
  - [ ] Predictive Coding: PC Hybrid
  - [ ] Attention-based: EqProp Transformer variants
- [ ] Generate family-level leaderboards and comparisons
- [ ] Analyze family strengths by task type (vision vs language vs RL)

### 3.5 Hyperparameter Sensitivity Analysis
- [ ] Compute sensitivity index per hyperparameter
  - [ ] Measure variance in accuracy when varying param (ANOVA)
  - [ ] Rank params by importance for each model
- [ ] Generate sensitivity heatmaps (param × model)
- [ ] Identify "robust" models (low sensitivity) vs "fragile" models

### 3.6 Enhanced Visualizations
- [ ] Replace scatter plots with hexbin plots (for dense data)
- [ ] Add hyperparameter heatmaps (LR × Hidden_Dim grid with accuracy color)
- [ ] Create interactive HTML reports with Plotly
  - [ ] Hover tooltips showing full config
  - [ ] Zoom/pan on Impact charts
- [ ] Add task difficulty ranking chart
  - [ ] Plot mean accuracy vs variance by task
  - [ ] Identify saturated (low variance) vs challenging (high variance) tasks

---

## Phase 4: Performance Optimization 🚀

### 4.1 Architecture Improvements
- [ ] Test modern architecture patterns on bioplausible algorithms
  - [ ] Skip connections (ResNet-style) for Deep Hebbian
  - [ ] Batch normalization integration with EqProp
  - [ ] Convolutional layers for CIFAR10 (replace MLP)
- [ ] Implement model scaling study
  - [ ] Test depth: 2, 4, 8, 16, 32 layers
  - [ ] Test width: 32, 64, 128, 256, 512 hidden_dim
  - [ ] Plot accuracy vs params (scaling laws)

### 4.2 Training Enhancements
- [ ] Add learning rate scheduling
  - [ ] Cosine annealing
  - [ ] Warmup for first 10% of training
  - [ ] Per-algorithm optimal schedules
- [ ] Test advanced optimizers
  - [ ] AdamW (currently just Adam)
  - [ ] Lookahead optimizer
  - [ ] Algorithm-specific optimizers (e.g., RMSprop for EqProp)
- [ ] Implement gradient clipping analysis
  - [ ] Track gradient norms
  - [ ] Auto-tune clipping threshold per model
- [ ] Data augmentation experiments
  - [ ] RandomCrop, RandomFlip for vision tasks
  - [ ] Measure robustness improvement

### 4.3 Ensemble & Transfer Learning
- [ ] Implement model ensembles
  - [ ] Average predictions from top-5 models per task
  - [ ] Measure ensemble vs single-model accuracy gain
- [ ] Test transfer learning
  - [ ] Pre-train on MNIST, fine-tune on Fashion-MNIST
  - [ ] Pre-train on char_ngram, fine-tune on tiny_shakespeare
  - [ ] Measure transfer effectiveness for bioplausible methods

### 4.4 Hardware Efficiency Optimization
- [ ] Profile computation bottlenecks
  - [ ] Use PyTorch profiler to identify slow ops
  - [ ] Optimize EqProp equilibrium settling (currently slow)
- [ ] Implement mixed precision training (FP16)
- [ ] Add gradient checkpointing for deep models
- [ ] Measure FLOPS and memory usage per model

---

## Phase 5: Scientific Rigor & Interpretability 🔬

### 5.1 Ablation Studies
- [ ] Systematic component ablations
  - [ ] EqProp: Test nudge_factor values (0.1, 0.5, 1.0, 2.0)
  - [ ] Transformers: Attention-only vs Recurrent vs Hybrid
  - [ ] Deep Hebbian: Test layer counts (10, 50, 100, 200)
- [ ] Feature attribution analysis
  - [ ] Which layers contribute most to accuracy?
  - [ ] Visualize learned representations (t-SNE of hidden states)

### 5.2 Theoretical Analysis
- [ ] Energy landscape visualization for Energy-based models
  - [ ] Plot energy convergence over equilibrium steps
  - [ ] Compare stable vs unstable trials
- [ ] Credit assignment analysis
  - [ ] Measure gradient alignment with true gradient (for bioplausible methods)
  - [ ] Compare feedback alignment effectiveness
- [ ] Sample complexity curves
  - [ ] Plot accuracy vs training samples (not just epochs)
  - [ ] Identify sample-efficient algorithms

### 5.3 Reproducibility & Uncertainty Quantification
- [ ] Add seed-based reproducibility testing
  - [ ] Run best configs with 5 different seeds
  - [ ] Report mean ± std across seeds
- [ ] Implement Bayesian neural networks for uncertainty
  - [ ] Test dropout-based uncertainty estimates
  - [ ] Compare bioplausible vs backprop calibration

### 5.4 Comparative Benchmarking
- [ ] Add standard benchmark datasets
  - [ ] CIFAR100 (harder than CIFAR10)
  - [ ] ImageNet (if feasible)
  - [ ] Penn Treebank (language)
- [ ] Compare against published baselines
  - [ ] Literature review: state-of-art for bioplausible methods
  - [ ] Gap analysis: How close are we to SOTA?

---

## Phase 6: Real-World Viability Assessment 🌍

### 6.1 Practical Constraint Testing
- [ ] Online learning capability
  - [ ] Test continual learning (sequential tasks without forgetting)
  - [ ] Measure catastrophic forgetting rate
- [ ] Low-data regime experiments
  - [ ] Train on 10%, 25%, 50%, 100% of dataset
  - [ ] Identify data-efficient algorithms
- [ ] Edge deployment feasibility
  - [ ] Model compression (pruning, quantization)
  - [ ] Measure accuracy vs model size tradeoff
  - [ ] Target: <1MB models for embedded systems

### 6.2 Robustness & Safety
- [ ] Adversarial robustness testing
  - [ ] FGSM, PGD attacks on best models
  - [ ] Compare bioplausible vs backprop adversarial vulnerability
- [ ] Out-of-distribution (OOD) detection
  - [ ] Test on corrupted MNIST-C, CIFAR-10-C
  - [ ] Measure OOD detection accuracy
- [ ] Noise robustness
  - [ ] Add Gaussian noise to inputs (σ = 0.1, 0.3, 0.5)
  - [ ] Identify robust algorithms

### 6.3 Domain-Specific Applications
- [ ] Medical imaging: Skin lesion classification (HAM10000 dataset)
  - [ ] Privacy-preserving learning (local updates without gradients)
  - [ ] Test bioplausible methods for federated learning
- [ ] Robotics: Continuous control (MuJoCo environments)
  - [ ] Test EqProp, Predictive Coding on RL tasks
  - [ ] Measure sample efficiency vs PPO/SAC
- [ ] Neuromorphic hardware simulation
  - [ ] Estimate energy efficiency on spiking architectures
  - [ ] Identify algorithms suited for Intel Loihi, IBM TrueNorth

### 6.4 Interpretability for Deployment
- [ ] Feature importance attribution
  - [ ] SHAP values for bioplausible models
  - [ ] Compare explainability vs backprop
- [ ] Decision boundary visualization
  - [ ] Plot 2D projections of learned boundaries
- [ ] Human-in-the-loop validation
  - [ ] User study: Can humans understand bioplausible model decisions?

### 6.5 Synthesis: Viability Report
- [ ] Generate comprehensive viability assessment
  - [ ] Accuracy: Where do bioplausible methods match/exceed backprop?
  - [ ] Efficiency: Training time, memory, energy comparison
  - [ ] Robustness: Adversarial, OOD, noise resistance
  - [ ] Deployability: Edge constraints, continual learning
- [ ] Identify "killer apps" for bioplausible methods
  - [ ] Where are they superior? (e.g., privacy, continual learning, neuromorphic)
  - [ ] Where are they insufficient? (e.g., large-scale vision)
- [ ] Publish findings as technical report or paper

---

## Phase 7: Infrastructure & Documentation 📚

### 7.1 Code Quality
- [ ] Add comprehensive unit tests (target 80% coverage)
- [ ] Type hints for all public APIs
- [ ] Refactor monolithic files (e.g., `core.py` is 1384 lines)
- [ ] Add docstrings with usage examples

### 7.2 Documentation
- [ ] Write user guide: "Getting Started with Scientist++"
- [ ] Document experimental workflow (from smoke to deep tier)
- [ ] Create algorithm catalog with pros/cons per method
- [ ] Add troubleshooting guide (common errors, solutions)

### 7.3 Automation
- [ ] CI/CD pipeline for regression testing
- [ ] Automatic report generation on experiment completion
- [ ] Slack/email notifications for significant findings
- [ ] Auto-backup of database and checkpoints

---

## Success Metrics

**By End of Phase 3**:
- [ ] 0 critical bugs (all P0 items resolved)
- [ ] All Impact charts disambiguated by task+tier
- [ ] ≥100 Standard tier trials completed
- [ ] Task allocation balanced (no single task >30%)

**By End of Phase 4**:
- [ ] ≥1 bioplausible model achieves 92%+ on MNIST (match backprop)
- [ ] CIFAR10 best result >45% (8% improvement over current 37%)
- [ ] RL tasks >50% average reward (2x current cartpole 26%)

**By End of Phase 5**:
- [ ] Statistical significance established for all model comparisons
- [ ] Confidence intervals <5% width for top-10 models
- [ ] Ablation studies published for 3 algorithm families

**By End of Phase 6**:
- [ ] Identified ≥3 real-world use cases where bioplausible > backprop
- [ ] Deployed 1 demo application (e.g., federated learning, edge inference)
- [ ] Technical report documenting viability findings

---

## Notes

- **Prioritize iteratively**: Complete Phase 1 fully before starting Phase 2
- **Document everything**: Each experiment should have hypothesis, setup, results
- **Fail fast**: If approach doesn't work after 3 attempts, pivot
- **Collaborate**: Share findings with neuroscience/ML community for feedback
