# Scientist++ User Guide

## Introduction

Scientist++ is an autonomous research agent designed to explore, optimize, and validate bioplausible neural network algorithms. It manages the entire experimental lifecycle, from initial "smoke tests" to rigorous cross-validation and robustness analysis.

## Workflow Tiers

The scientist operates using a tiered promotion system. A model/task pair must "pass" a tier to be promoted to the next.

1.  **Smoke Tier (3 Epochs)**
    *   **Goal**: Quick check if the model runs and learns *anything*.
    *   **Criteria**: Accuracy > 15% (for classification).
    *   **Budget**: Very low.

2.  **Shallow Tier (7 Epochs)**
    *   **Goal**: Initial exploration of hyperparameter space.
    *   **Criteria**: Accuracy > 40%.
    *   **Action**: If successful, used to refine search space for Standard tier.

3.  **Standard Tier (15 Epochs)**
    *   **Goal**: Balanced evaluation and optimization.
    *   **Criteria**: Accuracy > 60%.
    *   **Action**: High performers trigger:
        *   **Verification**: Repeats of the best config.
        *   **Ablation Studies**: Testing component impact (e.g., symmetric weights).
        *   **Continual Learning**: Split-MNIST sequences.
        *   **Low-Data Regime**: Testing on 10% and 25% of data.

4.  **Deep Tier (30 Epochs)**
    *   **Goal**: Thorough convergence analysis.
    *   **Criteria**: Accuracy > 80%.
    *   **Action**: Triggers **Robustness Checks** (Noise, Adversarial Attacks).

5.  **Cross-Validation Tier**
    *   **Goal**: Statistical rigor.
    *   **Action**: 5-Fold Cross-Validation for verified top performers.

## Advanced Features

### Robustness Testing
Models that reach the Deep tier undergo stress testing:
*   **Noise Injection**: Can the model recover from internal noise?
*   **Input Perturbation**: Resilience to noisy inputs.
*   **OOD Detection**: Ability to distinguish In-Distribution data from Out-of-Distribution noise (measured via Max Softmax Probability gap).
*   **Adversarial Attacks**: Resilience to FGSM (Fast Gradient Sign Method) attacks.

### Low-Data Regime
Top models in Standard tier are automatically tested on data subsets (10%, 25%) to evaluate sample efficiency, a key potential advantage of bioplausible algorithms.

### Interpretability
The system supports generating:
*   **Saliency Maps**: Visualizing input importance.
*   **Decision Boundaries**: 2D projections of classification boundaries.

## Usage

**Start the Scientist**:
```bash
python -m bioplausible.scientist.cli
# OR
./run_scientist.sh
```

**Generate Reports**:
```bash
python -m bioplausible.scientist.cli --report
# OR
./generate_report.sh
```

Reports are generated in `reports/` and include:
*   Leaderboards (Accuracy, Efficiency, Robustness)
*   Hyperparameter Impact Plots
*   Convergence Curves
*   Research Synthesis (High-level insights)
