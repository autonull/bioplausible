# AutoScientist: Autonomous Discovery System

The **AutoScientist** is a fully autonomous agent designed to discover, validate, and stress-test biologically plausible learning algorithms. It runs continuously, managing its own experiment schedule based on a "Discovery Funnel."

## 🚀 Quick Start

```bash
# Start the Scientist (runs until Ctrl+C)
biopl-scientist

# Enable resource monitoring (pause on high load)
biopl-scientist --check-resources

# Generate a Report (while scientist is running or after)
biopl-report --out ./my_report
```

## 🧠 The Discovery Funnel

The scientist manages experiments through 5 rigorous tiers:

1.  **🌫️ SMOKE** (1 min): "Does it run?"
    *   Goal: Verify code stability and basic learning (> random chance).
    *   Action: Prioritized for new models. Failures are pruned.
2.  **⛵ SHALLOW** (10 min): "Is it promising?"
    *   Goal: Hyperparameter sweep on small scale.
    *   Action: Promoted if > 80% of baseline.
3.  **⚖️ STANDARD** (1 hr): "Can it compete?"
    *   Goal: Full training on standard datasets.
    *   Action: Promoted if gap to Backprop < 5%.
4.  **✅ VERIFICATION** (Repeats): "Is it real?"
    *   Goal: Statistical significance.
    *   Action: Top models are automatically re-run with 3 different seeds.
5.  **🔄 CROSS-VAL** (Overnight): "Does it generalize?"
    *   Goal: 5-Fold Cross-Validation.
    *   Action: Scheduled only for verified, high-performing models.
6.  **🛡️ ROBUSTNESS** (Stress Test): "Is it fragile?"
    *   Goal: Noise injection, adversarial attacks.
    *   Action: Scheduled for top DEEP models.

## 📊 Reports

The `biopl-report` tool generates a publication-quality Markdown site containing:
*   **Leaderboards**: Best accuracy per task.
*   **Pareto Frontiers**: Accuracy vs. Parameter Efficiency.
*   **Significance Matrices**: Heatmaps of P-values between algorithms.
*   **ML Insights**: Decision Trees explaining *why* certain hyperparameters work (e.g., "If LR > 0.01 and Beta < 0.5, then Accuracy > 90%").

## 🛠️ Architecture

*   **State Persistence**: All state is stored in `bioplausible.db` via Optuna. The scientist can be stopped and restarted anytime without losing progress.
*   **Robustness**:
    *   **Crash Recovery**: Exponential backoff if trials fail.
    *   **Resource Aware**: Can pause if system CPU/RAM > 90%.
    *   **Dynamic Priority**: Boosts neglected models to prevent starvation.
