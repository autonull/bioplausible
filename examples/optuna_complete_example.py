"""
Complete Optuna Usage Example

Demonstrates end-to-end hyperparameter optimization using the new Optuna integration.
"""

from bioplausible.hyperopt import (HAS_OPTUNA, create_optuna_space,
                                   create_study, get_pareto_trials)
from bioplausible.models.registry import get_model_spec

if not HAS_OPTUNA:
    print("❌ This example requires Optuna. Install with: pip install optuna")
    exit(1)

print("=" * 70)
print("Bioplausible + Optuna: Complete Usage Example")
print("=" * 70)

# Example 1: Single-objective optimization (maximize accuracy)
print("\n📊 Example 1: Single-Objective Optimization")
print("-" * 70)

study_single = create_study(
    model_names=["EqProp MLP"],
    n_objectives=1,  # Single objective
    storage=None,  # In-memory (use "sqlite:///optuna.db" for persistence)
    study_name="eqprop_accuracy",
    use_pruning=True,
    sampler_name="tpe",  # Tree-structured Parzen Estimator
)


def objective_single(trial):
    """Simulate training and return accuracy."""
    config = create_optuna_space(trial, "EqProp MLP")

    # In real usage, you would:
    # accuracy = train_model(config)

    # For demo, simulate accuracy based on hyperparameters
    # Better lr and beta = higher accuracy
    simulated_accuracy = 0.7 + (config["lr"] * 10) + (config["beta"] * 0.2)
    simulated_accuracy = min(1.0, simulated_accuracy)  # Cap at 1.0

    return simulated_accuracy


print(f"Running {5} trials with TPE sampler...")
study_single.optimize(objective_single, n_trials=5, show_progress_bar=False)

print(f"\n✅ Best trial:")
print(f"   Trial: {study_single.best_trial.number}")
print(f"   Accuracy: {study_single.best_value:.4f}")
print(f"   Best hyperparameters:")
for key, value in study_single.best_trial.params.items():
    print(f"     {key}: {value}")

# Example 2: Multi-objective optimization (accuracy vs speed)
print("\n\n📊 Example 2: Multi-Objective Optimization (Pareto Front)")
print("-" * 70)

study_multi = create_study(
    model_names=["EqProp MLP"],
    n_objectives=2,  # Accuracy and loss
    storage=None,
    study_name="eqprop_pareto",
    use_pruning=False,  # Disable pruning for multi-objective
    sampler_name="nsga2",  # NSGA-II for multi-objective
)


def objective_multi(trial):
    """Simulate training and return (accuracy, loss)."""
    config = create_optuna_space(trial, "EqProp MLP")

    # Simulate accuracy and loss
    simulated_accuracy = 0.7 + (config["lr"] * 10)
    simulated_loss = 0.5 - (config["lr"] * 2)

    simulated_accuracy = min(1.0, max(0.0, simulated_accuracy))
    simulated_loss = max(0.0, simulated_loss)

    return simulated_accuracy, simulated_loss


print(f"Running {10} trials with NSGA-II sampler...")
study_multi.optimize(objective_multi, n_trials=10, show_progress_bar=False)

pareto_trials = get_pareto_trials(study_multi)
print(f"\n✅ Found {len(pareto_trials)} trials on Pareto frontier:")
for i, trial in enumerate(pareto_trials[:3], 1):  # Show top 3
    print(f"\n   Pareto Trial {i} (#{trial.number}):")
    print(f"     Accuracy: {trial.values[0]:.4f}")
    print(f"     Loss: {trial.values[1]:.4f}")
    print(f"     lr: {trial.params['lr']:.6f}")
    print(f"     beta: {trial.params.get('beta', 'N/A')}")


# Example 3: Using with model registry
print("\n\n📊 Example 3: Automatic Search Space from ModelSpec")
print("-" * 70)

# The bridge automatically configures search space based on model spec
spec = get_model_spec("Holomorphic EqProp")
print(f"Model: {spec.name}")
print(f"Family: {spec.family}")

study_auto = create_study(
    model_names=["Holomorphic EqProp"],
    n_objectives=1,
    sampler_name="tpe",
)

trial = study_auto.ask()
config = create_optuna_space(trial, "Holomorphic EqProp")

print(f"\n✅ Auto-generated config for {spec.name}:")
for key, value in config.items():
    print(f"   {key}: {value}")

# Example 4: Comparing multiple models
print("\n\n📊 Example 4: Multi-Model Comparison")
print("-" * 70)

models_to_compare = [
    "EqProp MLP",
    "DFA (Direct Feedback Alignment)",
    "Backprop Baseline",
]
results = {}

for model_name in models_to_compare:
    study = create_study(
        model_names=[model_name],
        n_objectives=1,
        sampler_name="random",
    )

    def objective(trial):
        config = create_optuna_space(trial, model_name)
        # Simulate accuracy (in reality, train the model)
        return 0.8 + (hash(str(config)) % 100) / 500  # Pseudo-random

    study.optimize(objective, n_trials=3, show_progress_bar=False)
    results[model_name] = study.best_value

print("\n✅ Model Comparison Results:")
for model, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"   {model}: {accuracy:.4f}")

# Example 5: Persistent storage
print("\n\n📊 Example 5: Persistent Storage (SQLite)")
print("-" * 70)

# Create study with SQLite storage
study_persistent = create_study(
    model_names=["EqProp MLP"],
    n_objectives=1,
    storage="sqlite:///examples/optuna_demo.db",  # Persistent!
    study_name="persistent_study",
    sampler_name="tpe",
)

print(f"✅ Created study with SQLite storage")
print(f"   Database: examples/optuna_demo.db")
print(f"   Study can be resumed later with:")
print(
    f"   optuna.load_study(study_name='persistent_study', storage='sqlite:///examples/optuna_demo.db')"
)

# Run a few trials
study_persistent.optimize(objective_single, n_trials=3, show_progress_bar=False)
print(f"   Completed {len(study_persistent.trials)} trials")
print(f"   Best accuracy: {study_persistent.best_value:.4f}")

print("\n" + "=" * 70)
print("✅ All Examples Complete!")
print("=" * 70)

print("\n📚 Next Steps:")
print("  1. Use in SearchTab UI (automatic)")
print("  2. Visualize results: optuna-dashboard sqlite:///examples/optuna_demo.db")
print("  3. Resume studies from database")
print("  4. Scale to distributed optimization")

print("\n💡 Key Benefits of Optuna Integration:")
print("  • Automatic search space from ModelSpec")
print("  • TPE sampler (10-100× more efficient than random)")
print("  • Multi-objective optimization (NSGA-II)")
print("  • Automatic pruning (saves 30-50% compute)")
print("  • Persistent storage (resume interrupted runs)")
print("  • Built-in visualization and analysis")
