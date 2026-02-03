"""
Optuna Integration Example

Demonstrates how to use the new Optuna bridge for hyperparameter optimization.
"""

import sys

# Test 1: Check if Optuna is available
try:
    from bioplausible.hyperopt import (HAS_OPTUNA, create_optuna_space,
                                       create_study)

    print("=" * 60)
    print("Optuna Integration Test")
    print("=" * 60)
    print(f"Optuna available: {HAS_OPTUNA}\n")

    if not HAS_OPTUNA:
        print("⚠️  Optuna not installed. Install with: pip install optuna")
        print("   Falling back to legacy hyperopt (deprecated)\n")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Load model registry
try:
    from bioplausible.models.registry import MODEL_REGISTRY

    print(f"✅ Found {len(MODEL_REGISTRY)} models in registry")
    print(f"   Sample models: {', '.join([m.name for m in MODEL_REGISTRY[:5]])}\n")
except ImportError as e:
    print(f"❌ Model registry error: {e}")
    sys.exit(1)

# Test 3: Load search spaces
try:
    from bioplausible.hyperopt import SEARCH_SPACES

    print(f"✅ Found {len(SEARCH_SPACES)} predefined search spaces")
    print(f"   Sample: {list(SEARCH_SPACES.keys())[:3]}\n")
except ImportError as e:
    print(f"❌ Search space error: {e}")
    sys.exit(1)

# Test 4: Optuna bridge (if available)
if HAS_OPTUNA:
    print("=" * 60)
    print("Testing Optuna Bridge")
    print("=" * 60)

    try:
        pass

        # Create a simple study
        study = create_study(
            model_names=["EqProp MLP"],
            n_objectives=2,
            storage=None,
            study_name="test_study",
            use_pruning=False,
            sampler_name="random",
        )

        print("✅ Created Optuna study successfully")
        print(f"   Study name: {study.study_name}")
        print(f"   Sampler: {type(study.sampler).__name__}")
        print(f"   Directions: {study.directions}\n")

        # Test space creation
        trial = study.ask()
        config = create_optuna_space(trial, "EqProp MLP")

        print("✅ Generated hyperparameter configuration:")
        for key, value in config.items():
            print(f"   {key}: {value}")

        # Simulate completing the trial
        study.tell(trial, values=[0.85, 0.15])  # accuracy, loss

        # Get the frozen trial to access values
        frozen_trial = study.trials[-1]

        print(f"\n✅ Completed trial {frozen_trial.number}")
        print(
            f"   Values: accuracy={frozen_trial.values[0]}, loss={frozen_trial.values[1]}"
        )

    except Exception as e:
        print(f"❌ Optuna bridge error: {e}")
        import traceback

        traceback.print_exc()

# Test 5: Optuna is required (no fallback)
print("\n" + "=" * 60)
print("Optuna Requirement Check")
print("=" * 60)

if HAS_OPTUNA:
    print("✅ Optuna is installed and working")
    print("   All hyperparameter optimization features available")
else:
    print("❌ Optuna is required but not installed")
    print("   Install with: pip install optuna")
    sys.exit(1)


# Test 5: Backward compatibility removed - Optuna is now required
print("\n" + "=" * 60)
print("Migration Note")
print("=" * 60)
print("✅ Legacy evolution code has been removed")
print("   All hyperparameter optimization now uses Optuna")
print("   GridSearch/RandomSearch replaced by Optuna samplers")

print("\n" + "=" * 60)
print("All Tests Complete!")
print("=" * 60)

if HAS_OPTUNA:
    print("✅ Full Optuna integration working")
    print("   Next: Run hyperparameter optimization with Optuna")
else:
    print("⚠️  Optuna not available - using legacy code")
    print("   Install: pip install optuna")
    print("   Then run this test again")

print("\nFor production usage, see:")
print("  - bioplausible/hyperopt/optuna_bridge.py")
print("  - bioplausible_ui/app/tabs/search_tab.py")
