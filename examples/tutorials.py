"""
Bioplausible Tutorial Examples

Complete examples for common use cases.

Contents:
- Tutorial 1: Quick Start with Presets
- Tutorial 2: Comparing Optimizers
- Tutorial 3: Hyperparameter Search
- Tutorial 4: Model Export and Deployment
- Tutorial 5: Statistical Analysis
- Tutorial 6: Custom Experiment Design
"""

# ============================================================================
# Tutorial 1: Quick Start with Presets
# ============================================================================

def tutorial_1_quick_start():
    """
    Tutorial 1: Quick Start with Research Presets
    
    This tutorial shows how to quickly run experiments using pre-configured
    research presets.
    """
    from bioplausible import run_preset, list_presets, get_preset
    from bioplausible.datasets import get_vision_dataset
    
    print("=" * 60)
    print("TUTORIAL 1: Quick Start with Presets")
    print("=" * 60)
    
    # List available presets
    print("\nAvailable presets:")
    for preset_name in list_presets()[:5]:
        print(f"  - {preset_name}")
    
    # Get preset details
    preset = get_preset('speed_vision_fast')
    print(f"\nPreset: {preset.name}")
    print(f"  Model: {preset.model_name}")
    print(f"  Optimizer: {preset.optimizer_name}")
    print(f"  Expected: {preset.expected_accuracy}")
    
    # Load data
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Run preset experiment
    print("\nRunning preset experiment...")
    result = run_preset(
        preset_name='speed_vision_fast',
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )
    
    print(f"\nResult: {result.val_accuracy:.2f}% validation accuracy")
    print(f"Training time: {result.training_time:.1f}s")


# ============================================================================
# Tutorial 2: Comparing Optimizers
# ============================================================================

def tutorial_2_compare_optimizers():
    """
    Tutorial 2: Comparing Multiple Optimizers
    
    This tutorial shows how to compare different optimizers on the same model.
    """
    from bioplausible import ExperimentRunner, quick_comparison
    from bioplausible.datasets import get_vision_dataset
    
    print("=" * 60)
    print("TUTORIAL 2: Comparing Optimizers")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Method 1: Quick comparison (simplest)
    print("\nMethod 1: Quick Comparison")
    results = quick_comparison(
        model_name='looped_mlp',
        optimizer_names=['smep', 'smep_fast', 'muon_backprop'],
        epochs=3,
    )
    
    print("\nResults (sorted by accuracy):")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.optimizer_name}: {r.val_accuracy:.2f}%")
    
    # Method 2: Full comparison with ExperimentRunner
    print("\nMethod 2: Full Experiment Runner")
    runner = ExperimentRunner(device='cpu')
    
    results = runner.compare_optimizers(
        model_name='looped_mlp',
        optimizer_names=['smep', 'smep_fast'],
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=3,
    )
    
    print("\nDetailed results:")
    for r in results:
        print(f"  {r.optimizer_name}:")
        print(f"    Val Accuracy: {r.val_accuracy:.2f}%")
        print(f"    Speed: {r.steps_per_second:.1f} steps/s")
        print(f"    Parameters: {r.num_parameters:,}")


# ============================================================================
# Tutorial 3: Hyperparameter Search
# ============================================================================

def tutorial_3_hyperparameter_search():
    """
    Tutorial 3: Hyperparameter Search
    
    This tutorial shows how to search for optimal hyperparameters.
    """
    from bioplausible import HyperparameterSearch
    from bioplausible.datasets import get_vision_dataset
    
    print("=" * 60)
    print("TUTORIAL 3: Hyperparameter Search")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Grid search
    search = HyperparameterSearch(device='cpu')
    
    print("\nRunning grid search...")
    best_params, best_result = search.grid_search(
        model_name='looped_mlp',
        optimizer_name='smep',
        param_grid={
            'lr': [0.001, 0.01],
            'settle_steps': [10, 20],
            'beta': [0.3, 0.5],
        },
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        verbose=True,
    )
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_result.val_accuracy:.2f}%")


# ============================================================================
# Tutorial 4: Model Export and Deployment
# ============================================================================

def tutorial_4_export_deploy():
    """
    Tutorial 4: Model Export and Deployment
    
    This tutorial shows how to export models for production deployment.
    """
    from bioplausible import ModelZoo, OptimizerZoo, export_model, load_model, InferenceEngine
    from bioplausible.datasets import get_vision_dataset
    import torch
    
    print("=" * 60)
    print("TUTORIAL 4: Model Export and Deployment")
    print("=" * 60)
    
    # Load data
    train_loader, _, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Create and train model
    print("\n1. Creating and training model...")
    model = ModelZoo.get('looped_mlp', input_dim=784, hidden_dim=256, output_dim=10)
    optimizer = OptimizerZoo.get('smep', model.parameters(), model=model)
    
    # Quick training
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= 10:
            break
        x = x.view(x.shape[0], -1)
        optimizer.step(x=x, target=y)
    
    print("   Training complete!")
    
    # Export model
    print("\n2. Exporting model...")
    info = export_model(
        model=model,
        model_name='looped_mlp',
        model_params={'input_dim': 784, 'hidden_dim': 256, 'output_dim': 10},
        output_dir='./tutorial_exports',
        formats=['config', 'state'],
        optimizer=optimizer,
        training_metrics={'train_steps': 10},
    )
    
    print(f"   Exported to: {info.export_path}")
    print(f"   Formats: {info.export_format}")
    
    # Load model
    print("\n3. Loading exported model...")
    loaded_model, config = load_model('./tutorial_exports')
    print(f"   Loaded: {config['model_name']}")
    
    # Create inference engine
    print("\n4. Creating inference engine...")
    engine = InferenceEngine.from_export('./tutorial_exports')
    
    # Run inference
    print("\n5. Running inference...")
    test_input = torch.randn(1, 784)
    prediction = engine.predict_class(test_input)
    confidence = engine.predict_proba(test_input).max().item()
    
    print(f"   Prediction: Class {prediction}")
    print(f"   Confidence: {confidence:.2%}")


# ============================================================================
# Tutorial 5: Statistical Analysis
# ============================================================================

def tutorial_5_statistical_analysis():
    """
    Tutorial 5: Statistical Analysis of Results
    
    This tutorial shows how to analyze results statistically.
    """
    from bioplausible import ExperimentRunner, analyze_results
    from bioplausible.datasets import get_vision_dataset
    
    print("=" * 60)
    print("TUTORIAL 5: Statistical Analysis")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Run multiple experiments
    print("\n1. Running experiments...")
    runner = ExperimentRunner(device='cpu')
    
    results = []
    for opt_name in ['smep', 'smep_fast', 'muon_backprop']:
        print(f"   Training with {opt_name}...")
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name=opt_name,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            verbose=False,
        )
        results.append(result)
    
    # Analyze results
    print("\n2. Analyzing results...")
    report = analyze_results(results)
    
    print("\n" + report.summary())
    
    # Statistical comparison
    print("\n3. Statistical comparison...")
    from bioplausible import ResultAnalyzer
    
    analyzer = ResultAnalyzer()
    analyzer.add_results(results)
    
    comp = analyzer.compare_optimizers('smep', 'muon_backprop')
    if comp:
        print(comp.summary())


# ============================================================================
# Tutorial 6: Custom Experiment Design
# ============================================================================

def tutorial_6_custom_experiment():
    """
    Tutorial 6: Custom Experiment Design
    
    This tutorial shows how to design custom experiments.
    """
    from bioplausible import ModelZoo, OptimizerZoo, ExperimentRunner
    from bioplausible.datasets import get_vision_dataset
    
    print("=" * 60)
    print("TUTORIAL 6: Custom Experiment Design")
    print("=" * 60)
    
    # Load data
    train_loader, val_loader, _ = get_vision_dataset(
        dataset='mnist',
        batch_size=128,
        normalize=True,
    )
    
    # Design 1: Scaling study (model size)
    print("\n1. Scaling Study: Model Size")
    runner = ExperimentRunner(device='cpu')
    
    hidden_sizes = [64, 128, 256]
    results = []
    
    for hidden in hidden_sizes:
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name='smep',
            train_loader=train_loader,
            val_loader=val_loader,
            model_params={'hidden_dim': hidden},
            epochs=2,
            verbose=False,
        )
        results.append((hidden, result.num_parameters, result.val_accuracy))
        print(f"   Hidden={hidden}: {result.num_parameters:,} params, {result.val_accuracy:.2f}%")
    
    # Design 2: Ablation study (optimizer components)
    print("\n2. Ablation Study: Settling Steps")
    
    settle_steps_list = [5, 10, 20]
    results = []
    
    for steps in settle_steps_list:
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name='smep',
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_params={'settle_steps': steps},
            epochs=2,
            verbose=False,
        )
        results.append((steps, result.val_accuracy, result.training_time))
        print(f"   Steps={steps}: {result.val_accuracy:.2f}%, {result.training_time:.1f}s")
    
    # Design 3: Cross-validation style
    print("\n3. Multiple Runs for Stability")
    
    accuracies = []
    for run in range(3):
        result = runner.run(
            model_name='looped_mlp',
            optimizer_name='smep',
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            verbose=False,
        )
        accuracies.append(result.val_accuracy)
        print(f"   Run {run+1}: {result.val_accuracy:.2f}%")
    
    import numpy as np
    print(f"   Mean: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")


# ============================================================================
# Main: Run all tutorials
# ============================================================================

if __name__ == '__main__':
    import sys
    
    tutorials = {
        '1': tutorial_1_quick_start,
        '2': tutorial_2_compare_optimizers,
        '3': tutorial_3_hyperparameter_search,
        '4': tutorial_4_export_deploy,
        '5': tutorial_5_statistical_analysis,
        '6': tutorial_6_custom_experiment,
    }
    
    if len(sys.argv) > 1:
        # Run specific tutorial
        tutorial_id = sys.argv[1]
        if tutorial_id in tutorials:
            tutorials[tutorial_id]()
        else:
            print(f"Unknown tutorial: {tutorial_id}")
            print(f"Available: {', '.join(tutorials.keys())}")
    else:
        # Run all tutorials
        print("Running all tutorials...\n")
        for tutorial_id, tutorial_fn in tutorials.items():
            try:
                tutorial_fn()
                print("\n" + "=" * 60 + "\n")
            except Exception as e:
                print(f"\nTutorial {tutorial_id} failed: {e}\n")
                print("=" * 60 + "\n")
