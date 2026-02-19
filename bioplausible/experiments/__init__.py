"""
Bioplausible Experiments Package

Utilities for experimentation, research, and discovery of novel
machine learning approaches.

Quick Start:
    from bioplausible.experiments import ExperimentRunner, quick_comparison
    
    # Quick comparison of optimizers
    results = quick_comparison(
        model_name='looped_mlp',
        optimizer_names=['smep', 'smep_fast', 'muon_backprop'],
        epochs=3,
    )
    
    # Full experiment
    runner = ExperimentRunner()
    result = runner.run(
        model_name='looped_mlp',
        optimizer_name='smep',
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
    )
    
    # Use research presets
    from bioplausible.experiments import get_preset, list_presets, run_preset
    
    print(list_presets())  # All presets
    print(list_presets('performance'))  # Performance presets
    
    preset = get_preset('performance_vision_default')
    result = run_preset('performance_vision_default', train_loader, val_loader)
"""

from bioplausible.experiments.utils import (
    ExperimentResult,
    ExperimentConfig,
    ExperimentRunner,
    HyperparameterSearch,
    quick_comparison,
    benchmark_model,
)

from bioplausible.experiments.presets import (
    ResearchPreset,
    PERFORMANCE_PRESETS,
    SPEED_PRESETS,
    EFFICIENCY_PRESETS,
    BIOPLAUSIBLE_PRESETS,
    ROBUSTNESS_PRESETS,
    EXPLORATORY_PRESETS,
    ALL_PRESETS,
    get_preset,
    list_presets,
    get_preset_by_category,
    run_preset,
)

__all__ = [
    # Utils
    'ExperimentResult',
    'ExperimentConfig',
    'ExperimentRunner',
    'HyperparameterSearch',
    'quick_comparison',
    'benchmark_model',
    # Presets
    'ResearchPreset',
    'PERFORMANCE_PRESETS',
    'SPEED_PRESETS',
    'EFFICIENCY_PRESETS',
    'BIOPLAUSIBLE_PRESETS',
    'ROBUSTNESS_PRESETS',
    'EXPLORATORY_PRESETS',
    'ALL_PRESETS',
    'get_preset',
    'list_presets',
    'get_preset_by_category',
    'run_preset',
]
