# TorEqProp Experiments & Benchmarks

This directory contains the implementation of research tracks and comprehensive benchmarks for the Equilibrium Propagation project.

## Research Tracks (Tracks 34-40)

These scripts correspond to the "Breakthrough" phase of the roadmap.

- **Track 34**: [`cifar_breakthrough.py`](./cifar_breakthrough.py) - ModernConvEqProp on CIFAR-10.
- **Track 35**: [`memory_scaling_demo.py`](./memory_scaling_demo.py) - O(1) memory scaling verification.
- **Track 36**: [`energy_confidence.py`](./energy_confidence.py) - Energy-based OOD detection.
- **Track 37**: 
    - [`language_modeling.py`](./language_modeling.py) - Single EqProp character-level LM training
    - [`language_modeling_comparison.py`](./language_modeling_comparison.py) - **EqProp vs Backprop comparison** with parameter efficiency analysis
    - [`lm_scale_study.py`](./lm_scale_study.py) - Dataset size and sequence length scaling study (See [LM_SCALE_STUDY.md](./LM_SCALE_STUDY.md))

- **Track 38**: [`adaptive_compute.py`](./adaptive_compute.py) - Variable computation limits per sample.
- **Track 39**: [`diffusion_mnist.py`](./diffusion_mnist.py) - Score-based generative modeling.
- **Track 40**: [`flop_analysis.py`](./flop_analysis.py) - Hardware efficiency and FLOP counting.

## Comprehensive Benchmarks

These scripts provide rigorous statistical validation of key claims.

- **Spectral Normalization**: 
    - [`sn_benchmark_datasets.py`](./sn_benchmark_datasets.py) - Effects across multiple datasets (KMNIST, SVHN, etc.)
    - [`sn_benchmark_model_size.py`](./sn_benchmark_model_size.py) - Effects across model sizes
    - [`sn_stress_test.py`](./sn_stress_test.py) - Stability under extreme conditions (high LR, many steps)
    - [`track_a1_stability.py`](./track_a1_stability.py) - Architecture-agnostic stability analysis

## Usage

Run any script directly from the project root:

```bash
python experiments/cifar_breakthrough.py
python experiments/sn_benchmark_datasets.py
```

Note: These scripts are also invoked by the main `verify.py` harness during full validation.
