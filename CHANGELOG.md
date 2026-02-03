# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-22

### Added
- **Scientific Infrastructure**:
    - `ExperimentTracker`: Unified interface for Weights & Biases logging.
    - `ResultVisualizer`: Automated generation of publication-quality plots (Leaderboards, Pareto frontiers, Lipschitz trajectories).
    - `StatisticalAnalyzer`: Tools for rigorous algorithm comparison (Cohen's d, t-tests).
- **Core Improvements**:
    - Integrated Optuna for hyperparameter optimization, replacing custom evolutionary algorithms.
    - Triton kernel support for accelerated Equilibrium Propagation dynamics (optional).
    - `LoopedMLP` now supports spectral normalization for stable fixed-point convergence.
- **UI Refactoring**:
    - Split `bioplausible_ui` into `studio`, `leaderboard`, `lab`, and `app` subpackages.
    - Enhanced "Deploy" tab logic.
- **Validation Tracks**:
    - Added tracks 34-40 including EqProp Diffusion, O(1) Memory Scaling, and Hardware Efficiency Analysis.

### Changed
- Refactored `ScientistReporter` to use the new visualization and statistics modules.
- Cleaned up codebase by removing conversational comments and deprecated "Archive" code.
- Updated `bioplausible.models` imports to prevent circular dependencies.

### Fixed
- Fixed Triton import errors on systems without Triton installed.
- Fixed circular imports in `benchmark.py`.
- Improved test suite robustness with better mocking.
