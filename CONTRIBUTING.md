# Contributing to Bioplausible

Thank you for your interest in contributing to Bioplausible! We welcome contributions from researchers and developers to help advance equilibrium propagation and bio-inspired AI.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/your-username/bioplausible.git
    cd bioplausible
    ```
3.  **Install dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```
    *Note: Triton and CuPy are optional but recommended for GPU acceleration.*

## Development Workflow

1.  **Create a branch** for your feature or fix:
    ```bash
    git checkout -b feature/my-new-feature
    ```
2.  **Implement your changes**.
3.  **Run tests** to ensure no regressions:
    ```bash
    pytest tests/
    ```
4.  **Format your code**:
    ```bash
    black bioplausible bioplausible_ui
    isort bioplausible bioplausible_ui
    ```
5.  **Commit your changes** with descriptive messages.

## Code Style

- We use **Black** for code formatting.
- We use **isort** for import sorting.
- We use **Flake8** for linting.
- Avoid conversational comments (e.g., "Let's try this..."). Use professional, descriptive comments.

## Adding New Models

1.  Implement your model in `bioplausible/models/`.
2.  Inherit from `BioModel` or `EqPropModel`.
3.  Register your model using the `@register_model` decorator.
4.  Add a test case in `tests/test_model_registry_instantiation.py`.

## Reporting Issues

Please use the GitHub Issues tracker to report bugs or request features. Provide as much detail as possible, including reproduction steps and environment information.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
