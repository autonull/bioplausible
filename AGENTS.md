# AGENTS.md: Coding Standards and Agent Guidelines

This document defines the coding standards, cleanup rules, and behavioral guidelines for agents (human or autonomous) working on the **Bioplausible** codebase.

## 1. Code Style

*   **Formatting**: All code must be formatted with **Black**.
*   **Imports**: Imports must be sorted with **Isort**.
*   **Linting**: Code must be free of **Flake8** errors (unused imports, undefined variables, etc.).
*   **Line Length**: Adhere to Black's default (88 characters), with flexibility up to 100 for complex scientific expressions.

## 2. Type Hinting

*   **Mandatory**: All public functions and methods must have type hints for arguments and return values.
*   **Specifics**: Use `List`, `Dict`, `Optional`, `Union` from `typing` (or built-ins in Python 3.9+) rather than `Any` where possible.
*   **Data Classes**: Use `dataclasses` for structured data transfer objects.

## 3. Documentation

*   **Docstrings**: All modules, classes, and public methods must have **Google-style** docstrings.
    *   *Args*: Description of arguments.
    *   *Returns*: Description of return values.
    *   *Raises*: potential exceptions.
*   **Comments**: Use comments to explain *why*, not *what*. Remove commented-out code unless it is a specific, labeled "TODO" or "NOTE".

## 4. Refactoring Guidelines

*   **Function Length**: Aim for functions shorter than 50 lines. Extract logic into helper methods (prefixed with `_`) if a function grows too large.
*   **Cyclomatic Complexity**: Avoid deeply nested `if/else` blocks. Use guard clauses (`if not condition: return`) to flatten logic.
*   **Variable Naming**: Use descriptive names (e.g., `experiment_task` instead of `t`).
*   **Modularity**: Keep related logic together. Move disparate utility functions to `utils.py` or specific modules.

## 5. Agent Behavior (The "Scientist")

*   **Logging**: Use the standard `logging` library. Do **not** use `print()` for status updates (use `DASHBOARD` or `logger`).
*   **Resource Management**: Explicitly close connections (DB, files) and clean up heavy resources (GPU memory) using `try...finally` blocks.
*   **State Persistence**: Agents must be stateless between runs or persist state to the database (`bioplausible.db`). Do not rely on in-memory global state across restarts.
*   **Error Handling**: Catch specific exceptions. If catching `Exception`, log the traceback using `logger.error(..., exc_info=True)`.

## 6. Cleanup Checklist

When refactoring, perform the following:
1.  [ ] Run `isort .` and `black .`.
2.  [ ] Remove unused imports.
3.  [ ] Fix "bare" `except:` clauses.
4.  [ ] Replace magic numbers with named constants.
5.  [ ] Ensure `__init__.py` files expose only necessary symbols.
6.  [ ] Verify all tests pass: `pytest tests/`.
