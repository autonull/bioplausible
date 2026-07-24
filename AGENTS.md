## Toolchain
*   Language: **Python 3.14+**
*   **uv**: For dependency management, virtualenvs, and task running (`uv run`, `uv add`). Single lockfile (`uv.lock`); no `requirements.txt`.
*   **Ruff**: For formatting, linting, and import sorting. All config in `pyproject.toml`.
*   **Pyright** (or mypy): In **strict mode**. Type errors fail the build.
*   **pre-commit**: Runs ruff + type checker + tests before every commit.
*   **Line Length**: Ruff default (88). Relax per-line with `# noqa: <code>` and a reason, never globally.
*   Rules are enforced by tooling — if a rule can be a config or pre-commit hook, it is one.

## Type System
*   **Modern Syntax**: Built-in generics (`list[str]`), PEP 604 unions (`X | None`). Never import `List`, `Dict`, `Optional`, or `Union`.
*   **PEP 695**: Use `class Cache[T]: ...` and `type UserId = int` for generics and aliases.
*   **Interfaces & Narrowing**: Prefer `Protocol` over ABCs. Use `Self` for fluent returns. Use **`TypeIs`** for custom type narrowing to preserve original type context on failure.
*   **Value Sets**: Use `Literal` / `StrEnum` instead of bare strings.
*   **Data Modeling**:
    *   `@dataclass(frozen=True, slots=True)` for internal value objects.
    *   `TypedDict` for unvalidated structured dicts.
    *   **Pydantic v2** at I/O boundaries for runtime validation.
*   **No `Any`**: Replace with `object`, generics, or `Protocol`. If unavoidable, isolate and document why.

## Architecture & Control Flow
*   **Pattern Matching**: Use `match`/`case` for complex state/data routing, favoring it over chained `if/elif`.
*   **Complexity**: Let Ruff rules (`C901`, `PLR09xx`) enforce function size. Extract `_`-prefixed helpers rather than nesting deeper.
*   **Control Flow**: Flatten with guard clauses (`if not condition: return`). 
*   **Finally Blocks**: **Never** use `return`, `break`, or `continue` inside a `finally` block (PEP 765). This is a hard `SyntaxError` and silently swallows exceptions.
*   **Composition over Inheritance**: Favor small pure functions. Isolate side effects so core logic stays testable.
*   **Immutability**: Default to immutable structures (`tuple`, `frozenset`, frozen dataclasses) unless mutation is strictly required.

## Async Concurrency & Thread Safety
*   **Structured Concurrency**: Use `asyncio.TaskGroup` for concurrent tasks. Avoid `asyncio.gather` for complex flows.
*   **Event Loop Hygiene**: Never mix blocking I/O/CPU tasks with `async` code. Use `asyncio.to_thread` for legacy blocking calls.
*   **Thread Safety (PEP 703)**: With free-threaded CPython, **do not rely on the GIL**. Use explicit locks (`threading.Lock`), thread-local storage, or immutable data for shared mutable state.

## Documentation & Modules
*   **Docstrings**: **Google-style** on all public APIs. Type hints replace argument type documentation — focus on behavior, side effects, and invariants.
*   **Comments**: Explain *why*, not *what*. Delete dead code; use `# TODO(name): ...` for deferred work.
*   **Import Hygiene**: Avoid heavy computations or I/O at the module level. Prevent circular imports via local imports or dependency injection.

## Errors, Logging & Resources
*   **Safe Interpolation**: Use **t-strings** (PEP 750) for logging and templating to enable safe, deferred interpolation. Never use f-strings for database queries (use driver parameterization) or untrusted log inputs.
*   **Logging**: Standard `logging` (or `structlog`). Never `print()`. Include context: `logger.error("msg", extra={"task_id": id})`.
*   **Exceptions**: Define a small custom hierarchy per domain. Always chain: `raise DomainError("msg") from original_exception`. Use `except*` (PEP 654) for concurrent independent failures.
*   **Resources**: Use context managers (`with` / `async with`) for all resource lifecycles.

## Testing
*   **pytest + pytest-cov**: Enforce a coverage floor in CI (e.g., ≥85%).
*   **hypothesis**: Use for property-based tests on pure logic.
*   **Mocking**: Prefer Dependency Injection over `unittest.mock`. Use `pytest-mock` when strictly required.
*   **Fixtures**: Use fixtures over setup/teardown; `@pytest.mark.parametrize` over duplicated tests.

## Security & Project Structure
*   **Dependency Scanning**: Run `pip-audit` in CI.
*   **Static Analysis**: Enable Ruff's `S` (bandit) rule set. Never hardcode secrets.
*   **Project Structure**: `pyproject.toml` is the single source of truth. `__init__.py` exposes only the public API via `__all__`; internal modules are `_`-prefixed.
*   **CI Gate Order**: `ruff format --check` → `ruff check` → `pyright` → `pytest --cov` → `pip-audit`.

## Agent Commit Checklist
**Automated (pre-commit / CI — do not bypass):**
- [ ] `ruff format .` && `ruff check --fix .`
- [ ] `pyright .` — zero errors in strict mode
- [ ] `pytest --cov` — all tests pass, coverage floor met
