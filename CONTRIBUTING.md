# Contributing to Solve_IVP_NS

Thanks for your interest in contributing!

## Development setup

- Python 3.10–3.13 recommended.
- Create a virtual environment and install the project in editable mode:

```
pip install -e .[test]
```

## Running tests

```
pytest -q
```

## Style

- Keep public APIs stable.
- Add docstrings (Args/Returns/Shapes/Example).
- Prefer small PRs with focused changes.

## Pull requests

- Link issues and describe the change and rationale.
- Include tests for new behavior.
- Update docs when public APIs change.
