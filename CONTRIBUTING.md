# Contributing to Solve_IVP_NS

Thanks for your interest in contributing!

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[test]
```

## Running tests

```bash
pytest -q
```

## Building documentation

```bash
cd docs && make clean html
```

## Style

- Keep public APIs stable.
- Use NumPy-style docstrings (Args/Returns/Shapes/Example).
- Prefer small PRs with focused changes.

## Pull requests

- Link issues and describe the change and rationale.
- Include tests for new behavior.
- Update docs when public APIs change.
