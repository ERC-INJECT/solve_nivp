# Solve_IVP_NS

## RL experiments (optional, user-defined)

The `RL_Adaption/` folder contains optional experiments where
`Solve_IVP_NS` is wrapped in a Gym-style environment and controlled by
reinforcement-learning agents (for example, TD3 or TQC from Stable
Baselines3).

In these notebooks the reward signal, observation mapping, and policy
configuration are defined directly in notebook cells. The core
`Solve_IVP_NS` package only supplies the nonsmooth solvers and the
`AdaptiveStepperEnv` wrapper; all RL-specific choices are intentionally
left to the user so they can adapt the workflow to their own
applications.

These experiments are not required for installing, testing, or using the
library. To try them out install the optional RL dependencies:

```bash
pip install -e .[rl]
```

A Python toolkit for integrating nonsmooth ODE/DAE systems via projection-based constraints and semismooth Newton solves. It provides implicit integrators (Backward Euler, Trapezoidal, theta/composite), projection operators (identity, sign, Coulomb-like, SOC), and an adaptive controller with optional acceleration.

## Installation

Recommended developer install:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[test]
```

Optional extras:

```bash
# RL experiments
pip install -e .[rl]
```

## Quickstart

```python
import numpy as np
from Solve_IVP_NS import solve_ivp_ns

# simple smooth rhs: y' = -y
rhs = lambda t, y: -y

t_span = (0.0, 1.0)
y0 = np.array([1.0])

# identity projection, VI solver via composite integrator
sol = solve_ivp_ns(
    fun=rhs,
    t_span=t_span,
    y0=y0,
    method='composite',
    projection='identity',
    solver='VI',
)

print(sol[0][:5], sol[1][:5])  # t, y samples
```

See `examples/` for notebooks on friction stick–slip, bouncing ball (contact/impact), SOC constraints, and sliding-mode control.

## Running tests

```bash
pytest -q
```

## Building the documentation

```bash
cd docs
make clean html
```
Open `docs/_build/html/index.html`.

## RL experiments (optional)

The `RL_Adaption/` folder contains optional experiments (TD3/TQC) for learned adaptivity on challenging nonsmooth problems. Large artifacts are ignored by Git and not required for core installation or testing.

## Citation

See `CITATION.cff`. If you use this software, please cite the JOSS paper once available.

## License

MIT License (see `LICENSE`).
