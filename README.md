# solve_nivp

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09775/status.svg)](https://doi.org/10.21105/joss.09775)

A Python library for time integration of **nonsmooth** ODE/DAE systems—models
with abrupt changes such as impacts, switching, or inequality constraints.
Such models arise in frictional contact mechanics, piecewise and switching
behaviour in circuits, sliding-mode control, and discontinuous rules in
finance and energy markets. Classical solvers, which assume smoothness, often
require regularisation or very small steps due to the inherent stiffness
of these models. **solve_nivp** builds nonsmooth rules directly into the
implicit time-stepping scheme, enabling users to encode constraints and advance
the state robustly.

## Key features

- **Projection-based constraint encoding.** Users express set-valued or
  nonsmooth relations as projections onto convex sets (Coulomb friction cone,
  sign / normal cone, second-order cone, algebraic constraints). Custom
  projections need only implement `project()` and an optional `tangent_cone()`.

- **Nonlinear solvers for nonsmooth problems.** A semismooth Newton method with
  Armijo line search and a variational-inequality (VI) fixed-point iteration,
  both with standard tolerances, safeguards, and iteration diagnostics.

- **Implicit integrators.** Backward Euler, Trapezoidal, θ-method, a composite
  TR–BE scheme (Bathe-type, second-order), and an embedded BE–TR error estimator.

- **Adaptive step-size control** with Richardson extrapolation.

- **Optional RL add-on.** Exposes the time integrator as a Gym-style
  environment for learning adaptive step-size policies (TD3 / TQC via Stable
  Baselines 3).

The library is organised around three interchangeable components—projection,
nonlinear solver, and integrator—so that swapping algorithms during
experimentation is straightforward. Linear-algebra routines operate on dense or
sparse arrays in the SciPy ecosystem.

## Installation

From PyPI:

```bash
pip install solve_nivp
```

Optional extras:

```bash
pip install solve_nivp[test]   # includes pytest
pip install solve_nivp[rl]     # RL experiments (gymnasium, stable-baselines3)
pip install solve_nivp[docs]   # Sphinx documentation build
```

### Developer install (from source)

```bash
git clone https://github.com/ERC-INJECT/solve_nivp.git
cd solve_nivp
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[test]
```

## Quickstart

```python
import numpy as np
from solve_nivp import solve_ivp_ns

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

If you use this software, please cite the JOSS paper:

> Riley, D. M. & Stefanou, I. (2025). solve_nivp: A Python toolkit for integrating
> nonsmooth dynamical systems. *Journal of Open Source Software*.
> [doi:10.21105/joss.09775](https://doi.org/10.21105/joss.09775)

BibTeX:

```bibtex
@article{Riley2025solve_nivp,
  author  = {Riley, David Michael and Stefanou, Ioannis},
  title   = {solve\_nivp: A Python toolkit for integrating nonsmooth dynamical systems},
  journal = {Journal of Open Source Software},
  year    = {2025},
  doi     = {10.21105/joss.09775},
  url     = {https://doi.org/10.21105/joss.09775}
}
```

## License

MIT License (see `LICENSE`).
