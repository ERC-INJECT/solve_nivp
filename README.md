# solve_nivp

[![DOI](https://joss.theoj.org/papers/10.21105/joss.09775/status.svg)](https://doi.org/10.21105/joss.09775)

A Python library for time integration of **nonsmooth** ODE/DAE systems: models
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

- **Implicit integrators.** Backward Euler, Trapezoidal, theta-method, a
  composite TR-BE scheme (Bathe-type, second-order), an embedded BE-TR error
  estimator, and SDIRK2 on this development branch.

- **Adaptive step-size control** with Richardson or embedded error estimates.

- **Optional RL add-on.** Exposes the time integrator as a Gym-style
  environment for learning adaptive step-size policies (TD3 / TQC via Stable
  Baselines 3).

The library is organised around three interchangeable components: projection,
nonlinear solver, and integrator, so that swapping algorithms during
experimentation is straightforward. Linear-algebra routines operate on dense or
sparse arrays in the SciPy ecosystem.

## Installation

From PyPI:

```bash
pip install solve_nivp
```

Latest development pre-release from PyPI:

```bash
pip install --pre solve_nivp
```

Optional extras:

```bash
pip install solve_nivp[test]   # includes pytest
pip install solve_nivp[rl]     # RL experiments (gymnasium, stable-baselines3)
pip install solve_nivp[docs]   # Sphinx documentation build
pip install solve_nivp[all]    # branch-local heavy extras (PETSc + RL stack)
```

### Developer install (from source)

```bash
git clone https://github.com/ERC-INJECT/solve_nivp.git
cd solve_nivp
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[test]
```

Conda environments are also provided for HPC or other user-space installs:

```bash
# lean environment for core development, tests, docs, and notebooks
conda env create -f environment.yml

# broader environment for PETSc, RL, and the heavier notebook stack
conda env create -f environment-full.yml
```

### HPC / user-space install

If you are working on an HPC system and want the environment to live in your
project or scratch space rather than a shared default location, create it with
an explicit prefix:

```bash
export ENV_ROOT=/path/to/your/project-or-scratch-space/.conda
export CONDA_PKGS_DIRS="$ENV_ROOT/pkgs"

conda env create -p "$ENV_ROOT/envs/solve_nivp" -f environment.yml
# or, for the broader stack:
# conda env create -p "$ENV_ROOT/envs/solve_nivp-full" -f environment-full.yml

conda activate "$ENV_ROOT/envs/solve_nivp"
pip install -e .
```

The sliding-block and prestressed-fault examples also depend on a separate
`poroelasticity` repository that is not bundled as a package dependency here.
If you need those examples, install that repo into the same environment as
well, for example:

```bash
pip install -e /path/to/Poroelasticity
```

## Quickstart

```python
import numpy as np
from solve_nivp import solve_nivp

# simple smooth rhs: y' = -y
rhs = lambda t, y: -y

t_span = (0.0, 1.0)
y0 = np.array([1.0])

# identity projection, VI solver via composite integrator
sol = solve_nivp(
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

In these notebooks the reward signal, observation mapping, and policy
configuration are defined directly in notebook cells. The core `solve_nivp`
package only supplies the nonsmooth solvers and the `AdaptiveStepperEnv`
wrapper; all RL-specific choices are intentionally left to the user so they can
adapt the workflow to their own applications.

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
