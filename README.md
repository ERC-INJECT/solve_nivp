# Solve_IVP_NS

A solver package for implicit ODEs with nonsmooth projections.

## Installation

```bash
pip install Solve_IVP_NS
# or from source (editable):
pip install -e .[test]
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

## Dependencies
- numpy, scipy
- optional: numba (for certain accelerated paths). You can disable projector numba usage by passing `use_numba=False` to the projection constructor.

## Contributing
See CONTRIBUTING.md
