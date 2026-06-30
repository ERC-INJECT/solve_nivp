import numpy as np
import pytest

from solve_nivp import ODESystem, ODESolver


def _rhs(t, y):
    return -y


def _system():
    return ODESystem(fun=_rhs, y0=np.array([1.0]), method='backward_euler',
                     adaptive=False)


def test_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        ODESolver(_system(), t_span=(0.0, 1.0), h=0.0)
    with pytest.raises(ValueError):
        ODESolver(_system(), t_span=(0.0, 1.0), h=-0.1)


def test_rejects_non_increasing_t_span():
    with pytest.raises(ValueError):
        ODESolver(_system(), t_span=(1.0, 0.0), h=0.1)
    with pytest.raises(ValueError):
        ODESolver(_system(), t_span=(0.5, 0.5), h=0.1)


def test_accepts_valid_arguments():
    solver = ODESolver(_system(), t_span=(0.0, 1.0), h=0.1)
    assert solver.tf > solver.t0
    assert solver.h_initial > 0.0
