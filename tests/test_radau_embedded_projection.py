import numpy as np
import pytest

from solve_nivp.integrations import SDIRK2, RadauIIA
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


def _rhs(t, y):
    return -y


class _ShiftProjection:
    """Post-step projection that adds a fixed offset to the candidate."""

    def __init__(self, offset):
        self.offset = np.asarray(offset, dtype=float)

    def project(self, current_state, candidate, **kwargs):
        return candidate + self.offset


def _solver():
    return ImplicitEquationSolver(method='semismooth_newton',
                                  proj=IdentityProjection())


@pytest.mark.parametrize("Method", [SDIRK2, RadauIIA])
def test_embedded_error_reflects_post_step_projection(Method):
    y0 = np.array([1.0, 2.0])
    h = 0.05
    offset = np.array([0.1, -0.2])

    _y, _fk, err_base, ok_base, _it = Method(solver=_solver()).step(
        _rhs, 0.0, y0.copy(), h)
    assert ok_base

    proj_method = Method(solver=_solver(),
                         post_step_projection=_ShiftProjection(offset))
    _yp, _fkp, err_proj, ok_proj, _itp = proj_method.step(
        _rhs, 0.0, y0.copy(), h)
    assert ok_proj

    np.testing.assert_allclose(err_proj - err_base, offset, atol=1e-9)
