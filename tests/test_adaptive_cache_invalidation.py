import types

import numpy as np

from solve_nivp.integrations import IntegrationMethod
from solve_nivp.adaptive_integrator import AdaptiveStepping


def _rhs(t, y):
    return -y


class _HalfStepFailIntegrator(IntegrationMethod):
    """Full step of size ``full_h`` succeeds; any other (half) step fails."""

    def __init__(self, full_h):
        self.full_h = full_h
        self.solver = types.SimpleNamespace(
            _lu="STALE",
            _lu_shape=(2, 2),
            _J_cross_call="STALE",
            _petsc_needs_matrix_update=False,
            proj=None,
        )

    def step(self, fun, t, y, h):
        ok = abs(h - self.full_h) < 1e-15
        return y + h, np.zeros_like(y), np.zeros_like(y), ok, 1


def test_half_step_failure_invalidates_solver_caches():
    h = 1e-2
    integ = _HalfStepFailIntegrator(full_h=h)
    stepper = AdaptiveStepping(integrator=integ, atol=1e-6, rtol=1e-3)

    out = stepper._step_richardson(_rhs, 0.0, np.array([1.0, 2.0]), h)

    # The half step failed, so the step is not accepted ...
    success = out[4]
    assert success is False
    # ... and the stale factorisation must have been dropped.
    assert integ.solver._lu is None
    assert integ.solver._lu_shape is None
    assert integ.solver._J_cross_call is None
    assert integ.solver._petsc_needs_matrix_update is True
