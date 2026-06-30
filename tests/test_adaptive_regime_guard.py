import numpy as np

from solve_nivp.integrations import SDIRK2
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection
from solve_nivp.adaptive_integrator import AdaptiveStepping


class _HooklessProjection:
    """Functional projection that is NOT a ``Projection`` subclass, so it
    lacks the optional ``regime_snapshot`` / ``regime_changed_mask`` hooks."""

    def __init__(self):
        self._inner = IdentityProjection()

    def project(self, current_state, candidate, **kwargs):
        return self._inner.project(current_state, candidate, **kwargs)

    def tangent_cone(self, candidate, current_state, **kwargs):
        return self._inner.tangent_cone(candidate, current_state, **kwargs)


def _rhs(t, y):
    return -y


def test_active_set_filter_tolerates_projection_without_regime_hooks():
    solver = ImplicitEquationSolver(method='semismooth_newton',
                                    proj=_HooklessProjection())
    integ = SDIRK2(solver=solver)
    stepper = AdaptiveStepping(integrator=integ, atol=1e-6, rtol=1e-3,
                               active_set_filter=True)

    out = stepper._step_embedded(_rhs, 0.0, np.array([1.0, 2.0]), 1e-2)

    assert out is not None
