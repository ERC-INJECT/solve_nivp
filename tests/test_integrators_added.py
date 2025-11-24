import numpy as np

from solve_nivp.integrations import BackwardEuler, Trapezoidal, CompositeMethod
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection
from solve_nivp.adaptive_integrator import AdaptiveStepping


def _rhs(t, y):
    return -y


def test_integrator_single_step_contracts():
    y0 = np.array([1.0, 2.0])
    solver = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-12)

    for Integrator in (BackwardEuler, Trapezoidal, CompositeMethod):
        integ = Integrator(solver=solver)
        out = integ.step(_rhs, 0.0, y0.copy(), 0.1)
        assert isinstance(out, tuple)
        assert len(out) == 5
        y_new, fk, err, ok, it = out
        assert y_new.shape == y0.shape
        assert np.all(np.isfinite(y_new))


def test_adaptive_controller_skip_indices_and_no_stall():
    y0 = np.array([1.0, 2.0, 3.0, 4.0])
    solver = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-10)
    integ = CompositeMethod(solver=solver)
    ctrl = AdaptiveStepping(integrator=integ, atol=1e-6, rtol=1e-3, h0=1e-2, h_min=1e-6, component_slices=[slice(0, 2), slice(2, 4)], skip_error_indices=[1])

    t = 0.0
    y = y0.copy()
    h = 1e-2

    # take a few steps; ensure h doesn't get stuck at minimum and skip indices respected (smoke)
    for _ in range(10):
        y, fk, h_new, E, ok, solver_err, it = ctrl.step(_rhs, t, y, h)
        assert h_new >= ctrl.h_min
        t += min(h, 1e-3)
        y = y
        h = h_new


def test_api_imports_and_solve_shapes():
    from solve_nivp import solve_ivp_ns, SignProjection, CoulombProjection, IdentityProjection

    t, y, h, fk, info = solve_ivp_ns(
        fun=_rhs,
        t_span=(0.0, 0.05),
        y0=np.array([1.0]),
        method='composite',
        projection='identity',
        solver='VI',
    )
    assert t.ndim == 1 and y.ndim == 2 and h.ndim == 1
    assert y.shape[1] == 1
