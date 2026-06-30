import numpy as np

from solve_nivp import ODESystem, ODESolver


def _rhs(t, y):
    return -y


def test_step_fixed_does_not_mutate_current_y():
    """``ODESystem.step_fixed`` must not commit state; the driver owns the
    accepted-state update. A successful step therefore leaves ``current_y``
    untouched, and a failed step cannot corrupt it either."""
    y0 = np.array([1.0, 2.0])
    system = ODESystem(fun=_rhs, y0=y0, method='backward_euler', adaptive=False)

    y_new, _fk, _err, success, _it = system.step_fixed(0.0, 0.1)

    assert success
    assert not np.allclose(y_new, y0)
    np.testing.assert_array_equal(system.current_y, y0)


def test_injected_method_with_A_clears_use_identity():
    """Passing ``A`` to ``ODESystem`` alongside a prebuilt method (constructed
    with ``A=None``) must make the method actually use ``A``, not identity."""
    from solve_nivp.integrations import BackwardEuler
    from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
    from solve_nivp.projections import IdentityProjection

    method = BackwardEuler(
        solver=ImplicitEquationSolver(method='semismooth_newton',
                                      proj=IdentityProjection()),
        A=None,
    )
    assert method.use_identity is True

    A = np.array([[2.0, 0.0], [0.0, 3.0]])
    system = ODESystem(fun=_rhs, y0=np.array([1.0, 1.0]),
                       method=method, A=A)

    assert system.method.use_identity is False
    np.testing.assert_array_equal(system.method._get_A(2), A)


def test_solver_stores_scalar_residual():
    """A method returning a Python-float residual must not crash the driver
    (scalars have no ``.copy()``)."""
    from solve_nivp.integrations import IntegrationMethod

    class _ScalarFkMethod(IntegrationMethod):
        def step(self, fun, t, y, h):
            return y + h, 0.0, 0.0, True, 1

    system = ODESystem(fun=_rhs, y0=np.array([1.0]),
                       method=_ScalarFkMethod(), adaptive=False)
    solver = ODESolver(system, t_span=(0.0, 0.2), h=0.1)
    _t, _y, _h, fk, _info = solver.solve()

    assert fk[-1] == 0.0


def test_driver_commits_accepted_state():
    """The driver advances state on accepted steps even though ``step_fixed``
    no longer mutates it."""
    y0 = np.array([1.0])
    system = ODESystem(fun=_rhs, y0=y0, method='backward_euler', adaptive=False)
    solver = ODESolver(system, t_span=(0.0, 0.5), h=0.05)
    t, y, _h, _fk, _info = solver.solve()

    assert t[-1] >= 0.5 - 1e-9
    assert y[-1, 0] < y0[0]
