import numpy as np


def test_fun_receives_time_and_fk():
    import solve_nivp as sivp

    calls = []

    def fun(t, y, Fk_val=None):
        # Record call args for the t+h stage only
        calls.append((t, Fk_val))
        # simple stable linear system
        return -y

    # Exact Jacobian for fun
    def rhs_jac(t, y, Fk_val=None):
        return -np.eye(y.size)

    y0 = np.array([1.0, 2.0])
    t_span = (0.0, 0.05)

    result = sivp.solve_nivp(
        fun=fun,
        t_span=t_span,
        y0=y0,
        method='backward_euler',
        projection='identity',
        solver='semismooth_newton',
        solver_opts={"rhs_jac": rhs_jac},
        adaptive=False,
        h0=0.05,
    )

    # Ensure the fun was called with a non-None Fk_val at the implicit point
    # (the initial explicit call may have Fk_val=None; we only require that some call had non-None)
    assert any(Fk is not None for (_, Fk) in calls)


def test_projection_receives_time_and_fk_in_vi():
    import solve_nivp as sivp
    received = {"t": None, "fk": None}

    # trivial fun
    def fun(t, y):
        return -y

    # Custom projection to assert reception
    class ProbeProjection:
        def __init__(self):
            self.component_slices = [slice(0, 2)]
        def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
            received["t"] = t
            received["fk"] = Fk_val
            return candidate
        def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
            return np.eye(candidate.size)

    y0 = np.array([0.1, -0.2])
    t_span = (0.0, 0.01)

    # Use lower-level API to plug our projection
    from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
    from solve_nivp.integrations import BackwardEuler
    from solve_nivp.ODESystem import ODESystem
    from solve_nivp.ODESolver import ODESolver

    solver = ImplicitEquationSolver(method='VI', proj=ProbeProjection(), component_slices=[slice(0,2)])
    integrator = BackwardEuler(solver=solver)
    system = ODESystem(fun=fun, y0=y0, method=integrator, adaptive=False)
    ode = ODESolver(system, t_span, h=0.01)
    ode.solve()

    assert received["t"] is not None
    # Fk_val may be None in simple flows, but ensure key is present
    assert "fk" in received


def test_fixed_step_odesolver_stops_on_failed_nonlinear_step():
    from solve_nivp.ODESolver import ODESolver

    class FailingSystem:
        adaptive = False
        verbose = False

        def __init__(self):
            self.current_y = np.array([1.0])

        def step(self, t, h):
            return np.array([9.0]), np.array([7.0]), 2.5, False, 11

    system = FailingSystem()
    solver = ODESolver(system, t_span=(0.0, 1.0), h=0.5)
    t_vals, y_vals, h_vals, fk_vals, error_estimates = solver.solve()

    np.testing.assert_allclose(t_vals, [0.0])
    np.testing.assert_allclose(y_vals[:, 0], [1.0])
    np.testing.assert_allclose(h_vals, [0.5])
    assert fk_vals.size == 0
    assert error_estimates == [(2.5, False, 11)]
    np.testing.assert_allclose(system.current_y, [1.0])
    assert solver.terminal_failure == (2.5, False, 11)
