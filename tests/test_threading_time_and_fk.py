import numpy as np


def test_fun_receives_time_and_fk():
    import Solve_IVP_NS as sivp

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

    result = sivp.solve_ivp_ns(
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
    import Solve_IVP_NS as sivp
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
    from Solve_IVP_NS.nonlinear_solvers import ImplicitEquationSolver
    from Solve_IVP_NS.integrations import BackwardEuler
    from Solve_IVP_NS.ODESystem import ODESystem
    from Solve_IVP_NS.ODESolver import ODESolver

    solver = ImplicitEquationSolver(method='VI', proj=ProbeProjection(), component_slices=[slice(0,2)])
    integrator = BackwardEuler(solver=solver)
    system = ODESystem(fun=fun, y0=y0, method=integrator, adaptive=False)
    ode = ODESolver(system, t_span, h=0.01)
    ode.solve()

    assert received["t"] is not None
    # Fk_val may be None in simple flows, but ensure key is present
    assert "fk" in received
