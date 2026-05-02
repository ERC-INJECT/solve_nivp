"""Focused regressions for the unilateral Moreau projection path."""

import numpy as np

import solve_nivp
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import GeneralMoreauVIProjection, IdentityProjection


def test_vi_identity_does_not_false_converge_with_tiny_rho():
    solver = ImplicitEquationSolver(
        method="VI",
        proj=IdentityProjection(),
        component_slices=[slice(0, 1)],
        tol=1.0e-6,
        rho0=1.0e-12,
        max_iter=3,
        adaptive_lam=False,
    )

    y, F, err, success, iterations = solver.solve(
        lambda yy: np.ones_like(yy, dtype=float),
        np.array([0.0], dtype=float),
    )

    assert not success
    assert iterations == 3
    assert err > 0.1
    np.testing.assert_allclose(F, [1.0], atol=0.0)
    np.testing.assert_allclose(y, [-3.0e-12], atol=1.0e-15)


def test_general_moreau_projection_forwards_step_size_to_g_apply():
    seen = {}

    def gap(t, y):
        return np.array([y[1]], dtype=float)

    def u_map(t, y):
        return np.array([y[0]], dtype=float)

    def J_u(t, y):
        return np.array([[1.0, 0.0]], dtype=float)

    def G_apply(t, y, lam_full, step_size=None):
        seen["step_size"] = step_size
        lam = float(lam_full[0])
        return np.array([lam, 0.5 * float(step_size) * lam], dtype=float)

    proj = GeneralMoreauVIProjection(
        gap=gap,
        u_map=u_map,
        J_u=J_u,
        G_apply=G_apply,
        e=0.0,
        gap_tol=1.0e-12,
        lcp_tol=1.0e-14,
    )

    state = np.array([-1.0, 0.0], dtype=float)
    projected = proj.project(
        state,
        state.copy(),
        rhok=1.0,
        t=0.0,
        Fk_val=np.zeros_like(state),
        prev_state=state.copy(),
        step_size=1.0e-2,
    )

    assert seen["step_size"] == 1.0e-2
    np.testing.assert_allclose(projected, [0.0, 5.0e-3], atol=1.0e-12)


def test_semismooth_newton_crosses_first_bounce_step():
    mass = 1.0
    gravity = 9.81
    restitution = 0.8
    h = 1.0e-3
    # One fixed step immediately before the first impact in the notebook case.
    y_pre = np.array([-6.337879999999952, 0.0036328800000074418], dtype=float)

    def rhs(t, y):
        return np.array([-gravity, y[0]], dtype=float)

    def rhs_jacobian(t, y, Fk=None):
        J = np.zeros((2, 2), dtype=float)
        J[1, 0] = 1.0
        return J

    def gap(t, y):
        return np.array([y[1]], dtype=float)

    def u_map(t, y):
        return np.array([y[0]], dtype=float)

    def J_u(t, y):
        return np.array([[1.0, 0.0]], dtype=float)

    def G_apply(t, y, lam_full, step_size=None):
        lam = float(lam_full[0])
        dv = lam / mass
        ds = 0.0 if step_size is None else 0.5 * float(step_size) * dv
        return np.array([dv, ds], dtype=float)

    proj = GeneralMoreauVIProjection(
        gap=gap,
        u_map=u_map,
        J_u=J_u,
        G_apply=G_apply,
        e=restitution,
        gap_tol=1.0e-10,
        lcp_tol=1.0e-10,
        tc_tol=1.0e-10,
    )

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=rhs,
        t_span=(0.0, h),
        y0=y_pre,
        method="trapezoidal",
        projection=proj,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-9,
            "max_iter": 80,
            "rhs_jac": rhs_jacobian,
            "linear_solver": "splu",
            "globalization": "linesearch",
            "linear_tol_strategy": "eisenstat",
            "adaptive_lam": False,
        },
        adaptive=False,
        h0=h,
        component_slices=[slice(0, 1), slice(1, 2)],
        A=None,
        store_fk=False,
    )

    solver_error, success, iterations = info[-1]
    assert success
    assert solver_error < 1.0e-10
    assert iterations <= 4
    assert y[-1, 0] > 0.0
    assert y[-1, 1] > y_pre[1]
