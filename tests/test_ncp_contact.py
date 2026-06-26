"""Tests for the full-state NCP contact backend."""

import numpy as np
import pytest

import solve_nivp
from solve_nivp.contact import ContactSystem
from solve_nivp.alart_curnier_contact import _project_ball_and_jac
from solve_nivp.ncp_contact import (
    _contact_block_residual_and_jac_2d,
    _friction_compliance_and_jac,
    _normal_ncp_residual_and_jac,
    build_dynamic_ncp_contact,
    build_ncp_contact,
)
from solve_nivp.projections import IdentityProjection


def _bouncing_ball_setup(mu=0.3):
    mass = 1.0
    gravity = np.array([0.0, -9.81])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        v = y[0:2]
        return np.concatenate([mass * gravity, v])

    def gap_func(y, t):
        return np.array([y[3]])

    y0 = np.array([2.0, 0.0, 0.0, 1.0])
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=0.0)]
    return A, rhs, y0, contacts, gap_func


def _reference_contact_block_2d(
    gap,
    u_blk,
    r_blk,
    mu,
    normal_ncp_type,
    friction_ncp_type,
    normal_scale,
    friction_scale,
    friction_law,
    *,
    tie_tol=1.0e-14,
):
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)
    f_blk = np.zeros(2, dtype=float)
    df_dgap = np.zeros(2, dtype=float)
    df_du = np.zeros((2, 2), dtype=float)
    df_dr = np.zeros((2, 2), dtype=float)

    phi_n, dphi_dgap, dphi_drn = _normal_ncp_residual_and_jac(
        gap, r_blk[0], normal_ncp_type, normal_scale, tie_tol=tie_tol
    )
    f_blk[0] = phi_n
    df_dgap[0] = dphi_dgap
    df_dr[0, 0] = dphi_drn

    mu_lambda_n = float(mu * r_blk[0])
    if mu_lambda_n <= tie_tol:
        f_blk[1] = r_blk[1]
        df_dr[1, 1] = 1.0
        return f_blk, df_dgap, df_du, df_dr

    if friction_law == "natural_map":
        y_vec = np.array([r_blk[1] - float(friction_scale) * u_blk[1]])
        proj, dproj_ddelta, dproj_dy = _project_ball_and_jac(
            y_vec, mu_lambda_n, tie_tol=tie_tol
        )
        f_blk[1] = proj[0] - r_blk[1]
        df_du[1, 1] = -float(friction_scale) * dproj_dy[0, 0]
        df_dr[1, 1] = dproj_dy[0, 0] - 1.0
        df_dr[1, 0] = float(mu) * dproj_ddelta[0]
        return f_blk, df_dgap, df_du, df_dr

    u_t = float(u_blk[1])
    r_t = float(r_blk[1])
    speed = abs(u_t)
    r_t_norm = abs(r_t)
    cone_gap = mu_lambda_n - r_t_norm
    W, dW_dspeed, dW_dgap, dW_dmulambda = _friction_compliance_and_jac(
        speed, cone_gap, mu_lambda_n, friction_ncp_type, friction_scale,
        tie_tol=tie_tol,
    )

    f_blk[1] = u_t + W * r_t
    if speed > tie_tol:
        dspeed_du = u_t / speed
    elif r_t_norm > tie_tol:
        dspeed_du = -r_t / r_t_norm
    else:
        dspeed_du = 0.0

    drnorm_dr = r_t / r_t_norm if r_t_norm > tie_tol else 0.0
    dW_du = dW_dspeed * dspeed_du
    dW_dr_t = -dW_dgap * drnorm_dr
    dW_dr_n = (dW_dmulambda + dW_dgap) * float(mu)

    df_du[1, 1] = 1.0 + r_t * dW_du
    df_dr[1, 1] = W + r_t * dW_dr_t
    df_dr[1, 0] = r_t * dW_dr_n
    return f_blk, df_dgap, df_du, df_dr


@pytest.mark.parametrize("friction_law", ["compliance", "natural_map"])
@pytest.mark.parametrize(
    "gap,u_blk,r_blk,mu",
    [
        (-0.2, np.array([0.0, 0.4]), np.array([0.7, -0.2]), 0.6),
        (0.1, np.array([0.0, 0.0]), np.array([0.5, 0.3]), 0.5),
        (-0.1, np.array([0.0, 0.2]), np.array([0.0, 0.1]), 0.6),
    ],
)
def test_contact_block_2d_fast_path_matches_reference(
    friction_law, gap, u_blk, r_blk, mu
):
    got = _contact_block_residual_and_jac_2d(
        gap,
        u_blk,
        r_blk,
        mu,
        "fischer_burmeister",
        "fischer_burmeister",
        1.7,
        0.8,
        friction_law,
    )
    expected = _reference_contact_block_2d(
        gap,
        u_blk,
        r_blk,
        mu,
        "fischer_burmeister",
        "fischer_burmeister",
        1.7,
        0.8,
        friction_law,
    )

    for got_arr, expected_arr in zip(got, expected):
        np.testing.assert_allclose(got_arr, expected_arr, rtol=1.0e-13, atol=1.0e-13)


def test_return_type_and_projection():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup()
    cs = build_ncp_contact(A, rhs, y0, contacts, gap)
    assert isinstance(cs, ContactSystem)
    assert isinstance(cs.projection, IdentityProjection)
    assert cs.y0.shape == (6,)


def test_open_contact_enforces_zero_reaction_rows():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup()
    cs = build_ncp_contact(A, rhs, y0, contacts, gap)
    y = np.array([1.5, -0.5, 0.0, 1.0, 5.0, -3.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    np.testing.assert_allclose(out[4:], [-5.0, 3.0], atol=1.0e-14)

    J = cs.rhs_jac(0.0, y, prev, h).toarray()
    np.testing.assert_allclose(J[4:, :4], 0.0, atol=1.0e-14)
    np.testing.assert_allclose(J[4:, 4:], -np.eye(2), atol=1.0e-14)


def test_minimum_map_normal_row_matches_expected_value():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_ncp_contact(A, rhs, y0, contacts, gap, ncp_type="minimum_map")

    y = np.array([0.0, 0.0, 0.0, -0.2, 0.0, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    # phi_n = min(gap, r_n) = min(-0.2, 0) = -0.2, rhs stores -phi_n.
    assert abs(out[4] - 0.2) < 1.0e-14


def test_fischer_burmeister_normal_row_matches_expected_value():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_ncp_contact(
        A,
        rhs,
        y0,
        contacts,
        gap,
        ncp_type="fischer_burmeister",
        normal_r=2.0,
    )

    y = np.array([0.0, 0.0, 0.0, -0.3, 0.2, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    phi = -0.3 + 2.0 * 0.2 - np.sqrt((-0.3) ** 2 + (2.0 * 0.2) ** 2)
    assert abs(out[4] - (-phi)) < 1.0e-14


def test_minimum_map_friction_row_uses_compliance_formula():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.5)
    cs = build_ncp_contact(A, rhs, y0, contacts, gap, ncp_type="minimum_map")

    y = np.array([0.0, 0.0, 0.0, -1.0, 1.0, 0.8])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    # speed = 0, cone_gap = mu*r_n - |r_t| = 0.5 - 0.8 = -0.3
    # W = (0 - (-0.3)) / 0.5 = 0.6, so phi_t = W * r_t = 0.48.
    assert abs(out[5] + 0.48) < 1.0e-14


def test_dynamic_wrapper_returns_force_like_contact_system():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.2)
    cs = build_dynamic_ncp_contact(
        A=A,
        rhs_smooth=rhs,
        y0=y0,
        contacts=contacts,
        gap_func=gap,
    )
    assert isinstance(cs, ContactSystem)
    assert isinstance(cs.projection, IdentityProjection)

    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.5])
    prev = y.copy()
    out1 = cs.rhs(0.0, y, prev, 1.0e-1)
    out2 = cs.rhs(0.0, y, prev, 1.0e-4)
    np.testing.assert_allclose(out1[:4], out2[:4], atol=1.0e-14)


def test_backward_euler_open_contact_step_converges_minimum_map():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_ncp_contact(A, rhs, y0, contacts, gap, ncp_type="minimum_map")

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.02),
        y0=cs.y0,
        method="backward_euler",
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-11,
            "max_iter": 40,
            "globalization": "linesearch",
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
        },
        adaptive=False,
        h0=0.02,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    solver_error, success, iterations = info[-1]
    assert success
    assert iterations <= 40
    assert solver_error is not None
    assert np.linalg.norm(y[-1][cs.n_phys:]) < 1.0e-8


@pytest.mark.parametrize(
    "method,max_iter",
    [
        ("backward_euler", 40),
        ("trapezoidal", 40),
        ("theta", 40),
        ("composite", 40),
        ("sdirk2", 50),
        ("embedded_betr", 40),
    ],
)
def test_dynamic_wrapper_runs_with_core_integrators_fischer_burmeister(method, max_iter):
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_dynamic_ncp_contact(
        A=A,
        rhs_smooth=rhs,
        y0=y0,
        contacts=contacts,
        gap_func=gap,
        ncp_type="fischer_burmeister",
    )

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.02),
        y0=cs.y0,
        method=method,
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-11,
            "max_iter": max_iter,
            "globalization": "linesearch",
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
        },
        adaptive=False,
        h0=0.02,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    solver_error, success, iterations = info[-1]
    assert success
    assert iterations <= max_iter
    assert solver_error is not None
    assert np.all(np.isfinite(y[-1]))
