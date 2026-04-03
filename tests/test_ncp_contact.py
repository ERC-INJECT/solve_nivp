"""Tests for the full-state NCP contact backend."""

import numpy as np
import pytest

import solve_nivp
from solve_nivp.contact import ContactSystem
from solve_nivp.ncp_contact import build_dynamic_ncp_contact, build_ncp_contact
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
