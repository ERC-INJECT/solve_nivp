"""Tests for the full-state Alart-Curnier contact benchmark backend."""

import numpy as np
import pytest

import solve_nivp
from solve_nivp.alart_curnier_contact import (
    build_alart_curnier_contact,
    build_dynamic_alart_curnier_contact,
)
from solve_nivp.contact import ContactSystem
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
    cs = build_alart_curnier_contact(A, rhs, y0, contacts, gap)
    assert isinstance(cs, ContactSystem)
    assert isinstance(cs.projection, IdentityProjection)
    assert cs.y0.shape == (6,)


def test_open_contact_enforces_zero_reaction_rows():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup()
    cs = build_alart_curnier_contact(A, rhs, y0, contacts, gap)
    y = np.array([1.5, -0.5, 0.0, 1.0, 5.0, -3.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    np.testing.assert_allclose(out[4:], [-5.0, 3.0], atol=1e-14)

    J = cs.rhs_jac(0.0, y, prev, h).toarray()
    np.testing.assert_allclose(J[4:, :4], 0.0, atol=1e-14)
    np.testing.assert_allclose(J[4:, 4:], -np.eye(2), atol=1e-14)


def test_active_contact_rows_match_alart_curnier_formula():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.3)
    cs = build_alart_curnier_contact(A, rhs, y0, contacts, gap)

    # Closed contact, zero reaction, closing normal velocity.
    y = np.array([2.0, -1.5, 0.0, 0.0, 0.0, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    # f_N = Proj_R+(r_N - rho_N u_N) - r_N = Proj_R+(1.5) - 0 = 1.5
    # rhs contact row stores -f_AC.
    assert abs(out[4] - (-1.5)) < 1e-14
    # Tangential radius is mu * r_N = 0, so the projected tangential term is 0.
    assert abs(out[5]) < 1e-14


def test_rate_form_uses_prev_state_in_contact_velocity():
    A = np.eye(2)

    def rhs_zero(t, y):
        return np.zeros_like(y)

    def gap_closed(y, t):
        return np.array([-1.0])

    y0 = np.zeros(2)
    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.5, e=0.0)]

    cs = build_alart_curnier_contact(
        A=A,
        rhs_smooth=rhs_zero,
        y0=y0,
        contacts=contacts,
        gap_func=gap_closed,
        C_extract=np.eye(2),
        D_extract=np.eye(2),
        rate_form=True,
    )

    # u = (y - y_prev) / h = [-2, 1]
    y = np.array([-0.2, 0.1, 0.0, 0.0])
    prev = np.array([0.0, 0.0, 0.0, 0.0])
    h = 0.1
    out = cs.rhs(0.0, y, prev, h)
    assert abs(out[2] - (-2.0)) < 1e-14
    assert abs(out[3]) < 1e-14


def test_backward_euler_open_contact_step_converges():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup()
    cs = build_alart_curnier_contact(A, rhs, y0, contacts, gap)

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.02),
        y0=cs.y0,
        method="backward_euler",
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-12,
            "max_iter": 30,
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
    assert iterations <= 30
    assert solver_error is not None
    assert np.linalg.norm(y[-1][cs.n_phys:]) < 1.0e-10


def test_active_contact_supports_soc_style_constant_offsets():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.5)
    cs = build_alart_curnier_contact(
        A,
        rhs,
        y0,
        contacts,
        gap,
        get_s0=lambda y: 0.0,
        get_w0=lambda y, k: np.array([0.8]),
    )

    # Closed contact, positive normal reaction so the tangential ball has radius 0.5.
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    # Tangential residual uses r_eff_T = r_T + w0 = 0.8.
    # Proj_{B(0, 0.5)}(0.8) - 0.8 = 0.5 - 0.8 = -0.3, and rhs stores -f_AC.
    assert abs(out[5] - 0.3) < 1.0e-14


def test_tangential_damping_regularizes_tangential_block():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.5)
    cs = build_alart_curnier_contact(
        A,
        rhs,
        y0,
        contacts,
        gap,
        reaction_units="force",
        tangential_damping=0.25,
    )

    # Closed contact with positive normal force and nonzero tangential velocity.
    y = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    # Here r_T,fric = r_T - c u_T = -0.5 and y = r_T,fric - rho_t u_T = -2.5.
    # The ball projection radius is mu * r_N = 0.5, so the tangential
    # residual becomes Proj(-2.5) - (-0.5) = -0.5 + 0.5 = 0.
    assert abs(out[5]) < 1.0e-14


def test_state_dependent_tangential_offset_jacobian_is_included():
    A = np.eye(2)

    def rhs_zero(t, y):
        return np.zeros_like(y)

    def gap_closed(y, t):
        return np.array([-1.0])

    y0 = np.zeros(2)
    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.0, e=0.0)]

    def get_w0(y, k):
        return np.array([2.0 * y[0]])

    def get_dw0_dz(y, k):
        out = np.zeros((1, 4))
        out[0, 0] = 2.0
        return out

    cs = build_alart_curnier_contact(
        A=A,
        rhs_smooth=rhs_zero,
        y0=y0,
        contacts=contacts,
        gap_func=gap_closed,
        get_w0=get_w0,
        get_dw0_dz=get_dw0_dz,
    )

    y = np.array([3.0, 0.0, 0.0, 0.0])
    prev = y.copy()
    h = 1.0e-2

    out = cs.rhs(0.0, y, prev, h)
    assert abs(out[3] - 6.0) < 1.0e-14

    J = cs.rhs_jac(0.0, y, prev, h).toarray()
    np.testing.assert_allclose(J[3, 0], 2.0, atol=1.0e-14)
    np.testing.assert_allclose(J[3, 3], 1.0, atol=1.0e-14)


def test_time_dependent_mu_callable_is_supported():
    A, rhs, y0, _, gap = _bouncing_ball_setup(mu=0.0)
    contacts = [
        dict(
            vel_normal_idx=1,
            vel_tangential_idx=[0],
            mu=lambda y, t: 0.25 + 0.5 * float(t),
            e=0.0,
        )
    ]
    cs = build_alart_curnier_contact(A, rhs, y0, contacts, gap)

    # At t=1, mu=0.75 and with r_N=1 the tangential ball has radius 0.75.
    y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    prev = y.copy()
    h = 1.0e-2
    out = cs.rhs(1.0, y, prev, h)
    assert abs(out[5] - 0.25) < 1.0e-14


def test_force_reaction_units_are_not_scaled_by_step_size():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_alart_curnier_contact(
        A,
        rhs,
        y0,
        contacts,
        gap,
        reaction_units="force",
    )

    y = np.array([0.0, 0.0, 0.0, 0.0, 2.0, -3.0])
    prev = y.copy()

    out_h1 = cs.rhs(0.0, y, prev, 1.0e-1)
    out_h2 = cs.rhs(0.0, y, prev, 1.0e-4)

    np.testing.assert_allclose(out_h1[:4], out_h2[:4], atol=1.0e-14)
    # B couples [r_N, r_T] onto physical rows [v_x, v_y, x, y] as [r_T, r_N, 0, 0].
    np.testing.assert_allclose(out_h1[:4], np.array([-3.0, -7.81, 0.0, 0.0]), atol=1.0e-14)


def test_dynamic_wrapper_returns_force_like_contact_system():
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.2)
    cs = build_dynamic_alart_curnier_contact(
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


@pytest.mark.parametrize(
    "method,max_iter",
    [
        ("backward_euler", 30),
        ("trapezoidal", 30),
        ("theta", 30),
        ("composite", 30),
        ("sdirk2", 40),
        ("embedded_betr", 30),
    ],
)
def test_dynamic_wrapper_runs_with_core_integrators(method, max_iter):
    A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.0)
    cs = build_dynamic_alart_curnier_contact(
        A=A,
        rhs_smooth=rhs,
        y0=y0,
        contacts=contacts,
        gap_func=gap,
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
            "tol": 1.0e-12,
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
    assert np.linalg.norm(y[-1][cs.n_phys:]) < 1.0e-10
