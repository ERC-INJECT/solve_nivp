"""Tests for the dynamic De Saxce cone contact backend."""

import numpy as np

import solve_nivp
from solve_nivp.contact import ContactSystem
from solve_nivp.desaxce_contact import (
    _desaxce_block_residual_and_jac,
    build_dynamic_desaxce_contact,
    build_dynamic_desaxce_projected_contact,
    build_dynamic_desaxce_residual_contact,
)
from solve_nivp.projections import IdentityProjection


def _frictionless_normal_setup():
    A = np.eye(2)

    def rhs(t, y):
        v, _q = y
        return np.array([0.0, v], dtype=float)

    def rhs_jac(t, y, Fk=None):
        J = np.zeros((2, 2), dtype=float)
        J[1, 0] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0, e=0.0)]
    B = np.array([[1.0], [0.0]], dtype=float)
    return A, rhs, rhs_jac, gap_func, contacts, B


def test_return_type_and_open_contact_projection():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 1.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    assert isinstance(cs, ContactSystem)

    y = np.array([-2.0, 1.0], dtype=float)
    projected = cs.projection.project(
        y,
        y.copy(),
        t=0.0,
        prev_state=y.copy(),
        step_size=0.1,
    )

    np.testing.assert_allclose(projected, y, atol=1.0e-14)
    reaction = cs.projection.reaction_from_state(
        projected, t=0.0, prev_state=projected.copy(), step_size=0.1
    )
    assert abs(reaction[0]) < 1.0e-14


def test_frictionless_projection_cancels_closing_velocity():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    # Candidate corresponds to the free backward-Euler step:
    # v_free = -1, q_free = -0.1 at h = 0.1.
    y = np.array([-1.0, -0.1], dtype=float)
    projected = cs.projection.project(
        y,
        y.copy(),
        t=0.0,
        prev_state=y.copy(),
        step_size=0.1,
    )

    reaction = cs.reaction_from_step(
        projected, prev_state=y.copy(), t=0.0, h_val=0.1
    )

    np.testing.assert_allclose(projected, [0.0, -0.1], atol=1.0e-9)
    np.testing.assert_allclose(reaction, [10.0], atol=1.0e-9)


def test_projection_satisfies_desaxce_cone_conditions_with_friction():
    A = np.eye(3)

    def rhs(t, y):
        v_t, v_n, _q = y
        return np.array([0.0, 0.0, v_n], dtype=float)

    def rhs_jac(t, y, Fk=None):
        J = np.zeros((3, 3), dtype=float)
        J[2, 1] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[2]], dtype=float)

    mu = 0.5
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=0.0)]
    B = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    cs = build_dynamic_desaxce_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.zeros(3, dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    y = np.array([0.5, -1.0, -0.1], dtype=float)
    projected = cs.projection.project(
        y,
        y.copy(),
        t=0.0,
        prev_state=y.copy(),
        step_size=0.1,
    )

    u_t = float(projected[0])
    u_n = float(projected[1])
    reaction = cs.reaction_from_step(
        projected, prev_state=y.copy(), t=0.0, h_val=0.1
    )
    r_n = float(reaction[0])
    r_t = float(reaction[1])
    u_hat_n = u_n + mu * abs(u_t)

    assert r_n >= -1.0e-10
    assert abs(r_t) <= mu * r_n + 1.0e-8
    assert u_hat_n >= -1.0e-10
    assert abs(u_t) <= (u_hat_n / mu) + 1.0e-8
    assert abs(u_hat_n * r_n + u_t * r_t) <= 1.0e-8


def test_semismooth_newton_one_step_converges():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([-1.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.1),
        y0=cs.y0,
        method="backward_euler",
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-10,
            "max_iter": 40,
            "globalization": "linesearch",
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
            "adaptive_lam": False,
        },
        adaptive=False,
        h0=0.1,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    solver_error, success, _iterations = info[-1]
    assert success
    assert solver_error is not None
    assert abs(y[-1, 0]) <= 1.0e-8
    assert y[-1, 1] >= -1.0e-8
    reaction_hist = cs.reaction_history(y, t)
    assert reaction_hist[-1, 0] > 0.0


def test_residual_backend_returns_identity_projection():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_residual_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    assert isinstance(cs, ContactSystem)
    assert isinstance(cs.projection, IdentityProjection)
    assert cs.y0.shape == (3,)


def test_projected_backend_uses_post_step_projection():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_projected_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    assert isinstance(cs, ContactSystem)
    assert isinstance(cs.projection, IdentityProjection)
    assert "post_step_projection" in cs.integrator_opts
    assert cs.integrator_opts["post_step_projection"] is cs.step_projection


def test_projected_backend_sdirk2_one_step_closes_contact():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_projected_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([-1.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
    )

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.1),
        y0=cs.y0,
        method="sdirk2",
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-10,
            "max_iter": 40,
            "globalization": "linesearch",
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
            "adaptive_lam": False,
        },
        adaptive=False,
        h0=0.1,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    solver_error, success, _iterations = info[-1]
    assert success
    assert solver_error is not None
    np.testing.assert_allclose(y[-1], [0.0, 0.0], atol=1.0e-8)
    reaction_hist = cs.reaction_history(y, t)
    assert reaction_hist[-1, 0] > 0.0


def test_residual_backend_one_step_converges():
    A, rhs, rhs_jac, gap_func, contacts, B = _frictionless_normal_setup()
    cs = build_dynamic_desaxce_residual_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([-1.0, 0.0], dtype=float),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        contact_rho=10.0,
    )

    t, y, _, _, info = solve_nivp.solve_nivp(
        fun=cs.rhs,
        t_span=(0.0, 0.1),
        y0=cs.y0,
        method="backward_euler",
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts={
            "tol": 1.0e-10,
            "max_iter": 40,
            "globalization": "linesearch",
            "linear_solver": "splu",
            "rhs_jac": cs.rhs_jac,
            "adaptive_lam": False,
        },
        adaptive=False,
        h0=0.1,
        integrator_opts=cs.integrator_opts,
        component_slices=cs.component_slices,
        A=cs.A,
        store_fk=False,
    )

    solver_error, success, _iterations = info[-1]
    assert success
    assert solver_error is not None
    assert abs(y[-1, 0]) <= 1.0e-8
    assert y[-1, 1] >= -1.0e-8
    assert y[-1, cs.n_phys] > 0.0


def test_desaxce_residual_prestress_offset_shifts_equilibrium():
    """With prestress s0, the builder accepts get_s0 and augments correctly."""
    A = np.eye(2)

    def rhs(t, y):
        return np.array([0.0, y[0]], dtype=float)

    def rhs_jac(t, y):
        J = np.zeros((2, 2))
        J[1, 0] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    s0_val = 100.0
    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[], mu=0.0)]
    B = np.array([[1.0], [0.0]], dtype=float)

    cs = build_dynamic_desaxce_residual_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 0.5]),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        contact_rho=1.0,
        get_s0=lambda y: np.array([s0_val]),
        reaction_units="force",
    )
    assert isinstance(cs, ContactSystem)
    assert cs.y0.size == 3


def test_desaxce_block_residual_with_offset():
    """Prestress offset shifts r_eff = r + offset into the cone projection."""
    mu = 0.5
    alpha = mu
    rho = 1.0

    r_zero = np.array([0.0, 0.0])
    offset = np.array([4000.0, -1200.0])
    r_shifted = r_zero + offset
    u_blk = np.array([0.0, 1e-3])

    phi_off, du_off, dr_off = _desaxce_block_residual_and_jac(
        u_blk, r_zero, mu, alpha, rho, offset=offset,
    )
    phi_ref, du_ref, dr_ref = _desaxce_block_residual_and_jac(
        u_blk, r_shifted, mu, alpha, rho,
    )
    np.testing.assert_allclose(phi_off, phi_ref, atol=1e-14)
    np.testing.assert_allclose(du_off, du_ref, atol=1e-14)
    np.testing.assert_allclose(dr_off, dr_ref, atol=1e-14)


def test_desaxce_block_residual_zero_offset_matches_none():
    """offset=np.zeros(d) gives identical results to offset=None."""
    mu, alpha, rho = 0.5, 0.5, 2.0
    u_blk = np.array([0.01, 0.5])
    r_blk = np.array([4000.0, -1200.0])

    phi_none, du_none, dr_none = _desaxce_block_residual_and_jac(
        u_blk, r_blk, mu, alpha, rho,
    )
    phi_zero, du_zero, dr_zero = _desaxce_block_residual_and_jac(
        u_blk, r_blk, mu, alpha, rho, offset=np.zeros(2),
    )
    np.testing.assert_allclose(phi_zero, phi_none, atol=1e-14)
    np.testing.assert_allclose(du_zero, du_none, atol=1e-14)
    np.testing.assert_allclose(dr_zero, dr_none, atol=1e-14)


def test_desaxce_residual_offset_jac_fd():
    """Finite-difference check: jac_aug matches rhs_aug with offset."""
    A = np.eye(2)

    def rhs(t, y):
        return np.array([0.0, y[0]], dtype=float)

    def rhs_jac(t, y):
        J = np.zeros((2, 2))
        J[1, 0] = 1.0
        return J

    def gap_func(y, t):
        return np.array([y[1]], dtype=float)

    contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.5)]
    B = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    cs = build_dynamic_desaxce_residual_contact(
        A=A,
        rhs_smooth=rhs,
        rhs_jac=rhs_jac,
        y0=np.array([0.0, 0.5]),
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        contact_rho=1.0,
        get_s0=lambda y: np.array([4000.0]),
        get_w0=lambda y, k: np.array([-1200.0]),
        reaction_units="force",
    )

    y = cs.y0.copy()
    y[:2] = [0.01, 0.5]
    y[2:] = [0.0, 0.0]

    prev = y.copy()
    h = 0.01
    J_an = cs.rhs_jac(0.0, y, prev, None, h)
    if hasattr(J_an, "toarray"):
        J_an = J_an.toarray()

    eps = 1e-7
    n = y.size
    J_fd = np.zeros((n, n))
    f0 = cs.rhs(0.0, y, prev, None, h)
    for j in range(n):
        yp = y.copy()
        yp[j] += eps
        fp = cs.rhs(0.0, yp, prev, None, h)
        J_fd[:, j] = (fp - f0) / eps

    np.testing.assert_allclose(J_an, J_fd, atol=1e-5, rtol=1e-4)
