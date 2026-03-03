"""Tests for the C_extract / rate_form generalisation of build_impulse_contact.

Three scenarios:
1. Identity C_extract reproduces the existing point-mass result.
2. Non-trivial C_extract on a toy 2-DOF-per-node FEM-like problem.
3. rate_form=True with displacement unknowns.
4. Auto-generated gap and B from C_extract.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import solve_nivp
from solve_nivp.contact import build_impulse_contact


# ──────────────────────────────────────────────────────────────────────
# Shared bouncing-ball setup (velocity unknowns, identity extraction)
# ──────────────────────────────────────────────────────────────────────
mass = 1.0
gravity = np.array([0.0, -9.81])
A_phys = np.diag([mass, mass, 1.0, 1.0])
mu = 0.3


def rhs_ball(t, y):
    return np.concatenate([mass * gravity, y[0:2]])


y0_ball = np.array([2.0, 0.0, 0.0, 1.0])
slices_ball = [slice(0, 2), slice(2, 4)]


def gap_ball(y, t):
    return np.array([y[3]])


contact_spec = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=0.0)]


# ──────────────────────────────────────────────────────────────────────
# 1. Identity C_extract = existing behaviour
# ──────────────────────────────────────────────────────────────────────
class TestIdentityCExtract:
    """C_extract = I should reproduce the identity-extraction path."""

    def test_identity_dense(self):
        """Dense identity C gives same result as no C."""
        # Reference: no C_extract
        cs_ref = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
            theta=1.0, component_slices=slices_ball,
        )
        t_ref, y_ref, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_ref.rhs, t_span=(0, 1.0), y0=cs_ref.y0, A=cs_ref.A,
            method='backward_euler', projection=cs_ref.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_ref.component_slices,
            integrator_opts=cs_ref.integrator_opts,
            adaptive=False, h0=0.005,
        )

        # With identity C_extract — vel_normal/tangential stay the same
        # because they refer to rows of C = physical DOF indices.
        C_id = np.eye(4)
        cs_c = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
            theta=1.0, component_slices=slices_ball,
            C_extract=C_id,
        )
        t_c, y_c, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_c.rhs, t_span=(0, 1.0), y0=cs_c.y0, A=cs_c.A,
            method='backward_euler', projection=cs_c.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_c.component_slices,
            integrator_opts=cs_c.integrator_opts,
            adaptive=False, h0=0.005,
        )

        assert len(t_ref) == len(t_c)
        np.testing.assert_allclose(y_ref[:, :4], y_c[:, :4], atol=1e-10)

    def test_identity_sparse(self):
        """Sparse identity C gives same result."""
        cs_ref = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
            theta=1.0, component_slices=slices_ball,
        )
        t_ref, y_ref, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_ref.rhs, t_span=(0, 0.5), y0=cs_ref.y0, A=cs_ref.A,
            method='backward_euler', projection=cs_ref.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_ref.component_slices,
            integrator_opts=cs_ref.integrator_opts,
            adaptive=False, h0=0.01,
        )

        C_sp = sp.eye(4, format='csr')
        cs_c = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
            theta=1.0, component_slices=slices_ball,
            C_extract=C_sp,
        )
        t_c, y_c, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_c.rhs, t_span=(0, 0.5), y0=cs_c.y0, A=cs_c.A,
            method='backward_euler', projection=cs_c.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_c.component_slices,
            integrator_opts=cs_c.integrator_opts,
            adaptive=False, h0=0.01,
        )

        np.testing.assert_allclose(y_ref[:, :4], y_c[:, :4], atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# 2. Non-trivial C_extract (FEM-like extraction)
# ──────────────────────────────────────────────────────────────────────
class TestFEMLikeExtraction:
    """Non-identity C_extract on a toy problem.

    Physical state: (v_x, v_y, q_x, q_y)  — same bouncing ball.
    C_extract picks only the velocity DOFs and scales them
    (simulating a non-trivial FEM evaluation operator):

        C = [[0, 1, 0, 0],    ← ''normal velocity''  (v_y)
             [1, 0, 0, 0]]    ← ''tangential velocity''  (v_x)

    Contacts: vel_normal_idx=0 (row 0 of C = v_y),
              vel_tangential_idx=[1] (row 1 of C = v_x).
    B = C^T.

    This is the same physics as the identity case, so results must match.
    """

    def test_permuted_extraction(self):
        C_perm = np.array([
            [0, 1, 0, 0],  # row 0 → v_y (normal)
            [1, 0, 0, 0],  # row 1 → v_x (tangential)
        ], dtype=float)

        contact_perm = [dict(
            vel_normal_idx=0,     # row 0 of C = v_y
            vel_tangential_idx=[1],  # row 1 of C = v_x
            mu=mu, e=0.0,
        )]

        cs_ref = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
            theta=1.0, component_slices=slices_ball,
        )
        t_ref, y_ref, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_ref.rhs, t_span=(0, 1.0), y0=cs_ref.y0, A=cs_ref.A,
            method='backward_euler', projection=cs_ref.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_ref.component_slices,
            integrator_opts=cs_ref.integrator_opts,
            adaptive=False, h0=0.005,
        )

        cs_c = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_perm, gap_ball,
            theta=1.0, component_slices=slices_ball,
            C_extract=C_perm,
        )
        t_c, y_c, *_ = solve_nivp.solve_ivp_ns(
            fun=cs_c.rhs, t_span=(0, 1.0), y0=cs_c.y0, A=cs_c.A,
            method='backward_euler', projection=cs_c.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs_c.component_slices,
            integrator_opts=cs_c.integrator_opts,
            adaptive=False, h0=0.005,
        )

        np.testing.assert_allclose(y_ref[:, :4], y_c[:, :4], atol=1e-10)

    def test_auto_gap_from_C(self):
        """gap_func=None should auto-generate gap from C normal rows."""
        C_perm = np.array([
            [0, 0, 0, 1],  # row 0 → q_y (gap = position)
            [1, 0, 0, 0],  # row 1 → v_x  (tangential vel)
        ], dtype=float)

        # For gap: C[normal_idx=0, :] @ y = y[3] = q_y ✓
        # For De Saxcé velocity: same C is used, row 0 → q_y, row 1 → v_x.
        # But De Saxcé needs velocities, not positions, for the reaction rows.
        # This is a subtlety — for the bouncing ball the state includes
        # both velocities and positions.  The gap uses position (q_y),
        # but De Saxcé needs velocity (v_y).  So C_extract and D_extract
        # must differ when the extraction for gap vs velocity is different.

        # Use separate D for velocity extraction:
        D_vel = np.array([
            [0, 1, 0, 0],  # row 0 → v_y (normal velocity)
            [1, 0, 0, 0],  # row 1 → v_x (tangential velocity)
        ], dtype=float)

        contact_auto = [dict(
            vel_normal_idx=0,        # row 0 of C/D
            vel_tangential_idx=[1],  # row 1 of C/D
            mu=mu, e=0.0,
        )]

        # gap_func=None → auto-generated from C_extract row 0
        cs_auto = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_auto,
            gap_func=None,  # auto from C normal rows
            theta=1.0, component_slices=slices_ball,
            C_extract=C_perm,
            D_extract=D_vel,
        )

        # Verify gap detects q_y correctly
        y_test = np.array([2.0, 0.0, 0.0, -0.01, 0.0, 0.0])
        gap = cs_auto.projection.gap_func(y_test, 0.0)
        assert gap[0] == pytest.approx(-0.01, abs=1e-15)

        y_test2 = np.array([2.0, 0.0, 0.0, 0.5, 0.0, 0.0])
        gap2 = cs_auto.projection.gap_func(y_test2, 0.0)
        assert gap2[0] == pytest.approx(0.5, abs=1e-15)

    def test_auto_B_equals_CT(self):
        """Auto-generated B should equal C_extract^T."""
        C = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        contact_c = [dict(vel_normal_idx=0, vel_tangential_idx=[1],
                          mu=mu, e=0.0)]
        cs = build_impulse_contact(
            A_phys, rhs_ball, y0_ball, contact_c, gap_ball,
            theta=1.0, component_slices=slices_ball,
            C_extract=C,
        )
        # B should be C^T (reordered: rows [0, 1] of C → columns [0, 1] of B)
        np.testing.assert_array_equal(cs.B, C.T)


# ──────────────────────────────────────────────────────────────────────
# 3. rate_form=True (displacement unknowns)
# ──────────────────────────────────────────────────────────────────────
class TestRateForm:
    """Displacement-based formulation with rate_form=True.

    Physics: a 1D spring-dashpot with a floor contact.
      State: (u, q)  where u = displacement, q = auxiliary.
      M du/dt = -k*u + gravity  → same as bouncing ball but u = position.

    Actually, let's test with a simpler model: free-fall with floor.
      State: (q,) — single displacement DOF.
      m*q̈ = -mg   →  first-order form:  (m, 1) * (v, q)' = (mg, v)
    But rate_form is for when the STATE is displacement.

    Simplest displacement-only test:
      State: (q_x, q_y)  — two displacement DOFs.
      M (q - q_prev)/h = v_ext  ← external velocity.
      M is diagonal [1, 1].
      Contact: normal = q_y, tangential = q_x.
      C_extract = I (2×2).  rate_form → v_c = (q - q_prev)/h.

    But this is not a standard ODE...  Let's just verify the mechanics:
    the RHS should produce contact velocities via C@(y-y_prev)/h.
    """

    def test_rate_form_rhs_produces_contact_velocity(self):
        """Verify that rate_form computes C @ (y-y_prev)/h for De Saxcé."""
        n = 2
        A_disp = np.eye(n)

        def rhs_disp(t, y):
            return np.array([0.0, -9.81])

        y0_disp = np.array([0.0, 1.0])
        C_id = np.eye(n)
        contact_disp = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                             mu=0.3, e=0.0)]

        def gap_disp(y, t):
            return np.array([y[1]])

        cs = build_impulse_contact(
            A_disp, rhs_disp, y0_disp, contact_disp, gap_disp,
            theta=1.0, component_slices=[slice(0, 2)],
            C_extract=C_id, rate_form=True,
        )

        # Simulate one step: y_prev = [0, 1], y_curr = [0.1, 0.9], h = 0.1
        h = 0.1
        y_curr = np.array([0.1, 0.9, 0.0, 0.0])   # augmented
        y_prev = np.array([0.0, 1.0, 0.0, 0.0])    # augmented

        out = cs.rhs(0.0, y_curr, y_prev, h)

        # Reaction rows (indices 2, 3): should be -û where
        # v_contact = C @ (y - y_prev)/h = [0.1/0.1, -0.1/0.1] = [1, -1]
        # v_N = v_contact[1] = -1 (normal vel idx = 1)
        # v_T = v_contact[0] = 1  (tangential)
        # û_N = v_N + μ|v_T| + 0 = -1 + 0.3*1 = -0.7
        # û_T = v_T = 1.0
        # outreact = [-û_N, -û_T] = [0.7, -1.0]
        assert out[2] == pytest.approx(0.7, abs=1e-12)   # -(-0.7)
        assert out[3] == pytest.approx(-1.0, abs=1e-12)   # -(1.0)

    def test_rate_form_rejects_nonzero_e(self):
        """rate_form=True with e>0 should raise ValueError."""
        contact_bad = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                            mu=0.3, e=0.5)]
        with pytest.raises(ValueError, match="rate_form.*c_coeff"):
            build_impulse_contact(
                np.eye(2), lambda t, y: y, np.zeros(2),
                contact_bad, lambda y, t: np.array([y[1]]),
                theta=1.0, C_extract=np.eye(2), rate_form=True,
            )


# ──────────────────────────────────────────────────────────────────────
# 4. Validation / error handling
# ──────────────────────────────────────────────────────────────────────
class TestValidation:
    def test_c_extract_wrong_cols(self):
        """C_extract with wrong number of columns → ValueError."""
        C_bad = np.eye(3)  # 3 cols, but n_phys = 4
        with pytest.raises(ValueError, match="columns"):
            build_impulse_contact(
                A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
                C_extract=C_bad,
            )

    def test_gap_func_required_without_c(self):
        """gap_func=None without C_extract → ValueError."""
        with pytest.raises(ValueError, match="gap_func"):
            build_impulse_contact(
                A_phys, rhs_ball, y0_ball, contact_spec, gap_func=None,
            )

    def test_incremental_coupling_rejected_with_c(self):
        """C_extract + incremental_coupling → NotImplementedError."""
        with pytest.raises(NotImplementedError, match="C_extract"):
            build_impulse_contact(
                A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
                C_extract=np.eye(4), incremental_coupling=True,
            )

    def test_fremond_rejected_with_c(self):
        """C_extract + fremond_contact → NotImplementedError."""
        with pytest.raises(NotImplementedError, match="C_extract"):
            build_impulse_contact(
                A_phys, rhs_ball, y0_ball, contact_spec, gap_ball,
                C_extract=np.eye(4), fremond_contact=True,
            )
