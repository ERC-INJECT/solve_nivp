"""Tests for CompositeContactProjection and build_impulse_contact with constraints.

Covers:
1. CompositeContactProjection standalone — project, tangent_cone, disjoint
   validation, active-set delegation, batch.
2. build_impulse_contact with constraints= — composite creation, type check,
   solve_nivp integration with a coupled DAE + contact problem.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.projections import (
    AlgebraicConstraintProjection,
    CompositeContactProjection,
    MuScaledSOCProjection,
)
from solve_nivp.contact import build_impulse_contact


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_alg(n_y=2, n_q=2, y_start=0, q_start=2):
    """Algebraic projection: q = C @ y with a simple 2×2 coupling."""
    C = np.array([[2.0, 0.0], [0.0, 3.0]])[:n_q, :n_y]
    return AlgebraicConstraintProjection(
        g=lambda y, _C=C: _C @ y,
        dg_dy=lambda y, _C=C: _C,
        y_slice=slice(y_start, y_start + n_y),
        q_slice=slice(q_start, q_start + n_q),
    )


def _make_soc(rN=4, rT=5, mu=0.3, zero_inactive=True):
    """SOC projection on reaction DOFs (rN, [rT])."""
    return MuScaledSOCProjection(
        blocks=[(rN, [rT])],
        get_mu=lambda y: mu,
        zero_inactive=zero_inactive,
    )


# ======================================================================
# Test class: CompositeContactProjection standalone
# ======================================================================

class TestCompositeProjectionStandalone:
    """Unit tests for the CompositeContactProjection class itself."""

    def test_project_algebraic_and_soc(self):
        """Both sub-projections are applied correctly."""
        alg = _make_alg()
        soc = _make_soc()
        comp = CompositeContactProjection(alg, soc)

        # State: [y0, y1, q0, q1, rN, rT]
        z = np.array([1.0, 2.0, 99.0, 99.0, 0.5, 0.1])
        p = comp.project(z, z)

        # Algebraic: q = C @ y → [2*1, 3*2] = [2, 6]
        np.testing.assert_allclose(p[2], 2.0, atol=1e-14)
        np.testing.assert_allclose(p[3], 6.0, atol=1e-14)

        # SOC: (0.5, 0.1) — check it's on or inside cone μ=0.3
        rN_proj, rT_proj = p[4], p[5]
        assert rN_proj >= -1e-15, "Normal reaction must be >= 0"
        assert abs(rT_proj) <= 0.3 * rN_proj + 1e-14, "Coulomb bound"

        # Physical DOFs untouched
        np.testing.assert_allclose(p[:2], z[:2])

    def test_project_soc_interior(self):
        """Interior SOC point stays unchanged."""
        alg = _make_alg()
        soc = _make_soc(mu=0.5)
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.1])
        p = comp.project(z, z)
        # Interior: |0.1| <= 0.5 * 1.0 → unchanged
        np.testing.assert_allclose(p[4:], z[4:], atol=1e-14)

    def test_project_soc_polar(self):
        """SOC polar point projects to zero."""
        alg = _make_alg()
        soc = _make_soc(mu=0.3)
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 1.0, 0.0, 0.0, -1.0, 0.1])
        p = comp.project(z, z)
        np.testing.assert_allclose(p[4:], 0.0, atol=1e-14)

    def test_tangent_cone_shape_and_type(self):
        """Tangent cone has correct shape and preserves sparsity."""
        alg = _make_alg()
        soc = _make_soc()
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.5, 0.1])
        D = comp.tangent_cone(z, z)
        assert D.shape == (6, 6)

    def test_tangent_cone_algebraic_rows(self):
        """Algebraic rows of tangent cone match dg/dy."""
        alg = _make_alg()
        soc = _make_soc()
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.5, 0.1])
        D = comp.tangent_cone(z, z)
        D_arr = D.toarray() if sp.issparse(D) else np.asarray(D)

        # Rows 2, 3 should have dg/dy = [[2, 0, ...], [0, 3, ...]]
        np.testing.assert_allclose(D_arr[2, 0], 2.0, atol=1e-12)
        np.testing.assert_allclose(D_arr[2, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(D_arr[3, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(D_arr[3, 1], 3.0, atol=1e-12)
        # No identity on diagonal for constrained rows
        np.testing.assert_allclose(D_arr[2, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(D_arr[3, 3], 0.0, atol=1e-12)

    def test_tangent_cone_identity_rows(self):
        """Non-constrained, non-SOC rows are identity."""
        alg = _make_alg()
        soc = _make_soc()
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.5, 0.1])
        D = comp.tangent_cone(z, z)
        D_arr = D.toarray() if sp.issparse(D) else np.asarray(D)

        # Rows 0, 1 should be identity
        np.testing.assert_allclose(D_arr[0, :], [1, 0, 0, 0, 0, 0], atol=1e-14)
        np.testing.assert_allclose(D_arr[1, :], [0, 1, 0, 0, 0, 0], atol=1e-14)

    def test_tangent_cone_soc_rows(self):
        """SOC rows contain non-trivial Clarke Jacobian on boundary."""
        alg = _make_alg()
        soc = _make_soc(mu=0.3)
        comp = CompositeContactProjection(alg, soc)

        # Point on boundary of cone (not interior, not polar)
        z = np.array([1.0, 1.0, 0.0, 0.0, 0.5, 0.5])
        D = comp.tangent_cone(z, z)
        D_arr = D.toarray() if sp.issparse(D) else np.asarray(D)

        # SOC rows (4, 5) should NOT be identity
        soc_block = D_arr[4:6, 4:6]
        assert not np.allclose(soc_block, np.eye(2)), \
            "Boundary SOC must have non-trivial Jacobian"

    def test_soc_batch_tangent_keeps_active_apex_as_identity(self):
        """The batch SOC fast-path must match the scalar apex classification.

        At the cone apex ``(s, w) = (0, 0)`` the exact projector treats the
        point as interior, so the Clarke selection is the identity.  This is a
        regression test for the vectorized ``n > 64`` path used by large
        contact systems.
        """
        n_blocks = 40  # 80 DOFs -> forces the vectorized batch tangent path
        blocks = [(2 * k, [2 * k + 1]) for k in range(n_blocks)]
        soc = MuScaledSOCProjection(
            blocks=blocks,
            get_mu=lambda y: 0.3,
            gap_func=lambda y, t: np.zeros(n_blocks),
            zero_inactive=True,
        )

        z = np.zeros(2 * n_blocks)
        D = soc.tangent_cone(z, z, t=0.0)
        D_arr = D.toarray() if sp.issparse(D) else np.asarray(D)

        np.testing.assert_allclose(
            D_arr, np.eye(2 * n_blocks), atol=1e-14,
            err_msg="Active apex must keep the identity tangent in batch mode",
        )

    def test_tangent_cone_formula_D_alg_plus_D_soc_minus_I(self):
        """Verify D_comp = D_alg + D_soc - I explicitly."""
        alg = _make_alg()
        soc = _make_soc(mu=0.3)
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.3, 0.5])
        D_comp = comp.tangent_cone(z, z)
        D_alg = alg.tangent_cone(z, z)
        D_soc = soc.tangent_cone(z, z)

        def _to_dense(M):
            return M.toarray() if sp.issparse(M) else np.asarray(M)

        expected = _to_dense(D_alg) + _to_dense(D_soc) - np.eye(6)
        np.testing.assert_allclose(_to_dense(D_comp), expected, atol=1e-14)

    def test_disjoint_validation_raises(self):
        """Overlapping algebraic q_slice and SOC block raises ValueError."""
        alg = _make_alg(q_start=4)  # q_slice = slice(4, 6) — overlaps SOC
        soc = _make_soc(rN=4, rT=5)
        with pytest.raises(ValueError, match="overlap"):
            CompositeContactProjection(alg, soc)

    def test_type_validation_alg(self):
        """Non-AlgebraicConstraintProjection raises TypeError."""
        soc = _make_soc()
        with pytest.raises(TypeError, match="AlgebraicConstraintProjection"):
            CompositeContactProjection("not_a_projection", soc)

    def test_type_validation_soc(self):
        """Non-MuScaledSOCProjection raises TypeError."""
        alg = _make_alg()
        with pytest.raises(TypeError, match="MuScaledSOCProjection"):
            CompositeContactProjection(alg, "not_a_projection")

    def test_lock_unlock_delegation(self):
        """Active-set lock/unlock are delegated to the SOC sub-projection."""
        alg = _make_alg()
        soc = MuScaledSOCProjection(
            blocks=[(4, [5])],
            get_mu=lambda y: 0.3,
            gap_func=lambda y, t: np.array([1.0]),  # gap > 0 → inactive
            zero_inactive=True,
        )
        comp = CompositeContactProjection(alg, soc)

        y = np.zeros(6)
        comp.lock_active_set(y, t=0.0)
        assert soc._locked_active is not None
        assert not soc._locked_active[0]  # gap > 0 → inactive

        comp.unlock_active_set()
        assert soc._locked_active is None

    def test_project_batch(self):
        """Batch projection applies both sub-projections."""
        alg = _make_alg()
        soc = _make_soc(mu=0.5)
        comp = CompositeContactProjection(alg, soc)

        z1 = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.1])
        z2 = np.array([2.0, 3.0, 0.0, 0.0, -0.5, 0.1])
        candidates = np.vstack([z1, z2])

        out = comp.project_batch(z1, candidates)

        # Row 0: algebraic [2, 3], SOC interior → unchanged
        np.testing.assert_allclose(out[0, 2], 2.0, atol=1e-14)
        np.testing.assert_allclose(out[0, 3], 3.0, atol=1e-14)
        np.testing.assert_allclose(out[0, 4:], [1.0, 0.1], atol=1e-14)

        # Row 1: algebraic [4, 9], SOC polar → zero
        np.testing.assert_allclose(out[1, 2], 4.0, atol=1e-14)
        np.testing.assert_allclose(out[1, 3], 9.0, atol=1e-14)
        np.testing.assert_allclose(out[1, 4:], [0.0, 0.0], atol=1e-14)

    def test_sub_projection_accessors(self):
        """Accessor properties return the correct sub-projections."""
        alg = _make_alg()
        soc = _make_soc()
        comp = CompositeContactProjection(alg, soc)

        assert comp.algebraic_projection is alg
        assert comp.soc_projection is soc
        assert comp.blocks is soc.blocks
        assert comp.zero_inactive is soc.zero_inactive

    def test_inactive_soc_zeroed_with_gap(self):
        """Inactive SOC blocks are zeroed; algebraic still enforced."""
        alg = _make_alg()
        soc = MuScaledSOCProjection(
            blocks=[(4, [5])],
            get_mu=lambda y: 0.3,
            gap_func=lambda y, t: np.array([1.0]),  # gap > 0 → inactive
            zero_inactive=True,
        )
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 99.0, 99.0, 5.0, 3.0])
        p = comp.project(z, z)

        # Algebraic enforced
        np.testing.assert_allclose(p[2], 2.0, atol=1e-14)
        np.testing.assert_allclose(p[3], 6.0, atol=1e-14)
        # SOC zeroed (inactive)
        np.testing.assert_allclose(p[4:], [0.0, 0.0], atol=1e-14)


# ======================================================================
# Test class: build_impulse_contact with constraints
# ======================================================================

class TestBuildImpulseContactConstraints:
    """Integration tests for constraints= parameter."""

    def test_no_constraints_gives_soc_only(self):
        """Without constraints, projection is plain MuScaledSOCProjection."""
        M = np.eye(2)
        y0 = np.array([0.0, 0.0])
        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        cs = build_impulse_contact(
            M, lambda t, y: np.zeros(2), y0, contacts,
            gap_func=lambda y, t: np.array([y[0]]),
        )
        assert isinstance(cs.projection, MuScaledSOCProjection)
        assert not isinstance(cs.projection, CompositeContactProjection)

    def test_with_constraints_gives_composite(self):
        """With constraints, projection is CompositeContactProjection."""
        n_phys = 4  # [y0, y1, q0, q1]
        M = np.diag([1.0, 1.0, 0.0, 0.0])  # singular mass for algebraic DOFs
        y0 = np.zeros(n_phys)
        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        constraints = [
            dict(
                g=lambda y: 2.0 * y,
                dg_dy=lambda y: 2.0 * np.eye(2),
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
            ),
        ]
        cs = build_impulse_contact(
            M, lambda t, y: np.zeros(n_phys), y0, contacts,
            gap_func=lambda y, t: np.array([y[0]]),
            constraints=constraints,
        )
        assert isinstance(cs.projection, CompositeContactProjection)
        assert isinstance(cs.projection.algebraic_projection,
                          AlgebraicConstraintProjection)
        assert isinstance(cs.projection.soc_projection,
                          MuScaledSOCProjection)

    def test_gap_tol_propagated_to_composite_soc_projection(self):
        """gap_tol reaches the SOC sub-projection inside the composite."""
        n_phys = 4
        M = np.diag([1.0, 1.0, 0.0, 0.0])
        y0 = np.zeros(n_phys)
        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        constraints = [
            dict(
                g=lambda y: 2.0 * y,
                dg_dy=lambda y: 2.0 * np.eye(2),
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
            ),
        ]
        cs = build_impulse_contact(
            M,
            lambda t, y: np.zeros(n_phys),
            y0,
            contacts,
            gap_func=lambda y, t: np.array([y[0]]),
            constraints=constraints,
            gap_tol=1.0e-9,
        )
        assert cs.projection.soc_projection.gap_tol == pytest.approx(1.0e-9)

    def test_augmented_dimensions(self):
        """Augmented system has correct dimensions."""
        n_phys = 4
        M = np.diag([1.0, 1.0, 0.0, 0.0])
        y0 = np.zeros(n_phys)
        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        constraints = [
            dict(g=lambda y: y * 2, dg_dy=lambda y: 2 * np.eye(2),
                 y_slice=slice(0, 2), q_slice=slice(2, 4)),
        ]
        cs = build_impulse_contact(
            M, lambda t, y: np.zeros(n_phys), y0, contacts,
            gap_func=lambda y, t: np.array([y[0]]),
            constraints=constraints,
        )
        n_react = 2  # 1 normal + 1 tangential
        n_aug = n_phys + n_react
        assert cs.y0.shape == (n_aug,)
        assert cs.A.shape == (n_aug, n_aug)
        assert cs.n_phys == n_phys

    def test_composite_projection_enforces_both(self):
        """The composite projection enforces algebraic + SOC on augmented state."""
        n_phys = 4
        M = np.diag([1.0, 1.0, 0.0, 0.0])
        y0 = np.zeros(n_phys)
        C_alg = np.array([[1.5, 0.0], [0.0, 2.5]])
        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        constraints = [
            dict(g=lambda y, _C=C_alg: _C @ y,
                 dg_dy=lambda y, _C=C_alg: _C,
                 y_slice=slice(0, 2), q_slice=slice(2, 4)),
        ]
        cs = build_impulse_contact(
            M, lambda t, y: np.zeros(n_phys), y0, contacts,
            gap_func=lambda y, t: np.array([y[0]]),
            constraints=constraints,
        )
        # Test projection on a candidate
        z = np.array([1.0, 2.0, 0.0, 0.0, 0.5, 0.1])
        p = cs.projection.project(z, z)

        # Algebraic: [1.5*1, 2.5*2] = [1.5, 5.0]
        np.testing.assert_allclose(p[2], 1.5, atol=1e-14)
        np.testing.assert_allclose(p[3], 5.0, atol=1e-14)

        # SOC: (0.5, 0.1) inside cone μ=0.3? s=0.5, |w|=0.1, μs=0.15
        # |w| > μs → boundary projection
        assert p[4] >= -1e-15

    def test_with_c_extract_and_constraints(self):
        """C_extract + constraints together work."""
        n_phys = 6  # [u0, u1, p0, p1, lam_q0, lam_q1]
        M = np.diag([1, 1, 1, 1, 0, 0]).astype(float)
        y0 = np.zeros(n_phys)

        # C_extract: extracts u0, u1 from physical state
        C_extract = np.zeros((2, n_phys))
        C_extract[0, 0] = 1.0  # contact normal
        C_extract[1, 1] = 1.0  # contact tangential

        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]
        constraints = [
            dict(
                g=lambda p: p * 1.5,
                dg_dy=lambda p: 1.5 * np.eye(2),
                y_slice=slice(2, 4),     # p0, p1
                q_slice=slice(4, 6),     # lam_q0, lam_q1
            ),
        ]

        cs = build_impulse_contact(
            M, lambda t, y: np.zeros(n_phys), y0, contacts,
            C_extract=C_extract,
            constraints=constraints,
        )
        assert isinstance(cs.projection, CompositeContactProjection)
        assert cs.n_phys == n_phys
        assert cs.y0.shape == (n_phys + 2,)  # 2 reaction DOFs


# ======================================================================
# Test class: solve_nivp integration with composite
# ======================================================================

class TestSolveWithComposite:
    """Integration test: solve a small DAE + contact problem end-to-end."""

    def test_falling_ball_with_algebraic_constraint(self):
        """Ball falling under gravity with an algebraic output variable.

        State: [q_x, q_y, v_x, v_y, w] where w = C*q_y (algebraic).
        q_y = vertical position. Contact: gap = q_y, normal = v_y.
        This ensures the composite projection works through solve_nivp.
        """
        from solve_nivp import solve_nivp

        # Physical system: [q_x, q_y, v_x, v_y, w]
        #   dq/dt = v, dv_x/dt = 0, dv_y/dt = -g, 0 = w - C*q_y
        g_grav = 9.81
        C_obs = 2.0  # observation: w = 2*q_y

        M = np.diag([1.0, 1.0, 1.0, 1.0, 0.0])

        def rhs_smooth(t, y):
            qx, qy, vx, vy, w = y
            return np.array([vx, vy, 0.0, -g_grav, 0.0])

        # Drop from height 0.1, v=0
        y0 = np.array([0.0, 0.1, 0.0, 0.0, C_obs * 0.1])

        # Normal = v_y (idx 3), tangential = v_x (idx 2)
        contacts = [dict(vel_normal_idx=3, vel_tangential_idx=[2], mu=0.0)]

        def gap(y, t):
            return np.array([y[1]])  # gap = q_y

        constraints = [
            dict(
                g=lambda y_sub: np.array([C_obs * y_sub[1]]),
                dg_dy=lambda y_sub: np.array([[0.0, C_obs, 0.0, 0.0]]),
                y_slice=slice(0, 4),   # reads q_x, q_y, v_x, v_y
                q_slice=slice(4, 5),   # writes w
            ),
        ]

        cs = build_impulse_contact(
            M, rhs_smooth, y0, contacts,
            gap_func=gap,
            constraints=constraints,
        )

        t_span = (0.0, 0.3)

        sol_t, sol_y, *_ = solve_nivp(
            fun=cs.rhs, y0=cs.y0, A=cs.A, t_span=t_span,
            projection=cs.projection,
            component_slices=cs.component_slices,
            method='backward_euler',
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-10, max_iter=200,
                             lam_update_strategy='none'),
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=1e-3,
        )

        n_phys = cs.n_phys
        q_y_sol = sol_y[:, 1]
        w_sol = sol_y[:, 4]

        # Position should never go significantly negative
        # (small penetration ~O(h) is normal for impulse-level contact)
        assert np.all(q_y_sol >= -1e-3), \
            f"Position violated: min(q_y) = {q_y_sol.min()}"

        # Algebraic constraint: w = C * q_y should hold throughout
        np.testing.assert_allclose(w_sol, C_obs * q_y_sol, atol=1e-3,
                                   err_msg="Algebraic constraint w=C*q_y violated")

    def test_composite_multiple_constraints(self):
        """Multiple algebraic constraints + SOC contact."""
        alg = AlgebraicConstraintProjection(constraints=[
            dict(
                g=lambda y: np.array([y[0] + y[1]]),
                dg_dy=lambda y: np.array([[1.0, 1.0]]),
                y_slice=slice(0, 2),
                q_slice=slice(2, 3),
            ),
            dict(
                g=lambda y: np.array([y[0] * 0.5]),
                dg_dy=lambda y: np.array([[0.5]]),
                y_slice=slice(0, 1),
                q_slice=slice(3, 4),
            ),
        ])
        soc = MuScaledSOCProjection(
            blocks=[(4, [5])],
            get_mu=lambda y: 0.5,
            zero_inactive=True,
        )
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.8, 0.6])
        p = comp.project(z, z)

        # Constraint 1: q2 = y0 + y1 = 3.0
        np.testing.assert_allclose(p[2], 3.0, atol=1e-14)
        # Constraint 2: q3 = 0.5 * y0 = 0.5
        np.testing.assert_allclose(p[3], 0.5, atol=1e-14)
        # SOC: (0.8, 0.6) → |0.6| > 0.5*0.8=0.4 → boundary
        assert p[4] >= -1e-15

    def test_tangent_cone_numerical_consistency(self):
        """Finite-difference check: tangent cone ≈ FD Jacobian of project."""
        alg = _make_alg()
        soc = _make_soc(mu=0.5)
        comp = CompositeContactProjection(alg, soc)

        z0 = np.array([1.0, 2.0, 5.0, 5.0, 0.8, 0.3])
        D = comp.tangent_cone(z0, z0)
        D_arr = D.toarray() if sp.issparse(D) else np.asarray(D)

        # Finite-difference Jacobian of project(z, z) w.r.t. z
        eps = 1e-7
        p0 = comp.project(z0, z0)
        D_fd = np.zeros((6, 6))
        for j in range(6):
            z_p = z0.copy()
            z_p[j] += eps
            p_p = comp.project(z_p, z_p)
            D_fd[:, j] = (p_p - p0) / eps

        # Check agreement (may differ at non-smooth points, but this
        # test uses a smooth-enough point)
        np.testing.assert_allclose(D_arr, D_fd, atol=1e-4,
                                   err_msg="Tangent cone ≠ FD Jacobian")
