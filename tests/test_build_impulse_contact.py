"""Tests for the build_impulse_contact helper."""

import numpy as np
import pytest

from solve_nivp.contact import build_impulse_contact, ContactSystem
from solve_nivp.projections import MuScaledSOCProjection


# =====================================================================
# Shared fixtures
# =====================================================================

def _bouncing_ball_setup(mu=0.3, e=0.0):
    """Return the standard bouncing ball physical system."""
    mass = 1.0
    gravity = np.array([0.0, -9.81])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        v = y[0:2]
        return np.concatenate([mass * gravity, v])

    def gap_func(y, t):
        return np.array([y[3]])

    y0 = np.array([2.0, 0.0, 0.0, 1.0])
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=e)]
    return A, rhs, y0, contacts, gap_func


# =====================================================================
# ContactSystem structure tests
# =====================================================================

class TestContactSystemStructure:
    """Verify dimensions, slices, and auto-generated B."""

    def test_return_type(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert isinstance(cs, ContactSystem)

    def test_augmented_dimensions(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.n_phys == 4
        assert cs.y0.shape == (6,)
        assert cs.A.shape == (6, 6)

    def test_augmented_y0_padded(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        np.testing.assert_array_equal(cs.y0[:4], y0)
        np.testing.assert_array_equal(cs.y0[4:], 0.0)

    def test_augmented_A_block_diag(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        # Physical block untouched
        np.testing.assert_array_equal(cs.A[:4, :4], A)
        # Reaction block = zero
        np.testing.assert_array_equal(cs.A[4:, :], 0.0)
        np.testing.assert_array_equal(cs.A[:, 4:], 0.0)

    def test_auto_B_generation(self):
        """Auto-generated B maps r_N → v_y (idx 1), r_T → v_x (idx 0)."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        B_expected = np.array([
            [0, 1],
            [1, 0],
            [0, 0],
            [0, 0],
        ], dtype=float)
        np.testing.assert_array_equal(cs.B, B_expected)

    def test_manual_B_accepted(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        B = np.array([[0, 1], [1, 0], [0, 0], [0, 0]], dtype=float)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap, B=B)
        np.testing.assert_array_equal(cs.B, B)

    def test_manual_B_wrong_shape_raises(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        B_wrong = np.eye(4)
        with pytest.raises(ValueError, match="B shape"):
            build_impulse_contact(A, rhs, y0, contacts, gap, B=B_wrong)

    def test_component_slices_extended(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 2), slice(2, 4)],
        )
        assert len(cs.component_slices) == 3
        assert cs.component_slices[-1] == slice(4, 6)

    def test_component_slices_default(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.component_slices == [slice(0, 4), slice(4, 6)]

    def test_integrator_opts(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.integrator_opts['pass_prev_state'] is True
        assert cs.integrator_opts['pass_step_size'] is True

    def test_projection_type(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert isinstance(cs.projection, MuScaledSOCProjection)


# =====================================================================
# Frémond coefficient tests
# =====================================================================

class TestFremondCoefficient:
    """Verify c = θ(1+e) - 1 computation via the RHS."""

    def test_c_zero_for_BE_e0(self):
        """BE (θ=1), e=0 → c=0 → no prev_state contribution."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.0)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap, theta=1.0)
        y_test = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.array([0.5, -2.0, 0.0, 0.5, 0.0, 0.0])
        # Call RHS with prev_state and h
        out = cs.rhs(0.0, y_test, prev, 0.001)
        # Reaction row for r_N: -(v_y + mu*|v_x| + c*v_y_prev)
        #   v_y=-1, mu=0.3, |v_x|=1, c=0, v_y_prev=-2
        #   = -(-1 + 0.3 + 0) = 0.7
        assert abs(out[4] - 0.7) < 1e-14

    def test_c_equals_e_for_BE(self):
        """BE (θ=1), e=0.5 → c=0.5 → includes restitution term."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.5)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap, theta=1.0)
        y_test = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.array([0.5, -2.0, 0.0, 0.5, 0.0, 0.0])
        out = cs.rhs(0.0, y_test, prev, 0.001)
        # c = 1*(1+0.5)-1 = 0.5
        # -(v_y + mu*|v_x| + c*v_y_prev) = -(-1 + 0.3 + 0.5*(-2)) = 1.7
        assert abs(out[4] - 1.7) < 1e-14

    def test_c_for_trapezoidal(self):
        """TR (θ=0.5), e=0 → c = 0.5*(1+0)-1 = -0.5."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.0)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap, theta=0.5)
        y_test = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.array([0.5, -2.0, 0.0, 0.5, 0.0, 0.0])
        out = cs.rhs(0.0, y_test, prev, 0.001)
        # c = -0.5
        # -(v_y + mu*|v_x| + c*v_y_prev) = -(-1 + 0.3 + (-0.5)*(-2)) = -0.3
        assert abs(out[4] - (-0.3)) < 1e-14


# =====================================================================
# RHS evaluation tests
# =====================================================================

class TestRHSEvaluation:
    """Verify the augmented RHS produces correct values."""

    def test_physical_rows_no_contact(self):
        """In free flight (r=0), physical rows = smooth RHS."""
        A, rhs_smooth, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap)
        y_test = np.array([2.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        out = cs.rhs(0.0, y_test, y_test, 0.001)
        f_phys = rhs_smooth(0.0, y_test[:4])
        np.testing.assert_allclose(out[:4], f_phys, atol=1e-14)

    def test_physical_rows_with_impulse(self):
        """Coupling: B @ r / h adds to physical rows."""
        A, rhs_smooth, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap)
        y_test = np.array([1.0, 0.0, 0.0, 0.0, 5.0, 3.0])
        h = 0.01
        out = cs.rhs(0.0, y_test, y_test, h)
        f_phys = rhs_smooth(0.0, y_test[:4])
        B = cs.B
        expected_phys = f_phys + B @ np.array([5.0, 3.0]) / h
        np.testing.assert_allclose(out[:4], expected_phys, atol=1e-12)

    def test_reaction_rows_desaxce(self):
        """Reaction rows: -û_N = -(v_y + μ|v_x|), -û_T = -v_x."""
        A, rhs_smooth, y0, contacts, gap = _bouncing_ball_setup(mu=0.3)
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap)
        y_test = np.array([2.0, -1.5, 0.0, 0.0, 0.0, 0.0])
        out = cs.rhs(0.0, y_test, y_test, 0.001)
        # -û_N = -(v_y + mu*|v_x|) = -(-1.5 + 0.3*2) = 0.9
        assert abs(out[4] - 0.9) < 1e-14
        # -û_T = -v_x = -2.0
        assert abs(out[5] - (-2.0)) < 1e-14

    def test_rhs_handles_no_prev_state(self):
        """When called without prev_state, c*v_N_prev term is zero."""
        A, rhs_smooth, y0, contacts, gap = _bouncing_ball_setup(e=0.5)
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap, theta=1.0)
        y_test = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        # Call with only (t, y) — no extra args
        out = cs.rhs(0.0, y_test)
        # c=0.5 but prev_state=None → v_N_prev=0
        # û_N = v_y + mu*|v_x| = -1 + 0.3 = -0.7
        assert abs(out[4] - 0.7) < 1e-14

    def test_rhs_accepts_fk_arg(self):
        """RHS must tolerate Fk array (passed by _get_bound_wrapper)."""
        A, rhs_smooth, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap)
        y_test = np.array([2.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        fk = np.zeros(6)  # Fk residual
        # Call with (t, y, prev, fk, h) — Fk in the middle
        out = cs.rhs(0.0, y_test, y_test, fk, 0.001)
        # Should still work; h extracted as last scalar
        f_phys = rhs_smooth(0.0, y_test[:4])
        np.testing.assert_allclose(out[:4], f_phys, atol=1e-14)


# =====================================================================
# Multi-contact tests
# =====================================================================

class TestMultiContact:
    """Support for multiple simultaneous contacts."""

    def test_two_contacts_dimensions(self):
        """Two contacts → 4 reaction DOFs appended."""
        n_phys = 6
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)

        def rhs(t, y):
            return np.zeros(n_phys)

        def gap(y, t):
            return np.array([y[2], y[5]])

        contacts = [
            dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3, e=0.0),
            dict(vel_normal_idx=4, vel_tangential_idx=[3], mu=0.5, e=0.0),
        ]
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.n_phys == 6
        assert cs.y0.shape == (10,)
        assert cs.A.shape == (10, 10)
        assert cs.B.shape == (6, 4)

    def test_two_contacts_B_structure(self):
        """Auto-B has correct column mapping for two contacts."""
        n_phys = 6
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)

        def rhs(t, y):
            return np.zeros(n_phys)

        def gap(y, t):
            return np.array([y[2], y[5]])

        contacts = [
            dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3),
            dict(vel_normal_idx=4, vel_tangential_idx=[3], mu=0.5),
        ]
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        # Col 0: r_N^1 → row 1, Col 1: r_T^1 → row 0
        # Col 2: r_N^2 → row 4, Col 3: r_T^2 → row 3
        expected_B = np.zeros((6, 4))
        expected_B[1, 0] = 1.0
        expected_B[0, 1] = 1.0
        expected_B[4, 2] = 1.0
        expected_B[3, 3] = 1.0
        np.testing.assert_array_equal(cs.B, expected_B)


# =====================================================================
# Callable mu tests
# =====================================================================

class TestCallableMu:
    """Support for state-dependent friction coefficient."""

    def test_callable_mu(self):
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        # Override mu with a callable
        contacts[0]['mu'] = lambda y: 0.1 + 0.01 * abs(y[0])
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        y_test = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        out = cs.rhs(0.0, y_test, y_test, 0.001)
        mu_val = 0.1 + 0.01 * 2.0  # = 0.12
        # û_N = v_y + mu*|v_x| = -1 + 0.12*2 = -0.76
        # -û_N = 0.76
        assert abs(out[4] - 0.76) < 1e-14


# =====================================================================
# Integration test (end-to-end via solve_ivp_ns)
# =====================================================================

class TestIntegration:
    """End-to-end solve matching the manual impulse-level setup."""

    def test_bouncing_ball_fixed_step(self):
        """Fixed-step solve matches manual setup to machine precision."""
        import solve_nivp

        A, rhs_smooth, y0, contacts, gap_func = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs_smooth, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        t_auto, y_auto, *_ = solve_nivp.solve_ivp_ns(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=0.001,
        )

        t_auto = np.array(t_auto)
        # Ball should impact around t ≈ 0.452
        # After impact: v_y should be ≈ 0 (inelastic), v_x should decrease
        assert not np.any(np.isnan(y_auto))
        assert len(t_auto) > 100
        # Normal reaction is non-negative during contact
        contact_mask = y_auto[:, 3] <= 1e-8  # q_y ≈ 0
        p_N = y_auto[contact_mask, 4]
        assert np.all(p_N >= -1e-10), f"Negative p_N: {p_N.min()}"


# =====================================================================
# Frémond θ-averaged contact mode tests
# =====================================================================

class TestFremondContact:
    """Tests for fremond_contact=True (JTCAM 2025, §3.2)."""

    # ── Dissipation condition validation ──────────────────────────────

    def test_dissipation_valid_theta_half_e0(self):
        """θ=0.5, e=0 → valid (θ_max = 1.0)."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.5, fremond_contact=True,
        )
        assert isinstance(cs, ContactSystem)

    def test_dissipation_valid_theta_half_e1(self):
        """θ=0.5, e=1 → valid (θ_max = 0.5, boundary)."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=1.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.5, fremond_contact=True,
        )
        assert isinstance(cs, ContactSystem)

    def test_dissipation_rejects_theta_too_large(self):
        """θ=0.8, e=0.5 → θ_max = 2/3, violation → ValueError."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.5)
        with pytest.raises(ValueError, match="dissipation condition"):
            build_impulse_contact(
                A, rhs, y0, contacts, gap,
                theta=0.8, fremond_contact=True,
            )

    def test_dissipation_rejects_theta_too_small(self):
        """θ=0.3, e=0 → below 0.5 minimum → ValueError."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(e=0.0)
        with pytest.raises(ValueError, match="dissipation condition"):
            build_impulse_contact(
                A, rhs, y0, contacts, gap,
                theta=0.3, fremond_contact=True,
            )

    def test_dissipation_multi_contact_uses_max_e(self):
        """Dissipation uses ē = max(e^α) across all contacts."""
        n_phys = 6
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)
        def rhs(t, y): return np.zeros(n_phys)
        def gap(y, t): return np.array([y[2], y[5]])

        contacts = [
            dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3, e=0.2),
            dict(vel_normal_idx=4, vel_tangential_idx=[3], mu=0.5, e=0.8),
        ]
        # θ_max = 1/(1+0.8) ≈ 0.5556
        # θ=0.55 should be valid
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.55, fremond_contact=True,
        )
        assert isinstance(cs, ContactSystem)
        # θ=0.6 should fail
        with pytest.raises(ValueError, match="dissipation condition"):
            build_impulse_contact(
                A, rhs, y0, contacts, gap,
                theta=0.6, fremond_contact=True,
            )

    # ── Reaction row: θ-averaged vs standard ─────────────────────────

    def test_fremond_uses_norm_of_theta_blend(self):
        """Frémond evaluates μ||v_{T,k+θ}|| (norm of blend), not
        θ-blend of norms."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.4, e=0.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.5, fremond_contact=True,
        )
        # New state with v_x=3, v_y=-1  (on contact surface)
        y_new = np.array([3.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        # Old state with v_x=1, v_y=-2
        y_old = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0])

        out_new = cs.rhs(0.0, y_new, y_old, 0.001)
        out_old = cs.rhs(0.0, y_old, y_old, 0.001)

        # Effective contact law after θ-blend:
        # θ f_new + (1-θ) f_old should equal -Θ(v_{k+θ})
        theta = 0.5
        f_eff_N = theta * out_new[4] + (1 - theta) * out_old[4]
        f_eff_T = theta * out_new[5] + (1 - theta) * out_old[5]

        # Expected: v_{k+θ} = 0.5*[3,-1] + 0.5*[1,-2] = [2, -1.5]
        v_N_th = 0.5 * (-1.0) + 0.5 * (-2.0)  # = -1.5
        v_T_th = 0.5 * 3.0 + 0.5 * 1.0  # = 2.0
        c = 0.5 * (1 + 0) - 1  # = -0.5
        mu = 0.4
        v_N_prev = -2.0  # from y_old
        Theta_N = v_N_th + c * v_N_prev + mu * abs(v_T_th)
        # = -1.5 + (-0.5)*(-2) + 0.4*2 = -1.5 + 1.0 + 0.8 = 0.3
        Theta_T = v_T_th  # = 2.0

        np.testing.assert_allclose(f_eff_N, -Theta_N, atol=1e-13)
        np.testing.assert_allclose(f_eff_T, -Theta_T, atol=1e-13)

    def test_fremond_reduces_to_BE_at_theta1_e0(self):
        """θ=1, e=0 → Frémond = standard BE (eq. 68–69)."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.3, e=0.0)
        cs_be = build_impulse_contact(
            A, rhs, y0, contacts, gap, theta=1.0,
        )
        cs_fr = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=1.0, fremond_contact=True,
        )
        y_test = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        prev = np.array([1.0, -3.0, 0.0, 0.5, 0.0, 0.0])
        out_be = cs_be.rhs(0.0, y_test, prev, 0.001)
        out_fr = cs_fr.rhs(0.0, y_test, prev, 0.001)
        np.testing.assert_allclose(out_fr, out_be, atol=1e-14)

    def test_fremond_self_consistent_old_eval(self):
        """When y == prev_state, rhs returns -Θ(y) directly."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup(mu=0.3, e=0.5)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.5, fremond_contact=True,
        )
        y_test = np.array([2.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        # Old eval: y == prev
        out = cs.rhs(0.0, y_test, y_test, 0.001)

        # Expected -Θ(v_old) with c = 0.5*(1+0.5)-1 = -0.25
        c = -0.25
        mu = 0.3
        v_N = -1.0
        v_T = 2.0
        Theta_N = (1 + c) * v_N + mu * abs(v_T)
        # = 0.75*(-1) + 0.3*2 = -0.75 + 0.6 = -0.15
        Theta_T = v_T  # = 2.0

        np.testing.assert_allclose(out[4], -Theta_N, atol=1e-14)
        np.testing.assert_allclose(out[5], -Theta_T, atol=1e-14)

    # ── Physical coupling anti-blend implied ─────────────────────────

    def test_fremond_implies_physical_antiblend(self):
        """fremond_contact=True implies anti-blend for B p/h coupling."""
        A, rhs, y0, contacts, gap = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            theta=0.5, fremond_contact=True,
        )
        # y with reaction p_N=5, p_T=3
        y_new = np.array([1.0, 0.0, 0.0, 0.0, 5.0, 3.0])
        y_old = np.array([0.5, 0.0, 0.0, 0.0, 2.0, 1.0])
        h = 0.01
        out = cs.rhs(0.0, y_new, y_old, h)

        # Anti-blend: B @ (r_prev + (r - r_prev)/θ) / h
        B = cs.B
        r_new = np.array([5.0, 3.0])
        r_old = np.array([2.0, 1.0])
        theta = 0.5
        coupling = B @ (r_old + (r_new - r_old) / theta) / h
        f_phys = rhs(0.0, y_new[:4])
        np.testing.assert_allclose(out[:4], f_phys + coupling, atol=1e-12)

    # ── End-to-end integration ────────────────────────────────────────

    def test_fremond_bouncing_ball_e2e(self):
        """Frémond mode solves bouncing ball without NaN."""
        import solve_nivp

        A, rhs_smooth, y0, contacts, gap_func = _bouncing_ball_setup(
            mu=0.3, e=0.0,
        )
        cs = build_impulse_contact(
            A, rhs_smooth, y0, contacts, gap_func,
            theta=0.5, fremond_contact=True,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        t_out, y_out, *_ = solve_nivp.solve_ivp_ns(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='theta',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=0.005,
        )

        assert not np.any(np.isnan(y_out)), "NaN in Frémond solution"
        # Normal reaction non-negative during contact
        contact_mask = y_out[:, 3] <= 1e-8
        p_N = y_out[contact_mask, 4]
        assert np.all(p_N >= -1e-10), f"Negative p_N: min={p_N.min()}"
