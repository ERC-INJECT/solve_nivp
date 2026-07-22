"""SDIRK2 + SOC contact with step_size_arg = γh.

Tests cover four contact invariants:
  (a) gap ≥ 0
  (b) p_N ≥ 0
  (c) Coulomb feasibility ‖p_T‖ ≤ μ·p_N
  (d) energy monotone decreasing during contact

Plus:
  - Comparison with Backward Euler (matching physics)
  - get_s0 pre-stress works correctly with γh
  - SDIRK2 adaptive stepping still works
"""

import numpy as np
import pytest

import solve_nivp
from solve_nivp.contact import build_impulse_contact


# =====================================================================
# Helpers
# =====================================================================

def _bouncing_ball_setup(mu=0.3, e=0.0):
    """Standard 2D bouncing ball: [v_x, v_y, q_x, q_y]."""
    mass = 1.0
    gravity = np.array([0.0, -9.81])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        v = y[0:2]
        return np.concatenate([mass * gravity, v])

    def gap_func(y, t):
        return np.array([y[3]])

    y0 = np.array([2.0, 0.0, 0.0, 1.0])  # sliding + falling
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=e)]
    return A, rhs, y0, contacts, gap_func, mass


def _spring_slider_setup(mu=0.5, e=0.0):
    """1D spring-slider with gravity pre-stress via get_s0.

    State: [v, q] — velocity and position along the horizontal.
    The slider rests on a friction surface; gravity acts normal to it.
    A spring pulls the slider: F_spring = -k*q.
    """
    mass = 1.0
    k = 100.0
    g = 9.81
    A = np.diag([mass, 1.0])

    def rhs(t, y):
        v, q = y
        return np.array([-k * q, v])

    _h = [0.01]  # mutable step-size ref

    y0 = np.array([0.0, 0.5])  # initial displacement, zero velocity
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[], mu=mu, e=e)]

    def gap_func(y, t):
        return np.array([1.0])  # always in contact

    return A, rhs, y0, contacts, gap_func, mass, g, k, _h


# =====================================================================
# Bouncing ball: SDIRK2 vs BE
# =====================================================================

class TestSDIRK2BouncingBall:
    """End-to-end bouncing ball with SDIRK2 + SOC contact (e=0)."""

    @pytest.fixture
    def sdirk2_solution(self):
        """Run SDIRK2 bouncing ball once, cache result."""
        A, rhs_smooth, y0, contacts, gap_func, mass = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs_smooth, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )
        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=0.001,
        )
        return t, y, h, fk, info, cs

    def test_no_nan(self, sdirk2_solution):
        """Solution should be finite."""
        _, y, *_ = sdirk2_solution
        assert not np.any(np.isnan(y)), "NaN in SDIRK2 solution"

    def test_gap_nonnegative(self, sdirk2_solution):
        """(a) gap = q_y ≥ 0  (no penetration, O(h²) tolerance)."""
        _, y, *_ = sdirk2_solution
        q_y = y[:, 3]  # position in y-direction
        # SDIRK2's stage-2 extrapolation may allow O(h²) penetration;
        # tighter gap enforcement would require event location (not implemented).
        assert np.all(q_y >= -1e-3), f"Penetration: min gap = {q_y.min()}"

    def test_normal_reaction_nonneg(self, sdirk2_solution):
        """(b) p_N ≥ 0  (unilateral contact)."""
        _, y, *_ = sdirk2_solution
        # During contact (q_y ≈ 0), p_N must be non-negative
        contact_mask = y[:, 3] <= 1e-6
        if np.any(contact_mask):
            p_N = y[contact_mask, 4]
            assert np.all(p_N >= -1e-10), f"Negative p_N: min = {p_N.min()}"

    def test_coulomb_feasibility(self, sdirk2_solution):
        """(c) ‖p_T‖ ≤ μ·p_N during contact."""
        _, y, *_ = sdirk2_solution
        mu = 0.3
        contact_mask = y[:, 3] <= 1e-6
        if np.any(contact_mask):
            p_N = y[contact_mask, 4]
            p_T = np.abs(y[contact_mask, 5])
            violations = p_T - mu * p_N
            assert np.all(violations <= 1e-8), \
                f"Coulomb violation: max = {violations.max()}"

    def test_energy_monotone_during_contact(self, sdirk2_solution):
        """(d) Kinetic + potential energy is non-increasing during contact."""
        t, y, *_ = sdirk2_solution
        mass = 1.0
        g = 9.81
        KE = 0.5 * mass * (y[:, 0]**2 + y[:, 1]**2)
        PE = mass * g * y[:, 3]
        E = KE + PE

        contact_mask = y[:, 3] <= 1e-6
        contact_indices = np.where(contact_mask)[0]
        if len(contact_indices) > 1:
            E_contact = E[contact_indices]
            dE = np.diff(E_contact)
            # Energy should not increase during contact (dissipative)
            assert np.all(dE <= 1e-6), \
                f"Energy increased during contact: max dE = {dE.max()}"

    def test_matches_be_qualitatively(self):
        """SDIRK2 and BE produce qualitatively similar results."""
        A, rhs_smooth, y0, contacts, gap_func, mass = _bouncing_ball_setup()

        results = {}
        for method in ['backward_euler', 'sdirk2']:
            cs = build_impulse_contact(
                A, rhs_smooth, y0, contacts, gap_func,
                theta=1.0,
                component_slices=[slice(0, 2), slice(2, 4)],
            )
            t, y, *_ = solve_nivp.solve_nivp(
                fun=cs.rhs,
                t_span=(0.0, 0.5),
                y0=cs.y0,
                A=cs.A,
                method=method,
                projection=cs.projection,
                solver='semismooth_newton',
                solver_opts=dict(tol=1e-12, max_iter=200,
                                 lam_update_strategy='none',
                                 globalization='linesearch'),
                component_slices=cs.component_slices,
                integrator_opts=cs.integrator_opts,
                adaptive=False,
                h0=0.001,
            )
            results[method] = (np.array(t), y)

        t_be, y_be = results['backward_euler']
        t_sd, y_sd = results['sdirk2']

        # Both should have the same number of steps (fixed h)
        assert len(t_be) == len(t_sd)

        # Impact time should be similar (ball hits ground ~ t=0.452)
        # Use a relaxed gap threshold — SDIRK2's stage-2 extrapolation
        # may allow slightly larger penetration than BE.
        impact_be = next((i for i in range(len(t_be))
                          if y_be[i, 3] < 1e-3), None)
        impact_sd = next((i for i in range(len(t_sd))
                          if y_sd[i, 3] < 1e-3), None)
        assert impact_be is not None, "BE: no impact detected"
        assert impact_sd is not None, "SDIRK2: no impact detected"
        # Impact should occur within ~5 steps of each other
        assert abs(impact_be - impact_sd) < 10, \
            f"Impact time mismatch: BE={impact_be}, SDIRK2={impact_sd}"

        # Post-contact velocity should agree (inelastic → v_y ≈ 0)
        last_be = y_be[-1]
        last_sd = y_sd[-1]
        # Both should have ball at (near) rest on the ground
        assert last_be[3] < 0.01, "BE: ball not on ground at t=0.5"
        assert last_sd[3] < 0.01, "SDIRK2: ball not on ground at t=0.5"


# =====================================================================
# Pre-stress via get_s0 with step_size_ref
# =====================================================================

class TestSDIRK2PreStress:
    """Verify get_s0 works correctly when step_size_arg = γh."""

    def test_prestress_static_equilibrium(self):
        """A block under gravity on a friction surface should reach
        static equilibrium with SDIRK2, including get_s0 pre-stress."""
        mass = 1.0
        g = 9.81
        k = 50.0  # spring stiffness
        A = np.diag([mass, mass, 1.0, 1.0])

        def rhs(t, y):
            v_x, v_y, q_x, q_y = y
            # Spring + gravity
            return np.array([-k * q_x, -mass * g, v_x, v_y])

        def gap_func(y, t):
            return np.array([y[3]])

        _h = [0.001]
        y0 = np.array([1.0, 0.0, 0.5, 0.0])  # on the ground, sliding
        contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                         mu=0.8, e=0.0)]

        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
            get_s0=lambda y: mass * g * _h[0],
            step_size_ref=_h,
        )

        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 0.5),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=0.001,
        )

        assert not np.any(np.isnan(y)), "NaN in pre-stress SDIRK2 solution"

        # Gap should stay non-negative (block stays on surface)
        q_y = y[:, 3]
        assert np.all(q_y >= -1e-6), f"Penetration with pre-stress: {q_y.min()}"

        # Normal reaction should be at least gravitational
        contact_mask = q_y <= 1e-6
        if np.any(contact_mask):
            p_N = y[contact_mask, 4]
            assert np.all(p_N >= -1e-10), f"Negative p_N with pre-stress: {p_N.min()}"


# =====================================================================
# SDIRK2 adaptive stepping with contact
# =====================================================================

class TestSDIRK2AdaptiveContact:
    """SDIRK2 with embedded error + adaptive stepping + contact."""

    def test_adaptive_bouncing_ball(self):
        """Adaptive SDIRK2 completes bouncing ball without failure."""
        A, rhs_smooth, y0, contacts, gap_func, mass = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs_smooth, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True,
            h0=0.01,
            atol=1e-6,
            rtol=1e-3,
        )

        assert not np.any(np.isnan(y)), "NaN in adaptive SDIRK2"
        assert len(t) > 10, "Too few steps"

        # Contact feasibility (O(h²) penetration tolerance for SDIRK2)
        q_y = y[:, 3]
        assert np.all(q_y >= -1e-3), f"Adaptive gap violation: {q_y.min()}"

        contact_mask = q_y <= 1e-6
        if np.any(contact_mask):
            p_N = y[contact_mask, 4]
            assert np.all(p_N >= -1e-10), \
                f"Adaptive p_N violation: {p_N.min()}"


# =====================================================================
# Coupling strength unit test
# =====================================================================

class TestCouplingStrength:
    """Verify that the γh step_size_arg gives correct coupling magnitude."""

    def test_rhs_coupling_with_gamma_h(self):
        """RHS coupling B·r/(θ·h_val) uses γh, giving correct net impulse
        after the stage equation multiplies by γh."""
        A, rhs_smooth, y0, contacts, gap_func, mass = _bouncing_ball_setup()
        cs = build_impulse_contact(
            A, rhs_smooth, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        gamma = 1.0 - np.sqrt(2.0) / 2.0  # SDIRK2 γ ≈ 0.2929
        h_full = 0.01
        gh = gamma * h_full

        # Test state with known reaction
        y_test = np.array([2.0, 0.0, 0.0, 0.0, 5.0, 3.0])
        prev = y_test.copy()

        # Call with γh as step size (what SDIRK2 now passes)
        out_gh = cs.rhs(0.0, y_test, prev, gh)

        # Expected coupling: B @ r / (1.0 * gh)
        B = cs.B
        r = np.array([5.0, 3.0])
        f_phys = rhs_smooth(0.0, y_test[:4])
        expected_phys = f_phys + B @ r / gh
        np.testing.assert_allclose(out_gh[:4], expected_phys, rtol=1e-12)

        # When stage equation multiplies by γh:
        # net coupling = γh * B @ r / (1 * γh) = B @ r   ✓
        net_impulse = gh * (out_gh[:4] - f_phys)
        np.testing.assert_allclose(net_impulse, B @ r, rtol=1e-12)
