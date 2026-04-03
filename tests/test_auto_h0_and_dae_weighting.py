"""Tests for automatic h₀ estimation (Hairer-Wanner) and DAE-aware error weighting.

Feature 2: Automatic h₀
    - Hairer-Wanner algorithm produces a reasonable initial step size
    - h0=None / h0='auto' triggers estimation on first step() call
    - Works through solve_nivp top-level API
    - Backward compat: explicit h0=float still works

Feature 3: DAE-aware error weighting
    - Algebraic DOFs (zero mass-matrix rows) detected and excluded from error norm
    - dae_var_weight='include' keeps old behaviour
    - dae_var_weight='auto' / 'exclude' skips algebraic DOFs
    - Works with sparse and dense mass matrices
    - Embedded (SDIRK2) and Richardson (BE) paths both respect the mask
"""

import math
import numpy as np
import pytest
import scipy.sparse as sp

from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp.integrations import BackwardEuler, SDIRK2
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_solver():
    return ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        tol=1e-12,
    )


def _rhs_decay(t, y):
    """dy/dt = -y"""
    return -y


def _rhs_2d(t, y):
    """Two-component linear: dy/dt = [-1, -10] * y (component-wise)."""
    return np.array([-y[0], -10.0 * y[1]])


# ===========================================================================
# FEATURE 2: Automatic h₀ estimation
# ===========================================================================

class TestHairerWarnerH0:
    """Unit tests for _estimate_h0_hairer."""

    def test_reasonable_order_of_magnitude(self):
        """For dy/dt = -y with y0=1, the estimated h should be O(0.01–0.1)."""
        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(integ, h0='auto', method_order=1)
        h_est = stepper._estimate_h0_hairer(_rhs_decay, 0.0, np.array([1.0]))
        # Should be sensible: not too tiny, not too huge
        assert 1e-4 < h_est < 1.0, f"h_est={h_est:.4e} out of expected range"

    def test_stiff_ode_gives_small_h(self):
        """For a stiff ODE (λ = -1000), h₀ should be very small."""
        def rhs_stiff(t, y):
            return -1000.0 * y

        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(integ, h0='auto', method_order=1)
        h_est = stepper._estimate_h0_hairer(rhs_stiff, 0.0, np.array([1.0]))
        assert h_est < 0.01, f"stiff h_est={h_est:.4e} too large"

    def test_auto_h0_none(self):
        """h0=None should behave the same as h0='auto'."""
        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(integ, h0=None, method_order=1)
        assert stepper._auto_h0 is True

    def test_auto_h0_triggers_on_first_step(self):
        """When h0='auto', the first step() call should use the estimated h."""
        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(integ, h0='auto', method_order=1)
        assert stepper._auto_h0 is True

        y0 = np.array([1.0])
        # The step should run without error and produce a result
        y_out, fk, h_next, E, success, se, iters = stepper.step(
            _rhs_decay, 0.0, y0, stepper.h  # h is placeholder
        )
        # After the call, auto_h0 should be consumed
        assert stepper._auto_h0 is False

    def test_explicit_h0_unchanged(self):
        """Explicit h0=0.05 should NOT trigger auto estimation."""
        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(integ, h0=0.05, method_order=1)
        assert stepper._auto_h0 is False
        assert stepper.h == 0.05

    def test_clamped_to_h_min_h_max(self):
        """The estimate should never exceed [h_min, h_max]."""
        integ = BackwardEuler(solver=_make_solver())
        stepper = AdaptiveStepping(
            integ, h0='auto', method_order=1, h_min=1e-3, h_max=0.01
        )
        h_est = stepper._estimate_h0_hairer(_rhs_decay, 0.0, np.array([1.0]))
        assert h_est >= stepper.h_min
        assert h_est <= stepper.h_max


class TestAutoH0HighLevel:
    """Test auto-h0 through solve_nivp."""

    def test_solve_nivp_auto_h0(self):
        from solve_nivp import solve_nivp

        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='backward_euler',
            projection='identity',
            solver='semismooth_newton',
            h0='auto',
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 0.05

    def test_solve_nivp_h0_none(self):
        from solve_nivp import solve_nivp

        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='backward_euler',
            projection='identity',
            solver='semismooth_newton',
            h0=None,
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 0.05

    def test_solve_nivp_sdirk2_auto_h0(self):
        """Auto h0 with SDIRK2 embedded path."""
        from solve_nivp import solve_nivp

        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0='auto',
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 0.01


# ===========================================================================
# FEATURE 3: DAE-aware error weighting
# ===========================================================================

class TestDetectAlgebraicDofs:
    """Unit tests for _detect_algebraic_dofs and _ensure_dae_mask."""

    def test_dense_mass_matrix_with_zero_rows(self):
        """A 4×4 mass matrix with rows 0,2 zero → those DOFs algebraic."""
        M = np.array([
            [0, 0, 0, 0],  # algebraic
            [0, 1, 0, 0],  # differential
            [0, 0, 0, 0],  # algebraic
            [0, 0, 0, 5],  # differential
        ], dtype=float)

        mask = AdaptiveStepping._detect_algebraic_dofs(M, 4)
        np.testing.assert_array_equal(mask, [0, 1, 0, 1])

    def test_sparse_mass_matrix_with_zero_rows(self):
        """Same test with a sparse mass matrix."""
        M = sp.diags([0, 1, 0, 5], 0, format='csr')
        mask = AdaptiveStepping._detect_algebraic_dofs(M, 4)
        np.testing.assert_array_equal(mask, [0, 1, 0, 1])

    def test_identity_mass_all_differential(self):
        """Identity → all DOFs differential."""
        M = np.eye(3)
        mask = AdaptiveStepping._detect_algebraic_dofs(M, 3)
        np.testing.assert_array_equal(mask, [1, 1, 1])

    def test_no_mass_matrix_all_differential(self):
        """When integrator has A=None (identity), mask should be all ones."""
        integ = BackwardEuler(solver=_make_solver())  # A=None → use_identity=True
        stepper = AdaptiveStepping(integ, h0=0.1, dae_var_weight='auto')
        mask = stepper._ensure_dae_mask(5)
        np.testing.assert_array_equal(mask, np.ones(5))

    def test_include_mode_ignores_mass_matrix(self):
        """dae_var_weight='include' should give all-ones mask regardless."""
        M = np.diag([0, 1, 0, 1])
        integ = BackwardEuler(solver=_make_solver(), A=M)
        stepper = AdaptiveStepping(integ, h0=0.1, dae_var_weight='include')
        mask = stepper._ensure_dae_mask(4)
        np.testing.assert_array_equal(mask, np.ones(4))


class TestDAEErrorWeighting:
    """Test that algebraic DOFs are actually excluded from the error norm."""

    def test_algebraic_dof_not_counted_embedded(self):
        """Error norm should ignore algebraic DOFs (embedded path).

        Set up a 2-DOF system with M = diag(0, 1): first DOF algebraic.
        If the embedded error is [100, 0.001], the algebraic component's
        huge error should NOT affect E when dae_var_weight='auto'.
        """
        # 2-DOF mass matrix: DOF-0 algebraic, DOF-1 differential
        M = np.diag([0.0, 1.0])
        integ = SDIRK2(solver=_make_solver(), A=M)
        stepper = AdaptiveStepping(integ, h0=0.1, method_order=2, dae_var_weight='auto')

        y_new = np.array([1.0, 1.0])
        err_vec = np.array([100.0, 0.001])  # huge algebraic error, tiny diff error

        E = stepper._scaled_error_embedded(y_new, err_vec)

        # With DAE mask, only DOF-1 contributes: E = |0.001 / (atol + rtol*1)|
        # atol=1e-6, rtol=1e-3 → tol = 1.001e-3 → E ≈ 0.999
        assert E < 1.5, f"E={E:.3e} should be small (algebraic error ignored)"

        # Now verify that with 'include' mode, E would be huge
        stepper_include = AdaptiveStepping(
            SDIRK2(solver=_make_solver(), A=M),
            h0=0.1, method_order=2, dae_var_weight='include',
        )
        E_include = stepper_include._scaled_error_embedded(y_new, err_vec)
        assert E_include > 10.0, f"E_include={E_include:.3e} should be large (algebraic counted)"

    def test_algebraic_dof_not_counted_richardson(self):
        """Same test but for Richardson path (_scaled_error)."""
        M = np.diag([0.0, 1.0])
        integ = BackwardEuler(solver=_make_solver(), A=M)
        stepper = AdaptiveStepping(integ, h0=0.1, method_order=1, dae_var_weight='auto')

        # y_lo = full-step, y_hi = half-step Richardson
        y_prev = np.array([1.0, 1.0])
        y_lo = np.array([1.1, 1.001])    # full step
        y_hi = np.array([1.0, 1.0005])   # half-step (more accurate)

        E = stepper._scaled_error(y_prev, y_lo, y_hi)

        # raw_err = (y_lo - y_hi)/(2^p - 1) = [0.1, 0.0005]
        # With DAE mask, DOF-0 (algebraic) is zeroed out → only DOF-1 counts
        # tol_1 = atol + rtol * max(|1.001|, |1.0005|) ≈ 1e-6 + 1e-3*1.001 ≈ 1.001e-3
        # E ≈ |0.0005 / 1.001e-3| = 0.499
        assert E < 1.0, f"E={E:.3e} should be < 1 (algebraic DOF excluded)"

    def test_all_differential_same_as_include(self):
        """When M has no zero rows, 'auto' should give same E as 'include'."""
        M = np.diag([2.0, 3.0])
        integ = SDIRK2(solver=_make_solver(), A=M)

        stepper_auto = AdaptiveStepping(integ, h0=0.1, method_order=2, dae_var_weight='auto')
        stepper_incl = AdaptiveStepping(
            SDIRK2(solver=_make_solver(), A=M),
            h0=0.1, method_order=2, dae_var_weight='include',
        )

        y_new = np.array([1.0, 2.0])
        err = np.array([0.001, 0.002])

        E_auto = stepper_auto._scaled_error_embedded(y_new, err)
        E_incl = stepper_incl._scaled_error_embedded(y_new, err)

        assert E_auto == pytest.approx(E_incl, rel=1e-12)


class TestDAEErrorWeightingSparse:
    """DAE mask with sparse mass matrix."""

    def test_sparse_biot_like_system(self):
        """Simulate a Biot-like system: M_uu=0 (displacement), M_pp!=0 (pressure).

        Build a block-diagonal mass matrix M = diag(0_n_u, I_n_p) and verify
        that displacement DOFs are excluded from the error norm.
        """
        n_u, n_p = 6, 4  # displacement, pressure DOFs
        n = n_u + n_p

        M_diag = np.zeros(n)
        M_diag[n_u:] = 1.0  # only pressure block has mass
        M = sp.diags(M_diag, 0, format='csr')

        def rhs(t, y):
            return -y  # simple test RHS

        integ = BackwardEuler(solver=_make_solver(), A=M)
        stepper = AdaptiveStepping(integ, h0=0.1, method_order=1, dae_var_weight='auto')

        mask = stepper._ensure_dae_mask(n)
        # First n_u should be 0 (algebraic), last n_p should be 1
        np.testing.assert_array_equal(mask[:n_u], 0.0)
        np.testing.assert_array_equal(mask[n_u:], 1.0)

        # Error norm should only reflect pressure DOFs
        y_new = np.ones(n)
        err = np.zeros(n)
        err[:n_u] = 999.0   # huge displacement error (should be ignored)
        err[n_u:] = 1e-4    # small pressure error

        # Force embedded-style evaluation
        E = stepper._scaled_error_embedded(y_new, err)
        # Only pressure error: tol ≈ 1e-6 + 1e-3*1 = 1.001e-3
        # E = sqrt(mean((1e-4 / 1.001e-3)^2)) ≈ 0.0999
        assert E < 0.5, f"E={E:.4f} should be small (displacement DOFs excluded)"


class TestDAEHighLevel:
    """Test DAE-aware weighting through solve_nivp."""

    def test_solve_nivp_dae_auto(self):
        """Verify dae_var_weight='auto' doesn't crash and integrates correctly."""
        from solve_nivp import solve_nivp

        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='backward_euler',
            projection='identity',
            solver='semismooth_newton',
            h0=0.05,
            dae_var_weight='auto',
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 0.05

    def test_solve_nivp_dae_include(self):
        """Verify dae_var_weight='include' (traditional) still works."""
        from solve_nivp import solve_nivp

        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0=0.05,
            dae_var_weight='include',
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 0.01

    def test_solve_nivp_dae_with_mass_matrix(self):
        """Full test with a mass matrix that has a small (near-zero) diagonal entry.

        We use M = diag(1e-15, 1, 1) so that DOF-0 is detected as algebraic
        by the row-norm threshold, while DOFs 1-2 are genuinely differential.
        Since f = [-y0, -y1, -y2], the system remains well-posed for the
        implicit solve and DOFs 1-2 should decay as exp(-t).
        """
        from solve_nivp import solve_nivp

        M = np.diag([1e-15, 1.0, 1.0])

        def rhs(t, y):
            return np.array([-y[0], -y[1], -y[2]])

        t, y, h, fk, info = solve_nivp(
            fun=rhs,
            t_span=(0.0, 1.0),
            y0=np.array([5.0, 1.0, 2.0]),
            method='backward_euler',
            projection='identity',
            solver='semismooth_newton',
            A=M,
            h0=0.05,
            dae_var_weight='auto',
        )
        # Differential DOFs (1, 2) should decay
        np.testing.assert_allclose(y[-1, 1], np.exp(-1.0), rtol=0.05)
        np.testing.assert_allclose(y[-1, 2], 2.0 * np.exp(-1.0), rtol=0.05)
