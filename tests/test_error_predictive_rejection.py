"""Tests for error-predictive rejection shrink logic.

Verifies that:
 - _rejection_shrink uses E_curr to compute the optimal next h
 - The formula matches safety * E^{-1/(p+1)} clamped by [_REJECT_FLOOR, 1]
 - Non-finite / zero E falls back to blind h_down
 - The floor clamp prevents extreme shrinks
 - Integration-level rejection actually uses the error-predictive h
   (not blind h_down) when E >> 1
"""

import numpy as np
import pytest

from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp.integrations import BackwardEuler, SDIRK2
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_solver(**kw):
    return ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        tol=1e-12,
        **kw,
    )


def _make_stepper(order=1, safety=0.9, h_down=0.6, h0=0.1, **kw):
    """Build an AdaptiveStepping wrapper around BackwardEuler."""
    integrator = BackwardEuler(solver=_make_solver())
    return AdaptiveStepping(
        integrator,
        h0=h0,
        safety=safety,
        h_down=h_down,
        method_order=order,
        mode="classic",
        **kw,
    )


# ---------------------------------------------------------------------------
# Unit tests for _rejection_shrink
# ---------------------------------------------------------------------------

class TestRejectionShrinkFormula:
    """Direct unit tests on _rejection_shrink."""

    def test_basic_formula(self):
        """h_new = h * safety * E^{-1/(p+1)}, within [floor, 1]."""
        stepper = _make_stepper(order=1, safety=0.9)
        h = 0.5
        E = 10.0
        p = 1

        expected_g = 0.9 * (E ** (-1.0 / (p + 1)))  # ~0.284
        expected_h = h * max(stepper._REJECT_FLOOR, expected_g)
        result = stepper._rejection_shrink(h, E)

        assert result == pytest.approx(expected_h, rel=1e-12)

    def test_moderate_error_not_clamped(self):
        """For modest E just above 1, shrink factor is between floor and 1."""
        stepper = _make_stepper(order=2, safety=0.9)
        h = 1.0
        E = 2.0
        p = 2

        g = 0.9 * (E ** (-1.0 / (p + 1)))  # ~0.714
        assert g > stepper._REJECT_FLOOR
        assert g < 1.0

        result = stepper._rejection_shrink(h, E)
        assert result == pytest.approx(h * g, rel=1e-12)

    def test_extreme_error_hits_floor(self):
        """Very large E should clamp the ratio at _REJECT_FLOOR."""
        stepper = _make_stepper(order=1, safety=0.9)
        h = 1.0
        E = 1e12  # enormous error

        g = 0.9 * (E ** (-1.0 / 2.0))  # ≈ 9e-7, far below floor
        assert g < stepper._REJECT_FLOOR

        result = stepper._rejection_shrink(h, E)
        assert result == pytest.approx(h * stepper._REJECT_FLOOR, rel=1e-12)

    def test_nonfinite_error_fallback(self):
        """np.inf or np.nan E should fall back to blind h_down."""
        stepper = _make_stepper(order=1, safety=0.9, h_down=0.5)
        h = 1.0

        assert stepper._rejection_shrink(h, np.inf) == pytest.approx(h * 0.5)
        assert stepper._rejection_shrink(h, np.nan) == pytest.approx(h * 0.5)

    def test_zero_error_fallback(self):
        """E == 0 should fall back to blind h_down."""
        stepper = _make_stepper(order=1, safety=0.9, h_down=0.5)
        h = 1.0
        assert stepper._rejection_shrink(h, 0.0) == pytest.approx(h * 0.5)

    def test_h_min_floor(self):
        """Result never goes below h_min."""
        stepper = _make_stepper(order=1, safety=0.9, h0=1e-9, h_min=1e-8)
        stepper.h_min = 1e-8
        h = 1e-8  # already at the minimum
        result = stepper._rejection_shrink(h, 100.0)
        assert result >= stepper.h_min

    def test_e_slightly_above_one(self):
        """For E just above 1.0, the shrink should be mild (close to safety)."""
        stepper = _make_stepper(order=2, safety=0.9)
        h = 1.0
        E = 1.01
        p = 2

        g = 0.9 * (E ** (-1.0 / 3.0))  # ~0.897
        result = stepper._rejection_shrink(h, E)
        assert result == pytest.approx(h * g, rel=1e-10)
        # Should be very close to h (only ~10% shrink)
        assert result > 0.85 * h


class TestRejectionShrinkVsBlind:
    """Compare error-predictive vs the old blind h_down approach."""

    def test_large_error_shrinks_more_than_h_down(self):
        """With E >> 1 the predictive shrink should be much smaller than h*h_down."""
        stepper = _make_stepper(order=1, safety=0.9, h_down=0.6)
        h = 1.0
        E = 50.0

        blind = h * stepper.h_down  # 0.6
        predictive = stepper._rejection_shrink(h, E)

        # The predictive value should be noticeably smaller
        assert predictive < blind, (
            f"Predictive {predictive:.4f} should be < blind {blind:.4f} for E={E}"
        )

    def test_small_error_shrinks_less_aggressively(self):
        """With E just above 1 the predictive shrink should be gentler than h*h_down."""
        stepper = _make_stepper(order=2, safety=0.9, h_down=0.6)
        h = 1.0
        E = 1.05

        blind = h * stepper.h_down  # 0.6
        predictive = stepper._rejection_shrink(h, E)

        # g ≈ 0.9 * 1.05^{-1/3} ≈ 0.885 → predictive > blind
        assert predictive > blind, (
            f"Predictive {predictive:.4f} should be > blind {blind:.4f} for small E={E}"
        )


# ---------------------------------------------------------------------------
# Integration-level test: verify rejection uses error-predictive h
# ---------------------------------------------------------------------------

class TestRejectionInIntegration:
    """End-to-end: the adaptive loop produces the error-predictive h on rejection."""

    @staticmethod
    def _rhs_stiff(t, y):
        """dy/dt = -1000*y  (very stiff scalar)."""
        return -1000.0 * y

    def test_embedded_classic_rejection_uses_predictive_h(self):
        """Force a rejection in embedded classic mode and check the returned h."""
        integrator = SDIRK2(solver=_make_solver())
        stepper = AdaptiveStepping(
            integrator,
            h0=1.0,         # deliberately too large for the stiff system
            safety=0.9,
            h_down=0.6,
            mode="classic",
            method_order=2,
            verbose=False,
        )

        y0 = np.array([1.0])

        result = stepper.step(self._rhs_stiff, 0.0, y0, 1.0)
        y_out, fk_out, h_next, E_curr, accepted, solver_err, iters = result

        # The first step with h=1.0 on a stiff system should be rejected
        if not accepted:
            # Compute what the error-predictive formula would give
            p = 2
            g = 0.9 * (E_curr ** (-1.0 / (p + 1)))
            g = max(stepper._REJECT_FLOOR, min(g, 1.0))
            h_expected = max(stepper.h_min, g * 1.0)

            assert h_next == pytest.approx(h_expected, rel=1e-10), (
                f"Rejection h={h_next:.6e} should match predictive "
                f"h={h_expected:.6e} (E={E_curr:.3e})"
            )
            # Also verify it differs from the old blind h_down
            h_blind = 1.0 * 0.6
            assert h_next != pytest.approx(h_blind, rel=0.05), (
                f"Rejection h should differ from blind h_down={h_blind}"
            )
