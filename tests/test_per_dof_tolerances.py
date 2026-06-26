"""Tests for per-DOF tolerance vectors + weighted RMS norm.

Verifies that:
 - Scalar tolerances still work (backward compat)
 - Per-DOF array tolerances are accepted and used
 - Per-slice tolerances are expanded correctly
 - Weighted RMS norm in nonlinear solver converges
 - Different tolerances per component actually affect acceptance
 - Top-level solve_nivp forwards nl_atol / nl_rtol correctly
"""

import math

import numpy as np
import pytest

from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.integrations import SDIRK2, BackwardEuler
from solve_nivp.projections import IdentityProjection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _rhs_decay(t, y):
    """dy/dt = −y"""
    return -y


def _rhs_2d(t, y):
    """2-D ODE: dy0/dt = -y0,  dy1/dt = -100*y1  (stiff + mild)"""
    return np.array([-y[0], -100.0 * y[1]])


def _make_solver(**kw):
    return ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        tol=1e-12,
        **kw,
    )


def _make_vi_solver(**kw):
    return ImplicitEquationSolver(
        method='VI',
        proj=IdentityProjection(),
        component_slices=[slice(0, 2)],
        tol=1e-12,
        **kw,
    )


# ===================================================================
# 1. AdaptiveStepping: tolerance expansion helpers
# ===================================================================

class TestAdaptiveSteppingTolExpansion:
    """Verify _expand_tol / _ensure_tol_vectors work correctly."""

    def _make_stepper(self, atol=1e-6, rtol=1e-3, slices=None):
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        return AdaptiveStepping(
            integrator=integ,
            component_slices=slices,
            atol=atol,
            rtol=rtol,
        )

    # --- scalar broadcast ---
    def test_scalar_broadcast(self):
        stepper = self._make_stepper(atol=1e-5, rtol=1e-2)
        atol_v, rtol_v = stepper._ensure_tol_vectors(4)
        np.testing.assert_array_equal(atol_v, np.full(4, 1e-5))
        np.testing.assert_array_equal(rtol_v, np.full(4, 1e-2))

    # --- per-DOF array (length matches n) ---
    def test_per_dof_array(self):
        a = np.array([1e-4, 1e-6, 1e-8])
        r = np.array([1e-1, 1e-2, 1e-3])
        stepper = self._make_stepper(atol=a, rtol=r)
        atol_v, rtol_v = stepper._ensure_tol_vectors(3)
        np.testing.assert_array_equal(atol_v, a)
        np.testing.assert_array_equal(rtol_v, r)

    # --- per-slice expansion ---
    def test_per_slice_expansion(self):
        slices = [slice(0, 2), slice(2, 5)]
        a = np.array([1e-3, 1e-6])   # two blocks
        r = np.array([1e-1, 1e-4])
        stepper = self._make_stepper(atol=a, rtol=r, slices=slices)
        atol_v, rtol_v = stepper._ensure_tol_vectors(5)
        expected_a = np.array([1e-3, 1e-3, 1e-6, 1e-6, 1e-6])
        expected_r = np.array([1e-1, 1e-1, 1e-4, 1e-4, 1e-4])
        np.testing.assert_array_almost_equal(atol_v, expected_a)
        np.testing.assert_array_almost_equal(rtol_v, expected_r)

    # --- wrong-length array raises ---
    def test_bad_length_raises(self):
        stepper = self._make_stepper(atol=np.array([1e-3, 1e-6]))
        with pytest.raises(ValueError, match="atol"):
            stepper._ensure_tol_vectors(5)

    # --- cache invalidation on setter ---
    def test_setter_invalidates_cache(self):
        stepper = self._make_stepper(atol=1e-5, rtol=1e-3)
        stepper._ensure_tol_vectors(3)
        assert stepper._atol_vec is not None
        stepper.atol = 1e-7
        assert stepper._atol_vec is None  # cache cleared
        atol_v, _ = stepper._ensure_tol_vectors(3)
        np.testing.assert_array_equal(atol_v, np.full(3, 1e-7))


# ===================================================================
# 2. AdaptiveStepping: integration with vector tolerances
# ===================================================================

class TestAdaptiveSteppingVectorTol:
    """End-to-end adaptive steps with per-DOF tolerances."""

    def test_scalar_still_works(self):
        """Backward compat: scalar atol/rtol produces correct result."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        stepper = AdaptiveStepping(
            integrator=integ,
            atol=1e-6,
            rtol=1e-3,
        )
        y0 = np.array([1.0])
        result = stepper.step(_rhs_decay, 0.0, y0, 0.1)
        y_new, h_next, fk, info = result[0], result[1], result[2], result[3:]
        assert np.isfinite(y_new).all()

    def test_vector_tol_integrates(self):
        """Per-DOF vector tolerances run without error on 2-D system."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        stepper = AdaptiveStepping(
            integrator=integ,
            atol=np.array([1e-4, 1e-8]),
            rtol=np.array([1e-2, 1e-5]),
        )
        y0 = np.array([1.0, 1.0])
        result = stepper.step(_rhs_2d, 0.0, y0, 0.01)
        y_new = result[0]
        assert y_new.shape == (2,)
        assert np.isfinite(y_new).all()


# ===================================================================
# 3. ImplicitEquationSolver: WRMS convergence helpers
# ===================================================================

class TestImplicitSolverWRMS:
    """Verify _wrms / _converged / _errf_metric helpers."""

    def test_wrms_uniform(self):
        """With constant atol and rtol=0, wrms ≈ ||F|| / atol / sqrt(n)."""
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            nl_atol=1.0,
            nl_rtol=0.0,
        )
        F = np.array([0.5, 0.5, 0.5, 0.5])
        y = np.zeros(4)
        solver._ensure_nl_tol_vectors(4)
        wrms = solver._wrms(F, y)
        # weight = atol + rtol*|y| = 1.0 + 0 = 1.0
        # wrms = sqrt(mean((0.5/1.0)^2)) = 0.5
        np.testing.assert_allclose(wrms, 0.5, atol=1e-14)

    def test_converged_true(self):
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            nl_atol=1.0,
            nl_rtol=0.0,
        )
        F = np.array([0.1, 0.1])
        y = np.zeros(2)
        solver._ensure_nl_tol_vectors(2)
        assert solver._converged(F, y) is True   # wrms = 0.1 <= 1

    def test_converged_false(self):
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            nl_atol=0.01,
            nl_rtol=0.0,
        )
        F = np.array([0.5, 0.5])
        y = np.zeros(2)
        solver._ensure_nl_tol_vectors(2)
        assert solver._converged(F, y) is False   # wrms = 0.5/0.01 = 50 >> 1

    def test_legacy_mode_when_none(self):
        """When nl_atol/nl_rtol are None, _use_weighted_norm is False."""
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
        )
        assert solver._use_weighted_norm is False

    def test_weighted_norm_block_diagnostics(self):
        """Diagnostics should identify which component dominates the WRMS norm."""
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            nl_atol=1.0,
            nl_rtol=0.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )
        solver.record_diagnostics = True
        solver.diagnostic_component_names = ['slow', 'dominant']

        F = np.array([0.5, 0.5, 2.0, 2.0])
        y = np.zeros(4)

        converged, metric = solver._converged_with_metric(F, y)

        assert converged is False
        assert metric == pytest.approx(math.sqrt((0.25 + 0.25 + 4.0 + 4.0) / 4.0))
        diag = solver._last_nl_block_diagnostics
        assert diag['global_metric'] == pytest.approx(metric)
        assert [blk['name'] for blk in diag['blocks']] == ['slow', 'dominant']
        assert diag['blocks'][0]['rms'] == pytest.approx(0.5)
        assert diag['blocks'][1]['rms'] == pytest.approx(2.0)
        assert diag['blocks'][1]['fraction'] > diag['blocks'][0]['fraction']


# ===================================================================
# 4. ImplicitEquationSolver: solve with weighted norm
# ===================================================================

class TestImplicitSolverSolveWRMS:
    """Verify nonlinear solves converge with weighted-norm tolerances."""

    def _run_one_step(self, solver_method, **extra_solver_kw):
        """Build solver + integrator and do a single implicit step."""
        solver = ImplicitEquationSolver(
            method=solver_method,
            proj=IdentityProjection(),
            tol=1e-10,
            nl_atol=1e-6,
            nl_rtol=1e-3,
            **extra_solver_kw,
        )
        integ = BackwardEuler(solver=solver)
        y0 = np.array([1.0, 2.0])
        y_new, fk, err, ok, iters = integ.step(_rhs_2d, 0.0, y0, 0.01)
        return y_new, ok, iters

    def test_semismooth_newton_converges(self):
        y, ok, iters = self._run_one_step('semismooth_newton')
        assert ok is True
        assert np.isfinite(y).all()

    def test_vi_converges(self):
        y, ok, iters = self._run_one_step(
            'VI', component_slices=[slice(0, 2)],
        )
        assert ok is True
        assert np.isfinite(y).all()


# ===================================================================
# 5. Top-level solve_nivp with per-DOF tolerances
# ===================================================================

class TestSolveIvpNsVectorTol:
    """Verify the full solve_nivp pipeline with vector tolerances."""

    def test_per_dof_atol_rtol(self):
        from solve_nivp import solve_nivp

        y0 = np.array([1.0, 1.0])
        t, y, h, fk, info = solve_nivp(
            fun=_rhs_2d,
            t_span=(0.0, 0.1),
            y0=y0,
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            atol=np.array([1e-4, 1e-8]),
            rtol=np.array([1e-2, 1e-6]),
            h0=0.01,
        )
        y_exact = np.array([np.exp(-0.1), np.exp(-10.0)])
        # Just check it runs and is in the right ballpark
        assert t[-1] >= 0.1 - 1e-12
        assert np.isfinite(y[-1]).all()
        # Loose check — mainly testing that it doesn't crash
        np.testing.assert_allclose(y[-1, 0], y_exact[0], rtol=0.05)

    def test_nl_atol_nl_rtol(self):
        """Forward nl_atol / nl_rtol to the nonlinear solver."""
        from solve_nivp import solve_nivp

        y0 = np.array([1.0, 1.0])
        t, y, h, fk, info = solve_nivp(
            fun=_rhs_2d,
            t_span=(0.0, 0.1),
            y0=y0,
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            atol=1e-6,
            rtol=1e-3,
            nl_atol=1e-6,
            nl_rtol=1e-3,
            h0=0.01,
        )
        assert t[-1] >= 0.1 - 1e-12
        assert np.isfinite(y[-1]).all()

    def test_scalar_backward_compat(self):
        """Scalar tolerances still work identically to before."""
        from solve_nivp import solve_nivp

        y0 = np.array([1.0])
        t, y, h, fk, info = solve_nivp(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=y0,
            method='sdirk2',
            projection='identity',
            solver='VI',
            atol=1e-6,
            rtol=1e-3,
            h0=0.05,
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 1e-2

    def test_per_slice_tol(self):
        """Per-slice tolerances expand correctly through the pipeline."""
        from solve_nivp import solve_nivp

        y0 = np.array([1.0, 1.0, 1.0])
        slices = [slice(0, 1), slice(1, 3)]

        t, y, h, fk, info = solve_nivp(
            fun=lambda t, y: -y,
            t_span=(0.0, 0.5),
            y0=y0,
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            atol=np.array([1e-4, 1e-8]),
            rtol=np.array([1e-2, 1e-5]),
            component_slices=slices,
            h0=0.01,
        )
        assert t[-1] >= 0.5 - 1e-12
        assert np.isfinite(y[-1]).all()

    def test_adaptive_opts_vector_override(self):
        """Vector atol/rtol passed through adaptive_opts are honoured."""
        from solve_nivp import solve_nivp

        y0 = np.array([1.0, 1.0])
        t, y, h, fk, info = solve_nivp(
            fun=_rhs_2d,
            t_span=(0.0, 0.05),
            y0=y0,
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            atol=1e-6,
            rtol=1e-3,
            adaptive_opts={
                'atol': np.array([1e-3, 1e-7]),
                'rtol': np.array([1e-1, 1e-4]),
            },
            h0=0.01,
        )
        assert t[-1] >= 0.05 - 1e-12
        assert np.isfinite(y[-1]).all()
