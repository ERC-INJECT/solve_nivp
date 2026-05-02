"""Tests for the RadauIIA integration method (all stage counts)."""

import math
import numpy as np
import pytest

from solve_nivp.integrations import RadauIIA, BackwardEuler
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection
from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp import solve_nivp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rhs_decay(t, y):
    """dy/dt = −y  →  y(t) = exp(−t)"""
    return -y


def _rhs_stiff(t, y):
    """dy/dt = −50 y  (stiff scalar ODE)"""
    return -50.0 * y


def _rhs_van_pol_slow(t, y):
    """Van der Pol with μ=1 (mildly stiff 2-D system)."""
    return np.array([y[1], (1.0 - y[0]**2) * y[1] - y[0]])


def _make_solver(tol=1e-12):
    return ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        tol=tol,
    )


def _make_solver_with_jac(tol=1e-12):
    """Solver with analytical Jacobian for _rhs_decay — enables coupled Newton."""
    solver = _make_solver(tol=tol)
    solver.rhs_jacobian = lambda t, y, *a, **kw: -np.eye(len(y))
    return solver


def _make_integ(stages, **kw):
    return RadauIIA(stages=stages, solver=_make_solver(), **kw)


def _make_integ_with_jac(stages, **kw):
    """Integrator with Jacobian — uses coupled Newton for full method order."""
    return RadauIIA(stages=stages, solver=_make_solver_with_jac(), **kw)


# ---------------------------------------------------------------------------
# 1. Tableau structure checks
# ---------------------------------------------------------------------------

class TestRadauIIATableau:
    """Verify Butcher tableau properties for all supported stage counts."""

    @pytest.mark.parametrize("stages", [1, 2, 3])
    def test_stiff_accuracy(self, stages):
        """Last row of A must equal b (stiffly accurate property)."""
        integ = _make_integ(stages)
        np.testing.assert_allclose(
            integ._rk_A[-1, :], integ._rk_b, atol=1e-14,
            err_msg=f"Stiff accuracy violated for stages={stages}",
        )

    def test_coupled_newton_stacks_and_restores_component_slices(self):
        """Stacked coupled Newton must shift PETSc field-split blocks by stage."""
        original_slices = [slice(0, 1), np.array([1, 3])]
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            component_slices=original_slices,
            petsc_field_slices=[slice(0, 2), np.array([2, 3])],
        )
        integ = RadauIIA(stages=2, solver=solver, use_coupled_newton=True)
        seen = {}

        def fake_solve(F, Z0):
            seen['component_slices'] = solver.component_slices
            seen['petsc_field_slices'] = solver.petsc_field_slices
            return Z0, F(Z0), 0.0, True, 1

        solver.solve = fake_solve
        y = np.array([1.0, 2.0, 3.0, 4.0])

        result = integ._step_coupled_newton_impl(
            0.0,
            y,
            0.1,
            len(y),
            np.eye(len(y)),
            integ._rk_A,
            integ._rk_c,
            integ.stages,
            lambda t, yy, stage_h_override=None: -yy,
            lambda t, yy, fk, prev, h_val: -np.eye(len(yy)),
            None,
            None,
        )

        assert result is not None
        assert solver.component_slices is original_slices
        stacked = seen['component_slices']
        assert len(stacked) == 4
        assert stacked[0] == slice(0, 1)
        np.testing.assert_array_equal(stacked[1], [1, 3])
        assert stacked[2] == slice(4, 5)
        np.testing.assert_array_equal(stacked[3], [5, 7])
        field_stacked = seen['petsc_field_slices']
        assert len(field_stacked) == 2
        np.testing.assert_array_equal(field_stacked[0], [0, 1, 4, 5])
        np.testing.assert_array_equal(field_stacked[1], [2, 3, 6, 7])

    @pytest.mark.parametrize("stages", [1, 2, 3])
    def test_abscissa_sum_one(self, stages):
        """Row sums of A must equal c (consistency condition)."""
        integ = _make_integ(stages)
        row_sums = integ._rk_A.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, integ._rk_c, atol=1e-14,
            err_msg=f"Abscissa consistency violated for stages={stages}",
        )

    @pytest.mark.parametrize("stages,order", [(1, 1), (2, 3), (3, 5)])
    def test_method_order_attribute(self, stages, order):
        integ = _make_integ(stages)
        assert integ.order == order

    def test_embedded_order_attribute(self):
        """Embedded pair order attribute must be 1 for stages=2."""
        integ = _make_integ(2)
        assert integ.embedded_order == 1

    @pytest.mark.parametrize("stages", [2])
    def test_err_coeffs_shape(self, stages):
        integ = _make_integ(stages)
        assert integ._err_coeffs.shape == (stages,)

    def test_has_embedded_error_s2(self):
        assert _make_integ(2).has_embedded_error is True

    def test_has_embedded_error_s3(self):
        # s=3 uses Richardson extrapolation, NOT embedded error
        assert _make_integ(3).has_embedded_error is False

    def test_invalid_stages_raises(self):
        with pytest.raises(ValueError, match="stages must be 1, 2, or 3"):
            RadauIIA(stages=4)


# ---------------------------------------------------------------------------
# 2. Step contract: 5-tuple with correct shapes
# ---------------------------------------------------------------------------

class TestRadauIIAContract:

    @pytest.mark.parametrize("stages", [1, 2, 3])
    def test_return_shape_scalar(self, stages):
        integ = _make_integ(stages)
        y0 = np.array([1.0])
        out = integ.step(_rhs_decay, 0.0, y0.copy(), 0.1)
        assert isinstance(out, tuple) and len(out) == 5
        y_new, fk, err, ok, iters = out
        assert y_new.shape == y0.shape
        assert np.all(np.isfinite(y_new))
        assert ok is True
        assert iters >= 1

    @pytest.mark.parametrize("stages", [2, 3])  # s=1 delegates to BE (err is scalar)
    def test_return_shape_vector(self, stages):
        integ = _make_integ(stages)
        y0 = np.array([1.0, 2.0, 0.5])
        out = integ.step(_rhs_decay, 0.0, y0.copy(), 0.05)
        y_new, fk, err, ok, iters = out
        assert y_new.shape == y0.shape
        assert np.asarray(err).shape == y0.shape

    @pytest.mark.parametrize("stages", [2])
    def test_embedded_error_finite(self, stages):
        integ = _make_integ(stages)
        y0 = np.array([1.0])
        _, _, err, _, _ = integ.step(_rhs_decay, 0.0, y0.copy(), 0.1)
        assert np.all(np.isfinite(err))
        assert err.shape == y0.shape


# ---------------------------------------------------------------------------
# 3. Convergence order verification
# ---------------------------------------------------------------------------

class TestRadauIIAConvergenceOrder:
    """Verify that the global error scales at the correct power of h."""

    @pytest.mark.parametrize("stages,expected_order", [(1, 1), (2, 3), (3, 5)])
    def test_convergence_order(self, stages, expected_order):
        # Use coupled Newton (analytical Jacobian) so full method order is achieved.
        # Waveform relaxation alone needs wf_maxiter >= expected_order to recover
        # the full order (each sweep adds one order of accuracy in the coupling);
        # coupled Newton achieves this in a single Newton loop.
        y0 = np.array([1.0])
        h_vals = [0.1, 0.05, 0.025]
        errors = []
        integ = _make_integ_with_jac(stages)
        t_end = 1.0
        for h in h_vals:
            Yi = y0.copy()
            t = 0.0
            while t + h <= t_end + 1e-15:
                Yi, _, _, ok, _ = integ.step(_rhs_decay, t, Yi, h)
                assert ok
                t += h
            errors.append(abs(Yi[0] - math.exp(-t_end)))

        # Check convergence at half the step size: order ≈ expected_order
        order_12 = math.log2(errors[0] / errors[1])
        order_23 = math.log2(errors[1] / errors[2])
        tol_order = 0.2 * expected_order  # 20% tolerance on order estimate
        assert abs(order_12 - expected_order) < tol_order, (
            f"stages={stages}: order h[0]→h[1]={order_12:.2f}, expect {expected_order}"
        )
        assert abs(order_23 - expected_order) < tol_order, (
            f"stages={stages}: order h[1]→h[2]={order_23:.2f}, expect {expected_order}"
        )

    @pytest.mark.parametrize("stages,expected_order", [(2, 3), (3, 5)])
    def test_wf_convergence_order(self, stages, expected_order):
        """WF path achieves full order only when wf_maxiter >= expected_order."""
        y0 = np.array([1.0])
        h_vals = [0.1, 0.05, 0.025]
        errors = []
        # Each WF sweep adds one order of coupling accuracy; use enough sweeps.
        integ = _make_integ(stages, wf_maxiter=expected_order + 1)
        t_end = 1.0
        for h in h_vals:
            Yi = y0.copy()
            t = 0.0
            while t + h <= t_end + 1e-15:
                Yi, _, _, ok, _ = integ.step(_rhs_decay, t, Yi, h)
                assert ok
                t += h
            errors.append(abs(Yi[0] - math.exp(-t_end)))

        order_12 = math.log2(errors[0] / errors[1])
        order_23 = math.log2(errors[1] / errors[2])
        tol_order = 0.2 * expected_order
        assert abs(order_12 - expected_order) < tol_order, (
            f"WF stages={stages}: order h[0]→h[1]={order_12:.2f}, expect {expected_order}"
        )
        assert abs(order_23 - expected_order) < tol_order, (
            f"WF stages={stages}: order h[1]→h[2]={order_23:.2f}, expect {expected_order}"
        )

    def test_s1_matches_backward_euler(self):
        """stages=1 must give identical output to BackwardEuler."""
        y0 = np.array([1.0, -0.5])
        h = 0.07
        solver = _make_solver()

        be = BackwardEuler(solver=solver)
        r_be = be.step(_rhs_decay, 0.0, y0.copy(), h)

        solver2 = _make_solver()
        r1 = RadauIIA(stages=1, solver=solver2).step(_rhs_decay, 0.0, y0.copy(), h)

        np.testing.assert_allclose(r1[0], r_be[0], atol=1e-13)


# ---------------------------------------------------------------------------
# 4. Stiff problem stability
# ---------------------------------------------------------------------------

class TestRadauIIAStiff:

    @pytest.mark.parametrize("stages", [2, 3])
    def test_stiff_decay_large_step(self, stages):
        """L-stable methods should damp stiff modes even with h >> 1/λ.

        The waveform-relaxation approximation with finite iterations may introduce
        small oscillations for very stiff steps (λh=25).  We verify that the result
        is bounded (|y| < 1) and has decayed substantially (|y| << 1).
        """
        integ = _make_integ(stages, wf_maxiter=6)   # extra iters for stiff coupling
        y0 = np.array([1.0])
        h = 0.2   # h * 50 = 10  (stiff but not extreme)
        y_new, _, _, ok, _ = integ.step(_rhs_stiff, 0.0, y0.copy(), h)
        assert ok
        # Solution should be substantially damped (exact: exp(-50*0.2) ≈ 4.5e-5)
        assert abs(y_new[0]) < 0.1, f"stiff test: expected |y| < 0.1, got {y_new[0]}"


# ---------------------------------------------------------------------------
# 5. Waveform-relaxation convergence
# ---------------------------------------------------------------------------

class TestWaveformRelaxation:

    def test_wf_converges_to_fixed_point(self):
        """More WF sweeps should converge stage values to the Radau fixed-point.

        WF converges to the exact Radau IIA stage solution, not to the ODE
        solution.  The correct check is that stage-value changes shrink and
        that the output eventually stabilises — not that ODE error monotonically
        decreases (early sweeps can accidentally land closer to the ODE truth).
        """
        y0 = np.array([1.0])
        h = 0.2

        # Capture y_new at successive wf_maxiter values
        results = {}
        for wf in (3, 6, 10):
            integ = _make_integ(2, wf_maxiter=wf)
            y_new, _, _, ok, _ = integ.step(_rhs_decay, 0.0, y0.copy(), h)
            assert ok
            results[wf] = y_new[0]

        # After convergence (wf >= 6), the output should be stable
        assert abs(results[10] - results[6]) < 1e-10, (
            "WF output not stable between wf=6 and wf=10"
        )
        # All results must be physically reasonable (positive, < 1)
        for wf, val in results.items():
            assert 0.0 < val < 1.0, f"wf={wf}: y_new={val} out of range"


# ---------------------------------------------------------------------------
# 6. Adaptive stepping via solve_nivp (end-to-end)
# ---------------------------------------------------------------------------

class TestRadauIIAAdaptive:

    def test_adaptive_s2_scalar_decay(self):
        """s=2 adaptive integration should converge to exp(−t)."""
        t, y, _, _, _ = solve_nivp(
            _rhs_decay, (0.0, 3.0),
            y0=np.array([1.0]),
            method='radau_iia',
            integrator_opts={'stages': 2},
            solver='semismooth_newton',
            projection=None,
            h0=0.1, adaptive=True, atol=1e-8, rtol=1e-6,
        )
        np.testing.assert_allclose(y[-1, 0], math.exp(-3.0), rtol=1e-5)

    def test_adaptive_s3_scalar_decay(self):
        """s=3 with analytical Jacobian (coupled Newton) achieves high accuracy."""
        t, y, _, _, _ = solve_nivp(
            _rhs_decay, (0.0, 3.0),
            y0=np.array([1.0]),
            method='radau_iia',
            integrator_opts={'stages': 3},
            solver='semismooth_newton',
            solver_opts={'rhs_jac': lambda t, y: -np.eye(len(y))},
            projection=None,
            h0=0.1, adaptive=True, atol=1e-9, rtol=1e-7,
        )
        np.testing.assert_allclose(y[-1, 0], math.exp(-3.0), rtol=1e-6)

    def test_s2_fewer_steps_than_sdirk2(self):
        """s=2 should use ≤ SDIRK2 steps for smooth ODE (higher order)."""
        common_kw = dict(
            t_span=(0.0, 5.0),
            y0=np.array([1.0]),
            solver='semismooth_newton',
            projection=None,
            h0=0.1, adaptive=True, atol=1e-8, rtol=1e-6,
        )
        t_s, _, _, _, _ = solve_nivp(_rhs_decay, method='sdirk2', **common_kw)
        t_r, _, _, _, _ = solve_nivp(
            _rhs_decay, method='radau_iia',
            integrator_opts={'stages': 2}, **common_kw
        )
        # Radau IIA s=2 (order 3) should need ≤ steps than SDIRK2 (order 2)
        assert len(t_r) <= len(t_s) * 1.1, (
            f"Radau IIA s=2 ({len(t_r)} steps) > SDIRK2 ({len(t_s)} steps) "
            "by more than 10 %% for smooth ODE — step-size controller may be miscalibrated"
        )

    def test_s3_very_few_steps_high_accuracy(self):
        """s=3 with coupled Newton should take very few steps for smooth ODE."""
        t, y, _, _, _ = solve_nivp(
            _rhs_decay, (0.0, 5.0),
            y0=np.array([1.0]),
            method='radau_iia',
            integrator_opts={'stages': 3},
            solver='semismooth_newton',
            solver_opts={'rhs_jac': lambda t, y: -np.eye(len(y))},
            projection=None,
            h0=0.1, adaptive=True, atol=1e-10, rtol=1e-8,
        )
        # High-order method should solve this with very few steps
        assert len(t) - 1 < 100, f"s=3 took {len(t)-1} steps — expected < 100"
        np.testing.assert_allclose(
            y[-1, 0], math.exp(-5.0), rtol=1e-6,
            err_msg="s=3 final value inaccurate",
        )

    def test_fixed_step_s2(self):
        """Fixed-step s=2 integration over unit interval."""
        t, y, _, _, _ = solve_nivp(
            _rhs_decay, (0.0, 1.0),
            y0=np.array([1.0]),
            method='radau_iia',
            integrator_opts={'stages': 2},
            solver='semismooth_newton',
            projection=None,
            h0=0.1, adaptive=False,
        )
        assert len(t) == 11  # t0 plus 10 steps
        # Order-3 method with h=0.1: error ~ 5e-6, so rtol=1e-4 is appropriate
        np.testing.assert_allclose(y[-1, 0], math.exp(-1.0), rtol=1e-4)

    def test_coupled_newton_large_magnitude_wrms(self):
        """Coupled Newton must use WRMS convergence, not plain L2.

        Regression test for the stacked-system refactor of
        ``_step_coupled_newton_impl``.  The original hand-rolled Newton
        loop tested ``|F| < tol=1e-6`` which was unreachable for
        physics-scale residuals (~10⁹ in poroelastic contact problems).
        Routing the stacked (s·n) system through ``self.solver.solve``
        inherits the solver's weighted-norm convergence so that large-
        magnitude states can converge with realistic per-DOF tolerances.
        """
        scale = 1.0e9

        def rhs(t, y):
            return -y

        def jac(t, y):
            return -np.eye(len(y))

        t, y, _, _, _ = solve_nivp(
            rhs, (0.0, 1.0),
            y0=np.array([scale, scale * 2.0]),
            method='radau_iia',
            integrator_opts={'stages': 2, 'use_coupled_newton': True},
            solver='semismooth_newton',
            solver_opts={'rhs_jac': jac},
            projection=None,
            nl_atol=scale * 1.0e-10,
            nl_rtol=1.0e-8,
            h0=0.1, adaptive=False,
        )
        np.testing.assert_allclose(
            y[-1], [scale * math.exp(-1.0), 2.0 * scale * math.exp(-1.0)],
            rtol=1.0e-4,
            err_msg='Coupled-Newton stacked-system path failed to converge '
                    'for physics-scale RHS magnitudes.',
        )

    def test_coupled_newton_matches_waveform(self):
        """Coupled Newton and waveform relaxation must agree on smooth ODEs.

        Both solve the same Radau IIA stage system, so they should
        converge to the same fixed point within the solver tolerance.
        """
        common = dict(
            t_span=(0.0, 2.0),
            y0=np.array([1.0, 0.5]),
            method='radau_iia',
            solver='semismooth_newton',
            solver_opts={'rhs_jac': lambda t, y: -np.eye(len(y))},
            projection=None,
            h0=0.05, adaptive=False,
        )
        t_c, y_c, _, _, _ = solve_nivp(
            _rhs_decay,
            integrator_opts={'stages': 2, 'use_coupled_newton': True},
            **common,
        )
        t_w, y_w, _, _, _ = solve_nivp(
            _rhs_decay,
            integrator_opts={'stages': 2, 'use_coupled_newton': False},
            **common,
        )
        np.testing.assert_allclose(y_c[-1], y_w[-1], rtol=1.0e-6)


# ---------------------------------------------------------------------------
# 7. Embedded-order attribute is used by the adaptive controller
# ---------------------------------------------------------------------------

class TestEmbeddedOrderControllerCalibration:

    def test_infer_method_order_uses_embedded_order(self):
        """AdaptiveStepping._infer_method_order must return embedded_order for s=2."""
        from solve_nivp.adaptive_integrator import AdaptiveStepping
        integ = _make_integ(2)
        # Build a minimal AdaptiveStepping with no fun/y0/etc.
        # Use only _infer_method_order
        stepper = object.__new__(AdaptiveStepping)
        stepper.integrator = integ
        p = stepper._infer_method_order(integ)
        assert p == integ.embedded_order, (
            f"Expected controller order={integ.embedded_order}, got {p}"
        )
