"""Tests for the SDIRK2 integration method."""

import numpy as np
import pytest

from solve_nivp.integrations import SDIRK2, BackwardEuler
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection
from solve_nivp.adaptive_integrator import AdaptiveStepping


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rhs_decay(t, y):
    """dy/dt = −y  →  y(t) = y0 exp(−t)"""
    return -y


def _rhs_stiff(t, y):
    """dy/dt = −50 y   (stiff scalar ODE)"""
    return -50.0 * y


def _make_solver():
    return ImplicitEquationSolver(
        method='semismooth_newton',
        proj=IdentityProjection(),
        tol=1e-12,
    )


# ---------------------------------------------------------------------------
# 1. Step returns the correct 5-tuple contract
# ---------------------------------------------------------------------------

class TestSDIRK2Contract:
    """Ensure SDIRK2.step() returns the standard 5-tuple."""

    def test_return_shape_scalar(self):
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0])
        out = integ.step(_rhs_decay, 0.0, y0, 0.1)
        assert isinstance(out, tuple)
        assert len(out) == 5

        y_new, fk, err, ok, iters = out
        assert y_new.shape == y0.shape
        assert np.all(np.isfinite(y_new))
        assert ok is True
        assert isinstance(iters, (int, np.integer))
        assert iters >= 2  # at least one iteration per stage

    def test_return_shape_vector(self):
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0, 2.0, 3.0])
        out = integ.step(_rhs_decay, 0.0, y0, 0.05)

        y_new, fk, err, ok, iters = out
        assert y_new.shape == (3,)
        assert err.shape == (3,)
        assert ok is True


# ---------------------------------------------------------------------------
# 2. Order-of-accuracy verification
# ---------------------------------------------------------------------------

class TestSDIRK2Order:
    """Verify that SDIRK2 achieves second-order convergence on a smooth ODE."""

    def test_second_order_convergence_decay(self):
        """Take a single step with halving h and check the error ratio → 4."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0])
        t0 = 0.0

        errors = []
        step_sizes = [0.2, 0.1, 0.05, 0.025]
        for h in step_sizes:
            y_new, _, _, ok, _ = integ.step(_rhs_decay, t0, y0.copy(), h)
            assert ok
            y_exact = y0 * np.exp(-h)
            errors.append(np.abs(y_new[0] - y_exact[0]))

        # Ratio of consecutive errors should approach 4 for order 2
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / errors[i]
            assert ratio > 3.5, f"ratio {ratio:.2f} at h={step_sizes[i]} not consistent with order 2"

    def test_multi_step_accuracy(self):
        """Integrate dy/dt = −y from 0 to 1 with fixed step and check final error."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y = np.array([1.0])
        t = 0.0
        h = 0.01
        t_end = 1.0

        while t < t_end - 1e-14:
            h_eff = min(h, t_end - t)
            y, _, _, ok, _ = integ.step(_rhs_decay, t, y, h_eff)
            assert ok
            t += h_eff

        y_exact = np.exp(-1.0)
        rel_err = np.abs(y[0] - y_exact) / y_exact
        # With h=0.01 and order 2 we expect relative error well below 1e-3
        assert rel_err < 1e-4, f"relative error {rel_err:.3e} too large"


# ---------------------------------------------------------------------------
# 3. Embedded error estimate
# ---------------------------------------------------------------------------

class TestSDIRK2EmbeddedError:
    """Check that the embedded error estimate is meaningful."""

    def test_error_estimate_nonzero(self):
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0])
        _, _, err, ok, _ = integ.step(_rhs_decay, 0.0, y0, 0.1)
        assert ok
        assert np.linalg.norm(err) > 0, "embedded error should be nonzero for a non-trivial ODE"

    def test_error_estimate_scales_with_h(self):
        """Error estimate should roughly halve³ ≈ /8 when h halves (O(h³) LTE for order 2)."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0])

        _, _, err_big, ok1, _ = integ.step(_rhs_decay, 0.0, y0.copy(), 0.2)
        _, _, err_small, ok2, _ = integ.step(_rhs_decay, 0.0, y0.copy(), 0.1)
        assert ok1 and ok2

        # The ratio |err_big|/|err_small| should be near 2^p = 2^2 = 4
        # (the embedded difference is order p, so LTE ~ h^{p+1}, but the
        # difference between order-2 and order-1 solutions is O(h^2)).
        # In practice it's the difference between the two Butcher rows,
        # so err ~ h^2 * C, and ratio ≈ 4.
        ratio = np.linalg.norm(err_big) / max(np.linalg.norm(err_small), 1e-30)
        assert 2.5 < ratio < 6.0, f"error scaling ratio {ratio:.2f} outside expected range"

    def test_error_zero_for_constant_rhs(self):
        """For dy/dt = c (constant), k1 = k2 ⇒ error = 0."""
        def rhs_const(t, y):
            return np.array([3.0])

        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([0.0])
        _, _, err, ok, _ = integ.step(rhs_const, 0.0, y0, 0.1)
        assert ok
        np.testing.assert_allclose(err, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 4. L-stability check (stiff problem)
# ---------------------------------------------------------------------------

class TestSDIRK2LStability:
    """Verify L-stability: |R(z)| → 0 as z → −∞."""

    def test_stiff_damping(self):
        """For dy/dt = −50y, large h*λ should still produce stable and damped solution."""
        solver = _make_solver()
        integ = SDIRK2(solver=solver)
        y0 = np.array([1.0])

        # h=1.0 with λ=−50 gives z = −50, well into the stiff regime
        y_new, _, _, ok, _ = integ.step(_rhs_stiff, 0.0, y0, 1.0)
        assert ok
        # L-stable ⇒ |y_new| should be very small (damped toward zero)
        assert np.abs(y_new[0]) < 0.15, f"|y_new| = {np.abs(y_new[0]):.4f} not damped enough"


# ---------------------------------------------------------------------------
# 5. Method attributes
# ---------------------------------------------------------------------------

class TestSDIRK2Attributes:
    """Check that SDIRK2 exposes the right metadata."""

    def test_order_attribute(self):
        integ = SDIRK2(solver=_make_solver())
        assert integ.order == 2

    def test_gamma_constant(self):
        import math
        assert abs(SDIRK2._GAMMA - (1.0 - math.sqrt(2.0) / 2.0)) < 1e-15


# ---------------------------------------------------------------------------
# 6. Integration via solve_ivp_ns
# ---------------------------------------------------------------------------

class TestSDIRK2HighLevel:
    """Test SDIRK2 via the high-level solve_ivp_ns entry point."""

    def test_solve_ivp_ns_sdirk2(self):
        from solve_nivp import solve_ivp_ns

        t, y, h, fk, info = solve_ivp_ns(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='VI',
            h0=0.05,
        )
        assert t.ndim == 1 and y.ndim == 2
        assert y.shape[1] == 1
        # Check final value is close to exp(−1)
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 1e-2, (
            f"Final value {y[-1, 0]:.6f} not close to exp(-1)={y_exact:.6f}"
        )

    def test_solve_ivp_ns_sdirk2_ssn(self):
        """Also works with semismooth_newton solver."""
        from solve_nivp import solve_ivp_ns

        t, y, h, fk, info = solve_ivp_ns(
            fun=_rhs_decay,
            t_span=(0.0, 0.5),
            y0=np.array([1.0, 2.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0=0.02,
        )
        y_exact = np.array([1.0, 2.0]) * np.exp(-0.5)
        np.testing.assert_allclose(y[-1], y_exact, rtol=1e-2)


# ---------------------------------------------------------------------------
# 7. Comparison with BackwardEuler (SDIRK2 should be more accurate)
# ---------------------------------------------------------------------------

class TestSDIRK2VsBackwardEuler:
    """SDIRK2 with order 2 should be strictly more accurate than BE (order 1)
    for the same step size on a smooth problem."""

    def test_more_accurate_than_be(self):
        solver_sdirk = _make_solver()
        solver_be = _make_solver()
        sdirk = SDIRK2(solver=solver_sdirk)
        be = BackwardEuler(solver=solver_be)

        y0 = np.array([1.0])
        h = 0.1
        y_exact = np.exp(-h)

        y_sdirk, _, _, ok_s, _ = sdirk.step(_rhs_decay, 0.0, y0.copy(), h)
        y_be, _, _, ok_b, _ = be.step(_rhs_decay, 0.0, y0.copy(), h)
        assert ok_s and ok_b

        err_sdirk = np.abs(y_sdirk[0] - y_exact)
        err_be = np.abs(y_be[0] - y_exact)
        assert err_sdirk < err_be, (
            f"SDIRK2 error {err_sdirk:.3e} should be smaller than BE error {err_be:.3e}"
        )


# ---------------------------------------------------------------------------
# 8. Mass-matrix (A ≠ I) correctness
# ---------------------------------------------------------------------------

class TestSDIRK2MassMatrix:
    """Verify SDIRK2 works correctly when A is a non-identity mass matrix.

    This tests the stage-2 shift and error estimate, which previously had a
    bug: they used k1 = f(t, Y1) instead of the stage derivative
    K1 = (Y1 - y)/(γh).  The two are only equal when A = I.
    """

    def test_scaled_mass_matrix_accuracy(self):
        """M dy/dt = f(t,y) with M = 2I, f = -2y  ⇒  dy/dt = -y  ⇒  y = e^{-t}.

        If the shift were computed as y + (1-γ)h·f  instead of
        y + ((1-γ)/γ)(Y1-y), the factor-of-M error would corrupt the solution.
        """
        M = 2.0 * np.eye(2)

        def rhs_scaled(t, y):
            return -2.0 * y  # M dy/dt = -2y  ⇒  dy/dt = -y

        solver = _make_solver()
        integ = SDIRK2(solver=solver, A=M)

        y0 = np.array([1.0, 3.0])
        h = 0.05
        y_new, _, err, ok, _ = integ.step(rhs_scaled, 0.0, y0, h)
        assert ok

        y_exact = y0 * np.exp(-h)
        np.testing.assert_allclose(y_new, y_exact, rtol=1e-3,
                                   err_msg="SDIRK2 with mass matrix M=2I is inaccurate")

        # Error estimate should also be finite and reasonable
        assert np.all(np.isfinite(err))
        assert np.linalg.norm(err) > 0

    def test_mass_matrix_convergence_order(self):
        """Verify second-order convergence with a non-trivial mass matrix."""
        M = np.array([[2.0, 0.5], [0.5, 3.0]])

        # M dy/dt = -M y  ⇒  dy/dt = -y  ⇒  y = y0 e^{-t}
        def rhs_M(t, y):
            return -M @ y

        errors = []
        step_sizes = [0.1, 0.05, 0.025]
        y0 = np.array([1.0, 2.0])

        for h in step_sizes:
            solver = _make_solver()
            integ = SDIRK2(solver=solver, A=M)
            y_new, _, _, ok, _ = integ.step(rhs_M, 0.0, y0.copy(), h)
            assert ok, f"SDIRK2 failed to converge at h={h}"
            y_exact = y0 * np.exp(-h)
            errors.append(np.linalg.norm(y_new - y_exact))

        # Check second-order: error ratio ≈ 4 when h halves
        for i in range(1, len(errors)):
            ratio = errors[i - 1] / errors[i]
            assert ratio > 3.0, (
                f"ratio {ratio:.2f} at h={step_sizes[i]} not consistent with order 2 "
                f"(errors: {errors})"
            )

    def test_mass_matrix_multi_step(self):
        """Integrate with M=diag(3,3) over multiple steps and check accuracy."""
        M = 3.0 * np.eye(2)

        def rhs_3(t, y):
            return -3.0 * y  # dy/dt = -y

        solver = _make_solver()
        integ = SDIRK2(solver=solver, A=M)
        y = np.array([1.0, 2.0])
        t = 0.0
        h = 0.02
        t_end = 0.5

        while t < t_end - 1e-14:
            h_eff = min(h, t_end - t)
            y, _, _, ok, _ = integ.step(rhs_3, t, y, h_eff)
            assert ok
            t += h_eff

        y_exact = np.array([1.0, 2.0]) * np.exp(-0.5)
        np.testing.assert_allclose(y, y_exact, rtol=1e-3)


# ---------------------------------------------------------------------------
# 9. Embedded error bypass in AdaptiveStepping
# ---------------------------------------------------------------------------

class TestSDIRK2EmbeddedAdaptive:
    """Verify that AdaptiveStepping uses the embedded error path for SDIRK2
    (no Richardson step-doubling), and that BackwardEuler still uses Richardson.
    """

    def test_has_embedded_error_flag(self):
        integ = SDIRK2(solver=_make_solver())
        assert getattr(integ, 'has_embedded_error', False) is True

    def test_be_no_embedded_flag(self):
        integ = BackwardEuler(solver=_make_solver())
        assert getattr(integ, 'has_embedded_error', False) is False

    def test_sdirk2_fewer_rhs_calls_than_be(self):
        """SDIRK2+adaptive should call integrator.step() once per attempt,
        while BE+adaptive calls it three times (Richardson).

        We instrument the step count to verify."""
        call_counts = {'sdirk2': 0, 'be': 0}

        # Wrap SDIRK2
        solver_s = _make_solver()
        sdirk = SDIRK2(solver=solver_s)
        orig_step_s = sdirk.step

        def counted_step_s(*args, **kwargs):
            call_counts['sdirk2'] += 1
            return orig_step_s(*args, **kwargs)
        sdirk.step = counted_step_s

        ctrl_s = AdaptiveStepping(
            integrator=sdirk, atol=1e-6, rtol=1e-3,
            h0=0.1, h_min=1e-8, mode='classic',
        )
        y0 = np.array([1.0])
        ctrl_s.step(_rhs_decay, 0.0, y0.copy(), 0.1)
        sdirk2_calls = call_counts['sdirk2']

        # Wrap BackwardEuler
        solver_b = _make_solver()
        be = BackwardEuler(solver=solver_b)
        orig_step_b = be.step

        def counted_step_b(*args, **kwargs):
            call_counts['be'] += 1
            return orig_step_b(*args, **kwargs)
        be.step = counted_step_b

        ctrl_b = AdaptiveStepping(
            integrator=be, atol=1e-6, rtol=1e-3,
            h0=0.1, h_min=1e-8, mode='classic',
        )
        ctrl_b.step(_rhs_decay, 0.0, y0.copy(), 0.1)
        be_calls = call_counts['be']

        # SDIRK2 should use 1 call; BE should use 3 (full + 2 halves)
        assert sdirk2_calls == 1, f"SDIRK2 made {sdirk2_calls} step calls, expected 1"
        assert be_calls == 3, f"BE made {be_calls} step calls, expected 3"

    def test_sdirk2_adaptive_produces_correct_result(self):
        """Full adaptive integration with SDIRK2 should still converge."""
        from solve_nivp import solve_ivp_ns

        t, y, h, fk, info = solve_ivp_ns(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0=0.1,
            adaptive=True,
            rtol=1e-4,
            atol=1e-6,
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 1e-3, (
            f"SDIRK2 adaptive: y_final={y[-1,0]:.6f}, expected {y_exact:.6f}"
        )

    def test_sdirk2_adaptive_ratio_mode(self):
        """Embedded path also works with ratio/digital-filter controller."""
        from solve_nivp import solve_ivp_ns

        t, y, h, fk, info = solve_ivp_ns(
            fun=_rhs_decay,
            t_span=(0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0=0.1,
            adaptive=True,
            adaptive_opts={'controller': 'h211b', 'method_order': 2},
            rtol=1e-3,
            atol=1e-6,
        )
        y_exact = np.exp(-1.0)
        assert np.abs(y[-1, 0] - y_exact) < 1e-2


# ---------------------------------------------------------------------------
# 8. SPLU factorisation reuse across SDIRK2 stages
# ---------------------------------------------------------------------------


class TestSPLUReuse:
    """Verify that SDIRK2's two stages share one SPLU factorisation."""

    def test_splu_reuse_constant_jacobian(self):
        """For a linear ODE with analytical Jacobian both SDIRK2 stages
        should share one SPLU factorisation per step, not two.

        We instrument scipy.sparse.linalg.splu to count calls.
        """
        import scipy.sparse as sp_mod
        import scipy.sparse.linalg as spla_mod
        from unittest.mock import patch

        n = 50
        # A = -diag(1..n)  (constant Jacobian)
        lam = -np.arange(1, n + 1, dtype=float)
        A_sparse = sp_mod.diags(lam).tocsc()

        def rhs(t, y, *_a, **_kw):
            return A_sparse @ y

        def rhs_jac(t, y, *_a, **_kw):
            return A_sparse

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-12,
            sparse=True,          # force sparse path to exercise SPLU caching
            linear_solver='splu', # explicitly select SPLU path
        )
        solver.rhs_jacobian = rhs_jac
        integ = SDIRK2(solver=solver)

        y0 = np.ones(n)
        h = 0.01

        # Reset solver's cached LU so we count from a clean state
        solver._lu = None
        solver._lu_shape = None

        # Count splu calls
        real_splu = spla_mod.splu
        call_count = [0]

        def counting_splu(*args, **kwargs):
            call_count[0] += 1
            return real_splu(*args, **kwargs)

        with patch.object(spla_mod, 'splu', side_effect=counting_splu):
            # First step: stage 1 must factorise, stage 2 should reuse → 1 call
            y1, fk1, err1, ok1, it1 = integ.step(rhs, 0.0, y0, h)
            first_step_calls = call_count[0]

            # Second step with same h: should reuse from previous step → 0 calls
            y2, fk2, err2, ok2, it2 = integ.step(rhs, h, y1, h)
            second_step_calls = call_count[0] - first_step_calls

        assert ok1 and ok2, "Both steps should converge"
        # Stage 1 of first step factorises (1), stage 2 reuses (0) → total 1
        assert first_step_calls == 1, (
            f"First step should need exactly 1 SPLU (stage 1 factorises, stage 2 reuses), "
            f"got {first_step_calls}"
        )
        # Both stages of second step reuse the factorisation → 0
        assert second_step_calls == 0, (
            f"Second step (same h, constant J) should reuse prior factorisation, "
            f"got {second_step_calls} new factorisations"
        )


class TestSDIRK2StaleLUInvalidation:
    """When h changes between steps, the cached SPLU must be invalidated.

    The iteration matrix  M/(γh)−J  depends on h; if a stale factorisation
    from a previous step is reused, Newton's first iterate is corrupted and
    may cause convergence failure (the classic symptom: every step shows a
    'nonlinear fail → shrink → accept' pattern).
    """

    def test_different_h_invalidates_lu(self):
        """After a step with h₁, a step with h₂≠h₁ must NOT reuse the old LU."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla_mod
        from unittest.mock import patch

        n = 20
        A_mat = sp.diags([-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)],
                         [0, 1, -1], format='csr') * 50.0
        M_mat = sp.eye(n, format='csr')

        def rhs(t, y):
            return A_mat @ y

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-12,
            max_iter=5,
            linear_solver='splu',
            sparse=True,        # force sparse path for small n
        )
        solver.rhs_jacobian = lambda t, y, *a, **kw: A_mat
        integ = SDIRK2(solver=solver, A=M_mat)

        y0 = np.random.RandomState(42).randn(n)

        # Step 1 with h₁ — builds and caches LU
        h1 = 0.01
        y1, _, _, ok1, it1 = integ.step(rhs, 0.0, y0, h1)
        assert ok1, "Step 1 should converge"
        assert solver._lu is not None, "LU should be cached after step 1"

        # Step 2 with h₂ ≠ h₁ — LU must be invalidated, new factorisation
        h2 = 0.02
        real_splu = spla_mod.splu
        splu_calls = [0]

        def counting_splu(*args, **kwargs):
            splu_calls[0] += 1
            return real_splu(*args, **kwargs)

        with patch.object(spla_mod, 'splu', side_effect=counting_splu):
            y2, _, _, ok2, it2 = integ.step(rhs, h1, y1, h2)

        assert ok2, "Step with different h should still converge"
        assert splu_calls[0] >= 1, (
            "A step with a different h must trigger at least one new SPLU factorisation"
        )
        # With the fix, Newton should converge in ≤ 3 iterations (linear system)
        assert it2 <= 4, (
            f"Linear system should converge quickly with fresh LU, got {it2} iterations"
        )

    def test_same_h_still_reuses_lu(self):
        """Consecutive steps with the same h should still reuse the cached LU."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla_mod
        from unittest.mock import patch

        n = 20
        A_mat = sp.diags([-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)],
                         [0, 1, -1], format='csr') * 50.0
        M_mat = sp.eye(n, format='csr')

        def rhs(t, y):
            return A_mat @ y

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-12,
            max_iter=5,
            linear_solver='splu',
            sparse=True,        # force sparse path for small n
        )
        solver.rhs_jacobian = lambda t, y, *a, **kw: A_mat
        integ = SDIRK2(solver=solver, A=M_mat)

        y0 = np.random.RandomState(42).randn(n)
        h = 0.01

        # Step 1 — caches LU
        y1, _, _, ok1, _ = integ.step(rhs, 0.0, y0, h)
        assert ok1

        # Step 2 (same h) — count SPLU calls, should be 0
        real_splu = spla_mod.splu
        splu_calls = [0]

        def counting_splu(*args, **kwargs):
            splu_calls[0] += 1
            return real_splu(*args, **kwargs)

        with patch.object(spla_mod, 'splu', side_effect=counting_splu):
            y2, _, _, ok2, _ = integ.step(rhs, h, y1, h)

        assert ok2
        assert splu_calls[0] == 0, (
            f"Same h should fully reuse the cached LU, got {splu_calls[0]} new factorisations"
        )

    def test_backward_euler_invalidates_lu_on_h_change(self):
        """BackwardEuler should also invalidate the cached LU when h changes."""
        import scipy.sparse as sp

        n = 20
        A_mat = sp.diags([-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)],
                         [0, 1, -1], format='csr') * 50.0
        M_mat = sp.eye(n, format='csr')

        def rhs(t, y):
            return A_mat @ y

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-12,
            max_iter=3,
            linear_solver='splu',
            sparse=True,        # force sparse path for small n
        )
        solver.rhs_jacobian = lambda t, y, *a, **kw: A_mat
        integ = BackwardEuler(solver=solver, A=M_mat)

        y0 = np.random.RandomState(42).randn(n)

        # Step 1 with h₁
        y1, _, _, ok1, it1 = integ.step(rhs, 0.0, y0, 0.01)
        assert ok1, "BE step 1 should converge"

        # Step 2 with h₂ ≠ h₁ — must converge quickly (max_iter=3 is enough for linear)
        y2, _, _, ok2, it2 = integ.step(rhs, 0.01, y1, 0.02)
        assert ok2, (
            "BE step with different h should converge (LU invalidated, fresh factorisation)"
        )

    def test_sdirk2_adaptive_no_repeated_nonlinear_failures(self):
        """An adaptive SDIRK2 run on a linear ODE should NOT show nonlinear
        failures when h changes, because the LU is invalidated."""
        import scipy.sparse as sp

        n = 10
        A_mat = sp.diags([-2 * np.ones(n), np.ones(n - 1), np.ones(n - 1)],
                         [0, 1, -1], format='csr') * 20.0
        M_mat = sp.eye(n, format='csr')

        def rhs(t, y):
            return A_mat @ y

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            max_iter=5,
            linear_solver='splu',
            sparse=True,        # force sparse path for small n
        )
        solver.rhs_jacobian = lambda t, y, *a, **kw: A_mat
        integ = SDIRK2(solver=solver, A=M_mat)

        y0 = np.ones(n)
        h = 0.01

        # Run several steps with varying h — none should fail
        t = 0.0
        y = y0.copy()
        h_values = [0.01, 0.02, 0.005, 0.03, 0.01, 0.04]
        for h_step in h_values:
            y_new, _, _, ok, it = integ.step(rhs, t, y, h_step)
            assert ok, (
                f"Linear system with h={h_step} should converge, "
                f"but failed after {it} iterations"
            )
            t += h_step
            y = y_new
