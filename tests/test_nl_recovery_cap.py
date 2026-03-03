"""Tests for the nonlinear-failure recovery cap in AdaptiveStepping.

The cap prevents the "death spiral" where every proposed h triggers a
nonlinear solve failure, the retry at h*h_down succeeds with tiny error,
the PI controller grows h, and the next attempt fails again — so h decays
monotonically to h_min.
"""
import numpy as np
import pytest
from solve_nivp import solve_ivp_ns
from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp.integrations import SDIRK2


# ---------------------------------------------------------------------------
# Helper: an integrator that *always* fails when h > threshold, succeeds
# otherwise.  This reproduces the coarse-mesh death-spiral pattern.
# ---------------------------------------------------------------------------
class _FakeIntegratorAlwaysFails:
    """Mock integrator that fails the NL solve when h exceeds a threshold."""

    has_embedded_error = True
    order = 2

    def __init__(self, h_threshold=0.02):
        self.h_threshold = h_threshold
        self.call_count = 0

    def step(self, fun, t, y, h):
        self.call_count += 1
        if h > self.h_threshold:
            # NL failure
            return y, None, np.zeros_like(y), False, 10
        # Success with small embedded error
        y_new = y + h * fun(t, y)
        err = np.full_like(y, 1e-8)  # very small error
        return y_new, fun(t + h, y_new), err, True, 3


class TestNLRecoveryCapUnit:
    """Unit tests for the recovery-cap state machine."""

    def test_cap_activates_after_persistent_failures(self):
        """After _NL_PERSIST_THRESH fail→succeed pairs, h_next is capped."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=0.02)
        stepper = AdaptiveStepping(
            integrator,
            atol=1e-6, rtol=1e-3,
            h0=0.05,
            h_min=1e-12, h_max=0.1,
            h_up=2.0, h_down=0.5,
            verbose=False,
        )
        fun = lambda t, y: -y

        y = np.array([1.0])
        h = 0.05
        t = 0.0

        h_values = []
        for _ in range(20):
            y_new, fk, h_next, E, success, se, it = stepper.step(fun, t, y, h)
            h_values.append(h)
            if success:
                t += h
                y = y_new
            h = h_next

        # Without the cap, h would decay geometrically to h_min.
        # With the cap, h should stabilise near h_threshold * h_down.
        # Check that h never reaches an extremely small value.
        assert min(h_values) > 1e-4, (
            f"Step size decayed to {min(h_values):.2e}; "
            f"expected stabilisation near {integrator.h_threshold * stepper.h_down:.2e}"
        )

    def test_cap_relaxes_after_clean_successes(self):
        """After enough successes without NL failure, the cap is removed."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=0.3)  # generous
        stepper = AdaptiveStepping(
            integrator,
            atol=1e-6, rtol=1e-3,
            h0=0.05,
            h_min=1e-12, h_max=0.5,
            h_up=3.0, h_down=0.5,
            verbose=False,
        )
        fun = lambda t, y: -y
        y = np.array([1.0])
        t = 0.0
        h = 0.05

        # Force 4 fail→succeed pairs to activate cap
        for _ in range(10):
            y_new, fk, h_next, E, success, se, it = stepper.step(fun, t, y, h)
            if success:
                t += h
                y = y_new
            h = h_next

        # Now disable the threshold so all steps succeed
        integrator.h_threshold = 100.0

        # Run enough successful steps to relax the cap
        for _ in range(30):
            y_new, fk, h_next, E, success, se, it = stepper.step(fun, t, y, h)
            if success:
                t += h
                y = y_new
            h = h_next

        # After relaxation, the cap counter should be 0
        assert stepper._nl_fail_recovery_count == 0, (
            f"Expected cap to be fully relaxed, "
            f"but _nl_fail_recovery_count={stepper._nl_fail_recovery_count}"
        )

    def test_no_cap_when_no_failures(self):
        """Normal operation without NL failures: no cap interference."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=100.0)  # never fails
        stepper = AdaptiveStepping(
            integrator,
            atol=1e-6, rtol=1e-3,
            h0=0.01,
            h_min=1e-12, h_max=0.2,
            h_up=2.0, h_down=0.5,
            verbose=False,
        )
        fun = lambda t, y: -y
        y = np.array([1.0])
        t = 0.0
        h = 0.01

        for _ in range(10):
            y_new, fk, h_next, E, success, se, it = stepper.step(fun, t, y, h)
            assert success
            t += h
            y = y_new
            h = h_next

        assert stepper._nl_fail_recovery_count == 0
        assert stepper._nl_success_no_fail > 0


class TestNLRecoveryIntegration:
    """Integration tests using solve_ivp_ns."""

    def test_stiff_scalar_completes(self):
        """A simple stiff ODE should integrate to completion."""
        lam = -100.0
        t, y, h, fk, info = solve_ivp_ns(
            lambda t, y: lam * y,
            (0.0, 1.0),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            adaptive=True,
            atol=1e-4, rtol=1e-2,
            verbose=False,
        )
        assert t[-1] == pytest.approx(1.0, abs=1e-8)
        # exp(-100) ≈ 3.7e-44, numerically indistinguishable from 0
        assert abs(y[-1, 0]) < 1e-4


class TestThinOutputAndMemory:
    """Test thin_output, store_fk, gc_interval options."""

    def _make_system(self):
        from solve_nivp import ODESystem
        from solve_nivp.integrations import BackwardEuler
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        from solve_nivp.projections import IdentityProjection

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
        )
        integrator = BackwardEuler(solver=solver)
        return ODESystem(
            fun=lambda t, y: -y,
            y0=np.array([1.0]),
            method=integrator,
            adaptive=True,
            atol=1e-4, rtol=1e-2,
        )

    def test_thin_output_reduces_stored_steps(self):
        """With thin_output=5, many fewer snapshots are stored."""
        from solve_nivp import ODESolver
        sys1 = self._make_system()
        solver_all = ODESolver(sys1, (0.0, 1.0), h=0.01, thin_output=1)
        t_all, y_all, *_ = solver_all.solve()

        sys2 = self._make_system()
        solver_thin = ODESolver(sys2, (0.0, 1.0), h=0.01, thin_output=5)
        t_thin, y_thin, *_ = solver_thin.solve()

        # Thin output should have strictly fewer entries
        assert len(t_thin) < len(t_all)
        # Both should reach the end
        assert t_thin[-1] == pytest.approx(1.0, abs=1e-6)
        assert t_all[-1] == pytest.approx(1.0, abs=1e-6)

    def test_store_fk_false(self):
        """When store_fk=False, fk entries are None."""
        from solve_nivp import ODESolver
        sys = self._make_system()
        solver = ODESolver(sys, (0.0, 1.0), h=0.01, store_fk=False)
        _, _, _, fk, _ = solver.solve()
        for entry in fk:
            assert entry is None

    def test_gc_interval_does_not_crash(self):
        """gc_interval>0 should not cause errors."""
        from solve_nivp import ODESolver
        sys = self._make_system()
        solver = ODESolver(sys, (0.0, 1.0), h=0.01, gc_interval=3)
        t, y, *_ = solver.solve()
        assert t[-1] == pytest.approx(1.0, abs=1e-6)

    def test_thin_output_via_solve_ivp_ns(self):
        """thin_output parameter works through the high-level API."""
        t, y, h, fk, info = solve_ivp_ns(
            lambda t, y: -y,
            (0.0, 1.0),
            y0=np.array([1.0]),
            method='backward_euler',
            projection='identity',
            thin_output=10,
        )
        # Should reach end
        assert t[-1] == pytest.approx(1.0, abs=1e-6)
        # Should have far fewer than 100 stored steps
        assert len(t) < 20


# ---------------------------------------------------------------------------
# Tests for consecutive NL failure tracking + solver rescue mechanism
# ---------------------------------------------------------------------------

class TestConsecutiveNLFailureRescue:
    """Tests for the stuck-failure rescue that destroys all solver caches."""

    def test_consecutive_counter_increments(self):
        """_consecutive_nl_fails increments on each failure step."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=0.0)  # fails always
        stepper = AdaptiveStepping(integrator, h_min=1e-6, h_max=1.0)
        stepper.h = 0.1

        fun = lambda t, y: -y
        y = np.array([1.0])
        # Each call should fail NL solve and increment the counter
        stepper.step(fun, 0.0, y, 0.1)
        assert stepper._consecutive_nl_fails == 1
        stepper.step(fun, 0.0, y, 0.05)
        assert stepper._consecutive_nl_fails == 2

    def test_consecutive_counter_resets_on_success(self):
        """Counter resets to 0 after a successful step."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=0.05)
        stepper = AdaptiveStepping(integrator, h_min=1e-6, h_max=1.0)
        stepper.h = 0.1

        fun = lambda t, y: -y
        y = np.array([1.0])
        # Fail with large h
        stepper.step(fun, 0.0, y, 0.1)
        assert stepper._consecutive_nl_fails == 1
        # Succeed with small h
        stepper.step(fun, 0.0, y, 0.01)
        assert stepper._consecutive_nl_fails == 0

    def test_rescue_threshold_is_configurable(self):
        """_NL_RESCUE_THRESH defaults to 5 and can be overridden."""
        integrator = _FakeIntegratorAlwaysFails(h_threshold=0.05)
        stepper = AdaptiveStepping(integrator, h_min=1e-6, h_max=1.0)
        assert stepper._NL_RESCUE_THRESH == 5
        stepper._NL_RESCUE_THRESH = 3


class TestJacobianScalingPassthrough:
    """Test that jacobian_scaling flows properly from solver_opts."""

    def test_scaling_from_solver_opts(self):
        """jacobian_scaling in solver_opts is forwarded when top-level is None."""
        t, y, h, fk, info = solve_ivp_ns(
            lambda t, y: -y,
            (0.0, 0.5),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            solver_opts={'jacobian_scaling': 'row'},
            h0=0.1,
        )
        np.testing.assert_allclose(y[-1, 0], np.exp(-0.5), rtol=0.05)

    def test_explicit_scaling_wins(self):
        """Explicit jacobian_scaling kwarg wins over solver_opts value."""
        t, y, h, fk, info = solve_ivp_ns(
            lambda t, y: -y,
            (0.0, 0.5),
            y0=np.array([1.0]),
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            solver_opts={'jacobian_scaling': 'ruiz'},
            jacobian_scaling='row',
            h0=0.1,
        )
        np.testing.assert_allclose(y[-1, 0], np.exp(-0.5), rtol=0.05)


class TestInvalidateAllCaches:
    """Tests for ImplicitEquationSolver.invalidate_all_caches()."""

    def test_invalidate_clears_lu(self):
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        from solve_nivp.projections import IdentityProjection

        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), linear_solver='splu')
        # Simulate cached state
        solver._lu = "some_factorisation"
        solver._lu_shape = (10, 10)
        solver._J_cross_call = "some_jacobian"
        solver._ilu = "some_ilu"
        solver._eq_Dr = np.ones(10)
        solver._eq_Dc = np.ones(10)

        solver.invalidate_all_caches()

        assert solver._lu is None
        assert solver._lu_shape is None
        assert solver._J_cross_call is None
        assert solver._ilu is None
        assert solver._eq_Dr is None
        assert solver._eq_Dc is None
