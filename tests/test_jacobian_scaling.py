"""Tests for Jacobian equilibration (jacobian_scaling parameter)."""

import numpy as np
import pytest
import scipy.sparse as sp

from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection, AlgebraicConstraintProjection
from solve_nivp import solve_ivp_ns


# ---------------------------------------------------------------------------
# Helper: poorly-conditioned linear system (disparate row scales)
# ---------------------------------------------------------------------------

def _make_poorly_conditioned_system():
    """Create an F(y) whose Jacobian has very disparate row scales.

    System:       y0 - 1   = 0
            1e8 * y1 - 1e8 = 0

    True solution: y = [1, 1].
    Jacobian: diag([1, 1e8]) — condition number 1e8.
    Row-equilibrated: diag([1, 1]) — condition number 1.
    """
    def F(y):
        return np.array([y[0] - 1.0, 1e8 * (y[1] - 1.0)])

    return F, np.array([0.0, 0.0]), np.array([1.0, 1.0])


# ---------------------------------------------------------------------------
# Unit tests for _equilibrate
# ---------------------------------------------------------------------------

class TestEquilibrateRow:
    """Row equilibration normalises row infinity-norms to 1."""

    def test_row_scaling_normalises_rows(self):
        """After row equilibration every row has ||·||_inf == 1."""
        J = sp.csr_matrix(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1e6, 2e6],
            [0.0, 0.0, 1e-3],
        ]))
        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='row')
        J_eq, Dr, Dc = solver._equilibrate(J)

        abs_eq = J_eq.copy()
        abs_eq.data = np.abs(abs_eq.data)
        _tmp = abs_eq.max(axis=1)
        row_max = np.asarray(_tmp.toarray() if sp.issparse(_tmp) else _tmp).ravel()
        np.testing.assert_allclose(row_max, 1.0, atol=1e-12)

        # Column scaling should be identity for 'row' mode
        np.testing.assert_array_equal(Dc, np.ones(3))

    def test_row_scaling_preserves_zero_rows(self):
        """Zero rows (e.g. algebraic DOFs before patching) are handled."""
        J = sp.csr_matrix(np.array([
            [2.0, 1.0],
            [0.0, 0.0],
        ]))
        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='row')
        J_eq, Dr, Dc = solver._equilibrate(J)
        # Zero row should remain zero; Dr for that row should be 1.0
        assert Dr[1] == 1.0
        row1 = J_eq[1, :].toarray().ravel()
        np.testing.assert_array_equal(row1, [0.0, 0.0])


class TestEquilibrateRuiz:
    """Ruiz iterative symmetric scaling balances rows AND columns."""

    def test_ruiz_normalises_both(self):
        """After Ruiz scaling, row and column norms are approximately 1."""
        J = sp.csr_matrix(np.array([
            [1e6, 0.0],
            [0.0, 1e-4],
        ]))
        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='ruiz')
        J_eq, Dr, Dc = solver._equilibrate(J)

        abs_eq = J_eq.copy()
        abs_eq.data = np.abs(abs_eq.data)

        def _dr(m):
            return np.asarray(m.toarray() if sp.issparse(m) else m).ravel()

        row_max = _dr(abs_eq.max(axis=1))
        col_max = _dr(abs_eq.max(axis=0))
        np.testing.assert_allclose(row_max, 1.0, atol=0.05)
        np.testing.assert_allclose(col_max, 1.0, atol=0.05)

    def test_ruiz_solution_recovery(self):
        """Ruiz column scaling correctly recovers the Newton step."""
        J = sp.csr_matrix(np.array([
            [1e6, 0.0],
            [0.0, 1e-4],
        ]))
        rhs = np.array([2e6, 3e-4])
        x_exact = np.linalg.solve(J.toarray(), rhs)

        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='ruiz')
        J_eq, Dr, Dc = solver._equilibrate(J)
        rhs_eq = Dr * rhs
        x_eq = np.linalg.solve(J_eq.toarray(), rhs_eq)
        x_recovered = Dc * x_eq

        np.testing.assert_allclose(x_recovered, x_exact, rtol=1e-10)


# ---------------------------------------------------------------------------
# Integration tests: Newton convergence with scaling
# ---------------------------------------------------------------------------

class TestNewtonWithScaling:
    """End-to-end Newton solve with jacobian_scaling enabled."""

    @pytest.mark.parametrize('mode', ['row', 'ruiz'])
    def test_identity_newton_with_scaling(self, mode):
        """Standard Newton (IdentityProjection) converges with scaling."""
        F, y0, y_exact = _make_poorly_conditioned_system()
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=IdentityProjection(),
            tol=1e-10,
            max_iter=50,
            jacobian_scaling=mode,
        )
        y_sol, F_sol, errF, converged, iters = solver.solve(F, y0)
        assert converged, f"Newton did not converge in {iters} iterations"
        np.testing.assert_allclose(y_sol, y_exact, atol=1e-8)

    @pytest.mark.parametrize('mode', ['row', 'ruiz'])
    def test_algebraic_newton_with_scaling(self, mode):
        """Newton with AlgebraicConstraintProjection converges with scaling."""
        # System: y0 solves y0 - 2 = 0 (field eq)
        #         q0 = 1e6 * y0         (algebraic constraint with large scale)
        C = np.array([[1e6]])
        proj = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=slice(0, 1),
            q_slice=slice(1, 2),
        )

        def F(y):
            return np.array([y[0] - 2.0, 0.0])

        y0 = np.array([0.0, 0.0])
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            tol=1e-10,
            max_iter=50,
            jacobian_scaling=mode,
        )
        y_sol, _, _, converged, _ = solver.solve(F, y0)
        assert converged
        np.testing.assert_allclose(y_sol[0], 2.0, atol=1e-8)
        np.testing.assert_allclose(y_sol[1], 1e6 * 2.0, atol=1e-4)

    def test_none_scaling_is_default_behaviour(self):
        """jacobian_scaling='none' produces same results as baseline."""
        F, y0, y_exact = _make_poorly_conditioned_system()
        for mode in ('none',):
            solver = ImplicitEquationSolver(
                method='semismooth_newton',
                proj=IdentityProjection(),
                tol=1e-10,
                max_iter=50,
                jacobian_scaling=mode,
            )
            y_sol, _, _, converged, _ = solver.solve(F, y0)
            assert converged
            np.testing.assert_allclose(y_sol, y_exact, atol=1e-8)


# ---------------------------------------------------------------------------
# solve_ivp_ns integration
# ---------------------------------------------------------------------------

class TestSolveIvpNsScaling:
    """jacobian_scaling parameter flows through solve_ivp_ns."""

    @pytest.mark.parametrize('mode', ['none', 'row', 'ruiz'])
    def test_simple_ode_with_scaling(self, mode):
        """dy/dt = -y solves correctly with each scaling mode."""
        t, y, h, fk, info = solve_ivp_ns(
            fun=lambda t, y: -y,
            t_span=(0, 1),
            y0=[1.0],
            method='sdirk2',
            projection='identity',
            solver='semismooth_newton',
            h0=0.1,
            atol=1e-6,
            rtol=1e-3,
            jacobian_scaling=mode,
        )
        # Expect y(1) ≈ exp(-1) ≈ 0.3679
        np.testing.assert_allclose(y[-1, 0], np.exp(-1), rtol=1e-2)


# ---------------------------------------------------------------------------
# Edge-case / validation
# ---------------------------------------------------------------------------

class TestScalingValidation:
    """Parameter validation for jacobian_scaling."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="jacobian_scaling"):
            ImplicitEquationSolver(
                proj=IdentityProjection(),
                jacobian_scaling='invalid',
            )

    def test_scale_rhs_noop_when_none(self):
        """_scale_rhs is a no-op when scaling is 'none'."""
        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='none')
        rhs = np.array([1.0, 2.0, 3.0])
        assert solver._scale_rhs(rhs) is rhs  # same object, not a copy

    def test_unscale_noop_for_row_mode(self):
        """_unscale_solution is a no-op for 'row' mode."""
        solver = ImplicitEquationSolver(
            proj=IdentityProjection(), jacobian_scaling='row')
        x = np.array([1.0, 2.0])
        assert solver._unscale_solution(x) is x
