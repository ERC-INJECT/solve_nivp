"""Tests for the block-structured Schur complement Newton solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solve_nivp.block_system import SchurComplementSolver


def _simple_block_system():
    """2-DOF velocity + 1-DOF contact: known analytical solution.

    With gap > 0 (inactive): λ = 0, u_new = u_old + h * f / m.
    """
    n_phys = 2
    n_react = 1
    h = 0.01
    mass = np.array([1.0, 2.0])
    H = np.diag(mass)
    f_ext = np.array([0.0, -9.81])

    J = np.array([[0.0, 1.0]])
    C = np.array([[1.0]])
    g_rhs = h * f_ext
    h_rhs = np.array([0.0])

    return H, J, C, g_rhs, h_rhs, n_phys, n_react


def test_schur_solver_inactive_contact():
    """Inactive contact: Schur solve matches direct solve of saddle-point."""
    H, J, C, g_rhs, h_rhs, n_phys, n_react = _simple_block_system()
    solver = SchurComplementSolver(maxiter=5, tol=1e-12, pcr_maxiter=20)
    delta_u, delta_lam, info = solver.solve_linear(H, J, C, g_rhs, h_rhs)

    A_full = np.block([[H, -J.T], [J, C]])
    rhs_full = np.concatenate([g_rhs, h_rhs])
    x_exact = np.linalg.solve(A_full, rhs_full)

    assert_allclose(delta_u, x_exact[:n_phys], atol=1e-10)
    assert_allclose(delta_lam, x_exact[n_phys:], atol=1e-10)
    assert info["converged"]


def test_schur_solver_active_contact():
    """Active contact: Schur complement couples velocity and reaction."""
    H = np.diag([1.0, 1.0])
    J = np.array([[0.0, 1.0]])
    C = np.array([[1e-5]])
    g_rhs = np.array([0.0, -9.81])
    h_rhs = np.array([-0.5])

    solver = SchurComplementSolver(maxiter=5, tol=1e-12, pcr_maxiter=50)
    delta_u, delta_lam, info = solver.solve_linear(H, J, C, g_rhs, h_rhs)

    A_full = np.block([
        [H, -J.T],
        [J, C],
    ])
    rhs_full = np.concatenate([g_rhs, h_rhs])
    x_exact = np.linalg.solve(A_full, rhs_full)

    assert_allclose(delta_u, x_exact[:2], atol=1e-8)
    assert_allclose(delta_lam, x_exact[2:], atol=1e-8)


def test_schur_matches_direct_solve_random():
    """Schur complement solution matches direct solve of the full system."""
    rng = np.random.default_rng(123)
    n_p, n_r = 10, 4
    L = rng.standard_normal((n_p, n_p))
    H = L.T @ L + 3.0 * np.eye(n_p)
    J = rng.standard_normal((n_r, n_p))
    C = 0.1 * np.eye(n_r)
    g = rng.standard_normal(n_p)
    h_c = rng.standard_normal(n_r)

    solver = SchurComplementSolver(
        pcr_maxiter=100, tol=1e-12, use_preconditioner=False,
    )
    delta_u, delta_lam, info = solver.solve_linear(H, J, C, g, h_c)

    A_full = np.block([[H, -J.T], [J, C]])
    rhs_full = np.concatenate([g, h_c])
    x_exact = np.linalg.solve(A_full, rhs_full)

    assert_allclose(delta_u, x_exact[:n_p], atol=1e-7)
    assert_allclose(delta_lam, x_exact[n_p:], atol=1e-7)
