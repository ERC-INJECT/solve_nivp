"""Tests for the block-structured Schur complement Newton solver."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solve_nivp.block_system import SchurComplementSolver


def test_schur_solver_inactive_contact():
    """Inactive contact: Schur solve matches direct solve of saddle-point."""
    H = np.diag([1.0, 2.0])
    B_top = np.array([[0.0], [-1.0]])
    B_bot = np.array([[0.0, 1.0]])
    C = np.array([[1.0]])
    g = 0.01 * np.array([0.0, -9.81])
    h_c = np.array([0.0])

    solver = SchurComplementSolver(maxiter=5, tol=1e-12)
    delta_u, delta_lam, info = solver.solve_linear(H, B_top, B_bot, C, g, h_c)

    A_full = np.block([[H, B_top], [B_bot, C]])
    x_exact = np.linalg.solve(A_full, np.concatenate([g, h_c]))
    assert_allclose(delta_u, x_exact[:2], atol=1e-10)
    assert_allclose(delta_lam, x_exact[2:], atol=1e-10)


def test_schur_solver_active_contact():
    """Active contact: Schur complement couples velocity and reaction."""
    H = np.diag([1.0, 1.0])
    B_top = np.array([[0.0], [-1.0]])
    B_bot = np.array([[0.0, 1.0]])
    C = np.array([[1e-5]])
    g = np.array([0.0, -9.81])
    h_c = np.array([-0.5])

    solver = SchurComplementSolver(maxiter=5, tol=1e-12)
    delta_u, delta_lam, info = solver.solve_linear(H, B_top, B_bot, C, g, h_c)

    A_full = np.block([[H, B_top], [B_bot, C]])
    x_exact = np.linalg.solve(A_full, np.concatenate([g, h_c]))
    assert_allclose(delta_u, x_exact[:2], atol=1e-8)
    assert_allclose(delta_lam, x_exact[2:], atol=1e-8)


def test_schur_nonsymmetric_direct():
    """Non-symmetric off-diagonals handled correctly by direct solver."""
    rng = np.random.default_rng(123)
    n_p, n_r = 10, 4
    L = rng.standard_normal((n_p, n_p))
    H = L.T @ L + 3.0 * np.eye(n_p)
    B_top = rng.standard_normal((n_p, n_r))
    B_bot = rng.standard_normal((n_r, n_p))
    C = 0.1 * np.eye(n_r) + 0.01 * rng.standard_normal((n_r, n_r))
    g = rng.standard_normal(n_p)
    h_c = rng.standard_normal(n_r)

    solver = SchurComplementSolver(tol=1e-12, linear_solver="direct")
    du, dl, info = solver.solve_linear(H, B_top, B_bot, C, g, h_c)

    A_full = np.block([[H, B_top], [B_bot, C]])
    x_exact = np.linalg.solve(A_full, np.concatenate([g, h_c]))
    assert_allclose(du, x_exact[:n_p], atol=1e-10)
    assert_allclose(dl, x_exact[n_p:], atol=1e-10)


def test_pcr_symmetric_macklin_case():
    """PCR path works for the symmetric Macklin case B_top = -B_bot^T."""
    rng = np.random.default_rng(42)
    n_p, n_r = 8, 3
    L = rng.standard_normal((n_p, n_p))
    H = L.T @ L + 5.0 * np.eye(n_p)
    J = rng.standard_normal((n_r, n_p))
    B_top = -J.T
    B_bot = J
    C = 0.2 * np.eye(n_r)
    g = rng.standard_normal(n_p)
    h_c = rng.standard_normal(n_r)

    solver = SchurComplementSolver(
        linear_solver="pcr", pcr_maxiter=200, use_preconditioner=False,
    )
    du_pcr, dl_pcr, info = solver.solve_linear(
        H, B_top, B_bot, C, g, h_c,
    )

    A_full = np.block([[H, B_top], [B_bot, C]])
    x_exact = np.linalg.solve(A_full, np.concatenate([g, h_c]))
    assert_allclose(du_pcr, x_exact[:n_p], atol=1e-6)
    assert_allclose(dl_pcr, x_exact[n_p:], atol=1e-6)
