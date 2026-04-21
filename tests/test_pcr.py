"""Tests for the Preconditioned Conjugate Residual (PCR) solver."""

import numpy as np
import scipy.sparse as sp
import pytest
from numpy.testing import assert_allclose

from solve_nivp.pcr import pcr_solve


def test_pcr_spd_2x2():
    """2x2 SPD system solved exactly in at most 2 iterations."""
    A = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_exact = np.linalg.solve(A, b)

    x, info = pcr_solve(lambda v: A @ v, b, tol=1e-12)

    assert info["converged"]
    assert info["iterations"] <= 2
    assert_allclose(x, x_exact, atol=1e-10)


def test_pcr_spd_diagonal_preconditioner():
    """20x20 SPD with Jacobi preconditioner converges."""
    rng = np.random.RandomState(99)
    R = rng.randn(20, 20)
    A = R.T @ R + 5.0 * np.eye(20)
    b = rng.randn(20)
    x_exact = np.linalg.solve(A, b)

    diag_inv = 1.0 / np.diag(A)
    precond = lambda v: diag_inv * v

    x, info = pcr_solve(lambda v: A @ v, b, maxiter=200, tol=1e-10,
                         preconditioner=precond)

    assert info["converged"]
    assert_allclose(x, x_exact, atol=1e-7)


def test_pcr_indefinite_saddle_point():
    """3x3 symmetric indefinite saddle-point system [[K, G^T], [G, 0]]."""
    K = np.array([[4.0, 1.0],
                  [1.0, 3.0]])
    G = np.array([[1.0, -1.0]])
    n = 3
    A = np.zeros((n, n))
    A[:2, :2] = K
    A[:2, 2:] = G.T
    A[2:, :2] = G

    b = np.array([1.0, 0.5, 0.0])
    x_exact = np.linalg.solve(A, b)

    x, info = pcr_solve(lambda v: A @ v, b, maxiter=50, tol=1e-12)

    assert info["converged"]
    assert_allclose(x, x_exact, atol=1e-9)


def test_pcr_residual_monotonically_decreasing():
    """Residual history is monotonically non-increasing for a 15x15 SPD system."""
    rng = np.random.RandomState(42)
    R = rng.randn(15, 15)
    A = R.T @ R + 2.0 * np.eye(15)
    b = rng.randn(15)

    _, info = pcr_solve(lambda v: A @ v, b, maxiter=50, tol=1e-12)

    hist = info["residual_history"]
    assert len(hist) >= 2
    eps = 1e-14
    for i in range(1, len(hist)):
        assert hist[i] <= hist[i - 1] + eps, (
            f"Residual increased at iteration {i}: {hist[i]} > {hist[i-1]}")


def test_pcr_sparse_matvec():
    """PCR works with a sparse diagonal matvec."""
    n = 50
    diag_vals = np.arange(1.0, n + 1.0)
    A_sp = sp.diags(diag_vals, 0, format="csr")
    b = np.ones(n)

    x_exact = b / diag_vals

    x, info = pcr_solve(lambda v: A_sp @ v, b, maxiter=200, tol=1e-12)

    assert info["converged"]
    assert_allclose(x, x_exact, atol=1e-10)
