import numpy as np
import scipy.sparse as sp
import pytest

import solve_nivp.solvers.block_system as bsmod
from solve_nivp.solvers.block_system import SchurComplementSolver


class _ConstResidualSystem:
    """Block system whose residual never drops below tol, so the Newton loop
    always reaches the linear-solve step."""

    n_phys = 1
    n_react = 1

    def assemble_blocks(self, y, t, h, y_prev):
        return {
            "H": np.array([[1.0]]),
            "B_top": np.array([[0.0]]),
            "B_bot": np.array([[0.0]]),
            "C": np.array([[1.0]]),
            "g": np.array([10.0]),
            "h_c": np.array([10.0]),
            "precond_diag": None,
        }


def test_unknown_linear_solver_is_rejected():
    with pytest.raises(ValueError):
        SchurComplementSolver(linear_solver="bogus")


def test_solve_does_not_apply_unconverged_linear_step():
    solver = SchurComplementSolver(maxiter=3, tol=1e-12, linear_solver="direct")

    def _fake_solve_linear(*args, **kwargs):
        return (np.array([1.0e6]), np.array([1.0e6]),
                {"converged": False, "iterations": 1, "residual_norm": 1.0})

    solver.solve_linear = _fake_solve_linear
    y, _err, converged, _iters = solver.solve(
        _ConstResidualSystem(), np.array([0.0, 0.0]), 0.0, 0.1,
        np.array([0.0, 0.0]))

    assert converged is False
    assert np.all(np.abs(y) < 1.0e-3)


def test_direct_solver_uses_sparse_path_for_sparse_blocks(monkeypatch):
    solver = SchurComplementSolver(linear_solver="direct")
    H = sp.csr_matrix(np.array([[2.0, 0.0], [0.0, 3.0]]))
    C = sp.csr_matrix(np.array([[4.0]]))
    B_top = sp.csr_matrix(np.zeros((2, 1)))
    B_bot = sp.csr_matrix(np.zeros((1, 2)))
    g = np.array([2.0, 6.0])
    h_c = np.array([8.0])

    def _no_dense(*args, **kwargs):
        raise AssertionError("dense la.solve used for sparse blocks")

    monkeypatch.setattr(bsmod.la, "solve", _no_dense)

    delta_u, delta_lam, _info = solver.solve_linear(H, B_top, B_bot, C, g, h_c)

    np.testing.assert_allclose(delta_u, [1.0, 2.0])
    np.testing.assert_allclose(delta_lam, [2.0])
