import numpy as np
import scipy.sparse as sp

from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import Projection


class _ConstTangentProjection(Projection):
    """Non-identity projection whose value leaves the candidate unchanged but
    whose tangent cone is a fixed, full (non-diagonal) matrix.  This forces the
    general semismooth-Newton assembly ``J = I - D + D @ diag(lam) @ J`` on both
    the dense and sparse code paths."""

    def __init__(self, D):
        super().__init__()
        self._D = sp.csr_matrix(np.asarray(D, dtype=float))

    def project(self, current_state, candidate, rhok=None, t=None, Fk_val=None):
        return candidate

    def tangent_cone(self, candidate, current_state, rhok=None, t=None, Fk_val=None):
        return self._D


def test_dense_vector_lam_matches_sparse_row_scaling():
    M = np.array([[2.0, 1.0, 0.0],
                  [0.0, 3.0, 1.0],
                  [1.0, 0.0, 4.0]])
    b = np.array([1.0, 2.0, 3.0])
    lam = np.array([0.2, 1.0, 5.0])
    D = np.array([[0.6, 0.1, 0.0],
                  [0.0, 0.7, 0.1],
                  [0.1, 0.0, 0.8]])
    y0 = np.array([0.3, -0.2, 0.5])

    def func(y):
        return M @ y - b

    common = dict(method='semismooth_newton', adaptive_lam=False,
                  max_iter=1, tol=1e-14, lam=lam)

    dense = ImplicitEquationSolver(
        proj=_ConstTangentProjection(D), sparse_threshold=1000, **common)
    sparse = ImplicitEquationSolver(
        proj=_ConstTangentProjection(D), sparse_threshold=0,
        linear_solver='splu', **common)

    y_dense = dense.solve(func, y0.copy())[0]
    y_sparse = sparse.solve(func, y0.copy())[0]

    np.testing.assert_allclose(y_dense, y_sparse, atol=1e-9)
