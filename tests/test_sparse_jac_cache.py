import numpy as np
import scipy.sparse as sp

from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection


def test_sparse_jacobian_cache_rejects_same_nnz_pattern_change():
    """The cached CSR may only be value-updated when the sparsity pattern is
    unchanged.  Two Jacobians with equal nnz but different ``indices`` must not
    have the second's data copied into the first's positions."""
    solver = ImplicitEquationSolver(method='semismooth_newton',
                                    proj=IdentityProjection())

    J1 = sp.csr_matrix(np.diag([1.0, 2.0, 3.0, 4.0]))          # nnz 4, diagonal
    J2 = sp.csr_matrix(np.array([[0.0, 5.0, 0.0, 0.0],
                                 [0.0, 0.0, 6.0, 0.0],
                                 [0.0, 0.0, 0.0, 7.0],
                                 [8.0, 0.0, 0.0, 0.0]]))        # nnz 4, cyclic
    assert J1.data.size == J2.data.size

    pending = [J1, J2]
    solver.jacobian = lambda y: pending.pop(0)
    y = np.zeros(4)

    first = solver._compute_jacobian_csr(lambda yy: yy, y, sparse_active=True)
    np.testing.assert_array_equal(first.toarray(), J1.toarray())

    second = solver._compute_jacobian_csr(lambda yy: yy, y, sparse_active=True)
    np.testing.assert_array_equal(second.toarray(), J2.toarray())
