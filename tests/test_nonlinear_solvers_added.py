import numpy as np
import scipy.sparse as sp
import time

from Solve_IVP_NS.nonlinear_solvers import ImplicitEquationSolver
from Solve_IVP_NS.projections import IdentityProjection


def test_vi_convergence_monotone_identity():
    # f(y) = A y + b, A SPD, identity projection
    rng = np.random.default_rng(0)
    n = 5
    M = rng.standard_normal((n, n))
    A = M.T @ M + np.eye(n)
    b = rng.standard_normal(n)
    f = lambda y: A @ y + b
    y0 = np.zeros(n)
    solver = ImplicitEquationSolver(method='VI', proj=IdentityProjection(), component_slices=[slice(0, n)], tol=1e-10, max_iter=200)
    y, Fk, err, ok, it = solver.solve(f, y0)
    assert ok and err < solver.tol and it <= solver.max_iter


def test_ssn_scalar_cubic_identity_both_globalizations():
    f = lambda y: y + 0.1 * y**3
    y0 = np.array([1.0])

    # no globalization
    s1 = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-12, globalization='none')
    y, Fk, err, ok, it = s1.solve(f, y0)
    assert ok and err < s1.tol

    # with line search
    s2 = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-12, globalization='linesearch')
    y, Fk, err, ok, it = s2.solve(f, y0)
    assert ok and err < s2.tol


def test_ssn_uses_analytical_jacobian_and_matches_numeric():
    # Linear system f(y) = A y - b; Jacobian is A
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([1.0, -1.0])
    f = lambda y: A @ y - b
    y0 = np.zeros(2)

    s = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-12)
    s.jacobian = lambda y: A

    y, Fk, err, ok, it = s.solve(f, y0)
    assert ok and err < s.tol

    # Compare numeric Jacobian to A at solution
    Jn = s._numerical_jacobian(f, y)
    np.testing.assert_allclose(Jn, A, rtol=1e-6, atol=1e-8)


def test_sparse_path_forced_success():
    # Force sparse path by size
    n = 220
    D = np.arange(1, n + 1, dtype=float)
    A = sp.diags(D, format='csr')
    b = np.ones(n)
    f = lambda y: A @ y - b
    y0 = np.zeros(n)

    solver = ImplicitEquationSolver(method='semismooth_newton', proj=IdentityProjection(), tol=1e-10, sparse=True, sparse_threshold=1)
    t0 = time.time()
    y, Fk, err, ok, it = solver.solve(f, y0)
    t1 = time.time()
    assert ok and err < solver.tol
    # ensure it didn't blow up in retries/time; this is a smoke check
    assert (t1 - t0) < 5.0
