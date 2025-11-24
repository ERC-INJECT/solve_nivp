import numpy as np
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
from solve_nivp.projections import IdentityProjection

def simple_func(y):
    # Root at y = 0, mildly nonlinear
    return y + 0.1*y**3

def test_semismooth_linesearch_converges():
    proj = IdentityProjection()
    solver = ImplicitEquationSolver(method='semismooth_newton', proj=proj, globalization='linesearch', max_iter=50, tol=1e-10)
    y0 = np.array([5.0, -3.0])
    y_star, Fk, err, success, it = solver.solve(simple_func, y0)
    assert success, 'Line search semismooth Newton failed to converge'
    assert err < 1e-8

def test_semismooth_no_globalization_converges():
    proj = IdentityProjection()
    solver = ImplicitEquationSolver(method='semismooth_newton', proj=proj, globalization='none', max_iter=50, tol=1e-10)
    y0 = np.array([1.0, -1.0])
    y_star, Fk, err, success, it = solver.solve(simple_func, y0)
    assert success
    assert err < 1e-8
