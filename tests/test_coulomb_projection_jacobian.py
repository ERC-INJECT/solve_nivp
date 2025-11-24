import numpy as np
from solve_nivp.projections import CoulombProjection

def conf(y, t=None, Fk_val=None):
    # Simple linear map for testing
    return y

def jac_conf(y, t=None, Fk_val=None):
    return np.eye(len(y))

def test_coulomb_projection_analytical_jacobian_matches_fd():
    n = 4
    y = np.linspace(-1, 1, n)
    proj = CoulombProjection(conf, rhok=np.ones(n), jac_func=jac_conf)
    J_analytical = proj._compute_jacobian(y)
    # Temporarily remove jac_func to force numerical
    proj.jac_func = None
    J_fd = proj._compute_jacobian(y)
    assert np.allclose(J_analytical, J_fd, atol=1e-6)
