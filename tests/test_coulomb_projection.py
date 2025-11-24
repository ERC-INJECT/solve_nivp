import numpy as np
import scipy.sparse as sp

from solve_nivp.projections import CoulombProjection


def test_coulomb_tangent_cone_tip_zero():
    # Single constrained pair at indices (0,1)
    # con_force returns zero so z_tilde = y[1]
    con_force = lambda y, t=None, Fk_val=None: np.array([0.0, 0.0])
    proj = CoulombProjection(con_force_func=con_force, rhok=np.array([1.0, 1.0]), constraint_indices=[0])

    y = np.zeros(2)
    cand = np.zeros_like(y)

    D = proj.tangent_cone(cand, y, rhok=1.0)
    assert sp.isspmatrix_csr(D)
    Dd = D.toarray()
    # At the tip, P=0 -> zero row for constrained v index; other indices remain identity
    assert np.allclose(Dd[0, :], 0.0)
    assert np.allclose(Dd[1, :], np.eye(2)[1])


def test_coulomb_tangent_cone_modes_jac_none_full():
    # con_force = [y0 + 2*y1, y1] so J_conf[0] = [1,2]
    def con_force(y, t=None, Fk_val=None):
        return np.array([y[0] + 2.0*y[1], y[1]])

    y = np.array([1.0, -0.25])
    cand = y.copy()

    # Mode 'none': d(conf)/dy ignored -> only direct cols
    proj_none = CoulombProjection(con_force_func=con_force, rhok=np.array([1.0, 1.0]),
                                  constraint_indices=[0], conf_jacobian_mode='none')
    D_none = proj_none.tangent_cone(cand, y, rhok=1.0).toarray()

    # Mode 'full': include chain rule term
    proj_full = CoulombProjection(con_force_func=con_force, rhok=np.array([1.0, 1.0]),
                                  constraint_indices=[0], conf_jacobian_mode='full')
    D_full = proj_full.tangent_cone(cand, y, rhok=1.0).toarray()

    # Matrices must differ when Jacobian contribution is active
    assert not np.allclose(D_none, D_full)


def test_coulomb_projection_project_maps_pair():
    # Test project path is consistent shape and modifies expected indices
    con_force = lambda y, t=None, Fk_val=None: y.copy()
    proj = CoulombProjection(con_force_func=con_force, rhok=np.ones(4), constraint_indices=[0,2])
    y = np.array([1.0, 0.2, -0.3, 0.1])
    cand = y + 0.5  # arbitrary candidate
    out = proj.project(y, cand, rhok=1.0)
    assert out.shape == y.shape
