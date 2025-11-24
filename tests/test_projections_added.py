import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.projections import IdentityProjection, SignProjection, CoulombProjection


def test_identity_tangent_is_identity_any_y():
    y = np.array([1.0, -2.0, 3.0])
    P = IdentityProjection()
    D = P.tangent_cone(candidate=y, current_state=y)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(D, np.eye(3))


def test_sign_projection_project_and_tangent():
    # scalar y,w
    P = SignProjection(y_indices=0, w_indices=1)
    cand = np.array([2.0, 0.3])
    out = P.project(cand, cand)
    assert out[1] == 1.0

    cand = np.array([-2.0, 0.3])
    out = P.project(cand, cand)
    assert out[1] == -1.0

    # y == 0 -> clamp within [-1,1]
    cand = np.array([0.0, 2.3])
    out = P.project(cand, cand)
    assert out[1] == 1.0
    cand = np.array([0.0, -2.3])
    out = P.project(cand, cand)
    assert out[1] == -1.0
    cand = np.array([0.0, 0.2])
    out = P.project(cand, cand)
    assert -1.0 <= out[1] <= 1.0

    # tangent_cone rules on the w diagonal
    # off kink (|y|>tol): derivative 0 on w
    D = P.tangent_cone(candidate=np.array([2.0, 0.0]), current_state=np.array([2.0, 0.0]))
    assert D.shape == (2, 2)
    assert D[1, 1] == 0.0

    # at kink with |w|<1: derivative 1 on w
    D = P.tangent_cone(candidate=np.array([0.0, 0.0]), current_state=np.array([0.0, 0.0]))
    assert D[1, 1] == 1.0

    # at kink with |w|≈1: 0.5 selection
    D = P.tangent_cone(candidate=np.array([0.0, 1.0]), current_state=np.array([0.0, 1.0]))
    assert np.isclose(D[1, 1], 0.5)


def test_coulomb_projection_shapes_and_jac_modes():
    # tiny problem with 2 pairs -> state length 4
    # constraint indices point to the v components (0 and 2)
    n_pairs = 2
    n = 2 * n_pairs

    def con_force(y, t=None, Fk_val=None):
        # simple linear conf force on v's only
        g = np.zeros_like(y)
        g[0] = 2.0 * y[0]
        g[2] = 3.0 * y[2]
        return g

    rhok = np.ones(n)
    P_full = CoulombProjection(con_force_func=con_force, rhok=rhok, constraint_indices=np.array([0, 2]), conf_jacobian_mode='full')
    P_none = CoulombProjection(con_force_func=con_force, rhok=rhok, constraint_indices=np.array([0, 2]), conf_jacobian_mode='none')

    y = np.array([0.1, 0.2, -0.3, 0.4])

    # project returns vector of same shape
    y_proj = P_full.project(y, y, rhok)
    assert y_proj.shape == y.shape

    # tangent_cone returns CSR sparse matrix with shape (n,n)
    D_full = P_full.tangent_cone(candidate=y, current_state=y, rhok=rhok)
    D_none = P_none.tangent_cone(candidate=y, current_state=y, rhok=rhok)
    assert sp.isspmatrix_csr(D_full) and D_full.shape == (n, n)
    assert sp.isspmatrix_csr(D_none) and D_none.shape == (n, n)

    # light numeric check: diagonals bounded and shapes correct
    d_full = D_full.diagonal()
    d_none = D_none.diagonal()
    assert np.all(np.isfinite(d_full)) and np.all(np.isfinite(d_none))
    assert np.all(d_full <= 1.0 + 1e-12)
    assert np.all(d_none <= 1.0 + 1e-12)


def test_coulomb_abs_v_mode_projects_pairs_and_tangent():
    # Single constrained index at 0 (no longer a pair concept)
    n = 2

    # conf returns zero so v_pre = v and z_tilde = |v|
    def con_force(y, t=None, Fk_val=None):
        return np.zeros_like(y)

    rhok = np.ones(n)
    # Note: z_mode parameter removed as it doesn't exist in modified implementation
    P_abs = CoulombProjection(con_force_func=con_force, rhok=rhok,
                              constraint_indices=np.array([0]), conf_jacobian_mode='full')

    # Candidate has v=0.3 at index 0, y[1]=100.0 is unconstrained
    y = np.array([0.3, 100.0])
    y_proj = P_abs.project(y, y, rhok)

    # With z_tilde = |0.3| - 1*0 = 0.3 and v=0.3
    # This is in region R1 (|z| <= v), so projects onto (1,1) ray at magnitude 0.3
    np.testing.assert_allclose(y_proj[0], 0.3, atol=1e-12)
    # y[1] should remain unchanged since it's not a constrained index
    np.testing.assert_allclose(y_proj[1], 100.0, atol=1e-12)

    # Test tangent cone at this point
    D = P_abs.tangent_cone(candidate=y, current_state=y, rhok=rhok)
    assert D.shape == (n, n)
    Dd = D.toarray()

    # Check that unconstrained index 1 has identity row
    np.testing.assert_allclose(Dd[1, :], [0, 1], atol=1e-12)

    # Test with negative v
    y_neg = np.array([-0.3, 50.0])
    y_proj_neg = P_abs.project(y_neg, y_neg, rhok)
    # z_tilde = |−0.3| = 0.3, v = -0.3
    # This is in region R3 (|z| <= -v), so different projection
    # y[1] should remain unchanged
    np.testing.assert_allclose(y_proj_neg[1], 50.0, atol=1e-12)

    # Test at origin (tip region)
    y_zero = np.array([0.0, 25.0])
    D_zero = P_abs.tangent_cone(candidate=y_zero, current_state=y_zero, rhok=rhok)
    Dd_zero = D_zero.toarray()
    # At tip, constrained index should have zero row
    np.testing.assert_allclose(Dd_zero[0, :], [0, 0], atol=1e-12)
    # Unconstrained index remains identity
    np.testing.assert_allclose(Dd_zero[1, :], [0, 1], atol=1e-12)
