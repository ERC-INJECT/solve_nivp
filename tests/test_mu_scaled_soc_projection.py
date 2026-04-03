"""Tests for the rewritten MuScaledSOCProjection (purely geometric)."""

import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.projections import MuScaledSOCProjection


# =====================================================================
# Helpers
# =====================================================================

def numerical_jacobian(proj, z_full, block_indices, eps=1e-7):
    """Finite-difference Jacobian of the projection restricted to block DOFs."""
    n = z_full.size
    J = np.zeros((n, n))
    f0 = proj.project(z_full, z_full)
    for j in range(n):
        z_pert = z_full.copy()
        z_pert[j] += eps
        f1 = proj.project(z_pert, z_pert)
        J[:, j] = (f1 - f0) / eps
    return J


# =====================================================================
# Static _proj_mu_scaled_soc tests
# =====================================================================

class TestProjStatic:
    """Tests for MuScaledSOCProjection._proj_mu_scaled_soc."""

    def test_interior_identity_2d(self):
        """Point inside K_mu → returned unchanged."""
        mu = 0.5
        z = np.array([4.0, 1.0])  # s=4, |w|=1, mu*s=2 >= 1 ✓
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        np.testing.assert_array_equal(p, z)

    def test_interior_jacobian_identity_2d(self):
        """Interior → Jacobian = I."""
        mu = 0.5
        z = np.array([4.0, 1.0])
        p, J = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu, return_jacobian=True)
        np.testing.assert_array_equal(p, z)
        np.testing.assert_array_equal(J, np.eye(2))

    def test_polar_zero_2d(self):
        """Point inside polar cone → projected to zero."""
        mu = 0.5
        # polar: s <= 0 and |w| <= -s/mu → s=-4, |w|=1, -s/mu=8 >= 1 ✓
        z = np.array([-4.0, 1.0])
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        np.testing.assert_array_equal(p, np.zeros(2))

    def test_polar_jacobian_zero_2d(self):
        """Polar → Jacobian = 0."""
        mu = 0.5
        z = np.array([-4.0, 1.0])
        p, J = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu, return_jacobian=True)
        np.testing.assert_array_equal(p, np.zeros(2))
        np.testing.assert_array_equal(J, np.zeros((2, 2)))

    def test_boundary_2d(self):
        """Point outside both interior and polar → boundary formula."""
        mu = 1.0
        z = np.array([0.0, 2.0])  # s=0, w=2, not in K_1 (need s>=|w|)
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        # alpha = 1/(1+1) = 0.5, beta = 0 + 1*2 = 2
        # p_s = 0.5*2 = 1, p_w = 1*1*1 = 1  (w_hat = 1)
        np.testing.assert_allclose(p, [1.0, 1.0])

    def test_boundary_jacobian_2d(self):
        """Boundary Jacobian matches finite differences."""
        mu = 0.7
        z = np.array([1.0, 3.0])
        p, J = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu, return_jacobian=True)
        # Finite-difference check
        eps = 1e-7
        J_fd = np.zeros((2, 2))
        for j in range(2):
            z_p = z.copy(); z_p[j] += eps
            p_p = MuScaledSOCProjection._proj_mu_scaled_soc(z_p, mu)
            J_fd[:, j] = (p_p - p) / eps
        np.testing.assert_allclose(J, J_fd, atol=1e-5)

    def test_interior_3d(self):
        """3-D: point inside K_mu."""
        mu = 1.0
        z = np.array([5.0, 1.0, 2.0])  # s=5, ||w||=sqrt(5)≈2.24, mu*s=5 ✓
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        np.testing.assert_array_equal(p, z)

    def test_boundary_3d(self):
        """3-D boundary projection."""
        mu = 0.5
        z = np.array([0.0, 3.0, 4.0])  # s=0, ||w||=5
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        # alpha = 1/(1+0.25) = 0.8, beta = 0+0.5*5=2.5
        # p_s = 0.8*2.5 = 2.0
        # p_w = 2.0 * 0.5 * w_hat = 1.0 * [0.6, 0.8]
        np.testing.assert_allclose(p, [2.0, 0.6, 0.8])

    def test_boundary_jacobian_3d(self):
        """3-D boundary Jacobian vs finite differences."""
        mu = 0.3
        z = np.array([1.0, 2.0, -1.5])
        p, J = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu, return_jacobian=True)
        eps = 1e-7
        J_fd = np.zeros((3, 3))
        for j in range(3):
            z_p = z.copy(); z_p[j] += eps
            p_p = MuScaledSOCProjection._proj_mu_scaled_soc(z_p, mu)
            J_fd[:, j] = (p_p - p) / eps
        np.testing.assert_allclose(J, J_fd, atol=1e-5)

    def test_idempotent(self):
        """Projection is idempotent: projecting the result again is a no-op."""
        mu = 0.6
        for z in [np.array([1.0, 3.0]),
                   np.array([-2.0, 0.5]),
                   np.array([0.0, 0.0]),
                   np.array([2.0, 0.5, -1.0])]:
            p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
            pp = MuScaledSOCProjection._proj_mu_scaled_soc(p, mu)
            np.testing.assert_allclose(pp, p, atol=1e-14,
                                       err_msg=f"Not idempotent for z={z}")

    def test_mu_zero_normal_only(self):
        """μ = 0 → K_0 = {(s,0): s>=0}, only normal clamped."""
        z = np.array([-1.0, 5.0])
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu=0.0)
        # With mu=0: interior check s>=0 and ||w||<=0 → fails (w=5)
        # polar check: s<=0 → True, ||w|| <= -s/0 → division issue
        # Actually mu=0: K_0 = half-line, polar = {t<=0}
        # s=-1 <=0, so in polar → project to zero
        np.testing.assert_array_equal(p, np.zeros(2))

    def test_mu_zero_positive_s(self):
        """μ = 0, s > 0 → boundary projection clamps w to 0."""
        z = np.array([3.0, 5.0])
        p = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu=0.0)
        # s=3>0, mu*s=0, ||w||=5>0 → not interior
        # polar: s>0 → not polar
        # boundary: alpha=1/(1+0)=1, beta=s+0*r=3, p_s=3, p_w=0*w_hat=0
        np.testing.assert_allclose(p, [3.0, 0.0])


# =====================================================================
# Full projection tests (block-wise)
# =====================================================================

class TestProjection:
    """Tests for MuScaledSOCProjection.project."""

    def test_single_block_2d_interior(self):
        """Single 2-D block, point inside cone → unchanged."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        z = np.array([4.0, 1.0, 99.0])  # idx 0=s, idx 1=w, idx 2=unrelated
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, z)

    def test_single_block_2d_boundary(self):
        """Single 2-D block, outside cone → projected."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
        )
        z = np.array([0.0, 2.0, 99.0])
        result = proj.project(z, z)
        # Block [0,2] → proj → [1,1], rest unchanged
        np.testing.assert_allclose(result[:2], [1.0, 1.0])
        assert result[2] == 99.0

    def test_two_blocks_2d(self):
        """Two independent 2-D blocks."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
        )
        z = np.array([0.0, 2.0, -4.0, 0.5])
        result = proj.project(z, z)
        # Block 0: boundary → [1, 1]
        np.testing.assert_allclose(result[:2], [1.0, 1.0])
        # Block 1: polar (s=-4, |w|=0.5, -s/mu=4 >= 0.5) → [0, 0]
        np.testing.assert_allclose(result[2:], [0.0, 0.0])

    def test_single_block_3d(self):
        """3-D contact: s + 2 tangential DOFs."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
        )
        z = np.array([0.0, 3.0, 4.0])  # ||w||=5
        result = proj.project(z, z)
        np.testing.assert_allclose(result, [2.0, 0.6, 0.8])

    def test_non_contiguous_indices(self):
        """Block indices don't need to be contiguous."""
        # s at index 3, w at index 0
        proj = MuScaledSOCProjection(
            blocks=[(3, [0])],
            get_mu=lambda y: 1.0,
        )
        z = np.array([2.0, 99.0, 99.0, 0.0])  # s=0, w=2
        result = proj.project(z, z)
        np.testing.assert_allclose(result[3], 1.0)   # projected s
        np.testing.assert_allclose(result[0], 1.0)   # projected w
        assert result[1] == 99.0
        assert result[2] == 99.0

    def test_slice_block_spec(self):
        """Blocks given as slices."""
        proj = MuScaledSOCProjection(
            blocks=[slice(0, 2)],
            get_mu=lambda y: 1.0,
        )
        z = np.array([0.0, 2.0])
        result = proj.project(z, z)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_mu_zero_batch_tangent_zeroes_tangential_rows(self):
        """Vectorized tangent path must use diag([1, 0]) on active K_0 blocks."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 0.0,
            gap_func=lambda y, t=None: np.array([0.0, 0.0]),
            gap_tol=1.0e-12,
            zero_inactive=True,
        )
        z = np.zeros(4)
        D = proj.tangent_cone(z, z)
        D = D.toarray() if sp.issparse(D) else np.asarray(D)
        np.testing.assert_allclose(D, np.diag([1.0, 0.0, 1.0, 0.0]), atol=0.0)

    def test_per_block_mu(self):
        """get_mu returns per-block array."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: np.array([1.0, 0.0]),
        )
        z = np.array([0.0, 2.0, 3.0, 5.0])
        result = proj.project(z, z)
        # Block 0 (mu=1): boundary
        np.testing.assert_allclose(result[:2], [1.0, 1.0])
        # Block 1 (mu=0): s=3>0, w=5, boundary → [3, 0]
        np.testing.assert_allclose(result[2:], [3.0, 0.0])


# =====================================================================
# Gap activation tests
# =====================================================================

class TestGapActivation:
    """Tests for gap_func-based block activation."""

    def test_inactive_block_unchanged(self):
        """Block with positive gap is not projected."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            gap_func=lambda y, t: np.array([1.0]),  # gap > 0 → inactive
        )
        z = np.array([0.0, 2.0])
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, z)  # unchanged

    def test_active_block_projected(self):
        """Block with negative gap is projected."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            gap_func=lambda y, t: np.array([-0.1]),  # gap < 0 → active
        )
        z = np.array([0.0, 2.0])
        result = proj.project(z, z)
        np.testing.assert_allclose(result, [1.0, 1.0])

    def test_mixed_activation(self):
        """Two blocks, one active, one not."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
            gap_func=lambda y, t: np.array([-1.0, 5.0]),
        )
        z = np.array([0.0, 2.0, 0.0, 2.0])
        result = proj.project(z, z)
        # Block 0 active → projected
        np.testing.assert_allclose(result[:2], [1.0, 1.0])
        # Block 1 inactive → unchanged
        np.testing.assert_allclose(result[2:], [0.0, 2.0])

    def test_gap_func_single_arg(self):
        """gap_func with only y argument."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            gap_func=lambda y: np.array([-1.0]),
        )
        z = np.array([0.0, 2.0])
        result = proj.project(z, z)
        np.testing.assert_allclose(result, [1.0, 1.0])


# =====================================================================
# Tangent cone tests
# =====================================================================

class TestTangentCone:
    """Tests for MuScaledSOCProjection.tangent_cone."""

    def test_interior_identity(self):
        """Interior point → tangent cone = identity."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        z = np.array([4.0, 1.0, 99.0])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)
        np.testing.assert_allclose(D_dense, np.eye(3))

    def test_polar_zero_rows(self):
        """Polar interior → block rows are zero."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        z = np.array([-4.0, 1.0, 99.0])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)
        # Block rows 0,1 should be zero
        np.testing.assert_allclose(D_dense[0, :], 0.0)
        np.testing.assert_allclose(D_dense[1, :], 0.0)
        # Non-block row 2 should be identity
        assert D_dense[2, 2] == 1.0

    def test_boundary_jacobian_matches_fd(self):
        """Tangent cone on boundary matches finite-difference Jacobian."""
        mu = 0.7
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
        )
        z = np.array([1.0, 3.0])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)

        # Finite-difference Jacobian of full projection
        eps = 1e-7
        f0 = proj.project(z, z)
        J_fd = np.zeros((2, 2))
        for j in range(2):
            z_p = z.copy(); z_p[j] += eps
            f1 = proj.project(z_p, z_p)
            J_fd[:, j] = (f1 - f0) / eps

        np.testing.assert_allclose(D_dense, J_fd, atol=1e-5)

    def test_3d_tangent_cone_matches_fd(self):
        """3-D tangent cone matches finite differences."""
        mu = 0.3
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: mu,
        )
        z = np.array([1.0, 2.0, -1.5])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)

        eps = 1e-7
        f0 = proj.project(z, z)
        J_fd = np.zeros((3, 3))
        for j in range(3):
            z_p = z.copy(); z_p[j] += eps
            f1 = proj.project(z_p, z_p)
            J_fd[:, j] = (f1 - f0) / eps

        np.testing.assert_allclose(D_dense, J_fd, atol=1e-5)

    def test_inactive_block_identity_rows(self):
        """Inactive block → identity rows in tangent cone."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            gap_func=lambda y, t: np.array([1.0]),
        )
        z = np.array([0.0, 2.0])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)
        np.testing.assert_allclose(D_dense, np.eye(2))

    def test_sparse_format(self):
        """Tangent cone returns sparse CSR for large n, dense for small n."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        # Small system — dense path (ndarray)
        z_small = np.array([1.0, 3.0])
        D_small = proj.tangent_cone(z_small, z_small)
        assert isinstance(D_small, np.ndarray)

        # Large system — sparse CSR path
        n = 100
        proj_big = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        z_big = np.zeros(n)
        z_big[0] = 1.0; z_big[1] = 3.0
        D_big = proj_big.tangent_cone(z_big, z_big)
        assert sp.issparse(D_big)
        assert D_big.format == 'csr'

    def test_two_blocks_mixed_regions(self):
        """Two blocks in different regions — tangent cone assembled correctly."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
        )
        z = np.array([4.0, 1.0, -4.0, 0.5])
        D = proj.tangent_cone(z, z)
        D_dense = D.toarray() if sp.issparse(D) else np.asarray(D)
        # Block 0: interior → identity rows
        np.testing.assert_allclose(D_dense[:2, :2], np.eye(2))
        np.testing.assert_allclose(D_dense[:2, 2:], 0.0)
        # Block 1: polar → zero rows
        np.testing.assert_allclose(D_dense[2:, :], 0.0)

    def test_large_batch_tangent_keeps_fixed_pattern_with_exact_values(self):
        """Large uniform SOC batches reuse one CSR pattern while updating values exactly."""
        blocks = [(2 * k, [2 * k + 1]) for k in range(40)]  # n = 80 > batch threshold
        mu = 0.7
        proj = MuScaledSOCProjection(
            blocks=blocks,
            get_mu=lambda y: mu,
        )

        z_interior = np.zeros(80)
        z_interior[0::2] = 4.0
        z_interior[1::2] = 1.0
        D_interior = proj.tangent_cone(z_interior, z_interior).copy()
        assert sp.isspmatrix_csr(D_interior)
        np.testing.assert_array_equal(D_interior[:2, :2].toarray(), np.eye(2))

        z_boundary = np.zeros(80)
        z_boundary[0] = 1.0
        z_boundary[1] = 3.0
        z_boundary[2::2] = 4.0
        z_boundary[3::2] = 1.0
        D_boundary = proj.tangent_cone(z_boundary, z_boundary).copy()
        assert sp.isspmatrix_csr(D_boundary)

        np.testing.assert_array_equal(D_interior.indptr, D_boundary.indptr)
        np.testing.assert_array_equal(D_interior.indices, D_boundary.indices)

        proj_small = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
        )
        D_expected = proj_small.tangent_cone(
            np.array([1.0, 3.0]),
            np.array([1.0, 3.0]),
        )
        D_expected = D_expected if isinstance(D_expected, np.ndarray) else D_expected.toarray()

        np.testing.assert_allclose(D_boundary[:2, :2].toarray(), D_expected, atol=1e-12)
        np.testing.assert_allclose(D_boundary[2:4, 2:4].toarray(), np.eye(2), atol=1e-12)


# =====================================================================
# get_mu arity detection
# =====================================================================

class TestMuArity:
    """Tests for auto-detection of get_mu signature."""

    def test_mu_1arg(self):
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.5)
        z = np.array([0.0, 2.0])
        result = proj.project(z, z)
        # alpha=0.8, beta=1 → [0.8, 0.4]
        np.testing.assert_allclose(result, [0.8, 0.4])

    def test_mu_2arg(self):
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y, t: 0.5)
        z = np.array([0.0, 2.0])
        result = proj.project(z, z, t=1.0)
        np.testing.assert_allclose(result, [0.8, 0.4])

    def test_mu_3arg(self):
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y, t, Fk: 0.5)
        z = np.array([0.0, 2.0])
        result = proj.project(z, z, t=1.0, Fk_val=np.zeros(2))
        np.testing.assert_allclose(result, [0.8, 0.4])


# =====================================================================
# Edge cases / validation
# =====================================================================

class TestEdgeCases:
    """Edge-case and validation tests."""

    def test_no_blocks_raises(self):
        with pytest.raises(ValueError, match="blocks must be provided"):
            MuScaledSOCProjection(blocks=None, get_mu=lambda y: 0.5)

    def test_empty_w_raises(self):
        with pytest.raises(ValueError, match="at least one tangential"):
            MuScaledSOCProjection(blocks=[(0, [])], get_mu=lambda y: 0.5)

    def test_wrong_mu_size_raises(self):
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y: np.array([0.5, 0.6]))
        z = np.array([0.0, 2.0])
        with pytest.raises(ValueError, match="scalar or array of length"):
            proj.project(z, z)

    def test_project_preserves_extra_dofs(self):
        """Non-block DOFs must be preserved exactly."""
        proj = MuScaledSOCProjection(blocks=[(1, [2])], get_mu=lambda y: 1.0)
        z = np.array([42.0, 0.0, 2.0, -7.0])
        result = proj.project(z, z)
        assert result[0] == 42.0
        assert result[3] == -7.0

    def test_zero_vector(self):
        """Zero input → zero output (on cone boundary / origin)."""
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.5)
        z = np.zeros(2)
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, np.zeros(2))

    def test_large_mu(self):
        """Very large μ → wide cone, most points interior."""
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 1000.0)
        z = np.array([0.001, 1.0])  # mu*s = 1 >= |w|=1 ✓
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, z)


# =====================================================================
# zero_inactive tests
# =====================================================================

class TestZeroInactive:
    """Tests for zero_inactive=True: inactive blocks project onto {0}."""

    def test_inactive_block_zeroed(self):
        """When gap > 0 and zero_inactive=True, block is set to zero."""
        gap = lambda y, t: np.array([1.0])  # gap > 0 → inactive
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=True,
        )
        z = np.array([3.0, 2.0, 99.0])
        result = proj.project(z, z)
        # Block DOFs zeroed, non-block DOFs preserved
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 99.0

    def test_inactive_block_identity_default(self):
        """When gap > 0 and zero_inactive=False (default), block is unchanged."""
        gap = lambda y, t: np.array([1.0])  # gap > 0 → inactive
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=False,
        )
        z = np.array([3.0, 2.0, 99.0])
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, z)  # all unchanged

    def test_active_block_projected_normally(self):
        """When gap <= 0 and zero_inactive=True, normal projection applies."""
        gap = lambda y, t: np.array([-1.0])  # gap <= 0 → active
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=True,
        )
        # Polar region: s < 0 → project to zero
        z = np.array([-3.0, 1.0, 99.0])
        result = proj.project(z, z)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 99.0

    def test_inactive_tangent_cone_zero(self):
        """Tangent cone Jacobian for inactive block with zero_inactive=True is zero."""
        gap = lambda y, t: np.array([1.0])  # inactive
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=True,
        )
        z = np.array([3.0, 2.0, 99.0])
        J = proj.tangent_cone(z, z)
        if hasattr(J, 'toarray'):
            J = J.toarray()
        # Block rows/cols should be zero
        np.testing.assert_array_equal(J[0, :], 0.0)
        np.testing.assert_array_equal(J[1, :], 0.0)
        # Non-block rows remain identity
        assert J[2, 2] == 1.0

    def test_inactive_tangent_cone_identity_default(self):
        """Tangent cone Jacobian for inactive block with zero_inactive=False is identity."""
        gap = lambda y, t: np.array([1.0])  # inactive
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=False,
        )
        z = np.array([3.0, 2.0, 99.0])
        J = proj.tangent_cone(z, z)
        if hasattr(J, 'toarray'):
            J = J.toarray()
        np.testing.assert_array_almost_equal(J, np.eye(3))

    def test_mixed_active_inactive(self):
        """Two blocks: one active, one inactive with zero_inactive=True."""
        def gap(y, t):
            return np.array([-1.0, 1.0])  # block 0 active, block 1 inactive
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])], get_mu=lambda y: 0.5,
            gap_func=gap, zero_inactive=True,
        )
        z = np.array([4.0, 1.0, 3.0, 2.0, 99.0])
        result = proj.project(z, z)
        # Block 0: active, inside cone (mu*4=2>=1) → unchanged
        assert result[0] == 4.0
        assert result[1] == 1.0
        # Block 1: inactive → zeroed
        assert result[2] == 0.0
        assert result[3] == 0.0
        # Non-block DOF preserved
        assert result[4] == 99.0
