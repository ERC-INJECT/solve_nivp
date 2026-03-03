"""Tests for AnisotropicSOCProjection."""

import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.projections import AnisotropicSOCProjection, MuScaledSOCProjection


# =====================================================================
# Helpers
# =====================================================================

def _fd_jacobian(proj, z_full, eps=1e-7):
    """Finite-difference Jacobian of the full projection."""
    n = z_full.size
    f0 = proj.project(z_full, z_full)
    J = np.zeros((n, n))
    for j in range(n):
        z_p = z_full.copy()
        z_p[j] += eps
        f1 = proj.project(z_p, z_p)
        J[:, j] = (f1 - f0) / eps
    return J


def _fd_jacobian_static(proj_func, z, eps=1e-7, **kwargs):
    """Finite-difference Jacobian of a static projection function."""
    f0 = proj_func(z, **kwargs)
    n = z.size
    J = np.zeros((n, n))
    for j in range(n):
        z_p = z.copy()
        z_p[j] += eps
        f1 = proj_func(z_p, **kwargs)
        J[:, j] = (f1 - f0) / eps
    return J


def _proj_aniso(z, mu, B, return_jacobian=False):
    """Convenience wrapper for the static anisotropic projector."""
    B_inv = np.linalg.inv(B)
    return AnisotropicSOCProjection._proj_anisotropic(
        z, mu, B, B_inv, return_jacobian=return_jacobian)


# =====================================================================
# Isotropic B=I should match MuScaledSOCProjection exactly
# =====================================================================

class TestIsotropicEquivalence:
    """AnisotropicSOCProjection with B=I must match isotropic results."""

    @pytest.mark.parametrize("z,mu", [
        (np.array([4.0, 1.0]), 0.5),        # interior 2D
        (np.array([-4.0, 1.0]), 0.5),       # polar 2D
        (np.array([0.0, 2.0]), 1.0),        # boundary 2D
        (np.array([1.0, 3.0]), 0.7),        # boundary 2D
        (np.array([5.0, 1.0, 2.0]), 1.0),   # interior 3D
        (np.array([0.0, 3.0, 4.0]), 0.5),   # boundary 3D
        (np.array([-2.0, 0.3, 0.1]), 0.5),  # polar 3D
        (np.array([1.0, 2.0, -1.5]), 0.3),  # boundary 3D
        (np.array([0.0, 0.0]), 1.0),        # origin 2D
        (np.array([3.0, 0.0]), 0.5),        # w=0, s>0
        (np.array([-1.0, 0.0]), 0.5),       # w=0, s<0
    ])
    def test_value_matches_isotropic(self, z, mu):
        """Projection value with B=I matches isotropic projector."""
        m = z.size - 1
        B = np.eye(m)
        p_aniso = _proj_aniso(z, mu, B)
        p_iso = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        np.testing.assert_allclose(p_aniso, p_iso, atol=1e-12)

    @pytest.mark.parametrize("z,mu", [
        (np.array([4.0, 1.0]), 0.5),
        (np.array([-4.0, 1.0]), 0.5),
        (np.array([0.0, 2.0]), 1.0),
        (np.array([1.0, 3.0]), 0.7),
        (np.array([5.0, 1.0, 2.0]), 1.0),
        (np.array([0.0, 3.0, 4.0]), 0.5),
        (np.array([1.0, 2.0, -1.5]), 0.3),
    ])
    def test_jacobian_matches_isotropic(self, z, mu):
        """Jacobian with B=I matches isotropic Clarke element."""
        m = z.size - 1
        B = np.eye(m)
        _, J_aniso = _proj_aniso(z, mu, B, return_jacobian=True)
        _, J_iso = MuScaledSOCProjection._proj_mu_scaled_soc(
            z, mu, return_jacobian=True)
        np.testing.assert_allclose(J_aniso, J_iso, atol=1e-10)


# =====================================================================
# Core anisotropic tests (B ≠ I)
# =====================================================================

class TestAnisotropicCore:
    """Tests for the anisotropic projector with non-trivial B."""

    def _make_B(self, mu1, mu2):
        """Build B = diag(1/μ₁², 1/μ₂²) for 3D contact (2 tangential)."""
        return np.diag([1.0 / mu1**2, 1.0 / mu2**2])

    def test_interior_anisotropic(self):
        """Interior of elliptic cone."""
        mu = 1.0
        B = self._make_B(2.0, 1.0)  # wider in first direction
        # q(w) = sqrt(w1²/4 + w2²) = sqrt(0.25 + 1) ≈ 1.12
        # mu*s = 3 > 1.12 → interior
        z = np.array([3.0, 1.0, 1.0])
        p = _proj_aniso(z, mu, B)
        np.testing.assert_array_equal(p, z)

    def test_polar_anisotropic(self):
        """Deep inside polar cone."""
        mu = 1.0
        B = self._make_B(2.0, 1.0)
        z = np.array([-10.0, 0.1, 0.1])
        p = _proj_aniso(z, mu, B)
        np.testing.assert_allclose(p, np.zeros(3), atol=1e-12)

    def test_boundary_anisotropic_feasibility(self):
        """Boundary projection lies on cone boundary."""
        mu = 0.5
        B = self._make_B(1.5, 0.8)
        z = np.array([1.0, 3.0, 2.0])
        p = _proj_aniso(z, mu, B)
        s_proj = p[0]
        w_proj = p[1:]
        q_proj = np.sqrt(w_proj @ B @ w_proj)
        # Must lie on boundary: q(w) = μs, s ≥ 0
        assert s_proj >= -1e-14, f"s_proj = {s_proj} < 0"
        np.testing.assert_allclose(q_proj, mu * s_proj, atol=1e-10)

    def test_idempotent_anisotropic(self):
        """Projection is idempotent."""
        mu = 0.6
        B = self._make_B(1.2, 0.7)
        for z in [np.array([1.0, 3.0, 2.0]),
                   np.array([-2.0, 0.5, -0.3]),
                   np.array([0.0, 0.0, 0.0]),
                   np.array([5.0, 0.1, 0.2])]:
            p = _proj_aniso(z, mu, B)
            pp = _proj_aniso(p, mu, B)
            np.testing.assert_allclose(
                pp, p, atol=1e-12,
                err_msg=f"Not idempotent for z={z}")

    def test_boundary_jacobian_matches_fd(self):
        """Boundary Jacobian matches finite differences."""
        mu = 0.5
        B = self._make_B(1.5, 0.8)
        z = np.array([1.0, 3.0, 2.0])
        _, J = _proj_aniso(z, mu, B, return_jacobian=True)

        def proj_fn(z_in, mu=mu, B=B):
            return _proj_aniso(z_in, mu, B)

        J_fd = _fd_jacobian_static(proj_fn, z)
        np.testing.assert_allclose(J, J_fd, atol=1e-5)

    def test_boundary_jacobian_2d_contact(self):
        """2D contact (scalar tangent) Jacobian matches FD."""
        mu = 0.7
        B = np.array([[2.0]])
        z = np.array([1.0, 3.0])
        _, J = _proj_aniso(z, mu, B, return_jacobian=True)

        def proj_fn(z_in, mu=mu, B=B):
            return _proj_aniso(z_in, mu, B)

        J_fd = _fd_jacobian_static(proj_fn, z)
        np.testing.assert_allclose(J, J_fd, atol=1e-5)

    def test_rotated_anisotropy(self):
        """Rotated B matrix — projection still feasible and idempotent."""
        mu = 0.4
        theta = np.pi / 6
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        D = np.diag([1.0, 4.0])
        B = R @ D @ R.T
        z = np.array([1.0, 5.0, -3.0])
        p = _proj_aniso(z, mu, B)
        # Check feasibility
        s_p = p[0]
        w_p = p[1:]
        q_p = np.sqrt(w_p @ B @ w_p)
        assert s_p >= -1e-14
        assert q_p <= mu * s_p + 1e-10
        # Idempotent
        pp = _proj_aniso(p, mu, B)
        np.testing.assert_allclose(pp, p, atol=1e-11)

    def test_mu_zero_anisotropic(self):
        """μ = 0 with anisotropic B still works."""
        B = np.diag([2.0, 3.0])
        z = np.array([3.0, 1.0, 2.0])
        p = _proj_aniso(z, 0.0, B)
        np.testing.assert_allclose(p, [3.0, 0.0, 0.0])


# =====================================================================
# Full block-wise projection API tests
# =====================================================================

class TestBlockProjection:
    """Tests for AnisotropicSOCProjection.project (block-wise API)."""

    def test_single_block_identity_B(self):
        """Single block with B=I should match isotropic."""
        iso = MuScaledSOCProjection(
            blocks=[(0, [1, 2])], get_mu=lambda y: 0.5)
        aniso = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])], get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.eye(2))
        z = np.array([1.0, 3.0, 4.0])
        p_iso = iso.project(z, z)
        p_aniso = aniso.project(z, z)
        np.testing.assert_allclose(p_aniso, p_iso, atol=1e-12)

    def test_two_blocks_different_B(self):
        """Two blocks with different B matrices."""
        def get_B(y, k):
            if k == 0:
                return np.array([[2.0]])
            return np.array([[0.5]])

        proj = AnisotropicSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
            get_B=get_B,
        )
        z = np.array([1.0, 3.0, 1.0, 3.0])
        result = proj.project(z, z)
        # Both should be on boundary (different projections due to B)
        assert result[0] >= 0
        assert result[2] >= 0

    def test_tangent_cone_matches_fd(self):
        """Full block tangent cone matches finite differences."""
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.diag([2.0, 0.5]),
        )
        z = np.array([1.0, 3.0, 2.0])
        D = proj.tangent_cone(z, z)
        D_dense = D if isinstance(D, np.ndarray) else D.toarray()
        J_fd = _fd_jacobian(proj, z)
        np.testing.assert_allclose(D_dense, J_fd, atol=1e-5)

    def test_gap_inactive_block_untouched(self):
        """Inactive block (gap > 0) is left untouched."""
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 1.0,
            get_B=lambda y, k: np.eye(2),
            gap_func=lambda y, t: np.array([5.0]),
        )
        z = np.array([0.0, 3.0, 4.0])
        result = proj.project(z, z)
        np.testing.assert_array_equal(result, z)

    def test_preserves_extra_dofs(self):
        """Non-block DOFs unchanged."""
        proj = AnisotropicSOCProjection(
            blocks=[(1, [2, 3])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.eye(2),
        )
        z = np.array([42.0, 0.0, 3.0, 4.0, -7.0])
        result = proj.project(z, z)
        assert result[0] == 42.0
        assert result[4] == -7.0

    def test_sparse_tangent_cone_large(self):
        """Sparse format for large systems."""
        n = 100
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.eye(2),
        )
        z = np.zeros(n)
        z[0] = 1.0; z[1] = 3.0; z[2] = 4.0
        D = proj.tangent_cone(z, z)
        assert sp.issparse(D)
        assert D.format == 'csr'


# =====================================================================
# ∂p/∂μ sensitivity (informational / future-proofing)
# =====================================================================

class TestMuSensitivity:
    """Verify ∂Π/∂μ formulas for state-dependent μ."""

    def test_boundary_dp_dmu_fd(self):
        """Finite-difference ∂p/∂μ on the boundary matches analytical."""
        mu = 0.5
        z = np.array([1.0, 3.0])
        # Analytical: on boundary, α = 1/(1+μ²), λ₊ = s + μr
        s, r = z[0], abs(z[1])
        w_hat = np.sign(z[1])

        alpha = 1.0 / (1.0 + mu**2)
        lam_plus = s + mu * r
        dalpha_dmu = -2.0 * mu * alpha**2
        dlam_plus_dmu = r

        dp_s_dmu = alpha * dlam_plus_dmu + lam_plus * dalpha_dmu
        p_s = alpha * lam_plus
        dp_w_dmu = (mu * dp_s_dmu + p_s) * w_hat

        # FD check
        eps_mu = 1e-7
        p0 = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu)
        p1 = MuScaledSOCProjection._proj_mu_scaled_soc(z, mu + eps_mu)
        dp_fd = (p1 - p0) / eps_mu

        np.testing.assert_allclose(dp_fd[0], dp_s_dmu, atol=1e-5)
        np.testing.assert_allclose(dp_fd[1], dp_w_dmu, atol=1e-5)
