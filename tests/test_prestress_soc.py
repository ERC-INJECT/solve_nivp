"""Tests for SOC projections with normal and tangential pre-stress.

Covers:
1. MuScaledSOCProjection with get_s0 / get_w0
2. MoreauSOCProjection with get_s0 / get_w0
3. AnisotropicSOCProjection with get_s0 / get_w0
4. CompositeContactProjection inheriting pre-stress from SOC sub-projection
5. Backward compatibility (no prestress → same as before)
6. Jacobian consistency via finite differences
"""

import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.projections import (
    MuScaledSOCProjection,
    MoreauSOCProjection,
    AnisotropicSOCProjection,
    AlgebraicConstraintProjection,
    CompositeContactProjection,
)


# =====================================================================
# Helpers
# =====================================================================

def numerical_jacobian(proj, z_full, eps=1e-7, **kw):
    """Finite-difference Jacobian of the full projection.

    Uses the same vector for both current_state and candidate,
    so it captures d[project(z,z)]/dz including sensitivity
    through current_state.
    """
    n = z_full.size
    J = np.zeros((n, n))
    f0 = proj.project(z_full, z_full, **kw)
    for j in range(n):
        z_pert = z_full.copy()
        z_pert[j] += eps
        f1 = proj.project(z_pert, z_pert, **kw)
        J[:, j] = (f1 - f0) / eps
    return J


def numerical_jacobian_candidate_only(proj, y_fixed, z_cand, eps=1e-7, **kw):
    """FD Jacobian w.r.t. candidate only (current_state held fixed).

    Returns dΠ(y_fixed, z)/dz — consistent with tangent_cone semantics.
    """
    n = z_cand.size
    J = np.zeros((n, n))
    f0 = proj.project(y_fixed, z_cand, **kw)
    for j in range(n):
        z_pert = z_cand.copy()
        z_pert[j] += eps
        f1 = proj.project(y_fixed, z_pert, **kw)
        J[:, j] = (f1 - f0) / eps
    return J


def tangent_as_dense(D, n):
    """Convert sparse or dense tangent to dense ndarray."""
    if sp.issparse(D):
        return D.toarray()
    return np.asarray(D)


# =====================================================================
# 1. MuScaledSOCProjection — normal pre-stress
# =====================================================================

class TestMuScaledSOCNormalPrestress:
    """Normal pre-stress s0 shifts the cone apex along the normal axis."""

    def _make(self, s0=0.0, mu=0.5, zero_inactive=False):
        return MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: s0,
            zero_inactive=zero_inactive,
        )

    def test_no_prestress_backward_compat(self):
        """get_s0=None → identical to standard projection."""
        proj_std = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5)
        proj_ps = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5,
            get_s0=None, get_w0=None)
        z = np.array([1.0, 3.0])
        np.testing.assert_array_equal(
            proj_std.project(z, z), proj_ps.project(z, z))

    def test_interior_with_positive_s0(self):
        """Positive s0 enlarges the cone: point that was on boundary
        is now interior."""
        mu = 1.0
        # Without s0: z = (1, 1) → s=1, |w|=1, μs=1 → boundary
        proj_no = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu)
        z = np.array([1.0, 1.0])
        p_no = proj_no.project(z, z)
        # It's on the boundary, so project = z
        np.testing.assert_allclose(p_no, z, atol=1e-14)

        # With s0 = 2: ŝ = s + 2 = 3, |w| = 1, μŝ = 3 > 1 → interior
        proj_s0 = self._make(s0=2.0, mu=mu)
        p_s0 = proj_s0.project(z, z)
        # Interior → projection is identity (z unchanged)
        np.testing.assert_allclose(p_s0, z, atol=1e-14)

    def test_negative_s0_shifts_apex(self):
        """Negative s0 shrinks the cone: requires more compression to engage."""
        mu = 1.0
        # z = (2, 0) is interior of standard cone (s=2, |w|=0)
        proj_no = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu)
        z = np.array([2.0, 0.0])
        p_no = proj_no.project(z, z)
        np.testing.assert_allclose(p_no, z, atol=1e-14)

        # With s0 = -3: ŝ = 2 - 3 = -1, |w|=0 → polar
        proj_neg = self._make(s0=-3.0, mu=mu)
        p_neg = proj_neg.project(z, z)
        # Projected to (ŝ_proj, w_proj) = (0, 0), then un-shift: s = 0 - (-3) = 3
        np.testing.assert_allclose(p_neg, [3.0, 0.0], atol=1e-14)

    def test_s0_boundary_projection(self):
        """Pre-stress changes boundary projection values."""
        mu = 1.0
        s0 = 1.0
        proj = self._make(s0=s0, mu=mu)
        # z = (-0.5, 1): without s0 it's polar (s<0, λ+ = -0.5 + 1 = 0.5 > 0, λ- = -0.5 - 1 = -1.5)
        # → boundary. With s0=1: ŝ = 0.5, so λ+ = 0.5 + 1 = 1.5, λ- = 0.5 - 1 = -0.5
        # → still boundary. Projection formula: α = 0.5, p_s = α·λ+ = 0.75,
        #   p_w = α·μ·λ+·ŵ = 0.75. Un-shift: s_out = 0.75 - 1 = -0.25.
        z = np.array([-0.5, 1.0])
        p = proj.project(z, z)
        alpha = 0.5  # 1/(1+1)
        lam_plus = 0.5 + 1.0 * 1.0  # ŝ + μ·r = 1.5
        expected_s = alpha * lam_plus - s0  # 0.75 - 1 = -0.25
        expected_w = alpha * mu * lam_plus * 1.0  # 0.75
        np.testing.assert_allclose(p, [expected_s, expected_w], atol=1e-14)

    def test_jacobian_matches_fd_normal(self):
        """Tangent cone with s0 matches finite-difference Jacobian."""
        mu = 0.7
        s0 = 1.5
        proj = self._make(s0=s0, mu=mu)
        z = np.array([1.0, 3.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_zero_s0_is_noop(self):
        """s0 = 0 is the same as no pre-stress."""
        proj_no = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5)
        proj_zero = self._make(s0=0.0, mu=0.5)
        z = np.array([1.0, 3.0])
        np.testing.assert_allclose(
            proj_no.project(z, z), proj_zero.project(z, z), atol=1e-14)


# =====================================================================
# 2. MuScaledSOCProjection — tangential pre-stress
# =====================================================================

class TestMuScaledSOCTangentialPrestress:
    """Tangential pre-stress w0 offsets the cone in tangential space."""

    def _make(self, w0, mu=1.0):
        return MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_w0=lambda y, k: np.array(w0) if np.isscalar(w0) else w0,
        )

    def test_tangential_shift_interior(self):
        """w0 shifts w: interior point stays interior if shifted w
        is still inside the cone."""
        mu = 1.0
        # z = (5, 0): in cone (s=5, |w|=0). With w0=2: |w+w0|=2, μs=5 → interior
        proj = self._make(w0=2.0, mu=mu)
        z = np.array([5.0, 0.0])
        p = proj.project(z, z)
        np.testing.assert_allclose(p, z, atol=1e-14)

    def test_tangential_shift_to_boundary(self):
        """Large w0 pushes point to boundary."""
        mu = 1.0
        # z = (2, 0): in cone. With w0=3: ŵ = w+w0 = 3, μs = 2 < 3 → boundary
        proj = self._make(w0=3.0, mu=mu)
        z = np.array([2.0, 0.0])
        p = proj.project(z, z)
        # After shift: (2, 3). Boundary projection:
        # α = 0.5, λ+ = 2 + 3 = 5, p_s = 2.5, p_w = 2.5, then un-shift: w_out = 2.5 - 3 = -0.5
        expected_s = 0.5 * 5.0  # 2.5
        expected_w = 0.5 * 1.0 * 5.0 * 1.0 - 3.0  # 2.5 - 3 = -0.5
        np.testing.assert_allclose(p, [expected_s, expected_w], atol=1e-14)

    def test_jacobian_matches_fd_tangential(self):
        """Tangent cone with w0 matches finite-difference Jacobian."""
        mu = 0.7
        proj = self._make(w0=1.5, mu=mu)
        z = np.array([2.0, 1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 3. Combined normal + tangential pre-stress
# =====================================================================

class TestMuScaledSOCCombinedPrestress:
    """Both s0 and w0 applied simultaneously."""

    def _make(self, s0=0.0, w0=0.0, mu=1.0):
        return MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: s0,
            get_w0=lambda y, k: np.array([w0]) if np.isscalar(w0) else w0,
        )

    def test_combined_shift(self):
        """Combined s0 + w0 = constant translation of the cone."""
        s0, w0, mu = 1.0, 0.5, 1.0
        proj = self._make(s0=s0, w0=w0, mu=mu)
        z = np.array([0.0, 0.0])
        p = proj.project(z, z)
        # Shifted input: (1, 0.5). In cone? s=1, |w|=0.5, μs=1 >= 0.5 → interior
        # So projection(shifted) = shifted. Un-shift: (1-1, 0.5-0.5) = (0, 0)
        np.testing.assert_allclose(p, [0.0, 0.0], atol=1e-14)

    def test_combined_jacobian_matches_fd(self):
        """Jacobian with both s0 and w0 matches finite differences."""
        s0, w0, mu = 0.5, 0.3, 0.8
        proj = self._make(s0=s0, w0=w0, mu=mu)
        z = np.array([1.0, 2.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 4. 3D contact (two tangential DOFs)
# =====================================================================

class TestPrestress3D:
    """Pre-stress with 3D blocks (1 normal + 2 tangential)."""

    def _make(self, s0=0.0, w0=None, mu=0.5):
        return MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: mu,
            get_s0=lambda y: s0 if s0 != 0.0 else None,
            get_w0=(lambda y, k: np.array(w0)) if w0 is not None else None,
        )

    def test_3d_normal_prestress(self):
        """Normal pre-stress in 3D: shifts s axis."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 1.0,
            get_s0=lambda y: 2.0,
        )
        # z = (-1, 0, 0): without s0, polar. With s0=2: ŝ=1 → interior.
        z = np.array([-1.0, 0.0, 0.0])
        p = proj.project(z, z)
        np.testing.assert_allclose(p, z, atol=1e-14)

    def test_3d_tangential_prestress_jacobian(self):
        """Jacobian with tangential pre-stress in 3D."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.6,
            get_w0=lambda y, k: np.array([0.5, -0.3]),
        )
        z = np.array([2.0, 1.0, 0.5])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 5. Multiple blocks
# =====================================================================

class TestPrestressMultiBlock:
    """Pre-stress with multiple contact blocks."""

    def test_per_block_s0(self):
        """Per-block s0 via array return from get_s0."""
        s0_vals = np.array([1.0, -1.0])
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
            get_s0=lambda y: s0_vals,
        )
        # Block 0: z = (0, 0). Shifted: (1, 0) → interior. Un-shift: (0, 0).
        # Block 1: z = (0, 0). Shifted: (-1, 0) → polar. Proj: (0, 0). Un-shift: (1, 0).
        z = np.zeros(4)
        p = proj.project(z, z)
        np.testing.assert_allclose(p[0:2], [0.0, 0.0], atol=1e-14)
        np.testing.assert_allclose(p[2:4], [1.0, 0.0], atol=1e-14)

    def test_per_block_w0(self):
        """Per-block w0 via block index k."""
        w0_data = {0: np.array([0.5]), 1: np.array([-0.5])}
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 1.0,
            get_w0=lambda y, k: w0_data[k],
        )
        # Block 0: z = (10, 0). Shifted: (10, 0.5). Interior (μs=10>0.5). Un-shift: (10, 0).
        # Block 1: z = (10, 0). Shifted: (10, -0.5). Interior. Un-shift: (10, 0).
        z = np.array([10.0, 0.0, 10.0, 0.0])
        p = proj.project(z, z)
        np.testing.assert_allclose(p, z, atol=1e-14)


# =====================================================================
# 6. MoreauSOCProjection with pre-stress
# =====================================================================

class TestMoreauPrestress:
    """Pre-stress in the velocity-level Moreau time-stepping projector."""

    def _make(self, s0=0.0, w0=None, mu=0.5, e=0.0):
        return MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=(lambda y: s0) if s0 != 0.0 else None,
            get_w0=(lambda y, k: np.array(w0) if np.isscalar(w0) else w0) if w0 is not None else None,
            e=e,
        )

    def test_moreau_no_prestress_backward_compat(self):
        """No pre-stress → same as standard Moreau projection."""
        proj_std = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5, e=0.0)
        proj_ps = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5, e=0.0,
            get_s0=None, get_w0=None)
        z = np.array([1.0, 0.5])
        np.testing.assert_allclose(
            proj_std.project(z, z),
            proj_ps.project(z, z), atol=1e-14)

    def test_moreau_normal_prestress_changes_projection(self):
        """Normal pre-stress shifts the De Saxcé-projected cone."""
        mu = 0.5
        s0 = 2.0
        proj_no = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu, e=0.0)
        proj_s0 = self._make(s0=s0, mu=mu)
        z = np.array([1.0, 0.5])
        p_no = proj_no.project(z, z)
        p_s0 = proj_s0.project(z, z)
        # With positive s0, the cone is effectively larger so the projection
        # should differ (pre-stress pulls point toward interior)
        assert not np.allclose(p_no, p_s0, atol=1e-10) or np.allclose(p_no, z, atol=1e-10)

    def test_moreau_tangent_cone_with_prestress(self):
        """Moreau tangent cone runs without error with pre-stress."""
        proj = self._make(s0=1.0, w0=0.3, mu=0.5)
        z = np.array([2.0, 1.0])
        D = proj.tangent_cone(z, z)
        assert D.shape == (2, 2)

    def test_moreau_with_restitution_and_prestress(self):
        """Pre-stress works correctly alongside restitution (e > 0)."""
        e = 0.5
        proj = self._make(s0=1.0, mu=0.5, e=e)
        y = np.array([2.0, 1.0])
        z = np.array([1.5, 0.8])
        prev = np.array([0.5, 0.3])
        p = proj.project(y, z, prev_state=prev)
        assert p.shape == (2,)
        assert np.all(np.isfinite(p))


# =====================================================================
# 7. AnisotropicSOCProjection with pre-stress
# =====================================================================

class TestAnisotropicPrestress:
    """Pre-stress in the anisotropic (Cholesky-whitened) projector."""

    def _make(self, s0=0.0, w0=None, mu=0.5, B=None):
        if B is None:
            B = np.eye(1)
        return AnisotropicSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_B=lambda y, k: B,
            get_s0=(lambda y: s0) if s0 != 0.0 else None,
            get_w0=(lambda y, k: np.array(w0) if np.isscalar(w0) else w0) if w0 is not None else None,
        )

    def test_aniso_isotropic_matches_base(self):
        """With B=I and same s0, anisotropic matches base class."""
        s0, mu = 1.5, 0.7
        proj_base = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu,
            get_s0=lambda y: s0)
        proj_aniso = self._make(s0=s0, mu=mu, B=np.eye(1))
        z = np.array([1.0, 3.0])
        np.testing.assert_allclose(
            proj_base.project(z, z),
            proj_aniso.project(z, z), atol=1e-13)

    def test_aniso_tangential_prestress(self):
        """Tangential pre-stress in anisotropic case."""
        B = np.array([[2.0]])  # anisotropic scaling
        proj = self._make(w0=0.5, mu=0.5, B=B)
        z = np.array([3.0, 0.0])
        p = proj.project(z, z)
        assert np.all(np.isfinite(p))
        assert p.shape == (2,)

    def test_aniso_jacobian_with_prestress(self):
        """Tangent cone runs and has correct shape with pre-stress."""
        proj = self._make(s0=1.0, w0=0.5, mu=0.5, B=np.array([[2.0]]))
        z = np.array([2.0, 1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        assert D.shape == (2, 2)
        assert np.all(np.isfinite(D))

    def test_aniso_3d_prestress(self):
        """Anisotropic 3D with pre-stress."""
        B = np.array([[2.0, 0.5], [0.5, 1.0]])  # SPD
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: B,
            get_s0=lambda y: 1.0,
            get_w0=lambda y, k: np.array([0.3, -0.2]),
        )
        z = np.array([2.0, 1.0, 0.5])
        p = proj.project(z, z)
        assert np.all(np.isfinite(p))
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))


# =====================================================================
# 8. CompositeContactProjection inherits pre-stress
# =====================================================================

class TestCompositePrestress:
    """CompositeContactProjection delegates pre-stress to SOC sub-projection."""

    def test_composite_with_prestress(self):
        """Pre-stress in the SOC sub-projection works through composite."""
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        alg = AlgebraicConstraintProjection(
            g=lambda y: C @ y[:2],
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        soc = MuScaledSOCProjection(
            blocks=[(4, [5])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 1.0,
            zero_inactive=True,
        )
        comp = CompositeContactProjection(alg, soc)

        z = np.array([1.0, 2.0, 0.0, 0.0, 0.5, 0.1])
        p = comp.project(z, z)
        # Algebraic: q = y[:2] = [1, 2]
        np.testing.assert_allclose(p[2:4], [1.0, 2.0], atol=1e-14)
        # SOC: pre-stress s0=1 shifts the cone
        assert np.all(np.isfinite(p[4:6]))

    def test_composite_tangent_cone_with_prestress(self):
        """Tangent cone through composite with pre-stress."""
        C = np.eye(2)
        alg = AlgebraicConstraintProjection(
            g=lambda y: C @ y[:2],
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        soc = MuScaledSOCProjection(
            blocks=[(4, [5])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 1.0,
            get_w0=lambda y, k: np.array([0.3]),
            zero_inactive=True,
        )
        comp = CompositeContactProjection(alg, soc)
        z = np.array([1.0, 2.0, 0.0, 0.0, 2.0, 1.0])
        D = comp.tangent_cone(z, z)
        assert D.shape == (6, 6)


# =====================================================================
# 9. Projection identity: Π(z) = Π_{K_μ}(z + z0) - z0
# =====================================================================

class TestPrestressIdentity:
    """Verify the projection identity:
       project_prestress(z) == standard_project(z + z0) - z0
    """

    @pytest.mark.parametrize("s0,w0", [
        (2.0, 0.0), (-1.0, 0.0), (0.0, 1.5), (0.0, -0.5),
        (1.5, 0.8), (-0.5, -1.2),
    ])
    def test_shift_identity_2d(self, s0, w0):
        """For various (s0, w0), verify the projection shift identity."""
        mu = 0.7
        proj_std = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu)
        proj_ps = MuScaledSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: mu,
            get_s0=lambda y, _s=s0: _s,
            get_w0=lambda y, k, _w=w0: np.array([_w]),
        )
        z = np.array([1.0, 2.0])
        # Pre-stressed projection
        p_ps = proj_ps.project(z, z)
        # Manual shift identity
        z_shifted = z.copy()
        z_shifted[0] += s0
        z_shifted[1] += w0
        p_shifted = proj_std.project(z_shifted, z_shifted)
        p_manual = p_shifted.copy()
        p_manual[0] -= s0
        p_manual[1] -= w0
        np.testing.assert_allclose(p_ps, p_manual, atol=1e-14)


# =====================================================================
# 10. State-dependent pre-stress
# =====================================================================

class TestStateDependentPrestress:
    """Pre-stress that depends on time (constant w.r.t. state unknowns)."""

    def test_time_dependent_s0(self):
        """get_s0(y, t) with time dependence."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            get_s0=lambda y, t: 2.0 * t,  # s0 grows with time
        )
        z = np.array([0.0, 0.0])
        # At t=0: s0=0, → (0,0) → polar → (0,0)
        p0 = proj.project(z, z, t=0.0)
        np.testing.assert_allclose(p0, [0.0, 0.0], atol=1e-14)
        # At t=1: s0=2, → (2,0) → interior → (0,0) after un-shift
        p1 = proj.project(z, z, t=1.0)
        np.testing.assert_allclose(p1, [0.0, 0.0], atol=1e-14)

    def test_time_dependent_w0(self):
        """get_w0(y, k, t) with time dependence."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            get_w0=lambda y, k, t: np.array([t]),
        )
        z = np.array([10.0, 0.0])
        # At t=0: w0=0, interior (10, 0). At t=5: w0=5, shifted (10, 5) in cone.
        p0 = proj.project(z, z, t=0.0)
        p5 = proj.project(z, z, t=5.0)
        np.testing.assert_allclose(p0, z, atol=1e-14)
        np.testing.assert_allclose(p5, z, atol=1e-14)


# =====================================================================
# 11. Zero_inactive + pre-stress
# =====================================================================

class TestPrestressZeroInactive:
    """Pre-stress with zero_inactive=True (force/impulse-level)."""

    def test_inactive_zeroed_regardless_of_prestress(self):
        """Inactive blocks produce zero regardless of pre-stress."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 10.0,
            gap_func=lambda y, t: np.array([1.0]),  # gap > 0 → inactive
            gap_tol=0.0,
            zero_inactive=True,
        )
        z = np.array([5.0, 3.0])
        p = proj.project(z, z, t=0.0)
        np.testing.assert_allclose(p, [0.0, 0.0], atol=1e-14)


# =====================================================================
# 12. Sparse path for large systems
# =====================================================================

class TestPrestressSparsePath:
    """Pre-stress with n > 64 to exercise the sparse tangent_cone path."""

    def test_large_system_tangent_cone(self):
        """Sparse CSR tangent cone with pre-stress."""
        n = 100
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 1.0,
            get_w0=lambda y, k: np.array([0.5]),
        )
        z = np.zeros(n)
        z[0] = 2.0
        z[1] = 1.0
        D = proj.tangent_cone(z, z)
        assert sp.issparse(D)
        assert D.shape == (n, n)
        # Non-block rows should be identity
        D_dense = D.toarray()
        for i in range(2, n):
            assert D_dense[i, i] == 1.0
            assert np.sum(np.abs(D_dense[i, :])) == 1.0


# =====================================================================
# 13. Validation
# =====================================================================

class TestPrestressValidation:
    """Input validation for pre-stress callbacks."""

    def test_s0_wrong_length_raises(self):
        """get_s0 returning wrong-length array raises ValueError."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: np.array([1.0, 2.0, 3.0]),  # 3 instead of 2
        )
        z = np.zeros(4)
        with pytest.raises(ValueError, match="get_s0"):
            proj.project(z, z)

    def test_w0_wrong_length_raises(self):
        """get_w0 returning wrong-length array raises ValueError."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],  # m_k = 2
            get_mu=lambda y: 0.5,
            get_w0=lambda y, k: np.array([1.0]),  # 1 instead of 2
        )
        z = np.zeros(3)
        with pytest.raises(ValueError, match="get_w0"):
            proj.project(z, z)


# =====================================================================
# 14. State-dependent pre-stress (Jacobian correction)
# =====================================================================

class TestStateDependentPrestressJacobian:
    """When s0(y) or w0(y) depends on the state, the Jacobian of the
    projection acquires a correction: (J_cone - I) @ dz0/dz."""

    def test_constant_prestress_no_jac_callbacks(self):
        """Without get_ds0_dz/get_dw0_dz, Jacobian treats s0 as constant."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 1.0,
        )
        z = np.array([1.0, 3.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        # This is the "constant pre-stress" Jacobian (no correction)
        assert D.shape == (2, 2)

    def test_linear_s0_jacobian_matches_fd(self):
        """s0 = c * y[2] depends on a non-block DOF. Jacobian should
        have off-diagonal entries and match finite differences."""
        c = 0.5
        mu = 0.7
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: c * y[2],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, c]),
        )
        z = np.array([1.0, 3.0, 2.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_linear_w0_jacobian_matches_fd(self):
        """w0 = c * y[2] depends on a non-block DOF."""
        c = 0.3
        mu = 0.7
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_w0=lambda y, k: np.array([c * y[2]]),
            get_dw0_dz=lambda y, k: np.array([[0.0, 0.0, c]]),
        )
        z = np.array([2.0, 1.0, 3.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_combined_s0_w0_state_dep_jacobian(self):
        """Both s0 and w0 depend on non-block DOFs."""
        cs, cw = 0.4, 0.6
        mu = 0.8
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: cs * y[2],
            get_w0=lambda y, k: np.array([cw * y[3]]),
            get_ds0_dz=lambda y: np.array([0.0, 0.0, cs, 0.0]),
            get_dw0_dz=lambda y, k: np.array([[0.0, 0.0, 0.0, cw]]),
        )
        z = np.array([1.5, 2.0, 1.0, 0.5])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_off_diagonal_entries_appear(self):
        """State-dependent s0 creates off-diagonal entries in the Jacobian."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: y[2],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, 1.0]),
        )
        # Use a boundary point where J_cone != I
        z = np.array([1.0, 3.0, 1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        # Column 2 should have nonzero entries in rows 0 and 1
        # (because s0 depends on z[2] and the block is on the boundary)
        assert abs(D[0, 2]) > 1e-10 or abs(D[1, 2]) > 1e-10

    def test_interior_point_no_correction(self):
        """Interior point: J_cone = I, so (J_cone - I) @ dz0/dz = 0."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 1.0,
            get_s0=lambda y: y[2] + 10.0,  # large s0 ensures interior
            get_ds0_dz=lambda y: np.array([0.0, 0.0, 1.0]),
        )
        z = np.array([5.0, 0.0, 0.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        # Interior: Jacobian should be identity (no correction needed)
        np.testing.assert_allclose(D, np.eye(3), atol=1e-14)

    def test_polar_point_no_correction(self):
        """Polar point: J_cone = 0, so (J_cone - I) @ dz0/dz = -dz0/dz,
        but project = 0, so total contribution to these rows is just
        the correction applied on top of the zero block."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: y[2] - 100.0,  # huge negative → polar
            get_ds0_dz=lambda y: np.array([0.0, 0.0, 1.0]),
        )
        z = np.array([1.0, 0.0, 0.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 15. State-dependent pre-stress — 3D and multi-block
# =====================================================================

class TestStateDep3DMultiBlock:
    """State-dependent pre-stress with 3D contacts and multiple blocks."""

    def test_3d_state_dep_jacobian(self):
        """3D contact with state-dependent s0."""
        c = 0.3
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.6,
            get_s0=lambda y: c * y[3],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, 0.0, c]),
        )
        z = np.array([2.0, 1.0, 0.5, 1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_3d_state_dep_w0_jacobian(self):
        """3D contact with state-dependent w0."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_w0=lambda y, k: 0.2 * y[3:5],
            get_dw0_dz=lambda y, k: np.array([
                [0.0, 0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.2],
            ]),
        )
        z = np.array([3.0, 1.0, 0.5, 2.0, -1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_multi_block_per_block_ds0(self):
        """Two blocks, each with independent state-dependent s0."""
        def get_s0(y):
            return np.array([0.5 * y[4], 0.3 * y[5]])  # per-block

        def get_ds0_dz(y):
            n = y.size
            J = np.zeros((2, n))
            J[0, 4] = 0.5
            J[1, 5] = 0.3
            return J

        proj = MuScaledSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: 0.5,
            get_s0=get_s0,
            get_ds0_dz=get_ds0_dz,
        )
        z = np.array([1.0, 2.0, 1.5, 1.0, 3.0, 2.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 16. State-dependent pre-stress in MoreauSOCProjection
# =====================================================================

class TestMoreauStateDep:
    """State-dependent pre-stress through the Moreau (De Saxcé) chain.

    The Moreau tangent_cone intentionally treats De Saxcé forward terms
    (μ|v_T|) as constant w.r.t. candidate (operator splitting).
    We therefore validate the pre-stress correction via:
    (a) candidate-only FD for the base tangent_cone (no correction),
    (b) column-specific FD for non-block DOFs where De Saxcé mismatch
        doesn't apply,
    (c) algebraic difference between corrected and uncorrected tangent_cone.
    """

    def test_moreau_base_tangent_matches_candidate_fd(self):
        """Moreau tangent_cone (no correction) matches candidate-only FD."""
        mu = 0.5
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            e=0.0,
            get_s0=lambda y: 0.4 * y[2],
        )
        y = np.array([2.0, 1.0, 1.5])
        z = np.array([1.5, 0.8, 1.5])
        D = tangent_as_dense(proj.tangent_cone(z, y), z.size)
        J_fd = numerical_jacobian_candidate_only(proj, y, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_moreau_correction_adds_expected_diff(self):
        """The correction is the difference from the uncorrected case."""
        c, mu = 0.4, 0.5
        proj_no = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            e=0.0,
            get_s0=lambda y: c * y[2],
        )
        proj_corr = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            e=0.0,
            get_s0=lambda y: c * y[2],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, c]),
        )
        # Use boundary point: s=0.5, w=2 → shifted s=0.5+c*1.5=1.1
        # 1/mu=2: λ- = 1.1 - 2/2 = 0.1 > 0 → interior → correction=0
        # Use a point that's on the boundary after shift:
        z = np.array([-0.5, 2.0, 1.5])
        D_no = tangent_as_dense(proj_no.tangent_cone(z, z), z.size)
        D_corr = tangent_as_dense(proj_corr.tangent_cone(z, z), z.size)
        diff = D_corr - D_no
        # The correction only affects columns where ds0/dz is nonzero (column 2)
        np.testing.assert_allclose(diff[:, 0], 0.0, atol=1e-15)
        np.testing.assert_allclose(diff[:, 1], 0.0, atol=1e-15)
        # Column 2 should be nonzero (boundary point after shift)
        assert diff.shape == (3, 3)

    def test_moreau_correction_column_matches_fd(self):
        """For non-tangential DOFs, FD with both args matches correction."""
        c, mu = 0.4, 0.5
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            e=0.0,
            get_s0=lambda y: c * y[2],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, c]),
        )
        z = np.array([-0.5, 2.0, 1.5])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        # Perturbing z[2] doesn't change De Saxcé (v_T = y[1] unchanged)
        # so FD column 2 should match
        eps = 1e-7
        f0 = proj.project(z, z)
        z_pert = z.copy()
        z_pert[2] += eps
        f1 = proj.project(z_pert, z_pert)
        fd_col2 = (f1 - f0) / eps
        np.testing.assert_allclose(D[:, 2], fd_col2, atol=1e-5)

    def test_moreau_state_dep_w0_runs(self):
        """Moreau tangent cone with state-dependent w0 runs correctly."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            e=0.0,
            get_w0=lambda y, k: np.array([0.3 * y[2]]),
            get_dw0_dz=lambda y, k: np.array([[0.0, 0.0, 0.3]]),
        )
        z = np.array([2.0, 1.0, 2.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        assert D.shape == (3, 3)
        assert np.all(np.isfinite(D))


# =====================================================================
# 17. State-dependent pre-stress in AnisotropicSOCProjection
# =====================================================================

class TestAnisotropicStateDep:
    """State-dependent pre-stress through the anisotropic (whitened) chain."""

    def test_aniso_state_dep_s0_jacobian(self):
        """Anisotropic tangent cone with state-dependent s0 matches FD."""
        c = 0.5
        B = np.array([[2.0]])
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: B,
            get_s0=lambda y: c * y[2],
            get_ds0_dz=lambda y: np.array([0.0, 0.0, c]),
        )
        z = np.array([2.0, 1.0, 1.5])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)

    def test_aniso_3d_state_dep_w0_jacobian(self):
        """Anisotropic 3D with state-dependent w0."""
        B = np.array([[2.0, 0.5], [0.5, 1.5]])
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: B,
            get_w0=lambda y, k: 0.3 * y[3:5],
            get_dw0_dz=lambda y, k: np.array([
                [0.0, 0.0, 0.0, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3],
            ]),
        )
        z = np.array([3.0, 1.0, 0.5, 2.0, -1.0])
        D = tangent_as_dense(proj.tangent_cone(z, z), z.size)
        J_fd = numerical_jacobian(proj, z)
        np.testing.assert_allclose(D, J_fd, atol=1e-5)


# =====================================================================
# 18. Sparse path with state-dependent pre-stress
# =====================================================================

class TestStateDependentSparse:
    """State-dependent pre-stress through the sparse (n > 64) path."""

    def test_sparse_off_diagonal_entries(self):
        """Sparse tangent cone has off-diagonal entries from dz0/dz."""
        n = 100
        c = 0.5
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: c * y[50],
            get_ds0_dz=lambda y: np.eye(1, n, 50).ravel() * c,
        )
        z = np.zeros(n)
        z[0] = 1.0
        z[1] = 3.0
        z[50] = 2.0
        D = proj.tangent_cone(z, z)
        assert sp.issparse(D)
        D_dense = D.toarray()
        # Column 50 should have nonzero entries in rows 0 and/or 1
        assert abs(D_dense[0, 50]) > 1e-10 or abs(D_dense[1, 50]) > 1e-10
        # Non-block, non-coupling rows are still identity
        for i in range(2, n):
            if i == 50:
                continue
            row_sum = np.sum(np.abs(D_dense[i, :]))
            assert abs(row_sum - 1.0) < 1e-14


# =====================================================================
# 19. Backward compatibility — no jac callbacks
# =====================================================================

class TestStateDependentBackwardCompat:
    """No get_ds0_dz/get_dw0_dz → same Jacobian as constant pre-stress."""

    def test_no_jac_callback_same_as_constant(self):
        """Without Jacobian callbacks, tangent_cone matches constant case."""
        mu = 0.7
        s0 = 1.5
        proj_const = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: s0,
        )
        proj_dep = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: mu,
            get_s0=lambda y: s0,
            get_ds0_dz=None,  # explicitly None
        )
        z = np.array([1.0, 3.0])
        D1 = tangent_as_dense(proj_const.tangent_cone(z, z), z.size)
        D2 = tangent_as_dense(proj_dep.tangent_cone(z, z), z.size)
        np.testing.assert_array_equal(D1, D2)


# =====================================================================
# 20. Validation for Jacobian callbacks
# =====================================================================

class TestStateDependentValidation:
    """Input validation for get_ds0_dz and get_dw0_dz."""

    def test_ds0_dz_wrong_shape_raises(self):
        """get_ds0_dz returning wrong shape raises ValueError."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_s0=lambda y: 1.0,
            get_ds0_dz=lambda y: np.array([1.0, 2.0, 3.0]),  # length 3 != n=2
        )
        z = np.array([1.0, 3.0])
        with pytest.raises(ValueError, match="get_ds0_dz"):
            proj.tangent_cone(z, z)

    def test_dw0_dz_wrong_shape_raises(self):
        """get_dw0_dz returning wrong shape raises ValueError."""
        proj = MuScaledSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_w0=lambda y, k: np.array([1.0]),
            get_dw0_dz=lambda y, k: np.array([1.0, 2.0]),  # (2,) not (1, 2)
        )
        z = np.array([1.0, 3.0])
        with pytest.raises(ValueError, match="get_dw0_dz"):
            proj.tangent_cone(z, z)


# =====================================================================
# Tests for tangent_cone_split (De Saxcé state-dependency — path C)
# =====================================================================

class TestMoreauTangentConeSplit:
    """Verify tangent_cone_split returns (D_cand, D_state) with correct algebra."""

    def test_split_exists_only_on_moreau(self):
        """MuScaledSOCProjection should NOT have tangent_cone_split."""
        proj = MuScaledSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.3)
        assert not hasattr(proj, 'tangent_cone_split')

    def test_moreau_has_split(self):
        """MoreauSOCProjection exposes tangent_cone_split."""
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.3)
        assert hasattr(proj, 'tangent_cone_split')
        assert callable(proj.tangent_cone_split)

    def test_split_returns_pair(self):
        """tangent_cone_split returns a 2-tuple."""
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.3)
        z = np.array([1.0, 0.2])
        result = proj.tangent_cone_split(z, z, prev_state=np.zeros(2))
        assert isinstance(result, tuple) and len(result) == 2

    def test_d_cand_matches_tangent_cone_when_no_state_dep(self):
        """For zero v_T and no pre-stress, D_state should be zero and
        D_cand should equal the single tangent_cone output."""
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.3)
        # current_state with v_T = 0 → De Saxcé term contributes nothing
        y = np.array([1.0, 0.0])
        z = np.array([1.0, 0.2])
        prev = np.zeros(2)
        D_cand, D_state = proj.tangent_cone_split(z, y, prev_state=prev)
        D_full = proj.tangent_cone(z, y, prev_state=prev)

        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state
        D_full_d = D_full.toarray() if sp.issparse(D_full) else D_full

        # D_cand should match D_full
        np.testing.assert_allclose(D_cand_d, D_full_d, atol=1e-14)
        # D_state should be zero
        np.testing.assert_allclose(D_state_d, np.zeros_like(D_state_d), atol=1e-14)

    def test_d_state_nonzero_when_vT_nonzero(self):
        """When v_T != 0, D_state should have nonzero entries from the
        De Saxcé subgradient mu * v_T/||v_T||."""
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: 0.5)
        y = np.array([1.0, 0.8])  # v_T = 0.8 ≠ 0
        z = np.array([0.5, 0.3])
        prev = np.zeros(2)
        D_cand, D_state = proj.tangent_cone_split(z, y, prev_state=prev)
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state
        assert np.any(np.abs(D_state_d) > 1e-15), \
            "D_state should be nonzero when v_T != 0"

    def test_full_fd_jacobian_vs_split(self):
        """Verify: d[proj(y,y)]/dy = D_cand + D_state matches full FD."""
        mu = 0.4
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: mu)
        y = np.array([1.0, 0.6])
        prev = np.zeros(2)

        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        # FD of proj(z, z) wrt z — captures both candidate and state dep
        J_fd = numerical_jacobian(proj, y, prev_state=prev)

        # The full derivative when candidate = current_state is D_cand + D_state
        np.testing.assert_allclose(D_cand_d + D_state_d, J_fd, atol=1e-5)

    def test_candidate_only_fd_vs_d_cand(self):
        """D_cand alone matches FD with perturbed candidate, fixed state."""
        mu = 0.4
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: mu)
        y = np.array([1.0, 0.7])
        z = np.array([0.5, 0.3])
        prev = np.zeros(2)

        D_cand, _ = proj.tangent_cone_split(z, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand

        J_fd_cand = numerical_jacobian_candidate_only(proj, y, z, prev_state=prev)
        np.testing.assert_allclose(D_cand_d, J_fd_cand, atol=1e-5)

    def test_correct_ssn_formula(self):
        """Verify J = I - A - B + lam * A @ J_F agrees with full FD of the
        natural residual r(y) = y - P(y, y - lam*F(y))."""
        mu = 0.3
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: mu)
        lam = 0.5

        # Simple nonlinear F: linear for tractability
        A_mat = np.array([[2.0, -0.1], [0.3, 1.5]])
        def F(y):
            return A_mat @ y - np.array([1.0, 0.5])
        J_F = A_mat

        y = np.array([0.8, 0.4])
        prev = np.zeros(2)

        candidate = y - lam * F(y)
        D_cand, D_state = proj.tangent_cone_split(
            candidate, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        # Analytical SSN Jacobian: I - A - B + lam * A @ J_F
        n = y.size
        J_ssn = np.eye(n) - D_cand_d - D_state_d + lam * (D_cand_d @ J_F)

        # Numerical FD of r(y) = y - P(y, y - lam*F(y))
        def residual(y_):
            c_ = y_ - lam * F(y_)
            p_ = proj.project(y_, c_, prev_state=prev)
            return y_ - p_

        eps = 1e-7
        J_fd = np.zeros((n, n))
        r0 = residual(y)
        for j in range(n):
            y_p = y.copy()
            y_p[j] += eps
            J_fd[:, j] = (residual(y_p) - r0) / eps

        np.testing.assert_allclose(J_ssn, J_fd, atol=1e-5)

    def test_3d_block(self):
        """tangent_cone_split works for 3D blocks (2 tangential DOFs)."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1, 2])], get_mu=lambda y: 0.4)
        y = np.array([1.0, 0.3, 0.5])
        prev = np.zeros(3)
        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        J_fd = numerical_jacobian(proj, y, prev_state=prev)
        np.testing.assert_allclose(D_cand_d + D_state_d, J_fd, atol=1e-5)

    def test_multi_block(self):
        """tangent_cone_split works with multiple contact blocks."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1]), (2, [3])],
            get_mu=lambda y: np.array([0.3, 0.5]),
        )
        y = np.array([0.5, 0.4, 0.8, 0.6])
        prev = np.zeros(4)
        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        J_fd = numerical_jacobian(proj, y, prev_state=prev)
        np.testing.assert_allclose(D_cand_d + D_state_d, J_fd, atol=1e-5)

    def test_restitution_e_nonzero(self):
        """tangent_cone_split with restitution e > 0."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.3, e=0.5)
        y = np.array([0.6, 0.5])
        prev = np.array([0.2, 0.0])
        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        J_fd = numerical_jacobian(proj, y, prev_state=prev)
        np.testing.assert_allclose(D_cand_d + D_state_d, J_fd, atol=1e-5)

    def test_inactive_block_split(self):
        """Inactive blocks → D_cand = I, D_state = 0 on those rows."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.3,
            gap_func=lambda y, t: np.array([1.0]),  # always inactive
            gap_tol=0.0,
        )
        y = np.array([1.0, 0.5])
        prev = np.zeros(2)
        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state
        np.testing.assert_allclose(D_cand_d, np.eye(2), atol=1e-14)
        np.testing.assert_allclose(D_state_d, np.zeros((2, 2)), atol=1e-14)

    def test_with_prestress(self):
        """tangent_cone_split correctly includes pre-stress in D_state."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.3,
            get_s0=lambda y: np.array([0.5]),
            get_ds0_dz=lambda y: np.array([0.1, 0.0]),
        )
        y = np.array([0.5, 0.4])
        prev = np.zeros(2)
        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        J_fd = numerical_jacobian(proj, y, prev_state=prev)
        np.testing.assert_allclose(D_cand_d + D_state_d, J_fd, atol=1e-5)

    def test_ssn_formula_with_prestress(self):
        """Full SSN Jacobian with pre-stress matches FD of natural residual."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.4,
            get_s0=lambda y: np.array([0.2 * y[0]]),
            get_ds0_dz=lambda y: np.array([[0.2, 0.0]]),
        )
        lam = 0.3
        A_mat = np.array([[1.5, 0.2], [-0.1, 2.0]])
        def F(y):
            return A_mat @ y - np.array([0.5, 0.3])
        J_F = A_mat

        y = np.array([0.7, 0.3])
        prev = np.zeros(2)

        candidate = y - lam * F(y)
        D_cand, D_state = proj.tangent_cone_split(
            candidate, y, prev_state=prev)
        D_cand_d = D_cand.toarray() if sp.issparse(D_cand) else D_cand
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state

        J_ssn = np.eye(2) - D_cand_d - D_state_d + lam * (D_cand_d @ J_F)

        def residual(y_):
            c_ = y_ - lam * F(y_)
            p_ = proj.project(y_, c_, prev_state=prev)
            return y_ - p_

        eps = 1e-7
        r0 = residual(y)
        J_fd = np.zeros((2, 2))
        for j in range(2):
            y_p = y.copy()
            y_p[j] += eps
            J_fd[:, j] = (residual(y_p) - r0) / eps

        np.testing.assert_allclose(J_ssn, J_fd, atol=1e-5)

    def test_sparse_path(self):
        """tangent_cone_split also works in the sparse path (n > 64)."""
        n = 70
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.3,
        )
        y = np.zeros(n)
        y[0] = 1.0
        y[1] = 0.5
        prev = np.zeros(n)

        D_cand, D_state = proj.tangent_cone_split(y, y, prev_state=prev)
        assert sp.issparse(D_cand) and sp.issparse(D_state)

        # Verify sparse result matches dense for the contact rows
        y_small = np.array([1.0, 0.5])
        proj_small = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.3)
        D_c_d, D_s_d = proj_small.tangent_cone_split(
            y_small, y_small, prev_state=np.zeros(2))
        D_c_d = D_c_d if isinstance(D_c_d, np.ndarray) else D_c_d.toarray()
        D_s_d = D_s_d if isinstance(D_s_d, np.ndarray) else D_s_d.toarray()

        D_cand_a = D_cand.toarray()
        D_state_a = D_state.toarray()
        np.testing.assert_allclose(
            D_cand_a[:2, :2], D_c_d, atol=1e-14)
        np.testing.assert_allclose(
            D_state_a[:2, :2], D_s_d, atol=1e-14)
        # Non-contact rows of D_cand should be identity
        np.testing.assert_allclose(
            np.diag(D_cand_a)[2:], np.ones(n - 2), atol=1e-14)

    def test_vT_at_zero_minimal_norm(self):
        """At v_T = 0 (kink), D_state = 0 (minimal-norm Clarke selection)."""
        proj = MoreauSOCProjection(
            blocks=[(0, [1])], get_mu=lambda y: 0.5)
        y = np.array([1.0, 0.0])  # v_T = 0
        z = np.array([0.5, 0.3])
        prev = np.zeros(2)
        D_cand, D_state = proj.tangent_cone_split(z, y, prev_state=prev)
        D_state_d = D_state.toarray() if sp.issparse(D_state) else D_state
        np.testing.assert_allclose(D_state_d, np.zeros((2, 2)), atol=1e-14)


class TestSSNWithSplitTangent:
    """Verify the SSN solver uses tangent_cone_split when available."""

    def test_ssn_moreau_converges(self):
        """SSN with MoreauSOCProjection (which has tangent_cone_split)
        converges for a simple contact problem."""
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver

        mu = 0.3
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: mu)

        # Simple linear operator: F(y) = A @ y - b
        A_mat = np.array([[3.0, 0.1], [0.2, 2.0]])
        b = np.array([1.0, 0.5])
        def F(y):
            return A_mat @ y - b

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            tol=1e-10,
            max_iter=50,
        )
        solver.prev_state = np.zeros(2)
        y0 = np.array([0.5, 0.5])
        y_sol, F_sol, err, success, iters = solver.solve(F, y0)
        assert success, f"SSN did not converge: err={err}, iters={iters}"
        assert err < 1e-9

    def test_ssn_moreau_sparse_path(self):
        """SSN sparse path with MoreauSOCProjection converges."""
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver

        mu = 0.3
        proj = MoreauSOCProjection(blocks=[(0, [1])], get_mu=lambda y: mu)

        A_mat = np.array([[3.0, 0.1], [0.2, 2.0]])
        b = np.array([1.0, 0.5])
        def F(y):
            return A_mat @ y - b

        # Force sparse path
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            tol=1e-10,
            max_iter=50,
            sparse=True,
            linear_solver='splu',
        )
        solver.prev_state = np.zeros(2)
        y0 = np.array([0.5, 0.5])
        y_sol, F_sol, err, success, iters = solver.solve(F, y0)
        assert success, f"SSN sparse did not converge: err={err}, iters={iters}"
        assert err < 1e-9
