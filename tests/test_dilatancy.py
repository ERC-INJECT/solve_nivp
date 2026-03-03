"""Tests for dilatancy support (β = tan ψ) in MoreauSOCProjection and contact.py.

The De Saxcé augmentation coefficient is α = μ − β.  When β = 0 (default), α = μ
(standard non-dilatant contact).  When β > 0, sliding imparts a positive normal
opening velocity v_N = β |v_T|.

Key physics to verify:
  - β = 0 reproduces standard Moreau (α = μ)
  - On sliding, v_N = β |v_T|  (dilatancy law)
  - Yield cone unchanged: reaction R ∈ K_μ
  - Jacobians (tangent_cone, tangent_cone_split) match finite-difference
  - contact.py builder accepts beta and produces correct RHS
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from solve_nivp.projections import MoreauSOCProjection, MuScaledSOCProjection


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _make_moreau(mu=0.3, beta=None, e=0.0, blocks=None):
    """Create a 2D (v_N, v_x) Moreau projector with optional dilatancy."""
    if blocks is None:
        blocks = [(0, [1])]
    kw = dict(blocks=blocks, get_mu=lambda y: mu, e=e)
    if beta is not None:
        kw['get_beta'] = beta if callable(beta) else (lambda y, _b=float(beta): _b)
    return MoreauSOCProjection(**kw)


def _fd_jacobian(proj, y, z, eps=1e-7, wrt='candidate', **kw):
    """Finite-difference Jacobian of project wrt candidate or current_state."""
    n = z.size
    J = np.zeros((n, n))
    if wrt == 'candidate':
        p0 = proj.project(y, z, **kw)
        for j in range(n):
            z_p = z.copy(); z_p[j] += eps
            z_m = z.copy(); z_m[j] -= eps
            J[:, j] = (proj.project(y, z_p, **kw) - proj.project(y, z_m, **kw)) / (2 * eps)
    elif wrt == 'current_state':
        p0 = proj.project(y, z, **kw)
        for j in range(n):
            y_p = y.copy(); y_p[j] += eps
            y_m = y.copy(); y_m[j] -= eps
            J[:, j] = (proj.project(y_p, z.copy(), **kw) - proj.project(y_m, z.copy(), **kw)) / (2 * eps)
    return J


def _is_near_kink(y, z, mu, beta=0.0, tol=1e-3):
    """Check if a 2D (v_N, v_T) point is near a nonsmooth kink.

    Kinks occur at:
    - |y_T| = 0  (norm of tangential velocity, affects D_state)
    - SOC eigenvalue λ₊ ≈ 0 or λ₋ ≈ 0  (region boundary of K_{1/μ})
    """
    alpha = mu - beta
    v_T_norm = float(np.linalg.norm(y[1:]))
    if v_T_norm < tol:
        return True
    # Forward De Saxcé transformed point
    s = z[0] + alpha * v_T_norm
    r = float(np.linalg.norm(z[1:]))
    kappa = 1.0 / mu
    lam_plus = s + kappa * r
    lam_minus = s - mu * r
    if abs(lam_plus) < tol or abs(lam_minus) < tol:
        return True
    return False


def _verify_dual_cone(p, mu, alpha, e=0.0, v_N_prev=0.0, atol=1e-10):
    """Verify that the reconstructed û lies in K_{1/μ}.

    û_N = v_N + α|v_T| + e·v_N_prev,  û_T = v_T.
    Must satisfy:  û_N ≥ 0  and  |û_T| ≤ (1/μ)·û_N.
    """
    kappa = 1.0 / mu
    v_N = p[0]
    v_T_norm = float(np.linalg.norm(p[1:]))
    uhat_N = v_N + alpha * v_T_norm + e * v_N_prev
    assert uhat_N >= -atol, f"û_N = {uhat_N} should be ≥ 0"
    assert v_T_norm <= kappa * uhat_N + atol, (
        f"|û_T| = {v_T_norm} should be ≤ κ·û_N = {kappa * uhat_N}")


# ──────────────────────────────────────────────────────────────────────
# Tests: MoreauSOCProjection with dilatancy
# ──────────────────────────────────────────────────────────────────────

class TestMoreauDilatancy:
    """Tests for MoreauSOCProjection with get_beta."""

    def test_beta_zero_matches_standard(self):
        """β = 0 should produce identical results to no-beta projector."""
        proj_std = _make_moreau(mu=0.3, beta=None)
        proj_b0 = _make_moreau(mu=0.3, beta=0.0)

        y = np.array([0.5, 1.2])
        z = np.array([0.8, 0.6])
        p_std = proj_std.project(y, z)
        p_b0 = proj_b0.project(y, z)
        assert_allclose(p_b0, p_std, atol=1e-14)

    def test_beta_zero_tangent_matches_standard(self):
        """β = 0 tangent_cone should match no-beta tangent_cone."""
        proj_std = _make_moreau(mu=0.3, beta=None)
        proj_b0 = _make_moreau(mu=0.3, beta=0.0)

        y = np.array([0.5, 1.2])
        z = np.array([0.8, 0.6])
        D_std = proj_std.tangent_cone(z, y)
        D_b0 = proj_b0.tangent_cone(z, y)
        assert_allclose(D_b0, D_std, atol=1e-14)

    def test_beta_zero_split_matches_standard(self):
        """β = 0 tangent_cone_split should match no-beta tangent_cone_split."""
        proj_std = _make_moreau(mu=0.3, beta=None)
        proj_b0 = _make_moreau(mu=0.3, beta=0.0)

        y = np.array([0.5, 1.2])
        z = np.array([0.8, 0.6])
        A_std, B_std = proj_std.tangent_cone_split(z, y)
        A_b0, B_b0 = proj_b0.tangent_cone_split(z, y)
        assert_allclose(A_b0, A_std, atol=1e-14)
        assert_allclose(B_b0, B_std, atol=1e-14)

    def test_dilatancy_sliding_is_enforced(self):
        """Deterministic boundary-region test: v_N = β|v_T| with û ∈ K_{1/μ}.

        Construct an input guaranteed to land in the boundary (sliding)
        region of the K_{1/μ} projection.  For μ = 0.5, κ = 2, a
        transformed point (s, w) = (0.5, 2.0) satisfies:
          λ₊ = s + κr = 4.5 > 0,  λ₋ = s − μr = −0.5 < 0  → boundary ✓
        """
        mu = 0.5
        beta = 0.2
        alpha = mu - beta   # 0.3
        kappa = 1.0 / mu    # 2.0
        proj = _make_moreau(mu=mu, beta=beta)

        # Forward De Saxcé: z̃_N = z_N + α|y_T|, z̃_T = z_T
        # Choose y_T = 1, z_T = 2.0, and z_N so that z̃ = (0.5, 2.0).
        y = np.array([0.0, 1.0])
        z = np.array([0.5 - alpha, 2.0])   # z_N = 0.2

        p = proj.project(y, z)
        vN, vT = p[0], abs(p[1])

        # 1. Must actually be sliding
        assert vT > 1e-12, "Should be in sliding regime"
        # 2. Dilatancy law: v_N = β|v_T|
        assert_allclose(vN, beta * vT, atol=1e-10)
        # 3. Reconstructed û must lie in K_{1/μ}
        _verify_dual_cone(p, mu, alpha)

    def test_dilatancy_sliding_multiple_beta(self):
        """Deterministic sliding for several β with û ∈ K_{1/μ}.

        Same boundary-region construction for each β value, verifying
        both the sliding law and dual-cone membership.
        """
        mu = 0.5
        kappa = 1.0 / mu  # 2.0
        s_target, w_target = 0.5, 2.0  # guaranteed boundary point

        for beta in [0.0, 0.1, 0.2, 0.3, 0.45]:
            alpha = mu - beta
            proj = _make_moreau(mu=mu, beta=beta)

            y = np.array([0.0, 1.0])   # |y_T| = 1
            z = np.array([s_target - alpha, w_target])

            p = proj.project(y, z)
            vN, vT = p[0], abs(p[1])

            assert vT > 1e-12, f"β={beta}: should be sliding"
            assert_allclose(vN, beta * vT, atol=1e-10,
                            err_msg=f"β={beta}: v_N should equal β|v_T|")
            _verify_dual_cone(p, mu, alpha)

    def test_tangent_cone_vs_fd(self):
        """tangent_cone matches finite difference with dilatancy."""
        mu = 0.4
        beta = 0.15
        proj = _make_moreau(mu=mu, beta=beta)

        y = np.array([0.3, 1.5])
        z = np.array([0.5, 0.8])

        D = proj.tangent_cone(z, y)
        D_fd = _fd_jacobian(proj, y, z, wrt='candidate')
        assert_allclose(D, D_fd, atol=1e-6,
                        err_msg="tangent_cone should match FD wrt candidate")

    def test_tangent_cone_fd_multiple_points(self):
        """tangent_cone matches FD at points away from nonsmooth kinks."""
        mu = 0.5
        beta = 0.2
        proj = _make_moreau(mu=mu, beta=beta)
        rng = np.random.default_rng(42)

        checked = 0
        for _ in range(30):
            y = rng.standard_normal(2) * 2
            z = rng.standard_normal(2) * 2
            if _is_near_kink(y, z, mu, beta):
                continue
            D = proj.tangent_cone(z, y)
            D_fd = _fd_jacobian(proj, y, z, wrt='candidate')
            assert_allclose(D, D_fd, atol=1e-6)
            checked += 1
        assert checked >= 5, f"Only {checked} samples away from kinks"

    def test_split_d_cand_vs_fd(self):
        """D_cand from tangent_cone_split matches FD wrt candidate."""
        mu = 0.4
        beta = 0.1
        proj = _make_moreau(mu=mu, beta=beta)

        y = np.array([0.3, 1.5])
        z = np.array([0.5, 0.8])

        A, B = proj.tangent_cone_split(z, y)
        A_fd = _fd_jacobian(proj, y, z, wrt='candidate')
        assert_allclose(A, A_fd, atol=1e-6,
                        err_msg="D_cand should match FD wrt candidate")

    def test_split_d_state_vs_fd(self):
        """D_state from tangent_cone_split matches FD wrt current_state."""
        mu = 0.4
        beta = 0.15
        proj = _make_moreau(mu=mu, beta=beta)

        y = np.array([0.3, 1.5])
        z = np.array([0.5, 0.8])

        A, B = proj.tangent_cone_split(z, y)
        B_fd = _fd_jacobian(proj, y, z, wrt='current_state')
        assert_allclose(B, B_fd, atol=1e-6,
                        err_msg="D_state should match FD wrt current_state")

    def test_split_full_jacobian_vs_fd(self):
        """A + B from tangent_cone_split matches FD, filtered from kinks."""
        mu = 0.5
        beta = 0.2
        proj = _make_moreau(mu=mu, beta=beta)
        rng = np.random.default_rng(123)

        checked = 0
        for _ in range(30):
            y = rng.standard_normal(2) * 2
            z = rng.standard_normal(2) * 2
            if _is_near_kink(y, z, mu, beta):
                continue

            A, B = proj.tangent_cone_split(z, y)

            # FD of candidate + state
            A_fd = _fd_jacobian(proj, y, z, wrt='candidate')
            B_fd = _fd_jacobian(proj, y, z, wrt='current_state')

            assert_allclose(A, A_fd, atol=1e-6)
            assert_allclose(B, B_fd, atol=1e-6)
            checked += 1
        assert checked >= 5, f"Only {checked} samples away from kinks"

    def test_ssn_formula_with_dilatancy(self):
        """SSN Jacobian I - A - B + λ A J_F matches FD of natural residual.

        Samples filtered away from SOC kinks and |y_T| = 0 so that the
        Clarke sub-differential is unique and FD is reliable.
        """
        mu = 0.5
        beta = 0.2
        proj = _make_moreau(mu=mu, beta=beta)

        n = 2
        lam = 0.5
        rng = np.random.default_rng(77)

        checked = 0
        for trial in range(30):
            y = rng.standard_normal(n) * 2
            J_F = np.eye(n) + 0.1 * rng.standard_normal((n, n))

            z_cand = y - lam * (J_F @ y)
            if _is_near_kink(y, z_cand, mu, beta):
                continue

            def F(y_, _J=J_F):
                return _J @ y_

            def residual(y_, _J=J_F):
                z_ = y_ - lam * (_J @ y_)
                return y_ - proj.project(y_, z_)

            A, B = proj.tangent_cone_split(z_cand, y)
            J_ssn = np.eye(n) - A - B + lam * A @ J_F

            # FD of residual
            eps = 1e-7
            J_fd = np.zeros((n, n))
            for j in range(n):
                y_p = y.copy(); y_p[j] += eps
                y_m = y.copy(); y_m[j] -= eps
                J_fd[:, j] = (residual(y_p) - residual(y_m)) / (2 * eps)

            assert_allclose(J_ssn, J_fd, atol=1e-5,
                            err_msg=f"SSN formula mismatch at trial {trial}")
            checked += 1
        assert checked >= 5, f"Only {checked} samples away from kinks"

    def test_beta_equals_mu_zero_dilatancy_coeff(self):
        """β = μ gives α = 0: De Saxcé term vanishes, maximal dilatancy."""
        mu = 0.3
        proj = _make_moreau(mu=mu, beta=mu)

        y = np.array([0.0, 2.0])
        z = np.array([1.0, 0.5])

        p = proj.project(y, z)
        # α = 0 means no normal augmentation.  The forward De Saxcé
        # doesn't add μ||v_T|| at all.
        # Just verify it runs without error and result is finite.
        assert np.all(np.isfinite(p))

    def test_beta_exceeds_mu_raises(self):
        """β > μ should raise ValueError."""
        mu = 0.3
        proj = _make_moreau(mu=mu, beta=0.5)

        y = np.array([0.0, 2.0])
        z = np.array([1.0, 0.5])

        with pytest.raises(ValueError, match="exceeding mu"):
            proj.project(y, z)

    def test_negative_beta_raises(self):
        """β < 0 should raise ValueError."""
        mu = 0.3
        proj = _make_moreau(mu=mu, beta=-0.1)

        y = np.array([0.0, 2.0])
        z = np.array([1.0, 0.5])

        with pytest.raises(ValueError, match="negative"):
            proj.project(y, z)

    def test_callable_beta(self):
        """beta as a callable should work the same as scalar."""
        mu = 0.4
        beta_val = 0.1
        proj_scalar = _make_moreau(mu=mu, beta=beta_val)
        proj_callable = _make_moreau(mu=mu, beta=lambda y: beta_val)

        y = np.array([0.3, 1.5])
        z = np.array([0.5, 0.8])

        p_s = proj_scalar.project(y, z)
        p_c = proj_callable.project(y, z)
        assert_allclose(p_c, p_s, atol=1e-14)

    def test_3d_block(self):
        """Dilatancy works with 3D contact (1 normal + 2 tangential)."""
        mu = 0.5
        beta = 0.15
        proj = _make_moreau(mu=mu, beta=beta, blocks=[(0, [1, 2])])

        y = np.array([0.0, 1.5, 0.8])
        z = np.array([-1.0, 1.2, 0.6])

        p = proj.project(y, z)
        assert np.all(np.isfinite(p))

        # FD check for tangent_cone
        D = proj.tangent_cone(z, y)
        D_fd = _fd_jacobian(proj, y, z, wrt='candidate')
        assert_allclose(D, D_fd, atol=1e-6)

        # FD check for tangent_cone_split
        A, B = proj.tangent_cone_split(z, y)
        B_fd = _fd_jacobian(proj, y, z, wrt='current_state')
        assert_allclose(A, D_fd, atol=1e-6)
        assert_allclose(B, B_fd, atol=1e-6)

    def test_multi_block_dilatancy(self):
        """Dilatancy works with multiple contact blocks."""
        mu = 0.4
        beta = 0.1
        blocks = [(0, [1]), (2, [3])]
        proj = _make_moreau(mu=mu, beta=beta, blocks=blocks)

        y = np.array([0.3, 1.5, 0.2, 0.9])
        z = np.array([0.5, 0.8, 0.4, 0.6])

        p = proj.project(y, z)
        assert np.all(np.isfinite(p))

        A, B = proj.tangent_cone_split(z, y)
        A_fd = _fd_jacobian(proj, y, z, wrt='candidate')
        B_fd = _fd_jacobian(proj, y, z, wrt='current_state')
        assert_allclose(A, A_fd, atol=1e-6)
        assert_allclose(B, B_fd, atol=1e-6)

    def test_restitution_with_dilatancy(self):
        """Dilatancy + restitution: both e and β active."""
        mu = 0.4
        beta = 0.1
        e = 0.5
        proj = _make_moreau(mu=mu, beta=beta, e=e)

        y = np.array([0.3, 1.5])
        z = np.array([0.5, 0.8])
        prev = np.array([-1.0, 0.5])

        p = proj.project(y, z, prev_state=prev)
        assert np.all(np.isfinite(p))

        D = proj.tangent_cone(z, y, prev_state=prev)
        D_fd = _fd_jacobian(proj, y, z, wrt='candidate', prev_state=prev)
        assert_allclose(D, D_fd, atol=1e-6)

    def test_uhat_in_dual_cone_3d(self):
        """Reconstructed û ∈ K_{1/μ} for a 3D block in the sliding regime."""
        mu = 0.5
        beta = 0.15
        alpha = mu - beta
        proj = _make_moreau(mu=mu, beta=beta, blocks=[(0, [1, 2])])

        # Transformed point (s, w) with |w| = sqrt(w1²+w2²) chosen so
        # that s < |w|/κ and s > -κ|w| (boundary region).
        w1, w2 = 1.5, 1.0
        w_norm = np.sqrt(w1**2 + w2**2)
        s_target = 0.5  # 0.5 < w_norm * mu = ~0.9, and 0.5 > -(1/mu)*w_norm = -3.6
        y_T_norm = 1.0
        y = np.array([0.0, 0.6, 0.8])   # |y_T| = 1.0
        z = np.array([s_target - alpha * y_T_norm, w1, w2])

        p = proj.project(y, z)
        vN = p[0]
        vT_norm = float(np.linalg.norm(p[1:]))

        assert vT_norm > 1e-12, "Should be sliding"
        assert_allclose(vN, beta * vT_norm, atol=1e-10)
        _verify_dual_cone(p, mu, alpha)

    def test_sparse_tangent_cone_dilatancy(self):
        """Sparse CSR path (n > 64) matches dense FD on selected columns."""
        import scipy.sparse as sp

        mu = 0.5
        beta = 0.15
        alpha = mu - beta
        n = 200
        # Single 2D block embedded at indices 50 (normal), 51 (tangential)
        blocks = [(50, [51])]
        proj = MoreauSOCProjection(
            blocks=blocks,
            get_mu=lambda y: mu,
            get_beta=lambda y: beta,
        )

        rng = np.random.default_rng(99)
        y = rng.standard_normal(n)
        y[51] = 2.0  # |y_T| well away from zero
        z = rng.standard_normal(n)
        # Place transformed point solidly in boundary region
        z[50] = 0.5 - alpha * abs(y[51])

        D_sp = proj.tangent_cone(z, y)
        assert sp.issparse(D_sp), f"Expected sparse, got {type(D_sp)}"
        D_dense = D_sp.toarray()

        # FD on selected columns: block columns + far identity columns
        eps = 1e-7
        for j in [49, 50, 51, 52, 100]:
            z_p = z.copy(); z_p[j] += eps
            z_m = z.copy(); z_m[j] -= eps
            fd_col = (proj.project(y, z_p) - proj.project(y, z_m)) / (2 * eps)
            assert_allclose(D_dense[:, j], fd_col, atol=1e-6,
                            err_msg=f"Sparse tangent_cone column {j} mismatch")

    def test_sparse_split_dilatancy(self):
        """Sparse CSR split (n > 64) matches FD on selected columns."""
        import scipy.sparse as sp

        mu = 0.5
        beta = 0.15
        alpha = mu - beta
        n = 200
        blocks = [(50, [51])]
        proj = MoreauSOCProjection(
            blocks=blocks,
            get_mu=lambda y: mu,
            get_beta=lambda y: beta,
        )

        rng = np.random.default_rng(101)
        y = rng.standard_normal(n)
        y[51] = 2.0
        z = rng.standard_normal(n)
        z[50] = 0.5 - alpha * abs(y[51])

        A_sp, B_sp = proj.tangent_cone_split(z, y)
        assert sp.issparse(A_sp)
        assert sp.issparse(B_sp)
        A_d = A_sp.toarray()
        B_d = B_sp.toarray()

        eps = 1e-7
        for j in [49, 50, 51, 52, 100]:
            # D_cand: FD wrt candidate
            z_p = z.copy(); z_p[j] += eps
            z_m = z.copy(); z_m[j] -= eps
            fd_A = (proj.project(y, z_p) - proj.project(y, z_m)) / (2 * eps)
            assert_allclose(A_d[:, j], fd_A, atol=1e-6,
                            err_msg=f"D_cand column {j} mismatch")

            # D_state: FD wrt current_state
            y_p = y.copy(); y_p[j] += eps
            y_m = y.copy(); y_m[j] -= eps
            fd_B = (proj.project(y_p, z.copy()) - proj.project(y_m, z.copy())) / (2 * eps)
            assert_allclose(B_d[:, j], fd_B, atol=1e-6,
                            err_msg=f"D_state column {j} mismatch")


# ──────────────────────────────────────────────────────────────────────
# Tests: contact.py builder with dilatancy
# ──────────────────────────────────────────────────────────────────────

class TestContactDilatancy:
    """Tests for build_impulse_contact with beta."""

    def test_contact_builder_accepts_beta(self):
        """build_impulse_contact should accept beta in contact spec."""
        from solve_nivp.contact import build_impulse_contact

        A = np.diag([1.0, 1.0, 1.0, 1.0])
        y0 = np.array([0.0, 0.0, 0.0, 1.0])

        def rhs(t, y):
            return np.array([0.0, -9.81, y[0], y[1]])

        def gap(y, t):
            return np.array([y[3]])

        cs = build_impulse_contact(
            A=A,
            rhs_smooth=rhs,
            y0=y0,
            contacts=[
                dict(vel_normal_idx=1, vel_tangential_idx=[0],
                     mu=0.3, beta=0.1, e=0.0),
            ],
            gap_func=gap,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        assert cs.n_phys == 4
        assert len(cs.y0) == 6

    def test_contact_builder_beta_zero_matches_default(self):
        """beta=0.0 should produce the same RHS as no beta."""
        from solve_nivp.contact import build_impulse_contact

        A = np.diag([1.0, 1.0, 1.0, 1.0])
        y0 = np.array([0.0, 0.0, 0.0, 1.0])

        def rhs(t, y):
            return np.array([0.0, -9.81, y[0], y[1]])

        def gap(y, t):
            return np.array([y[3]])

        contact_no_beta = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                                mu=0.3, e=0.0)]
        contact_beta_0 = [dict(vel_normal_idx=1, vel_tangential_idx=[0],
                               mu=0.3, beta=0.0, e=0.0)]

        cs1 = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0, contacts=contact_no_beta,
            gap_func=gap, component_slices=[slice(0, 2), slice(2, 4)],
        )
        cs2 = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0, contacts=contact_beta_0,
            gap_func=gap, component_slices=[slice(0, 2), slice(2, 4)],
        )

        # Evaluate RHS at the same state: results should match
        y_test = np.array([2.0, -1.0, 0.5, 0.0, 0.5, 0.1])
        prev = np.zeros(6)
        h = 0.01

        out1 = cs1.rhs(0.5, y_test, prev, h)
        out2 = cs2.rhs(0.5, y_test, prev, h)
        assert_allclose(out2, out1, atol=1e-14)

    def test_contact_builder_beta_changes_rhs(self):
        """Non-zero beta should change the De Saxcé augmentation in the RHS."""
        from solve_nivp.contact import build_impulse_contact

        A = np.diag([1.0, 1.0, 1.0, 1.0])
        y0 = np.array([0.0, 0.0, 0.0, 1.0])

        def rhs(t, y):
            return np.array([0.0, -9.81, y[0], y[1]])

        def gap(y, t):
            return np.array([y[3]])

        cs_std = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0,
            contacts=[dict(vel_normal_idx=1, vel_tangential_idx=[0],
                           mu=0.3, e=0.0)],
            gap_func=gap, component_slices=[slice(0, 2), slice(2, 4)],
        )
        cs_dil = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0,
            contacts=[dict(vel_normal_idx=1, vel_tangential_idx=[0],
                           mu=0.3, beta=0.15, e=0.0)],
            gap_func=gap, component_slices=[slice(0, 2), slice(2, 4)],
        )

        # State with nonzero tangential velocity (so De Saxcé term differs)
        y_test = np.array([2.0, -1.0, 0.5, 0.0, 0.5, 0.1])
        prev = np.zeros(6)
        h = 0.01

        out_std = cs_std.rhs(0.5, y_test, prev, h)
        out_dil = cs_dil.rhs(0.5, y_test, prev, h)

        # The reaction-row normal (idx 4) should differ: it has α||v_T|| vs μ||v_T||
        assert not np.isclose(out_std[4], out_dil[4], atol=1e-10), \
            "Non-zero beta should change the De Saxcé normal augmentation"
        # The tangential row should be unchanged (De Saxcé only affects normal)
        assert_allclose(out_std[5], out_dil[5], atol=1e-14)

    def test_contact_builder_callable_beta(self):
        """Callable beta in contact spec should work."""
        from solve_nivp.contact import build_impulse_contact

        A = np.diag([1.0, 1.0, 1.0, 1.0])
        y0 = np.array([0.0, 0.0, 0.0, 1.0])

        def rhs(t, y):
            return np.array([0.0, -9.81, y[0], y[1]])

        def gap(y, t):
            return np.array([y[3]])

        cs = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0,
            contacts=[dict(vel_normal_idx=1, vel_tangential_idx=[0],
                           mu=0.3, beta=lambda y: 0.1, e=0.0)],
            gap_func=gap, component_slices=[slice(0, 2), slice(2, 4)],
        )

        y_test = np.array([2.0, -1.0, 0.5, 0.0, 0.5, 0.1])
        prev = np.zeros(6)
        h = 0.01

        out = cs.rhs(0.5, y_test, prev, h)
        assert np.all(np.isfinite(out))

    def test_contact_single_step_sliding_law(self):
        """Single backward Euler step via contact builder: verify dilatancy.

        Build a minimal contact system, integrate one fixed step, and check
        that the converged iterate satisfies v_N ≈ β|v_T| (sliding law)
        and û ∈ K_{1/μ}.
        """
        from solve_nivp.contact import build_impulse_contact
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        from solve_nivp.integrations import BackwardEuler
        from solve_nivp.ODESystem import ODESystem
        from solve_nivp.ODESolver import ODESolver

        mu = 0.4
        beta_val = 0.15
        alpha = mu - beta_val
        kappa = 1.0 / mu
        mass = 1.0
        A_phys = np.diag([mass, mass, 1.0, 1.0])

        def rhs_smooth(t, y):
            return np.array([0.0, -9.81 * mass, y[0], y[1]])

        def gap(y, t):
            return np.array([y[3]])

        # Ball at ground (gap=0), sliding rightward, pressing down
        y0_phys = np.array([3.0, -2.0, 0.0, 0.0])

        cs = build_impulse_contact(
            A=A_phys, rhs_smooth=rhs_smooth, y0=y0_phys,
            contacts=[dict(vel_normal_idx=1, vel_tangential_idx=[0],
                           mu=mu, beta=beta_val, e=0.0)],
            gap_func=gap,
            component_slices=[slice(0, 2), slice(2, 4)],
        )

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=cs.projection,
            tol=1e-12,
            max_iter=100,
            component_slices=cs.component_slices,
        )
        integrator = BackwardEuler(solver=solver, A=cs.A, **cs.integrator_opts)

        h = 0.01
        system = ODESystem(fun=cs.rhs, y0=cs.y0, method=integrator,
                           adaptive=False)
        driver = ODESolver(system, t_span=(0, h), h=h)
        t, y_hist, _, _, _ = driver.solve()

        y_final = y_hist[-1]

        # Physical contact velocities
        v_N = y_final[1]   # vel_normal_idx
        v_T = y_final[0]   # vel_tangential_idx

        # If sliding, verify dilatancy law
        if abs(v_T) > 1e-8:
            assert_allclose(v_N, beta_val * abs(v_T), atol=1e-6,
                            err_msg="Sliding law v_N = β|v_T| not satisfied")

        # Verify û ∈ K_{1/μ}
        uhat_N = v_N + alpha * abs(v_T)
        assert uhat_N >= -1e-10, f"û_N = {uhat_N} should be ≥ 0"
        assert abs(v_T) <= kappa * uhat_N + 1e-10, (
            f"|û_T| = {abs(v_T)} should be ≤ κ·û_N = {kappa * uhat_N}")


# ──────────────────────────────────────────────────────────────────────
# Tests: SSN solver convergence with dilatancy
# ──────────────────────────────────────────────────────────────────────

class TestSSNDilatancy:
    """Verify SSN solver converges with dilatant Moreau projection."""

    def test_ssn_moreau_dilatancy_converges(self):
        """SSN converges and satisfies û ∈ K_{1/μ}; sliding law if on boundary."""
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver

        mu = 0.5
        beta = 0.2
        alpha = mu - beta
        kappa = 1.0 / mu
        proj = _make_moreau(mu=mu, beta=beta)

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            tol=1e-10,
            max_iter=50,
        )

        # Choose b so the unconstrained solution has v_N < 0 → forces sliding
        A_mat = np.array([[3.0, 0.1], [0.2, 2.0]])
        b = np.array([-1.0, 0.5])

        def F(y_):
            return A_mat @ y_ - b

        solver.prev_state = np.zeros(2)
        y0 = np.array([0.5, 0.5])
        y_sol, F_sol, err, success, iters = solver.solve(F, y0)
        assert success, f"SSN did not converge: err={err}, iters={iters}"
        assert err < 1e-9

        # Always valid: û ∈ K_{1/μ}
        _verify_dual_cone(y_sol, mu, alpha)

        # Check if on the boundary (sliding): û_N ≈ κ|û_T|
        vN = y_sol[0]
        vT = abs(y_sol[1])
        uhat_N = vN + alpha * vT
        if vT > 1e-8 and abs(uhat_N - kappa * vT) < 1e-6:
            # On the cone boundary → sliding law must hold
            assert_allclose(vN, beta * vT, atol=1e-8,
                            err_msg="Dilatancy law v_N = β|v_T| not satisfied")

    def test_ssn_moreau_dilatancy_bouncing_ball(self):
        """Full integration: bouncing ball with dilatancy completes."""
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        from solve_nivp.integrations import BackwardEuler
        from solve_nivp.ODESystem import ODESystem
        from solve_nivp.ODESolver import ODESolver

        mass = 1.0
        gravity = np.array([0.0, -9.81])
        A = np.diag([mass, mass, 1.0, 1.0])
        mu = 0.3
        beta = 0.05  # small dilatancy (large β can lift ball off floor)

        def rhs(t, y):
            v = y[0:2]
            return np.concatenate([mass * gravity, v])

        def gap_func(y, t):
            return np.array([y[3]])

        proj = MoreauSOCProjection(
            blocks=[(1, [0])],
            get_mu=lambda y: mu,
            get_beta=lambda y: beta,
            gap_func=gap_func,
            gap_tol=0.0,
            e=0.0,
        )

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            tol=1e-12,
            max_iter=200,
            component_slices=[slice(0, 2), slice(2, 4)],
        )
        integrator = BackwardEuler(solver=solver, A=A)

        y0 = np.array([2.0, 0.0, 0.0, 1.0])
        system = ODESystem(fun=rhs, y0=y0, method=integrator, adaptive=True)
        driver = ODESolver(system, t_span=(0.0, 0.6), h=0.001)
        t, y_hist, h_hist, fk_hist, info = driver.solve()

        # Should reach at least past impact (ball drops from h=1, impacts ~t=0.45)
        assert t[-1] >= 0.5, f"Integration stopped too early at t={t[-1]:.4f}"
        # Gap should stay non-negative (approximately)
        assert np.min(y_hist[:, 3]) > -1e-3
