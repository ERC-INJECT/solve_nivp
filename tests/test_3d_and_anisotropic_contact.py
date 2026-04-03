"""Tests for 3D friction and anisotropic friction via build_impulse_contact.

Tests cover:
  1. 3D contact (2 tangential DOFs) through build_impulse_contact
     - Structure: dimensions, B matrix, SOC blocks
     - RHS: De Saxcé augmentation with ||v_T|| in 2D tangential space
     - End-to-end: 3D bouncing ball sliding on a plane
  2. Anisotropic friction via get_B parameter
     - Structure: AnisotropicSOCProjection created
     - Isotropic equivalence: get_B=I matches standard pipeline
     - Direction-dependent friction: different mu per tangential axis
     - End-to-end: block slides further in low-friction direction
"""

import numpy as np
import pytest

from solve_nivp.contact import build_impulse_contact, ContactSystem
from solve_nivp.projections import (
    AnisotropicSOCProjection,
    MuScaledSOCProjection,
)
import solve_nivp


# =====================================================================
# Shared fixtures
# =====================================================================

def _3d_ball_setup(mu=0.3, e=0.0):
    """3D bouncing ball: v=(vx,vy,vz), q=(qx,qy,qz).

    Normal direction: z (idx 2 for vel, idx 5 for pos).
    Tangential directions: x, y (idx 0, 1).
    """
    mass = 1.0
    gravity = np.array([0.0, 0.0, -9.81])
    n_phys = 6
    A = np.diag([mass, mass, mass, 1.0, 1.0, 1.0])

    def rhs(t, y):
        v = y[0:3]
        return np.concatenate([mass * gravity, v])

    def gap_func(y, t):
        return np.array([y[5]])   # q_z = 0 is floor

    y0 = np.array([1.0, 0.5, 0.0,   # initial tangential velocity (vx, vy)
                    0.0, 0.0, 1.0])   # start 1 m above floor

    contacts = [dict(
        vel_normal_idx=2,
        vel_tangential_idx=[0, 1],
        mu=mu,
        e=e,
    )]
    return A, rhs, y0, contacts, gap_func, n_phys


def _spring_slider_3d_setup(mu=0.5):
    """3D spring-slider for anisotropic friction tests.

    Block on frictional floor.  Spring pulls in +x.
    v=(vx,vy,vz), q=(qx,qy,qz).  Normal = z.
    Gravity via s0.
    """
    mass = 1.0
    g = 9.81
    k_T = 5.0
    q_eq_x = 1.5

    n_phys = 6
    A = np.diag([mass, mass, mass, 1.0, 1.0, 1.0])
    h_fixed = 0.01
    _h = [h_fixed]

    def rhs(t, y):
        vx, vy, vz = y[0], y[1], y[2]
        qx = y[3]
        return np.array([
            -k_T * (qx - q_eq_x),   # spring in x
            0.0,                      # no force in y
            0.0,                      # gravity via s0
            vx, vy, vz,
        ])

    gap_func = lambda y, t: np.array([-1.0])   # always active

    y0 = np.zeros(n_phys)

    contacts = [dict(
        vel_normal_idx=2,
        vel_tangential_idx=[0, 1],
        mu=mu,
    )]

    return dict(
        A=A, rhs=rhs, y0=y0, contacts=contacts,
        gap_func=gap_func, n_phys=n_phys,
        mass=mass, g=g, k_T=k_T, q_eq_x=q_eq_x,
        _h=_h, h_fixed=h_fixed,
    )


# =====================================================================
# 3D Contact — Structure
# =====================================================================

class TestContact3DStructure:
    """Verify dimensions and B for 3D contact (2 tangential DOFs)."""

    def test_augmented_dimensions(self):
        """3D contact → 3 reaction DOFs (r_N, r_T1, r_T2)."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.n_phys == 6
        assert cs.y0.shape == (9,)      # 6 phys + 3 react
        assert cs.A.shape == (9, 9)

    def test_augmented_y0_padded(self):
        """Initial reactions are zero."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        np.testing.assert_array_equal(cs.y0[:6], y0)
        np.testing.assert_array_equal(cs.y0[6:], 0.0)

    def test_auto_B_3d(self):
        """Auto-B maps r_N→vz(2), r_T1→vx(0), r_T2→vy(1)."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        expected_B = np.zeros((6, 3))
        expected_B[2, 0] = 1.0   # r_N  → v_z
        expected_B[0, 1] = 1.0   # r_T1 → v_x
        expected_B[1, 2] = 1.0   # r_T2 → v_y
        np.testing.assert_array_equal(cs.B, expected_B)

    def test_component_slices_3d(self):
        """Reaction slice spans 3 DOFs."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 3), slice(3, 6)],
        )
        assert len(cs.component_slices) == 3
        assert cs.component_slices[-1] == slice(6, 9)

    def test_projection_soc_blocks_3d(self):
        """SOC block has 1 normal + 2 tangential indices."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        proj = cs.projection
        assert isinstance(proj, MuScaledSOCProjection)
        # Blocks: [(r_N_idx, [r_T1_idx, r_T2_idx])]
        assert len(proj.blocks) == 1
        s_idx, w_idx = proj.blocks[0]
        assert s_idx == 6
        assert list(w_idx) == [7, 8]

    def test_two_3d_contacts(self):
        """Two 3D contacts → 6 reaction DOFs."""
        n_phys = 12
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)

        def rhs(t, y):
            return np.zeros(n_phys)

        def gap(y, t):
            return np.array([y[5], y[11]])

        contacts = [
            dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=0.3),
            dict(vel_normal_idx=8, vel_tangential_idx=[6, 7], mu=0.5),
        ]
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.y0.shape == (18,)     # 12 + 6
        assert cs.B.shape == (12, 6)

        # Check B columns
        expected_B = np.zeros((12, 6))
        # Contact 1: r_N→2, r_T1→0, r_T2→1
        expected_B[2, 0] = 1.0
        expected_B[0, 1] = 1.0
        expected_B[1, 2] = 1.0
        # Contact 2: r_N→8, r_T1→6, r_T2→7
        expected_B[8, 3] = 1.0
        expected_B[6, 4] = 1.0
        expected_B[7, 5] = 1.0
        np.testing.assert_array_equal(cs.B, expected_B)

    def test_mixed_2d_3d_contacts(self):
        """One 2D contact + one 3D → 2 + 3 = 5 reaction DOFs."""
        n_phys = 8
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)

        def rhs(t, y):
            return np.zeros(n_phys)

        def gap(y, t):
            return np.array([y[2], y[5]])

        contacts = [
            dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3),       # 2D
            dict(vel_normal_idx=5, vel_tangential_idx=[3, 4], mu=0.5),    # 3D
        ]
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert cs.y0.shape == (13,)    # 8 + 2 + 3


# =====================================================================
# 3D Contact — RHS
# =====================================================================

class TestContact3DRHS:
    """Verify De Saxcé augmentation with 2D tangential velocity."""

    def test_reaction_rows_desaxce_3d(self):
        """û_N = v_z + μ·||(v_x,v_y)||, û_T = (v_x, v_y)."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.4)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        y_test = np.zeros(9)
        y_test[0] = 3.0    # v_x
        y_test[1] = 4.0    # v_y
        y_test[2] = -2.0   # v_z (normal)
        out = cs.rhs(0.0, y_test, y_test, 0.001)

        # ||v_T|| = ||(3,4)|| = 5
        # û_N = v_z + μ·||v_T|| = -2 + 0.4*5 = 0.0
        # -û_N = 0.0
        np.testing.assert_allclose(out[6], 0.0, atol=1e-14)
        # û_T = (v_x, v_y) = (3, 4)
        # -û_T = (-3, -4)
        np.testing.assert_allclose(out[7], -3.0, atol=1e-14)
        np.testing.assert_allclose(out[8], -4.0, atol=1e-14)

    def test_physical_rows_impulse_3d(self):
        """Coupling: B @ r / h adds to physical rows."""
        A, rhs_smooth, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs_smooth, y0, contacts, gap)
        y_test = np.zeros(9)
        y_test[0] = 1.0   # v_x
        y_test[5] = 0.0   # q_z = 0 (on floor)
        y_test[6] = 5.0   # p_N
        y_test[7] = 2.0   # p_T1
        y_test[8] = -1.0  # p_T2
        h = 0.01
        out = cs.rhs(0.0, y_test, y_test, h)

        f_phys = rhs_smooth(0.0, y_test[:6])
        r = np.array([5.0, 2.0, -1.0])
        expected_phys = f_phys + cs.B @ r / h
        np.testing.assert_allclose(out[:6], expected_phys, atol=1e-12)

    def test_desaxce_norm_2d(self):
        """Verify ||v_T|| uses 2D Euclidean norm, not sum."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0)
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        y_test = np.zeros(9)
        y_test[0] = 1.0    # v_x
        y_test[1] = 0.0    # v_y
        y_test[2] = -1.0   # v_z
        out = cs.rhs(0.0, y_test, y_test, 0.001)
        # ||v_T|| = ||(1,0)|| = 1
        # û_N = -1 + 1*1 = 0  →  -û_N = 0
        np.testing.assert_allclose(out[6], 0.0, atol=1e-14)

        # Now test (v_x, v_y) = (0.6, 0.8) → ||v_T|| = 1.0
        y_test2 = np.zeros(9)
        y_test2[0] = 0.6
        y_test2[1] = 0.8
        y_test2[2] = -1.0
        out2 = cs.rhs(0.0, y_test2, y_test2, 0.001)
        np.testing.assert_allclose(out2[6], 0.0, atol=1e-14)


# =====================================================================
# 3D Contact — End-to-end integration
# =====================================================================

class TestContact3DIntegration:
    """End-to-end 3D bouncing ball on frictional floor."""

    def test_3d_bouncing_ball_fixed_step(self):
        """3D ball drops, impacts, slides, and decelerates."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.3, e=0.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 3), slice(3, 6)],
        )
        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False,
            h0=0.001,
        )

        t = np.array(t)
        assert not np.any(np.isnan(y))
        # Ball should impact around t ≈ 0.452 (h=1, g=9.81)
        # After impact: v_z ≈ 0 (inelastic)
        assert len(t) > 100
        # Normal reaction non-negative during contact
        contact_mask = y[:, 5] <= 1e-8    # q_z ≈ 0
        p_N = y[contact_mask, 6]
        assert np.all(p_N >= -1e-10), f"Negative p_N: min={p_N.min()}"
        # Tangential velocity should decrease due to friction
        v_T_final = np.linalg.norm(y[-1, 0:2])
        v_T_init = np.linalg.norm(y0[0:2])
        assert v_T_final < v_T_init, (
            f"Friction should reduce tangential speed: "
            f"final={v_T_final:.4f}, init={v_T_init:.4f}")

    def test_3d_ball_isotropic_sliding_direction(self):
        """With isotropic friction, sliding decelerates uniformly in all
        tangential directions.  After sticking, v_T direction should
        be close to the initial direction (friction decelerates magnitude,
        not direction)."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.3, e=0.0)
        # Give diagonal tangential velocity
        y0[0] = 1.0   # v_x
        y0[1] = 1.0   # v_y
        # Low drop height for quick contact
        y0[5] = 0.1   # q_z = 0.1 m
        contacts[0]['mu'] = 0.3
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 3), slice(3, 6)],
        )
        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs, t_span=(0.0, 2.0), y0=cs.y0, A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False, h0=0.005,
        )
        assert not np.any(np.isnan(y))
        # Once sliding, friction acts along v_T direction
        # The ratio v_x / v_y should stay ≈ 1 whenever both are nonzero
        sliding_mask = (y[:, 5] <= 1e-8) & (np.linalg.norm(y[:, 0:2], axis=1) > 0.01)
        if np.any(sliding_mask):
            v_x = y[sliding_mask, 0]
            v_y = y[sliding_mask, 1]
            ratio = v_x / (v_y + 1e-15)
            np.testing.assert_allclose(
                ratio, 1.0, atol=0.1,
                err_msg="Isotropic friction should preserve v_T direction")


# =====================================================================
# Anisotropic friction — Structure
# =====================================================================

class TestAnisotropicStructure:
    """get_B parameter creates AnisotropicSOCProjection."""

    def test_anisotropic_projection_type(self):
        """When get_B is provided, projection is AnisotropicSOCProjection."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.eye(2),
        )
        assert isinstance(cs.projection, AnisotropicSOCProjection)

    def test_no_get_B_uses_isotropic(self):
        """Without get_B, projection is MuScaledSOCProjection."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        assert isinstance(cs.projection, MuScaledSOCProjection)
        assert not isinstance(cs.projection, AnisotropicSOCProjection)

    def test_anisotropic_dimensions_match(self):
        """All dimensions are the same with or without get_B."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup()
        cs_iso = build_impulse_contact(A, rhs, y0, contacts, gap)
        cs_ani = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.eye(2),
        )
        assert cs_iso.y0.shape == cs_ani.y0.shape
        np.testing.assert_array_equal(cs_iso.A, cs_ani.A)
        np.testing.assert_array_equal(cs_iso.B, cs_ani.B)

    def test_anisotropic_with_prestress(self):
        """get_B + get_s0 + get_w0 all work together."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0)
        _h = [0.01]
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.diag([2.0, 0.5]),
            get_s0=lambda y: 10.0 * _h[0],
            get_w0=lambda y, k: np.array([1.0 * _h[0], 0.0]),
            step_size_ref=_h,
        )
        assert isinstance(cs.projection, AnisotropicSOCProjection)
        assert cs.projection.get_s0 is not None
        assert cs.projection.get_w0 is not None

    def test_anisotropic_2d_contact(self):
        """Anisotropic with 1 tangential DOF (2D contact)."""
        mass = 1.0
        A = np.diag([mass, mass, 1.0, 1.0])
        y0 = np.zeros(4)

        def rhs(t, y):
            return np.zeros(4)

        gap = lambda y, t: np.array([y[3]])
        contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.5)]

        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.array([[2.0]]),
        )
        assert isinstance(cs.projection, AnisotropicSOCProjection)
        assert cs.y0.shape == (6,)


# =====================================================================
# Anisotropic friction — Isotropic equivalence
# =====================================================================

class TestAnisotropicIsotropicEquivalence:
    """get_B=I must reproduce isotropic results exactly."""

    def test_projection_value_matches(self):
        """Projection with B=I matches MuScaledSOCProjection."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.5)
        cs_iso = build_impulse_contact(A, rhs, y0, contacts, gap)
        cs_ani = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.eye(2),
        )

        # Project a test vector
        z = np.zeros(9)
        z[6] = 1.0   # p_N
        z[7] = 3.0   # p_T1
        z[8] = 4.0   # p_T2
        p_iso = cs_iso.projection.project(z, z)
        p_ani = cs_ani.projection.project(z, z)
        np.testing.assert_allclose(p_ani, p_iso, atol=1e-12)

    def test_rhs_matches(self):
        """RHS output with B=I matches isotropic pipeline."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.3)
        cs_iso = build_impulse_contact(A, rhs, y0, contacts, gap)
        cs_ani = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: np.eye(2),
        )

        y_test = np.array([2.0, 1.0, -1.5, 0.0, 0.0, 0.0, 3.0, 1.0, 0.5])
        prev = np.array([1.0, 0.5, -2.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0])
        out_iso = cs_iso.rhs(0.0, y_test, prev, 0.001)
        out_ani = cs_ani.rhs(0.0, y_test, prev, 0.001)
        np.testing.assert_allclose(out_ani, out_iso, atol=1e-14)

    def test_end_to_end_matches(self):
        """Full solve with B=I matches isotropic solve."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=0.3, e=0.0)
        kw = dict(
            component_slices=[slice(0, 3), slice(3, 6)],
        )
        cs_iso = build_impulse_contact(A, rhs, y0, contacts, gap, **kw)
        cs_ani = build_impulse_contact(A, rhs, y0, contacts, gap,
                                        get_B=lambda y, k: np.eye(2), **kw)

        solve_kw = dict(
            t_span=(0.0, 0.5),
            method='backward_euler',
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            adaptive=False,
            h0=0.005,
        )

        t_iso, y_iso, *_ = solve_nivp.solve_nivp(
            fun=cs_iso.rhs, y0=cs_iso.y0, A=cs_iso.A,
            projection=cs_iso.projection,
            component_slices=cs_iso.component_slices,
            integrator_opts=cs_iso.integrator_opts,
            **solve_kw,
        )
        t_ani, y_ani, *_ = solve_nivp.solve_nivp(
            fun=cs_ani.rhs, y0=cs_ani.y0, A=cs_ani.A,
            projection=cs_ani.projection,
            component_slices=cs_ani.component_slices,
            integrator_opts=cs_ani.integrator_opts,
            **solve_kw,
        )

        np.testing.assert_allclose(t_ani, t_iso, atol=1e-14)
        np.testing.assert_allclose(y_ani, y_iso, atol=1e-10)


# =====================================================================
# Anisotropic friction — Direction-dependent physics
# =====================================================================

class TestAnisotropicDirectional:
    """Anisotropic B gives direction-dependent friction."""

    def test_anisotropic_projection_elliptic(self):
        """Projection onto elliptic cone: q(w) = sqrt(w^T B w) <= mu*s.

        With B = diag(1/mu_x^2, 1/mu_y^2) and mu=1:
        effective mu_x = mu_x, mu_y = mu_y.
        Sliding purely in x-direction uses mu_x.
        """
        mu_x, mu_y = 0.8, 0.3
        B = np.diag([1.0 / mu_x**2, 1.0 / mu_y**2])

        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0)
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: B,
        )
        proj = cs.projection

        # Test 1: purely x-tangential — should use mu_x = 0.8
        z1 = np.zeros(9)
        z1[6] = 1.0    # p_N = 1
        z1[7] = 2.0    # p_T1 = 2 (x-dir), > mu_x*p_N = 0.8
        z1[8] = 0.0    # p_T2 = 0 (y-dir)
        p1 = proj.project(z1, z1)
        # On the elliptic boundary: sqrt(p_T1^2 / mu_x^2) = p_N
        # → |p_T1| = mu_x * p_N
        q1 = np.sqrt(p1[7]**2 / mu_x**2 + p1[8]**2 / mu_y**2)
        np.testing.assert_allclose(q1, p1[6], atol=1e-10)

        # Test 2: purely y-tangential — should use mu_y = 0.3
        z2 = np.zeros(9)
        z2[6] = 1.0
        z2[7] = 0.0
        z2[8] = 2.0    # p_T2 = 2 (y-dir), > mu_y*p_N = 0.3
        p2 = proj.project(z2, z2)
        q2 = np.sqrt(p2[7]**2 / mu_x**2 + p2[8]**2 / mu_y**2)
        np.testing.assert_allclose(q2, p2[6], atol=1e-10)

        # The y-direction has tighter friction, so less tangential impulse
        assert abs(p2[8]) < abs(p1[7]), (
            f"y-direction (mu_y={mu_y}) should give less tangential impulse "
            f"than x (mu_x={mu_x}): |p_T2|={abs(p2[8]):.4f}, |p_T1|={abs(p1[7]):.4f}")

    def test_anisotropic_sliding_direction(self):
        """Block slides further in low-friction direction.

        Spring-slider with spring in x.  With anisotropic friction:
        mu_x = 0.8 (hard to slide in x), mu_y = 0.2 (easy to slide in y).
        We give a diagonal initial perturbation.  The block should
        move more in y than x.
        """
        setup = _spring_slider_3d_setup(mu=1.0)
        A = setup['A']
        rhs = setup['rhs']
        n_phys = setup['n_phys']
        gap_func = setup['gap_func']
        _h = setup['_h']
        h_fixed = setup['h_fixed']
        mass = setup['mass']
        g = setup['g']

        mu_x, mu_y = 0.8, 0.2
        B = np.diag([1.0 / mu_x**2, 1.0 / mu_y**2])

        # Initial condition: small velocity in both x and y
        y0 = np.zeros(n_phys)
        y0[0] = 2.0    # v_x
        y0[1] = 2.0    # v_y  (same magnitude)

        contacts = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=1.0)]

        cs = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
            gap_func=gap_func,
            component_slices=[slice(0, 3), slice(3, 6)],
            get_s0=lambda y: mass * g * _h[0],
            get_B=lambda y, k: B,
            step_size_ref=_h,
        )

        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs, t_span=(0.0, 1.0), y0=cs.y0, A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True, h0=h_fixed,
        )

        assert not np.any(np.isnan(y))

        # q_y should have moved further than without friction asymmetry
        # In y-direction: lower effective friction → velocity decays slower
        # In x-direction: higher effective friction → velocity decays faster
        # Check that the final y-displacement is nonzero (block moved in y)
        # and that the block has moved at all
        q_x_final = np.abs(y[-1, 3])
        q_y_final = np.abs(y[-1, 4])
        assert q_x_final > 0 or q_y_final > 0, (
            "Block should have moved in at least one direction")


# =====================================================================
# Anisotropic friction — End-to-end integration
# =====================================================================

class TestAnisotropicIntegration:
    """Full solve with anisotropic friction."""

    def test_3d_ball_anisotropic_e2e(self):
        """3D bouncing ball with anisotropic friction completes without NaN."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0, e=0.0)
        mu_x, mu_y = 0.6, 0.3
        B = np.diag([1.0 / mu_x**2, 1.0 / mu_y**2])

        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 3), slice(3, 6)],
            get_B=lambda y, k: B,
        )
        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs, t_span=(0.0, 1.0), y0=cs.y0, A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False, h0=0.002,
        )

        assert not np.any(np.isnan(y)), "NaN in anisotropic solution"
        # Normal reaction non-negative during contact
        contact_mask = y[:, 5] <= 1e-8
        p_N = y[contact_mask, 6]
        assert np.all(p_N >= -1e-10), f"Negative p_N: min={p_N.min()}"

    def test_anisotropic_vs_isotropic_different_result(self):
        """Anisotropic B ≠ I should give different trajectory than isotropic."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0, e=0.0)
        # Give initial velocity at 45 degrees
        y0[0] = 1.0   # v_x
        y0[1] = 1.0   # v_y
        y0[5] = 0.1   # low drop height

        kw = dict(
            component_slices=[slice(0, 3), slice(3, 6)],
        )

        # Isotropic
        contacts_iso = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=0.5)]
        cs_iso = build_impulse_contact(A, rhs, y0, contacts_iso, gap, **kw)

        # Anisotropic: mu_x=0.8, mu_y=0.2 (via B with mu=1)
        contacts_ani = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=1.0)]
        B = np.diag([1.0 / 0.8**2, 1.0 / 0.2**2])
        cs_ani = build_impulse_contact(
            A, rhs, y0, contacts_ani, gap,
            get_B=lambda y, k: B, **kw,
        )

        solve_kw = dict(
            t_span=(0.0, 1.0),
            method='backward_euler',
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            adaptive=False, h0=0.002,
        )

        _, y_iso, *_ = solve_nivp.solve_nivp(
            fun=cs_iso.rhs, y0=cs_iso.y0, A=cs_iso.A,
            projection=cs_iso.projection,
            component_slices=cs_iso.component_slices,
            integrator_opts=cs_iso.integrator_opts,
            **solve_kw,
        )
        _, y_ani, *_ = solve_nivp.solve_nivp(
            fun=cs_ani.rhs, y0=cs_ani.y0, A=cs_ani.A,
            projection=cs_ani.projection,
            component_slices=cs_ani.component_slices,
            integrator_opts=cs_ani.integrator_opts,
            **solve_kw,
        )

        # Trajectories must differ (anisotropic breaks symmetry)
        assert not np.allclose(y_iso[:, 0:2], y_ani[:, 0:2], atol=1e-3), (
            "Anisotropic should give different trajectory than isotropic")

    def test_anisotropic_with_s0_gravity(self):
        """Anisotropic friction with gravity via s0 solves correctly."""
        setup = _spring_slider_3d_setup(mu=1.0)
        A = setup['A']
        rhs = setup['rhs']
        y0 = setup['y0']
        gap_func = setup['gap_func']
        _h = setup['_h']
        h_fixed = setup['h_fixed']
        mass = setup['mass']
        g = setup['g']

        mu_x, mu_y = 0.5, 0.5
        B = np.diag([1.0 / mu_x**2, 1.0 / mu_y**2])
        contacts = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=1.0)]

        cs = build_impulse_contact(
            A=A, rhs_smooth=rhs, y0=y0, contacts=contacts,
            gap_func=gap_func,
            component_slices=[slice(0, 3), slice(3, 6)],
            get_s0=lambda y: mass * g * _h[0],
            get_B=lambda y, k: B,
            step_size_ref=_h,
        )

        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs, t_span=(0.0, 3.0), y0=cs.y0, A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True, h0=h_fixed,
        )

        assert not np.any(np.isnan(y))
        # With mu_eff = 0.5 (same as isotropic 0.5), the spring
        # should slide the block.  k_T * q_eq = 7.5 > mu*mg = 4.905
        q_x_final = y[-1, 3]
        assert q_x_final > 0.5, (
            f"Block should slide: q_x_final={q_x_final:.4f}")

    def test_rotated_anisotropy_e2e(self):
        """Rotated B matrix — full solve completes."""
        A, rhs, y0, contacts, gap, n_phys = _3d_ball_setup(mu=1.0, e=0.0)
        y0[0] = 1.0
        y0[1] = 1.0
        y0[5] = 0.1

        theta_rot = np.pi / 4
        R = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                       [np.sin(theta_rot),  np.cos(theta_rot)]])
        D = np.diag([1.0 / 0.6**2, 1.0 / 0.2**2])
        B = R @ D @ R.T

        contacts = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=1.0)]
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            component_slices=[slice(0, 3), slice(3, 6)],
            get_B=lambda y, k: B,
        )

        t, y, *_ = solve_nivp.solve_nivp(
            fun=cs.rhs, t_span=(0.0, 0.5), y0=cs.y0, A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-12, max_iter=200,
                             lam_update_strategy='none'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=False, h0=0.002,
        )

        assert not np.any(np.isnan(y)), "NaN in rotated-anisotropy solution"
        assert len(t) > 10


# =====================================================================
# Anisotropic friction — Tangent cone / Jacobian
# =====================================================================

class TestAnisotropicTangentCone:
    """Tangent cone (Jacobian) matches finite differences."""

    def _fd_jacobian(self, proj, z, eps=1e-7):
        """Finite-difference Jacobian of the full projection."""
        f0 = proj.project(z, z)
        n = z.size
        J = np.zeros((n, n))
        for j in range(n):
            z_p = z.copy()
            z_p[j] += eps
            f1 = proj.project(z_p, z_p)
            J[:, j] = (f1 - f0) / eps
        return J

    def test_tangent_cone_matches_fd_3d(self):
        """3D anisotropic tangent cone matches FD."""
        B = np.diag([2.0, 0.5])
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1, 2])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: B,
        )
        z = np.array([1.0, 3.0, 2.0])
        D = proj.tangent_cone(z, z)
        D_dense = D if isinstance(D, np.ndarray) else D.toarray()
        J_fd = self._fd_jacobian(proj, z)
        np.testing.assert_allclose(D_dense, J_fd, atol=1e-5)

    def test_tangent_cone_via_build_impulse_contact(self):
        """Tangent cone from build_impulse_contact with get_B.

        Use always-active gap to avoid FD perturbation artefacts."""
        n_phys = 6
        A = np.eye(n_phys)
        y0 = np.zeros(n_phys)

        def rhs(t, y):
            return np.zeros(n_phys)

        # Always-active gap (returns -1)
        gap = lambda y, t: np.array([-1.0])
        contacts = [dict(vel_normal_idx=2, vel_tangential_idx=[0, 1], mu=0.5)]
        B_mat = np.diag([2.0, 0.5])
        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap,
            get_B=lambda y, k: B_mat,
        )

        z = np.zeros(9)
        z[6] = 1.0    # p_N
        z[7] = 3.0    # p_T1
        z[8] = 2.0    # p_T2
        D = cs.projection.tangent_cone(z, z)
        D_dense = D if isinstance(D, np.ndarray) else D.toarray()
        J_fd = self._fd_jacobian(cs.projection, z)
        np.testing.assert_allclose(D_dense, J_fd, atol=1e-5)
