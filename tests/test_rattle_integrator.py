"""Tests for the true nonsmooth RATTLE integrator (Breuling et al. 2024).

Covers:
- Unconstrained dynamics: free particle, harmonic oscillator, O(h^2) convergence
- Contact: bouncing ball (single impact, restitution), persistent contact
- Delassus proximal parameter computation
- Proximal maps: normal (gap > 0, penetration), friction (stick/slip)
- Bilateral constraints: pendulum
- Legacy adapter backward compatibility
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solve_nivp.rattle_contact import (
    RattleMechanicalSystem,
    RattleContactSpec,
    RattleBilateralSpec,
    RattleAlgebraicSpec,
    RattleContactSystem,
    RattleSolveResult,
    RattleSolver,
    build_rattle_system,
    build_dynamic_rattle_contact,
    solve_dynamic_rattle_contact,
    _project_ball,
    _project_negative_orthant,
    _wrms_norm,
)


# ===================================================================
#  A. Unit tests for building blocks
# ===================================================================

class TestProximalMaps:
    """Unit tests for proximal operators."""

    def test_project_negative_orthant(self):
        assert float(_project_negative_orthant(np.array([1.0]))[0]) == 0.0
        assert float(_project_negative_orthant(np.array([-2.0]))[0]) == -2.0
        assert float(_project_negative_orthant(np.array([0.0]))[0]) == 0.0

    def test_project_ball_inside(self):
        arg = np.array([0.3, 0.4])
        proj = _project_ball(arg, 1.0)
        np.testing.assert_allclose(proj, arg)

    def test_project_ball_outside(self):
        arg = np.array([3.0, 4.0])  # norm = 5
        proj = _project_ball(arg, 2.0)
        np.testing.assert_allclose(np.linalg.norm(proj), 2.0, atol=1e-14)
        # Direction preserved.
        np.testing.assert_allclose(proj / np.linalg.norm(proj),
                                   arg / np.linalg.norm(arg), atol=1e-14)

    def test_project_ball_zero_radius(self):
        arg = np.array([1.0, 2.0])
        proj = _project_ball(arg, 0.0)
        np.testing.assert_allclose(proj, [0.0, 0.0])


class TestWrmsNorm:
    def test_zero_delta(self):
        ref = np.array([1.0, 2.0, 3.0])
        assert _wrms_norm(np.zeros(3), 1e-8, 1e-6, ref) == 0.0

    def test_unit_delta(self):
        ref = np.ones(3)
        delta = np.ones(3) * (1e-8 + 1e-6)  # = atol + rtol * 1.0
        # Each (delta_i / sc_i)^2 = 1.0, so wrms = 1.0.
        val = _wrms_norm(delta, 1e-8, 1e-6, ref)
        assert abs(val - 1.0) < 0.1  # approximately 1


class TestDelassusProxParams:
    """Test Delassus diagonal proximal parameter estimation."""

    def test_single_dof_mass_spring(self):
        """For a 1-DOF system: r = alpha / (w^T M^{-1} w)."""
        m = 2.0
        w = 1.0  # unit normal direction

        mech = RattleMechanicalSystem(
            nq=1, nu=1,
            q0=np.array([0.0]), u0=np.array([0.0]),
            M=np.array([[m]]),
            h_force=lambda t, q, u: np.zeros(1),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[0]),
            W_N=np.array([w]),
            gamma_F=lambda t, q, u: np.zeros(1),
            W_F=np.array([[0.0]]),
            mu=0.0, n_F=1,
        )
        alpha = 0.5
        system = build_rattle_system(mech, contacts=[contact],
                                     prox_alpha=alpha)
        solver = RattleSolver(system)
        solver._compute_prox_params(0.0, np.array([0.0]))

        expected_r = alpha / (w * (1.0 / m) * w)
        np.testing.assert_allclose(solver.prox_r_N[0], expected_r, rtol=1e-12)

    def test_sparse_mass_matrix(self):
        """Delassus with sparse mass matrix."""
        m1, m2 = 3.0, 5.0
        M = sp.diags([m1, m2], format="csc")
        W_N = np.array([1.0, 0.0])  # contact on DOF 0

        mech = RattleMechanicalSystem(
            nq=2, nu=2,
            q0=np.zeros(2), u0=np.zeros(2),
            M=M,
            h_force=lambda t, q, u: np.zeros(2),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[0]),
            W_N=W_N,
            gamma_F=lambda t, q, u: np.zeros(1),
            W_F=np.array([[0.0], [0.0]]),
            mu=0.0, n_F=1,
        )
        alpha = 1.0
        system = build_rattle_system(mech, contacts=[contact],
                                     prox_alpha=alpha)
        solver = RattleSolver(system)
        solver._compute_prox_params(0.0, np.zeros(2))

        # W_N^T M^{-1} W_N = 1/m1 = 1/3
        expected_r = alpha / (1.0 / m1)
        np.testing.assert_allclose(solver.prox_r_N[0], expected_r, rtol=1e-12)


# ===================================================================
#  B. Integration tests -- unconstrained dynamics
# ===================================================================

class TestUnconstrainedDynamics:
    """Test RATTLE as a pure symplectic Lobatto IIIA-IIIB integrator."""

    def test_free_particle(self):
        """Free particle: q(t) = q0 + u0*t, u(t) = u0."""
        q0 = np.array([1.0])
        u0 = np.array([2.0])
        mech = RattleMechanicalSystem(
            nq=1, nu=1, q0=q0, u0=u0,
            M=np.eye(1),
            h_force=lambda t, q, u: np.zeros(1),
        )
        system = build_rattle_system(mech)
        solver = RattleSolver(system)
        result = solver.solve((0.0, 1.0), n_steps=10)

        assert result.failure is None
        # Check final state.
        q_final = result.states[-1, :1]
        u_final = result.states[-1, 1:]
        np.testing.assert_allclose(q_final, q0 + u0 * 1.0, atol=1e-12)
        np.testing.assert_allclose(u_final, u0, atol=1e-12)

    def test_harmonic_oscillator_energy(self):
        """Harmonic oscillator: energy should be approximately conserved."""
        omega = 2.0 * np.pi
        k = omega ** 2
        q0 = np.array([1.0])
        u0 = np.array([0.0])

        def h_force(t, q, u):
            return np.array([-k * q[0]])

        mech = RattleMechanicalSystem(
            nq=1, nu=1, q0=q0, u0=u0,
            M=np.eye(1), h_force=h_force,
            dh_dq=lambda t, q, u: np.array([[-k]]),
            dh_du=lambda t, q, u: np.zeros((1, 1)),
        )
        system = build_rattle_system(mech)
        solver = RattleSolver(system)
        result = solver.solve((0.0, 1.0), n_steps=100)

        assert result.failure is None
        # Energy at each step.
        E = 0.5 * result.states[:, 1] ** 2 + 0.5 * k * result.states[:, 0] ** 2
        E0 = E[0]
        # Symplectic: energy error should be bounded, not growing.
        assert np.max(np.abs(E - E0)) < 0.05 * E0

    def test_harmonic_oscillator_convergence_order(self):
        """Verify O(h^2) convergence for the harmonic oscillator."""
        omega = 1.0
        k = omega ** 2
        T = 0.5

        def h_force(t, q, u):
            return np.array([-k * q[0]])

        q0 = np.array([1.0])
        u0 = np.array([0.0])
        # Analytical: q(T) = cos(omega*T), u(T) = -omega*sin(omega*T).
        q_exact = np.cos(omega * T)
        u_exact = -omega * np.sin(omega * T)

        errors = []
        steps_list = [50, 100, 200, 400]
        for ns in steps_list:
            mech = RattleMechanicalSystem(
                nq=1, nu=1, q0=q0.copy(), u0=u0.copy(),
                M=np.eye(1), h_force=h_force,
                dh_dq=lambda t, q, u: np.array([[-k]]),
                dh_du=lambda t, q, u: np.zeros((1, 1)),
            )
            system = build_rattle_system(mech)
            solver = RattleSolver(system)
            res = solver.solve((0.0, T), n_steps=ns)
            assert res.failure is None
            q_err = abs(res.states[-1, 0] - q_exact)
            u_err = abs(res.states[-1, 1] - u_exact)
            errors.append(max(q_err, u_err))

        # Check convergence rate.
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            # Should be ~4 for O(h^2) when doubling steps.
            assert ratio > 3.0, f"Convergence ratio {ratio:.2f} < 3.0 at refinement {i}"


# ===================================================================
#  C. Integration tests -- contact
# ===================================================================

class TestContactDynamics:
    """Test RATTLE with frictional unilateral contacts."""

    def _bouncing_ball_system(self, e=0.0, mu=0.0):
        """1D bouncing ball: q = height, u = velocity, g_N = q."""
        g = 9.81
        q0 = np.array([1.0])
        u0 = np.array([0.0])

        mech = RattleMechanicalSystem(
            nq=1, nu=1, q0=q0, u0=u0,
            M=np.eye(1),
            h_force=lambda t, q, u: np.array([-g]),
            dh_dq=lambda t, q, u: np.zeros((1, 1)),
            dh_du=lambda t, q, u: np.zeros((1, 1)),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[0]),
            W_N=np.array([1.0]),
            gamma_F=lambda t, q, u: np.zeros(1),
            W_F=np.array([[0.0]]),
            mu=mu, e=e, n_F=1,
        )
        return build_rattle_system(mech, contacts=[contact])

    def test_bouncing_ball_no_restitution(self):
        """Ball dropped from height 1: should hit ground and stay."""
        system = self._bouncing_ball_system(e=0.0)
        solver = RattleSolver(system, newton_tol=1e-12)
        result = solver.solve((0.0, 2.0), n_steps=400)

        assert result.failure is None
        # After impact, height should stay >= 0.
        q_hist = result.states[:, 0]
        assert np.all(q_hist >= -1e-8), f"Penetration: min q = {np.min(q_hist)}"
        # Final velocity should be ~0 (resting on ground).
        u_final = result.states[-1, 1]
        assert abs(u_final) < 0.5

    def test_bouncing_ball_restitution(self):
        """With COR e, rebound apex ~ e^2 * h_initial."""
        e = 0.8
        system = self._bouncing_ball_system(e=e)
        solver = RattleSolver(system, newton_tol=1e-12)
        result = solver.solve((0.0, 2.0), n_steps=800)

        assert result.failure is None
        q_hist = result.states[:, 0]
        # Find first local maximum after the first impact.
        # First impact is near t = sqrt(2*1/9.81) ~ 0.45s.
        # Rebound apex should be near e^2 * 1.0 = 0.64.
        mid_idx = len(q_hist) // 3
        post_impact_max = np.max(q_hist[mid_idx:2 * mid_idx])
        expected_apex = e ** 2 * 1.0
        # Allow 20% tolerance due to time discretization.
        assert abs(post_impact_max - expected_apex) < 0.3 * expected_apex, \
            f"Rebound apex {post_impact_max:.3f}, expected ~{expected_apex:.3f}"

    def test_persistent_contact(self):
        """Ball starting on ground: should stay with reaction = mg."""
        g = 9.81
        q0 = np.array([0.0])
        u0 = np.array([0.0])

        mech = RattleMechanicalSystem(
            nq=1, nu=1, q0=q0, u0=u0,
            M=np.eye(1),
            h_force=lambda t, q, u: np.array([-g]),
            dh_dq=lambda t, q, u: np.zeros((1, 1)),
            dh_du=lambda t, q, u: np.zeros((1, 1)),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[0]),
            W_N=np.array([1.0]),
            gamma_F=lambda t, q, u: np.zeros(1),
            W_F=np.array([[0.0]]),
            mu=0.0, e=0.0, n_F=1,
        )
        system = build_rattle_system(mech, contacts=[contact])
        solver = RattleSolver(system, newton_tol=1e-12)
        result = solver.solve((0.0, 1.0), n_steps=100)

        assert result.failure is None
        q_hist = result.states[:, 0]
        u_hist = result.states[:, 1]
        # Should stay on ground.
        assert np.all(q_hist >= -1e-8)
        # Velocity should stay ~0.
        assert np.max(np.abs(u_hist)) < 0.1


class TestStage1SemismoothNewton:
    """Opt-in semismooth-Newton variant of RATTLE Stage 1."""

    def _coupled_patch_system(self, n_contacts: int = 3):
        """Dense-mass patch: several unilateral contacts sharing one M."""
        n = 2 * n_contacts
        rng = np.random.default_rng(0)
        M_raw = rng.standard_normal((n, n))
        M = M_raw @ M_raw.T + 0.5 * np.eye(n)  # SPD
        g_accel = 9.81

        def h_force(t, q, u):
            f = np.zeros(n)
            for k in range(n_contacts):
                f[2 * k] = -g_accel
            return f

        mech = RattleMechanicalSystem(
            nq=n, nu=n, q0=np.zeros(n), u0=np.zeros(n),
            M=M,
            h_force=h_force,
            dh_dq=lambda t, q, u: np.zeros((n, n)),
            dh_du=lambda t, q, u: np.zeros((n, n)),
        )

        contacts = []
        for k in range(n_contacts):
            w_n = np.zeros(n); w_n[2 * k] = 1.0
            w_f = np.zeros((n, 1)); w_f[2 * k + 1, 0] = 1.0
            contacts.append(RattleContactSpec(
                g_N=(lambda t, q, idx=k: float(q[2 * idx])),
                W_N=w_n,
                gamma_F=(lambda t, q, u, idx=k: np.array([float(u[2 * idx + 1])])),
                W_F=w_f,
                mu=0.3, e=0.0, n_F=1,
            ))
        return build_rattle_system(
            mech, contacts=contacts,
            initial_normal_forces=np.full(n_contacts, g_accel),
        )

    def test_ssn_matches_fixed_point_on_bouncing_ball(self):
        """SSN must produce a trajectory consistent with the fixed-point path."""
        g = 9.81
        mech = RattleMechanicalSystem(
            nq=1, nu=1, q0=np.array([1.0]), u0=np.array([0.0]),
            M=np.eye(1),
            h_force=lambda t, q, u: np.array([-g]),
            dh_dq=lambda t, q, u: np.zeros((1, 1)),
            dh_du=lambda t, q, u: np.zeros((1, 1)),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[0]),
            W_N=np.array([1.0]),
            gamma_F=lambda t, q, u: np.zeros(1),
            W_F=np.array([[0.0]]),
            mu=0.0, e=0.0, n_F=1,
        )
        system = build_rattle_system(mech, contacts=[contact])
        fp = RattleSolver(system, newton_tol=1e-12, stage1_method="fixed_point")
        ssn = RattleSolver(system, newton_tol=1e-12, stage1_method="semismooth_newton")
        r_fp = fp.solve((0.0, 2.0), n_steps=400)
        r_ssn = ssn.solve((0.0, 2.0), n_steps=400)
        assert r_fp.failure is None and r_ssn.failure is None
        # Bouncing ball stalls on the obstacle; both variants must agree to
        # better than the FP stage-1 precision floor.
        assert np.allclose(r_fp.states, r_ssn.states, atol=1e-7, rtol=1e-7)

    def test_ssn_drives_gap_to_roundoff_on_dense_delassus(self):
        """On a coupled dense-mass patch, SSN drives g_N to roundoff."""
        system = self._coupled_patch_system(n_contacts=3)
        ssn = RattleSolver(
            system,
            newton_tol=1.0e-12,
            # Stage-2 is still the velocity-level prox: give it plenty of
            # sweeps so it is not the bottleneck for this precision test.
            fixed_point_max_iter=400,
            fixed_point_tol=1.0e-10,
            stage1_method="semismooth_newton",
        )
        r_ssn = ssn.solve((0.0, 0.5), n_steps=200)
        assert r_ssn.failure is None
        n_q = system.mech.nq
        q_ssn = r_ssn.states[:, 0:n_q][:, 0::2]
        max_gap_ssn = float(np.max(np.abs(q_ssn)))
        assert max_gap_ssn < 1.0e-13, f"SSN max |g_N| = {max_gap_ssn:.3e}"

    def test_ssn_rejects_multidim_friction(self):
        """n_F > 1 must raise until SOC friction is implemented for SSN."""
        mech = RattleMechanicalSystem(
            nq=3, nu=3, q0=np.zeros(3), u0=np.zeros(3),
            M=np.eye(3),
            h_force=lambda t, q, u: np.array([0.0, 0.0, -9.81]),
            dh_dq=lambda t, q, u: np.zeros((3, 3)),
            dh_du=lambda t, q, u: np.zeros((3, 3)),
        )
        contact = RattleContactSpec(
            g_N=lambda t, q: float(q[2]),
            W_N=np.array([0.0, 0.0, 1.0]),
            gamma_F=lambda t, q, u: np.array([float(u[0]), float(u[1])]),
            W_F=np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
            mu=0.3, e=0.0, n_F=2,
        )
        system = build_rattle_system(mech, contacts=[contact])
        solver = RattleSolver(system, stage1_method="semismooth_newton")
        with pytest.raises(NotImplementedError, match="scalar friction"):
            solver.solve((0.0, 0.1), n_steps=10)


# ===================================================================
#  D. Legacy adapter compatibility
# ===================================================================

class TestLegacyAdapter:
    """Test backward compatibility of build_dynamic_rattle_contact."""

    def test_api_signature(self):
        """Old call signature should produce a valid RattleContactSystem."""
        n = 4
        A = np.eye(n)
        y0 = np.array([0.0, 0.0, 1.0, 0.0])  # [q_x, q_y, v_x, v_y]

        def rhs(t, y):
            return np.array([y[2], y[3], 0.0, -9.81])

        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.3)]

        B = np.eye(n, 2)  # 2 reaction DOFs
        gap_extract = np.zeros((2, n))
        gap_extract[0, 0] = -1.0  # gap = -q_x
        vel_extract = np.zeros((2, n))
        vel_extract[0, 2] = 1.0  # v_x
        vel_extract[1, 3] = 1.0  # v_y

        system = build_dynamic_rattle_contact(
            A, rhs, y0, contacts,
            B=B, gap_extract=gap_extract, vel_extract=vel_extract,
            n_base=2, velocity_slice=slice(2, 4),
        )

        assert isinstance(system, RattleContactSystem)
        assert system.mech.nq == 2
        assert system.mech.nu == 2
        assert len(system.contacts) == 1

    def test_solve_api(self):
        """solve_dynamic_rattle_contact should return RattleSolveResult."""
        m = 1.0
        g = 9.81
        n = 4
        A = np.diag([1.0, 1.0, m, m])
        y0 = np.array([0.0, 1.0, 0.0, 0.0])

        def rhs(t, y):
            return np.array([y[2], y[3], 0.0, -m * g])

        def rhs_jac(t, y):
            J = np.zeros((4, 4))
            J[0, 2] = 1.0
            J[1, 3] = 1.0
            return J

        contacts = [dict(vel_normal_idx=0, vel_tangential_idx=[1], mu=0.0)]
        B = np.zeros((n, 2))
        B[2, 0] = 1.0
        B[3, 1] = 1.0
        gap_extract = np.zeros((2, n))
        gap_extract[0, 1] = -1.0
        vel_extract = np.zeros((2, n))
        vel_extract[0, 3] = 1.0
        vel_extract[1, 2] = 1.0

        system = build_dynamic_rattle_contact(
            A, rhs, y0, contacts,
            B=B, gap_extract=gap_extract, vel_extract=vel_extract,
            n_base=2, velocity_slice=slice(2, 4),
            rhs_jac=rhs_jac,
        )

        result = solve_dynamic_rattle_contact(system, (0.0, 0.2), n_steps=20)
        assert isinstance(result, RattleSolveResult)
        assert result.times.shape[0] > 1
        assert result.states.shape[0] == result.times.shape[0]


# ===================================================================
#  E. Data model tests
# ===================================================================

class TestDataModel:
    def test_mechanical_system_identity_B(self):
        """When B_kin=None and nq=nu, kinematic map is identity."""
        mech = RattleMechanicalSystem(
            nq=3, nu=3,
            q0=np.zeros(3), u0=np.ones(3),
        )
        B = mech.eval_B(0.0, np.zeros(3))
        assert sp.issparse(B)
        np.testing.assert_allclose(B.toarray(), np.eye(3))

    def test_mechanical_system_nq_ne_nu_requires_B(self):
        """nq != nu without B_kin should raise."""
        with pytest.raises(ValueError, match="nq must equal nu"):
            RattleMechanicalSystem(nq=4, nu=3, q0=np.zeros(4), u0=np.zeros(3))

    def test_rattle_system_counts(self):
        mech = RattleMechanicalSystem(nq=2, nu=2, q0=np.zeros(2), u0=np.zeros(2))
        c1 = RattleContactSpec(
            g_N=lambda t, q: 0.0, W_N=np.zeros(2),
            gamma_F=lambda t, q, u: np.zeros(2), W_F=np.zeros((2, 2)),
            mu=0.3, n_F=2,
        )
        c2 = RattleContactSpec(
            g_N=lambda t, q: 0.0, W_N=np.zeros(2),
            gamma_F=lambda t, q, u: np.zeros(1), W_F=np.zeros((2, 1)),
            mu=0.0, n_F=1,
        )
        sys = RattleContactSystem(mech=mech, contacts=[c1, c2])
        assert sys.n_contacts == 2
        assert sys.nla_N == 2
        assert sys.nla_F == 3
        assert sys.has_contact is True
        assert sys.has_bilateral is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
