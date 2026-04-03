"""Tests for AlgebraicConstraintProjection.

Exercises the projection with a known DAE system:
    m dy/dt = A y + B q,   q = C y
with singular mass matrix M = diag(I, 0) via all integration methods
and both nonlinear solver strategies (VI, semismooth_newton).
"""

import numpy as np
import pytest
from solve_nivp import (
    solve_nivp,
    AlgebraicConstraintProjection,
    ImplicitEquationSolver,
    BackwardEuler,
    Trapezoidal,
    CompositeMethod,
    SDIRK2,
    ODESystem,
    ODESolver,
)


# ---------- Fixtures / helpers ----------

def _build_linear_dae():
    """Build a simple linear DAE:  dy/dt = A_ode y + B q, q = C y.

    Analytic equivalent: dy/dt = (A_ode + B C) y,  so the exact
    solution is  y(t) = expm((A_ode + B C) t) y0.

    Returns (A_ode, B, C, M, y0_full, y_slice, q_slice, exact_at_T).
    """
    # 2-DOF differential + 2-DOF algebraic
    A_ode = np.array([[-1.0, 0.0],
                      [ 0.0, -2.0]])
    B = np.array([[0.5, 0.0],
                  [0.0, 0.5]])
    C = np.array([[2.0, 0.0],
                  [0.0, 3.0]])

    # Effective dynamics: dy/dt = (A_ode + B C) y = [[-1+1, 0], [0, -2+1.5]] y = [[0, 0], [0, -0.5]] y
    # So y1(t)=y1(0), y2(t)=y2(0)*exp(-0.5 t)
    y0_y = np.array([1.0, 1.0])
    y0_q = C @ y0_y  # consistent initialisation
    y0_full = np.hstack([y0_y, y0_q])

    # Singular mass matrix
    M = np.diag([1.0, 1.0, 0.0, 0.0])

    T = 1.0
    exact_y = np.array([1.0, np.exp(-0.5 * T)])
    exact_q = C @ exact_y
    exact_full = np.hstack([exact_y, exact_q])

    y_slice = slice(0, 2)
    q_slice = slice(2, 4)
    return A_ode, B, C, M, y0_full, y_slice, q_slice, T, exact_full


# ---------- Unit tests for the projection itself ----------

class TestProjectionUnit:
    """Direct tests of project() and tangent_cone()."""

    def test_project_enforces_constraint(self):
        C = np.array([[2.0, 0.0], [0.0, 3.0]])
        proj = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        state = np.array([1.0, 1.0, 0.0, 0.0])
        # candidate has wrong q values
        cand = np.array([1.5, 0.5, 99.0, -99.0])
        result = proj.project(state, cand)
        # y untouched, q = C @ y_candidate
        np.testing.assert_allclose(result[:2], [1.5, 0.5])
        np.testing.assert_allclose(result[2:], C @ np.array([1.5, 0.5]))

    def test_project_with_none_g_gives_zero(self):
        proj = AlgebraicConstraintProjection(
            g=None,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        cand = np.array([1.0, 2.0, 5.0, 6.0])
        result = proj.project(np.zeros(4), cand)
        np.testing.assert_allclose(result[2:], [0.0, 0.0])

    def test_tangent_cone_structure(self):
        """D should be identity on y-rows, dg/dy on q-rows (with 0 on q-cols)."""
        C = np.array([[2.0, 0.5], [0.3, 3.0]])
        proj = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        z = np.array([1.0, 1.0, 2.5, 3.3])
        D = proj.tangent_cone(z, z).toarray()
        # y-rows: identity
        np.testing.assert_allclose(D[:2, :2], np.eye(2))
        np.testing.assert_allclose(D[:2, 2:], 0.0)
        # q-rows: dg/dy on y-cols, 0 on q-cols
        np.testing.assert_allclose(D[2:, :2], C)
        np.testing.assert_allclose(D[2:, 2:], 0.0)

    def test_tangent_cone_fd_jacobian(self):
        """Finite-difference Jacobian fallback should match analytical."""
        C = np.array([[2.0, 0.5], [0.3, 3.0]])
        proj_exact = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        proj_fd = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=None,  # finite-difference
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        z = np.array([1.0, 2.0, 3.0, 4.0])
        D_exact = proj_exact.tangent_cone(z, z).toarray()
        D_fd = proj_fd.tangent_cone(z, z).toarray()
        np.testing.assert_allclose(D_fd, D_exact, atol=1e-5)

    def test_project_batch(self):
        C = np.array([[2.0, 0.0], [0.0, 3.0]])
        proj = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        state = np.zeros(4)
        candidates = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
        ])
        result = proj.project_batch(state, candidates)
        np.testing.assert_allclose(result[0, 2:], [2.0, 3.0])
        np.testing.assert_allclose(result[1, 2:], [4.0, 9.0])

    def test_nonlinear_constraint(self):
        """g(y) = y^2 — a nonlinear algebraic constraint."""
        proj = AlgebraicConstraintProjection(
            g=lambda y: y ** 2,
            dg_dy=lambda y: np.diag(2.0 * y),
            y_slice=slice(0, 2),
            q_slice=slice(2, 4),
        )
        cand = np.array([3.0, 4.0, 0.0, 0.0])
        result = proj.project(np.zeros(4), cand)
        np.testing.assert_allclose(result[2:], [9.0, 16.0])
        D = proj.tangent_cone(cand, cand).toarray()
        np.testing.assert_allclose(D[2:, :2], np.diag([6.0, 8.0]))


# ---------- Integration tests ----------

class TestAlgebraicDAEIntegration:
    """Full DAE integration using the algebraic projection through solve_nivp."""

    @pytest.fixture
    def dae_setup(self):
        return _build_linear_dae()

    def _run_dae(self, dae_setup, method, solver_type, atol=1e-8, rtol=1e-6, h0=1e-3):
        A_ode, B, C, M, y0, y_sl, q_sl, T, exact = dae_setup
        n = len(y0)

        def rhs(t, z):
            y, q = z[y_sl], z[q_sl]
            dy = A_ode @ y + B @ q
            # Algebraic residual: the projection handles enforcement,
            # so return (q - C y) here (Newton will try to zero it,
            # but the projection overwrites q exactly each iteration).
            dq = q - C @ y
            return np.concatenate([dy, dq])

        t, y, h, fk, info = solve_nivp(
            rhs, (0.0, T), y0,
            method=method,
            projection='algebraic',
            projection_opts=dict(
                g=lambda y_sub: C @ y_sub,
                dg_dy=lambda y_sub: C,
                y_slice=y_sl,
                q_slice=q_sl,
            ),
            solver=solver_type,
            A=M,
            h0=h0,
            atol=atol,
            rtol=rtol,
            adaptive=True,
            component_slices=[y_sl, q_sl],
            skip_error_indices=[1],  # skip algebraic block in error norm
        )
        return t, y, exact

    @pytest.mark.parametrize("method", [
        'backward_euler', 'trapezoidal', 'composite', 'sdirk2',
    ])
    def test_ssn_all_methods(self, dae_setup, method):
        t, y, exact = self._run_dae(dae_setup, method, 'semismooth_newton')
        # Check final state matches analytical solution
        np.testing.assert_allclose(y[-1], exact, atol=1e-3,
                                   err_msg=f"SSN + {method} failed")

    @pytest.mark.parametrize("method", [
        'backward_euler', 'trapezoidal', 'composite', 'sdirk2',
    ])
    def test_vi_all_methods(self, dae_setup, method):
        t, y, exact = self._run_dae(dae_setup, method, 'VI')
        np.testing.assert_allclose(y[-1], exact, atol=1e-3,
                                   err_msg=f"VI + {method} failed")

    def test_constraint_satisfied_exactly(self, dae_setup):
        """q = C y should hold to machine precision at every step."""
        A_ode, B, C, M, y0, y_sl, q_sl, T, exact = dae_setup
        n = len(y0)

        def rhs(t, z):
            y, q = z[y_sl], z[q_sl]
            dy = A_ode @ y + B @ q
            dq = q - C @ y
            return np.concatenate([dy, dq])

        t, y_hist, h, fk, info = solve_nivp(
            rhs, (0.0, T), y0,
            method='backward_euler',
            projection='algebraic',
            projection_opts=dict(
                g=lambda y_sub: C @ y_sub,
                dg_dy=lambda y_sub: C,
                y_slice=y_sl,
                q_slice=q_sl,
            ),
            solver='semismooth_newton',
            A=M,
            h0=5e-2,
            adaptive=False,
            component_slices=[y_sl, q_sl],
        )
        # At every time step, q should equal C @ y to machine precision
        for i in range(len(t)):
            y_i = y_hist[i, y_sl]
            q_i = y_hist[i, q_sl]
            np.testing.assert_allclose(
                q_i, C @ y_i, atol=1e-12,
                err_msg=f"Constraint violated at step {i}, t={t[i]:.4f}"
            )

    def test_no_drift_long_integration(self, dae_setup):
        """Over many steps, constraint error stays at machine eps (no drift)."""
        A_ode, B, C, M, y0, y_sl, q_sl, T, _ = dae_setup
        T_long = 10.0

        def rhs(t, z):
            y, q = z[y_sl], z[q_sl]
            dy = A_ode @ y + B @ q
            dq = q - C @ y
            return np.concatenate([dy, dq])

        t, y_hist, h, fk, info = solve_nivp(
            rhs, (0.0, T_long), y0,
            method='composite',
            projection='algebraic',
            projection_opts=dict(
                g=lambda y_sub: C @ y_sub,
                dg_dy=lambda y_sub: C,
                y_slice=y_sl,
                q_slice=q_sl,
            ),
            solver='semismooth_newton',
            A=M,
            h0=1e-2,
            adaptive=True,
            atol=1e-6,
            rtol=1e-4,
            component_slices=[y_sl, q_sl],
            skip_error_indices=[1],
        )
        # Constraint error at every accepted step
        max_viol = 0.0
        for i in range(len(t)):
            y_i = y_hist[i, y_sl]
            q_i = y_hist[i, q_sl]
            viol = np.max(np.abs(q_i - C @ y_i))
            max_viol = max(max_viol, viol)
        assert max_viol < 1e-12, f"Constraint drift detected: max violation = {max_viol}"


# ---------- Low-level construction test ----------

class TestLowLevelConstruction:
    """Verify manual construction (no solve_nivp) works end-to-end."""

    def test_manual_pipeline(self):
        """Build Projection → Solver → Integrator → ODESystem → ODESolver by hand."""
        C = np.array([[2.0, 0.0], [0.0, 3.0]])
        A_ode = np.array([[-1.0, 0.0], [0.0, -2.0]])
        B = np.array([[0.5, 0.0], [0.0, 0.5]])
        M = np.diag([1.0, 1.0, 0.0, 0.0])

        y_sl, q_sl = slice(0, 2), slice(2, 4)
        y0_y = np.array([1.0, 1.0])
        y0 = np.hstack([y0_y, C @ y0_y])

        proj = AlgebraicConstraintProjection(
            g=lambda y: C @ y,
            dg_dy=lambda y: C,
            y_slice=y_sl,
            q_slice=q_sl,
        )

        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=proj,
            component_slices=[y_sl, q_sl],
            tol=1e-10,
        )

        integrator = BackwardEuler(solver=solver, A=M)

        def rhs(t, z):
            y, q = z[y_sl], z[q_sl]
            dy = A_ode @ y + B @ q
            dq = q - C @ y
            return np.concatenate([dy, dq])

        system = ODESystem(rhs, y0, method=integrator, adaptive=False)
        driver = ODESolver(system, (0.0, 1.0), h=0.01)
        t, y_hist, h, fk, info = driver.solve()

        # Should get reasonable solution
        assert t[-1] == pytest.approx(1.0)
        # Constraint satisfied at final time
        np.testing.assert_allclose(y_hist[-1, q_sl], C @ y_hist[-1, y_sl], atol=1e-10)


# ---------- Timing comparison: projection vs substituted dynamics ----------

class TestSDIRK2TimingComparison:
    """Compare SDIRK2 performance: algebraic projection vs direct substitution.

    Both formulations integrate the *same* underlying system
        dy/dt = A y + B q,   q = C y
    to the same final time and tolerances, using the **same** augmented
    state [y; q] and singular mass matrix M = diag(I, 0).  They differ
    only in how the algebraic relation q = C y is enforced:

    * **Projection**: AlgebraicConstraintProjection overwrites q = C y
      exactly at every Newton iteration → machine-precision constraint.
    * **In-dynamics (identity proj)**: the algebraic residual q − C y
      lives entirely in the RHS; Newton drives 0 = q_{n+1} − C y_{n+1}
      to solver tolerance with IdentityProjection → constraint only
      satisfied to tol.
    """

    @staticmethod
    def _build_larger_dae(n_y=20, stiffness=100.0):
        """Build a larger linear DAE with stiff algebraic coupling.

        Parameters
        ----------
        n_y : int
            Number of differential DOFs; algebraic DOFs match (n_q = n_y).
        stiffness : float
            Scaling factor for the coupling matrix B.  Large values make
            constraint violations in q amplify strongly into dy/dt,
            creating a stiff algebraic-differential coupling where exact
            constraint enforcement matters.
        """
        rng = np.random.RandomState(42)
        # Stiff coupling: large B amplifies any constraint violation q-Cy
        B = stiffness * rng.randn(n_y, n_y)
        C = 0.5 * rng.randn(n_y, n_y)

        # Build a stable *effective* dynamics  A_eff = A_ode + B @ C,
        # then recover A_ode.  This guarantees stability regardless of
        # the stiffness level.
        A_eff = -np.eye(n_y) + 0.1 * rng.randn(n_y, n_y)
        A_eff = A_eff - np.eye(n_y) * (np.max(np.real(np.linalg.eigvals(A_eff))) + 1.0)
        A_ode = A_eff - B @ C   # A_ode + B C = A_eff  (stable)

        M = np.block([
            [np.eye(n_y),           np.zeros((n_y, n_y))],
            [np.zeros((n_y, n_y)),  np.zeros((n_y, n_y))],
        ])

        y0_y = rng.randn(n_y)
        y0_q = C @ y0_y
        y0_full = np.hstack([y0_y, y0_q])

        y_sl = slice(0, n_y)
        q_sl = slice(n_y, 2 * n_y)

        return A_ode, B, C, A_eff, M, y0_y, y0_full, y_sl, q_sl

    def test_sdirk2_projection_vs_in_dynamics(self):
        """Time both formulations and report; verify they agree on y(T).

        Both use the same augmented state [y; q] with the same singular
        mass matrix and the same RHS — the *only* difference is whether
        q = C y is enforced by projection or by Newton.
        """
        import time
        from scipy.linalg import expm

        n_y = 20
        T = 2.0
        atol, rtol = 1e-7, 1e-5
        h0 = 1e-3

        A_ode, B, C, A_eff, M, y0_y, y0_full, y_sl, q_sl = self._build_larger_dae(n_y)

        # Exact solution: y(T) = expm(A_eff * T) @ y0_y,  q(T) = C @ y(T)
        y_exact = expm(A_eff * T) @ y0_y
        q_exact = C @ y_exact

        # -- Shared RHS (identical for both formulations) --
        def rhs_augmented(t, z):
            y, q = z[y_sl], z[q_sl]
            dy = A_ode @ y + B @ q
            dq = q - C @ y   # algebraic residual
            return np.concatenate([dy, dq])

        n_runs = 5

        # ---- Formulation 1: Algebraic projection (exact constraint) ----
        times_proj = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            t_p, y_p, h_p, fk_p, info_p = solve_nivp(
                rhs_augmented, (0.0, T), y0_full,
                method='sdirk2',
                projection='algebraic',
                projection_opts=dict(
                    g=lambda y_sub: C @ y_sub,
                    dg_dy=lambda y_sub: C,
                    y_slice=y_sl,
                    q_slice=q_sl,
                ),
                solver='semismooth_newton',
                A=M,
                h0=h0,
                atol=atol,
                rtol=rtol,
                adaptive=True,
                component_slices=[y_sl, q_sl],
                skip_error_indices=[1],
            )
            times_proj.append(time.perf_counter() - t0)

        # ---- Formulation 2: Constraint in dynamics, IdentityProjection ----
        #      Newton naturally solves 0 = q_{n+1} - C y_{n+1} on the
        #      algebraic rows (M has zero rows there).
        times_dyn = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            t_d, y_d, h_d, fk_d, info_d = solve_nivp(
                rhs_augmented, (0.0, T), y0_full,
                method='sdirk2',
                projection='identity',
                solver='semismooth_newton',
                A=M,
                h0=h0,
                atol=atol,
                rtol=rtol,
                adaptive=True,
                component_slices=[y_sl, q_sl],
                skip_error_indices=[1],
            )
            times_dyn.append(time.perf_counter() - t0)

        # ---- Report ----
        med_proj = np.median(times_proj)
        med_dyn = np.median(times_dyn)
        ratio = med_proj / med_dyn if med_dyn > 0 else float('inf')

        # Constraint violation
        max_viol_proj = 0.0
        for i in range(len(t_p)):
            viol = np.max(np.abs(y_p[i, q_sl] - C @ y_p[i, y_sl]))
            max_viol_proj = max(max_viol_proj, viol)

        max_viol_dyn = 0.0
        for i in range(len(t_d)):
            viol = np.max(np.abs(y_d[i, q_sl] - C @ y_d[i, y_sl]))
            max_viol_dyn = max(max_viol_dyn, viol)

        # Solution accuracy vs exact (expm)
        err_y_proj = np.linalg.norm(y_p[-1, y_sl] - y_exact)
        err_y_dyn  = np.linalg.norm(y_d[-1, y_sl] - y_exact)
        err_q_proj = np.linalg.norm(y_p[-1, q_sl] - q_exact)
        err_q_dyn  = np.linalg.norm(y_d[-1, q_sl] - q_exact)

        print("\n" + "=" * 70)
        print("SDIRK2 Timing: Projection vs In-Dynamics  "
              "(n_y={}, T={}, stiffness=100)".format(n_y, T))
        print("=" * 70)
        print(f"  Projection (exact constraint via AlgebraicConstraintProjection):")
        print(f"    median = {med_proj*1e3:.2f} ms   "
              f"({len(t_p)} steps accepted)")
        print(f"  In-dynamics (Newton enforces q-Cy=0, IdentityProjection):")
        print(f"    median = {med_dyn*1e3:.2f} ms   "
              f"({len(t_d)} steps accepted)")
        print(f"  Ratio (projection / in-dynamics) = {ratio:.2f}x")
        print(f"  --- Constraint violation ---")
        print(f"  Max |q - Cy| (projection):   {max_viol_proj:.2e}")
        print(f"  Max |q - Cy| (in-dynamics):   {max_viol_dyn:.2e}")
        print(f"  --- Solution accuracy vs expm ---")
        print(f"  ||y(T) - y_exact|| (projection):  {err_y_proj:.2e}")
        print(f"  ||y(T) - y_exact|| (in-dynamics):  {err_y_dyn:.2e}")
        print(f"  ||q(T) - q_exact|| (projection):  {err_q_proj:.2e}")
        print(f"  ||q(T) - q_exact|| (in-dynamics):  {err_q_dyn:.2e}")
        print("=" * 70)

        # ---- Verify both produce reasonable solutions ----
        # Projection must maintain exact constraint
        assert max_viol_proj < 1e-12, (
            f"Projection constraint drift: {max_viol_proj}"
        )
        # In-dynamics only to solver tolerance
        assert max_viol_dyn < 1e-2, (
            f"In-dynamics constraint violation unexpectedly large: {max_viol_dyn}"
        )
        # Both should recover a sensible y(T) (the exact tolerance
        # depends on integrator order and step count)
        assert err_y_proj < 1.0, (
            f"Projection y(T) error unexpectedly large: {err_y_proj}"
        )
        assert err_y_dyn < 1.0, (
            f"In-dynamics y(T) error unexpectedly large: {err_y_dyn}"
        )


# ---------- Multi-constraint tests ----------

class TestMultiConstraint:
    """Tests for the ``constraints=[...]`` multi-constraint API."""

    def test_project_two_independent_constraints(self):
        """Two constraints with disjoint y_slices and q_slices."""
        # z = [p0, p1, u0, lam_q0, lam_q1, lam_s0]
        C_qp = np.array([[2.0, 0.5],
                         [0.3, 3.0]])   # lam_q = C_qp @ p
        C_su = np.array([[4.0]])         # lam_s = C_su @ u

        proj = AlgebraicConstraintProjection(constraints=[
            dict(g=lambda p: C_qp @ p, dg_dy=lambda p: C_qp,
                 y_slice=slice(0, 2), q_slice=slice(3, 5)),
            dict(g=lambda u: C_su @ u, dg_dy=lambda u: C_su,
                 y_slice=slice(2, 3), q_slice=slice(5, 6)),
        ])

        state = np.zeros(6)
        cand = np.array([1.0, 2.0, 3.0, 99.0, 99.0, 99.0])
        result = proj.project(state, cand)

        # Differential DOFs untouched
        np.testing.assert_allclose(result[:3], [1.0, 2.0, 3.0])
        # Constraint 1: lam_q = C_qp @ [1, 2]
        np.testing.assert_allclose(result[3:5], C_qp @ np.array([1.0, 2.0]))
        # Constraint 2: lam_s = C_su @ [3]
        np.testing.assert_allclose(result[5:6], C_su @ np.array([3.0]))

    def test_tangent_cone_two_constraints(self):
        """Tangent cone has correct block structure with two constraints."""
        C_qp = np.array([[2.0, 0.5],
                         [0.3, 3.0]])
        C_su = np.array([[4.0]])

        proj = AlgebraicConstraintProjection(constraints=[
            dict(g=lambda p: C_qp @ p, dg_dy=lambda p: C_qp,
                 y_slice=slice(0, 2), q_slice=slice(3, 5)),
            dict(g=lambda u: C_su @ u, dg_dy=lambda u: C_su,
                 y_slice=slice(2, 3), q_slice=slice(5, 6)),
        ])

        z = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        D = proj.tangent_cone(z, z).toarray()

        # Rows 0,1,2 (differential): identity
        np.testing.assert_allclose(D[:3, :3], np.eye(3))
        np.testing.assert_allclose(D[:3, 3:], 0.0)

        # Rows 3,4 (lam_q): dg1/dp on cols 0,1; zero elsewhere
        np.testing.assert_allclose(D[3:5, 0:2], C_qp)
        np.testing.assert_allclose(D[3:5, 2:], 0.0)

        # Row 5 (lam_s): dg2/du on col 2; zero elsewhere
        np.testing.assert_allclose(D[5:6, 2:3], C_su)
        np.testing.assert_allclose(D[5:6, 0:2], 0.0)
        np.testing.assert_allclose(D[5:6, 3:], 0.0)

    def test_tangent_cone_cache_updates(self):
        """Second call reuses structure and only updates Jg values."""
        # Nonlinear constraint: tangent changes with state
        proj = AlgebraicConstraintProjection(constraints=[
            dict(g=lambda y: y ** 2,
                 dg_dy=lambda y: np.diag(2.0 * y),
                 y_slice=slice(0, 2), q_slice=slice(2, 4)),
        ])

        z1 = np.array([1.0, 2.0, 0.0, 0.0])
        D1 = proj.tangent_cone(z1, z1).toarray().copy()
        # Jg block should be diag(2, 4)
        np.testing.assert_allclose(D1[2:, :2], np.diag([2.0, 4.0]))

        z2 = np.array([3.0, 5.0, 0.0, 0.0])
        D2 = proj.tangent_cone(z2, z2).toarray()
        # Same sparsity, updated values
        np.testing.assert_allclose(D2[2:, :2], np.diag([6.0, 10.0]))

    def test_project_batch_multi(self):
        """project_batch applies all constraints to every row."""
        C1 = np.array([[2.0]])
        C2 = np.array([[3.0]])

        proj = AlgebraicConstraintProjection(constraints=[
            dict(g=lambda y: C1 @ y, dg_dy=lambda y: C1,
                 y_slice=slice(0, 1), q_slice=slice(2, 3)),
            dict(g=lambda y: C2 @ y, dg_dy=lambda y: C2,
                 y_slice=slice(1, 2), q_slice=slice(3, 4)),
        ])

        candidates = np.array([
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
        ])
        result = proj.project_batch(np.zeros(4), candidates)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 2.0, 6.0])
        np.testing.assert_allclose(result[1], [3.0, 4.0, 6.0, 12.0])

    def test_fd_jacobian_multi(self):
        """FD fallback works for multi-constraint (no dg_dy provided)."""
        C_qp = np.array([[2.0, 0.5]])
        C_su = np.array([[4.0]])

        proj = AlgebraicConstraintProjection(constraints=[
            dict(g=lambda p: C_qp @ p,
                 y_slice=slice(0, 2), q_slice=slice(3, 4)),
            dict(g=lambda u: C_su @ u,
                 y_slice=slice(2, 3), q_slice=slice(4, 5)),
        ])

        z = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
        D = proj.tangent_cone(z, z).toarray()
        np.testing.assert_allclose(D[3, :2], [2.0, 0.5], atol=1e-5)
        np.testing.assert_allclose(D[4, 2], 4.0, atol=1e-5)

    def test_overlapping_q_slices_raises(self):
        """Overlapping q_slices must raise ValueError."""
        with pytest.raises(ValueError, match="overlap"):
            AlgebraicConstraintProjection(constraints=[
                dict(g=lambda y: y, y_slice=slice(0, 2), q_slice=slice(2, 4)),
                dict(g=lambda y: y, y_slice=slice(0, 2), q_slice=slice(3, 5)),
            ])

    def test_both_api_forms_raises(self):
        """Passing both single + constraints raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            AlgebraicConstraintProjection(
                g=lambda y: y,
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
                constraints=[dict(g=lambda y: y,
                                  y_slice=slice(0, 2),
                                  q_slice=slice(2, 4))],
            )

    def test_backward_compat_single_constraint(self):
        """Single-constraint form still exposes .g, .y_slice etc."""
        C = np.array([[2.0, 0.0], [0.0, 3.0]])
        g_fn = lambda y: C @ y
        proj = AlgebraicConstraintProjection(
            g=g_fn, dg_dy=lambda y: C,
            y_slice=slice(0, 2), q_slice=slice(2, 4),
        )
        assert proj.g is g_fn
        assert proj.y_slice == slice(0, 2)
        assert proj.q_slice == slice(2, 4)


class TestMultiConstraintDAEIntegration:
    """Full DAE integration with two independent algebraic constraints.

    System: 4-field poromechanics-like DAE
        p  (2 DOFs) — parabolic (pressure)
        u  (2 DOFs) — quasi-static (displacement)
        λ_q (2 DOFs) — algebraic: λ_q = C_qp @ p
        λ_σ (2 DOFs) — algebraic: λ_σ = C_su @ u

    Mass matrix M = diag(I_p, 0, 0, 0) — only pressure is dynamic.
    """

    @staticmethod
    def _build_system():
        Np, Nu, Nc = 2, 2, 2
        n = Np + Nu + Nc + Nc  # = 8

        p_sl  = slice(0, Np)
        u_sl  = slice(Np, Np + Nu)
        lq_sl = slice(Np + Nu, Np + Nu + Nc)
        ls_sl = slice(Np + Nu + Nc, n)

        # Coupling matrices
        K_pp = np.array([[-2.0, 0.1], [0.1, -3.0]])
        K_uu = np.array([[-4.0, 0.2], [0.2, -5.0]])
        B_q  = np.array([[0.5, 0.0], [0.0, 0.5]])
        B_s  = np.array([[0.3, 0.0], [0.0, 0.3]])
        C_up = np.array([[0.1, 0.0], [0.0, 0.1]])
        C_qp = np.array([[1.5, 0.2], [0.1, 2.0]])
        C_su = np.array([[2.0, 0.3], [0.2, 1.5]])

        # Mass: only pressure is dynamic
        M = np.zeros((n, n))
        M[:Np, :Np] = np.eye(Np)

        # IC
        p0 = np.array([1.0, 0.5])
        # Quasi-static u0 from momentum equation at p0 with λ_σ = 0
        u0 = np.linalg.solve(K_uu, -C_up @ p0)
        lq0 = C_qp @ p0
        ls0 = C_su @ u0
        z0 = np.concatenate([p0, u0, lq0, ls0])

        slices = dict(p=p_sl, u=u_sl, lq=lq_sl, ls=ls_sl)
        mats = dict(K_pp=K_pp, K_uu=K_uu, B_q=B_q, B_s=B_s,
                    C_up=C_up, C_qp=C_qp, C_su=C_su)
        return n, M, z0, slices, mats

    def test_multi_constraint_integration(self):
        """Both algebraic constraints satisfied to machine precision."""
        n, M, z0, sl, m = self._build_system()

        def rhs(t, z):
            p  = z[sl['p']]
            u  = z[sl['u']]
            lq = z[sl['lq']]
            ls = z[sl['ls']]

            dp = m['K_pp'] @ p + m['B_q'] @ lq
            du_res = m['C_up'] @ p + m['K_uu'] @ u + m['B_s'] @ ls  # 0 = ...
            dlq = lq - m['C_qp'] @ p
            dls = ls - m['C_su'] @ u
            return np.concatenate([dp, du_res, dlq, dls])

        C_qp, C_su = m['C_qp'], m['C_su']

        t, z, h, fk, info = solve_nivp(
            rhs, (0.0, 1.0), z0,
            method='sdirk2',
            projection='algebraic',
            projection_opts=dict(constraints=[
                dict(g=lambda p: C_qp @ p, dg_dy=lambda p: C_qp,
                     y_slice=sl['p'], q_slice=sl['lq']),
                dict(g=lambda u: C_su @ u, dg_dy=lambda u: C_su,
                     y_slice=sl['u'], q_slice=sl['ls']),
            ]),
            solver='semismooth_newton',
            A=M,
            h0=1e-2,
            atol=1e-7,
            rtol=1e-5,
            adaptive=True,
            component_slices=[sl['p'], sl['u'], sl['lq'], sl['ls']],
            skip_error_indices=[2, 3],
        )

        # Both constraints satisfied at every step
        max_viol_q = 0.0
        max_viol_s = 0.0
        for i in range(len(t)):
            viol_q = np.max(np.abs(z[i, sl['lq']] - C_qp @ z[i, sl['p']]))
            viol_s = np.max(np.abs(z[i, sl['ls']] - C_su @ z[i, sl['u']]))
            max_viol_q = max(max_viol_q, viol_q)
            max_viol_s = max(max_viol_s, viol_s)

        assert max_viol_q < 1e-12, f"λ_q constraint drift: {max_viol_q}"
        assert max_viol_s < 1e-12, f"λ_σ constraint drift: {max_viol_s}"
        assert t[-1] == pytest.approx(1.0)

    @pytest.mark.parametrize("method", [
        'backward_euler', 'trapezoidal', 'composite', 'sdirk2',
    ])
    def test_multi_constraint_all_methods(self, method):
        """Multi-constraint works with all integration methods (SSN)."""
        n, M, z0, sl, m = self._build_system()

        def rhs(t, z):
            p  = z[sl['p']]
            u  = z[sl['u']]
            lq = z[sl['lq']]
            ls = z[sl['ls']]
            dp = m['K_pp'] @ p + m['B_q'] @ lq
            du_res = m['C_up'] @ p + m['K_uu'] @ u + m['B_s'] @ ls
            dlq = lq - m['C_qp'] @ p
            dls = ls - m['C_su'] @ u
            return np.concatenate([dp, du_res, dlq, dls])

        C_qp, C_su = m['C_qp'], m['C_su']

        t, z, h, fk, info = solve_nivp(
            rhs, (0.0, 0.5), z0,
            method=method,
            projection='algebraic',
            projection_opts=dict(constraints=[
                dict(g=lambda p: C_qp @ p, dg_dy=lambda p: C_qp,
                     y_slice=sl['p'], q_slice=sl['lq']),
                dict(g=lambda u: C_su @ u, dg_dy=lambda u: C_su,
                     y_slice=sl['u'], q_slice=sl['ls']),
            ]),
            solver='semismooth_newton',
            A=M,
            h0=1e-2,
            atol=1e-6,
            rtol=1e-4,
            adaptive=True,
            component_slices=[sl['p'], sl['u'], sl['lq'], sl['ls']],
            skip_error_indices=[2, 3],
        )

        for i in range(len(t)):
            np.testing.assert_allclose(
                z[i, sl['lq']], C_qp @ z[i, sl['p']], atol=1e-12,
                err_msg=f"{method}: λ_q violated at step {i}")
            np.testing.assert_allclose(
                z[i, sl['ls']], C_su @ z[i, sl['u']], atol=1e-12,
                err_msg=f"{method}: λ_σ violated at step {i}")


class TestClosureCaptureArity:
    """Ensure lambdas with default-parameter closure capture are handled.

    The common Python pattern ``lambda p, _C=matrix: _C @ p`` has 1
    **required** positional arg but 2 total.  The arity detector must
    count only required args so that the time value ``t`` is never
    passed as ``_C``.
    """

    def test_default_param_g_is_1_arg(self):
        """g=lambda p, _C=M: _C @ p  must be treated as 1-arg."""
        C = np.array([[1.0, 2.0], [3.0, 4.0]])
        proj = AlgebraicConstraintProjection(
            constraints=[dict(
                g=lambda p, _C=C: _C @ p,
                dg_dy=lambda p, _C=C: _C,
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
            )]
        )
        y = np.array([1.0, 0.5, 0.0, 0.0])
        out = proj.project(y, y.copy(), t=99.9)
        np.testing.assert_allclose(out[2:4], C @ y[:2], atol=1e-14)

    def test_default_param_with_sparse_matrix(self):
        """Sparse-matrix closure capture — mirrors real poromechanics usage."""
        from scipy import sparse as sp
        C_dense = np.array([[2.0, 0.0], [0.0, 3.0]])
        C_sp = sp.csr_matrix(C_dense)
        proj = AlgebraicConstraintProjection(
            constraints=[dict(
                g=lambda p, _C=C_sp: np.asarray((_C @ p)).ravel(),
                dg_dy=lambda p, _C=C_sp: _C,
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
            )]
        )
        y = np.array([1.0, 0.5, 0.0, 0.0])
        out = proj.project(y, y.copy(), t=0.42, Fk_val=np.zeros(4))
        np.testing.assert_allclose(out[2:4], C_dense @ y[:2], atol=1e-14)

    def test_default_param_tangent_cone(self):
        """tangent_cone must also handle closure-capture dg_dy correctly."""
        C = np.array([[1.0, 0.0], [0.0, 1.0]])
        proj = AlgebraicConstraintProjection(
            constraints=[dict(
                g=lambda p, _C=C: _C @ p,
                dg_dy=lambda p, _C=C: _C,
                y_slice=slice(0, 2),
                q_slice=slice(2, 4),
            )]
        )
        y = np.array([1.0, 2.0, 0.0, 0.0])
        T = proj.tangent_cone(y, y, t=1.5)
        # Rows 0-1 (y DOFs): identity.
        # Rows 2-3 (q DOFs): dg/dy=I placed in y-columns → [1,0,0,0],[0,1,0,0]
        expected = np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
        ])
        np.testing.assert_allclose(T.toarray(), expected, atol=1e-14)

    def test_multi_constraint_closure_capture_integration(self):
        """Full integration with 4 closure-capture constraints (user pattern)."""
        n = 2
        C1 = np.eye(n) * 1.5
        C2 = np.eye(n) * 0.8
        C3 = np.eye(n) * 2.0
        C4 = np.eye(n) * 0.3

        N = 6 * n  # p, u, lq+, lq-, ls+, ls-
        sl = {
            'p': slice(0, n), 'u': slice(n, 2*n),
            'lqp': slice(2*n, 3*n), 'lqm': slice(3*n, 4*n),
            'lsp': slice(4*n, 5*n), 'lsm': slice(5*n, 6*n),
        }

        constraints = [
            dict(g=lambda p, _C=C1: _C @ p, dg_dy=lambda p, _C=C1: _C,
                 y_slice=sl['p'], q_slice=sl['lqp']),
            dict(g=lambda p, _C=C2: _C @ p, dg_dy=lambda p, _C=C2: _C,
                 y_slice=sl['p'], q_slice=sl['lqm']),
            dict(g=lambda u, _C=C3: _C @ u, dg_dy=lambda u, _C=C3: _C,
                 y_slice=sl['u'], q_slice=sl['lsp']),
            dict(g=lambda u, _C=C4: _C @ u, dg_dy=lambda u, _C=C4: _C,
                 y_slice=sl['u'], q_slice=sl['lsm']),
        ]

        A_mass = np.zeros((N, N))
        A_mass[:n, :n] = np.eye(n)  # only p is dynamic

        # Quasi-static coupling so u rows are not trivially zero
        K_uu = np.eye(n) * 2.0
        C_up = np.eye(n) * 0.5

        def rhs(t, y):
            f = np.zeros(N)
            f[sl['p']] = -0.1 * y[sl['p']]                        # pressure decay
            f[sl['u']] = -C_up @ y[sl['p']] - K_uu @ y[sl['u']]   # quasi-static momentum
            return f

        y0 = np.zeros(N)
        y0[:n] = 1.0
        # Consistent u0: 0 = -C_up p0 - K_uu u0  =>  u0 = -K_uu^{-1} C_up p0
        y0[sl['u']] = np.linalg.solve(K_uu, -C_up @ y0[sl['p']])
        y0[sl['lqp']] = C1 @ y0[:n]
        y0[sl['lqm']] = C2 @ y0[:n]
        y0[sl['lsp']] = C3 @ y0[n:2*n]
        y0[sl['lsm']] = C4 @ y0[n:2*n]

        component_slices = [sl['p'], sl['u'], sl['lqp'], sl['lqm'],
                            sl['lsp'], sl['lsm']]
        t, z, *_ = solve_nivp(
            fun=rhs, t_span=(0.0, 0.5), y0=y0,
            method='backward_euler', projection='algebraic',
            solver='semismooth_newton',
            projection_opts=dict(constraints=constraints),
            A=A_mass, h0=0.1,
            component_slices=component_slices,
            skip_error_indices=[2, 3, 4, 5],  # algebraic blocks
        )

        assert len(t) >= 2
        for i in range(len(t)):
            np.testing.assert_allclose(z[i, sl['lqp']], C1 @ z[i, sl['p']], atol=1e-10)
            np.testing.assert_allclose(z[i, sl['lqm']], C2 @ z[i, sl['p']], atol=1e-10)
            np.testing.assert_allclose(z[i, sl['lsp']], C3 @ z[i, sl['u']], atol=1e-10)
            np.testing.assert_allclose(z[i, sl['lsm']], C4 @ z[i, sl['u']], atol=1e-10)

    def test_build_constraint_patch_repeated_zero_jacobian_keeps_identity_rows(self):
        """Zero-dg constraints must not lose their q=q identity rows on cache reuse."""
        n = 6
        q_sl = slice(3, 6)
        proj = AlgebraicConstraintProjection(
            constraints=[dict(
                g=lambda y: np.zeros(3),
                dg_dy=lambda y: np.zeros((3, 3)),
                y_slice=q_sl,
                q_slice=q_sl,
            )]
        )

        y0 = np.zeros(n)
        y1 = np.array([0.0, 0.0, 0.0, 1.0, -2.0, 3.0])

        patch0 = proj.build_constraint_patch(y0, n).toarray()
        patch1 = proj.build_constraint_patch(y1, n).toarray()

        expected = np.zeros((n, n))
        expected[q_sl, q_sl] = np.eye(3)

        np.testing.assert_allclose(patch0, expected, atol=1e-14)
        np.testing.assert_allclose(patch1, expected, atol=1e-14)
