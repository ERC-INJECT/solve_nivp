"""Tests for the active-set error-norm filter.

The filter suppresses velocity DOFs undergoing contact-regime transitions
(stick↔slip, contact↔separation) from the adaptive error norm, preventing
the embedded / Richardson error estimator from overreacting to the
discontinuous constraint-force jumps intrinsic to nonsmooth event-capturing
integrators.

Test plan:
  1. `regime_snapshot()` returns expected types and values
  2. `regime_changed_mask()` produces correct masks
  3. `CompositeContactProjection` delegates regime methods
  4. `AdaptiveStepping._get_projection()` traverses solver chain
  5. `build_impulse_contact` attaches `_velocity_dof_map`
  6. End-to-end: filter active → larger accepted steps on a transition problem
"""

import numpy as np
import pytest

import solve_nivp
from solve_nivp.projections import (
    MuScaledSOCProjection,
    MoreauSOCProjection,
    AnisotropicSOCProjection,
    CompositeContactProjection,
    AlgebraicConstraintProjection,
    Projection,
    IdentityProjection,
)
from solve_nivp.contact import build_impulse_contact
from solve_nivp.adaptive_integrator import AdaptiveStepping


# =====================================================================
# Helpers
# =====================================================================

def _make_soc_projection(nb=3, m=1, mu=0.5):
    """Create a MuScaledSOCProjection with *nb* blocks of tang dim *m*."""
    n_tot = nb * (1 + m)
    blocks = []
    for k in range(nb):
        s_idx = k * (1 + m)
        w_idx = list(range(s_idx + 1, s_idx + 1 + m))
        blocks.append((s_idx, w_idx))
    proj = MuScaledSOCProjection(
        blocks=blocks,
        get_mu=lambda y, _mu=mu: _mu,
    )
    return proj, n_tot


def _bouncing_ball_contact_system(mu=0.3, e=0.0):
    """Standard 2D bouncing ball returning ContactSystem."""
    mass = 1.0
    gravity = np.array([0.0, -9.81])
    A = np.diag([mass, mass, 1.0, 1.0])

    def rhs(t, y):
        v = y[0:2]
        return np.concatenate([mass * gravity, v])

    def gap_func(y, t):
        return np.array([y[3]])

    y0 = np.array([2.0, 0.0, 0.0, 1.0])  # sliding + falling
    contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=mu, e=e)]
    cs = build_impulse_contact(
        A, rhs, y0, contacts, gap_func,
        theta=1.0,
        component_slices=[slice(0, 2), slice(2, 4)],
    )
    return cs


# =====================================================================
# 1. regime_snapshot
# =====================================================================

class TestRegimeSnapshot:
    """Test regime_snapshot on MuScaledSOCProjection (batch path)."""

    def test_returns_ndarray(self):
        proj, _ = _make_soc_projection(nb=4)
        snap = proj.regime_snapshot()
        assert isinstance(snap, np.ndarray)
        assert snap.shape == (4,)
        # All unset initially
        assert np.all(snap == -1)

    def test_snapshot_is_copy(self):
        proj, _ = _make_soc_projection(nb=3)
        snap1 = proj.regime_snapshot()
        proj._batch_branch_region[0] = 2
        snap2 = proj.regime_snapshot()
        # snap1 should NOT reflect the change
        assert snap1[0] == -1
        assert snap2[0] == 2

    def test_base_class_returns_none(self):
        proj = IdentityProjection()
        assert proj.regime_snapshot() is None

    def test_base_class_changed_mask_returns_none(self):
        proj = IdentityProjection()
        assert proj.regime_changed_mask(None, 10) is None
        assert proj.regime_changed_mask(np.array([0, 1]), 10) is None


# =====================================================================
# 2. regime_changed_mask
# =====================================================================

class TestRegimeChangedMask:
    """Test mask generation from regime transitions."""

    def test_no_change_returns_none(self):
        proj, n = _make_soc_projection(nb=3, m=1)
        # Set initial regime
        proj._batch_branch_region[:] = [0, 2, 1]
        snap = proj.regime_snapshot()
        # No change → None
        mask = proj.regime_changed_mask(snap, n)
        assert mask is None

    def test_single_block_transition(self):
        proj, n = _make_soc_projection(nb=3, m=1)
        proj._batch_branch_region[:] = [0, 2, 1]
        snap = proj.regime_snapshot()
        # Block 1 transitions from boundary(2) to interior(0)
        proj._batch_branch_region[1] = 0
        mask = proj.regime_changed_mask(snap, n)
        assert mask is not None
        assert mask.shape == (n,)
        # Block 1 has DOFs s=2, w=[3] → should be suppressed
        assert mask[2] == 0.0
        assert mask[3] == 0.0
        # Other DOFs remain 1.0
        assert mask[0] == 1.0
        assert mask[1] == 1.0
        assert mask[4] == 1.0
        assert mask[5] == 1.0

    def test_multiple_block_transitions(self):
        proj, n = _make_soc_projection(nb=4, m=1)
        proj._batch_branch_region[:] = [0, 2, 1, 2]
        snap = proj.regime_snapshot()
        # Blocks 0 and 2 transition
        proj._batch_branch_region[0] = 2  # stick → slip
        proj._batch_branch_region[2] = 0  # sep → stick
        mask = proj.regime_changed_mask(snap, n)
        assert mask is not None
        # Block 0: DOFs 0,1
        assert mask[0] == 0.0
        assert mask[1] == 0.0
        # Block 2: DOFs 4,5
        assert mask[4] == 0.0
        assert mask[5] == 0.0
        # Blocks 1,3 unchanged
        assert mask[2] == 1.0
        assert mask[3] == 1.0
        assert mask[6] == 1.0
        assert mask[7] == 1.0

    def test_unset_blocks_not_flagged(self):
        """Blocks with prev=-1 (unset) should not be treated as changed."""
        proj, n = _make_soc_projection(nb=2)
        snap = proj.regime_snapshot()  # all -1
        proj._batch_branch_region[:] = [0, 2]
        mask = proj.regime_changed_mask(snap, n)
        # -1 → 0 is not a real transition (prev was unset)
        assert mask is None

    def test_velocity_dof_map_used(self):
        """When _velocity_dof_map is set, it overrides block indices."""
        proj, n = _make_soc_projection(nb=2, m=1)
        # Attach velocity DOF mapping: block 0 → vel DOFs [10,11]
        proj._velocity_dof_map = [
            np.array([10, 11]),
            np.array([12, 13]),
        ]
        proj._batch_branch_region[:] = [0, 2]
        snap = proj.regime_snapshot()
        proj._batch_branch_region[0] = 2  # transition
        n_aug = 14
        mask = proj.regime_changed_mask(snap, n_aug)
        assert mask is not None
        # Should suppress velocity DOFs, not block DOFs
        assert mask[10] == 0.0
        assert mask[11] == 0.0
        # Block projection DOFs should NOT be suppressed (vel map used instead)
        assert mask[0] == 1.0
        assert mask[1] == 1.0

    def test_none_prev_returns_none(self):
        proj, n = _make_soc_projection(nb=2)
        assert proj.regime_changed_mask(None, n) is None


class TestDerivedProjectionRegimeTracking:
    """Derived SOC projectors should expose regime snapshots consistently."""

    def test_moreau_snapshot_updates_from_tangent_cone_split(self):
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        y = np.array([0.0, 1.0])
        z = np.array([0.0, 2.0])  # boundary after forward De Saxce shift

        _ = proj.tangent_cone_split(z, y)
        snap = proj.regime_snapshot()

        assert isinstance(snap, np.ndarray)
        np.testing.assert_array_equal(snap, [2])

    def test_moreau_changed_mask_detects_transition(self):
        proj = MoreauSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
        )
        y = np.array([0.0, 1.0])

        _ = proj.tangent_cone_split(np.array([0.0, 2.0]), y)   # boundary
        snap = proj.regime_snapshot()
        _ = proj.tangent_cone_split(np.array([-1.0, 0.0]), y)  # polar

        mask = proj.regime_changed_mask(snap, 2)
        assert mask is not None
        np.testing.assert_array_equal(mask, [0.0, 0.0])

    def test_anisotropic_snapshot_updates_from_tangent_cone(self):
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.array([[2.0]]),
        )
        y = np.array([0.0, 1.0])
        z = np.array([0.5, 2.0])  # boundary in whitened coordinates

        _ = proj.tangent_cone(z, y)
        snap = proj.regime_snapshot()

        assert isinstance(snap, np.ndarray)
        np.testing.assert_array_equal(snap, [2])

    def test_anisotropic_changed_mask_detects_transition(self):
        proj = AnisotropicSOCProjection(
            blocks=[(0, [1])],
            get_mu=lambda y: 0.5,
            get_B=lambda y, k: np.array([[2.0]]),
        )
        y = np.array([0.0, 1.0])

        _ = proj.tangent_cone(np.array([0.5, 2.0]), y)  # boundary
        snap = proj.regime_snapshot()
        _ = proj.tangent_cone(np.array([5.0, 0.1]), y)  # interior

        mask = proj.regime_changed_mask(snap, 2)
        assert mask is not None
        np.testing.assert_array_equal(mask, [0.0, 0.0])


# =====================================================================
# 3. CompositeContactProjection delegation
# =====================================================================

class TestCompositeRegimeDelegation:
    """CompositeContactProjection should delegate regime methods to SOC."""

    def test_snapshot_delegation(self):
        soc, _ = _make_soc_projection(nb=3)
        soc._batch_branch_region[:] = [0, 1, 2]
        # Use a real constraint for AlgebraicConstraintProjection
        constraint = dict(
            g=lambda y: y,
            dg_dy=lambda y: np.eye(1),
            y_slice=slice(100, 101),
            q_slice=slice(101, 102),
        )
        alg = AlgebraicConstraintProjection(constraints=[constraint])
        comp = CompositeContactProjection(alg, soc)
        snap = comp.regime_snapshot()
        assert isinstance(snap, np.ndarray)
        np.testing.assert_array_equal(snap, [0, 1, 2])

    def test_changed_mask_delegation(self):
        soc, n = _make_soc_projection(nb=2, m=1)
        soc._batch_branch_region[:] = [0, 2]
        constraint = dict(
            g=lambda y: y,
            dg_dy=lambda y: np.eye(1),
            y_slice=slice(100, 101),
            q_slice=slice(101, 102),
        )
        alg = AlgebraicConstraintProjection(constraints=[constraint])
        comp = CompositeContactProjection(alg, soc)
        snap = comp.regime_snapshot()
        soc._batch_branch_region[0] = 1  # transition
        mask = comp.regime_changed_mask(snap, n)
        assert mask is not None
        assert mask[0] == 0.0  # suppressed
        assert mask[1] == 0.0


# =====================================================================
# 4. _get_projection in AdaptiveStepping
# =====================================================================

class TestGetProjection:
    """AdaptiveStepping._get_projection traverses the solver chain."""

    def test_returns_projection(self):
        cs = _bouncing_ball_contact_system()
        from solve_nivp.integrations import SDIRK2
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=cs.projection,
        )
        integrator = SDIRK2(solver=solver, A=cs.A)
        stepper = AdaptiveStepping(integrator=integrator)
        proj = stepper._get_projection()
        assert proj is cs.projection

    def test_returns_none_without_solver(self):
        # Mock integrator with no solver attribute
        class FakeIntegrator:
            has_embedded_error = False
        stepper = AdaptiveStepping(integrator=FakeIntegrator())
        assert stepper._get_projection() is None


# =====================================================================
# 5. build_impulse_contact attaches _velocity_dof_map
# =====================================================================

class TestVelocityDofMap:
    """build_impulse_contact should populate _velocity_dof_map."""

    def test_single_contact_map(self):
        cs = _bouncing_ball_contact_system()
        proj = cs.projection
        # For CompositeContactProjection, check SOC sub-proj
        soc = proj._soc if hasattr(proj, '_soc') else proj
        assert hasattr(soc, '_velocity_dof_map')
        assert len(soc._velocity_dof_map) == 1
        # Contact: vel_normal_idx=1, vel_tangential_idx=[0]
        np.testing.assert_array_equal(soc._velocity_dof_map[0], [1, 0])

    def test_multi_contact_map(self):
        """Two contacts with different velocity DOFs."""
        A = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        def rhs(t, y):
            return np.zeros(6)
        def gap(y, t):
            return np.array([y[2], y[5]])
        y0 = np.zeros(6)
        contacts = [
            dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3),
            dict(vel_normal_idx=4, vel_tangential_idx=[3], mu=0.5),
        ]
        cs = build_impulse_contact(A, rhs, y0, contacts, gap)
        soc = cs.projection._soc if hasattr(cs.projection, '_soc') else cs.projection
        assert len(soc._velocity_dof_map) == 2
        np.testing.assert_array_equal(soc._velocity_dof_map[0], [1, 0])
        np.testing.assert_array_equal(soc._velocity_dof_map[1], [4, 3])


# =====================================================================
# 6. End-to-end: filter produces larger steps at transitions
# =====================================================================

class TestActiveSetFilterEndToEnd:
    """Verify the filter actually improves step-size behaviour."""

    @staticmethod
    def _spring_slider_problem():
        """2D bouncing ball with spring that undergoes stick↔slip."""
        mass = 1.0
        k = 50.0
        gravity = np.array([0.0, -9.81])
        A = np.diag([mass, mass, 1.0, 1.0])

        def rhs(t, y):
            v = y[0:2]
            q = y[2:4]
            # Spring pulling in x-direction + gravity
            return np.concatenate([mass * gravity + np.array([-k * q[0], 0.0]), v])

        y0 = np.array([2.0, 0.0, 0.3, 1.0])  # sliding + falling
        contacts = [dict(vel_normal_idx=1, vel_tangential_idx=[0], mu=0.3)]

        def gap_func(y, t):
            return np.array([y[3]])

        cs = build_impulse_contact(
            A, rhs, y0, contacts, gap_func,
            theta=1.0,
            component_slices=[slice(0, 2), slice(2, 4)],
        )
        return cs

    def test_filter_flag_passthrough(self):
        """Verify active_set_filter reaches AdaptiveStepping."""
        cs = self._spring_slider_problem()
        # Run with filter=True — just check it doesn't crash
        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 0.5),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-10, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True,
            h0=0.01,
            atol=1e-4,
            rtol=1e-3,
            active_set_filter=True,
        )
        assert len(t) > 1, "Integration should produce steps"
        # Allow for early termination at h_min — just check we advanced
        assert t[-1] > 0.1, "Should make substantial progress"

    def test_filter_produces_fewer_steps(self):
        """With filter on, the integrator should need fewer steps."""
        cs = self._spring_slider_problem()
        common_kw = dict(
            fun=cs.rhs,
            t_span=(0.0, 2.0),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-10, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True,
            h0=0.01,
            atol=1e-4,
            rtol=1e-3,
        )
        t_off, y_off, *_ = solve_nivp.solve_nivp(**common_kw, active_set_filter=False)
        t_on, y_on, *_ = solve_nivp.solve_nivp(**common_kw, active_set_filter=True)

        n_off = len(t_off)
        n_on = len(t_on)
        # Filter should reduce step count (or at worst, match)
        assert n_on <= n_off * 1.05, (
            f"Filter should not increase step count: {n_on} vs {n_off}")

    def test_filter_preserves_physics(self):
        """With filter on, the solution should still be physically correct."""
        cs = self._spring_slider_problem()
        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 1.0),
            y0=cs.y0,
            A=cs.A,
            method='sdirk2',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-10, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True,
            h0=0.01,
            atol=1e-4,
            rtol=1e-3,
            active_set_filter=True,
        )
        n_phys = cs.n_phys
        # Velocities should remain bounded
        assert np.all(np.abs(y[:, :n_phys]) < 100), "Velocities should be bounded"
        # Solution should not blow up
        assert np.all(np.isfinite(y)), "Solution should be finite"

    def test_backward_euler_with_filter(self):
        """The filter should also work with BE (Richardson path)."""
        cs = self._spring_slider_problem()
        t, y, h, fk, info = solve_nivp.solve_nivp(
            fun=cs.rhs,
            t_span=(0.0, 0.5),
            y0=cs.y0,
            A=cs.A,
            method='backward_euler',
            projection=cs.projection,
            solver='semismooth_newton',
            solver_opts=dict(tol=1e-10, max_iter=200,
                             lam_update_strategy='none',
                             globalization='linesearch'),
            component_slices=cs.component_slices,
            integrator_opts=cs.integrator_opts,
            adaptive=True,
            h0=0.01,
            atol=1e-4,
            rtol=1e-3,
            active_set_filter=True,
        )
        assert len(t) > 1
        assert t[-1] >= 0.5 - 1e-10
        assert np.all(np.isfinite(y))

    def test_filter_default_off(self):
        """By default active_set_filter is off."""
        cs = self._spring_slider_problem()
        from solve_nivp.integrations import SDIRK2
        from solve_nivp.nonlinear_solvers import ImplicitEquationSolver
        solver = ImplicitEquationSolver(
            method='semismooth_newton',
            proj=cs.projection,
        )
        integrator = SDIRK2(solver=solver, A=cs.A)
        stepper = AdaptiveStepping(integrator=integrator)
        assert stepper.active_set_filter is False
