"""Horizon-landing regression tests.

Fixed/adaptive drivers accumulate ``t += h`` in floating point.  After many
steps ``t`` drifts a sliver (~1e-13) off the exact grid; a driver that then
appends a remainder step to reach ``tf`` takes a degenerate micro-step
(``h ~ 1e-13``).  For impulse/reaction-based steppers the reported reaction
(~ impulse/h) blows up and the cone/nonlinear solve stalls.  The drivers must
instead *snap* the final full step to land ``tf`` exactly.  These tests fail on
the pre-snap code (final ``t`` lands ~1e-13 short and, for the fixed-step path,
the appended micro-step's nonlinear solve fails).
"""
import numpy as np
import pytest


@pytest.mark.parametrize("tf", [10.0, 7.3, 3.33])
def test_fixed_step_lands_horizon_exactly_no_microstep(tf):
    import solve_nivp as sivp

    t, y, h, fk, e = sivp.solve_nivp(
        fun=lambda t, y: -y,
        t_span=(0.0, tf),
        y0=np.array([1.0]),
        method="backward_euler",
        projection="identity",
        solver="semismooth_newton",
        adaptive=False,
        h0=0.02,
    )
    t = np.asarray(t)
    h = np.asarray(h)
    assert t[-1] == tf                       # lands exactly, not ~1e-13 short
    assert np.all(np.isfinite(y))
    # no degenerate micro-step: every accepted step is a real fraction of h0
    assert h[1:].min() > 1.0e-6


def test_mjf_adaptive_lands_horizon_exactly_no_microstep():
    # Minimal 2-DOF De Saxce cone plant driven into slip; land a horizon that
    # does not divide evenly by the step (float drift near tf).
    import scipy.sparse as sp

    from solve_nivp.moreau_jean_fremond import (
        DescriptorMoreauJeanFremondStepper,
        solve_mjf_adaptive,
    )

    mu, N0 = 0.3, 1.0e6
    st = DescriptorMoreauJeanFremondStepper(
        A=sp.eye(2, format="csr"),
        rhs_callable=lambda t, y: np.array([0.0, 4.0e5]),
        rhs_jac_callable=lambda t, y: sp.csr_matrix((2, 2)),
        D_extract=np.eye(2), B=np.eye(2),
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu, "e": 0.0}],
        contact_offset_force=lambda y, t: np.array([N0, -mu * N0]),
        theta=0.5, aux_law="constant", contact_solver="petsc_ssn",
        contact_linear_solver="dense", contact_residual="soc_fb",
        theta_linear_solver="scipy",
    )
    tmax = 3.7
    T, Y, h, info, recs = solve_mjf_adaptive(
        st, (0.0, tmax), np.zeros(2), aux0={"mu": np.array([mu])},
        error_mask=np.array([False, True]), rtol=1e-4,
        atol=1e-6 * np.ones(2), h0=1e-2, h_max=0.2,
    )
    T = np.asarray(T)
    assert T[-1] == tmax                     # exact landing
    assert np.all(np.isfinite(Y))
    assert np.diff(T).min() > 1.0e-9         # no micro-step
