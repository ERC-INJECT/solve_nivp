"""Complete Eq. 19 activation for the descriptor MJF stepper.

The paper's old-state forecast admits a contact to the cone problem only when
both ``g_N(q_k) <= gap_tol`` and ``u_N,k <= 0``.  The velocity zero needs a
row-scaled numerical tolerance because the state and extraction matvec are
already floating-point approximations.
"""

import numpy as np
import pytest

from solve_nivp.moreau_jean_fremond import (
    DescriptorMoreauJeanFremondStepper,
)


def build_stepper(
    *,
    D=None,
    gap=True,
    acceleration=0.0,
    offset_force=None,
    restitution=0.0,
    normal_velocity_atol=0.0,
    normal_velocity_rtol=np.sqrt(np.finfo(float).eps),
):
    """Return a minimal point contact with state ``[gap, velocity, ...]``."""

    D = (
        np.array([[0.0, 1.0]], dtype=float)
        if D is None
        else np.asarray(D, dtype=float)
    )
    n_state = D.shape[1]
    A = np.eye(n_state)
    B = np.zeros((n_state, 1))
    B[1, 0] = 1.0

    def rhs(_t, y):
        out = np.zeros(n_state)
        out[0] = y[1]
        out[1] = acceleration
        return out

    def rhs_jac(_t, _y):
        out = np.zeros((n_state, n_state))
        out[0, 1] = 1.0
        return out

    offset = None
    if offset_force is not None:
        offset = lambda _y, _t, load=float(offset_force): np.array([load])

    return DescriptorMoreauJeanFremondStepper(
        A=A,
        rhs_callable=rhs,
        rhs_jac_callable=rhs_jac,
        D_extract=D,
        B=B,
        contacts=[
            {
                "vel_normal_idx": 0,
                "vel_tangential_idx": [],
                "mu_init": 0.0,
                "e": restitution,
            }
        ],
        gap_callable=(
            (lambda y, _t=0.0: np.array([y[0]])) if gap else None
        ),
        gap_tol=0.0,
        normal_velocity_atol=normal_velocity_atol,
        normal_velocity_rtol=normal_velocity_rtol,
        contact_offset_force=offset,
        theta=0.5,
        contact_solver="pgs",
        theta_linear_solver="scipy",
        combined_projection=False,
    )


def active_indices(stepper, y):
    active = stepper._geometric_active_blocks(
        np.asarray(y, dtype=float), 0.0
    )
    if active is None:
        return np.arange(stepper.n_contacts)
    return np.asarray(active)


def test_eq19_excludes_closed_resolved_separating_contact():
    assert active_indices(build_stepper(), [0.0, 1.0]).size == 0


def test_eq19_keeps_closed_closing_contact():
    np.testing.assert_array_equal(
        active_indices(build_stepper(), [0.0, -1.0]), [0]
    )


def test_scale_aware_zero_keeps_cancellation_level_positive_velocity():
    D = np.array([[0.0, 1.0, -1.0]])
    y = np.array([0.0, 1.0, 1.0 - 1.0e-12])
    assert (D @ y).item() > 0.0
    np.testing.assert_array_equal(active_indices(build_stepper(D=D), y), [0])


def test_scale_aware_zero_does_not_count_zero_valued_row_terms():
    # The base tolerance is the componentwise matvec scale abs(D)@abs(y).
    # Coefficients multiplying exact zero state entries must not inflate it;
    # state-solve error is handled separately by the prior-step consistency
    # defect rather than by a broad support norm.
    rtol = np.sqrt(np.finfo(float).eps)
    D = np.array([[0.0, 1.0, -1.0, 1.0, -1.0]])
    y = np.array([0.0, 1.0, 1.0 - 3.0 * rtol, 0.0, 0.0])
    u_n = (D @ y).item()
    componentwise_tol = rtol * (np.abs(D) @ np.abs(y)).item()
    support_normwise_tol = rtol * np.linalg.norm(D, ord=1, axis=1).item()

    assert componentwise_tol < u_n < support_normwise_tol
    assert active_indices(build_stepper(D=D), y).size == 0


def test_scale_aware_zero_rejects_resolved_positive_velocity():
    D = np.array([[0.0, 1.0, -1.0]])
    y = np.array([0.0, 1.0, 1.0 - 1.0e-5])
    assert active_indices(build_stepper(D=D), y).size == 0


def test_relative_velocity_classification_is_state_scale_invariant():
    D = np.array([[0.0, 1.0, -1.0]])
    base = np.array([0.0, 1.0, 1.0 - 1.0e-12])
    for factor in (1.0e-6, 1.0, 1.0e6):
        y = base.copy()
        y[1:] *= factor
        np.testing.assert_array_equal(
            active_indices(build_stepper(D=D), y), [0]
        )


def test_velocity_scale_ignores_unrelated_descriptor_fields():
    D = np.array([[0.0, 1.0, 0.0]])
    y = np.array([0.0, 1.0, 1.0e20])
    assert active_indices(build_stepper(D=D), y).size == 0


def test_prior_closed_contact_defect_supplements_scale_aware_zero():
    stepper = build_stepper()
    y = np.array([0.0, 2.0e-8])
    assert active_indices(stepper, y).size == 0

    active = stepper._geometric_active_blocks(
        y,
        0.0,
        normal_velocity_error=np.array([3.0e-8]),
    )
    assert active is None


def test_prior_contact_defect_classification_is_unit_scale_invariant():
    stepper = build_stepper()
    for factor in (1.0e-6, 1.0, 1.0e6):
        active = stepper._geometric_active_blocks(
            np.array([0.0, 2.0e-8 * factor]),
            0.0,
            normal_velocity_error=np.array([3.0e-8 * factor]),
        )
        assert active is None


def test_prior_contact_defect_does_not_hide_resolved_separation():
    stepper = build_stepper()
    active = stepper._geometric_active_blocks(
        np.array([0.0, 1.0e-5]),
        0.0,
        normal_velocity_error=np.array([1.0e-12]),
    )
    assert active.size == 0


def test_endpoint_defect_removes_physical_rebound_and_ignores_inactive_contact():
    stepper = build_stepper(restitution=0.5)
    numerical_error = 3.0e-13
    defect = stepper._normal_endpoint_consistency_error(
        u_N_new=np.array([1.0 + numerical_error]),
        u_N_old=np.array([-2.0]),
        regimes=["stick"],
    )
    np.testing.assert_allclose(defect, [numerical_error], rtol=1.0e-3)

    inactive = stepper._normal_endpoint_consistency_error(
        u_N_new=np.array([1.0e-4]),
        u_N_old=np.array([0.0]),
        regimes=["inactive"],
    )
    np.testing.assert_array_equal(inactive, [0.0])


def test_genuinely_inactive_contact_is_not_latched_by_prior_error_bound():
    stepper = build_stepper(acceleration=1.0)
    y0 = np.array([0.0, 0.0])
    y1, aux1, info1 = stepper.step(
        0.0, y0, {"mu": np.array([0.0])}, 0.1
    )

    assert info1["regime"] == ["separation"]
    np.testing.assert_array_equal(
        aux1["normal_velocity_error_bound"], [0.0]
    )
    assert y1[1] > 0.0
    active = stepper._geometric_active_blocks(
        y1,
        0.1,
        normal_velocity_error=aux1["normal_velocity_error_bound"],
    )
    assert active.size == 0


def test_gap_callable_none_preserves_persistent_contact():
    np.testing.assert_array_equal(
        active_indices(build_stepper(gap=False), [2.0, 1.0]), [0]
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("normal_velocity_atol", -1.0),
        ("normal_velocity_atol", np.nan),
        ("normal_velocity_atol", np.inf),
        ("normal_velocity_rtol", -1.0),
        ("normal_velocity_rtol", np.nan),
        ("normal_velocity_rtol", np.inf),
    ],
)
def test_eq19_velocity_tolerances_must_be_finite_nonnegative(name, value):
    with pytest.raises(ValueError, match=name):
        build_stepper(**{name: value})


def test_eq19_separating_contact_has_zero_total_reaction_and_releases_offset():
    # The constant offset is the background prestress in an affine force
    # decomposition.  Opening sets the total reaction to zero by applying the
    # equal-and-opposite reaction increment; the offset itself is never gated.
    stepper = build_stepper(offset_force=2.0)
    y0 = np.array([0.0, 1.0])
    y1, _aux, info = stepper.step(
        0.0, y0, {"mu": np.array([0.0])}, 1.0
    )

    np.testing.assert_allclose(
        info["p_contact_effective"], [0.0], atol=1.0e-12
    )
    np.testing.assert_allclose(info["p_contact"], [-2.0], atol=1.0e-12)
    np.testing.assert_allclose(y1, [0.0, -1.0], atol=1.0e-12)

    energy = lambda y: 0.5 * y[1] ** 2 + 2.0 * y[0]
    np.testing.assert_allclose(energy(y1), energy(y0), atol=1.0e-12)
