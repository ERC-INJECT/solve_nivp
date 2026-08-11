"""Exact affine-RHS fast-path tests for descriptor theta stepping."""

import numpy as np
import pytest
import scipy.sparse as sp

from solve_nivp import moreau_jean_fremond as mjf


def _stepper(rhs, jac, *, affine=None, lowrank=None, constraints=None):
    return mjf.DescriptorMoreauJeanFremondStepper(
        A=sp.eye(3, format="csr"),
        rhs_callable=rhs,
        rhs_jac_callable=jac,
        rhs_affine_callable=affine,
        D_extract=sp.csr_matrix((0, 3)),
        B=sp.csr_matrix((3, 0)),
        contacts=[],
        constraints=list(constraints or []),
        theta=0.5,
        theta_linear_solver="scipy",
        theta_lowrank_jac=lowrank,
        theta_operator_revision=mjf.THETA_OPERATOR_STATIC,
    )


def _march(stepper, *, steps=4, h=0.05):
    y = np.array([0.2, -0.1, 0.4])
    states = []
    for k in range(steps):
        y, _, _ = stepper.step(k * h, y, {}, h)
        states.append(y.copy())
    return np.asarray(states)


def test_affine_callback_matches_legacy_and_skips_cached_full_rhs_calls():
    J = sp.csr_matrix(
        [[-0.4, 0.2, 0.0], [0.0, -0.3, 0.1], [0.0, 0.0, -0.2]]
    )
    legacy_calls = {"rhs": 0, "jac": 0}
    fast_calls = {"rhs": 0, "jac": 0, "affine": 0}

    def forcing(t):
        return np.array([np.sin(t), 1.0 + t, -0.5 * t])

    def legacy_rhs(t, y):
        legacy_calls["rhs"] += 1
        return np.asarray(J @ y).ravel() + forcing(t)

    def legacy_jac(_t, _y):
        legacy_calls["jac"] += 1
        return J

    def fast_rhs(t, y):
        fast_calls["rhs"] += 1
        return np.asarray(J @ y).ravel() + forcing(t)

    def fast_jac(_t, _y):
        fast_calls["jac"] += 1
        return J

    def affine(t):
        fast_calls["affine"] += 1
        return forcing(t)

    legacy = _stepper(legacy_rhs, legacy_jac)
    fast = _stepper(fast_rhs, fast_jac, affine=affine)

    np.testing.assert_allclose(_march(fast), _march(legacy), rtol=0.0, atol=1e-14)
    assert legacy_calls == {"rhs": 4, "jac": 1}
    assert fast_calls == {"rhs": 1, "jac": 1, "affine": 4}


def test_affine_callback_is_evaluated_at_theta_time():
    seen = []

    def affine(t):
        seen.append(t)
        return np.array([t, 2.0 * t, -t])

    rhs = lambda t, y: affine_value(t)
    affine_value = lambda t: np.array([t, 2.0 * t, -t])
    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    stepper = _stepper(rhs, jac, affine=affine)
    y = np.zeros(3)

    y, _, _ = stepper.step(0.0, y, {}, 0.2)
    stepper.step(0.2, y, {}, 0.2)

    np.testing.assert_allclose(seen, [0.1, 0.3], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize(
    ("affine", "match"),
    [
        (lambda _t: np.zeros(2), "length"),
        (lambda _t: np.array([0.0, np.nan, 0.0]), "finite"),
    ],
)
def test_affine_callback_rejects_invalid_vectors(affine, match):
    rhs = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    stepper = _stepper(rhs, jac, affine=affine)

    with pytest.raises(ValueError, match=match):
        stepper.step(0.0, np.zeros(3), {}, 0.1)


def test_affine_callback_rejects_incorrect_identity_on_cold_build():
    rhs = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    stepper = _stepper(rhs, jac, affine=lambda _t: np.ones(3))

    with pytest.raises(ValueError, match="affine RHS certification failed"):
        stepper.step(0.0, np.zeros(3), {}, 0.1)


@pytest.mark.parametrize(
    ("rhs_value", "match"),
    [
        (np.zeros(2), "rhs_callable returned a vector with invalid length"),
        (np.array([0.0, np.nan, 0.0]), "rhs_callable must return only finite"),
        (np.array([0.0, np.inf, 0.0]), "rhs_callable must return only finite"),
    ],
)
def test_affine_certification_rejects_invalid_full_rhs(rhs_value, match):
    rhs = lambda _t, _y: rhs_value
    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    stepper = _stepper(rhs, jac, affine=lambda _t: np.zeros(3))

    with pytest.raises(ValueError, match=match):
        stepper.step(0.0, np.zeros(3), {}, 0.1)


@pytest.mark.parametrize(
    ("jacobian", "match"),
    [
        (
            sp.csr_matrix((2, 3)),
            "rhs_jac_callable returned a matrix with invalid shape",
        ),
        (
            sp.csr_matrix(([np.nan], ([0], [0])), shape=(3, 3)),
            "rhs_jac_callable must return only finite",
        ),
        (
            sp.csr_matrix(([np.inf], ([0], [0])), shape=(3, 3)),
            "rhs_jac_callable must return only finite",
        ),
    ],
)
def test_affine_certification_rejects_invalid_jacobian(jacobian, match):
    rhs = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: jacobian
    stepper = _stepper(rhs, jac, affine=lambda _t: np.zeros(3))

    with pytest.raises(ValueError, match=match):
        stepper.step(0.0, np.zeros(3), {}, 0.1)


def test_affine_callback_matches_legacy_with_lowrank_jacobian():
    U = np.array([[1.0], [-0.5], [0.25]])
    V = np.array([[0.3], [0.2], [-0.4]])
    forcing = lambda t: np.array([1.0 + t, -0.2 * t, 0.5])

    def rhs(t, y):
        return -(U @ (V.T @ y)).ravel() + forcing(t)

    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    legacy = _stepper(rhs, jac, lowrank=(U, V))
    fast = _stepper(rhs, jac, affine=forcing, lowrank=(U, V))

    np.testing.assert_allclose(_march(fast), _march(legacy), rtol=0.0, atol=1e-13)


def test_lowrank_feedback_respects_replaced_constraint_rows():
    J_sparse = sp.diags([-0.4, -0.3, -0.2], format="csr")
    U = np.array([[0.3], [-0.2], [1.4]])
    V = np.array([[0.5], [-0.4], [0.7]])
    J_full = (J_sparse - sp.csr_matrix(U @ V.T)).tocsr()
    forcing = lambda t: np.array([1.0 + t, -0.2 * t, 0.5])

    def rhs(t, y):
        return np.asarray(J_full @ y).ravel() + forcing(t)

    constraint = {
        "g": lambda y_sub, t, *_: 1.7 * y_sub + np.array([0.2 + t]),
        "dg_dy": lambda _y_sub, _t, *_: np.array([[1.7]]),
        "y_slice": np.array([0]),
        "q_slice": np.array([2]),
    }
    folded = _stepper(
        rhs,
        lambda _t, _y: J_full,
        affine=forcing,
        constraints=[constraint],
    )
    lowrank_legacy = _stepper(
        rhs,
        lambda _t, _y: J_sparse,
        lowrank=(U, V),
        constraints=[constraint],
    )
    lowrank_affine = _stepper(
        rhs,
        lambda _t, _y: J_sparse,
        affine=forcing,
        lowrank=(U, V),
        constraints=[constraint],
    )

    expected = _march(folded)
    np.testing.assert_allclose(
        _march(lowrank_legacy), expected, rtol=0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        _march(lowrank_affine), expected, rtol=0.0, atol=1e-13
    )
