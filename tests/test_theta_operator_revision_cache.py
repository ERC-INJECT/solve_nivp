"""Revision-aware theta-operator cache tests.

The explicit revision contract permits a solver to skip sparse operator
reconstruction while still rebuilding all time/state-dependent RHS terms.
"""

import numpy as np
import pytest
import scipy.sparse as sp

from solve_nivp import moreau_jean_fremond as mjf


def _contact_free_stepper(
    *,
    rhs,
    jac,
    revision=None,
    cache_size=4,
    constraints=None,
    lowrank=None,
):
    n = 3
    kwargs = {}
    if revision is not None:
        kwargs["theta_operator_revision"] = revision
    return mjf.DescriptorMoreauJeanFremondStepper(
        A=sp.eye(n, format="csr"),
        rhs_callable=rhs,
        rhs_jac_callable=jac,
        D_extract=sp.csr_matrix((0, n)),
        B=sp.csr_matrix((n, 0)),
        contacts=[],
        constraints=list(constraints or []),
        theta=0.5,
        theta_linear_solver="scipy",
        theta_lowrank_jac=lowrank,
        theta_cache_size=cache_size,
        **kwargs,
    )


def test_static_revision_sentinel_is_accepted():
    zero = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))

    stepper = _contact_free_stepper(
        rhs=zero,
        jac=jac,
        revision=mjf.THETA_OPERATOR_STATIC,
    )

    assert stepper.theta_operator_revision is mjf.THETA_OPERATOR_STATIC


def test_noncallable_revision_provider_is_rejected():
    zero = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))

    with pytest.raises(
        TypeError,
        match="theta_operator_revision must be None, THETA_OPERATOR_STATIC",
    ):
        _contact_free_stepper(rhs=zero, jac=jac, revision="static")


def _affine_callbacks(counter):
    J = sp.csr_matrix(
        [
            [-0.4, 0.2, 0.0],
            [0.0, -0.3, 0.1],
            [0.0, 0.0, -0.2],
        ]
    )

    def rhs(t, y):
        counter["rhs"] += 1
        forcing = np.array([np.sin(t), 1.0 + t, -0.5 * t])
        return np.asarray(J @ y).ravel() + forcing

    def jac(_t, _y):
        counter["jac"] += 1
        return J

    return rhs, jac


def _march(stepper, *, count=3, h=0.05):
    y = np.array([0.2, -0.1, 0.4])
    states = []
    for k in range(count):
        y, _, _ = stepper.step(k * h, y, {}, h)
        states.append(y.copy())
    return np.asarray(states)


def test_static_revision_rebuilds_rhs_but_not_operator():
    conservative_calls = {"rhs": 0, "jac": 0}
    static_calls = {"rhs": 0, "jac": 0}
    rhs_c, jac_c = _affine_callbacks(conservative_calls)
    rhs_s, jac_s = _affine_callbacks(static_calls)
    conservative = _contact_free_stepper(rhs=rhs_c, jac=jac_c)
    static = _contact_free_stepper(
        rhs=rhs_s,
        jac=jac_s,
        revision=mjf.THETA_OPERATOR_STATIC,
    )

    np.testing.assert_allclose(
        _march(static),
        _march(conservative),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert static_calls == {"rhs": 3, "jac": 1}
    assert conservative_calls == {"rhs": 3, "jac": 3}
    assert static._theta_operator_builds == 1
    assert static._theta_rhs_only_builds == 2
    assert static._theta_factorizations == 1
    entry = static._theta_cache[(0.05, mjf.THETA_OPERATOR_STATIC)]
    assert "operator_template" in entry
    assert "op_data" not in entry


def _affine_constraint(counter):
    def g(y_sub, t, *_):
        counter["g"] += 1
        return 2.0 * y_sub + np.array([t])

    def dg(_y_sub, _t, *_):
        counter["dg"] += 1
        return np.array([[2.0]])

    return {
        "g": g,
        "dg_dy": dg,
        "y_slice": np.array([0]),
        "q_slice": np.array([2]),
    }


def test_static_revision_recomputes_constraint_values_only():
    conservative_rhs_calls = {"rhs": 0, "jac": 0}
    static_rhs_calls = {"rhs": 0, "jac": 0}
    conservative_constraint_calls = {"g": 0, "dg": 0}
    static_constraint_calls = {"g": 0, "dg": 0}
    rhs_c, jac_c = _affine_callbacks(conservative_rhs_calls)
    rhs_s, jac_s = _affine_callbacks(static_rhs_calls)
    conservative = _contact_free_stepper(
        rhs=rhs_c,
        jac=jac_c,
        constraints=[_affine_constraint(conservative_constraint_calls)],
    )
    static = _contact_free_stepper(
        rhs=rhs_s,
        jac=jac_s,
        revision=mjf.THETA_OPERATOR_STATIC,
        constraints=[_affine_constraint(static_constraint_calls)],
    )

    np.testing.assert_allclose(
        _march(static),
        _march(conservative),
        rtol=0.0,
        atol=1.0e-14,
    )
    assert static_constraint_calls == {"g": 3, "dg": 1}
    assert conservative_constraint_calls == {"g": 3, "dg": 3}


def _mutable_callbacks(state, counter):
    def rhs(t, y):
        counter["rhs"] += 1
        forcing = np.array([1.0 + t, -0.25, 0.5])
        return np.asarray(state["J"] @ y).ravel() + forcing

    def jac(_t, _y):
        counter["jac"] += 1
        return state["J"]

    def revision(_t, _y):
        counter["revision"] += 1
        return state["revision"]

    return rhs, jac, revision


def _fresh_mutable_step(state):
    calls = {"rhs": 0, "jac": 0, "revision": 0}
    rhs, jac, _revision = _mutable_callbacks(state, calls)
    return _contact_free_stepper(rhs=rhs, jac=jac)


def test_dynamic_revision_invalidates_exact_reentry_and_reuses_old_revision():
    J0 = sp.diags([-0.2, -0.3, -0.4], format="csr")
    J1 = sp.csr_matrix(
        [
            [-0.5, 0.1, 0.0],
            [0.0, -0.6, 0.2],
            [0.0, 0.0, -0.7],
        ]
    )
    state = {"revision": 0, "J": J0}
    calls = {"rhs": 0, "jac": 0, "revision": 0}
    rhs, jac, revision = _mutable_callbacks(state, calls)
    stepper = _contact_free_stepper(
        rhs=rhs,
        jac=jac,
        revision=revision,
        cache_size=3,
    )
    y0 = np.array([0.3, -0.2, 0.1])
    h = 0.04

    y_rev0, _, _ = stepper.step(0.0, y0, {}, h)
    y_ref0, _, _ = _fresh_mutable_step(state).step(0.0, y0, {}, h)
    np.testing.assert_allclose(y_rev0, y_ref0, rtol=0.0, atol=1.0e-14)

    state.update(revision=1, J=J1)
    y_rev1, _, _ = stepper.step(0.0, y0, {}, h)
    y_ref1, _, _ = _fresh_mutable_step(state).step(0.0, y0, {}, h)
    np.testing.assert_allclose(y_rev1, y_ref1, rtol=0.0, atol=1.0e-14)

    state.update(revision=0, J=J0)
    y_rev0_again, _, _ = stepper.step(0.1, y0, {}, h)
    y_ref0_again, _, _ = _fresh_mutable_step(state).step(0.1, y0, {}, h)
    np.testing.assert_allclose(
        y_rev0_again,
        y_ref0_again,
        rtol=0.0,
        atol=1.0e-14,
    )

    assert calls == {"rhs": 3, "jac": 2, "revision": 3}
    assert stepper._theta_factorizations == 2
    assert stepper._theta_operator_builds == 2
    assert stepper._theta_rhs_only_builds == 1
    assert (h, 0) in stepper._theta_cache
    assert (h, 1) in stepper._theta_cache


def test_unhashable_revision_is_rejected_before_cache_mutation():
    zero = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))
    stepper = _contact_free_stepper(
        rhs=zero,
        jac=jac,
        revision=lambda _t, _y: [],
    )

    with pytest.raises(TypeError, match="must return a hashable token"):
        stepper.step(0.0, np.zeros(3), {}, 0.1)
    assert len(stepper._theta_cache) == 0


def test_revision_callback_error_propagates_before_cache_mutation():
    zero = lambda _t, y: np.zeros_like(y)
    jac = lambda _t, _y: sp.csr_matrix((3, 3))

    def broken_revision(_t, _y):
        raise RuntimeError("revision failed")

    stepper = _contact_free_stepper(
        rhs=zero,
        jac=jac,
        revision=broken_revision,
    )

    with pytest.raises(RuntimeError, match="revision failed"):
        stepper.step(0.0, np.zeros(3), {}, 0.1)
    assert len(stepper._theta_cache) == 0


def test_revision_lru_evicts_and_destroys_oldest_factorization():
    state = {
        "revision": 0,
        "J": sp.diags([-0.2, -0.3, -0.4], format="csr"),
    }
    calls = {"rhs": 0, "jac": 0, "revision": 0}
    rhs, jac, revision = _mutable_callbacks(state, calls)
    stepper = _contact_free_stepper(
        rhs=rhs,
        jac=jac,
        revision=revision,
        cache_size=2,
    )
    y0 = np.array([0.2, 0.1, -0.1])
    h = 0.03

    stepper.step(0.0, y0, {}, h)
    first = stepper._theta_cache[(h, 0)]
    destroyed = {"count": 0}
    original_destroy = first["fac"].destroy

    def destroy_spy(*args, **kwargs):
        destroyed["count"] += 1
        return original_destroy(*args, **kwargs)

    first["fac"].destroy = destroy_spy
    for revision_value in (1, 2):
        state["revision"] = revision_value
        state["J"] = sp.diags(
            [
                -0.2 - 0.1 * revision_value,
                -0.3 - 0.1 * revision_value,
                -0.4 - 0.1 * revision_value,
            ],
            format="csr",
        )
        stepper.step(revision_value * h, y0, {}, h)

    assert destroyed["count"] == 1
    assert stepper._theta_cache_evictions == 1
    assert stepper._theta_factorizations == 3
    assert (h, 0) not in stepper._theta_cache
    assert list(stepper._theta_cache) == [(h, 1), (h, 2)]


def test_static_lowrank_revision_matches_folded_jacobian():
    J_sparse = sp.diags([-0.5, -0.4, -0.3], format="csr")
    U = np.array([[0.2], [-0.1], [0.3]])
    V = np.array([[0.4], [0.2], [-0.2]])
    J_full = (J_sparse - sp.csr_matrix(U @ V.T)).tocsr()
    lowrank_calls = {"rhs": 0, "jac": 0}
    folded_calls = {"rhs": 0, "jac": 0}

    def make_callbacks(counter, jacobian):
        def rhs(t, y):
            counter["rhs"] += 1
            return np.asarray(J_full @ y).ravel() + np.array([t, 0.1, -0.2])

        def jac(_t, _y):
            counter["jac"] += 1
            return jacobian

        return rhs, jac

    rhs_lr, jac_lr = make_callbacks(lowrank_calls, J_sparse)
    rhs_folded, jac_folded = make_callbacks(folded_calls, J_full)
    lowrank = _contact_free_stepper(
        rhs=rhs_lr,
        jac=jac_lr,
        revision=mjf.THETA_OPERATOR_STATIC,
        lowrank=(U, V),
    )
    folded = _contact_free_stepper(
        rhs=rhs_folded,
        jac=jac_folded,
        revision=mjf.THETA_OPERATOR_STATIC,
    )

    np.testing.assert_allclose(
        _march(lowrank, count=5),
        _march(folded, count=5),
        rtol=1.0e-12,
        atol=1.0e-13,
    )
    assert lowrank_calls == {"rhs": 5, "jac": 1}
    assert folded_calls == {"rhs": 5, "jac": 1}
    assert lowrank._theta_factorizations == 1


def test_static_revision_keeps_distinct_step_size_rungs():
    calls = {"rhs": 0, "jac": 0}
    rhs, jac = _affine_callbacks(calls)
    stepper = _contact_free_stepper(
        rhs=rhs,
        jac=jac,
        revision=mjf.THETA_OPERATOR_STATIC,
        cache_size=2,
    )
    y0 = np.array([0.2, -0.1, 0.4])

    stepper.step(0.0, y0, {}, 0.05)
    stepper.step(0.05, y0, {}, 0.025)
    stepper.step(0.075, y0, {}, 0.05)

    assert calls == {"rhs": 3, "jac": 2}
    assert stepper._theta_factorizations == 2
    assert stepper._theta_operator_builds == 2
    assert stepper._theta_rhs_only_builds == 1
    assert list(stepper._theta_cache) == [
        (0.025, mjf.THETA_OPERATOR_STATIC),
        (0.05, mjf.THETA_OPERATOR_STATIC),
    ]


def test_failed_new_revision_preserves_other_cached_revisions(monkeypatch):
    J0 = sp.diags([-0.2, -0.3, -0.4], format="csr")
    J1 = sp.diags([-0.5, -0.6, -0.7], format="csr")
    state = {"revision": 0, "J": J0}
    calls = {"rhs": 0, "jac": 0, "revision": 0}
    rhs, jac, revision = _mutable_callbacks(state, calls)
    stepper = _contact_free_stepper(
        rhs=rhs,
        jac=jac,
        revision=revision,
    )
    y0 = np.array([0.1, -0.2, 0.3])
    h = 0.04
    stepper.step(0.0, y0, {}, h)
    old_entry = stepper._theta_cache[(h, 0)]

    state.update(revision=1, J=J1)
    with monkeypatch.context() as patch:
        def fail_factorization(*_args, **_kwargs):
            raise RuntimeError("factorization failed")

        patch.setattr(mjf, "_ThetaFactorization", fail_factorization)
        with pytest.raises(RuntimeError, match="factorization failed"):
            stepper.step(0.1, y0, {}, h)

    assert stepper._theta_cache[(h, 0)] is old_entry
    assert (h, 1) not in stepper._theta_cache

    state.update(revision=0, J=J0)
    y_cached, _, _ = stepper.step(0.2, y0, {}, h)
    y_fresh, _, _ = _fresh_mutable_step(state).step(0.2, y0, {}, h)
    np.testing.assert_allclose(y_cached, y_fresh, rtol=0.0, atol=1.0e-14)


def test_static_revision_rejects_constraint_count_mutation():
    calls = {"rhs": 0, "jac": 0}
    rhs, jac = _affine_callbacks(calls)
    stepper = _contact_free_stepper(
        rhs=rhs,
        jac=jac,
        revision=mjf.THETA_OPERATOR_STATIC,
    )
    y0 = np.array([0.2, -0.1, 0.4])
    stepper.step(0.0, y0, {}, 0.05)
    stepper.constraints.append(_affine_constraint({"g": 0, "dg": 0}))

    with pytest.raises(
        RuntimeError,
        match="constraint count changed without a theta operator revision",
    ):
        stepper.step(0.05, y0, {}, 0.05)
