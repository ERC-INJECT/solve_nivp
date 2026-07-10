"""Multi-level theta-factorization cache (Task A).

The theta operator op(h) = (1/theta)A - hJ is affine and byte-identical at a
fixed h, so each distinct h needs exactly one factorization.  The stepper keeps
an LRU-bounded ``OrderedDict`` keyed by ``float(h)`` so that h-revisits under an
adaptive driver are free.  ``_theta_factorizations`` is a permanent counter that
increments once per genuine (re)factorization; ``_theta_cache_evictions`` counts
LRU evictions.

Fixtures mirror ``test_theta_lowrank_petsc.py`` (MUMPS options, small sparse
systems) and the stepper-construction pattern of
``test_moreau_jean_fremond.py::test_descriptor_theta_cache_consistent_across_mu_and_h``.
"""
import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.moreau_jean_fremond import (
    DescriptorMoreauJeanFremondStepper,
    PETSC_AVAILABLE,
)

pytestmark = pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py required")

MUMPS_OPTS = {"ksp_type": "preonly", "pc_type": "lu",
              "pc_factor_mat_solver_type": "mumps"}


def _sliding_block(mu, F_T, *, theta_cache_size=4, mass=1.0, g=9.81, n_slip=0):
    """First-order descriptor sliding block with a configurable cache depth.

    Copy of ``test_moreau_jean_fremond._descriptor_sliding_block`` plus the
    ``theta_cache_size`` kwarg under test.
    """
    n = 4 + n_slip
    A = np.eye(n)

    def rhs(t, y):
        out = np.zeros(n)
        out[0] = y[2]
        out[1] = y[3]
        out[2] = -mass * g
        out[3] = F_T
        if n_slip:
            out[4] = abs(y[3])
        return out

    def rhs_jac(t, y):
        J = np.zeros((n, n))
        J[0, 2] = 1.0
        J[1, 3] = 1.0
        if n_slip:
            J[4, 3] = 1.0 if y[3] >= 0.0 else -1.0
        return J

    D = np.zeros((2, n))
    D[0, 2] = 1.0
    D[1, 3] = 1.0
    B = np.zeros((n, 2))
    B[2, 0] = 1.0
    B[3, 1] = 1.0
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac,
        D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu, "e": 0.0}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
        theta_cache_size=theta_cache_size,
    )


def _mutable_jac_stepper(state, *, n=4, theta=0.5, theta_cache_size=4):
    """Contact-free descriptor stepper whose Jacobian reads ``state['J']``.

    Mutating ``state['J']`` between steps changes the theta operator so the
    per-entry byte-identity guard can be exercised at a fixed h.
    """
    A = np.eye(n)

    def rhs(t, y):
        return np.asarray(state["J"] @ np.asarray(y).ravel()).ravel() + state["b"]

    def jac(t, y):
        return state["J"]

    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=np.zeros((0, n)), B=np.zeros((n, 0)), contacts=[],
        theta=theta, theta_linear_solver="scipy",
        theta_cache_size=theta_cache_size,
    )


def _lowrank_stepper(J_use, lowrank, *, n, theta, J_folded, b, theta_cache_size):
    """Contact-free MUMPS stepper (mirror of the folded/low-rank pair in
    ``test_theta_lowrank_petsc.py``) with a configurable cache depth."""
    A = sp.eye(n, format="csr")

    def rhs(t, y):
        return np.asarray(J_folded @ np.asarray(y).ravel()).ravel() + b

    def jac(t, y):
        return J_use

    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=np.zeros((0, n)), B=np.zeros((n, 0)), contacts=[],
        theta=theta,
        theta_linear_solver="petsc", theta_petsc_options=MUMPS_OPTS,
        theta_lowrank_jac=lowrank,
        theta_cache_size=theta_cache_size,
    )


# ---------------------------------------------------------------------------
# (a) same-(t, y, h) re-entry -> counter increments exactly once
# ---------------------------------------------------------------------------
def test_same_tyh_reentry_factorizes_once():
    F_T = 1.5 * 0.4 * 1.0 * 9.81
    y0 = np.zeros(4)
    h = 0.01
    s = _sliding_block(0.4, F_T, theta_cache_size=4)
    assert s._theta_factorizations == 0
    s.step(0.0, y0, {"mu": np.array([0.4])}, h)
    assert s._theta_factorizations == 1
    # aux fixed-point re-entry: identical (t, y, h), only mu differs -> exact hit
    s.step(0.0, y0, {"mu": np.array([0.05])}, h)
    assert s._theta_factorizations == 1
    assert len(s._theta_cache) == 1


# ---------------------------------------------------------------------------
# (b) sequence h, h/2, h, h/2 with cache_size>=2 -> counter == 2; per-step
#     y / p_contact byte-match a fresh (cacheless) stepper
# ---------------------------------------------------------------------------
def test_h_revisit_is_free_multilevel():
    F_T = 1.5 * 0.4 * 1.0 * 9.81
    y0 = np.zeros(4)
    mu = np.array([0.4])
    h = 0.01
    shared = _sliding_block(0.4, F_T, theta_cache_size=2)
    y = y0.copy()
    t = 0.0
    for hk in (h, 0.5 * h, h, 0.5 * h):
        fresh = _sliding_block(0.4, F_T, theta_cache_size=1)
        y_f, _, info_f = fresh.step(t, y, {"mu": mu}, hk)
        y_s, _, info_s = shared.step(t, y, {"mu": mu}, hk)
        np.testing.assert_allclose(y_s, y_f, atol=1.0e-12)
        np.testing.assert_allclose(info_s["p_contact"], info_f["p_contact"],
                                   atol=1.0e-12)
        y = y_s
        t = t + hk
    # only h and h/2 were ever factorized; the two revisits reused their rungs
    assert shared._theta_factorizations == 2
    assert len(shared._theta_cache) == 2


# ---------------------------------------------------------------------------
# (c) cache_size=2, 3 distinct h -> len==2, oldest absent, evictions==1,
#     evicted fac.destroy() called (spy)
# ---------------------------------------------------------------------------
def test_lru_evicts_oldest_and_destroys():
    F_T = 1.5 * 0.4 * 1.0 * 9.81
    y0 = np.zeros(4)
    mu = np.array([0.4])
    h1, h2, h3 = 0.01, 0.02, 0.03
    shared = _sliding_block(0.4, F_T, theta_cache_size=2)

    shared.step(0.0, y0, {"mu": mu}, h1)
    fac1 = shared._theta_cache[float(h1)]["fac"]
    calls = {"n": 0}
    orig_destroy = fac1.destroy

    def spy(*a, **k):
        calls["n"] += 1
        return orig_destroy(*a, **k)

    fac1.destroy = spy

    shared.step(0.0, y0, {"mu": mu}, h2)
    assert len(shared._theta_cache) == 2
    assert shared._theta_cache_evictions == 0

    shared.step(0.0, y0, {"mu": mu}, h3)
    assert len(shared._theta_cache) == 2
    assert float(h1) not in shared._theta_cache
    assert float(h2) in shared._theta_cache
    assert float(h3) in shared._theta_cache
    assert shared._theta_cache_evictions == 1
    assert shared._theta_factorizations == 3
    assert calls["n"] == 1     # evicted rung's factorization was destroyed


# ---------------------------------------------------------------------------
# (e) LRU recency-on-hit: a cache HIT must move its rung to most-recently-used
#     so a later insertion evicts the *other* (now-oldest) rung, not the one
#     just revisited.  cap=2: h1, h2, revisit h1 (exact re-entry hit), h3 ->
#     h2 evicted, h1 retained.  This pins the move_to_end() on the hit paths,
#     which the adaptive ratio driver relies on when it re-holds an earlier h.
# ---------------------------------------------------------------------------
def test_lru_recency_on_hit_retains_revisited_rung():
    F_T = 1.5 * 0.4 * 1.0 * 9.81
    y0 = np.zeros(4)
    mu = np.array([0.4])
    h1, h2, h3 = 0.01, 0.02, 0.03
    s = _sliding_block(0.4, F_T, theta_cache_size=2)

    s.step(0.0, y0, {"mu": mu}, h1)          # factorize h1
    s.step(0.0, y0, {"mu": mu}, h2)          # factorize h2; cache = {h1(old), h2(mru)}
    assert s._theta_factorizations == 2
    assert s._theta_cache_evictions == 0

    # revisit h1 with identical (t, y): exact (t, y, h) re-entry -> cache HIT,
    # no refactorization, and h1 must be promoted to most-recently-used.
    s.step(0.0, y0, {"mu": mu}, h1)
    assert s._theta_factorizations == 2      # hit: no new factorization
    assert list(s._theta_cache.keys()) == [float(h2), float(h1)]  # h1 now MRU

    # insert a third distinct h: the oldest rung (h2) is evicted, not h1.
    s.step(0.0, y0, {"mu": mu}, h3)
    assert s._theta_factorizations == 3
    assert s._theta_cache_evictions == 1
    assert float(h1) in s._theta_cache       # retained by the recency bump
    assert float(h3) in s._theta_cache
    assert float(h2) not in s._theta_cache   # evicted as least-recently-used


# ---------------------------------------------------------------------------
# (d) same h, genuine J change -> byte-identity guard refactorizes; result
#     matches a fresh stepper.  Control: same op / different y reuses.
# ---------------------------------------------------------------------------
def test_genuine_jac_change_at_same_h_refactorizes():
    n = 4
    b = np.zeros(n)
    state = {"J": -np.eye(n) - 0.1, "b": b}
    s = _mutable_jac_stepper(state, n=n, theta_cache_size=4)
    y0 = np.ones(n)
    h = 0.05

    s.step(0.0, y0, {}, h)
    assert s._theta_factorizations == 1

    # control: same operator, different y at the same h -> byte-identity reuse
    s.step(0.0, y0 * 2.0, {}, h)
    assert s._theta_factorizations == 1

    # genuine J change at the same h -> guard forces a rebuild
    state["J"] = -np.eye(n) - 0.2
    y3, _, _ = s.step(0.0, y0 * 3.0, {}, h)
    assert s._theta_factorizations == 2

    fresh = _mutable_jac_stepper({"J": state["J"], "b": b}, n=n,
                                 theta_cache_size=4)
    y_fresh, _, _ = fresh.step(0.0, y0 * 3.0, {}, h)
    np.testing.assert_allclose(y3, y_fresh, atol=1.0e-12)


# ---------------------------------------------------------------------------
# (f) Woodbury + multilevel: alternating h, h/2 matches the folded-Jacobian
#     reference trajectory AND the counter stays flat on revisits.
# ---------------------------------------------------------------------------
def test_woodbury_multilevel_trajectory_and_free_revisit():
    rng = np.random.default_rng(3)
    n, k, h, theta = 40, 2, 0.05, 0.5
    J_sparse = (sp.random(n, n, density=0.1, random_state=3, format="csr")
                - sp.eye(n) * 2.0).tocsr()
    U = rng.standard_normal((n, k)) * 0.3
    V = rng.standard_normal((n, k)) * 0.3
    J_folded = (J_sparse - sp.csr_matrix(U @ V.T)).tocsr()
    b = rng.standard_normal(n)

    s_ref = _lowrank_stepper(J_folded, None, n=n, theta=theta,
                             J_folded=J_folded, b=b, theta_cache_size=4)
    s_lr = _lowrank_stepper(J_sparse, (U, V), n=n, theta=theta,
                            J_folded=J_folded, b=b, theta_cache_size=4)

    y_ref = np.ones(n)
    y_lr = np.ones(n)
    aux_ref, aux_lr = {}, {}
    h_seq = [h, 0.5 * h, h, 0.5 * h, h, 0.5 * h]
    for i, hk in enumerate(h_seq):
        t = i * h
        y_ref, aux_ref, _ = s_ref.step(t, y_ref, aux_ref, hk)
        y_lr, aux_lr, _ = s_lr.step(t, y_lr, aux_lr, hk)

    np.testing.assert_allclose(y_lr, y_ref, rtol=1.0e-11, atol=1.0e-12)
    # only two rungs (h and h/2) factorized across six steps
    assert s_lr._theta_factorizations == 2
    assert len(s_lr._theta_cache) == 2


# ---------------------------------------------------------------------------
# (bc) theta_cache_size=1 reproduces today's one-deep behavior
#      (rerun of test_descriptor_theta_cache_consistent_across_mu_and_h)
# ---------------------------------------------------------------------------
def test_cache_size_one_reproduces_one_deep():
    mass, g = 1.0, 9.81
    F_T = 1.5 * 0.4 * mass * g
    y0 = np.zeros(4)
    h = 0.01
    shared = _sliding_block(0.4, F_T, theta_cache_size=1)

    cases = [
        ({"mu": np.array([0.4])}, h),
        ({"mu": np.array([0.05])}, h),
        ({"mu": np.array([0.05])}, 0.5 * h),
    ]
    for aux, h_k in cases:
        y_s, _, info_s = shared.step(0.0, y0, dict(aux), h_k)
        fresh = _sliding_block(0.4, F_T, theta_cache_size=1)
        y_f, _, info_f = fresh.step(0.0, y0, dict(aux), h_k)
        np.testing.assert_allclose(y_s, y_f, atol=1.0e-12)
        np.testing.assert_allclose(info_s["p_contact"], info_f["p_contact"],
                                   atol=1.0e-12)
        # one-deep: the cache never holds more than a single rung
        assert len(shared._theta_cache) == 1

    y_a, _, info_a = shared.step(0.0, y0, {"mu": np.array([0.4])}, h)
    y_b, _, info_b = shared.step(0.0, y0, {"mu": np.array([0.05])}, h)
    assert not np.allclose(y_a, y_b)
    assert abs(info_a["p_contact"][1]) > abs(info_b["p_contact"][1])
