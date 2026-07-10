"""Ratio-mode adaptive MJF driver (Task B).

``solve_mjf_adaptive_ratio`` delegates step-size control to the generic
:class:`~solve_nivp.adaptive_integrator.AdaptiveStepping` in ``mode="ratio"``
(Gustafsson / Soderlind digital filter with the LENIENT ratio-band acceptance,
DAE-aware error weighting, active-set filter), driving the
:class:`DescriptorMoreauJeanFremondStepper` through a thin integrator-contract
adapter.  A HOLD layer above the controller keeps the committed step size fixed
while the controller's proposal drifts inside a band, so the per-h theta
factorization cache (Task A) is reused and the factorization count tracks the
number of *distinct* committed step sizes, not the number of steps.

Test map (per the Task B brief):
  (e)   trajectory vs analytical on the sliding block (mirrors
        test_moreau_jean_fremond::test_solve_mjf_adaptive_sliding_block_*)
  (r1)  ratio-band leniency: a step with E > 1 is accepted (band, not E<=1)
  (r2)  factorizations ~= #distinct committed h  <<  #steps
  (r3)  DAE weighting active: zero-mass (algebraic) rows excluded from the norm
  (r4)  active-set filter suppresses regime-change velocity DOFs
  (hz)  horizon landing is exact, no degenerate micro-step
plus: mu fixed-point (slip-weakening) aux threading, and that the additive
driver leaves ``solve_mjf_adaptive`` untouched.
"""
import numpy as np
import scipy.sparse as sp
import pytest

from solve_nivp.moreau_jean_fremond import (
    DescriptorMoreauJeanFremondStepper,
    solve_mjf_adaptive,
    solve_mjf_adaptive_ratio,
    _MJFRatioAdapter,
    _MJFRegimeProjection,
    _mjf_velocity_dof_map,
)
from solve_nivp.adaptive_integrator import AdaptiveStepping
from solve_nivp.nonlinear_solvers import PETSC_AVAILABLE

pytestmark = pytest.mark.skipif(not PETSC_AVAILABLE, reason="petsc4py required")


# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------
def _sliding_block(mu, F_T, *, mass=1.0, g=9.81, n_slip=0):
    """First-order descriptor sliding block y = [q_n, q_t, v_n, v_t(, s)].

    Copy of ``test_moreau_jean_fremond._descriptor_sliding_block``.  The bulk
    Jacobian is CONSTANT, so the theta operator is byte-identical at a fixed h
    and the Task-A cache applies.
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
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu, "e": 0.0}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
    )


def _linear_oscillator(w2=50.0):
    """Contact-free linear oscillator q'' = -w2 q (constant Jacobian).

    Constant J keeps the theta operator byte-identical per h (cache applies),
    while the genuine curvature gives a nonzero Richardson error so the ratio
    band's leniency is actually exercised.
    """
    n = 2
    A = np.eye(n)

    def rhs(t, y):
        return np.array([y[1], -w2 * y[0]])

    def jac(t, y):
        J = np.zeros((n, n))
        J[0, 1] = 1.0
        J[1, 0] = -w2
        return J

    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=np.zeros((0, n)), B=np.zeros((n, 0)), contacts=[],
        theta=0.5, theta_linear_solver="scipy",
    )


def _dae_plant():
    """Index-1 descriptor DAE with a genuine algebraic row.

    A = diag(1, 1, 0): the 3rd DOF is algebraic (zero mass).  The dynamics
    x' = v, v' = -x, and the algebraic relation w = x are index-1 and the
    theta operator is nonsingular, so the plant integrates and the constraint
    w = x is enforced exactly.
    """
    n = 3
    A = np.diag([1.0, 1.0, 0.0])

    def rhs(t, y):
        return np.array([y[1], -y[0], y[0] - y[2]])

    def jac(t, y):
        J = np.zeros((n, n))
        J[0, 1] = 1.0
        J[1, 0] = -1.0
        J[2, 0] = 1.0
        J[2, 2] = -1.0
        return J

    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=np.zeros((0, n)), B=np.zeros((n, 0)), contacts=[],
        theta=0.5, theta_linear_solver="scipy",
    )


# ---------------------------------------------------------------------------
# (e) trajectory vs analytical
# ---------------------------------------------------------------------------
def test_e_ratio_sliding_block_matches_analytical():
    mass, g, mu = 1.0, 9.81, 0.4
    F_T = 1.5 * mu * mass * g
    stepper = _sliding_block(mu, F_T)
    t_end = 0.05
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, t_end), np.zeros(4), {"mu": np.array([mu])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-9, h_max=t_end,
    )
    assert t[-1] == pytest.approx(t_end)
    np.testing.assert_allclose(np.diff(t), h, atol=1.0e-15)
    accel = (F_T - mu * mass * g) / mass
    assert y[-1, 3] == pytest.approx(accel * t_end, rel=2.0e-2)
    assert y[-1, 2] == pytest.approx(0.0, abs=1.0e-8)
    assert all("adaptive_error" in rec for rec in info)
    assert "p_contact_force" in info[-1]
    # sliding contact rides the cone edge with the mu used by the law
    p_eff = info[-1]["p_contact_effective"]
    assert abs(abs(p_eff[1]) - mu * p_eff[0]) < 1.0e-8


# ---------------------------------------------------------------------------
# (r1) ratio-band leniency: E > 1 accepted
# ---------------------------------------------------------------------------
def test_r1_ratio_band_accepts_error_above_one():
    stepper = _linear_oscillator(w2=50.0)
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 3.0), np.array([1.0, 0.0]), {},
        rtol=3.0e-3, atol=3.0e-6, h0=3.0e-2, h_min=1.0e-9, h_max=0.3,
        record_attempts=True,
    )
    log = attempts["attempt_log"]
    err = np.asarray(log["error"], dtype=float)
    accepted = np.asarray(log["accepted"], dtype=bool)
    finite = np.isfinite(err)
    # Classic E<=1 acceptance could NEVER accept these; the ratio band does.
    assert np.any((err > 1.0) & accepted & finite), (
        "ratio band should accept at least one step with E > 1"
    )
    # And the run is otherwise healthy: it lands the horizon and stays bounded.
    assert t[-1] == pytest.approx(3.0)
    assert np.all(np.isfinite(y))
    # Energy of the linear oscillator stays bounded (no runaway from leniency).
    E0 = 0.5 * y[0, 1] ** 2 + 0.5 * 50.0 * y[0, 0] ** 2
    Ef = 0.5 * y[-1, 1] ** 2 + 0.5 * 50.0 * y[-1, 0] ** 2
    assert abs(Ef - E0) / E0 < 0.2


# ---------------------------------------------------------------------------
# (r2) factorizations ~= #distinct committed h  <<  #steps
# ---------------------------------------------------------------------------
def test_r2_factorizations_track_distinct_committed_h_contact():
    # Long contact march: a short growth phase (a few commits), then a long
    # plateau where the held h is reused -- the factorization count tracks the
    # number of DISTINCT committed step sizes, not the number of steps.
    mass, g, mu = 1.0, 9.81, 0.4
    F_T = 1.5 * mu * mass * g
    stepper = _sliding_block(mu, F_T)
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 5.0), np.zeros(4), {"mu": np.array([mu])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-9, h_max=5.0e-3,
    )
    n_acc = len(h)
    n_fac = attempts["n_factorizations"]
    n_dist = len(attempts["distinct_committed_h"])
    assert n_acc >= 500                                   # a real march
    # each committed h needs op(h) [full step] + op(h/2) [half steps]
    assert n_fac <= 2 * n_dist + 2
    # factorization ECONOMY: far fewer factorizations than steps
    assert n_fac < n_acc // 20
    assert n_fac == stepper._theta_factorizations         # counter is the source


def test_r2b_hold_layer_grows_h_from_too_small_h0():
    # DEAD-BAND regression (review blocking-1): the controller clamps the
    # accepted-step ratio to [r_min, r_max] = [0.8, 1.2], so a single-step
    # commit test against hold_threshold = 0.2 (== the band edge) could never
    # fire and h stayed at h0 forever (fixed-step degeneration).  The hold
    # layer must ACCUMULATE drift across held steps: from a deliberately
    # 30x-too-small h0, h must actually grow, and the total step count must be
    # comparable to the classic driver's.
    h0 = 1.0e-3
    st_ratio = _linear_oscillator(w2=50.0)
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        st_ratio, (0.0, 3.0), np.array([1.0, 0.0]), {},
        rtol=3.0e-3, atol=3.0e-6, h0=h0, h_min=1.0e-9, h_max=0.3,
    )
    assert max(h) >= 10.0 * h0            # h actually grew (dead band would pin it)
    st_classic = _linear_oscillator(w2=50.0)
    t2, y2, h2, info2, a2 = solve_mjf_adaptive(
        st_classic, (0.0, 3.0), np.array([1.0, 0.0]), {},
        rtol=3.0e-3, atol=3.0e-6, h0=h0, h_min=1.0e-9, h_max=0.3,
    )
    assert len(h) <= 2 * len(h2)          # step count comparable to classic


# ---------------------------------------------------------------------------
# (r3) DAE-aware error weighting excludes algebraic (zero-mass) rows
# ---------------------------------------------------------------------------
def test_r3_dae_weighting_excludes_algebraic_rows():
    stepper = _dae_plant()
    adapter = _MJFRatioAdapter(stepper)
    assert adapter.use_identity is False                  # genuine mass matrix
    ctrl = AdaptiveStepping(integrator=adapter, dae_var_weight="auto")
    mask = ctrl._ensure_dae_mask(3)
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0])  # 3rd DOF algebraic
    # And the plant integrates while holding the algebraic constraint w = x.
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 1.0), np.array([1.0, 0.0, 1.0]), {},
        rtol=1.0e-4, atol=1.0e-7, h0=1.0e-2, h_min=1.0e-9, h_max=0.1,
    )
    assert t[-1] == pytest.approx(1.0)
    assert abs(y[-1, 2] - y[-1, 0]) < 1.0e-10             # w = x enforced


def test_r3_identity_mass_gives_all_differential_mask():
    # Control: A = I -> no false algebraic detection -> all-ones weight.
    mu, F_T = 0.4, 1.5 * 0.4 * 9.81
    stepper = _sliding_block(mu, F_T)
    adapter = _MJFRatioAdapter(stepper)
    assert adapter.use_identity is True
    ctrl = AdaptiveStepping(integrator=adapter, dae_var_weight="auto")
    np.testing.assert_array_equal(ctrl._ensure_dae_mask(4), np.ones(4))


# ---------------------------------------------------------------------------
# (r4) active-set filter
# ---------------------------------------------------------------------------
def test_r4a_regime_shim_suppresses_transition_dofs():
    # Mirrors test_active_set_filter's MuScaledSOCProjection mask tests, but on
    # the MJF regime shim that feeds AdaptiveStepping's active-set filter.
    vmap = [np.array([2, 3]), np.array([4, 5])]
    proj = _MJFRegimeProjection(vmap)
    assert proj.regime_snapshot() is None                 # unset before any step
    proj.update_regime(["stick", "slip"])
    snap = proj.regime_snapshot()
    # no transition -> no mask
    assert proj.regime_changed_mask(snap, 6) is None
    # block 0 flips stick -> slip: its velocity DOFs 2,3 suppressed
    proj.update_regime(["slip", "slip"])
    mask = proj.regime_changed_mask(snap, 6)
    assert mask is not None
    np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    # None snapshot -> None
    assert proj.regime_changed_mask(None, 6) is None


def test_r4b_adapter_exposes_projection_to_controller():
    mu, F_T = 0.4, 1.5 * 0.4 * 9.81
    stepper = _sliding_block(mu, F_T)
    adapter = _MJFRatioAdapter(stepper)
    # velocity DOF map points at the contact's state velocity DOFs (v_n, v_t)
    np.testing.assert_array_equal(_mjf_velocity_dof_map(stepper)[0], [2, 3])
    ctrl = AdaptiveStepping(integrator=adapter, active_set_filter=True)
    assert ctrl._get_projection() is adapter._proj


def test_r4b2_velocity_dof_map_interleaved_rows():
    # Review blocking-2: production crack plants interleave the contact
    # velocity rows in D_extract (all normals first, then all tangentials:
    # contact k uses rows [k, n_c + k]), so indexing D_extract by the reaction
    # block-slice positions [2k, 2k+2) mixes DOFs across contacts and the
    # filter silently suppresses the WRONG contact's DOFs.  The map must be
    # built from D_contact, whose rows are already in reaction order.
    #
    # State: [qA_n, qA_t, qB_n, qB_t, vA_n, vA_t, vB_n, vB_t]
    # D_extract rows: [n_A, n_B, t_A, t_B] -> contact A = rows (0, 2),
    # contact B = rows (1, 3).
    n = 8
    D = np.zeros((4, n))
    D[0, 4] = 1.0    # n_A -> vA_n
    D[1, 6] = 1.0    # n_B -> vB_n
    D[2, 5] = 1.0    # t_A -> vA_t
    D[3, 7] = 1.0    # t_B -> vB_t
    B = np.zeros((n, 4))
    B[4, 0] = 1.0; B[5, 1] = 1.0; B[6, 2] = 1.0; B[7, 3] = 1.0

    def rhs(t, y):
        out = np.zeros(n)
        out[0:4] = y[4:8]
        return out

    def jac(t, y):
        J = np.zeros((n, n))
        for i in range(4):
            J[i, 4 + i] = 1.0
        return J

    stepper = DescriptorMoreauJeanFremondStepper(
        A=np.eye(n), rhs_callable=rhs, rhs_jac_callable=jac,
        D_extract=D, B=B,
        contacts=[
            {"vel_normal_idx": 0, "vel_tangential_idx": [2], "mu_init": 0.4, "e": 0.0},
            {"vel_normal_idx": 1, "vel_tangential_idx": [3], "mu_init": 0.4, "e": 0.0},
        ],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
    )
    # sanity: the rows really interleave in D_extract order
    assert stepper.contact_velocity_rows == [0, 2, 1, 3]
    vmap = _mjf_velocity_dof_map(stepper)
    # contact A must map to ITS velocity DOFs (4, 5) -- the old D_extract
    # block-slice indexing gave {4, 6} (A's normal + B's normal: wrong).
    np.testing.assert_array_equal(vmap[0], [4, 5])
    np.testing.assert_array_equal(vmap[1], [6, 7])
    # end-to-end: a regime flip on contact A must suppress ONLY A's DOFs
    proj = _MJFRegimeProjection(vmap)
    proj.update_regime(["stick", "stick"])
    snap = proj.regime_snapshot()
    proj.update_regime(["slip", "stick"])
    mask = proj.regime_changed_mask(snap, n)
    np.testing.assert_array_equal(
        mask, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])


def test_r4c_filter_collapses_transition_error_spike():
    # Order-collapse at a contact transition: raw Richardson pins h because the
    # regime-changing (tangential) velocity DOF shows a huge full-vs-refined
    # discrepancy.  The active-set filter suppresses exactly that DOF, so the
    # estimator no longer overreacts -> the step is accepted and h is NOT pinned.
    mu, F_T = 0.4, 1.5 * 0.4 * 9.81
    stepper = _sliding_block(mu, F_T)
    adapter = _MJFRatioAdapter(stepper)
    ctrl = AdaptiveStepping(
        integrator=adapter, atol=1.0e-6, rtol=1.0e-3, method_order=2, mode="ratio",
    )
    # DOF 3 = tangential velocity (the regime-changing DOF); others clean.
    y = np.array([0.0, 0.0, 0.0, 0.10])
    y_full = np.array([0.0, 0.0, 0.0, 0.10 + 2.0e-3])   # coarse full step
    y_hi = np.array([0.0, 0.0, 0.0, 0.10 + 1.0e-6])     # refined: transition collapse

    ctrl._transition_mask = None
    E_raw = ctrl._scaled_error(y, y_full, y_hi)
    assert E_raw > 1.0                                   # raw estimator REJECTS -> pins h

    m = np.ones(4)
    m[stepper.block_slices[0].start + 0] = 0.0           # (guard) suppress block-0 DOFs
    m[2] = 0.0
    m[3] = 0.0                                            # suppress the transitioning v_t
    ctrl._transition_mask = m
    E_filtered = ctrl._scaled_error(y, y_full, y_hi)
    assert E_filtered <= 1.0                             # filtered estimator ACCEPTS -> h recovers
    assert E_filtered < E_raw                            # spike collapsed


def test_r4d_active_set_filter_end_to_end_contact_march():
    mass, g, mu = 1.0, 9.81, 0.4
    F_T = 1.5 * mu * mass * g
    t_end = 0.05
    common = dict(
        y0=np.zeros(4), aux0={"mu": np.array([mu])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-9, h_max=t_end,
    )
    st_on = _sliding_block(mu, F_T)
    t1, y1, h1, info1, a1 = solve_mjf_adaptive_ratio(
        st_on, (0.0, t_end), active_set_filter=True, **common,
    )
    st_off = _sliding_block(mu, F_T)
    t2, y2, h2, info2, a2 = solve_mjf_adaptive_ratio(
        st_off, (0.0, t_end), active_set_filter=False, **common,
    )
    # Filter on/off both integrate the same physics to the horizon.
    assert t1[-1] == pytest.approx(t_end)
    assert t2[-1] == pytest.approx(t_end)
    assert np.all(np.isfinite(y1)) and np.all(np.isfinite(y2))
    accel = (F_T - mu * mass * g) / mass
    assert y1[-1, 3] == pytest.approx(accel * t_end, rel=2.0e-2)
    np.testing.assert_allclose(y1[-1], y2[-1], atol=1.0e-6)


# ---------------------------------------------------------------------------
# (hz) horizon landing exact, no micro-step (also covers error_mask +
#      contact_offset_force through the ratio driver)
# ---------------------------------------------------------------------------
def test_hz_ratio_lands_horizon_exactly_no_microstep():
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
    T, Y, h, info, recs = solve_mjf_adaptive_ratio(
        st, (0.0, tmax), np.zeros(2), aux0={"mu": np.array([mu])},
        error_mask=np.array([False, True]), rtol=1e-4,
        atol=1e-6 * np.ones(2), h0=1e-2, h_max=0.2,
    )
    T = np.asarray(T)
    assert T[-1] == tmax                                   # exact landing
    assert np.all(np.isfinite(Y))
    assert np.diff(T).min() > 1.0e-9                       # no micro-step


# ---------------------------------------------------------------------------
# mu fixed point (slip-weakening): exercises aux threading through the 3
# Richardson sub-steps AND the outer mu iteration.
# ---------------------------------------------------------------------------
def test_mu_fixed_point_slip_weakening_ratio():
    mass, g = 1.0, 9.81
    mu_s, mu_d, D_c = 0.6, 0.2, 1.0e-3
    F_T = 1.2 * mu_s * mass * g
    stepper = _sliding_block(mu_s, F_T, n_slip=1)

    def mu_from_state(y_state):
        frac = min(float(y_state[4]) / D_c, 1.0)
        return np.array([mu_s - (mu_s - mu_d) * frac])

    t_end = 0.08
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, t_end), np.zeros(5), {"mu": np.array([mu_s])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-10, h_max=5.0e-3,
        mu_from_state=mu_from_state, mu_fixed_point_tol=1.0e-10,
    )
    assert t[-1] == pytest.approx(t_end)
    assert y[-1, 4] > D_c                                  # fully weakened
    assert info[-1]["mu_law"][0] == pytest.approx(mu_d)
    assert max(rec["mu_fixed_point_iters"] for rec in info) >= 1
    assert max(rec["mu_fixed_point_error"] for rec in info) <= 1.0e-9 + 1.0e-12
    p_eff = info[-1]["p_contact_effective"]
    mu_law = info[-1]["mu_law"][0]
    assert abs(abs(p_eff[1]) - mu_law * p_eff[0]) < 1.0e-8
    # aux threading did not wreck factorization economy: still one
    # factorization pair (op(h), op(h/2)) per distinct committed h
    n_dist = len(attempts["distinct_committed_h"])
    assert attempts["n_factorizations"] <= 2 * n_dist + 2


# ---------------------------------------------------------------------------
# slip-weakening causal-DRIVER route: mu MUST evolve (a frozen-mu bug passes
# every trajectory smoke test otherwise).  On a plant whose slip row is
# identity/rhs=0 (the causal build), a mu_from_state(y) reading the state slip
# row FREEZES mu; the explicit-slip route (s += h|v_t|, written back) evolves it.
# ---------------------------------------------------------------------------
def _frozen_slip_block(mu_s, F_T, *, mass=1.0, g=9.81):
    """Sliding block whose slip row (idx 4) has rhs=0: the state slip row does
    NOT self-integrate, so only an explicit-slip writeback can advance it."""
    n = 5
    A = np.eye(n)

    def rhs(t, y):
        out = np.zeros(n)
        out[0] = y[2]
        out[1] = y[3]
        out[2] = -mass * g
        out[3] = F_T
        # out[4] left 0 -> slip row frozen unless written back explicitly
        return out

    def rhs_jac(t, y):
        J = np.zeros((n, n))
        J[0, 2] = 1.0
        J[1, 3] = 1.0
        return J

    D = np.zeros((2, n)); D[0, 2] = 1.0; D[1, 3] = 1.0
    B = np.zeros((n, 2)); B[2, 0] = 1.0; B[3, 1] = 1.0
    return DescriptorMoreauJeanFremondStepper(
        A=A, rhs_callable=rhs, rhs_jac_callable=rhs_jac, D_extract=D, B=B,
        contacts=[{"vel_normal_idx": 0, "vel_tangential_idx": [1],
                   "mu_init": mu_s, "e": 0.0}],
        theta=0.5, contact_solver="petsc_ssn", theta_linear_solver="scipy",
    )


def test_slip_weakening_route_actually_evolves_mu():
    mass, g = 1.0, 9.81
    mu_s, mu_d, D_c = 0.6, 0.2, 1.0e-3
    F_T = 1.2 * mu_s * mass * g

    def mu_from_slip(s):
        frac = np.minimum(np.asarray(s, dtype=float) / D_c, 1.0)
        return mu_s - (mu_s - mu_d) * frac

    slip_weakening = dict(
        slip_slice=slice(4, 5),
        mu_from_slip=mu_from_slip,
        vel_t_extract=np.array([[0.0, 0.0, 0.0, 1.0]]),   # v_t = y[3]
        n_phys=4,
    )
    err_mask = np.array([True, True, True, True, False])   # exclude the slip row

    # explicit-slip route: mu must strictly DECREASE and slip must accumulate.
    stepper = _frozen_slip_block(mu_s, F_T)
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.08), np.zeros(5), {"mu": np.array([mu_s])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-10, h_max=5.0e-3,
        slip_weakening=slip_weakening, error_mask=err_mask,
    )
    assert y[-1, 4] > D_c                                  # slip accumulated
    assert info[-1]["mu_law"][0] < mu_s - 1.0e-6           # mu DECREASED (not frozen)
    assert info[-1]["mu_law"][0] == pytest.approx(mu_d)    # fully weakened

    # contrast (documents the hazard): the naive mu_from_state route on the SAME
    # rhs=0 slip plant FREEZES mu -- passes trajectory smoke tests but is wrong.
    frozen = _frozen_slip_block(mu_s, F_T)
    t2, y2, h2, info2, a2 = solve_mjf_adaptive_ratio(
        frozen, (0.0, 0.08), np.zeros(5), {"mu": np.array([mu_s])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-10, h_max=5.0e-3,
        mu_from_state=lambda ys: mu_from_slip(ys[4:5]), error_mask=err_mask,
    )
    assert y2[-1, 4] == pytest.approx(0.0, abs=1.0e-14)    # slip row stayed frozen
    assert info2[-1]["mu_law"][0] == pytest.approx(mu_s)   # mu FROZEN -> the bug


def test_slip_weakening_trajectory_pins_production_fixed_route():
    # Production-route regression pin (review request 4): force the adaptive
    # driver onto a fixed h (h0 = h_max = h, lenient tolerance -> every attempt
    # accepted), so each accepted step is exactly two h/2 theta-steps; the
    # trajectory must then reproduce solve_mjf_fixed_step (the production
    # MJFIntegrationMethod route) marched at h/2, state for state, on the
    # slip-weakening plant: states, slip, and mu to ~1e-10.
    from solve_nivp.mjf_integration import solve_mjf_fixed_step

    mass, g = 1.0, 9.81
    mu_s, mu_d, D_c = 0.6, 0.2, 1.0e-3
    F_T = 1.2 * mu_s * mass * g
    h = 2.0e-3
    t_end = 0.04

    def mu_from_slip(s):
        frac = np.minimum(np.asarray(s, dtype=float) / D_c, 1.0)
        return mu_s - (mu_s - mu_d) * frac

    vel_t_extract = np.array([[0.0, 0.0, 0.0, 1.0]])       # v_t = y[3]
    sw = dict(slip_slice=slice(4, 5), mu_from_slip=mu_from_slip,
              vel_t_extract=vel_t_extract, n_phys=4)

    st_ad = _frozen_slip_block(mu_s, F_T)
    t_ad, y_ad, h_ad, info_ad, a_ad = solve_mjf_adaptive_ratio(
        st_ad, (0.0, t_end), np.zeros(5), {"mu": np.array([mu_s])},
        rtol=1.0e-4, atol=1.0e-8, h0=h, h_min=1.0e-10, h_max=h,
        slip_weakening=sw,
        error_mask=np.array([True, True, True, True, False]),
    )
    assert len(h_ad) == 20                                  # no rejects: exact pairing
    np.testing.assert_allclose(h_ad, h, rtol=1.0e-12)

    st_fx = _frozen_slip_block(mu_s, F_T)
    (t_fx, y_fx, h_fx, _fk, info_fx, att_fx), _rh = solve_mjf_fixed_step(
        st_fx, np.zeros(5), {"mu": np.array([mu_s])}, t_end, 0.5 * h,
        n_c=1, reaction_scale=1.0,
        slip_slice=slice(4, 5), mu_from_slip=mu_from_slip,
        vel_t_extract=vel_t_extract, n_phys=4,
        mu_fp_tol=1.0e-10, mu_fp_max_iter=30, verbose=False,
    )
    assert len(t_fx) == 41                                  # 40 half steps

    # adaptive accepted state k == fixed-route state 2k (two h/2 sub-steps)
    np.testing.assert_allclose(y_ad, y_fx[::2], rtol=0.0, atol=1.0e-10)
    # slip and mu explicitly
    np.testing.assert_allclose(y_ad[:, 4], y_fx[::2, 4], rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        mu_from_slip(y_ad[:, 4]), mu_from_slip(y_fx[::2, 4]),
        rtol=0.0, atol=1.0e-10,
    )


def test_slip_weakening_and_mu_from_state_are_mutually_exclusive():
    stepper = _sliding_block(0.4, 1.5 * 0.4 * 9.81, n_slip=1)
    with pytest.raises(ValueError):
        _MJFRatioAdapter(
            stepper, mu_from_state=lambda y: np.array([0.4]),
            slip_weakening=dict(slip_slice=slice(4, 5), mu_from_slip=lambda s: s,
                                vel_t_extract=np.zeros((1, 4)), n_phys=4),
        )


# ---------------------------------------------------------------------------
# additive: the new driver leaves solve_mjf_adaptive's defaults untouched
# ---------------------------------------------------------------------------
def test_solve_mjf_adaptive_classic_still_matches():
    mass, g, mu = 1.0, 9.81, 0.4
    F_T = 1.5 * mu * mass * g
    stepper = _sliding_block(mu, F_T)
    t_end = 0.05
    t, y, h, info, attempts = solve_mjf_adaptive(
        stepper, (0.0, t_end), np.zeros(4), {"mu": np.array([mu])},
        rtol=1.0e-4, atol=1.0e-8, h0=1.0e-3, h_min=1.0e-9, h_max=t_end,
    )
    accel = (F_T - mu * mass * g) / mass
    assert t[-1] == pytest.approx(t_end)
    assert y[-1, 3] == pytest.approx(accel * t_end, rel=2.0e-2)
