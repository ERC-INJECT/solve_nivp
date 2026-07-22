"""Run-control features of the MJF drivers: progress, checkpointing, memory.

Covers the shared machinery in ``solve_nivp.mjf_run_control`` and its wiring
into all three drivers:

* checkpoint I/O   -- atomic write, round-trip, schema guard, scalar aux
* MJFRunMonitor    -- heartbeat throttling, step/wall checkpoint triggers,
                      validation of the option combinations
* fixed-step       -- on_step callback, checkpoint cadence, exact resume,
                      thinning alignment (MJFContactView contract preserved)
* adaptive         -- on_step, thinning does not change the march, resume
* adaptive-ratio   -- same, against the linear-oscillator plant

The driver-level fixed-step tests use a deterministic linear-decay fake
stepper (same pattern as ``test_mjf_integration_fixes``), so resume equality
can be asserted exactly; the adaptive tests use a real
``DescriptorMoreauJeanFremondStepper`` on a contact-free linear oscillator
(same plant as ``test_mjf_adaptive_ratio``) and compare against the analytic
solution.
"""
import os

import numpy as np
import pytest

from solve_nivp.mjf_integration import (
    MJFContactView,
    MJFIntegrationMethod,
    solve_mjf_fixed_step,
)
from solve_nivp.mjf_run_control import (
    MJFRunMonitor,
    load_mjf_checkpoint,
    save_mjf_checkpoint,
)
from solve_nivp.moreau_jean_fremond import (
    DescriptorMoreauJeanFremondStepper,
    solve_mjf_adaptive,
    solve_mjf_adaptive_ratio,
)


# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------
class _LinearDecayStepper:
    """Deterministic theta-step for y' = -lam*y, with MJF-shaped aux/info.

    ``aux`` carries a per-step counter and a warm-start mirror so checkpoint
    round-trips of the auxiliary state are observable.  The map is a pure
    function of (y, h), so identical (t, y, aux) inputs reproduce identical
    outputs bitwise -- which is what makes exact resume assertions possible.
    """

    def __init__(self, lam=1.0, n_react=2, theta=0.5):
        self.lam = float(lam)
        self.theta = float(theta)
        self.n_react = int(n_react)

    def step(self, t, y, aux, h):
        y = np.asarray(y, dtype=float)
        y1 = y * (1.0 - h * (1.0 - self.theta) * self.lam) / (1.0 + h * self.theta * self.lam)
        p_contact = np.full(self.n_react, h * (1.0 + abs(float(y[0]))))
        aux1 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in aux.items()}
        aux1["counter"] = int(aux.get("counter", 0)) + 1
        aux1["warm"] = p_contact.copy()
        info = {
            "p_contact": p_contact,
            "soccp_converged": True,
            "soccp_residual": 1.0e-12,
            "soccp_outer_iters": 2,
        }
        return y1, aux1, info


def _linear_oscillator(w2=50.0):
    """Contact-free linear oscillator q'' = -w2 q (constant Jacobian)."""
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


def _oscillator_exact(t, w2=50.0, y0=(1.0, 0.0)):
    w = np.sqrt(w2)
    q0, v0 = y0
    return np.array([q0 * np.cos(w * t) + (v0 / w) * np.sin(w * t),
                     -q0 * w * np.sin(w * t) + v0 * np.cos(w * t)])


def _run_fixed(tmax, h, *, y0=None, aux0=None, **kwargs):
    stepper = _LinearDecayStepper()
    y0 = np.array([1.0, -2.0]) if y0 is None else y0
    aux0 = {"mu": np.array([0.5]), "counter": 0} if aux0 is None else aux0
    return solve_mjf_fixed_step(
        stepper, y0, aux0, tmax, h, 1, 1.0, verbose=False, **kwargs)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "ck.npz"
    aux = {"mu": np.array([0.4, 0.6]), "counter": 7, "flag": True}
    extras = {"reaction_row": np.array([1.0, 2.0])}
    out = save_mjf_checkpoint(path, t=1.5, y=np.array([3.0, 4.0]), aux=aux,
                              h=0.01, step_index=42, driver="fixed",
                              label="TEST", extras=extras)
    ck = load_mjf_checkpoint(out)
    assert ck["t"] == 1.5 and ck["h"] == 0.01 and ck["step_index"] == 42
    assert ck["driver"] == "fixed" and ck["label"] == "TEST"
    np.testing.assert_array_equal(ck["y"], [3.0, 4.0])
    np.testing.assert_array_equal(ck["aux"]["mu"], [0.4, 0.6])
    assert ck["aux"]["counter"] == 7 and ck["aux"]["flag"] is True
    np.testing.assert_array_equal(ck["extras"]["reaction_row"], [1.0, 2.0])


def test_checkpoint_write_is_atomic_and_overwrites(tmp_path):
    path = str(tmp_path / "ck")          # suffix added automatically
    p1 = save_mjf_checkpoint(path, t=0.0, y=np.zeros(3), aux={}, h=0.1,
                             step_index=0)
    p2 = save_mjf_checkpoint(path, t=1.0, y=np.ones(3), aux={}, h=0.1,
                             step_index=5)
    assert p1 == p2 and p1.endswith(".npz")
    assert not os.path.exists(p1 + ".tmp")
    assert load_mjf_checkpoint(p1)["t"] == 1.0


def test_checkpoint_rejects_non_numeric_aux(tmp_path):
    with pytest.raises(TypeError, match="aux\\['bad'\\]"):
        save_mjf_checkpoint(tmp_path / "ck.npz", t=0.0, y=np.zeros(1),
                            aux={"bad": "a string"}, h=0.1, step_index=0)


def test_checkpoint_rejects_foreign_npz(tmp_path):
    path = tmp_path / "foreign.npz"
    np.savez(path, a=np.zeros(3))
    with pytest.raises(ValueError, match="not an MJF checkpoint"):
        load_mjf_checkpoint(path)


# ---------------------------------------------------------------------------
# MJFRunMonitor
# ---------------------------------------------------------------------------
def test_monitor_rejects_triggers_without_path():
    with pytest.raises(ValueError, match="require"):
        MJFRunMonitor(t_start=0.0, t_end=1.0, checkpoint_every_steps=5)


@pytest.mark.parametrize("kwargs", [
    {"progress_interval_s": 0.0},
    {"checkpoint_path": "x.npz", "checkpoint_every_steps": 0},
    {"checkpoint_path": "x.npz", "checkpoint_every_walltime_s": -1.0},
])
def test_monitor_rejects_bad_intervals(kwargs):
    with pytest.raises(ValueError):
        MJFRunMonitor(t_start=0.0, t_end=1.0, **kwargs)


def test_monitor_step_trigger_cadence(tmp_path):
    path = tmp_path / "ck.npz"
    mon = MJFRunMonitor(t_start=0.0, t_end=1.0, checkpoint_path=path,
                        checkpoint_every_steps=3)
    y = np.zeros(2)
    for k in range(1, 8):
        mon.after_step(0.1 * k, y, {"counter": k}, {}, h=0.1)
    # steps 3 and 6 fire; 7 does not.
    assert mon.n_checkpoints_written == 2
    assert load_mjf_checkpoint(path)["aux"]["counter"] == 6


def test_monitor_walltime_trigger(tmp_path):
    path = tmp_path / "ck.npz"
    mon = MJFRunMonitor(t_start=0.0, t_end=1.0, checkpoint_path=path,
                        checkpoint_every_walltime_s=1000.0)
    mon.after_step(0.1, np.zeros(1), {}, {})
    assert mon.n_checkpoints_written == 0
    mon._wall_last_ckpt -= 2000.0        # pretend 2000 s passed
    mon.after_step(0.2, np.zeros(1), {}, {})
    assert mon.n_checkpoints_written == 1


def test_monitor_heartbeat_throttle_and_final_line(capsys):
    mon = MJFRunMonitor(t_start=0.0, t_end=1.0, label="BEAT",
                        progress_interval_s=1.0e9)
    for k in range(1, 4):
        mon.after_step(0.1 * k, np.zeros(1), {}, {}, h=0.1)
    assert capsys.readouterr().out == ""      # throttled: nothing yet
    mon.finish(0.3)
    out = capsys.readouterr().out
    assert out.count("BEAT: step 3") == 1 and "done" in out


def test_monitor_heartbeat_reports_progress(capsys):
    mon = MJFRunMonitor(t_start=0.0, t_end=2.0, label="RUN",
                        progress_interval_s=1.0e-9)
    mon.after_step(0.5, np.zeros(1), {}, {}, h=0.5)
    out = capsys.readouterr().out
    assert "RUN: step 1" in out and "25.0%" in out and "h=5.000e-01" in out


# ---------------------------------------------------------------------------
# Fixed-step driver
# ---------------------------------------------------------------------------
def test_fixed_defaults_unchanged_structure():
    (t, y, h, fk, info, attempts), rhist = _run_fixed(1.0, 0.1)
    assert len(t) == 11 and y.shape == (11, 2) and fk is None
    assert len(info) == 10 and len(h) == 10
    assert attempts["accepted"] == [True] * 10
    assert len(attempts["records"]) == 10
    assert attempts["n_accepted"] == 10 and attempts["n_rejected"] == 0
    assert rhist.shape == (11, 2)


def test_fixed_on_step_called_every_step():
    seen = []
    (t, _y, _h, _fk, _info, _at), _r = _run_fixed(
        1.0, 0.1, on_step=lambda tk, yk, aux, info: seen.append((tk, aux["counter"])))
    assert len(seen) == 10
    assert [c for _, c in seen] == list(range(1, 11))
    np.testing.assert_allclose([tk for tk, _ in seen], t[1:], rtol=0, atol=1e-12)


def test_fixed_checkpoint_cadence_and_final(tmp_path):
    path = tmp_path / "run.npz"
    _run_fixed(1.0, 0.1, checkpoint_path=path, checkpoint_every_steps=4)
    ck = load_mjf_checkpoint(path)
    # periodic writes at steps 4 and 8, then the final checkpoint at step 10.
    assert ck["step_index"] == 10 and ck["t"] == pytest.approx(1.0)
    assert ck["aux"]["counter"] == 10
    assert "reaction_row" in ck["extras"]


def test_fixed_resume_is_exact(tmp_path):
    path = tmp_path / "seg.npz"
    (t_full, y_full, _h, _fk, _info, _at), r_full = _run_fixed(1.0, 0.1)
    _run_fixed(0.5, 0.1, checkpoint_path=path)          # final ckpt at t=0.5
    (t2, y2, _h2, _fk2, _info2, at2), r2 = _run_fixed(
        1.0, 0.1, resume_from=path)
    assert t2[0] == pytest.approx(0.5) and t2[-1] == pytest.approx(1.0)
    # identical arithmetic from the identical restart datum: exact equality.
    np.testing.assert_array_equal(y2[-1], y_full[-1])
    np.testing.assert_array_equal(r2[-1], r_full[-1])
    assert at2["n_accepted"] == 5
    # a checkpoint already at the horizon yields the trivial segment.
    _run_fixed(1.0, 0.1, checkpoint_path=path)
    (t3, y3, h3, _fk3, info3, at3), _r3 = _run_fixed(1.0, 0.1, resume_from=path)
    assert len(t3) == 1 and len(info3) == 0 and at3["n_accepted"] == 0
    np.testing.assert_array_equal(y3[0], y_full[-1])


def test_fixed_resume_accepts_loaded_dict(tmp_path):
    path = tmp_path / "seg.npz"
    _run_fixed(0.5, 0.1, checkpoint_path=path)
    ck = load_mjf_checkpoint(path)
    (t2, _y2, _h2, _fk2, _info2, _at2), _r2 = _run_fixed(1.0, 0.1, resume_from=ck)
    assert t2[0] == pytest.approx(0.5)


def test_fixed_thinning_alignment_and_exactness():
    (t1, y1, _h1, _fk1, _info1, _at1), r1 = _run_fixed(1.0, 0.1)
    (t3, y3, h3, _fk3, info3, at3), r3 = _run_fixed(1.0, 0.1, thin_output=3)
    # stored: initial + steps 3, 6, 9 + final step 10.
    np.testing.assert_allclose(t3, [0.0, 0.3, 0.6, 0.9, 1.0], atol=1e-12)
    assert y3.shape[0] == len(t3) and r3.shape[0] == len(t3)
    assert len(info3) == len(t3) - 1 and len(h3) == len(t3) - 1
    # thinning only subsamples storage; the march itself is unchanged.
    np.testing.assert_array_equal(y3[-1], y1[-1])
    np.testing.assert_array_equal(r3[-1], r1[-1])
    np.testing.assert_array_equal(y3[1], y1[3])
    # true step count is reported even though storage is thinned.
    assert at3["n_accepted"] == 10 and len(at3["accepted"]) == 10
    assert all(i["fixed_h"] == pytest.approx(0.1) for i in info3)
    # the MJFContactView alignment contract survives thinning.
    view = MJFContactView(None, y3[0], None, None, 1.0)
    view._reaction_history = r3
    assert view.reaction_history(y3).shape[0] == y3.shape[0]


def test_fixed_thinning_boundary_multiple():
    # 10 steps with thin=5: final step is ON the grid; no duplicate row.
    (t5, y5, _h5, _fk5, _info5, _at5), r5 = _run_fixed(1.0, 0.1, thin_output=5)
    np.testing.assert_allclose(t5, [0.0, 0.5, 1.0], atol=1e-12)
    assert r5.shape[0] == 3


def test_fixed_gc_interval_smoke():
    (t, _y, _h, _fk, _info, _at), _r = _run_fixed(1.0, 0.1, gc_interval=2)
    assert len(t) == 11


def test_fixed_progress_heartbeat_prints(capsys):
    _run_fixed(1.0, 0.1, progress_interval_s=1.0e-9, on_step=None)
    out = capsys.readouterr().out
    assert "MJF: step 1" in out and "MJF: step 10" in out and "done" in out


def test_fixed_failed_step_not_in_history():
    class _FailAt(_LinearDecayStepper):
        def __init__(self, fail_at):
            super().__init__()
            self._fail_at = fail_at
            self._n = 0

        def step(self, t, y, aux, h):
            y1, aux1, info = super().step(t, y, aux, h)
            self._n += 1
            if self._n == self._fail_at:
                info["soccp_converged"] = False
            return y1, aux1, info

    stepper = _FailAt(4)
    (t, y, _h, _fk, info, attempts), rhist = solve_mjf_fixed_step(
        stepper, np.array([1.0, -2.0]), {"counter": 0}, 1.0, 0.1, 1, 1.0,
        verbose=False)
    # 3 successful steps stored, the failed 4th aborts the march but must not
    # desynchronise the reaction history from the stored states.
    assert len(t) == 4 and rhist.shape[0] == 4 and len(info) == 3
    assert attempts["n_accepted"] == 3


# ---------------------------------------------------------------------------
# Adaptive (step-doubling) driver
# ---------------------------------------------------------------------------
def test_adaptive_on_step_and_counters():
    stepper = _linear_oscillator()
    seen = []
    t, y, h, info, attempts = solve_mjf_adaptive(
        stepper, (0.0, 0.5), np.array([1.0, 0.0]),
        rtol=1e-4, atol=1e-7, h0=1e-2,
        on_step=lambda tk, yk, aux, ii: seen.append(tk))
    assert len(seen) == attempts["n_accepted"] == len(t) - 1
    assert attempts["n_rejected"] == sum(
        1 for a in attempts["accepted"] if not a)
    np.testing.assert_allclose(y[-1], _oscillator_exact(t[-1]), rtol=5e-3)


def test_adaptive_thinning_does_not_change_march():
    stepper = _linear_oscillator()
    t1, y1, _h1, _i1, a1 = solve_mjf_adaptive(
        stepper, (0.0, 0.5), np.array([1.0, 0.0]),
        rtol=1e-4, atol=1e-7, h0=1e-2)
    t3, y3, h3, i3, a3 = solve_mjf_adaptive(
        stepper, (0.0, 0.5), np.array([1.0, 0.0]),
        rtol=1e-4, atol=1e-7, h0=1e-2, thin_output=3)
    assert a3["n_accepted"] == a1["n_accepted"]
    assert len(t3) < len(t1)
    np.testing.assert_array_equal(y3[-1], y1[-1])
    assert t3[-1] == t1[-1]
    assert len(t3) == len(y3) == len(h3) + 1 == len(i3) + 1
    # pin the off-grid-final reconcile branch: if a controller change makes
    # n_accepted a multiple of 3, pick a different thin_output here.
    assert a3["n_accepted"] % 3 != 0
    assert len(a3["records"]) == len(t3) - 1


def test_adaptive_checkpoint_resume(tmp_path):
    path = tmp_path / "ad.npz"
    stepper = _linear_oscillator()
    solve_mjf_adaptive(stepper, (0.0, 0.25), np.array([1.0, 0.0]),
                       rtol=1e-4, atol=1e-7, h0=1e-2, checkpoint_path=path)
    ck = load_mjf_checkpoint(path)
    assert ck["driver"] == "adaptive" and 0.0 < ck["t"] <= 0.25 + 1e-12
    t2, y2, _h2, _i2, a2 = solve_mjf_adaptive(
        stepper, (0.0, 0.5), np.array([999.0, 999.0]),   # ignored on resume
        rtol=1e-4, atol=1e-7, resume_from=path)
    assert t2[0] == pytest.approx(ck["t"]) and t2[-1] == pytest.approx(0.5)
    np.testing.assert_allclose(y2[-1], _oscillator_exact(0.5), rtol=5e-3)


# ---------------------------------------------------------------------------
# Adaptive-ratio driver
# ---------------------------------------------------------------------------
def test_ratio_on_step_thinning_and_resume(tmp_path):
    path = tmp_path / "ratio.npz"
    stepper = _linear_oscillator()
    seen = []
    t1, y1, _h1, _i1, a1 = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.5), np.array([1.0, 0.0]),
        rtol=1e-4, atol=1e-7, h0=1e-2,
        on_step=lambda tk, yk, aux, ii: seen.append(tk),
        checkpoint_path=path)
    assert len(seen) == a1["n_accepted"] == len(t1) - 1
    ck = load_mjf_checkpoint(path)
    assert ck["driver"] == "ratio"
    assert ck["extras"]["h_held"] > 0.0
    assert ck["t"] == pytest.approx(t1[-1])

    t3, y3, _h3, _i3, a3 = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.5), np.array([1.0, 0.0]),
        rtol=1e-4, atol=1e-7, h0=1e-2, thin_output=4)
    assert a3["n_accepted"] == a1["n_accepted"]
    assert len(t3) < len(t1)
    np.testing.assert_array_equal(y3[-1], y1[-1])
    # pin the off-grid-final reconcile branch (see adaptive twin test).
    assert a3["n_accepted"] % 4 != 0
    assert len(a3["records"]) == len(t3) - 1

    # resume from the mid-horizon checkpoint of a shorter segment
    path2 = tmp_path / "ratio_seg.npz"
    solve_mjf_adaptive_ratio(stepper, (0.0, 0.25), np.array([1.0, 0.0]),
                             rtol=1e-4, atol=1e-7, h0=1e-2,
                             checkpoint_path=path2)
    t2, y2, _h2, _i2, _a2 = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.5), np.array([0.0, 0.0]),      # ignored on resume
        rtol=1e-4, atol=1e-7, resume_from=path2)
    assert t2[-1] == pytest.approx(0.5)
    np.testing.assert_allclose(y2[-1], _oscillator_exact(0.5), rtol=5e-3)


def test_ratio_resume_at_horizon_is_trivial(tmp_path):
    path = tmp_path / "done.npz"
    stepper = _linear_oscillator()
    solve_mjf_adaptive_ratio(stepper, (0.0, 0.5), np.array([1.0, 0.0]),
                             rtol=1e-4, atol=1e-7, h0=1e-2,
                             checkpoint_path=path)
    t, y, h, info, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.5), np.array([0.0, 0.0]),
        rtol=1e-4, atol=1e-7, resume_from=path)
    assert len(t) == 1 and len(h) == 0 and len(info) == 0
    assert attempts["n_accepted"] == 0


# ---------------------------------------------------------------------------
# Abort semantics: triggers, propagation, edge shapes
# ---------------------------------------------------------------------------
class _FailAtStepper(_LinearDecayStepper):
    """Linear-decay stepper whose N-th step reports a failed contact solve."""

    def __init__(self, fail_at, key="soccp_converged"):
        super().__init__()
        self._fail_at = int(fail_at)
        self._key = key
        self._n = 0

    def step(self, t, y, aux, h):
        y1, aux1, info = super().step(t, y, aux, h)
        self._n += 1
        if self._n == self._fail_at:
            info[self._key] = False
            if self._key == "mu_fp_converged":
                info["mu_fp_resid"] = 3.7e-2
        return y1, aux1, info


def test_fixed_fresh_bad_horizon_still_raises():
    # R1: a fresh call with a non-increasing horizon must fail loudly, not
    # return a trivial segment (the trivial path is resume-only).
    with pytest.raises(ValueError, match="t_span must be increasing"):
        _run_fixed(0.0, 0.1)
    with pytest.raises(ValueError, match="t_span must be increasing"):
        _run_fixed(-0.5, 0.1)


def test_fixed_abort_checkpoint_holds_march_front(tmp_path):
    # Failure abort with thinning: the returned trajectory ends at the last
    # STORED state, but the final checkpoint must hold the last SUCCESSFUL
    # state (march front) with its matching aux -- never the failed attempt's
    # aux, and never a stored state paired with newer aux.
    path = tmp_path / "abort.npz"
    stepper = _FailAtStepper(4)
    (t, y, _h, _fk, info, attempts), rhist = solve_mjf_fixed_step(
        stepper, np.array([1.0, -2.0]), {"counter": 0}, 1.0, 0.1, 1, 1.0,
        verbose=False, thin_output=2, checkpoint_path=path)
    # stored grid: initial + step 2 (step 3 is off-grid and unstored on abort)
    np.testing.assert_allclose(t, [0.0, 0.2], atol=1e-12)
    assert rhist.shape[0] == len(t)
    assert attempts["failure"] is not None
    assert attempts["failure"]["soccp_converged"] is False
    ck = load_mjf_checkpoint(path)
    assert ck["t"] == pytest.approx(0.3)          # march front, not t[-1]
    assert ck["aux"]["counter"] == 3              # last successful step's aux
    # and the checkpoint restarts cleanly
    (t2, _y2, _h2, _fk2, _info2, at2), _r2 = _run_fixed(
        1.0, 0.1, resume_from=path)
    assert t2[0] == pytest.approx(0.3) and at2["failure"] is None


def test_fixed_abort_verbose_reports_failure(capsys):
    stepper = _FailAtStepper(4, key="mu_fp_converged")
    (_t, _y, _h, _fk, _info, attempts), _r = solve_mjf_fixed_step(
        stepper, np.array([1.0, -2.0]), {"counter": 0}, 1.0, 0.1, 1, 1.0,
        verbose=True)
    out = capsys.readouterr().out
    assert "ABORTED at step 4" in out and "mu fixed-point not converged" in out
    assert attempts["failure"]["mu_fp_converged"] is False
    assert attempts["failure"]["mu_fp_resid"] == pytest.approx(3.7e-2)


def test_fixed_abort_heartbeat_says_aborted(capsys):
    stepper = _FailAtStepper(3)
    solve_mjf_fixed_step(
        stepper, np.array([1.0, -2.0]), {"counter": 0}, 1.0, 0.1, 1, 1.0,
        verbose=False, progress_interval_s=1e9)
    out = capsys.readouterr().out
    assert "aborted" in out and "done" not in out


def test_fixed_trivial_resume_writes_final_checkpoint(tmp_path):
    seg_a = tmp_path / "a.npz"
    seg_b = tmp_path / "b.npz"
    _run_fixed(0.5, 0.1, checkpoint_path=seg_a)
    _run_fixed(0.5, 0.1, resume_from=seg_a, checkpoint_path=seg_b)
    ck = load_mjf_checkpoint(seg_b)   # promised final checkpoint exists
    assert ck["t"] == pytest.approx(0.5)


def test_fixed_partial_final_step_fixed_h_under_thinning():
    # tmax not a multiple of h: the clipped final step must report its actual
    # size, and the private _step_h marker must not leak into returned info.
    (_t, _y, _h, _fk, info, _at), _r = _run_fixed(1.05, 0.1, thin_output=3)
    assert info[-1]["fixed_h"] == pytest.approx(0.05)
    assert info[0]["fixed_h"] == pytest.approx(0.1)
    assert all("_step_h" not in i for i in info)


def test_walltime_trigger_fires_through_driver(monkeypatch, tmp_path):
    from solve_nivp import mjf_run_control as mrc
    calls = []
    real_save = mrc.save_mjf_checkpoint
    monkeypatch.setattr(
        mrc, "save_mjf_checkpoint",
        lambda *a, **k: (calls.append(k.get("t")), real_save(*a, **k))[1])
    _run_fixed(1.0, 0.1, checkpoint_path=tmp_path / "w.npz",
               checkpoint_every_walltime_s=1e-12)
    # every step's elapsed wall time exceeds 1e-12 s: 10 periodic + 1 final.
    assert len(calls) == 11


def test_on_step_exception_propagates():
    def boom(t, y, aux, info):
        raise RuntimeError("user callback boom")

    with pytest.raises(RuntimeError, match="user callback boom"):
        _run_fixed(1.0, 0.1, on_step=boom)


def test_gc_interval_collects_on_all_drivers(monkeypatch):
    import gc as gc_mod
    counts = {"n": 0}
    real_collect = gc_mod.collect
    monkeypatch.setattr(gc_mod, "collect",
                        lambda *a, **k: (counts.__setitem__("n", counts["n"] + 1),
                                         real_collect(*a, **k))[1])
    _run_fixed(1.0, 0.1, gc_interval=2)
    n_fixed = counts["n"]
    assert n_fixed >= 5
    stepper = _linear_oscillator()
    solve_mjf_adaptive(stepper, (0.0, 0.2), np.array([1.0, 0.0]),
                       rtol=1e-4, atol=1e-7, h0=1e-2, gc_interval=3)
    assert counts["n"] > n_fixed
    n_ad = counts["n"]
    solve_mjf_adaptive_ratio(stepper, (0.0, 0.2), np.array([1.0, 0.0]),
                             rtol=1e-4, atol=1e-7, h0=1e-2, gc_interval=3)
    assert counts["n"] > n_ad


def test_checkpoint_scalar_float_aux_roundtrip(tmp_path):
    # the production stepper always writes aux['p_contact_prev_h'] = float(h)
    path = save_mjf_checkpoint(tmp_path / "s.npz", t=0.0, y=np.zeros(1),
                               aux={"p_contact_prev_h": 0.05}, h=0.05,
                               step_index=1)
    restored = load_mjf_checkpoint(path)["aux"]["p_contact_prev_h"]
    assert isinstance(restored, float) and restored == 0.05


def test_adaptive_resume_reads_extras_h_next():
    stepper = _linear_oscillator()
    ck = {"t": 0.2, "y": _oscillator_exact(0.2), "aux": {},
          "h": 999.0, "step_index": 3, "extras": {"h_next": 0.01}}
    _t, _y, _h, _i, attempts = solve_mjf_adaptive(
        stepper, (0.0, 0.5), np.zeros(2), rtol=1e-4, atol=1e-7,
        resume_from=ck)
    # first attempt must start at extras['h_next'], not the bogus ckpt['h'].
    assert attempts["records"][0]["h"] == pytest.approx(0.01)


def test_ratio_trivial_resume_keeps_attempt_log_key(tmp_path):
    path = tmp_path / "done.npz"
    stepper = _linear_oscillator()
    solve_mjf_adaptive_ratio(stepper, (0.0, 0.3), np.array([1.0, 0.0]),
                             rtol=1e-4, atol=1e-7, h0=1e-2,
                             checkpoint_path=path)
    _t, _y, _h, _i, attempts = solve_mjf_adaptive_ratio(
        stepper, (0.0, 0.3), np.zeros(2), rtol=1e-4, atol=1e-7,
        resume_from=path, record_attempts=True)
    assert "attempt_log" in attempts and attempts["thin_output"] == 1


# ---------------------------------------------------------------------------
# MJFIntegrationMethod unit-level (thinning counter parity with ODESolver)
# ---------------------------------------------------------------------------
def test_method_thinning_counter():
    stepper = _LinearDecayStepper()
    mjf = MJFIntegrationMethod(stepper, aux0={"counter": 0}, n_c=1,
                               reaction_scale=1.0, thin_output=2)
    y = np.array([1.0, 0.0])
    for k in range(5):
        y, _fk, _e, ok, _it = mjf.step(None, 0.1 * k, y, 0.1)
        assert ok
    # kept: steps 2 and 4 (initial row is the warm start) + last buffered.
    assert len(mjf.reaction_history) == 3
    assert mjf.steps_taken == 5
    assert mjf._last_entry[0] == 5
