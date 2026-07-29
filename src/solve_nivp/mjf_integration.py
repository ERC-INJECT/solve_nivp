"""MJF as a first-class integrator under the solve_nivp framework.

The Descriptor-Moreau-Jean-Fremond stepper is wrapped as an
:class:`~solve_nivp.integrations.IntegrationMethod` so the generic
:class:`~solve_nivp.ODESolver` drives it -- fixed-step is simply
``ODESolver(adaptive=False)``.  This replaces the bespoke fixed-step loop
(formerly the notebook-local ``mjf_driver.run_mjf_fixed_step``) with the
package's own driver, while the contact solve (petsc_ssn / MUMPS) inside the
stepper is left untouched.

Public API
----------
MJFIntegrationMethod  : the integrator adapter
solve_mjf_fixed_step  : thin ODESolver-based driver returning the same
                        ``(result, reaction_history)`` tuple the old helper did
MJFContactView        : reaction-history view for existing post-processing
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np

from .integrations import IntegrationMethod
from .mjf_run_control import MJFRunMonitor, resolve_mjf_checkpoint


class MJFContactView:
    """Stand-in for the Radau projected-contact object in post-processing.

    MJF keeps reactions in per-step ``info``; the driver collects them and this
    view replays them through the ``reaction_history`` interface used by the
    notebooks' ``_contact_history``.
    """

    def __init__(self, A, y0, component_slices, reaction_state_indices,
                 reaction_state_to_reported_scale):
        self.A = A
        self.y0 = y0
        self.component_slices = component_slices
        self.reaction_state_indices = reaction_state_indices
        self.reaction_state_to_reported_scale = reaction_state_to_reported_scale
        self.projected_radau_contact = None
        self._reaction_history = None

    def reaction_history(self, y_hist):
        hist = np.asarray(self._reaction_history, dtype=float)
        y = np.asarray(y_hist, dtype=float)
        if y.ndim == 1:
            return hist[-1]
        if hist.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"MJF reaction history has {hist.shape[0]} rows; "
                f"expected {y.shape[0]}")
        return hist


def _copy_aux(aux: dict) -> dict:
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in aux.items()}


class MJFIntegrationMethod(IntegrationMethod):
    """Adapt a ``DescriptorMoreauJeanFremondStepper`` to the integrator API.

    ``step(fun, t, y, h)`` ignores ``fun`` (MJF carries its own descriptor
    dynamics), advances one fixed theta-step, carries the MJF ``aux`` state
    internally, and accumulates per-step reactions in ``reaction_history``
    (initial warm reaction + one per step).  Two friction modes:

    * constant Coulomb -- leave ``slip_slice=None``
    * slip-weakening   -- pass ``slip_slice``/``mu_from_slip``/``vel_t_extract``
      /``n_phys`` and the contact law uses ``mu(y_{k+theta})`` via an outer
      fixed point each step.

    History storage and thinning
    ----------------------------
    ``reaction_history`` / ``info_history`` receive entries only for
    *successful* steps (a failed fixed step is not committed by the driver, so
    storing it would misalign the histories against the stored states; its
    diagnostics land in ``failure_info`` instead).  When ``thin_output > 1``
    only every N-th successful step is stored, matching the identical
    predicate in :class:`~solve_nivp.ODESolver`, and the most recent step is
    additionally buffered in ``_last_entry`` so the driver can align the
    histories with the always-stored final state.
    """

    def __init__(self, stepper, aux0: dict, *, n_c: int, reaction_scale: float,
                 warm_r: Optional[np.ndarray] = None,
                 slip_slice: Optional[slice] = None,
                 mu_from_slip: Optional[Callable] = None,
                 vel_t_extract=None, n_phys: Optional[int] = None,
                 mu_fp_tol: float = 1.0e-10, mu_fp_max_iter: int = 30,
                 weaken_while_open: bool = True,
                 run_monitor: Optional[MJFRunMonitor] = None,
                 thin_output: int = 1):
        self.stepper = stepper
        self.theta = float(stepper.theta)
        self.aux = _copy_aux(aux0)
        self.n_c = int(n_c)
        self.reaction_scale = float(reaction_scale)
        r0 = (np.zeros(self.stepper.n_react) if warm_r is None
              else np.asarray(warm_r, float).copy())
        self.reaction_history = [r0]
        self.info_history: list = []
        self.A = None                       # ODESystem may assign; unused by MJF
        self.slip_slice = slip_slice
        self.mu_from_slip = mu_from_slip
        self.vel_t_extract = vel_t_extract
        self.n_phys = n_phys
        self.mu_fp_tol = float(mu_fp_tol)
        self.mu_fp_max_iter = int(mu_fp_max_iter)
        self.weaken_while_open = bool(weaken_while_open)
        # run control / memory options
        self.run_monitor = run_monitor
        self.thin_output = max(1, int(thin_output))
        self.failure_info: Optional[dict] = None
        self._n_steps = 0                   # successful steps taken
        self._last_entry = None             # (step_index, reaction_row, info)
        self._march_front = None            # (t, y) of the last successful step
        # running convergence maxima over ALL successful steps (thinning must
        # not change the end-of-run convergence report)
        self._mu_fp_resid_max = None
        self._mu_fp_iters_max = 0
        self._mu_fp_nonconverged = 0
        self._soc_resid_max = None

    @property
    def steps_taken(self) -> int:
        """Number of successful steps advanced (independent of thinning)."""
        return self._n_steps

    def _closed_mask(self, info):
        """Per-contact 1.0 where the step's TOTAL normal reaction is compressive.

        An open contact transmits no friction, so its tangential free-flight
        motion must not feed the slip-weakening law.  ``p_contact_effective``
        is the total reaction (0 on gap-deactivated contacts); ``p_contact``
        is the increment, which equals the total for steppers without offset
        forces (with offsets it is -offset on open rows, still <= 0).
        """
        if self.weaken_while_open:
            return 1.0
        p_eff = info.get("p_contact_effective", info.get("p_contact"))
        if p_eff is None:
            return 1.0
        p_eff = np.asarray(p_eff, float).ravel()
        return np.array([float(p_eff[sl.start] > 0.0)
                         for sl in self.stepper.block_slices])

    def _step_weakening(self, t, y, h):
        assert self.mu_from_slip is not None and self.vel_t_extract is not None
        s_k = np.asarray(y[self.slip_slice], float).ravel()
        mu_guess = np.asarray(self.mu_from_slip(s_k), float).ravel()
        v_t_theta = np.zeros(self.n_c)
        closed = 1.0
        y1 = aux1 = info1 = None
        last_resid = np.inf
        n_iter = 0
        for n_iter in range(1, self.mu_fp_max_iter + 1):
            aux_in = _copy_aux(self.aux)
            aux_in["mu"] = mu_guess.copy()
            y1, aux1, info1 = self.stepper.step(t, y, aux_in, h)
            y_theta = y + self.theta * (y1 - y)
            v_t_theta = np.asarray(self.vel_t_extract @ y_theta[:self.n_phys], float).ravel()
            closed = self._closed_mask(info1)
            s_theta = s_k + self.theta * h * np.abs(v_t_theta) * closed
            mu_theta = np.asarray(self.mu_from_slip(s_theta), float).ravel()
            last_resid = float(np.linalg.norm(mu_theta - mu_guess, ord=np.inf))
            mu_guess = mu_theta
            if last_resid <= self.mu_fp_tol:
                break
        s_next = s_k + h * np.abs(v_t_theta) * closed
        y1 = np.asarray(y1, float).copy()
        y1[self.slip_slice] = s_next
        aux1 = _copy_aux(aux1)
        aux1["mu"] = np.asarray(self.mu_from_slip(s_next), float).ravel()
        aux1["cum_slip"] = s_next.copy()
        # record mu fixed-point convergence so the caller can verify friction
        info1 = dict(info1)
        info1["mu_fp_resid"] = last_resid
        info1["mu_fp_iters"] = int(n_iter)
        info1["mu_fp_converged"] = bool(last_resid <= self.mu_fp_tol)
        return y1, aux1, info1

    def _track_convergence(self, info: dict) -> None:
        if "mu_fp_resid" in info:
            resid = float(info["mu_fp_resid"])
            if self._mu_fp_resid_max is None or resid > self._mu_fp_resid_max:
                self._mu_fp_resid_max = resid
            self._mu_fp_iters_max = max(
                self._mu_fp_iters_max, int(info.get("mu_fp_iters", 0)))
            if info.get("mu_fp_converged") is False:
                self._mu_fp_nonconverged += 1
        soc = float(info.get("soccp_residual", 0.0) or 0.0)
        if self._soc_resid_max is None or soc > self._soc_resid_max:
            self._soc_resid_max = soc

    def step(self, fun, t, y, h):
        if self.slip_slice is None:
            y1, aux1, info = self.stepper.step(t, y, self.aux, h)
        else:
            y1, aux1, info = self._step_weakening(t, y, h)
        p_contact = np.asarray(info.get("p_contact", np.zeros(self.stepper.n_react)), float)
        reaction_row = p_contact * self.reaction_scale / h
        success = bool(
            np.all(np.isfinite(y1))
            and info.get("soccp_converged", True)
            and info.get("mu_fp_converged", True)
        )
        err = float(info.get("soccp_residual", 0.0) or 0.0)
        iters = int(info.get("soccp_outer_iters", 0) or 0)
        if not success:
            # ODESolver does not commit a failed fixed step (it aborts by
            # default), so the failed step must not enter the aligned
            # histories nor contaminate ``aux`` -- otherwise the post-abort
            # checkpoint would pair a stored state with the failed attempt's
            # friction/warm-start state.  Its diagnostics are kept for the
            # driver's abort report instead.
            self.failure_info = dict(info)
            return y1, None, err, success, iters
        self.aux = aux1
        self._n_steps += 1
        entry_info = dict(info)
        entry_info["_step_h"] = float(h)    # actual step size (final step may
        self._track_convergence(entry_info)  # be clipped to land on tmax)
        if self._n_steps % self.thin_output == 0:
            self.reaction_history.append(reaction_row)
            self.info_history.append(entry_info)
        self._last_entry = (self._n_steps, reaction_row, entry_info)
        # march front: the last SUCCESSFUL (t, y) even when thinning left it
        # unstored -- the driver checkpoints this, never a rolled-back state.
        self._march_front = (t + h, y1)
        if self.run_monitor is not None:
            self.run_monitor.after_step(
                t + h, y1, self.aux, entry_info, h=h,
                extras={"reaction_row": reaction_row})
        return y1, None, err, success, iters


def solve_mjf_fixed_step(stepper, y0, aux0, tmax, h_fixed, n_c, reaction_scale,
                         *, slip_slice=None, mu_from_slip=None, vel_t_extract=None,
                         n_phys=None, warm_r=None, mu_fp_tol=1.0e-10,
                         mu_fp_max_iter=30, weaken_while_open=True,
                         max_steps=200000, label="MJF",
                         verbose=True,
                         on_step=None, progress_interval_s=None,
                         checkpoint_path=None, checkpoint_every_steps=None,
                         checkpoint_every_walltime_s=None,
                         thin_output=1, gc_interval=0, resume_from=None):
    """Drive ``stepper`` to ``tmax`` with fixed step ``h_fixed`` via ``ODESolver``.

    Drop-in for the old ``run_mjf_fixed_step``: returns the same 6-tuple
    ``(t, y, h, None, info_list, attempts)`` plus the reaction history.

    Run control (all opt-in; defaults reproduce the previous behaviour)
    -------------------------------------------------------------------
    on_step : callable, optional
        ``on_step(t, y, aux, info)`` invoked after every successful step.
        Use it for custom progress bars, live diagnostics, or user-side
        snapshots.  Exceptions propagate and abort the march.
    progress_interval_s : float, optional
        When set, print a one-line status (step count, simulation time,
        percent of horizon, step size, elapsed wall time, ETA) at most every
        this many wall-clock seconds, plus a final line when the march ends.
    checkpoint_path : str or os.PathLike, optional
        Destination ``.npz`` for restart checkpoints, written atomically via
        :func:`~solve_nivp.mjf_run_control.save_mjf_checkpoint`.  Each write
        replaces the previous file.  A final checkpoint is always written at
        the end of the march; after a fixed-step failure abort it holds the
        last SUCCESSFUL state (the march front), which under thinning may be
        ahead of the last stored trajectory row.  A failed run is flagged in
        ``attempts['failure']`` (the failed step's info dict; ``None`` on a
        clean run).
    checkpoint_every_steps : int, optional
        Write a checkpoint every N successful steps.
    checkpoint_every_walltime_s : float, optional
        Write a checkpoint whenever this much wall time has passed since the
        last one.  May be combined with ``checkpoint_every_steps`` (whichever
        trigger fires first wins).  With ``checkpoint_path`` set and neither
        trigger given, only the final checkpoint is written.
    resume_from : str, os.PathLike or dict, optional
        A checkpoint path (or the dict from
        :func:`~solve_nivp.mjf_run_control.load_mjf_checkpoint`).  The march
        then starts from the checkpointed ``(t, y, aux)`` instead of
        ``(0, y0, aux0)`` and runs to ``tmax``; the returned histories cover
        only this segment (chain segments for the full trajectory).  The
        checkpoint's reaction row seeds the segment's initial history row.

    Memory options (all opt-in)
    ---------------------------
    thin_output : int, default 1
        Store only every N-th step in the returned time/state/reaction/info
        histories (the final state is always stored).  The march itself is
        unchanged -- every step is still taken at ``h_fixed`` -- only the
        stored trajectory is subsampled, cutting history memory by ~N.  The
        reaction and info histories are thinned with the identical predicate,
        so ``MJFContactView`` alignment is preserved.  Note the returned ``h``
        array is the spacing of the *stored* states (``N * h_fixed``) and
        ``attempts['records']`` covers stored steps only, while
        ``attempts['accepted']`` and the verbose summary count every step.
    gc_interval : int, default 0
        Run ``gc.collect()`` every N successful steps (0 disables).  Useful
        on very long marches to promptly release solver temporaries.
    """
    from .ODESystem import ODESystem
    from .ODESolver import ODESolver

    thin_output = max(1, int(thin_output))

    # --- resume: override the initial state from a checkpoint ---------------
    t0 = 0.0
    step_index0 = 0
    if resume_from is not None:
        ckpt = resolve_mjf_checkpoint(resume_from)
        t0 = float(ckpt["t"])
        y0 = np.asarray(ckpt["y"], float)
        aux0 = ckpt["aux"]
        step_index0 = int(ckpt.get("step_index", 0))
        ckpt_reaction = ckpt.get("extras", {}).get("reaction_row")
        if ckpt_reaction is not None:
            warm_r = np.asarray(ckpt_reaction, float)
        if verbose:
            print(f"{label}: resuming from checkpoint t={t0:.6e} "
                  f"(step {step_index0})", flush=True)

    monitor = None
    if (on_step is not None or progress_interval_s is not None
            or checkpoint_path is not None
            or checkpoint_every_steps is not None
            or checkpoint_every_walltime_s is not None):
        monitor = MJFRunMonitor(
            t_start=t0, t_end=float(tmax), label=label, driver="fixed",
            progress_interval_s=progress_interval_s, on_step=on_step,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_every_walltime_s=checkpoint_every_walltime_s,
            step_index0=step_index0)

    method = MJFIntegrationMethod(
        stepper, aux0, n_c=n_c, reaction_scale=reaction_scale, warm_r=warm_r,
        slip_slice=slip_slice, mu_from_slip=mu_from_slip,
        vel_t_extract=vel_t_extract, n_phys=n_phys,
        mu_fp_tol=mu_fp_tol, mu_fp_max_iter=mu_fp_max_iter,
        weaken_while_open=weaken_while_open,
        run_monitor=monitor, thin_output=thin_output)

    # A resumed checkpoint may already sit at (or past) the horizon; return
    # the trivial single-state segment instead of asking ODESolver for a
    # non-increasing t_span.  Fresh (non-resume) calls keep the old fail-fast
    # behaviour: a non-increasing horizon raises in ODESolver.__init__.
    _t_eps = 4.0 * np.finfo(float).eps * max(abs(t0), abs(float(tmax)), 1.0)
    if resume_from is not None and float(tmax) - t0 <= _t_eps:
        if verbose:
            print(f"{label}: checkpoint already at tmax; nothing to do")
        t = np.asarray([t0], float)
        y = np.asarray([np.asarray(y0, float)], float)
        if monitor is not None:
            monitor.finish(
                t0, np.asarray(y0, float), method.aux, h=float(h_fixed),
                extras={"reaction_row": np.asarray(method.reaction_history[0], float)})
        result = (t, y, np.zeros(0), None, [],
                  {"accepted": [], "records": [],
                   "n_accepted": 0, "n_rejected": 0,
                   "thin_output": thin_output, "failure": None})
        return result, np.asarray(method.reaction_history, float)

    system = ODESystem(fun=lambda _t, _y: np.zeros_like(_y),
                       y0=np.asarray(y0, float).copy(), method=method, adaptive=False)
    t, y, _h, _fk, _errs = ODESolver(
        system, (t0, float(tmax)), h=float(h_fixed),
        thin_output=thin_output, gc_interval=int(gc_interval)).solve()

    # Align the (possibly thinned) reaction/info histories with the stored
    # states: ODESolver always stores the final state even off the thinning
    # grid, so mirror that here from the buffered last entry.
    if len(method.reaction_history) == len(t) - 1 and method._last_entry is not None:
        _, r_last, i_last = method._last_entry
        method.reaction_history.append(r_last)
        method.info_history.append(i_last)
    if len(method.reaction_history) != len(t):
        raise RuntimeError(
            f"{label}: reaction history has {len(method.reaction_history)} "
            f"rows for {len(t)} stored states (thin_output={thin_output}); "
            f"this is a bookkeeping bug, please report it")

    n_total = method.steps_taken
    h_hist = list(np.diff(t)) if len(t) > 1 else []
    info_hist = []
    for k, info in enumerate(method.info_history):
        ia = dict(info)
        _step_h = ia.pop("_step_h", float(h_fixed))
        if thin_output == 1 and k < len(h_hist):
            ia["fixed_h"] = h_hist[k]
        else:
            # thinned storage: report each stored step's ACTUAL size (the
            # final step may be clipped to land on tmax), not the stored-grid
            # spacing that np.diff(t) would give.
            ia["fixed_h"] = _step_h
        info_hist.append(ia)
    attempts = {"accepted": [True] * n_total,
                "records": [{"accepted": True, "t": float(t[k]), "h": h_hist[k]}
                            for k in range(len(h_hist))],
                "n_accepted": n_total, "n_rejected": 0,
                "thin_output": thin_output,
                "failure": method.failure_info}
    if verbose:
        print(f"{label}: done {len(t)} states, t_final={t[-1]:.4e}, "
              f"steps={n_total}, h={h_fixed:.3e} (via ODESolver)")
        if method._mu_fp_resid_max is not None:
            _bad = method._mu_fp_nonconverged
            ok = "CONVERGED" if _bad == 0 else f"NOT CONVERGED ({_bad} steps)"
            print(f"{label}: mu fixed-point -- max resid={method._mu_fp_resid_max:.2e} "
                  f"(tol {mu_fp_tol:.0e}), max iters={method._mu_fp_iters_max}"
                  f"/{mu_fp_max_iter} -> {ok}")
        if method._soc_resid_max is not None:
            print(f"{label}: contact SSN -- max soccp_residual="
                  f"{method._soc_resid_max:.2e}")
        if method.failure_info is not None:
            fi = method.failure_info
            _causes = []
            if fi.get("mu_fp_converged", True) is False:
                _causes.append(
                    f"mu fixed-point not converged (resid={fi.get('mu_fp_resid', float('nan')):.2e})")
            if not fi.get("soccp_converged", True):
                _causes.append(
                    f"contact SSN not converged (residual={float(fi.get('soccp_residual', 0.0) or 0.0):.2e})")
            print(f"{label}: march ABORTED at step {n_total + 1} -- "
                  f"{'; '.join(_causes) or 'non-finite state'} "
                  f"(diagnostics in attempts['failure'])")
    if monitor is not None:
        # Checkpoint the march front (last SUCCESSFUL state), never a stored
        # state paired with newer aux: on a failure abort with thinning the
        # last stored state can be several committed steps behind the front.
        if method._march_front is not None:
            _t_front, _y_front = method._march_front
        else:
            _t_front, _y_front = t0, np.asarray(y0, float)
        _r_front = (method._last_entry[1] if method._last_entry is not None
                    else method.reaction_history[0])
        monitor.finish(
            float(_t_front), np.asarray(_y_front, float), method.aux,
            h=float(h_fixed),
            extras={"reaction_row": np.asarray(_r_front, float)},
            status="done" if method.failure_info is None else "aborted")
    result = (np.asarray(t, float), np.asarray(y, float),
              np.asarray(h_hist, float), None, info_hist, attempts)
    return result, np.asarray(method.reaction_history, float)
