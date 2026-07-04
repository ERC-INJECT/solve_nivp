import numpy as np
import gc
from typing import Any, Tuple, List, Optional

class ODESolver:
    """Time integration driver (fixed or adaptive) for an ``ODESystem``.

    Stores growing histories of time grid, states, step sizes and solver
    diagnostics. For adaptive runs, rejected steps do not append entries.

    Memory-saving options
    ---------------------
    For large systems (100k+ DOFs), storing the full state at every time step
    can consume significant memory.  Several knobs are provided:

    ``thin_output``
        Store only every *N*-th accepted step (first and last are always kept).
    ``store_fk``
        Set ``False`` to skip storing the residual vectors (which have the same
        size as the state vector).
    ``gc_interval``
        Call ``gc.collect()`` every *N* accepted steps to free unreferenced
        Python objects (e.g. stale sparse factorisations released by the
        solver).

    Residual semantics
    ------------------
    ``fk`` list stores the raw implicit residual / function value ``F(y_k)``
    returned by the integrator's nonlinear solve at each *accepted* step. This
    can be useful for post-process diagnostics (e.g. monitoring equilibrium of
    projected components). Entries may be ``None`` if a method does not return
    a residual (should not occur with current integrators).
    """
    def __init__(
        self,
        system: Any,
        t_span: Tuple[float, float],
        h: float = 1e-2,
        thin_output: int = 1,
        store_fk: bool = True,
        gc_interval: int = 0,
        abort_on_fixed_failure: bool = True,
        t_eval: Optional[np.ndarray] = None,
    ):
        """Initialize the time integration driver.

        Parameters
        ----------
        system : object
            ODE system to integrate.
        t_span : tuple of float
            ``(t0, tf)`` start and end times.
        h : float, default 1e-2
            Initial time step size.
        thin_output : int, default 1
            Store every *N*-th accepted step. First and last steps are always
            stored.
        store_fk : bool, default True
            Whether to store per-step residual vectors. Setting this to
            ``False`` saves about one state-vector of memory per step.
        gc_interval : int, default 0
            Call ``gc.collect()`` every *N* accepted steps. ``0`` disables
            explicit garbage collection.
        abort_on_fixed_failure : bool, default True
            Stop fixed-step integration at the first nonlinear failure instead
            of marching forward with the failed state. The failed attempt is
            still recorded in ``error_estimates``.
        t_eval : ndarray or None, optional
            Strictly increasing array of times in ``[t0, tf]`` that must be
            evaluated. When provided, the time loop clips each step so it lands
            exactly on each requested entry, and the returned histories contain
            only those entries. Pass ``None`` to keep the adaptive or fixed
            integration grid as the output.
        """
        self.system = system
        self.t0, self.tf = t_span
        self.h_initial = h

        if not (h > 0.0):
            raise ValueError(f"step size h must be positive, got {h}")
        if not (self.tf > self.t0):
            raise ValueError(
                f"t_span must be increasing, got (t0={self.t0}, tf={self.tf})"
            )

        # ---- t_eval validation and bookkeeping ----
        self._use_t_eval: bool = False
        self._t_eval: Optional[np.ndarray] = None
        self._t_eval_idx: int = 0
        if t_eval is not None:
            t_eval_arr = np.asarray(t_eval, dtype=float).reshape(-1)
            if t_eval_arr.size > 0:
                if np.any(np.diff(t_eval_arr) <= 0.0):
                    raise ValueError("t_eval must be strictly increasing")
                tf_eps = 1.0e-12 * max(abs(self.t0), abs(self.tf), 1.0)
                if (t_eval_arr[0] < self.t0 - tf_eps
                        or t_eval_arr[-1] > self.tf + tf_eps):
                    raise ValueError(
                        f"t_eval out of range [{self.t0}, {self.tf}]: "
                        f"got [{t_eval_arr[0]}, {t_eval_arr[-1]}]"
                    )
                self._t_eval = np.clip(t_eval_arr, self.t0, self.tf)
                self._use_t_eval = True

        if self._use_t_eval:
            self.t_values: List[float] = []
            self.y_values: List[np.ndarray] = []
            self.h_values: List[float] = []
            self.fk: List[Any] = []
            # If t_eval[0] coincides with t0, record the initial state.
            te0 = float(self._t_eval[0])
            t0_eps = 1.0e-12 * max(abs(self.t0), 1.0)
            if abs(te0 - self.t0) <= t0_eps:
                self.t_values.append(te0)
                self.y_values.append(self.system.current_y.copy())
                self.h_values.append(h)
                self.fk.append(None)
                self._t_eval_idx = 1
        else:
            self.t_values = [self.t0]
            self.y_values = [self.system.current_y.copy()]
            self.h_values = [h]
            self.fk = []

        self.error_estimates: List[Tuple[Any, bool, int]] = []
        # Memory-saving options
        self.thin_output = max(1, int(thin_output))
        self.store_fk = bool(store_fk)
        self.gc_interval = max(0, int(gc_interval))
        self.abort_on_fixed_failure = bool(abort_on_fixed_failure)
        self.terminal_failure: Optional[Tuple[Any, bool, int]] = None

    def solve(self, return_attempts: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[Any, bool, int]]]:
        """Integrate from ``t0`` to ``tf``.

        Parameters
        ----------
        return_attempts : bool, default False
            When ``True`` and adaptive stepping is enabled with attempt logging,
            include the raw attempt log as a sixth return value.

        Returns
        -------
        t_values : ndarray (m,)
            Time points (monotone, includes final time).
        y_values : ndarray (m, n)
            State history.
        h_values : ndarray (m,)
            Step-size history aligned with ``t_values``. The first entry is the
            initial ``h`` guess; later entries are the accepted step sizes for
            the stored states.
        fk : object ndarray (m-1,)
            Residual / implicit function evaluations for the stored accepted
            steps (one entry per stored state after the initial condition).
        error_estimates : list[tuple]
            Nonlinear-solver diagnostics ``(solver_error, success, iterations)``
            for the stored accepted states. In fixed-step mode a terminal
            failure tuple is appended even if the failed state is not stored.
        attempts : dict or None, optional
            Only returned when ``return_attempts`` is ``True``. Contains arrays of
            attempted times, step sizes, acceptance flags, etc., if recorded.
        """
        t = self.t0
        h = self.h_initial
        self.terminal_failure = None
        stepper = getattr(self.system, 'adaptive_stepper', None)
        if stepper is not None and hasattr(stepper, 'reset_attempt_log'):
            stepper.reset_attempt_log()
        
        _step_count = 0          # accepted-step counter
        _thin = self.thin_output
        _gc_iv = self.gc_interval

        # Floating-point tolerance for the "close enough to tf" check.
        # After many accepted steps the accumulated t may land a rounding
        # error below tf.  Attempting a micro-step of order eps would make
        # the implicit Jacobian A/(γh) explode and the nonlinear solver
        # fail, falsely reporting "reached minimum step size".
        _tf_eps = 4.0 * np.finfo(float).eps * max(abs(self.t0), abs(self.tf), 1.0)
        # Horizon-snap fraction: if a full step would leave only a sliver smaller
        # than this fraction of the step before tf, fold the sliver into that step
        # (land tf exactly) instead of appending a degenerate micro-step.  The
        # accumulated ``t += h_step`` drift after N steps is ~N*eps*tf, which can
        # far exceed _tf_eps, so relying on _tf_eps alone still lets a micro-step
        # through (e.g. tf=10, h=0.02 lands ~1e-13 short and appends h~1e-13).
        _snap_frac = 1.0e-3

        # Helper to record a stored sample (handles store_fk + fk fallback).
        def _record(t_store, y, fk_val, h_taken, errinfo):
            self.t_values.append(t_store)
            self.y_values.append(y.copy())
            if self.store_fk:
                self.fk.append(fk_val.copy() if hasattr(fk_val, "copy") else fk_val)
            else:
                self.fk.append(None)
            self.h_values.append(h_taken)
            self.error_estimates.append(errinfo)

        # Helper to drain any t_eval points that the latest accepted step
        # advanced over (catches both exact landings and tiny float drift).
        def _drain_t_eval(t_now, y_now, fk_val, h_taken, errinfo):
            if not self._use_t_eval:
                return
            te_eps = 1.0e-9 * max(abs(t_now), 1.0)
            while (self._t_eval_idx < len(self._t_eval)
                   and self._t_eval[self._t_eval_idx] <= t_now + te_eps):
                te = float(self._t_eval[self._t_eval_idx])
                _record(te, y_now, fk_val, h_taken, errinfo)
                self._t_eval_idx += 1

        while self.tf - t > _tf_eps:
            # Ensure we do not overshoot the final time.
            h_step = min(h, self.tf - t)
            # When t_eval is given, also clip the step so it lands exactly
            # on the next required output time.
            if self._use_t_eval and self._t_eval_idx < len(self._t_eval):
                next_te = float(self._t_eval[self._t_eval_idx])
                if next_te > t:
                    h_step = min(h_step, next_te - t)
            # Land the horizon exactly: if taking this step would leave only a
            # sub-step sliver before tf (float drift, not a wanted fractional
            # step), fold it into this step rather than appending a degenerate
            # micro-step next iteration.  A micro-step (h ~ 1e-13) makes an
            # impulse/reaction-based stepper (reaction ~ impulse/h) blow up and
            # the nonlinear cone solve stall.  Fractional final steps that are a
            # real fraction of h (>= _snap_frac*h) are left untouched.
            if 0.0 < (self.tf - t) - h_step < _snap_frac * h:
                h_step = self.tf - t
            if self.system.adaptive:
                # Adaptive stepping returns:
                # (y_new, fk_new, h_new, E, success, solver_error, iterations)
                y_new, fk_new, h_new, E, success, solver_error, iterations = self.system.step(t, h_step)
                if success:
                    t += h_step
                    _step_count += 1
                    if self._use_t_eval:
                        _drain_t_eval(t, y_new, fk_new, h_step,
                                      (solver_error, success, iterations))
                    else:
                        # Thin output: only store every Nth step (always store last)
                        _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                        if _step_count % _thin == 0 or _is_last:
                            _record(t, y_new, fk_new, h_step,
                                    (solver_error, success, iterations))
                    self.system.current_y = y_new
                    h = h_new  # Update step size for next iteration.
                    # Periodic garbage collection for large problems
                    if _gc_iv > 0 and _step_count % _gc_iv == 0:
                        gc.collect()
                else:
                    h = h_new
                    # If the adaptive step fails, reduce the step size and try again.
                    if h<=  self.system.adaptive_stepper.h_min:
                        if self.system.verbose:
                            print(f"Failed integration: reached minimum step size at t={t:.5f} and step did not converge.")
                        break
            else:
                # Fixed stepping mode.
                y_new, fk_new, solver_error, success, iterations = self.system.step(t, h_step)
                if success:
                    t += h_step
                    _step_count += 1
                    if self._use_t_eval:
                        _drain_t_eval(t, y_new, fk_new, h_step,
                                      (solver_error, success, iterations))
                    else:
                        _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                        if _step_count % _thin == 0 or _is_last:
                            _record(t, y_new, fk_new, h_step,
                                    (solver_error, success, iterations))
                    self.system.current_y = y_new
                    if _gc_iv > 0 and _step_count % _gc_iv == 0:
                        gc.collect()
                else:
                    self.terminal_failure = (solver_error, success, iterations)
                    if self.abort_on_fixed_failure:
                        self.error_estimates.append((solver_error, success, iterations))
                        if self.system.verbose:
                            print(
                                f"Failed fixed-step integration: nonlinear solve did not converge "
                                f"at t={t:.5f} with h={h_step:.5f}."
                            )
                        break
                    t += h_step
                    _step_count += 1
                    if self._use_t_eval:
                        _drain_t_eval(t, y_new, fk_new, h_step,
                                      (solver_error, success, iterations))
                    else:
                        _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                        if _step_count % _thin == 0 or _is_last:
                            _record(t, y_new, fk_new, h_step,
                                    (solver_error, success, iterations))
                    self.system.current_y = y_new
                    if _gc_iv > 0 and _step_count % _gc_iv == 0:
                        gc.collect()
            # pbar.update(h_step)
        # pbar.close()
        t_arr = np.array(self.t_values)
        y_arr = np.array(self.y_values)
        h_arr = np.array(self.h_values)
        fk_arr = np.array(self.fk, dtype=object)

        if return_attempts:
            attempt_log = None
            if stepper is not None and hasattr(stepper, 'get_attempt_log'):
                attempt_log = stepper.get_attempt_log()
            return t_arr, y_arr, h_arr, fk_arr, self.error_estimates, attempt_log

        return t_arr, y_arr, h_arr, fk_arr, self.error_estimates
