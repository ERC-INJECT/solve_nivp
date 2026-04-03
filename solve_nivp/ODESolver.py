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
    ):
        """
        Initialize the ODESolver.
        
        Parameters:
            system: The ODE system to be integrated.
            t_span: A tuple (t0, tf) specifying the start and end times.
            h: The initial time step size.
            thin_output: Store every *N*-th accepted step (1 = store all).
                First and last steps are always stored.
            store_fk: Whether to store per-step residual vectors.
                Setting to False saves ~1× state-vector memory per step.
            gc_interval: Call ``gc.collect()`` every *N* accepted steps
                (0 = disabled). Useful for large problems where stale
                solver factorisations may linger.
            abort_on_fixed_failure: Stop fixed-step integration at the first
                nonlinear failure instead of marching forward with the failed
                state. The failed attempt is still recorded in
                ``error_estimates``.
        """
        self.system = system
        self.t0, self.tf = t_span
        self.h_initial = h
        self.t_values: List[float] = [self.t0]
        self.y_values: List[np.ndarray] = [self.system.current_y.copy()]
        self.h_values: List[float] = [h]
        self.error_estimates: List[Tuple[Any, bool, int]] = []
        self.fk: List[Any] = []
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

        while self.tf - t > _tf_eps:
            # Ensure we do not overshoot the final time.
            h_step = min(h, self.tf - t)
            if self.system.adaptive:
                # Adaptive stepping returns:
                # (y_new, fk_new, h_new, E, success, solver_error, iterations)
                y_new, fk_new, h_new, E, success, solver_error, iterations = self.system.step(t, h_step)
                if success:
                    t += h_step
                    _step_count += 1
                    # Thin output: only store every Nth step (always store last)
                    _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                    if _step_count % _thin == 0 or _is_last:
                        self.t_values.append(t)
                        self.y_values.append(y_new.copy())
                        if self.store_fk:
                            self.fk.append(fk_new.copy() if fk_new is not None else None)
                        else:
                            self.fk.append(None)
                        self.h_values.append(h_step)
                        self.error_estimates.append((solver_error, success, iterations))
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
                    _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                    if _step_count % _thin == 0 or _is_last:
                        self.t_values.append(t)
                        self.y_values.append(y_new.copy())
                        if self.store_fk:
                            self.fk.append(fk_new.copy() if fk_new is not None else None)
                        else:
                            self.fk.append(None)
                        self.h_values.append(h_step)
                        self.error_estimates.append((solver_error, success, iterations))
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
                    _is_last = (t >= self.tf - 1e-14 * abs(self.tf))
                    if _step_count % _thin == 0 or _is_last:
                        self.t_values.append(t)
                        self.y_values.append(y_new.copy())
                        if self.store_fk:
                            self.fk.append(fk_new.copy() if fk_new is not None else None)
                        else:
                            self.fk.append(None)
                        self.h_values.append(h_step)
                        self.error_estimates.append((solver_error, success, iterations))
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
