import numpy as np
from typing import Any, Tuple, List

class ODESolver:
    """Time integration driver (fixed or adaptive) for an ``ODESystem``.

    Stores growing histories of time grid, states, step sizes and solver
    diagnostics. For adaptive runs, rejected steps do not append entries.

    Residual semantics
    ------------------
    ``fk`` list stores the raw implicit residual / function value ``F(y_k)``
    returned by the integrator's nonlinear solve at each *accepted* step. This
    can be useful for post-process diagnostics (e.g. monitoring equilibrium of
    projected components). Entries may be ``None`` if a method does not return
    a residual (should not occur with current integrators).
    """
    def __init__(self, system: Any, t_span: Tuple[float, float], h: float = 1e-2):
        """
        Initialize the ODESolver.
        
        Parameters:
            system: The ODE system to be integrated.
            t_span: A tuple (t0, tf) specifying the start and end times.
            h: The initial time step size.
        """
        self.system = system
        self.t0, self.tf = t_span
        self.h_initial = h
        self.t_values: List[float] = [self.t0]
        self.y_values: List[np.ndarray] = [self.system.current_y.copy()]
        self.h_values: List[float] = [h]
        self.error_estimates: List[Tuple[Any, bool, int]] = []
        self.fk: List[Any] = []

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
            Step sizes used; first entry equals initial ``h`` guess.
        fk : object ndarray (m,)
            Residual / implicit function evaluations per accepted step.
        error_estimates : list[tuple]
            Per-step tuples ``(solver_error, success, iterations)`` coming from
            the nonlinear solver (solver_error is typically final residual norm).
        attempts : dict or None, optional
            Only returned when ``return_attempts`` is ``True``. Contains arrays of
            attempted times, step sizes, acceptance flags, etc., if recorded.
        """
        t = self.t0
        h = self.h_initial
        stepper = getattr(self.system, 'adaptive_stepper', None)
        if stepper is not None and hasattr(stepper, 'reset_attempt_log'):
            stepper.reset_attempt_log()
        
        # Initialize progress bar for integration.
        # pbar = tqdm(total=self.tf - self.t0, desc='Integration Progress', unit='time unit')
        while t < self.tf:
            # Ensure we do not overshoot the final time.
            h_step = min(h, self.tf - t)
            if self.system.adaptive:
                # Adaptive stepping returns:
                # (y_new, fk_new, h_new, E, success, solver_error, iterations)
                y_new, fk_new, h_new, E, success, solver_error, iterations = self.system.step(t, h_step)
                if success:
                    t += h_step
                    self.t_values.append(t)
                    self.y_values.append(y_new.copy())
                    self.fk.append(fk_new.copy() if fk_new is not None else None)
                    self.h_values.append(h_new)
                    self.error_estimates.append((solver_error, success, iterations))
                    self.system.current_y = y_new
                    h = h_new  # Update step size for next iteration.
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
                t += h_step
                self.t_values.append(t)
                self.y_values.append(y_new.copy())
                self.fk.append(fk_new.copy() if fk_new is not None else None)
                self.h_values.append(h_step)
                self.error_estimates.append((solver_error, success, iterations))
                self.system.current_y = y_new
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
