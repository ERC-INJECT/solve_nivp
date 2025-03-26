import numpy as np

class AdaptiveStepping:
    """
    Adaptive time-stepping controller for non-BDF integrators using a full-step/two-half-step strategy.
    
    This class estimates the local error of a proposed step by comparing a single full step
    with two successive half steps. Based on the computed error, the step size is adapted for
    the next integration step.
    
    Parameters:
        integrator : object
            An integrator instance that implements a 'step' method.
        component_slices : list of slice objects, optional
            Slices to partition the state vector into components for error estimation.
        atol : float, default 1e-6
            Absolute tolerance for error estimation.
        rtol : float, default 1e-3
            Relative tolerance for error estimation.
        h0 : float, default 1e-2
            Initial time step size.
        h_min : float, default 1e-5
            Minimum allowable time step.
        h_max : float, default 1e3
            Maximum allowable time step.
        h_up : float, default 2.0
            Maximum factor for increasing the step size.
        h_down : float, default 0.6
            Factor for decreasing the step size.
        q : float, default 0.5
            Order exponent used in the step size control.
        verbose : bool, default False
            If True, prints diagnostic messages during stepping.
        skip_error_indices : list of int, optional
            Component indices to skip during error estimation.
    """
    def __init__(self, integrator, component_slices=None, atol=1e-6, rtol=1e-3, h0=1e-2,
                 h_min=1e-5, h_max=1e3, h_up=2.0, h_down=0.6, q=0.5, verbose=False,
                 skip_error_indices=None):
        self.integrator = integrator
        self.atol = atol
        self.rtol = rtol
        self.h = h0
        self.h_min = h_min
        self.h_max = h_max
        self.h_up = h_up
        self.h_down = h_down
        self.q = q
        self.verbose = verbose
        self.component_slices = component_slices
        self.skip_error_indices = skip_error_indices if skip_error_indices is not None else []

    def step(self, fun, t, y, h):
        """
        Perform one adaptive time step using a full-step and two half-step strategy.
        
        The method:
          1. Computes a full-step using the integrator.
          2. Computes two half-steps and compares the result with the full-step.
          3. Estimates the error based on the difference between the full-step and half-step solutions.
          4. Adjusts the time step size accordingly.
        
        Parameters:
            fun : callable
                Function representing the right-hand side (derivative) of the ODE.
            t : float
                Current time.
            y : array_like
                Current state vector.
            h : float
                Proposed time step size.
                
        Returns:
            y_new : array_like
                New state computed from the two half-steps.
            fk_new : array_like
                Derivative evaluated at the new state.
            h_new : float
                Adapted time step for the next step.
            E : float
                Estimated scaled error.
            success : bool
                True if the error is acceptable (E ≤ 1), False otherwise.
            solver_error : float
                Error reported by the integrator's solver for the full-step.
            solver_iterations : int
                Number of iterations taken by the integrator for the full-step.
        """
        # Compute the full step.
        try:
            y_full, fk_full, solver_error_full, solver_success_full, iter_full = \
                self.integrator.step(fun, t, y, h)
        except RuntimeError as e:
            if self.verbose:
                print(f"Error during full step at t={t:.5f}: {e}")
            return y, None, h, np.inf, False, np.inf, 0

        if not solver_success_full:
            return y, None, h, np.inf, False, solver_error_full, iter_full

        # Compute two half-steps.
        half_h = 0.5 * h
        try:
            y_half, _, _, success_half1, _ = self.integrator.step(fun, t, y, half_h)
            if not success_half1:
                return y, None, half_h, np.inf, False, np.inf, 0

            y_half_full, fk_half_full, _, success_half2, _ = \
                self.integrator.step(fun, t + half_h, y_half, half_h)
            if not success_half2:
                return y, None, half_h, np.inf, False, np.inf, 0
        except RuntimeError as e:
            if self.verbose:
                print(f"Error during half steps at t={t:.5f}: {e}")
            return y, None, h, np.inf, False, np.inf, 0

        # Determine the state components to estimate error.
        if self.component_slices is not None:
            components_prev = [y[s] for s in self.component_slices]
            components_LO = [y_full[s] for s in self.component_slices]
            components_HI = [y_half_full[s] for s in self.component_slices]
        else:
            components_prev = [y]
            components_LO = [y_full]
            components_HI = [y_half_full]

        # Compute error order exponent (p = 1/q - 1).
        p = 1 / self.q - 1
        all_scaled_errors = []

        # Loop over each component for error estimation.
        for idx, (comp_prev, comp_LO, comp_HI) in enumerate(zip(components_prev, components_LO, components_HI)):
            # Skip components if specified.
            if idx in self.skip_error_indices:
                continue
            # Estimate the error using the difference between full and half-step approximations.
            error_est = (comp_LO - comp_HI) / (2**p - 1)
            # Compute the tolerance for the component.
            etol = self.atol + self.rtol * np.maximum(np.abs(comp_HI), np.abs(comp_prev))
            # Scale the error.
            scaled_error = error_est / etol
            all_scaled_errors.append(scaled_error.flatten())
            if self.verbose:
                err_str = ", ".join(f"{e:.3e}" for e in scaled_error.flatten())
                print(f"Component {idx+1} scaled errors: [{err_str}]")

        # Combine the scaled errors.
        if all_scaled_errors:
            all_scaled_errors = np.concatenate(all_scaled_errors)
            E = np.sqrt(np.mean(all_scaled_errors**2))
        else:
            E = 0.0

        # Determine if the step is acceptable.
        success = (E <= 1.0)

        # Adjust the step size using the computed error.
        s = self.h_up if E == 0 else (1.0 / E)**(self.q)
        h_new = min(self.h_up * h, s * h)
        h_new = min(max(h_new, self.h_min), self.h_max)

        # If the error is too large and we are already at h_min, abort gracefully.
        if not success:
            # If the adaptive step fails, reduce the step size and try again.
            h_new = self.h_down*h
            if self.verbose:
                print(f"Step at t={t:.5f} rejected. Reducing step size to {h:.5e}.")
            return y, fk_full, h_new, E, False, solver_error_full, iter_full

        return y_half_full, fk_half_full, h_new, E, success, solver_error_full, iter_full
