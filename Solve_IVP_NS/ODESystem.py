import numpy as np
from typing import Callable, Optional, Union

# Import integration methods from integrations.py.
# It is assumed that integrations.py is in the same package or accessible via the PYTHONPATH.
from .integrations import (
    IntegrationMethod,
    BackwardEuler,
    Trapezoidal,
    ThetaMethod,
    CompositeMethod,
    EmbeddedBETR,
    # BDFMethod,
    # AdaptiveSteppingBDF
)

# from .adaptive_integrator import AdaptiveStepping

class ODESystem:
    """Encapsulate RHS, initial state and integration method configuration.

    The system binds a user RHS ``fun`` with an implicit integration method
    (possibly adaptive) and stores current state for the driver. The RHS may
    support signature variants ``fun(t, y)`` or ``fun(t, y, Fk)`` (third
    argument ignored if unused).

    Parameters
    ----------
    fun : callable
        ODE right-hand side ``fun(t, y) -> ndarray``.
    y0 : array_like, shape (n,)
        Initial state vector.
    method : str | IntegrationMethod, default 'backward_euler'
        Integration scheme name or pre-instantiated method object.
    a : float, default 1.0
        Auxiliary parameter (currently only placeholder for composite schemes).
    adaptive : bool, default False
        Enable adaptive step controller (two half-step Richardson + PI).
    atol, rtol : float
        Absolute / relative tolerances (adaptive only).
    component_slices : list[slice], optional
        Partition for per-block error norm and projection logic.
    verbose : bool, default False
        Emit basic rejection diagnostics.
    A : ndarray, optional
        Constant mass / descriptor matrix; identity if omitted.

    Notes
    -----
    Integrator ``step`` methods must return a 5‑tuple ``(y_new, Fk_new, err, success, iterations)``.
    ``Fk_new`` is propagated upward and recorded by the driver for diagnostics.
    """
    def __init__(self, 
                 fun: Callable[[float, np.ndarray], np.ndarray], 
                 y0: Union[np.ndarray, list],
                 method: Union[str, IntegrationMethod] = 'backward_euler',
                 a: float = 1.0,
                 adaptive: bool = False,
                 atol: float = 1e-6,
                 rtol: float = 1e-3,
                 component_slices: Optional[list] = None,
                 verbose: bool = False,
                 A: Optional[np.ndarray] = None):
        self.fun = fun
        self.y0 = np.array(y0, dtype=float)
        self.current_y = self.y0.copy()
        self.adaptive = adaptive
        self.atol = atol
        self.rtol = rtol
        self.verbose = verbose
        self.component_slices = component_slices
        self.A = A

        # If method is an instance of IntegrationMethod, use it directly.
        if isinstance(method, IntegrationMethod):
            self.method = method
            if A is not None:
                self.method.A = A
        # Otherwise, select the integration method based on the provided string.
        elif isinstance(method, str):
            self.method = self._select_method(method.lower(), a, A)
        else:
            raise ValueError("Invalid integration method specification.")

        # Set up adaptive stepping if requested.
        if self.adaptive:
            # # For BDF methods, use a specialized adaptive stepper.
            # if self.method.__class__.__name__.lower() == 'bdfmethod':
            #     self.adaptive_stepper = AdaptiveSteppingBDF(
            #         self.method,
            #         atol=self.atol,
            #         rtol=self.rtol
            #     )
            # else:
                # Otherwise, use the generic adaptive stepping mechanism.
                # Always use the generic adaptive stepper
                from .adaptive_integrator import AdaptiveStepping
                self.adaptive_stepper = AdaptiveStepping(
                    integrator=self.method,
                    component_slices=self.component_slices,
                    atol=self.atol,
                    rtol=self.rtol,
                    verbose=self.verbose
                )

    def _select_method(self, method_name: str, a: float, A: Optional[np.ndarray]):
        """
        Select and return an integration method based on the provided method name.
        
        Parameters:
            method_name : str
                Name of the integration method.
            a : float
                Parameter used by some integrators.
            A : optional
                Matrix parameter to pass to the integration method.
        
        Returns:
            An instance of the selected integration method.
        """
        if method_name == 'backward_euler':
            return BackwardEuler(A=A)
        elif method_name == 'trapezoidal':
            return Trapezoidal(A=A)
        elif method_name == 'theta':
            return ThetaMethod(theta=0.5, A=A)
        elif method_name == 'composite':
            return CompositeMethod(a=a, A=A)
        elif method_name == 'embedded_betr':
            return EmbeddedBETR(A=A)
        else:
            raise ValueError(f"Unknown integration method: {method_name}")
    
    def step_fixed(self, t: float, h: float):
        """
        Perform one fixed-step integration.
        
        Parameters:
            t : float
                Current time.
            h : float
                Fixed time step size.
        
        Returns:
            A tuple containing:
                y_new : array_like, updated state vector.
                f_new : array_like, derivative evaluated at the new state.
                solver_error : float, error reported by the integrator.
                success : bool, solver success flag.
                iterations : int, number of iterations taken by the solver.
        """
        y_new, f_new, solver_error, success, iterations = self.method.step(self.fun, t, self.current_y, h)
        self.current_y = y_new  # Update the system state.
        return y_new, f_new, solver_error, success, iterations

    def step_adaptive(self, t: float, h: float):
        """
        Perform one adaptive-step integration using the adaptive stepper.
        
        Parameters:
            t : float
                Current time.
            h : float
                Proposed time step size.
        
        Returns:
            The result of the adaptive stepping procedure.
        """
        return self.adaptive_stepper.step(self.fun, t, self.current_y, h)
    
    def step(self, t: float, h: float):
        """
        Take one integration step using either adaptive or fixed stepping.
        
        Parameters:
            t : float
                Current time.
            h : float
                Time step size.
        
        Returns:
            The integration step results, which vary depending on the stepping mode.
        """
        if self.adaptive:
            return self.step_adaptive(t, h)
        else:
            return self.step_fixed(t, h)
