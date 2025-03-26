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
    # BDFMethod,
    # AdaptiveSteppingBDF
)

# from .adaptive_integrator import AdaptiveStepping

class ODESystem:
    """
    Encapsulates an ordinary differential equation (ODE) system together with a numerical integrator.
    
    This class sets up the system using an ODE function, initial state, and a selected integration method.
    It supports both fixed-step and adaptive-step integration.
    
    Parameters:
        fun : callable
            Function defining the ODE, with signature fun(t, y).
        y0 : array_like
            Initial state vector.
        method : str or IntegrationMethod, default 'backward_euler'
            The integration method to use. This can be a string specifying the method (e.g., 'backward_euler',
            'trapezoidal', 'theta', 'composite', 'bdf') or an instance of an IntegrationMethod.
        a : float, default 1.0
            Parameter used for some integrators (e.g., CompositeMethod).
        adaptive : bool, default False
            Flag indicating whether to use adaptive time stepping.
        atol : float, default 1e-6
            Absolute tolerance for adaptive stepping.
        rtol : float, default 1e-3
            Relative tolerance for adaptive stepping.
        component_slices : list of slice objects, optional
            Slices that partition the state vector into components (used for error estimation).
        verbose : bool, default False
            If True, enables verbose logging during stepping.
        A : optional (e.g., np.array)
            A matrix parameter to pass to the integration method.
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
                from .adaptive_integrator import AdaptiveStepping  # Adjust this import if AdaptiveStepping is in a different module.
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
        elif method_name == 'bdf':
            return BDFMethod(A=A, atol=self.atol, rtol=self.rtol)
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
