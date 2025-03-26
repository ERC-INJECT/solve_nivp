import numpy as np
import math
from abc import ABC, abstractmethod
from scipy.optimize import root
from .nonlinear_solvers import ImplicitEquationSolver  # Relative import for a solver class

class IntegrationMethod(ABC):
    """
    Abstract base class for integration methods.
    
    Classes derived from IntegrationMethod must implement the `step` method,
    which advances the solution of an ODE from time t to t+h.
    """
    
    @abstractmethod
    def step(self, fun, t, y, h):
        """
        Advance the solution of an ODE by one time step.
        
        Parameters:
            fun: callable
                The function defining the ODE (dy/dt = fun(t, y)).
            t: float
                The current time.
            y: np.array
                The current state vector.
            h: float
                The time step size.
                
        Returns:
            The new state after taking the step (and possibly additional diagnostic info).
        """
        pass

class BackwardEuler(IntegrationMethod):
    """
    Implements the Backward Euler implicit integration method.
    
    Attributes:
        solver: ImplicitEquationSolver
            A solver instance used to solve the implicit equations.
        A: np.array or None
            A matrix used in the formulation of the method. If None, the identity matrix is used.
        use_identity: bool
            Flag to indicate whether to use the identity matrix.
        _ID_CACHE: dict
            Class-level cache for identity matrices to avoid recomputation.
    """
    
    # Cache for identity matrices to avoid repeated allocation.
    _ID_CACHE = {}
    
    def __init__(self, solver=None, A=None):
        """
        Initialize a Backward Euler integration method.
        
        Parameters:
            solver: ImplicitEquationSolver, optional
                The solver to use for solving the implicit equation. Defaults to using 
                an ImplicitEquationSolver with method 'root'.
            A: np.array, optional
                The matrix used in the formulation. If not provided, identity matrix is used.
        """
        self.solver = solver or ImplicitEquationSolver(method='root')
        self.A = A
        self.use_identity = A is None

    def _get_A(self, n):
        """
        Retrieve or compute the appropriate matrix A of size (n x n).
        
        If no specific matrix A is provided during initialization, the identity matrix is used.
        Identity matrices are cached to avoid repeated computation.
        
        Parameters:
            n: int
                The size of the square matrix needed.
                
        Returns:
            np.array: The matrix A (either the identity matrix or the user-specified matrix).
        """
        if self.use_identity:
            if n not in self._ID_CACHE:
                self._ID_CACHE[n] = np.eye(n)
            return self._ID_CACHE[n]
        return self.A

    def step(self, fun, t, y, h):
        """
        Take one Backward Euler step for the ODE.
        
        Formulates the implicit equation:
            A @ ((y_new - y) / h) - fun(t+h, y_new) = 0
        and solves for y_new.
        
        Parameters:
            fun: callable
                The function defining the ODE.
            t: float
                The current time.
            y: np.array
                The current state.
            h: float
                The time step size.
                
        Returns:
            The updated state computed by the solver.
        """
        A_local = self._get_A(len(y))
        # Define the implicit equation for the backward Euler step.
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - fun(t+h, y_new)
        return self.solver.solve(implicit_eq, y)

class Trapezoidal(BackwardEuler):
    """
    Implements the Trapezoidal (Crank-Nicolson) integration method.
    
    Inherits the matrix handling from BackwardEuler.
    """
    
    def step(self, fun, t, y, h):
        """
        Take one Trapezoidal method step for the ODE.
        
        Formulates the implicit equation:
            A @ ((y_new - y) / h) - 0.5*(fun(t, y) + fun(t+h, y_new)) = 0
        and solves for y_new.
        
        Parameters:
            fun: callable
                The function defining the ODE.
            t: float
                The current time.
            y: np.array
                The current state.
            h: float
                The time step size.
                
        Returns:
            The updated state computed by the solver.
        """
        A_local = self._get_A(len(y))
        f_n = fun(t, y)  # Evaluate the ODE function at the current state
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - 0.5 * (f_n + fun(t+h, y_new))
        return self.solver.solve(implicit_eq, y)

class ThetaMethod(BackwardEuler):
    """
    Implements the Theta integration method, a generalization of Backward Euler and Trapezoidal methods.
    
    The method uses a parameter theta in [0, 1]:
      - theta = 1 gives Backward Euler,
      - theta = 0.5 gives Trapezoidal method.
    """
    
    def __init__(self, theta=0.5, **kwargs):
        """
        Initialize a ThetaMethod instance.
        
        Parameters:
            theta: float
                The weighting parameter between 0 and 1.
            **kwargs:
                Additional keyword arguments passed to the BackwardEuler initializer.
                
        Raises:
            ValueError: if theta is not in the interval [0, 1].
        """
        super().__init__(**kwargs)
        if not (0 <= theta <= 1):
            raise ValueError("Theta must be between 0 and 1")
        self.theta = theta

    def step(self, fun, t, y, h):
        """
        Take one Theta method step for the ODE.
        
        Formulates the implicit equation:
            A @ ((y_new - y) / h) - (theta * fun(t+h, y_new) + (1-theta)*fun(t, y)) = 0
        and solves for y_new.
        
        Parameters:
            fun: callable
                The function defining the ODE.
            t: float
                The current time.
            y: np.array
                The current state.
            h: float
                The time step size.
                
        Returns:
            The updated state computed by the solver.
        """
        A_local = self._get_A(len(y))
        f_n = fun(t, y)
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - (
            self.theta * fun(t+h, y_new) + (1 - self.theta) * f_n
        )
        return self.solver.solve(implicit_eq, y)

class CompositeMethod(IntegrationMethod):
    """
    Implements a composite integration method that combines two steps:
      1. A half-step using the Trapezoidal method.
      2. A full step using a modified Backward Euler method.
      
    The composite method first advances the solution halfway in time, then uses this intermediate
    value to compute the final step.
    """
    
    def __init__(self, a=1.0, solver=None, A=None):
        """
        Initialize the CompositeMethod.
        
        Parameters:
            a: float, optional
                A parameter that may be used for weighting (currently not used in the implementation).
            solver: ImplicitEquationSolver, optional
                The solver used to solve the implicit equations.
            A: np.array, optional
                The matrix used in the formulation. If None, identity is used.
        """
        self.solver = solver or ImplicitEquationSolver(method='root')
        # Create instances for sub-steps using Trapezoidal and Backward Euler methods.
        self.trapezoidal = Trapezoidal(solver=self.solver, A=A)
        self.backward_euler = BackwardEuler(solver=self.solver, A=A)

    def step(self, fun, t, y, h):
        """
        Take one composite integration step for the ODE.
        
        The composite step is broken into two parts:
          1. Compute an intermediate solution y_half at t + h/2 using the Trapezoidal method.
          2. Compute the final solution y_new at t + h using a Backward Euler step with a specific implicit equation:
             (3*y_new - 4*y_half + y)/h - fun(t+h, y_new) = 0
        
        If the first sub-step fails (i.e., the solver does not converge), the method returns early.
        
        Parameters:
            fun: callable
                The function defining the ODE.
            t: float
                The current time.
            y: np.array
                The current state.
            h: float
                The time step size.
                
        Returns:
            tuple: (y_new, Fk_new, err_new, overall_success, total_iters)
                y_new: The updated state.
                Fk_new: The residual or function value from the second solver.
                err_new: The error estimate from the second solver.
                overall_success: Boolean flag indicating if both sub-steps were successful.
                total_iters: Combined iteration count from both sub-steps.
        """
        half_h = 0.5 * h

        # First half-step with the Trapezoidal method.
        y_half, Fk_half, err_half, success_half, iters_half = \
            self.trapezoidal.step(fun, t, y, half_h)
        if not success_half:
            # Return early if the half-step fails.
            return y, Fk_half, err_half, False, iters_half  

        # Define the implicit equation for the second step (Backward Euler type):
        # (3*y_new - 4*y_half + y)/h - fun(t+h, y_new) = 0
        def implicit_eq(y_new):
            A_local = self.backward_euler._get_A(len(y))  # or from self.trapezoidal._get_A(len(y))
            return A_local @ ((3.0 * y_new - 4.0 * y_half + y) / h) - fun(t + h, y_new)
        
        # Solve for y_new starting from the intermediate solution y_half.
        y_new, Fk_new, err_new, success_new, iters_new = \
            self.backward_euler.solver.solve(implicit_eq, y_half)

        # Combine iteration counts and determine overall success.
        total_iters = iters_half + iters_new
        overall_success = success_half and success_new
        
        # Return the final result along with diagnostics.
        return (y_new, Fk_new, err_new, overall_success, total_iters)



# class BDFMethod(IntegrationMethod):
#     # Precompute coefficients for all supported orders
#     _BDF_COEFFS = {
#         1: {'alpha': [1.0, -1.0], 'c': 1.0, 'l': [1.0, 1.0]},
#         2: {'alpha': [1.5, -2.0, 0.5], 'c': 2/3, 'l': [1.0, 1.0, 0.5]},
#         3: {'alpha': [11/6, -3.0, 1.5, -1/3], 'c': 6/11, 'l': [1.0, 1.0, 0.5, 1/6]},
#         4: {'alpha': [25/12, -4.0, 3.0, -4/3, 0.25], 'c': 12/25, 'l': [1.0, 1.0, 0.5, 1/6, 1/24]},
#         5: {'alpha': [137/60, -5.0, 5.0, -10/3, 1.25, -0.2], 'c': 60/137, 
#             'l': [1.0, 1.0, 0.5, 1/6, 1/24, 1/120]}
#     }
    
#     def __init__(self, solver=None, max_order=5, A=None, atol=1e-6, rtol=1e-3):
#         self.solver = solver or ImplicitEquationSolver(method='newton')
#         self.max_order = min(max_order, 5)
#         self.A = A
#         self.use_identity = A is None
#         self.atol = atol
#         self.rtol = rtol
#         self.nordsieck = None
#         self.current_order = 1
#         self.h = None
#         self.t = None

#     def _predict_nordsieck(self, order):
#         """Vectorized prediction using precomputed coefficients"""
#         coeffs = self._BDF_COEFFS[order]['l']
#         return sum(c * z for c, z in zip(coeffs, self.nordsieck[:order+1]))

#     # ... [rest of BDF implementation with vectorized operations] ...