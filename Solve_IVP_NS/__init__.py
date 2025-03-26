# In Solve_IVP_NS/__init__.py or a separate file like Solve_IVP_NS/api.py:

import numpy as np
from .projections import CoulombProjection, SignProjection, IdentityProjection
from .nonlinear_solvers import ImplicitEquationSolver
from .integrations import BackwardEuler, Trapezoidal, ThetaMethod, CompositeMethod #, BDFMethod
from .ODESystem import ODESystem
from .ODESolver import ODESolver

def solve_ivp_ns(
    fun,
    t_span,
    y0,
    method='composite',
    projection=None,
    solver='VI',
    projection_opts=None,
    solver_opts=None,
    adaptive=True,
    atol=1e-6,
    rtol=1e-3,
    h0=1e-2,
    component_slices=None,
    verbose=False,
    A=None
):
    """
    A high-level function to solve an ODE with optional projection and custom solver settings.
    
    Parameters:
      fun : callable
        ODE function fun(t, y).
      t_span : (float, float)
        Start and end times.
      y0 : array_like
        Initial condition.
      method : str
        Integration method name, e.g. 'composite', 'backward_euler', 'bdf', etc.
      projection : str or None
        Projection type, e.g. 'coulomb', 'sign', 'identity', or None for no projection.
      solver : str
        Nonlinear solver method, e.g. 'VI', 'newton_raphson', 'root', etc.
      projection_opts : dict
        Extra keyword arguments for the projection class (e.g. friction parameters).
      solver_opts : dict
        Extra keyword arguments for the ImplicitEquationSolver.
      adaptive : bool
        Whether to use adaptive stepping.
      atol, rtol : float
        Tolerances for adaptive stepping.
      h0 : float
        Initial step size.
      component_slices : list of slices or None
        For partitioning the state vector.
      verbose : bool
        Print debug info if True.
    
    Returns:
      (t_vals, y_vals, h_vals, fk_vals, solver_info)
        Arrays of time, solution, step sizes, residuals, and solver diagnostics.
    """
    if projection_opts is None:
        projection_opts = {}
    if solver_opts is None:
        solver_opts = {}

    # 1) Create the projection (if any)
    proj_instance = None
    if projection is not None:
        if projection.lower() == 'coulomb':
            proj_instance = CoulombProjection(**projection_opts)
        elif projection.lower() == 'sign':
            proj_instance = SignProjection(**projection_opts)
        elif projection.lower() == 'identity':
            proj_instance = IdentityProjection()
        else:
            raise ValueError(f"Unknown projection: {projection}")

    # 2) Create the nonlinear solver
    solver_instance = ImplicitEquationSolver(
        method=solver,
        proj=proj_instance,
        component_slices=component_slices,
        **solver_opts
    )

    # 3) Choose the integration method
    if method.lower() == 'backward_euler':
        integrator = BackwardEuler(solver=solver_instance, A = A)
    elif method.lower() == 'trapezoidal':
        integrator = Trapezoidal(solver=solver_instance, A = A)
    elif method.lower() == 'theta':
        integrator = ThetaMethod(theta=0.5, solver=solver_instance, A = A)
    elif method.lower() == 'composite':
        integrator = CompositeMethod(solver=solver_instance, A = A)
    # elif method.lower() == 'bdf':
    #     integrator = BDFMethod(solver=solver_instance, atol=atol, rtol=rtol)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 4) Create the ODE system
    system = ODESystem(
        fun=fun,
        y0=y0,
        method=integrator,
        adaptive=adaptive,
        atol=atol,
        rtol=rtol,
        component_slices=component_slices,
        verbose=verbose
    )

    # 5) Create the solver and solve
    solver_obj = ODESolver(system, t_span, h=h0)
    return solver_obj.solve()
