"""solve_nivp: Nonsmooth implicit IVP / (simple) DAE solver tooling.

This package provides building blocks for integrating systems of the form::

    A dy/dt = f(t, y)   (possibly A = I) ,

optionally coupled with nonsmooth complementarity / frictional style relations
expressed through projection operators. Two nonlinear solution strategies are
available each implicit step:

* ``semismooth_newton`` (projection based semismooth Newton using generalized
  Jacobians) 
* ``VI`` (fixed-point style projected iteration / variational inequality map)

High-level entry point
----------------------
``solve_ivp_ns`` wraps construction of a projection, nonlinear solver and
integration method and returns the integrated time grid, states and diagnostic
information.

Low-level workflow
------------------
1. Instantiate a projection (e.g. :class:`CoulombProjection`).
2. Create an :class:`ImplicitEquationSolver` with that projection.
3. Pick an integration method (``BackwardEuler``, ``Trapezoidal``, ``ThetaMethod``,
   ``CompositeMethod``, ``EmbeddedBETR``) and pass the solver instance.
4. Build an :class:`ODESystem` specifying your RHS ``fun(t, y)`` and options.
5. Drive the time loop with :class:`ODESolver` or let ``solve_ivp_ns`` do it.

Returned residual / fk semantics
--------------------------------
Throughout the package ``Fk`` (or ``fk`` at the solver level) denotes the raw
implicit residual function evaluation *before* projection for the particular
implicit equation of the step, i.e. the value ``F(y_k)`` being driven to zero by
the nonlinear solver. For the projection based methods this is typically the
implicit equation residual (e.g. Backward Euler) not the projected gap.

Quick start
-----------
>>> import numpy as np
>>> from solve_nivp import solve_ivp_ns, CoulombProjection
>>> def rhs(t, y):
...     return -y  # simple stable linear test
>>> t, y, h, fk, info = solve_ivp_ns(rhs, (0.0, 1.0), y0=np.array([1.0]), method='backward_euler')
>>> y[-1]
array([0.3679])  # ~ exp(-1)

See the Sphinx documentation (``docs/``) for extended examples.
"""

import numpy as np
from .projections import (
  CoulombProjection,
  SignProjection,
  IdentityProjection,
  GeneralMoreauVIProjection,   # ← add this
  MuScaledSOCProjection
)
from .nonlinear_solvers import ImplicitEquationSolver
from .integrations import BackwardEuler, Trapezoidal, ThetaMethod, CompositeMethod, EmbeddedBETR  # , BDFMethod
from .ODESystem import ODESystem
from .ODESolver import ODESolver

# Curated public API
__all__ = [
  'solve_ivp_ns',
  # Core system / driver
  'ODESystem', 'ODESolver',
  # Nonlinear solver
  'ImplicitEquationSolver',
  # Integrators
  'BackwardEuler', 'Trapezoidal', 'ThetaMethod', 'CompositeMethod', 'EmbeddedBETR',
  # Projections
  'CoulombProjection', 'SignProjection', 'IdentityProjection','GeneralMoreauVIProjection'
]


def solve_ivp_ns(
  fun,
  t_span,
  y0,
  method='composite',
  projection=None,
  solver='VI',
  projection_opts=None,
  solver_opts=None,
  integrator_opts=None,
  adaptive_opts=None,
  adaptive=True,
  atol=1e-6,
  rtol=1e-3,
  h0=1e-2,
  component_slices=None,
  verbose=False,
  A=None,
  skip_error_indices=None,
  return_attempts=False,
):
  """Integrate an ODE / simple index–1 DAE with optional nonsmooth projection.

  Parameters
  ----------
  fun : callable
    Right-hand side ``fun(t, y) -> ndarray`` (broadcast / vector valued). A third
    argument ``Fk`` is tolerated (``fun(t, y, Fk)``) and ignored if supplied.
  t_span : (float, float)
    Time interval ``(t0, tf)`` to integrate over.
  y0 : array_like, shape (n,)
    Initial state.
  method : str, default 'composite'
    Time stepping scheme: ``'backward_euler'``, ``'trapezoidal'``, ``'theta'``,
    ``'composite'`` (TR-BE like second order), ``'embedded_betr'``.
  projection : str or None, default None
    Name of projection to build: ``'coulomb'``, ``'sign'``, ``'identity'`` or
    ``None`` for no projection (only meaningful if solver supports that path).
  solver : str, default 'VI'
    Nonlinear solve strategy per implicit step: ``'VI'`` or ``'semismooth_newton'``.
  projection_opts : dict or None
    Keyword arguments forwarded to the projection constructor.
  solver_opts : dict or None
    Keyword arguments forwarded to :class:`ImplicitEquationSolver` (e.g.
    ``tol``, ``gmres_tol``, ``eisenstat_c``...). If ``rhs_jac`` or
    ``fun_jacobian`` is present it is used as an analytical Jacobian.
  integrator_opts : dict or None
    Optional keyword arguments forwarded to the integration method
    constructor (e.g. ``pass_prev_state=True``, ``pass_step_size=True``).
  adaptive_opts : dict or None
    Optional controls for the adaptive stepper. Recognized keys include
    ``h_min``, ``h_max``, ``h_up``, ``h_down``, ``safety``, ``use_PI``,
    ``method_order`` (alias ``p``), ``atol``, ``rtol``, and
    ``skip_error_indices``. ``h0`` here (if provided) overrides the top-level
    ``h0`` for the initial step guess. Unrecognized keys are ignored.
  adaptive : bool, default True
    Enable Richardson extrapolation based adaptive step size control.
  atol, rtol : float
    Absolute / relative tolerances for the adaptive controller.
  h0 : float, default 1e-2
    Initial step size guess.
  component_slices : list[slice] or None
    Optional partition of the state for block error control and projections.
  verbose : bool, default False
    Print basic diagnostics (mainly adaptive rejection messages).
  A : ndarray or None
    Optional constant mass / descriptor matrix. If ``None`` identity is assumed.
  skip_error_indices : iterable[int] or None
    Indices (w.r.t. ``component_slices`` order) to exclude from adaptive error norm
    (useful for algebraic / projected-only components).
  return_attempts : bool, default False
    When ``True`` (and adaptive stepping is enabled) capture every attempted
    step size along with acceptance, error estimate, and reason. Returning this
    diagnostic data introduces minor overhead and is therefore opt-in.

  Returns
  -------
  t : ndarray, shape (m,)
    Monotone sequence of time points including ``t0`` and final time.
  y : ndarray, shape (m, n)
    State history; ``y[i]`` corresponds to time ``t[i]``.
  h : ndarray, shape (m,)
    Step sizes actually used (first entry is the initial guess).
  fk : object ndarray, shape (m,)
    Residual / implicit function evaluations associated with accepted steps.
  info : list of tuple
    Per-step diagnostics: ``(solver_error, success, iterations)``.
  attempts : dict or None, optional
    Only returned when ``return_attempts`` is True. Contains arrays describing
    each attempted adaptive step (time, proposed ``h``, accepted flag, etc.).

  Notes
  -----
  This helper builds internal objects but does not retain them; for more
  granular control (e.g. custom restart, continuing integration) construct
  :class:`ODESystem` and :class:`ODESolver` directly.
  """
  if projection_opts is None:
    projection_opts = {}
  if solver_opts is None:
    solver_opts = {}
  if integrator_opts is None:
    integrator_opts = {}
  if adaptive_opts is None:
    adaptive_opts = {}

  # 1) Projection instance
  proj_instance = None
  if projection is not None:
    p = projection.lower()
    print(p)
    if p == 'coulomb':
      proj_instance = CoulombProjection(**projection_opts)
    elif p == 'sign':
      proj_instance = SignProjection(**projection_opts)
    elif p == 'identity':
      proj_instance = IdentityProjection()
    elif p == 'unilateral':
      proj_instance = GeneralMoreauVIProjection(**projection_opts)
    elif p == 'soccp':
      proj_instance = MuScaledSOCProjection(**projection_opts)
    else:
      raise ValueError(f"Unknown projection: {projection}")

  # 2) Nonlinear solver
  # Filter out keys not accepted by ImplicitEquationSolver.__init__
  _solver_opts = dict(solver_opts) if solver_opts is not None else {}
  rhs_jac = _solver_opts.pop('rhs_jac', None) or _solver_opts.pop('fun_jacobian', None)

  # Provide a sensible default component_slices for VI if not supplied
  if isinstance(solver, str) and solver.lower() == 'vi' and component_slices is None:
    try:
      n0 = int(np.atleast_1d(y0).shape[0])
      component_slices = [slice(0, n0)]
    except Exception:
      component_slices = None

  solver_instance = ImplicitEquationSolver(
    method=solver,
    proj=proj_instance,
    component_slices=component_slices,
    **_solver_opts,
  )

  # Optional: attach analytical RHS Jacobian (rhs_jac(t, y) -> df/dy)
  if callable(rhs_jac):
    setattr(solver_instance, 'rhs_jacobian', rhs_jac)

  # 3) Integration method
  m = method.lower()
  # Filter out reserved ctor keys to avoid duplication
  _integrator_opts = dict(integrator_opts) if integrator_opts is not None else {}
  for reserved in ('solver', 'A'):
    _integrator_opts.pop(reserved, None)

  if m == 'backward_euler':
    integrator = BackwardEuler(solver=solver_instance, A=A, **_integrator_opts)
  elif m == 'trapezoidal':
    integrator = Trapezoidal(solver=solver_instance, A=A, **_integrator_opts)
  elif m == 'theta':
    integrator = ThetaMethod(theta=0.5, solver=solver_instance, A=A, **_integrator_opts)
  elif m == 'composite':
    integrator = CompositeMethod(solver=solver_instance, A=A, **_integrator_opts)
  elif m == 'embedded_betr':
    integrator = EmbeddedBETR(solver=solver_instance, A=A)
  # elif m == 'bdf':
  #     integrator = BDFMethod(solver=solver_instance, atol=atol, rtol=rtol)
  else:
    raise ValueError(f"Unknown method: {method}")

  # 4) ODE system assembly
  system = ODESystem(
    fun=fun,
    y0=y0,
    method=integrator,
    adaptive=adaptive,
    atol=atol,
    rtol=rtol,
    component_slices=component_slices,
    verbose=verbose,
    record_attempts=return_attempts,
  )

  # Optionally tune the adaptive controller
  initial_h = h0
  if adaptive:
    stepper = getattr(system, 'adaptive_stepper', None)
    if stepper is not None:
      # Merge/override supported scalar options
      def _set(name, cast=float):
        if name in adaptive_opts:
          try:
            setattr(stepper, name, cast(adaptive_opts[name]))
          except Exception:
            pass

      for key in ('h_min', 'h_max', 'h_up', 'h_down', 'safety'):
        _set(key, float)
      _set('use_PI', bool)
      _set('atol', float)
      _set('rtol', float)
      _set('verbose', bool)

      # Ratio / digital-filter controller knobs
      _set('mode', lambda v: str(v))
      _set('controller', lambda v: str(v))
      _set('b_param', float)
      _set('r_min', float)
      _set('r_max', float)
      _set('reject_reboot_thresh', int)

      # Method order (alias 'p') needs alpha/beta refresh
      mo = None
      if 'method_order' in adaptive_opts:
        mo = adaptive_opts.get('method_order')
      elif 'p' in adaptive_opts:
        mo = adaptive_opts.get('p')
      if mo is not None:
        try:
          stepper.p = int(mo)
          stepper._alpha = 0.7 / (stepper.p + 1.0)
          stepper._beta = 0.4 / (stepper.p + 1.0)
        except Exception:
          pass

      # Skip error indices: merge top-level and adaptive_opts
      merged_skip = set()
      try:
        if 'skip_error_indices' in adaptive_opts and adaptive_opts['skip_error_indices'] is not None:
          merged_skip |= set(adaptive_opts['skip_error_indices'])
      except Exception:
        pass
      try:
        if skip_error_indices is not None:
          merged_skip |= set(skip_error_indices)
      except Exception:
        pass
      if merged_skip:
        try:
          stepper.skip_error_indices = set(merged_skip)
        except Exception:
          pass

      # Optional 'h0' override inside adaptive_opts
      if 'h0' in adaptive_opts:
        try:
          initial_h = float(adaptive_opts['h0'])
        except Exception:
          pass

  # 5) Integrate
  solver_obj = ODESolver(system, t_span, h=initial_h)
  return solver_obj.solve(return_attempts=return_attempts)
