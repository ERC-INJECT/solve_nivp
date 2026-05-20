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
``solve_nivp`` wraps construction of a projection, nonlinear solver and
integration method and returns the integrated time grid, states and diagnostic
information.

Low-level workflow
------------------
1. Instantiate a projection (e.g. :class:`CoulombProjection`).
2. Create an :class:`ImplicitEquationSolver` with that projection.
3. Pick an integration method (``BackwardEuler``, ``Trapezoidal``, ``ThetaMethod``,
   ``CompositeMethod``, ``EmbeddedBETR``, ``SDIRK2``) and pass the solver instance.
4. Build an :class:`ODESystem` specifying your RHS ``fun(t, y)`` and options.
5. Drive the time loop with :class:`ODESolver` or let ``solve_nivp`` do it.

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
>>> from solve_nivp import solve_nivp, CoulombProjection
>>> def rhs(t, y):
...     return -y  # simple stable linear test
>>> t, y, h, fk, info = solve_nivp(rhs, (0.0, 1.0), y0=np.array([1.0]), method='backward_euler')
>>> y[-1]
array([0.3679])  # ~ exp(-1)

See the Sphinx documentation (``docs/``) for extended examples.
"""

import numpy as np

__version__ = "0.2.0.dev1"

from .projections import (
  Projection,
  CoulombProjection,
  SignProjection,
  IdentityProjection,
  GeneralMoreauVIProjection,
  MuScaledSOCProjection,
  MoreauSOCProjection,
  AnisotropicSOCProjection,
  AlgebraicConstraintProjection,
  CompositeContactProjection,
)
from .solvers.nonlinear_solvers import ImplicitEquationSolver, UMFPACK_AVAILABLE, PETSC_AVAILABLE
from .integrations import BackwardEuler, BackwardEulerSchur, RadauIIASchur, Trapezoidal, ThetaMethod, CompositeMethod, EmbeddedBETR, SDIRK2, RadauIIA  # , BDFMethod
from .solvers.block_system import SchurComplementSolver, BlockStructuredSystem
from .ODESystem import ODESystem
from .ODESolver import ODESolver
from .contact import build_impulse_contact, ContactSystem
from .alart_curnier_contact import (
  build_alart_curnier_contact,
  build_dynamic_alart_curnier_contact,
)
from .ncp_contact import (
  build_ncp_contact,
  build_dynamic_ncp_contact,
  build_ncp_contact_blocked,
)
from .desaxce_contact import (
  build_dynamic_desaxce_contact,
  build_dynamic_desaxce_projected_contact,
  build_dynamic_desaxce_residual_contact,
)
from .rattle_contact import (
  build_dynamic_rattle_contact,
  solve_dynamic_rattle_contact,
  build_rattle_system,
  RattleMechanicalSystem,
  RattleContactSpec,
  RattleBilateralSpec,
  RattleAlgebraicSpec,
  RattleContactSystem,
  RattleSolveResult,
  RattleSolver,
)

_STABLE_PUBLIC_API = [
  '__version__',
  'solve_nivp',
  # Core system / driver
  'ODESystem', 'ODESolver',
  # Nonlinear solver
  'ImplicitEquationSolver',
  # Integrators
  'BackwardEuler', 'BackwardEulerSchur', 'RadauIIASchur', 'Trapezoidal', 'ThetaMethod', 'CompositeMethod', 'EmbeddedBETR', 'SDIRK2', 'RadauIIA',
  # Projections
  'Projection',
  'CoulombProjection', 'SignProjection', 'IdentityProjection',
  'GeneralMoreauVIProjection', 'MuScaledSOCProjection', 'MoreauSOCProjection',
  'AnisotropicSOCProjection', 'AlgebraicConstraintProjection',
  'CompositeContactProjection',
]

_EXPERIMENTAL_PUBLIC_API = [
  # Schur and block helpers
  'BackwardEulerSchur', 'RadauIIASchur',
  'SchurComplementSolver', 'BlockStructuredSystem',
  # Contact helpers
  'build_impulse_contact', 'build_alart_curnier_contact',
  'build_dynamic_alart_curnier_contact', 'build_ncp_contact',
  'build_dynamic_ncp_contact', 'build_ncp_contact_blocked',
  'build_dynamic_desaxce_contact',
  'build_dynamic_desaxce_projected_contact',
  'build_dynamic_desaxce_residual_contact',
  'build_dynamic_rattle_contact',
  'solve_dynamic_rattle_contact',
  'build_rattle_system',
  'ContactSystem',
  'RattleMechanicalSystem',
  'RattleContactSpec',
  'RattleBilateralSpec',
  'RattleAlgebraicSpec',
  'RattleContactSystem',
  'RattleSolveResult',
  'RattleSolver',
]

# Curated top-level API.  Experimental names remain importable for
# compatibility; docs/source/public_api.rst defines the support policy.
__all__ = _STABLE_PUBLIC_API + _EXPERIMENTAL_PUBLIC_API


def solve_nivp(
  fun,
  t_span,
  y0,
  method='composite',
  projection='identity',
  solver='VI',
  projection_opts=None,
  solver_opts=None,
  integrator_opts=None,
  adaptive_opts=None,
  adaptive=True,
  atol=1e-6,
  rtol=1e-3,
  nl_atol=None,
  nl_rtol=None,
  h0=1e-2,
  component_slices=None,
  verbose=False,
  A=None,
  skip_error_indices=None,
  return_attempts=False,
  dae_var_weight='auto',
  thin_output=1,
  store_fk=True,
  gc_interval=0,
  abort_on_fixed_failure=True,
  jacobian_scaling=None,
  active_set_filter=False,
  t_eval=None,
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
    ``'composite'`` (TR-BE like second order), ``'embedded_betr'``, ``'sdirk2'``,
    ``'radau_iia'`` (L-stable, stiffly accurate, order 3 or 5; stages controlled
    via ``integrator_opts={'stages': 2}`` or ``{'stages': 3}``).
  projection : str or Projection or None, default 'identity'
    Name of projection to build: ``'coulomb'``, ``'sign'``, ``'identity'`` or
    ``None``. At the high-level API, ``None`` is promoted to the identity
    projection so the smooth unconstrained path works out of the box.
  solver : str, default 'VI'
    Nonlinear solve strategy per implicit step: ``'VI'`` or ``'semismooth_newton'``.
  projection_opts : dict or None
    Keyword arguments forwarded to the projection constructor.
  solver_opts : dict or None
    Keyword arguments forwarded to :class:`ImplicitEquationSolver` (e.g.
    ``tol``, ``gmres_tol``, ``eisenstat_c``...). If ``rhs_jac`` or
    ``fun_jacobian`` is present it is used as an analytical Jacobian.
    For large sparse problems, ``jacobian_sparsity`` can be supplied to
    enable colored finite-difference Jacobian approximation when an
    analytical Jacobian is not available.
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
  atol, rtol : float or array_like
    Absolute / relative tolerances for the adaptive controller.  A scalar is
    broadcast to every DOF; an array of length ``len(component_slices)`` is
    expanded per block; a full per-DOF array is used directly.
  nl_atol, nl_rtol : float, array_like, or None
    Per-DOF absolute / relative tolerances for the nonlinear solver
    convergence test (weighted RMS norm).  When ``None`` (default) the
    nonlinear solver falls back to its scalar ``tol`` parameter.
  h0 : float, str, or None, default 1e-2
    Initial step size guess.  Set to ``None`` or ``'auto'`` to enable the
    Hairer-Wanner automatic initial step-size estimator.
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
  dae_var_weight : str, default 'auto'
    DAE-aware error weighting.  ``'auto'`` / ``'exclude'`` detects algebraic
    DOFs from the mass matrix (zero rows) and excludes them from the error
    norm à la SUNDIALS IDA.  ``'include'`` keeps all DOFs in the norm
    (traditional behaviour).
  thin_output : int, default 1
    Store only every *N*-th accepted step in the returned history arrays.
    First and last steps are always stored.  Set to a value > 1 for large
    problems where storing every state vector is prohibitive.
  store_fk : bool, default True
    Whether to keep per-step residual vectors in the returned ``fk`` array.
    Setting ``False`` saves ~1× state-vector memory per stored step.
  gc_interval : int, default 0
    Call ``gc.collect()`` every *N* accepted steps to free unreferenced
    objects (stale sparse factorisations, etc.).  0 disables.
  abort_on_fixed_failure : bool, default True
    In fixed-step mode, stop at the first nonlinear failure instead of
    continuing with the failed state. Adaptive stepping is unchanged.
  jacobian_scaling : str, default 'none'
    Row / column equilibration of the Newton Jacobian before each linear
    solve, improving conditioning for saddle-point systems with disparate
    scales (e.g. mixed physics with large spring constants).

    * ``'none'``  — no scaling (default).
    * ``'row'``   — row equilibration: normalises each row infinity-norm
      to 1.  Sufficient for direct solvers (SPLU, MUMPS).
    * ``'ruiz'``  — Ruiz iterative symmetric scaling (5 iterations):
      simultaneously normalises row **and** column infinity-norms.
      Better for iterative solvers (GMRES+ILU).

  active_set_filter : bool, default False
    When *True*, DOFs whose contact regime changed during a step
    (stick↔slip, contact↔separation) are excluded from the adaptive
    error norm.  This prevents the embedded / Richardson error estimator
    from overreacting to the discontinuous constraint-force jumps that
    are intrinsic to nonsmooth event-capturing integrators, allowing the
    step-size controller to maintain larger steps through transitions.
    Requires a projection with regime tracking (e.g.
    :class:`MuScaledSOCProjection` or :class:`CompositeContactProjection`).
    Has no effect when no projection is present or regime tracking is
    not available.

  Returns
  -------
  t : ndarray, shape (m,)
    Monotone sequence of time points including ``t0`` and final time.
  y : ndarray, shape (m, n)
    State history; ``y[i]`` corresponds to time ``t[i]``.
  h : ndarray, shape (m,)
    Stored step-size history. ``h[0]`` is the initial guess; later entries are
    the accepted step sizes for the stored states.
  fk : object ndarray, shape (m-1,)
    Residual / implicit function evaluations for the stored accepted steps.
  info : list of tuple
    Nonlinear-solver diagnostics for the stored accepted steps. In fixed-step
    mode a terminal failure tuple is appended even if the failed state is not
    stored.
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
  if projection is None:
    # The public API treats "no projection specified" as the smooth identity
    # map so the default solver configuration is immediately usable.
    proj_instance = IdentityProjection()
  else:
    # Accept a pre-built Projection instance directly (bypass string lookup)
    if isinstance(projection, Projection):
      proj_instance = projection
    elif isinstance(projection, str):
      p = projection.lower()
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
      elif p == 'algebraic':
        proj_instance = AlgebraicConstraintProjection(**projection_opts)
      else:
        raise ValueError(f"Unknown projection: {projection}")
    else:
      raise TypeError(
        f"projection must be a string or Projection instance, got {type(projection).__name__}")

  # 2) Nonlinear solver
  # Filter out keys not accepted by ImplicitEquationSolver.__init__
  _solver_opts = dict(solver_opts) if solver_opts is not None else {}
  rhs_jac = _solver_opts.pop('rhs_jac', None) or _solver_opts.pop('fun_jacobian', None)

  # Merge jacobian_scaling: top-level explicit wins over solver_opts value.
  _js_from_opts = _solver_opts.pop('jacobian_scaling', None)
  if jacobian_scaling is None:
    jacobian_scaling = _js_from_opts if _js_from_opts is not None else 'none'

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
    nl_atol=nl_atol,
    nl_rtol=nl_rtol,
    jacobian_scaling=jacobian_scaling,
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
  elif m == 'sdirk2':
    integrator = SDIRK2(solver=solver_instance, A=A, **_integrator_opts)
  elif m in ('radau_iia', 'radau'):
    integrator = RadauIIA(solver=solver_instance, A=A, **_integrator_opts)
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
      # Forward DAE-aware error weighting setting
      stepper._dae_var_weight_mode = str(dae_var_weight).lower().strip()
      stepper._dae_mask = None  # reset so it re-detects

      # Forward active-set filter setting
      stepper.active_set_filter = bool(active_set_filter)

      # Handle auto-h0 from top level
      if h0 is None or (isinstance(h0, str) and str(h0).lower() == 'auto'):
        stepper._auto_h0 = True
        initial_h = stepper.h_min   # placeholder; real h comes from estimator
      else:
        stepper._auto_h0 = False
        initial_h = float(h0)
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
      # atol / rtol may be array_like – pass through without float cast
      if 'atol' in adaptive_opts:
        stepper.atol = adaptive_opts['atol']
      if 'rtol' in adaptive_opts:
        stepper.rtol = adaptive_opts['rtol']
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
        ao_h0 = adaptive_opts['h0']
        if ao_h0 is None or (isinstance(ao_h0, str) and str(ao_h0).lower() == 'auto'):
          stepper._auto_h0 = True
          initial_h = stepper.h_min
        else:
          try:
            initial_h = float(ao_h0)
          except Exception:
            pass

  # Ensure initial_h is numeric before passing to ODESolver
  if initial_h is None or isinstance(initial_h, str):
    initial_h = 1e-2   # safe fallback

  # 5) Integrate
  solver_obj = ODESolver(
    system, t_span, h=initial_h,
    thin_output=thin_output,
    store_fk=store_fk,
    gc_interval=gc_interval,
    abort_on_fixed_failure=abort_on_fixed_failure,
    t_eval=t_eval,
  )
  return solver_obj.solve(return_attempts=return_attempts)


# Backward-compatible alias for older user code and examples.
solve_ivp_ns = solve_nivp
