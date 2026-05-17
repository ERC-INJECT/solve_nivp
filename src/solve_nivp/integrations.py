import inspect
import math
import numpy as np
from abc import ABC, abstractmethod
import scipy.sparse as sp
from .solvers.nonlinear_solvers import ImplicitEquationSolver  # Relative import for a solver class


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


def _call_projection_with_context(
    projection,
    current_state,
    candidate,
    *,
    rhok=None,
    t=None,
    Fk_val=None,
    prev_state=None,
    step_size=None,
):
    """Call ``projection.project`` while passing only supported context."""
    params = inspect.signature(projection.project).parameters
    kwargs = {}
    if "rhok" in params:
        kwargs["rhok"] = rhok
    elif "rho" in params:
        kwargs["rho"] = rhok
    if "t" in params:
        kwargs["t"] = t
    if "Fk_val" in params:
        kwargs["Fk_val"] = Fk_val
    if "prev_state" in params:
        kwargs["prev_state"] = prev_state
    if "step_size" in params:
        kwargs["step_size"] = step_size
    return projection.project(current_state, candidate, **kwargs)


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
    # Keys: ('dense'| 'csr', n)
    _ID_CACHE = {}

    def __init__(
        self,
        solver=None,
        A=None,
        pass_prev_state=False,
        pass_step_size=False,
        post_step_projection=None,
        post_step_rhok=1.0,
    ):
        """
        Initialize a Backward Euler integration method.

        Parameters:
            solver: ImplicitEquationSolver, optional
                The solver to use for solving the implicit equation. Defaults to using
                an ImplicitEquationSolver with method 'semismooth_newton'.
            A: np.array, optional
                The matrix used in the formulation. If not provided, identity matrix is used.
            pass_prev_state: bool, optional
                When True, the previously accepted state ``y`` will be supplied (when
                supported by the callable) as an additional argument to both the RHS
                function and its Jacobian.
            pass_step_size: bool, optional
                When True, the step size ``h`` for the current implicit solve will be
                forwarded to the RHS and Jacobian callables (when their signatures
                accept it).
        """
        self.solver = solver or ImplicitEquationSolver(method='semismooth_newton')
        self.A = A
        self.use_identity = (A is None)
        self.pass_prev_state = pass_prev_state
        self.pass_step_size = pass_step_size
        self.post_step_projection = post_step_projection
        self.post_step_rhok = post_step_rhok
        # Method order (for adaptive controllers)
        self.order = 1
        # Per-instance caches for bound call wrappers to avoid repeated try/except dispatch
        # Keys are tuples of (id(func), has_prev, has_h)
        self._fun_bindings = {}
        self._jac_bindings = {}

    def _get_bound_wrapper(self, func, has_prev, has_h, cache):
        """Bind a lightweight caller for func based on available arguments and cache it.

        Returns a callable wrapper(tt, yy, fk_val, prev_state, step_size) -> result
        that uses a fixed argument order discovered once.
        """
        key = (id(func), bool(has_prev), bool(has_h))
        wrapper = cache.get(key)
        if wrapper is not None:
            return wrapper

        # Candidate orders filtered by availability
        # Prefer t,y first; fall back to y-only variants
        orders = [
            ('t', 'y', 'prev', 'Fk', 'h'),
            ('t', 'y', 'prev', 'Fk'),
            ('t', 'y', 'prev', 'h'),
            ('t', 'y', 'Fk', 'h'),
            ('t', 'y', 'prev'),
            ('t', 'y', 'Fk'),
            ('t', 'y', 'h'),
            ('t', 'y'),
            ('y', 'prev', 'Fk', 'h'),
            ('y', 'prev', 'Fk'),
            ('y', 'prev', 'h'),
            ('y', 'Fk', 'h'),
            ('y', 'prev'),
            ('y', 'Fk'),
            ('y', 'h'),
            ('y',),
        ]

        def _build(order):
            def _call(tt, yy, fk, prev, h):
                # Skip labels not available
                args = []
                for lab in order:
                    if lab == 't':
                        if tt is None:
                            return _sentinel
                        args.append(tt)
                    elif lab == 'y':
                        if yy is None:
                            return _sentinel
                        args.append(yy)
                    elif lab == 'prev':
                        if not has_prev or prev is None:
                            return _sentinel
                        args.append(prev)
                    elif lab == 'Fk':
                        if fk is None:
                            return _sentinel
                        args.append(fk)
                    elif lab == 'h':
                        if not has_h or h is None:
                            return _sentinel
                        args.append(h)
                return func(*args)
            return _call

        _sentinel = object()
        # Probe using a real call at first use; we rely on the caller to give real values
        def resolve_and_cache(tt, yy, fk, prev, h):
            for order in orders:
                # quick availability filter
                if ('prev' in order and (not has_prev or prev is None)):
                    continue
                if ('h' in order and (not has_h or h is None)):
                    continue
                caller = _build(order)
                try:
                    out = caller(tt, yy, fk, prev, h)
                    if out is _sentinel:
                        continue
                    # Freeze wrapper with fixed order
                    def bound(tt2, yy2, fk2, prev2, h2, _order=order, _func=func):
                        args = []
                        for lab in _order:
                            if lab == 't':
                                args.append(tt2)
                            elif lab == 'y':
                                args.append(yy2)
                            elif lab == 'prev':
                                args.append(prev2)
                            elif lab == 'Fk':
                                args.append(fk2)
                            elif lab == 'h':
                                args.append(h2)
                        return _func(*args)
                    cache[key] = bound
                    return out
                except TypeError:
                    continue
            raise TypeError(
                "Unable to call {} with supported signatures.".format(getattr(func, '__name__', 'callable'))
            )

        # Return a wrapper that resolves on first call, then reuses cached bound
        def wrapper_first(tt, yy, fk, prev, h):
            # Attempt resolve; this stores the bound wrapper in cache
            out = resolve_and_cache(tt, yy, fk, prev, h)
            # Use cached for subsequent calls
            return out

        return wrapper_first

    def _get_A(self, n):
        """
        Retrieve or compute the appropriate matrix A of size (n x n).

        If no specific matrix A is provided during initialization, return an identity
        matrix. For large systems (as indicated by the solver's sparse preference),
        a CSR identity is returned to avoid allocating a dense n×n array.

        Returns a numpy.ndarray or scipy.sparse.csr_matrix matching the chosen path.
        """
        if not self.use_identity:
            return self.A

        # Prefer sparse identity for large n (same heuristic as solver sparse path if available)
        want_sparse = False
        try:
            want_sparse = bool(self.solver._sparse_active(n))  # may not exist for custom solvers
        except Exception:
            # Fallback heuristic using solver's threshold when present
            thr = getattr(self.solver, 'sparse_threshold', 200)
            try:
                want_sparse = (n >= int(thr))
            except Exception:
                want_sparse = (n >= 200)

        if want_sparse:
            key = ('csr', n)
            if key not in self._ID_CACHE:
                self._ID_CACHE[key] = sp.eye(n, format='csr')
            return self._ID_CACHE[key]
        else:
            key = ('dense', n)
            if key not in self._ID_CACHE:
                self._ID_CACHE[key] = np.eye(n)
            return self._ID_CACHE[key]

    def _refresh_solver_step_cache(self, h):
        """Swap / invalidate cached linear solver state when the step size changes."""
        _prev_h = getattr(self, '_cached_step_h', None)
        if _prev_h is not None and _prev_h != h:
            _cache = getattr(self, '_lu_h_cache', None)
            if _cache is None:
                self._lu_h_cache = _cache = {}
            _cache[_prev_h] = (
                self.solver._lu,
                getattr(self.solver, '_lu_shape', None),
                getattr(self.solver, '_lu_pattern', None),
                getattr(self.solver, '_lu_use_count', 0),
                getattr(self.solver, '_J_cross_call', None),
            )
            if len(_cache) > 2:
                _oldest = next(iter(_cache))
                del _cache[_oldest]

            _cached_state = _cache.get(h)
            if _cached_state is not None:
                (self.solver._lu,
                 self.solver._lu_shape,
                 self.solver._lu_pattern,
                 self.solver._lu_use_count,
                 self.solver._J_cross_call) = _cached_state
            else:
                self.solver._lu = None
                self.solver._lu_shape = None
                if hasattr(self.solver, '_J_cross_call'):
                    self.solver._J_cross_call = None
            self.solver._petsc_needs_matrix_update = True
        self._cached_step_h = h

    def _set_solver_lam_from_step(self, A_local, h, diag_factor=1.0):
        """Equilibrate the natural-map parameter with the implicit diagonal."""
        try:
            if getattr(self.solver, 'method', None) != 'semismooth_newton':
                return
            if getattr(self.solver, '_is_identity_proj', False):
                return
            if getattr(self.solver, '_is_rho_independent', False):
                return

            h_eff = float(h) / float(diag_factor)
            if not np.isfinite(h_eff) or h_eff <= 0.0:
                return

            A_diag = (A_local.diagonal() if sp.issparse(A_local)
                      else np.diag(np.asarray(A_local)))
            lam_vec = np.ones_like(np.asarray(A_diag, dtype=float))
            phys_mask = np.abs(A_diag) > 0.0
            lam_vec[phys_mask] = h_eff / A_diag[phys_mask]
            self.solver.lam = lam_vec
        except Exception:
            pass

    def _apply_post_step_projection(
        self, prev_state, candidate, *, t_new, h, Fk_val=None, residual_eval=None
    ):
        """Apply an optional end-of-step projection stage."""
        projection = getattr(self, "post_step_projection", None)
        if projection is None:
            return candidate, Fk_val, None

        projected = _call_projection_with_context(
            projection,
            prev_state,
            candidate,
            rhok=getattr(self, "post_step_rhok", 1.0),
            t=t_new,
            Fk_val=Fk_val,
            prev_state=prev_state,
            step_size=h,
        )
        projected = np.asarray(projected, dtype=float).reshape(candidate.shape)

        projected_fk = Fk_val
        if residual_eval is not None:
            try:
                projected_fk = residual_eval(projected)
            except Exception:
                projected_fk = Fk_val

        proj_delta = projected - candidate
        return projected, projected_fk, proj_delta

    def step(self, fun, t, y, h):
        """Perform one implicit Backward Euler step.

        Solves the nonlinear system::

            A ((y_new - y) / h) - f(t + h, y_new) = 0

        Parameters
        ----------
        fun : callable
            RHS function ``fun(t, y)`` (optionally ``fun(t, y, Fk)`` tolerated).
        t : float
            Current time of the known state ``y``.
        y : ndarray, shape (n,)
            Current state.
        h : float
            Proposed step size.

        Returns
        -------
        y_new : ndarray
            Next state.
        Fk_new : ndarray or None
            Residual / implicit function evaluation at the converged iterate.
        err_est : float
            Solver's local nonlinear residual norm (not LTE). For multi-stage
            composites this is the last stage residual norm.
        success : bool
            True if nonlinear solve converged to tolerance.
        iterations : int
            Number of nonlinear iterations executed.
        """
        A_local = self._get_A(len(y))
        self._refresh_solver_step_cache(h)
        self._set_solver_lam_from_step(A_local, h)

        # Helper to flexibly call fun with optional Fk_val
        prev_state_arg = y if self.pass_prev_state else None
        step_size_arg = h if self.pass_step_size else None

        # Bind RHS once with available context (prev_state, step_size)
        _rhs = self._get_bound_wrapper(fun, has_prev=(prev_state_arg is not None), has_h=(step_size_arg is not None), cache=self._fun_bindings)

        def _call_fun(f, tt, yy, Fk=None):
            # f is ignored; wrapper is bound to func shape
            return _rhs(tt, yy, Fk, prev_state_arg, step_size_arg)

        # Define the implicit equation for the backward Euler step.
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - _call_fun(
            fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None)
        )

        # If an analytical RHS Jacobian is provided, set an exact residual Jacobian for this step.
        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':
            A_over_h = A_local / h

            def jac_eq(y_new, _Aoh=A_over_h, _t=t, _h=h, _rhs_jac=rhs_jac, _solver=self.solver):
                # J_res = (A_local/h) - d f/dy (t+h, y_new)
                fk_val = getattr(_solver, 'last_Fk_val', None)
                _jac = self._get_bound_wrapper(_rhs_jac, has_prev=(prev_state_arg is not None), has_h=(step_size_arg is not None), cache=self._jac_bindings)
                rhs_jac_val = _jac(_t + _h, y_new, fk_val, prev_state_arg, step_size_arg)
                return _Aoh - rhs_jac_val

            self.solver.jacobian = jac_eq

        # Thread step-context to the solver so projections can access it
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = h
        except Exception:
            pass

        y_new, Fk_new, err_est, success, iterations = self.solver.solve(implicit_eq, y)
        if success:
            y_new, Fk_new, proj_delta = self._apply_post_step_projection(
                y,
                y_new,
                t_new=t + h,
                h=h,
                Fk_val=Fk_new,
                residual_eval=implicit_eq,
            )
            if proj_delta is not None:
                try:
                    self.last_post_step_delta = np.asarray(proj_delta, dtype=float)
                except Exception:
                    self.last_post_step_delta = None
            if Fk_new is not None:
                try:
                    err_est = float(np.linalg.norm(np.asarray(Fk_new).ravel(), ord=np.inf))
                except Exception:
                    pass
        return y_new, Fk_new, err_est, success, iterations


class BackwardEulerSchur(IntegrationMethod):
    """Backward Euler with Schur-complement Newton solver.

    Uses the Macklin (2019) compliance-formulation: the 2x2 block system
    [H, -J^T; J, C] is reduced via Schur complement to an n_react-sized
    PCR solve, then back-substituted for the velocity update.

    The ``fun`` argument to ``step()`` must be a ``BlockStructuredSystem``
    (i.e. an ``NCPBlockSystem`` from ``build_ncp_contact_blocked``),
    not a plain callable.

    Parameters
    ----------
    A : ndarray or sparse or None
        Mass / descriptor matrix.
    schur_solver_opts : dict
        Keyword arguments forwarded to ``SchurComplementSolver``.
    """

    def __init__(self, A=None, schur_solver_opts=None):
        self.A = A
        self.order = 1
        self.has_embedded_error = False
        self._schur_opts = schur_solver_opts or {}

    def step(self, fun, t, y, h):
        from .solvers.block_system import SchurComplementSolver

        solver = SchurComplementSolver(**self._schur_opts)
        y_new, err, converged, iters = solver.solve(
            fun, y, t + h, h, y,
        )
        Fk = None
        return y_new, Fk, err, converged, iters


class RadauIIASchur(IntegrationMethod):
    """RadauIIA with coupled Newton via Schur-complement reduction.

    All *s* stages are solved simultaneously.  The velocity block
    ``H_full`` carries Butcher cross-stage coupling; the contact blocks
    ``B_bot``, ``C`` are block-diagonal.  ``B_top`` has off-diagonal
    blocks from the ``B·λ`` coupling in the physical RHS.

    For ``stages=1`` this delegates to :class:`BackwardEulerSchur`.

    Parameters
    ----------
    stages : {1, 2, 3}
    A : ndarray or sparse or None
        Augmented mass matrix (n_aug × n_aug).
    schur_solver_opts : dict
        Forwarded to ``SchurComplementSolver``.
    """

    def __init__(self, stages=2, A=None, schur_solver_opts=None):
        if stages not in (1, 2, 3):
            raise ValueError(f"stages must be 1, 2, or 3; got {stages}")
        self.stages = int(stages)
        self.A = A
        self._schur_opts = schur_solver_opts or {}
        self.has_embedded_error = False

        _sq6 = math.sqrt(6.0)
        if stages == 1:
            self._rk_A = np.array([[1.0]])
            self._rk_c = np.array([1.0])
            self.order = 1
        elif stages == 2:
            self._rk_A = np.array([
                [5.0 / 12.0, -1.0 / 12.0],
                [3.0 / 4.0,   1.0 / 4.0],
            ])
            self._rk_c = np.array([1.0 / 3.0, 1.0])
            self.order = 3
        else:
            self._rk_A = np.array([
                [(88 - 7 * _sq6) / 360,
                 (296 - 169 * _sq6) / 1800,
                 (-2 + 3 * _sq6) / 225],
                [(296 + 169 * _sq6) / 1800,
                 (88 + 7 * _sq6) / 360,
                 (-2 - 3 * _sq6) / 225],
                [(16 - _sq6) / 36,
                 (16 + _sq6) / 36,
                 1.0 / 9.0],
            ])
            self._rk_c = np.array(
                [(4 - _sq6) / 10, (4 + _sq6) / 10, 1.0],
            )
            self.order = 5

    def step(self, fun, t, y, h):
        from .solvers.block_system import SchurComplementSolver

        s = self.stages
        solver = SchurComplementSolver(**self._schur_opts)

        if s == 1:
            y_new, err, ok, iters = solver.solve(fun, y, t + h, h, y)
            return y_new, None, err, ok, iters

        A_phys = getattr(fun, '_A_phys', None)
        if A_phys is None:
            n_p = fun.n_phys
            A_aug = self.A
            if A_aug is not None:
                A_phys = (A_aug.toarray()[:n_p, :n_p]
                          if sp.issparse(A_aug)
                          else np.asarray(A_aug)[:n_p, :n_p])
            else:
                A_phys = np.eye(n_p)

        y_new, err, ok, iters = solver.solve_coupled(
            fun, y, t, h, y,
            rk_A=self._rk_A,
            rk_c=self._rk_c,
            A_phys=A_phys,
        )
        return y_new, None, err, ok, iters


class AlgebraicBackwardEuler(IntegrationMethod):
    """Backward Euler variant supporting a subset of algebraic (index-1) constraints.

    Differential indices D use standard BE residual:
        (y_new - y)/h - f(t+h, y_new) = 0   (or A-scaled if A provided)

    Algebraic indices A enforce g(y_new)_A = 0 directly (no time derivative term).

    Parameters
    ----------
    algebraic_indices : sequence[int]
        Indices in the state treated as algebraic.
    g_func : callable
        Constraint function g(y) returning vector with at least entries for algebraic indices.
        Only entries at algebraic indices are used. If None, will default to identity constraint
        (i.e. forces y_A = 0).
    g_jac : callable, optional
        Jacobian dg/dy. If provided, an exact residual Jacobian is formed for algebraic rows.
    solver : ImplicitEquationSolver, optional
        Nonlinear solver to use (must be semismooth_newton or VI with projection).
    A : ndarray or None
        Mass / coefficient matrix for differential part; identity if None.
    """
    def __init__(self, algebraic_indices, g_func=None, g_jac=None, solver=None, A=None):
        self.algebraic_indices = np.array(sorted(algebraic_indices), dtype=int)
        self.g_func = g_func
        self.g_jac = g_jac
        self.solver = solver or ImplicitEquationSolver(method='semismooth_newton')
        self.A = A
        self.use_identity = A is None

    def _get_A(self, n):
        if not self.use_identity:
            return self.A
        # Prefer CSR identity for large n to reduce memory/compute
        want_sparse = False
        try:
            want_sparse = bool(self.solver._sparse_active(n))
        except Exception:
            thr = getattr(self.solver, 'sparse_threshold', 200)
            try:
                want_sparse = (n >= int(thr))
            except Exception:
                want_sparse = (n >= 200)
        if want_sparse:
            return sp.eye(n, format='csr')
        return np.eye(n)

    def step(self, fun, t, y, h):
        n = len(y)
        A_local = self._get_A(n)
        alg_idx = self.algebraic_indices
        diff_mask = np.ones(n, dtype=bool)
        diff_mask[alg_idx] = False
        diff_idx = np.nonzero(diff_mask)[0]

        def _call_fun(f, tt, yy, Fk=None):
            try:
                return f(tt, yy, Fk)
            except TypeError:
                try:
                    return f(tt, yy)
                except TypeError:
                    return f(yy)

        def residual(y_new):
            F = np.zeros(n, dtype=float)
            f_val = _call_fun(fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
            # Differential part rows
            if diff_idx.size:
                F[diff_idx] = (A_local[diff_idx][:, :] @ ((y_new - y) / h)) - f_val[diff_idx]
            # Algebraic part rows: g(y_new)_A = 0
            if alg_idx.size:
                if self.g_func is None:
                    F[alg_idx] = y_new[alg_idx]  # default constraint y_A = 0
                else:
                    g_val = self.g_func(y_new)
                    # Allow g to return full vector or just algebraic subset
                    if g_val.shape[0] == n:
                        F[alg_idx] = g_val[alg_idx]
                    else:
                        # assume ordered alignment with alg_idx
                        F[alg_idx] = g_val
            return F

        # Attach Jacobian if possible
        def jacobian(y_new):
            J = np.zeros((n, n), dtype=float)
            # Differential rows
            f_val = _call_fun(fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
            # Need df/dy for diff rows; reuse solver.rhs_jacobian if present
            rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
            if callable(rhs_jac):
                try:
                    dfdy = rhs_jac(t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
                except TypeError:
                    dfdy = rhs_jac(t + h, y_new)
            else:
                # fallback finite-difference (small system expected)
                eps = 1e-8
                dfdy = np.zeros((n, n))
                f0 = _call_fun(fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
                for j in range(n):
                    y_pert = y_new.copy(); y_pert[j] += eps
                    f_eps = _call_fun(fun, t + h, y_pert, getattr(self.solver, 'last_Fk_val', None))
                    dfdy[:, j] = (f_eps - f0)/eps
            if diff_idx.size:
                J[diff_idx, :] = (A_local[diff_idx][:, :] / h) - dfdy[diff_idx, :]
            if alg_idx.size:
                if self.g_jac is not None:
                    gJ = self.g_jac(y_new)
                    if gJ.shape[0] == len(alg_idx):
                        for row_pos, gi in enumerate(alg_idx):
                            J[gi, :] = gJ[row_pos]
                    else:
                        # assume full sized
                        J[alg_idx, :] = gJ[alg_idx, :]
                else:
                    # Numerical for g only
                    if self.g_func is None:
                        J[alg_idx, alg_idx] = 1.0
                    else:
                        eps = 1e-8
                        g0 = self.g_func(y_new)
                        full_g0 = np.zeros(len(alg_idx)) if g0.shape[0]==n else g0
                        if g0.shape[0]==n:
                            full_g0 = g0[alg_idx]
                        for j in range(n):
                            y_pert = y_new.copy(); y_pert[j] += eps
                            g_eps = self.g_func(y_pert)
                            if g_eps.shape[0]==n:
                                g_eps_sub = g_eps[alg_idx]
                            else:
                                g_eps_sub = g_eps
                            J[alg_idx, j] = (g_eps_sub - full_g0)/eps
            return J

        # Provide Jacobian to solver (only if semismooth_newton path)
        if getattr(self.solver, 'method', None) != 'VI':
            self.solver.jacobian = jacobian

        # Thread step-context
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = h
        except Exception:
            pass

        return self.solver.solve(residual, y)


class Trapezoidal(BackwardEuler):
    def __init__(self, solver=None, A=None, **kwargs):
        super().__init__(solver=solver, A=A, **kwargs)
        self.order = 2
    """
    Implements the Trapezoidal (Crank-Nicolson) integration method.

    Inherits the matrix handling from BackwardEuler.
    """

    def step(self, fun, t, y, h):
        """Perform one implicit Trapezoidal (Crank–Nicolson) step.

        Nonlinear system::

            A ((y_new - y)/h) - 0.5 ( f(t, y) + f(t+h, y_new) ) = 0

        Returns follow the same 5-tuple convention described in
        :meth:`BackwardEuler.step`.
        """
        A_local = self._get_A(len(y))
        self._refresh_solver_step_cache(h)
        self._set_solver_lam_from_step(A_local, h)

        prev_state_arg = y if self.pass_prev_state else None
        step_size_arg = h if self.pass_step_size else None

        # Bind RHS once (allows step_size override at call)
        _rhs = self._get_bound_wrapper(fun, has_prev=(prev_state_arg is not None), has_h=True, cache=self._fun_bindings)

        def _call_fun(f, tt, yy, Fk=None, h_override=None):
            return _rhs(tt, yy, Fk, prev_state_arg, (h_override if h_override is not None else step_size_arg))

        fk_last = getattr(self.solver, 'last_Fk_val', None)
        f_n = _call_fun(fun, t, y, fk_last)
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - 0.5 * (
            f_n + _call_fun(fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
        )

        # Exact Jacobian when analytical RHS Jacobian is available
        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':
            A_over_h = A_local / h

            def jac_eq(y_new, _Aoh=A_over_h, _t=t, _h=h, _rhs_jac=rhs_jac, _solver=self.solver):
                # J_res = (A_local/h) - 0.5 * d f/dy (t+h, y_new)
                fk_val = getattr(_solver, 'last_Fk_val', None)
                _jac = self._get_bound_wrapper(_rhs_jac, has_prev=(prev_state_arg is not None), has_h=(step_size_arg is not None), cache=self._jac_bindings)
                rhs_jac_val = _jac(_t + _h, y_new, fk_val, prev_state_arg, step_size_arg)
                return _Aoh - 0.5 * rhs_jac_val

            self.solver.jacobian = jac_eq

        # Thread step-context
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = h
        except Exception:
            pass

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
        # Theta=0.5 is TR (order 2); otherwise default to order 1
        self.order = 2 if abs(self.theta - 0.5) < 1e-12 else 1

    def step(self, fun, t, y, h):
        """Perform one Theta method step.

        Nonlinear system::

            A ((y_new - y)/h) - ( theta f(t+h, y_new) + (1-\theta) f(t, y) ) = 0

        Returns: 5-tuple as in :meth:`BackwardEuler.step`.
        """
        A_local = self._get_A(len(y))
        self._refresh_solver_step_cache(h)
        self._set_solver_lam_from_step(A_local, h)

        prev_state_arg = y if self.pass_prev_state else None
        step_size_arg = h if self.pass_step_size else None

        _rhs = self._get_bound_wrapper(fun, has_prev=(prev_state_arg is not None), has_h=(step_size_arg is not None), cache=self._fun_bindings)

        def _call_fun(f, tt, yy, Fk=None):
            return _rhs(tt, yy, Fk, prev_state_arg, step_size_arg)

        fk_last = getattr(self.solver, 'last_Fk_val', None)
        f_n = _call_fun(fun, t, y, fk_last)
        implicit_eq = lambda y_new: A_local @ ((y_new - y) / h) - (
            self.theta * _call_fun(fun, t+h, y_new, getattr(self.solver, 'last_Fk_val', None)) + (1 - self.theta) * f_n
        )

        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':
            A_over_h = A_local / h
            theta_val = self.theta

            def jac_eq(y_new, _Aoh=A_over_h, _t=t, _h=h, _rhs_jac=rhs_jac, _theta=theta_val, _solver=self.solver):
                # J_res = (A_local/h) - theta * d f/dy (t+h, y_new)
                fk_val = getattr(_solver, 'last_Fk_val', None)
                _jac = self._get_bound_wrapper(_rhs_jac, has_prev=(prev_state_arg is not None), has_h=(step_size_arg is not None), cache=self._jac_bindings)
                rhs_jac_val = _jac(_t + _h, y_new, fk_val, prev_state_arg, step_size_arg)
                return _Aoh - _theta * rhs_jac_val

            self.solver.jacobian = jac_eq

        # Thread step-context
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = h
        except Exception:
            pass

        return self.solver.solve(implicit_eq, y)


class CompositeMethod(IntegrationMethod):
    """
    Implements a composite integration method that combines two steps:
      1. A half-step using the Trapezoidal method.
      2. A full step using a modified Backward Euler method.

    The composite method first advances the solution halfway in time, then uses this intermediate
    value to compute the final step.
    """

    def __init__(self, a=1.0, solver=None, A=None, **kwargs):
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
        self.solver = solver or ImplicitEquationSolver(method='semismooth_newton')
        # Create instances for sub-steps using Trapezoidal and Backward Euler methods.
        self.trapezoidal = Trapezoidal(solver=self.solver, A=A, **kwargs)
        self.backward_euler = BackwardEuler(solver=self.solver, A=A, **kwargs)
        # TR-BE composite is second-order (TR-BDF2 style)
        self.order = 2

    def step(self, fun, t, y, h):
        """Composite TR / BE second-order step (TR-BDF2 style variant).

        Two stages:
          1. Half-step TR to obtain ``y_half``.
          2. Modified BE relation ``(3*y_new - 4*y_half + y)/h - f(t+h, y_new) = 0``.

        Returns
        -------
        y_new, Fk_new, err_new, success, iterations : as in other integrators with
        iteration count the sum over both stages.
        """
        half_h = 0.5 * h

        # ----- Stage 1: half-step TR from (t, y) to (t+half_h, y_half)
        # The TR.step call itself will set prev_state=y etc. via shared solver.
        y_half, Fk_half, err_half, success_half, iters_half = \
            self.trapezoidal.step(fun, t, y, half_h)
        if not success_half:
            return y, Fk_half, err_half, False, iters_half

        # ----- Stage 2: BE-like relation from (t+half_h, y_half) to (t+h, y_new)
        prev_state_arg = y_half if getattr(self.backward_euler, 'pass_prev_state', False) else None
        step_size_arg = h if getattr(self.backward_euler, 'pass_step_size', False) else None

        # Bind RHS once for the second stage using BackwardEuler's binder/caches
        _rhs_be = self.backward_euler._get_bound_wrapper(
            fun,
            has_prev=(prev_state_arg is not None),
            has_h=(step_size_arg is not None),
            cache=self.backward_euler._fun_bindings,
        )

        def _call_fun(f, tt, yy, Fk=None):
            return _rhs_be(tt, yy, Fk, prev_state_arg, step_size_arg)

        def implicit_eq(y_new):
            A_local = self.backward_euler._get_A(len(y))
            return A_local @ ((3.0 * y_new - 4.0 * y_half + y) / h) - _call_fun(
                fun, t + h, y_new, getattr(self.backward_euler.solver, 'last_Fk_val', None)
            )

        A_stage2 = self.backward_euler._get_A(len(y))
        self.backward_euler._refresh_solver_step_cache(h)
        self.backward_euler._set_solver_lam_from_step(A_stage2, h, diag_factor=3.0)

        rhs_jac = getattr(self.backward_euler.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.backward_euler.solver, 'method', None) != 'VI':
            A_over_h = A_stage2 / h

            def jac_eq_second(y_new, _Aoh=A_over_h, _t=t, _h=h, _rhs_jac=rhs_jac, _solver=self.backward_euler.solver):
                fk_val = getattr(_solver, 'last_Fk_val', None)
                _jac = self.backward_euler._get_bound_wrapper(
                    _rhs_jac,
                    has_prev=(prev_state_arg is not None),
                    has_h=(step_size_arg is not None),
                    cache=self.backward_euler._jac_bindings,
                )
                rhs_jac_val = _jac(_t + _h, y_new, fk_val, prev_state_arg, step_size_arg)
                return (3.0 * _Aoh) - rhs_jac_val

            self.backward_euler.solver.jacobian = jac_eq_second

        # Thread step-context for stage 2 (previous accepted state is y_half)
        try:
            self.backward_euler.solver.current_time = t + h
            self.backward_euler.solver.prev_state = y_half
            self.backward_euler.solver.prev_time = t + half_h
            self.backward_euler.solver.prev_step = half_h
        except Exception:
            pass

        y_guess = y_half
        y_new, Fk_new, err_new, success_new, iters_new = self.backward_euler.solver.solve(implicit_eq, y_guess)
        total_iters = iters_half + iters_new
        overall_success = success_half and success_new
        return (y_new, Fk_new, err_new, overall_success, total_iters)


class SDIRK2(BackwardEuler):
    r"""Two-stage, L-stable, singly diagonally implicit Runge–Kutta method of
    order 2 with an embedded order-1 error estimate.

    Butcher tableau (Alexander 1977)::

        γ   | γ     0
        1   | 1-γ   γ
        ----|----------
            | 1-γ   γ     ← order 2
            | 1     0     ← order 1 (embedded)

    where γ = 1 − √2/2 ≈ 0.2929.

    Each stage requires one implicit solve with the *same* diagonal
    coefficient γ, which means the iteration matrix
    ``(A/(γh) − df/dy)`` can in principle be reused across both stages.

    The embedded pair yields a cheap local-truncation-error estimate
    expressed purely in terms of the stage values (mass-matrix agnostic)::

        err = (Y₂ − y_shift) − (Y₁ − y)

    where ``y_shift = y + ((1−γ)/γ)(Y₁ − y)``.  This is returned as
    the ``err`` component of the 5-tuple.  The adaptive stepper
    (:class:`AdaptiveStepping`) can therefore skip the three-solve
    Richardson extrapolation when this integrator is used.

    Parameters
    ----------
    solver : ImplicitEquationSolver, optional
        Nonlinear solver instance.
    A : ndarray or sparse matrix, optional
        Mass / descriptor matrix; identity if *None*.
    pass_prev_state, pass_step_size : bool
        Forwarded to :class:`BackwardEuler`.
    """

    # Diagonal coefficient  γ = 1 − √2/2
    _GAMMA = 1.0 - math.sqrt(2.0) / 2.0

    def __init__(self, solver=None, A=None, **kwargs):
        super().__init__(solver=solver, A=A, **kwargs)
        self.order = 2  # method order for the adaptive controller
        self.has_embedded_error = True  # bypass Richardson in AdaptiveStepping

    # ------------------------------------------------------------------
    def step(self, fun, t, y, h):
        r"""Advance by one SDIRK2 step of size *h*.

        Returns
        -------
        y_new : ndarray
            State at ``t + h`` (second-order accurate).
        Fk_new : ndarray or None
            Residual at the converged iterate of stage 2.
        err : ndarray
            Element-wise embedded error estimate ``(Y₂ − y_shift) − (Y₁ − y)``,
            equivalent to ``h γ (K₂ − K₁)`` where ``K_i`` are the stage
            derivatives.  Mass-matrix agnostic.
        success : bool
            True if *both* stage solves converged.
        iterations : int
            Sum of nonlinear iterations over the two stages.
        """
        gamma = self._GAMMA
        n = len(y)
        A_local = self._get_A(n)
        gh = gamma * h  # diagonal sub-step length

        # -- Cache A/(γh) across steps to avoid redundant sparse division --
        # Both SDIRK2 stages share the same diagonal coefficient γh, so the
        # matrix  A/(γh)  only changes when h changes.  Caching it avoids a
        # full sparse-scalar division (O(nnz) allocation + copy) on every
        # call to step() when h is unchanged — which is the common case for
        # an adaptive controller on a smooth trajectory.
        _prev_gh = getattr(self, '_cached_gh', None)
        if _prev_gh is None or _prev_gh != gh:
            self._A_over_gh = A_local / gh
            self._cached_gh = gh
            # ---- Invalidate solver caches ----
            # The iteration matrix  M/(γh) − J  depends on h.  When h
            # changes, a cached SPLU factorisation or Jacobian from the
            # previous step is stale and MUST be discarded.  Without
            # this, the modified-Newton loop's first iteration reuses
            # the stale factor (because errF > 0.5*inf is always False
            # in IEEE-754), corrupts the iterate, and wastes iterations
            # recovering — often exceeding max_iter.
            self.solver._lu = None
            self.solver._lu_shape = None
            if hasattr(self.solver, '_J_cross_call'):
                self.solver._J_cross_call = None
            # Also invalidate PETSc/MUMPS factorisation — the direct
            # solver path in _solve_with_petsc will otherwise skip the
            # matrix update and keep solving with the stale factor.
            self.solver._petsc_needs_matrix_update = True
        A_over_gh = self._A_over_gh

        # -- helpers for flexible RHS calling ----------------------------
        prev_state_arg = y if self.pass_prev_state else None
        # Pass the *stage* sub-step γh (not full h).  The contact RHS
        # computes  B·r/(θ·h_val);  the stage equation divides by γh,
        # so net coupling is  B·r·γh / (θ·γh) = B·r/θ — correct for
        # θ = 1.  Pre-stress callbacks (get_s0) that multiply force×h_val
        # also get the stage-level impulse, which is physically right.
        step_size_arg = gh if self.pass_step_size else None
        _rhs = self._get_bound_wrapper(
            fun,
            has_prev=(prev_state_arg is not None),
            has_h=(step_size_arg is not None),
            cache=self._fun_bindings,
        )

        def _call_fun(tt, yy, Fk=None):
            return _rhs(tt, yy, Fk, prev_state_arg, step_size_arg)

        # ── Block-diagonal ρ equilibration ──────────────────────────────
        # The natural-map Jacobian  J_Φ = I − DΠ(I − ρ J_F)  has the
        # implicit-equation diagonal  A_ii/(γh)  on the physical rows.
        # When γ < 1 (e.g. SDIRK2, γ ≈ 0.293) this diagonal scales
        # as 1/γ relative to Backward Euler, pushing the condition
        # number above the IEEE-754 accuracy floor and stalling Newton.
        # Setting  ρ_i = γh / A_ii  for M-rows and  ρ_i = 1  for
        # algebraic (zero-mass) rows equilibrates the system to O(1).
        _A_diag = (A_local.diagonal() if sp.issparse(A_local)
                   else np.diag(A_local))
        _phys_mask = np.abs(_A_diag) > 0.0
        _rho_vec = np.ones(n)
        _rho_vec[_phys_mask] = gh / _A_diag[_phys_mask]
        self.solver.lam = _rho_vec

        # ================================================================
        # Stage 1:  solve  A (Y1 − y)/(γh) − f(t + γh, Y1) = 0
        # ================================================================
        def implicit_s1(Y1):
            return A_over_gh @ (Y1 - y) - _call_fun(
                t + gh, Y1, getattr(self.solver, 'last_Fk_val', None)
            )

        # Exact Jacobian for stage 1 (if available)
        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':

            def jac_s1(Y1, _Aogh=A_over_gh, _t=t, _gh=gh):
                fk_val = getattr(self.solver, 'last_Fk_val', None)
                _jac = self._get_bound_wrapper(
                    rhs_jac,
                    has_prev=(prev_state_arg is not None),
                    has_h=(step_size_arg is not None),
                    cache=self._jac_bindings,
                )
                J_rhs = _jac(_t + _gh, Y1, fk_val, prev_state_arg, step_size_arg)
                return _Aogh - J_rhs

            self.solver.jacobian = jac_s1

        # Thread step-context for stage 1
        try:
            self.solver.current_time = t + gh
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = gh
        except Exception:
            pass

        Y1, Fk1, err1, ok1, it1 = self.solver.solve(implicit_s1, y)
        if not ok1:
            return y, Fk1, np.zeros(n), False, it1

        # ================================================================
        # Stage 2:  solve  A (Y2 − y_shift) / (γh) − f(t+h, Y2) = 0
        #
        # The Butcher stage derivative K1 = (Y1 − y)/(γh).  The stage-2
        # shift uses K1 directly (no f evaluation or M⁻¹ needed):
        #   y_shift = y + (1−γ)h K1 = y + ((1−γ)/γ)(Y1 − y)
        # ================================================================
        dY1 = Y1 - y                                   # stage-1 increment
        y_shift = y + ((1.0 - gamma) / gamma) * dY1    # mass-matrix agnostic

        def implicit_s2(Y2):
            return A_over_gh @ (Y2 - y_shift) - _call_fun(
                t + h, Y2, getattr(self.solver, 'last_Fk_val', None)
            )

        # Exact Jacobian for stage 2 (same structure, different time)
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':

            def jac_s2(Y2, _Aogh=A_over_gh, _t=t, _h=h):
                fk_val = getattr(self.solver, 'last_Fk_val', None)
                _jac = self._get_bound_wrapper(
                    rhs_jac,
                    has_prev=(prev_state_arg is not None),
                    has_h=(step_size_arg is not None),
                    cache=self._jac_bindings,
                )
                J_rhs = _jac(_t + _h, Y2, fk_val, prev_state_arg, step_size_arg)
                return _Aogh - J_rhs

            self.solver.jacobian = jac_s2

        # Thread step-context for stage 2
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = Y1
            self.solver.prev_time = t + gh
            self.solver.prev_step = gh
        except Exception:
            pass

        # Good initial guess: extrapolate from stage 1
        Y2_guess = y_shift
        Y2, Fk2, err2, ok2, it2 = self.solver.solve(implicit_s2, Y2_guess)
        if not ok2:
            return y, Fk2, np.zeros(n), False, it1 + it2

        # ================================================================
        # Output
        # ================================================================
        # y_{n+1} = Y2  (already the second-order solution)
        y_new = Y2

        y_new, Fk2, proj_delta = self._apply_post_step_projection(
            y,
            y_new,
            t_new=t + h,
            h=h,
            Fk_val=Fk2,
            residual_eval=implicit_s2,
        )
        if proj_delta is not None:
            try:
                self.last_post_step_delta = np.asarray(proj_delta, dtype=float)
            except Exception:
                self.last_post_step_delta = None

        # Embedded error estimate (mass-matrix agnostic, uses stage values):
        #   err = h γ (K2 − K1)
        #       = (Y2 − y_shift) − (Y1 − y)
        #       = (Y2 − y_shift) − dY1
        err_embed = (Y2 - y_shift) - dY1

        total_iters = it1 + it2
        return y_new, Fk2, err_embed, True, total_iters


class RadauIIA(BackwardEuler):
    r"""s-stage Radau IIA collocation method.

    Radau IIA methods are collocation schemes at shifted Gauss–Legendre nodes that
    include the right endpoint (``c_s = 1``).  Their key properties for stiff and
    nonsmooth dynamics are:

    * **L-stability**: ``R(infinity) = 0``, so spurious modes are annihilated in
      one step.
    * **Stiff accuracy**: ``a_{s,j} = b_j`` for all j, so the last stage is the
      step output (``y_{n+1} = Y_s``); no additional combination is needed.
    * **Order 2s-1**: the highest order achievable with ``s`` stages for a
      one-step method.

    Supported stage counts:

    * ``stages=1``: order 1, equivalent to Backward Euler.
    * ``stages=2``: order 3, preferred for contact/impact problems.
    * ``stages=3``: order 5, high accuracy in smooth regions.

    Multi-stage solve via waveform relaxation
    ------------------------------------------
    For s ≥ 2 the stage equations are **fully coupled** (the Butcher matrix A is
    not lower-triangular).  The implementation uses *waveform relaxation* (block
    Gauss–Seidel on the stage index): stage i is solved as a Backward-Euler-like
    system::

        A_M (Y_i − y) / (a_{ii} h) − f(t + c_i h, Y_i) = C_i

    where ``C_i = Σ_{j≠i} (a_{ij}/a_{ii}) · f(t + c_j h, Y_j)`` is the explicit
    coupling contribution from all other stages.  Typically 1–3 outer sweeps
    suffice for moderate stiffness; for purely smooth problems a single sweep
    (``wf_maxiter=1``) already achieves full order.

    The constant ``C_i`` shifts only the RHS by an additive term — it does **not**
    change the Jacobian structure — so the same solver/projection infrastructure
    (including semismooth Newton with contact projections) is reused unchanged
    for every stage.

    LU factorisation reuse
    ----------------------
    Within a single stage, the iteration matrix ``A_M/(a_{ii}h) − ∂f/∂Y`` is
    fixed.  The existing cross-call LU cache in :class:`ImplicitEquationSolver`
    reuses the factorisation across Newton iterations of the same stage, and
    across waveform-relaxation outer sweeps when ``h`` is unchanged.  Separate
    per-stage-step-size caches are maintained so that the LU for stage 1
    (step ``a_{11}h``) and stage 2 (step ``a_{22}h``) are swapped rather than
    recomputed when the solver alternates between stages.

    Velocity projection (Breuling Stage 2)
    ----------------------------------------
    Pass a ``post_step_projection`` to enforce Newton's impact law after the
    RK stage system is solved.  The projection receives ``(y_prev, Y_s)`` and
    returns the corrected step output.  This corresponds exactly to Stage 2 of
    the *nonsmooth projected Radau IIA* described in Breuling (2024), Chapters 4–5.

    Embedded error estimate
    -----------------------
    For ``s=2`` the order-3 result ``Y_2`` is compared with the linear
    extrapolation of stage 1 to the end of the interval (a free first-order
    estimate)::

        err = Σ_k e_k · (Y_k − y),   e = e_s − b̂ᵀ A⁻¹,  b̂ = [1, 0, …]
            ≡ 0.5·Y_2 − 1.5·Y_1 + y   (s=2 Radau IIA)

    For ``s=3`` a quadratic extrapolation through stages 1 and 2 serves as the
    lower-order companion.  Both estimates are mass-matrix agnostic (no M⁻¹
    required) and are returned as element-wise vectors compatible with the
    adaptive step-size controller.

    Parameters
    ----------
    stages : {1, 2, 3}, default 2
        Number of collocation stages.  ``stages=1`` is equivalent to
        :class:`BackwardEuler` and is delegated directly.
    wf_maxiter : int, default 3
        Maximum outer waveform-relaxation sweeps.  1 = single Gauss–Seidel
        pass; sufficient for smooth problems.  Increase to 4–6 near impact
        events where stage coupling is strong.
    wf_tol : float, default 1e-12
        Outer-loop early-exit tolerance.  Iteration stops when the maximum
        relative change in any stage value falls below this threshold.
    solver : :class:`~solve_nivp.ImplicitEquationSolver`, optional
    A : ndarray or sparse matrix, optional
        Mass / descriptor matrix; identity when *None*.
    pass_prev_state, pass_step_size : bool
        Forwarded to :class:`BackwardEuler`.
    post_step_projection : callable or None
        Velocity projection (Breuling Stage 2).  Uses the standard
        :func:`_call_projection_with_context` interface.
    """

    def __init__(
        self,
        stages: int = 2,
        solver=None,
        A=None,
        wf_maxiter: int = 3,
        wf_tol: float = 1e-12,
        use_coupled_newton: bool = True,
        projected_radau_contact=None,
        **kwargs,
    ):
        if stages not in (1, 2, 3):
            raise ValueError(f"RadauIIA: stages must be 1, 2, or 3; got {stages!r}")
        super().__init__(solver=solver, A=A, **kwargs)
        self.stages = int(stages)
        self.wf_maxiter = max(1, int(wf_maxiter))
        self.wf_tol = float(wf_tol)
        self.use_coupled_newton = bool(use_coupled_newton)
        self.projected_radau_contact = projected_radau_contact

        # ── Butcher tableaux (Hairer & Wanner, Solving ODEs II, §II.7) ────────
        _sq6 = math.sqrt(6.0)
        if stages == 1:
            self._rk_A = np.array([[1.0]])
            self._rk_b = np.array([1.0])
            self._rk_c = np.array([1.0])
            self.order = 1
        elif stages == 2:
            # Table II.7.1 — order 3
            self._rk_A = np.array(
                [[ 5.0 / 12.0, -1.0 / 12.0],
                 [ 3.0 /  4.0,  1.0 /  4.0]],
                dtype=float,
            )
            self._rk_b = np.array([3.0 / 4.0, 1.0 / 4.0])
            self._rk_c = np.array([1.0 / 3.0, 1.0])
            self.order = 3
        else:  # stages == 3
            # Table II.7.2 — order 5
            self._rk_A = np.array(
                [
                    [(88 -   7*_sq6) / 360,  (296 - 169*_sq6) / 1800,  (-2 + 3*_sq6) / 225],
                    [(296 + 169*_sq6) / 1800, (88 +   7*_sq6) / 360,   (-2 - 3*_sq6) / 225],
                    [(16 -     _sq6) /  36,  (16 +     _sq6) /  36,     1.0 / 9.0          ],
                ],
                dtype=float,
            )
            self._rk_b = np.array(
                [(16 - _sq6) / 36, (16 + _sq6) / 36, 1.0 / 9.0]
            )
            self._rk_c = np.array([(4 - _sq6) / 10, (4 + _sq6) / 10, 1.0])
            self.order = 5

        # Stiff accuracy: a_{s,j} = b_j (guaranteed for Radau IIA)
        # → y_{n+1} = Y_s, no extra combination step required.
        #
        # Embedded error strategy (stage-count dependent):
        #   s=2: Use an O(h²) embedded estimate from the first-order companion
        #        b̂=[1,0].  The adaptive controller uses exponent 1/(q+1)=1/2 via
        #        ``embedded_order=1``, giving proper step-size calibration.
        #   s=3: Fall back to Richardson extrapolation (step-doubling).  The
        #        b̂=[1,0,0] companion has a larger error constant than the s=2
        #        analogue, making the embedded step inefficient.  Richardson gives
        #        an O(h^6) estimate correctly calibrated for the p=5 exponent
        #        (1/6), yielding ~4–10× larger steps for smooth problems.
        self.has_embedded_error = (stages == 2)

        # ── Embedded error estimate coefficients (mass-matrix agnostic) ───────
        # For s=2: companion b̂=[1,0] — mass-matrix-free formula using A^{-1}.
        # err = Σ_k _err_coeffs[k] · (Y_k − y)
        # = (e_s − b̂^T · A^{-1}) · (Y − y)   [no M^{-1} needed]
        if stages == 2:
            _rk_Ainv = np.linalg.inv(self._rk_A)
            # b̂^T @ Ainv = [1,0] @ Ainv = first row of Ainv
            _b_hat_Ainv = _rk_Ainv[0, :]          # shape (2,)
            _e_s = np.array([0.0, 1.0])
            self._err_coeffs = _e_s - _b_hat_Ainv  # shape (2,)
        else:
            self._err_coeffs = np.zeros(max(stages, 1))

        # Order of the embedded companion for s=2.  AdaptiveStepping reads this
        # via ``embedded_order`` and uses exponent 1/(q+1) = 1/2 for q=1.
        self.embedded_order: int = 1

        # Per-stage-step-size LU swap cache (key: stage_h float, value: LU state tuple)
        # Allows O(1) LU restoration when cycling through stages with different aii*h.
        self._stage_lu_cache: dict = {}
        self._cached_stage_h: float | None = None

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _swap_stage_lu(self, new_stage_h: float) -> None:
        """Swap the solver's LU state when moving to a stage with a different step size.

        Maintains a small cache keyed by ``a_{ii}·h`` so that returning to a
        previously computed stage restores the factorisation rather than
        recomputing it from scratch.  The cache is bounded to 6 entries
        (3 stages × 2 consecutive step sizes) to avoid unbounded growth.
        """
        prev_h = self._cached_stage_h
        if prev_h is None or prev_h == new_stage_h:
            self._cached_stage_h = new_stage_h
            return

        # Save current LU under prev_h
        self._stage_lu_cache[prev_h] = (
            getattr(self.solver, '_lu', None),
            getattr(self.solver, '_lu_shape', None),
            getattr(self.solver, '_lu_pattern', None),
            getattr(self.solver, '_lu_use_count', 0),
            getattr(self.solver, '_J_cross_call', None),
        )
        # Bound the cache size
        while len(self._stage_lu_cache) > 6:
            oldest = next(iter(self._stage_lu_cache))
            del self._stage_lu_cache[oldest]

        # Restore (or invalidate) the LU for new_stage_h
        cached = self._stage_lu_cache.get(new_stage_h)
        if cached is not None:
            (self.solver._lu,
             self.solver._lu_shape,
             self.solver._lu_pattern,
             self.solver._lu_use_count,
             self.solver._J_cross_call) = cached
        else:
            self.solver._lu = None
            self.solver._lu_shape = None
            if hasattr(self.solver, '_J_cross_call'):
                self.solver._J_cross_call = None
        self.solver._petsc_needs_matrix_update = True
        self._cached_stage_h = new_stage_h

    # ──────────────────────────────────────────────────────────────────────────
    # Coupled Newton
    # ──────────────────────────────────────────────────────────────────────────

    def _step_coupled_newton_impl(
        self,
        t: float,
        y: np.ndarray,
        h: float,
        n: int,
        A_local,
        rk_A: np.ndarray,
        rk_c: np.ndarray,
        s: int,
        _call_fun,
        _jac_wrapper,
        prev_state_arg,
        step_size_arg,
    ):
        """Full Newton on the stacked (s·n) stage system.

        Assembles the (s·n × s·n) block Jacobian from per-stage analytical
        Jacobians and factorises it with SPLU on every Newton iteration.
        Cross-stage coupling is represented in the Jacobian, eliminating
        the linear-stability constraint on ``|a_{ij}/a_{ii}|`` that limits
        waveform-relaxation convergence for Radau IIA s=2
        (where ``a[1,0]/a[1,1] = 3``).

        Block Jacobian structure (i = row block, j = column block)::

            J[i, i] = A_M / (a_{ii}·h) − ∂f/∂Y_i(t + c_i·h, Y_i)
            J[i, j] = −(a_{ij}/a_{ii}) · ∂f/∂Y_j(t + c_j·h, Y_j)   j ≠ i

        Parameters
        ----------
        _call_fun : callable
            ``_call_fun(tt, yy, stage_h_override=v)`` returns f(tt, yy).
        _jac_wrapper : callable
            ``_jac_wrapper(tt, yy, fk, prev, h_val)`` returns ∂f/∂y as
            a dense ndarray or sparse matrix.

        Returns
        -------
        tuple ``(Y_list, f_stage, Fk_last, converged, total_iters)``
            Y_list : list of s ndarrays, shape (n,) each
            f_stage : list of s ndarrays — RHS evaluated at final Y_i
            Fk_last : last element of f_stage (last stage value)
            converged : bool
            total_iters : number of SPLU linear solves performed
        ``None``
            Returned when the method cannot proceed (Jacobian evaluation
            error or SPLU failure on the first iteration).  The caller
            should fall back to waveform relaxation.
        """
        solver = self.solver

        # Route the (s·n) stacked system through the standard Newton path so
        # it inherits WRMS convergence, damped-step fraction, diagonal
        # regularisation, cold-start slices and modified-Newton Jacobian
        # reuse.  The path is only available for identity projections (the
        # common case for NCP / Alart-Curnier / DAE residual formulations);
        # other projectors fall back to waveform relaxation.
        if not getattr(solver, '_is_identity_proj', False):
            return None

        A_sp = A_local.tocsr() if sp.issparse(A_local) else sp.csr_matrix(A_local)
        sn = s * n

        # Stage RHS values cached by ``F_stacked`` and consumed by
        # ``J_stacked`` to avoid a second round of per-stage evaluations.
        _cache = {'f_stage': None, 'Y_list': None}

        def F_stacked(Z):
            Y_list = [Z[i * n:(i + 1) * n] for i in range(s)]
            f_stage = [
                _call_fun(
                    t + rk_c[i] * h, Y_list[i],
                    stage_h_override=rk_A[i, i] * h,
                )
                for i in range(s)
            ]
            _cache['f_stage'] = f_stage
            _cache['Y_list'] = Y_list
            F_out = np.empty(sn, dtype=float)
            for i in range(s):
                aii = rk_A[i, i]
                Fi = A_local @ ((Y_list[i] - y) / (aii * h)) - f_stage[i]
                for j in range(s):
                    if j != i:
                        Fi = Fi - (rk_A[i, j] / aii) * f_stage[j]
                F_out[i * n:(i + 1) * n] = Fi
            return F_out

        def J_stacked(Z):
            f_stage = _cache['f_stage']
            Y_list = _cache['Y_list']
            if Y_list is None:
                Y_list = [Z[i * n:(i + 1) * n] for i in range(s)]
                f_stage = [None] * s
            J_rhs = []
            for j in range(s):
                h_j = (rk_A[j, j] * h) if step_size_arg is not None else None
                fk_hint_j = f_stage[j] if f_stage is not None else None
                Jj = _jac_wrapper(
                    t + rk_c[j] * h, Y_list[j], fk_hint_j, prev_state_arg, h_j,
                )
                J_rhs.append(
                    Jj.tocsr() if sp.issparse(Jj) else sp.csr_matrix(Jj)
                )
            block_rows = []
            for i in range(s):
                aii = rk_A[i, i]
                row = []
                for j in range(s):
                    if j == i:
                        row.append((A_sp / (aii * h) - J_rhs[j]).tocsr())
                    else:
                        row.append((-(rk_A[i, j] / aii) * J_rhs[j]).tocsr())
                block_rows.append(sp.hstack(row, format='csr'))
            return sp.vstack(block_rows, format='csr')

        # Save solver state that must be restored after the call.
        # * cold_start_slices: index ranges get shifted to stacked layout,
        #   must not leak into subsequent (n, n) solver.solve invocations.
        # * jacobian: set to J_stacked, must be restored for the embedded
        #   error / post-projection paths that still use (n, n) sizes.
        # * _nl_atol_vec / _nl_rtol_vec: expanded to stacked size; restored
        #   so the next call lazily re-expands at the right shape.
        # Shape-sensitive caches (_J_cached, _lu, _J_cross_call) are guarded
        # by shape checks in the solver and self-invalidate on mismatch, so
        # they are left alone to preserve SPLU reuse across coupled steps.
        saved_cold = solver._cold_start_slices
        saved_jac = solver.jacobian
        saved_atol_vec = solver._nl_atol_vec
        saved_rtol_vec = solver._nl_rtol_vec
        saved_component_slices = solver.component_slices
        saved_field_slices = solver.petsc_field_slices

        def _expand_slice(sl, off):
            start = (sl.start if sl.start is not None else 0) + off
            stop = (sl.stop if sl.stop is not None else n) + off
            return slice(start, stop, sl.step)

        def _expand_indices(idx, off):
            arr = np.arange(*idx.indices(n)) if isinstance(idx, slice) else np.asarray(idx)
            return arr + off

        if saved_cold:
            stacked_cs = []
            for stage_idx in range(s):
                off = stage_idx * n
                for sl in saved_cold:
                    stacked_cs.append(_expand_slice(sl, off))
            solver._cold_start_slices = stacked_cs

        if saved_component_slices:
            stacked_components = []
            for stage_idx in range(s):
                off = stage_idx * n
                for entry in saved_component_slices:
                    if isinstance(entry, slice):
                        stacked_components.append(_expand_slice(entry, off))
                    else:
                        stacked_components.append(np.asarray(entry) + off)
            solver.component_slices = stacked_components

        if saved_field_slices:
            stacked_fields = []
            for entry in saved_field_slices:
                merged = np.concatenate(
                    [_expand_indices(entry, stage_idx * n) for stage_idx in range(s)]
                )
                stacked_fields.append(merged)
            solver.petsc_field_slices = stacked_fields

        if solver._use_weighted_norm:
            atol_n, rtol_n = solver._ensure_nl_tol_vectors(n)
            solver._nl_atol_vec = np.tile(np.asarray(atol_n), s)
            solver._nl_rtol_vec = np.tile(np.asarray(rtol_n), s)

        solver.jacobian = J_stacked
        Z0 = np.tile(y, s)

        try:
            try:
                Z_sol, _F_sol, _err_sol, ok, iters = solver.solve(F_stacked, Z0)
            except Exception:
                return None
        finally:
            solver._cold_start_slices = saved_cold
            solver.jacobian = saved_jac
            solver._nl_atol_vec = saved_atol_vec
            solver._nl_rtol_vec = saved_rtol_vec
            solver.component_slices = saved_component_slices
            solver.petsc_field_slices = saved_field_slices

        Y_list = [Z_sol[i * n:(i + 1) * n].copy() for i in range(s)]
        f_stage_out = [
            _call_fun(
                t + rk_c[i] * h, Y_list[i], stage_h_override=rk_A[i, i] * h
            )
            for i in range(s)
        ]
        Fk_last = f_stage_out[-1]
        return Y_list, f_stage_out, Fk_last, bool(ok), int(iters)

    # ──────────────────────────────────────────────────────────────────────────
    # Main step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, fun, t, y, h):
        r"""Advance by one Radau IIA step of size *h*.

        For ``stages=1`` this is identical to :meth:`BackwardEuler.step`.

        For ``stages≥2`` the step consists of:

        1. **Waveform-relaxation stage loop** — up to ``wf_maxiter`` outer
           sweeps over the ``s`` stages, each solved as a Backward-Euler-like
           implicit equation augmented with explicit coupling contributions.
        2. **Post-step velocity projection** (optional) — applied if
           ``post_step_projection`` is set (Breuling Stage 2).

        Returns
        -------
        y_new : ndarray
            State at ``t + h``.  Equal to the last stage value ``Y_s``
            (stiff accuracy).
        Fk_new : ndarray or None
            Residual at convergence of the last stage solve.
        err : ndarray
            Element-wise embedded error estimate (mass-matrix agnostic).
        success : bool
            *True* only if every stage solve converged.
        iterations : int
            Total nonlinear iterations summed over all stage solves.
        """
        if self.projected_radau_contact is not None:
            return self.projected_radau_contact.step(self, t, y, h)

        # 1-stage: delegate unchanged to BackwardEuler
        if self.stages == 1:
            return super().step(fun, t, y, h)

        n = len(y)
        A_local = self._get_A(n)
        rk_A = self._rk_A   # shape (s, s)
        rk_c = self._rk_c   # shape (s,)
        s = self.stages

        # ── RHS / Jacobian wrapper (same pattern as SDIRK2) ────────────────
        prev_state_arg = y if self.pass_prev_state else None
        # pass_step_size: use full h as the outer scale hint;
        # each stage overrides to aii*h via explicit arg in _rhs calls below.
        step_size_arg = h if self.pass_step_size else None

        _rhs = self._get_bound_wrapper(
            fun,
            has_prev=(prev_state_arg is not None),
            has_h=(step_size_arg is not None),
            cache=self._fun_bindings,
        )

        def _call_fun(tt, yy, Fk=None, stage_h_override=None):
            # stage_h_override lets us pass the diagonal sub-step as h_val
            h_arg = stage_h_override if (stage_h_override is not None and step_size_arg is not None) else step_size_arg
            return _rhs(tt, yy, Fk, prev_state_arg, h_arg)

        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        _jac_wrapper = None
        if callable(rhs_jac) and getattr(self.solver, 'method', None) != 'VI':
            _jac_wrapper = self._get_bound_wrapper(
                rhs_jac,
                has_prev=(prev_state_arg is not None),
                has_h=(step_size_arg is not None),
                cache=self._jac_bindings,
            )

        # ── Diagonal mass-matrix ρ (proximal equilibration) ────────────────
        _A_diag = (A_local.diagonal() if sp.issparse(A_local) else np.diag(A_local))
        _phys_mask = np.abs(_A_diag) > 0.0

        def _set_stage_rho(aii):
            rho_vec = np.ones(n, dtype=float)
            rho_vec[_phys_mask] = (aii * h) / _A_diag[_phys_mask]
            self.solver.lam = rho_vec

        # ── Initial stage values and cached RHS evaluations ────────────────
        Y = [y.copy() for _ in range(s)]
        # f_stage[i] = fun(t + c_i*h, Y[i]) — re-evaluated after each stage solve
        f_stage = [
            _call_fun(t + rk_c[i] * h, Y[i], stage_h_override=rk_A[i, i] * h)
            for i in range(s)
        ]

        total_iters = 0
        Fk_last: np.ndarray | None = None
        ok_all = True
        _cn_used = False
        _implicit_i = None  # set by WF path; used for _residual_last

        # ── Coupled-Newton primary path (requires analytical Jacobian) ──────
        # Solves the fully coupled (s·n × s·n) block system in one Newton loop,
        # bypassing the waveform-relaxation coupling-ratio stability limit.
        if self.use_coupled_newton and _jac_wrapper is not None:
            cn_result = self._step_coupled_newton_impl(
                t, y, h, n, A_local, rk_A, rk_c, s,
                _call_fun, _jac_wrapper, prev_state_arg, step_size_arg,
            )
            if cn_result is not None:
                Y, f_stage, Fk_last, ok_all, total_iters = cn_result
                _cn_used = True
                if not ok_all:
                    return y, Fk_last, np.zeros(n), False, total_iters

        if not _cn_used:
            # ── Waveform-relaxation outer loop ────────────────────────────────
            # Fallback used when no analytical Jacobian is available (VI method).
            for wf_iter in range(self.wf_maxiter):
                Y_prev_wf = [Yi.copy() for Yi in Y]

                for i in range(s):
                    aii = rk_A[i, i]
                    stage_h = aii * h

                    # Explicit coupling: C_i = Σ_{j≠i} (a_{ij}/a_{ii}) * f_stage[j]
                    # On the first sweep (wf_iter==0) use zero coupling so each stage
                    # is solved as a backward-Euler sub-step at size a_{ii}*h.  This
                    # gives a stable DIRK predictor that avoids the destabilising effect
                    # of large negative cross-coupling terms (e.g. a[0,1] = −1/12 for
                    # s=2) when f_stage is initialised from a seeded or transient state.
                    # Subsequent sweeps correct the coupling to the full Radau value.
                    C_i = np.zeros(n, dtype=float)
                    if wf_iter > 0:
                        for j in range(s):
                            if j != i:
                                C_i += (rk_A[i, j] / aii) * f_stage[j]
                    # Freeze for closure
                    _C = C_i  # already a fresh array each iteration

                    # Stage i implicit residual:
                    #   A_M (Yi − y)/(a_{ii}·h) − f(t+c_i·h, Yi) − C_i = 0
                    # The constant C_i does NOT affect the Jacobian structure.
                    def _implicit_i(
                        Yi,
                        _Aloc=A_local,
                        _y=y,
                        _ci=rk_c[i],
                        _aii=aii,
                        _h=h,
                        _C=_C,
                    ):
                        return (
                            _Aloc @ ((Yi - _y) / (_aii * _h))
                            - _call_fun(t + _ci * _h, Yi, stage_h_override=_aii * _h)
                            - _C
                        )

                    # Swap in (or invalidate) the LU for this stage's step size
                    self._swap_stage_lu(stage_h)
                    # Proximal parameter equilibration for this stage
                    _set_stage_rho(aii)

                    # Exact Jacobian (if available): same as BackwardEuler at step=stage_h
                    if _jac_wrapper is not None:
                        A_over_sth = A_local / stage_h
                        def _jac_i(
                            Yi,
                            _A=A_over_sth,
                            _ci=rk_c[i],
                            _h=h,
                            _aii=aii,
                            _jw=_jac_wrapper,
                        ):
                            fk_val = getattr(self.solver, 'last_Fk_val', None)
                            J_rhs = _jw(
                                t + _ci * _h, Yi, fk_val,
                                prev_state_arg,
                                (_aii * _h) if step_size_arg is not None else None,
                            )
                            return _A - J_rhs

                        self.solver.jacobian = _jac_i

                    # Thread step-context
                    try:
                        self.solver.current_time = t + rk_c[i] * h
                        self.solver.prev_state = y
                        self.solver.prev_time = t
                        self.solver.prev_step = stage_h
                    except Exception:
                        pass

                    Yi_new, Fki, erri, oki, itsi = self.solver.solve(_implicit_i, Y[i])
                    total_iters += itsi

                    if not oki:
                        return y, Fki, np.zeros(n), False, total_iters

                    Y[i] = Yi_new
                    Fk_last = Fki
                    # Update cached RHS for stage i (needed for coupling in next stages/sweeps)
                    f_stage[i] = _call_fun(
                        t + rk_c[i] * h, Yi_new, stage_h_override=aii * h
                    )

                # Early exit if outer-loop converged
                if wf_iter > 0:
                    max_rel = max(
                        np.linalg.norm(Y[i] - Y_prev_wf[i])
                        / (1.0e-15 + np.linalg.norm(Y_prev_wf[i]))
                        for i in range(s)
                    )
                    if max_rel < self.wf_tol:
                        break

        # ── Stiffly accurate output: y_new = Y_s ───────────────────────────
        y_new = Y[-1].copy()

        # ── Post-step velocity projection (Breuling Stage 2) ───────────────
        # Define the last-stage residual for re-evaluating Fk after projection.
        if _cn_used:
            # Freeze coupling from converged coupled-Newton solution
            _f_coup = list(f_stage)
            _aii_s = rk_A[s - 1, s - 1]
            _ci_s = rk_c[s - 1]

            def _residual_last(Yev,
                               _A=A_local, _y=y, _h=h,
                               _ci=_ci_s, _aii=_aii_s,
                               _fc=_f_coup, _s=s, _rkA=rk_A):
                f_ev = _call_fun(t + _ci * _h, Yev, stage_h_override=_aii * _h)
                C_last = np.zeros(len(Yev), dtype=float)
                for jj in range(_s - 1):
                    C_last += (_rkA[_s - 1, jj] / _aii) * _fc[jj]
                return _A @ ((Yev - _y) / (_aii * _h)) - f_ev - C_last
        else:
            # WF path: _implicit_i is the closure from the last stage solve
            def _residual_last(Yev):
                return _implicit_i(Yev)

        y_new, Fk_last, proj_delta = self._apply_post_step_projection(
            y,
            y_new,
            t_new=t + h,
            h=h,
            Fk_val=Fk_last,
            residual_eval=_residual_last,
        )
        if proj_delta is not None:
            try:
                self.last_post_step_delta = np.asarray(proj_delta, dtype=float)
            except Exception:
                self.last_post_step_delta = None

        # ── Embedded error estimate (mass-matrix agnostic) ─────────────────
        # Uses precomputed coefficients from __init__:
        #   err = Σ_k _err_coeffs[k] · (Y[k] − y)
        # Derived from the difference between the main method (b = Radau weights)
        # and a first-order companion (b̂ = [1, 0, …]) expressed in terms of
        # stage increments via the Butcher A inverse.  Scales as O(h²) for all
        # stage counts (leading term of the order-1 companion LTE).
        # The adaptive controller uses exponent 1/(embedded_order+1) = 1/2 via
        # the ``embedded_order`` attribute, giving correct step-size selection.
        coeffs = self._err_coeffs          # shape (s,)
        err_embed = np.zeros(n, dtype=float)
        for k in range(s):
            err_embed += coeffs[k] * (Y[k] - y)

        return y_new, Fk_last, err_embed, True, total_iters


class EmbeddedBETR(IntegrationMethod):
    """
    Trapezoidal implicit integrator (kept under the historical name EmbeddedBETR).

    Acts as a plain TR stepper with optional exact residual Jacobian.
    """

    def __init__(self, solver=None, A=None):
        self.solver = solver or ImplicitEquationSolver(method='semismooth_newton')
        self.A = A
        self.use_identity = (A is None)
        self.order = 2

    def _get_A(self, n):
        if not self.use_identity:
            return self.A
        want_sparse = False
        try:
            want_sparse = bool(self.solver._sparse_active(n))
        except Exception:
            thr = getattr(self.solver, 'sparse_threshold', 200)
            try:
                want_sparse = (n >= int(thr))
            except Exception:
                want_sparse = (n >= 200)
        if want_sparse:
            return sp.eye(n, format='csr')
        return np.eye(n)

    def _attach_be_jac(self, A_local, t, h):
        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac):
            def jac_eq(y_new, _A=A_local, _h=h, _t=t, _rhs_jac=rhs_jac):
                return (_A / _h) - _rhs_jac(_t + _h, y_new)
            self.solver.jacobian = jac_eq

    def _attach_tr_jac(self, A_local, t, h):
        rhs_jac = getattr(self.solver, 'rhs_jacobian', None)
        if callable(rhs_jac):
            def jac_eq(y_new, _A=A_local, _h=h, _t=t, _rhs_jac=rhs_jac, _solver=self.solver):
                try:
                    return (_A / _h) - 0.5 * _rhs_jac(_t + _h, y_new, getattr(_solver, 'last_Fk_val', None))
                except TypeError:
                    return (_A / _h) - 0.5 * _rhs_jac(_t + _h, y_new)
            self.solver.jacobian = jac_eq

    def step(self, fun, t, y, h):
        """Single TR step (legacy name kept for backward compatibility).

        Returns
        -------
        y_new, Fk_new, err_new, success, iterations : standard solver 5-tuple.
        """
        n = len(y)
        A_local = self._get_A(n)

        def _call_fun(f, tt, yy, Fk=None):
            try:
                return f(tt, yy, Fk)
            except TypeError:
                try:
                    return f(tt, yy)
                except TypeError:
                    return f(yy)

        f_n = _call_fun(fun, t, y, getattr(self.solver, 'last_Fk_val', None))
        implicit_tr = lambda y_new: A_local @ ((y_new - y) / h) - 0.5 * (
            f_n + _call_fun(fun, t + h, y_new, getattr(self.solver, 'last_Fk_val', None))
        )
        # Attach TR residual Jacobian using shared helper
        self._attach_tr_jac(A_local, t, h)

        # Thread step-context
        try:
            self.solver.current_time = t + h
            self.solver.prev_state = y
            self.solver.prev_time = t
            self.solver.prev_step = h
        except Exception:
            pass

        return self.solver.solve(implicit_tr, y)


if __name__ == "__main__":
    # Optional quick smoke test when running this module directly
    def rhs(t, y, Fk=None):
        return -y

    y0 = np.array([1.0])
    t0 = 0.0
    h = 0.2
    be = BackwardEuler(solver=ImplicitEquationSolver(method='semismooth_newton', proj=lambda *args, **kw: None))
    tr = Trapezoidal(solver=ImplicitEquationSolver(method='semismooth_newton', proj=lambda *args, **kw: None))
    comp = CompositeMethod(solver=ImplicitEquationSolver(method='semismooth_newton', proj=lambda *args, **kw: None))
    print("TR:", tr.step(rhs, t0, y0.copy(), h))
    print("Composite:", comp.step(rhs, t0, y0.copy(), h))
