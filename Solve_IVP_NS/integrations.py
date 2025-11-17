import numpy as np
import math
from abc import ABC, abstractmethod
import scipy.sparse as sp
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
    # Keys: ('dense'| 'csr', n)
    _ID_CACHE = {}

    def __init__(self, solver=None, A=None, pass_prev_state=False, pass_step_size=False):
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

        return self.solver.solve(implicit_eq, y)


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

        rhs_jac = getattr(self.backward_euler.solver, 'rhs_jacobian', None)
        if callable(rhs_jac) and getattr(self.backward_euler.solver, 'method', None) != 'VI':
            A_over_h = self.backward_euler._get_A(len(y)) / h

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
