# nonlinear_solvers.py  (fast-path projector dispatch)

from __future__ import annotations

import inspect
import logging
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
import math

# Optional numba acceleration for hot kernels
try:
    from ._numba_accel import NUMBA_AVAILABLE as _NUMBA_OK, wrms_kernel as _wrms_numba
except Exception:
    _NUMBA_OK = False
    _wrms_numba = None

# Configure logger for PETSc timing (works in VS Code Jupyter)
_petsc_logger = logging.getLogger('solve_nivp.petsc')
_petsc_logger.setLevel(logging.DEBUG)
if not _petsc_logger.handlers:
    _handler = logging.StreamHandler()  # stderr by default
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _petsc_logger.addHandler(_handler)

# Optional UMFPACK support (much faster than SuperLU for n > ~20k)
try:
    from scikits.umfpack import UmfpackContext, UMFPACK_A as _UMFPACK_A
    UMFPACK_AVAILABLE = True
except Exception:          # ImportError, OSError, or missing shared lib
    UMFPACK_AVAILABLE = False
    UmfpackContext = None
    _UMFPACK_A = None

# Optional PETSc support
try:
    from petsc4py import PETSc
    PETSC_AVAILABLE = True
except ImportError:
    PETSC_AVAILABLE = False
    PETSc = None

_PETSC_GPU_MAT_TYPES = frozenset({'aijcusparse', 'aijkokkos'})
_PETSC_GPU_VEC_TYPES = frozenset({'cuda', 'kokkos'})
_PETSC_GPU_PAIR_FOR_MAT = {
    'aijcusparse': 'cuda',
    'aijkokkos': 'kokkos',
}
_PETSC_GPU_PAIR_FOR_VEC = {
    'cuda': 'aijcusparse',
    'kokkos': 'aijkokkos',
}
_PETSC_TYPE_SUPPORT_CACHE = {}

class ImplicitEquationSolver:
    """Solve F(y)=0 with projection-aware VI or semismooth Newton (fast path)."""

    def __init__(
        self,
        method: str = 'semismooth_newton',
        jacobian=None,
        tol: float = 1e-10,
        max_iter: int = 100,
        proj=None,
        rho0: float = 0.9,
        delta: float = 0.7,
        component_slices=None,
        L: float = 0.9,
        Lmin: float = 0.3,
        nu: float = 0.66,
        lam: float = 1.0,
        lam_min: float = 1e-6,
        sparse: bool | str = 'auto',
        sparse_threshold: int = 200,
        linear_solver: str = 'gmres',
        precond_reuse_steps: int = 20,
        ilu_drop_tol: float = 1e-4,
        ilu_fill_factor: float = 10.0,
        gmres_tol: float = 1e-6,
        gmres_maxiter: int | None = None,
        gmres_restart: int | None = None,
        splu_permc_spec: str = 'COLAMD',
        ilu_permc_spec: str = 'COLAMD',
        linear_tol_strategy: str = 'fixed',
        eisenstat_c: float = 0.5,
        eisenstat_exp: float = 0.5,
        adaptive_lam: bool = True,
        lam_update_strategy: str = 'vi',
        globalization: str = 'none',
        ls_c1: float = 1e-4,
        ls_beta: float = 0.5,
        ls_min_alpha: float = 1e-8,
        max_backtracks: int = 15,
        use_broyden: bool = False,
        # VI strict per-block Lipschitz enforcement (opt-in)
        vi_strict_block_lipschitz: bool = True,
        vi_max_block_adjust_iters: int = 10,
        # PETSc options
        petsc_options: dict | None = None,
        petsc_reuse_steps: int = 10,
        petsc_comm='self',
        # Per-DOF tolerance vectors (SUNDIALS convention) — opt-in
        nl_atol=None,
        nl_rtol=None,
        # Jacobian equilibration for improved conditioning
        jacobian_scaling: str = 'none',
        jacobian_sparsity=None,
    ) -> None:
        if method not in ['VI', 'semismooth_newton']:
            raise ValueError("Unsupported solver method. Use 'VI' or 'semismooth_newton'.")
        self.method = method
        self.jacobian = jacobian
        self.tol = tol
        self.max_iter = max_iter
        self.proj = proj
        self.delta = delta
        self.component_slices = component_slices
        self.L = L
        self.Lmin = Lmin
        self.nu = nu
        self.lam = lam
        self._lam_floor = float(lam_min)

        # Sparse / linear solver configuration
        self.sparse = sparse
        self.sparse_threshold = int(sparse_threshold)
        self.linear_solver = (linear_solver or 'gmres').lower()
        self.precond_reuse_steps = max(0, int(precond_reuse_steps))
        self.ilu_drop_tol = ilu_drop_tol
        self.ilu_fill_factor = ilu_fill_factor
        self.gmres_tol = gmres_tol
        self.gmres_maxiter = gmres_maxiter
        self.gmres_restart = gmres_restart
        self.splu_permc_spec = splu_permc_spec
        self.ilu_permc_spec = ilu_permc_spec
        self.linear_tol_strategy = (linear_tol_strategy or 'fixed').lower()
        self.eisenstat_c = float(eisenstat_c)
        self.eisenstat_exp = float(eisenstat_exp)
        self.adaptive_lam = bool(adaptive_lam)
        strategy = (lam_update_strategy or 'vi').lower()
        if strategy not in ('vi', 'none'):
            raise ValueError(
                "Unsupported lam_update_strategy. Use 'vi' for variational inequality adaptation or 'none'."
            )
        if self.adaptive_lam and strategy == 'none':
            self.adaptive_lam = False
        self.lam_update_strategy = strategy
        self.globalization = (globalization or 'none').lower()
        if self.globalization not in ('none', 'linesearch'):
            self.globalization = 'none'
        self.ls_c1 = float(ls_c1)
        self.ls_beta = float(ls_beta)
        self.ls_min_alpha = float(ls_min_alpha)
        self.max_backtracks = int(max_backtracks)

        # Quasi-Newton (dense)
        self.use_broyden = bool(use_broyden)
        self._B = None
        self._y_prev_broyden = None
        self._F_prev_broyden = None

        # VI strict block Lipschitz options
        self.vi_strict_block_lipschitz = bool(vi_strict_block_lipschitz)
        self.vi_max_block_adjust_iters = int(vi_max_block_adjust_iters)

        # PETSc configuration
        # Default: direct LU via MUMPS — robust for all matrix types
        # including indefinite / saddle-point systems.  MUMPS uses METIS
        # nested-dissection ordering (much better fill-reducing than COLAMD
        # for mixed FE systems) and is multithreaded via OpenMP.
        # For iterative solving, pass e.g.:
        #   petsc_options={'ksp_type': 'gmres', 'pc_type': 'hypre',
        #                  'pc_hypre_type': 'boomeramg'}
        self.petsc_options = petsc_options if petsc_options is not None else {
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        self.petsc_reuse_steps = int(petsc_reuse_steps)
        self.petsc_comm = petsc_comm
        self._petsc_ksp = None
        self._petsc_mat = None
        self._petsc_build_count = 0
        self._petsc_shape = None
        self._petsc_field_is = None  # Index sets for field-split
        self._petsc_needs_matrix_update = False  # set True when Newton recomputes J
        self._petsc_use_gpu = False
        self._petsc_comm_obj = None
        self._petsc_effective_mat_type = None
        self._petsc_effective_vec_type = None
        self._petsc_gpu_warned = set()

        # Rho adaptation safeguards (bounds and "stuck" thresholds)
        # These are conservative defaults; they can be adjusted by users after construction if needed.
        self.rho_min = 1e-12
        self.rho_max = 1e6
        # A component is considered "stuck" if the change is below this absolute threshold times a scale
        self.stuck_eps_abs = 1e-14
        # Optional relative scale for stuck detection; set small to avoid false positives on tiny states
        self.stuck_eps_rel = 1e-12

        # GMRES preconditioner cache
        self._ilu = None
        self._ilu_steps_since_build = 0
        self._last_shape = None
        self._last_pattern = None

        # SPLU / UMFPACK cache
        self._lu = None
        self._lu_use_count = 0
        self._lu_pattern = None
        self._lu_shape = None
        self._lu_matrix = None          # UMFPACK needs original CSC for solve
        self._umf_ctx = None            # persists across h-change invalidations
        self._umf_symbolic_key = None   # (shape, nnz) — reuse symbolic analysis

        # Pre-allocated workspace for _wrms to avoid repeated temporaries
        self._wrms_buf = None

        # ---- Jacobian equilibration ----
        _js = (jacobian_scaling or 'none').lower()
        if _js not in ('none', 'row', 'ruiz'):
            raise ValueError(
                f"jacobian_scaling must be 'none', 'row', or 'ruiz', got '{jacobian_scaling}'"
            )
        self.jacobian_scaling = _js
        self._eq_Dr = None   # row scaling vector (cached)
        self._eq_Dc = None   # column scaling vector (cached)

        # Cross-call Jacobian cache for non-SPLU iterative solvers.
        # When linear_solver is 'gmres' or 'petsc', the identity Newton
        # path cannot rely on a cached SPLU factor for cross-call modified
        # Newton.  Instead it caches the last Jacobian CSR here so that
        # back-to-back solve() calls (e.g. SDIRK stages) can skip the
        # (potentially expensive) Jacobian evaluation.
        self._J_cross_call = None

        # Identity caches
        self._I_cache = {}

        # Jacobian structure cache (for dense-to-sparse optimization)
        self._J_cached = None
        self._J_rows = None
        self.jacobian_sparsity = None
        self._jacobian_sparsity = None
        self.set_jacobian_sparsity(jacobian_sparsity)

        # Tangent structure cache (for dense-to-sparse optimization)
        self._D_cached = None
        self._D_rows = None

        # Exact sparse assembly cache for diagonal-tangent Newton operators
        self._diag_newton_key = None
        self._diag_newton_row_idx = None
        self._diag_newton_diag_pos = None
        self._diag_newton_out = None

        # Initialize rho state (scalar default + structured cache)
        self._set_initial_rho(rho0)

        # ---- Per-DOF weighted-norm convergence (SUNDIALS convention) ----
        # When nl_atol / nl_rtol are provided, convergence is tested via a
        # weighted RMS norm  ``wrms(F, y) = sqrt(mean((F_i / w_i)^2)) <= 1``
        # with ``w_i = nl_atol_i + nl_rtol_i * |y_i|``.
        # When *not* provided (default), the legacy ``||F|| < tol`` test is used.
        self._nl_atol_raw = nl_atol      # None, scalar, or array_like
        self._nl_rtol_raw = nl_rtol
        self._use_weighted_norm: bool = (nl_atol is not None or nl_rtol is not None)
        self._nl_atol_vec: np.ndarray | None = None
        self._nl_rtol_vec: np.ndarray | None = None

        # Basic checks
        if self.method == 'VI':
            if self.proj is None:
                raise ValueError("Projection operator 'proj' must be provided for method 'VI'.")
            if self.component_slices is None:
                raise ValueError("component_slices must be provided for method 'VI'.")
        if self.method == 'semismooth_newton' and self.proj is None:
            raise ValueError("Projection operator 'proj' must be provided for 'semismooth_newton'.")

        # ---- Bind fast projector dispatchers once (no per-iteration try/except) ----
        self._bind_projector_fastpaths()

    # ---------- Fastpath binding ----------
    def _bind_projector_fastpaths(self):
        """Bind self._project and self._tangent with only the args supported by the projector."""
        if self.proj is None:
            self._project = None
            self._tangent = None
            return

        def _supports(fn, name):
            try:
                return name in inspect.signature(fn).parameters
            except Exception:
                return False

        P = self.proj
        _self = self  # avoid closure over self for step_size lookup

        # --- detect what the projector exposes ---
        has_prev_p = _supports(P.project, 'prev_state')
        has_step_p = _supports(P.project, 'step_size')
        has_rhok_p = _supports(P.project, 'rhok')          # keyword form
        has_rho_p  = _supports(P.project, 'rho')           # positional/keyword form
        has_t_p    = _supports(P.project, 't')
        has_Fk_p   = _supports(P.project, 'Fk_val')

        # ---- PROJECT BINDER ----
        # Build a specialized closure that directly passes only the needed args,
        # avoiding the construction of a kwargs dict on every call.
        _proj_fn = P.project

        # Precompute a bitmask of which kwargs are needed
        # bits: 1=t, 2=Fk, 4=step_size, 8=prev_state, 16=rhok-keyword, 32=rho-positional
        p_mask = 0
        if has_t_p:    p_mask |= 1
        if has_Fk_p:   p_mask |= 2
        if has_step_p: p_mask |= 4
        if has_prev_p: p_mask |= 8
        if has_rhok_p: p_mask |= 16
        elif has_rho_p: p_mask |= 32

        if p_mask == 16:  # common: just rhok keyword
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand, rhok=rho)
        elif p_mask == 32:  # just rho positional
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand, rho)
        elif p_mask == 0:  # no extras
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand)
        elif p_mask == (1 | 2 | 16):  # t + Fk + rhok
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand, rhok=rho, t=t, Fk_val=Fk)
        elif p_mask == (1 | 2 | 8 | 16):  # t + Fk + prev + rhok
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand, rhok=rho, t=t, Fk_val=Fk, prev_state=prev)
        elif p_mask == (1 | 2 | 4 | 8 | 16):  # everything with rhok
            def _project(cur, cand, rho, t, Fk, prev):
                return _proj_fn(cur, cand, rhok=rho, t=t, Fk_val=Fk,
                                step_size=getattr(_self, 'prev_step', None), prev_state=prev)
        else:
            # Generic fallback — still builds a dict, but rare
            def _project(cur, cand, rho, t, Fk, prev):
                kw = {}
                if has_t_p:   kw['t'] = t
                if has_Fk_p:  kw['Fk_val'] = Fk
                if has_step_p: kw['step_size'] = getattr(_self, 'prev_step', None)
                if has_prev_p: kw['prev_state'] = prev
                if has_rhok_p:
                    kw['rhok'] = rho
                    return _proj_fn(cur, cand, **kw)
                elif has_rho_p:
                    return _proj_fn(cur, cand, rho, **kw)
                else:
                    return _proj_fn(cur, cand, **kw)

        self._project = _project

        # ---- TANGENT BINDER ----
        has_prev_t = _supports(P.tangent_cone, 'prev_state')
        has_step_t = _supports(P.tangent_cone, 'step_size')
        has_rhok_t = _supports(P.tangent_cone, 'rhok')
        has_rho_t  = _supports(P.tangent_cone, 'rho')
        has_t_t    = _supports(P.tangent_cone, 't')
        has_Fk_t   = _supports(P.tangent_cone, 'Fk_val')

        _tang_fn = P.tangent_cone
        t_mask = 0
        if has_t_t:    t_mask |= 1
        if has_Fk_t:   t_mask |= 2
        if has_step_t: t_mask |= 4
        if has_prev_t: t_mask |= 8
        if has_rhok_t: t_mask |= 16
        elif has_rho_t: t_mask |= 32

        if t_mask == 16:
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur, rhok=rho)
        elif t_mask == 32:
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur, rho)
        elif t_mask == 0:
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur)
        elif t_mask == (1 | 2 | 16):
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur, rhok=rho, t=t, Fk_val=Fk)
        elif t_mask == (1 | 2 | 8 | 16):
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur, rhok=rho, t=t, Fk_val=Fk, prev_state=prev)
        elif t_mask == (1 | 2 | 4 | 8 | 16):
            def _tangent(cand, cur, rho, t, Fk, prev):
                return _tang_fn(cand, cur, rhok=rho, t=t, Fk_val=Fk,
                                step_size=getattr(_self, 'prev_step', None), prev_state=prev)
        else:
            def _tangent(cand, cur, rho, t, Fk, prev):
                kw = {}
                if has_t_t:   kw['t'] = t
                if has_Fk_t:  kw['Fk_val'] = Fk
                if has_step_t: kw['step_size'] = getattr(_self, 'prev_step', None)
                if has_prev_t: kw['prev_state'] = prev
                if has_rhok_t:
                    kw['rhok'] = rho
                    return _tang_fn(cand, cur, **kw)
                elif has_rho_t:
                    return _tang_fn(cand, cur, rho, **kw)
                else:
                    return _tang_fn(cand, cur, **kw)

        self._tangent = _tangent

        # Identity projection fast-path flag: when True, solve() bypasses all
        # projection machinery (tangent, lam adaptation, sparse assembly) and
        # runs a standard Newton or Richardson iteration directly.
        self._is_identity_proj = (
            type(self.proj).__name__ == 'IdentityProjection'
            or getattr(self.proj, 'is_identity', False)
        )

        # Rho-independent projections (e.g. AlgebraicConstraintProjection)
        # enforce constraints regardless of the rhok / lam parameter, so
        # Lipschitz-based lam adaptation wastes func evals without benefit.
        self._is_rho_independent = getattr(self.proj, 'rho_independent', False)

    def _normalize_jacobian_sparsity(self, sparsity):
        if sparsity is None:
            return None
        if sp.issparse(sparsity):
            return sparsity.tocsr()
        arr = np.asarray(sparsity)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("jacobian_sparsity must be a square 2-D array or sparse matrix")
        return sp.csr_matrix(arr)

    def set_jacobian_sparsity(self, sparsity) -> None:
        """Register a public Jacobian sparsity pattern for colored FD."""
        norm = self._normalize_jacobian_sparsity(sparsity)
        self.jacobian_sparsity = norm
        # Keep the legacy private alias for backwards compatibility.
        self._jacobian_sparsity = norm

    def _component_indices(self, field, n: int | None = None, dtype=int) -> np.ndarray:
        """Normalize a component partition entry to a 1-D index array."""
        if isinstance(field, slice):
            start = 0 if field.start is None else int(field.start)
            if field.stop is None:
                if n is None:
                    raise ValueError("Open-ended component slice requires problem size")
                stop = int(n)
            else:
                stop = int(field.stop)
            step = 1 if field.step is None else int(field.step)
            return np.arange(start, stop, step, dtype=dtype)

        arr = np.asarray(field)
        if arr.ndim == 0:
            return np.array([arr.item()], dtype=dtype)
        if arr.dtype == bool:
            if n is None and arr.ndim != 1:
                raise ValueError("Boolean component mask requires problem size")
            return np.flatnonzero(arr).astype(dtype, copy=False)
        return np.asarray(arr, dtype=dtype).ravel()

    # ---------- Per-DOF weighted-norm helpers ----------

    def _ensure_nl_tol_vectors(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(nl_atol_vec, nl_rtol_vec)`` of length *n*.

        Lazily expands the raw tolerances provided at construction:

        * **scalar** → broadcast to length *n*.
        * **per-slice sequence** (length == ``len(component_slices)``) →
          expanded to per-DOF via the slice mapping.
        * **per-DOF array** (length == *n*) → used directly.
        """
        if self._nl_atol_vec is not None and self._nl_atol_vec.shape == (n,):
            return self._nl_atol_vec, self._nl_rtol_vec

        # Defaults when only one of atol/rtol was supplied
        raw_a = self._nl_atol_raw if self._nl_atol_raw is not None else self.tol
        raw_r = self._nl_rtol_raw if self._nl_rtol_raw is not None else 0.0

        self._nl_atol_vec = self._expand_nl_tol(raw_a, n, 'nl_atol')
        self._nl_rtol_vec = self._expand_nl_tol(raw_r, n, 'nl_rtol')
        return self._nl_atol_vec, self._nl_rtol_vec

    def _expand_nl_tol(self, raw, n: int, name: str) -> np.ndarray:
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            return np.full(n, float(arr))
        if arr.shape == (n,):
            return arr.copy()
        if self.component_slices is not None and arr.size == len(self.component_slices):
            out = np.empty(n, dtype=float)
            for val, sl in zip(arr.ravel(), self.component_slices):
                out[sl] = float(val)
            return out
        raise ValueError(
            f"{name} has length {arr.size}, expected scalar, "
            f"len(component_slices)={len(self.component_slices) if self.component_slices else 'N/A'}, "
            f"or n={n}"
        )

    def _wrms(self, F: np.ndarray, y: np.ndarray) -> float:
        """Weighted RMS norm: ``sqrt(mean((F_i / w_i)^2))`` with ``w_i = atol_i + rtol_i*|y_i|``.

        Returns a value comparable to the ``E <= 1`` convention used by
        SUNDIALS CVODE for Newton convergence.
        """
        n = F.size
        atol_v, rtol_v = self._ensure_nl_tol_vectors(n)
        # Fast path: single-pass numba kernel (no intermediate arrays)
        if _NUMBA_OK and _wrms_numba is not None:
            return float(_wrms_numba(F, y, atol_v, rtol_v))
        # Reuse pre-allocated buffer to avoid 3 temporary arrays per call
        buf = self._wrms_buf
        if buf is None or buf.size != n:
            buf = self._wrms_buf = np.empty(n, dtype=float)
        np.abs(y, out=buf)       # buf = |y|
        buf *= rtol_v             # buf = rtol * |y|
        buf += atol_v             # buf = atol + rtol * |y|  (= weights)
        np.divide(F, buf, out=buf)  # buf = F / weights
        return float(math.sqrt(np.dot(buf, buf) / n))

    def _converged(self, F: np.ndarray, y: np.ndarray) -> bool:
        """Check nonlinear convergence using either legacy or weighted-norm test.

        * Legacy (default): ``||F|| < self.tol``
        * Weighted-norm (when ``nl_atol`` or ``nl_rtol`` was provided):
          ``wrms(F, y) <= 1``
        """
        if self._use_weighted_norm:
            return self._wrms(F, y) <= 1.0
        return float(np.linalg.norm(F)) < self.tol

    def _errf_metric(self, F: np.ndarray, y: np.ndarray) -> float:
        """Return the convergence metric value (for reporting / modified Newton logic)."""
        if self._use_weighted_norm:
            return self._wrms(F, y)
        return float(np.linalg.norm(F))

    def _converged_with_metric(self, F: np.ndarray, y: np.ndarray) -> tuple[bool, float]:
        """Compute convergence metric and test in a single call.

        Returns ``(is_converged, errF)`` without evaluating the norm twice.
        This replaces the common pattern::

            errF = self._errf_metric(F, y)
            if self._converged(F, y):  # re-computes the same norm
        """
        if self._use_weighted_norm:
            val = self._wrms(F, y)
            return (val <= 1.0, val)
        val = float(np.linalg.norm(F))
        return (val < self.tol, val)


    # ---------- Jacobian equilibration ----------

    def _equilibrate(self, J_csr):
        r"""Row (and optionally column) equilibrate a CSR Jacobian.

        Parameters
        ----------
        J_csr : scipy.sparse.csr_matrix
            Jacobian in CSR format.

        Returns
        -------
        J_eq : scipy.sparse.csr_matrix
            Equilibrated Jacobian.
        Dr : ndarray
            Row scaling vector such that ``J_eq = diag(Dr) @ J @ diag(Dc)``.
        Dc : ndarray
            Column scaling vector (ones for ``'row'`` mode).
        """
        n = J_csr.shape[0]
        mode = self.jacobian_scaling

        def _dense_ravel(sparse_or_matrix):
            """Convert sparse/matrix .max() result to a 1-D dense array."""
            if sp.issparse(sparse_or_matrix):
                return np.asarray(sparse_or_matrix.toarray()).ravel()
            return np.asarray(sparse_or_matrix).ravel()

        if mode == 'row':
            # Row equilibration: normalise each row's infinity-norm to 1.
            abs_J = J_csr.copy()
            abs_J.data = np.abs(abs_J.data)
            row_max = _dense_ravel(abs_J.max(axis=1))
            row_max = np.where(row_max > 0, row_max, 1.0)
            Dr = 1.0 / row_max

            J_eq = J_csr.copy()
            if J_eq.nnz > 0:
                row_idx = np.repeat(np.arange(n), np.diff(J_eq.indptr))
                J_eq.data = J_eq.data * Dr[row_idx]
            return J_eq, Dr, np.ones(n)

        elif mode == 'ruiz':
            # Ruiz iterative symmetric scaling (5 iterations).
            # After convergence both row and column infinity-norms ≈ 1.
            J_eq = J_csr.copy()
            Dr_total = np.ones(n)
            Dc_total = np.ones(n)

            for _ in range(5):
                abs_J = J_eq.copy()
                abs_J.data = np.abs(abs_J.data)

                row_max = _dense_ravel(abs_J.max(axis=1))
                row_max = np.where(row_max > 0, row_max, 1.0)
                Dr = 1.0 / np.sqrt(row_max)

                col_max = _dense_ravel(abs_J.max(axis=0))
                col_max = np.where(col_max > 0, col_max, 1.0)
                Dc = 1.0 / np.sqrt(col_max)

                if J_eq.nnz > 0:
                    row_idx = np.repeat(np.arange(n), np.diff(J_eq.indptr))
                    J_eq.data = J_eq.data * Dr[row_idx] * Dc[J_eq.indices]

                Dr_total *= Dr
                Dc_total *= Dc

            return J_eq, Dr_total, Dc_total

        # 'none' — identity (should not normally be called)
        return J_csr, np.ones(n), np.ones(n)

    def _apply_jacobian_scaling(self, J_csr):
        """Equilibrate *J_csr* and cache the scaling vectors on *self*.

        Returns the equilibrated matrix.  Subsequent calls to
        :meth:`_scale_rhs` and :meth:`_unscale_solution` use the cached
        vectors until this method is called again.
        """
        J_eq, self._eq_Dr, self._eq_Dc = self._equilibrate(J_csr)
        return J_eq

    def _scale_rhs(self, rhs):
        """Scale a right-hand side vector with the cached row scaling."""
        if self.jacobian_scaling == 'none' or self._eq_Dr is None:
            return rhs
        return self._eq_Dr * rhs

    def _unscale_solution(self, x):
        """Undo column scaling on the linear-solve result.

        For ``'row'`` mode the column scaling is the identity, so this
        is a no-op.  For ``'ruiz'`` mode the solution is multiplied by
        the column scaling vector.
        """
        if self.jacobian_scaling in ('none', 'row') or self._eq_Dc is None:
            return x
        return self._eq_Dc * x

    # ---------- Rho helpers ----------
    def _rho_scalar_value(self, rho):
        arr = np.asarray(rho, dtype=float)
        if arr.ndim == 0:
            val = float(arr)
        else:
            val = float(np.mean(arr))
        if not np.isfinite(val) or val <= 0.0:
            val = 1.0
        val = float(np.clip(val, self.rho_min, self.rho_max))
        return val

    def _set_rho_last(self, rho, *, update_default=False):
        rho_struct = float(rho) if np.isscalar(rho) else np.asarray(rho, dtype=float).copy()
        self.rho_last = rho_struct
        if update_default:
            self.rho0 = self._rho_scalar_value(rho_struct)

    def _set_initial_rho(self, rho):
        self._set_rho_last(rho, update_default=True)


    # ---------- Public API ----------
    def _func_wrapper(self, y):
        return self.func(y)

    def set_func(self, func):
        self.func = func

    def set_projection(self, proj):
        """Allow swapping projector at runtime, with fastpath rebind."""
        self.proj = proj
        self._bind_projector_fastpaths()

    def invalidate_all_caches(self):
        """Force-discard every cached factorisation, Jacobian, and PETSc KSP.

        This is called by the adaptive stepper when many consecutive NL
        failures indicate that stale solver state is contributing to
        divergence.  The next ``solve()`` call will compute a fresh Jacobian,
        fresh factorisation (SPLU/UMFPACK/MUMPS), and fresh preconditioner.
        """
        # SPLU / UMFPACK cache
        self._lu = None
        self._lu_shape = None
        self._lu_pattern = None
        self._lu_use_count = 0
        self._lu_matrix = None

        # Cross-call Jacobian cache (for non-SPLU paths)
        self._J_cross_call = None

        # GMRES/ILU preconditioner cache
        self._ilu = None
        self._ilu_steps_since_build = 0
        self._last_shape = None
        self._last_pattern = None

        # PETSc KSP / Mat / factorisation — must be destroyed to release
        # MUMPS memory and force a full refactorisation.
        self._petsc_needs_matrix_update = False
        if self._petsc_ksp is not None:
            try:
                self._petsc_ksp.destroy()
            except Exception:
                pass
            self._petsc_ksp = None
        if self._petsc_mat is not None:
            try:
                self._petsc_mat.destroy()
            except Exception:
                pass
            self._petsc_mat = None
        self._petsc_build_count = 0
        self._petsc_shape = None
        self._petsc_field_is = None
        self._petsc_use_gpu = False
        self._petsc_comm_obj = None
        self._petsc_effective_mat_type = None
        self._petsc_effective_vec_type = None

        # Jacobian equilibration cache
        self._eq_Dr = None
        self._eq_Dc = None

    def solve(self, func, y0):
        self.set_func(func)
        if self.method == 'VI':
            if self._is_identity_proj:
                return self._solve_vi_identity(func, y0)
            return self._solve_with_VI(func, y0)
        else:
            if self._is_identity_proj:
                return self._solve_newton_identity(func, y0)
            if self._is_rho_independent:
                return self._solve_newton_algebraic(func, y0)
            return self._solve_with_semismooth_newton(func, y0)

    # ================================================================
    # Algebraic-constraint fast path (direct Newton with row patching)
    # ================================================================

    def _solve_newton_algebraic(self, func, y0):
        r"""Newton solver for algebraic-constraint projections.

        Bypasses the general semismooth Newton assembly
        ``J = I - D + lam·D·J_func`` (which requires an expensive sparse
        matrix–matrix product ``D @ J_func`` plus three additional sparse
        additions) and instead directly patches the zero constraint rows
        of the iteration-matrix Jacobian with the algebraic constraint
        equations.

        The augmented system solved at each Newton step is::

            F_field(y) = 0          (implicit equation — field rows)
            q - g(y_field) = 0      (algebraic constraint — constraint rows)

        whose Jacobian is::

            J_mod[field, :]      = J_func[field, :]        (unchanged)
            J_mod[q_i, q_i]      = +1                      (identity diagonal)
            J_mod[q_i, y_field]  = -∂g/∂y                  (constraint Jacobian)

        Since ``J_func`` already has zero rows on the constraint DOFs
        (mass and stiffness stripped), the patch is an additive sparse
        correction — O(nnz_patch) rather than O(nnz(D) × nnz_row(J)).

        SPLU caching is delegated to ``_solve_linear_sparse`` so the
        factorisation reuse policy (``precond_reuse_steps``) is identical
        to the general semismooth Newton path.  When the SPLU will be
        reused, Jacobian computation and row-patching are skipped
        entirely.
        """
        y = np.asarray(y0, dtype=float).reshape(-1).copy()
        n = len(y)
        sparse_active = self._sparse_active(n)
        P = self.proj
        _has_patch = hasattr(P, 'build_constraint_patch') and callable(P.build_constraint_patch)
        _has_resid = hasattr(P, 'constraint_residual') and callable(P.constraint_residual)

        if not (_has_patch and _has_resid):
            return self._solve_with_semismooth_newton(func, y0)

        # Pre-fetch constraint q_slices for row replacement
        _q_slices = P.constraint_q_slices if hasattr(P, 'constraint_q_slices') else []

        # Cross-call Jacobian cache (patched J persists across solve() calls)
        J_local = None
        if self._J_cross_call is not None and self._J_cross_call.shape == (n, n):
            J_local = self._J_cross_call

        prev_errF = np.inf

        # Force row equilibration for the algebraic saddle-point system
        # when using SPLU or UMFPACK.  The patched Jacobian has constraint
        # rows with O(1) entries and field rows with O(||stiffness||/h_time)
        # entries.  Without equilibration the direct solver receives a
        # heavily unbalanced matrix, producing an inaccurate LU factor that
        # stalls Newton.  This worsens with mesh refinement (stiffness ∝
        # h_mesh⁻²) and with larger time steps (A/(γh) → 0).
        #
        # When linear_solver='petsc' with MUMPS (the default PETSc config),
        # MUMPS ICNTL(8)=77 automatically selects the best scaling strategy
        # *inside* the factorisation — tightly coupled with pivot selection
        # and generally superior to a separate row-equilibration pass.
        # We therefore skip the manual scaling for the PETSc path.
        _force_scaling = False
        _is_petsc = (self.linear_solver == 'petsc')
        if sparse_active and self.jacobian_scaling == 'none' and not _is_petsc:
            _force_scaling = True
            self.jacobian_scaling = 'row'

        # Verbose Newton diagnostics — enable via solver._nl_debug = True
        _nl_debug = getattr(self, '_nl_debug', False)

        def _build_patched_J(y_cur, t_cur, Fk_cur):
            """Compute base Jacobian and patch constraint rows."""
            J_base = self._compute_jacobian_csr(func, y_cur, sparse_active)
            patch = P.build_constraint_patch(
                y_cur, n, t=t_cur, Fk_val=Fk_cur,
                step_size=getattr(self, 'prev_step', None),
                prev_state=getattr(self, 'prev_state', None),
            )
            if sparse_active:
                J_mod = J_base.copy()
                # Vectorised row zeroing via CSR indptr slicing
                for qs in _q_slices:
                    s = J_mod.indptr[qs.start]
                    e = J_mod.indptr[qs.stop]
                    if e > s:
                        J_mod.data[s:e] = 0.0
                J_mod.eliminate_zeros()
                return J_mod + patch
            else:
                J_arr = J_base.toarray() if sp.issparse(J_base) else np.array(J_base, copy=True)
                patch_d = patch.toarray() if sp.issparse(patch) else np.asarray(patch)
                for qs in _q_slices:
                    J_arr[qs, :] = 0.0
                return J_arr + patch_d

        for iteration in range(1, self.max_iter + 1):
            tcur = getattr(self, 'current_time', None)

            # ---- Combined residual: field eqs + constraint violation ----
            F_field = func(y)
            self.last_Fk_val = F_field
            c_resid = P.constraint_residual(
                y, t=tcur, Fk_val=F_field,
                step_size=getattr(self, 'prev_step', None),
                prev_state=getattr(self, 'prev_state', None),
            )
            # REPLACE constraint rows (don't add — func may already carry
            # algebraic residual on those DOFs, e.g. dq = q - C y).
            F_combined = F_field.copy()
            for qs in _q_slices:
                F_combined[qs] = c_resid[qs]

            converged, errF = self._converged_with_metric(F_combined, y)
            if _nl_debug:
                print(f'  [alg-newton] it={iteration} errF={errF:.4e}'
                      f' ||F_field||={float(np.linalg.norm(F_field)):.4e}'
                      f' ||c_resid||={float(np.linalg.norm(c_resid)):.4e}'
                      f' converged={converged}')
            if converged:
                # Enforce constraints to machine precision before returning
                proj_val = self._project(y, y, 1.0, tcur, F_field,
                                         getattr(self, 'prev_state', None))
                y[:] = proj_val
                F_final = func(y)
                if _force_scaling:
                    self.jacobian_scaling = 'none'
                return (y.copy(), F_final, errF, True, iteration)

            # ---- Recompute J only when stale or first time ----
            need_J = (J_local is None or errF > 0.5 * prev_errF)
            prev_errF = errF

            if need_J:
                J_local = _build_patched_J(y, tcur, F_field)
                # ---- Jacobian equilibration (row / Ruiz) ----
                if sparse_active and self.jacobian_scaling != 'none':
                    J_local = self._apply_jacobian_scaling(
                        self._to_csr(J_local) if not sp.issparse(J_local) else J_local.tocsr())
                self._J_cross_call = J_local
                # Invalidate cached factorisations so the fresh J is used
                self._lu = None
                self._lu_pattern = None
                self._lu_shape = None
                self._petsc_needs_matrix_update = True

            # ---- Linear solve (SPLU caching delegated to _solve_linear_sparse) ----
            rhs = -F_combined

            if sparse_active:
                J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local

                rtol_dyn = self.gmres_tol
                if self.linear_solver not in ('splu', 'umfpack') and self.linear_tol_strategy != 'fixed':
                    eta = min(0.5, self.eisenstat_c * (errF ** self.eisenstat_exp))
                    rtol_dyn = max(self.gmres_tol, eta)

                delta, ok = self._solve_linear_sparse(
                    J_csr, self._scale_rhs(rhs), rtol=rtol_dyn)
                if ok:
                    delta = self._unscale_solution(delta)
                if not ok:
                    if not need_J:
                        # Force fresh Jacobian and SPLU rebuild
                        J_local = _build_patched_J(y, tcur, F_field)
                        if self.jacobian_scaling != 'none':
                            J_local = self._apply_jacobian_scaling(
                                self._to_csr(J_local) if not sp.issparse(J_local)
                                else J_local.tocsr())
                        self._J_cross_call = J_local
                        J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local
                        self._lu = None
                        self._lu_pattern = None
                        self._lu_shape = None
                        self._petsc_needs_matrix_update = True
                        delta, ok = self._solve_linear_sparse(
                            J_csr, self._scale_rhs(rhs), rtol=rtol_dyn)
                        if ok:
                            delta = self._unscale_solution(delta)
                    if not ok:
                        if _force_scaling:
                            self.jacobian_scaling = 'none'
                        return (y, F_field, errF, False, iteration)
            else:
                J_dense = J_local.toarray() if sp.issparse(J_local) else J_local
                try:
                    delta = np.linalg.solve(J_dense, rhs)
                except np.linalg.LinAlgError:
                    if _force_scaling:
                        self.jacobian_scaling = 'none'
                    return (y, F_field, errF, False, iteration)

            if _nl_debug:
                delta_norm = float(np.linalg.norm(delta))
                y_norm = max(1.0, float(np.linalg.norm(y)))
                print(f'  [alg-newton] it={iteration} ||delta||={delta_norm:.4e}'
                      f' ||y||={y_norm:.4e} need_J={need_J}')

            # ---- Update ----
            np.add(y, delta, out=y)

        # Max iterations exhausted
        F_in = func(y)
        self.last_Fk_val = F_in
        c_resid = P.constraint_residual(y, t=getattr(self, 'current_time', None))
        F_combined = F_in.copy()
        for qs in _q_slices:
            F_combined[qs] = c_resid[qs]
        errF = self._errf_metric(F_combined, y)
        if _force_scaling:
            self.jacobian_scaling = 'none'
        if _nl_debug:
            print(f'  [alg-newton] FAILED after {self.max_iter} iters, errF={errF:.4e}')
        return (y, F_in, errF, False, self.max_iter)

    # ================================================================
    # Identity-projection fast paths (no projection overhead at all)
    # ================================================================

    def _solve_newton_identity(self, func, y0):
        """Standard Newton solver — fast path for IdentityProjection.

        Bypasses *all* projection machinery that the general semismooth Newton
        path performs:

        * No ``_update_rho`` / lam adaptation  (saves 2+ func evals / iter)
        * No projection call                   (saves function-call overhead)
        * No tangent-cone computation           (D = I is trivial)
        * No sparse assembly ``I - D + lam*D@J`` (saves 3 large sparse ops)

        Additionally implements **modified Newton**: the Jacobian (and its
        SPLU factorization when on the sparse path) is reused across
        iterations within a solve, and only recomputed when the error
        did not at least halve compared to the previous iteration.
        The SPLU factorization is also persisted across successive
        ``solve()`` calls (via ``self._lu``), so back-to-back time steps
        that share a similar Jacobian benefit from factorization reuse
        — analogous to what SciPy's BDF does internally.

        For multi-stage SDIRK methods this is especially beneficial:
        every stage within one time step shares the *same* diagonal
        coefficient γh and (for constant-Jacobian problems) the same
        iteration matrix ``A/(γh) − J``, so the SPLU from stage 1 is
        reused for stage 2 with zero refactorisation cost.
        """
        y = y0.copy()
        n = len(y)
        sparse_active = self._sparse_active(n)

        # ---- Modified-Newton state ----
        # Seed from persistent cache so that back-to-back solve() calls
        # (e.g. SDIRK2 stage 1 → stage 2, or consecutive time steps with
        # the same step size and constant Jacobian) reuse the factorisation.
        J_local = None          # current Jacobian (dense or CSR)
        lu_local = self._lu if (self._lu is not None and self._lu_shape == (n, n)) else None

        # For non-SPLU iterative solvers (GMRES, PETSc, …), seed J from
        # the cross-call cache so that stage 2 of SDIRK and consecutive
        # same-h steps skip the Jacobian evaluation — analogous to how
        # the SPLU path seeds lu_local from self._lu.
        _use_splu_path = (self.linear_solver == 'splu')
        if not _use_splu_path and self._J_cross_call is not None:
            if self._J_cross_call.shape == (n, n):
                J_local = self._J_cross_call

        prev_errF = np.inf      # previous *iteration's* errF (always updated)

        for iteration in range(1, self.max_iter + 1):
            F_in = np.asarray(func(y), dtype=float).reshape(-1)
            self.last_Fk_val = F_in
            converged, errF = self._converged_with_metric(F_in, y)

            if converged:
                return (y.copy(), F_in, errF, True, iteration)

            # --- Decide whether to recompute J (modified Newton) ---
            # Recompute J when convergence factor > 0.5 (error didn't at
            # least halve since last iteration).  On the very first
            # iteration, if we inherited a cached SPLU from a prior
            # solve() call, *skip* the Jacobian computation and try the
            # cached factorisation directly (cross-call modified Newton).
            # This is the key path for SDIRK multi-stage reuse: both
            # stages share the same iteration matrix A/(γh)−J, so stage 2
            # re-uses stage 1's factorisation at zero extra cost.
            need_J = (
                (J_local is None and lu_local is None)
                or errF > 0.5 * prev_errF
            )
            prev_errF = errF          # always track for next iteration

            if need_J:
                J_local = self._compute_jacobian_csr(func, y, sparse_active)
                # ---- Jacobian equilibration (row / Ruiz) ----
                if sparse_active and self.jacobian_scaling != 'none':
                    J_csr_eq = self._to_csr(J_local) if not sp.issparse(J_local) else J_local.tocsr()
                    J_local = self._apply_jacobian_scaling(J_csr_eq)
                lu_local = None               # invalidate cached factorization
                self._lu = None               # also invalidate persistent cache
                self._lu_shape = None
                self._J_cross_call = None     # will be re-set after linear solve
                self._petsc_needs_matrix_update = True  # force MUMPS re-factorisation

            # --- Linear solve: J @ delta = -F_in ---
            rhs = -F_in

            if sparse_active:
                # When J_local is available, prepare CSR form for fallback paths
                J_csr = None
                if J_local is not None:
                    J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local

                if _use_splu_path:
                    # ---- Inline SPLU path with cross-call LU reuse ----
                    # Fastest for moderate-size systems where direct
                    # factorisation is affordable.  LU is reused within a
                    # solve and across consecutive solve() calls (SDIRK
                    # stage reuse, consecutive same-h steps).
                    rhs_solve = self._scale_rhs(rhs)
                    if lu_local is not None:
                        try:
                            delta = lu_local.solve(rhs_solve)
                            delta = np.asarray(
                                self._unscale_solution(delta), dtype=float
                            ).reshape(-1)
                        except Exception:
                            lu_local = None
                            self._lu = None
                            self._lu_shape = None

                    if lu_local is None:
                        if J_csr is None:
                            J_local = self._compute_jacobian_csr(func, y, sparse_active)
                            if self.jacobian_scaling != 'none':
                                J_local = self._apply_jacobian_scaling(
                                    self._to_csr(J_local) if not sp.issparse(J_local)
                                    else J_local.tocsr())
                                rhs_solve = self._scale_rhs(rhs)
                            J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local
                        try:
                            J_csc = J_csr.tocsc() if not sp.isspmatrix_csc(J_csr) else J_csr
                            lu_local = spla.splu(J_csc, permc_spec=self.splu_permc_spec)
                            self._lu = lu_local
                            self._lu_shape = (n, n)
                            delta = lu_local.solve(rhs_solve)
                            delta = np.asarray(
                                self._unscale_solution(delta), dtype=float
                            ).reshape(-1)
                        except Exception:
                            # SPLU failed — fall back to _solve_linear_sparse
                            lu_local = None
                            rtol_dyn = self.gmres_tol
                            if self.linear_tol_strategy != 'fixed':
                                eta = min(0.5, self.eisenstat_c * (errF ** self.eisenstat_exp))
                                rtol_dyn = max(self.gmres_tol, eta)
                            delta, ok = self._solve_linear_sparse(
                                J_csr, rhs_solve, rtol=rtol_dyn)
                            if ok:
                                delta = np.asarray(
                                    self._unscale_solution(delta), dtype=float
                                ).reshape(-1)
                            if not ok:
                                if not need_J:
                                    J_local = self._compute_jacobian_csr(func, y, sparse_active)
                                    if self.jacobian_scaling != 'none':
                                        J_local = self._apply_jacobian_scaling(
                                            self._to_csr(J_local) if not sp.issparse(J_local)
                                            else J_local.tocsr())
                                        rhs_solve = self._scale_rhs(rhs)
                                    J_csr = self._to_csr(J_local)
                                    self._petsc_needs_matrix_update = True
                                    delta, ok = self._solve_linear_sparse(
                                        J_csr, rhs_solve, rtol=rtol_dyn)
                                    if ok:
                                        delta = np.asarray(
                                            self._unscale_solution(delta), dtype=float
                                        ).reshape(-1)
                                if not ok:
                                    return (y, F_in, errF, False, iteration)
                else:
                    # ---- General iterative path (GMRES, PETSc, …) ----
                    # Honours the user's ``linear_solver`` setting and
                    # enables physics-based preconditioners such as PETSc
                    # field-split for saddle-point systems (e.g. Biot).
                    #
                    # Cross-call modified Newton: when *need_J* was False
                    # (convergence is healthy), the Jacobian seeded from
                    # ``_J_cross_call`` is reused — SDIRK stage 2 and
                    # consecutive same-h steps avoid redundant evaluations.
                    if J_csr is None:
                        # Fall back to computing J now
                        J_local = self._compute_jacobian_csr(func, y, sparse_active)
                        if self.jacobian_scaling != 'none':
                            J_local = self._apply_jacobian_scaling(
                                self._to_csr(J_local) if not sp.issparse(J_local)
                                else J_local.tocsr())
                        J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local

                    # Persist for cross-call reuse (analogous to self._lu)
                    self._J_cross_call = J_csr

                    rtol_dyn = self.gmres_tol
                    if self.linear_tol_strategy != 'fixed':
                        eta = min(0.5, self.eisenstat_c * (errF ** self.eisenstat_exp))
                        rtol_dyn = max(self.gmres_tol, eta)

                    rhs_solve = self._scale_rhs(rhs)
                    delta, ok = self._solve_linear_sparse(J_csr, rhs_solve, rtol=rtol_dyn)
                    if ok:
                        delta = np.asarray(
                            self._unscale_solution(delta), dtype=float
                        ).reshape(-1)
                    if not ok:
                        # Retry with a fresh Jacobian if we hadn't already
                        if not need_J:
                            J_local = self._compute_jacobian_csr(func, y, sparse_active)
                            if self.jacobian_scaling != 'none':
                                J_local = self._apply_jacobian_scaling(
                                    self._to_csr(J_local) if not sp.issparse(J_local)
                                    else J_local.tocsr())
                                rhs_solve = self._scale_rhs(rhs)
                            J_csr = self._to_csr(J_local) if not sp.issparse(J_local) else J_local
                            self._J_cross_call = J_csr
                            self._petsc_needs_matrix_update = True
                            delta, ok = self._solve_linear_sparse(
                                J_csr, rhs_solve, rtol=rtol_dyn)
                            if ok:
                                delta = np.asarray(
                                    self._unscale_solution(delta), dtype=float
                                ).reshape(-1)
                        if not ok:
                            return (y, F_in, errF, False, iteration)
            else:
                J_dense = J_local.toarray() if sp.issparse(J_local) else J_local
                try:
                    delta = np.asarray(
                        np.linalg.solve(J_dense, rhs), dtype=float
                    ).reshape(-1)
                except np.linalg.LinAlgError:
                    return (y, F_in, errF, False, iteration)

            # --- Globalization (optional Armijo line search) ---
            # When J_local is None (using cached LU from a prior solve call),
            # skip linesearch and accept the full Newton step.  The merit
            # function gradient requires J^T F which we cannot compute
            # without J.  In practice this only happens when the cached
            # factorisation is very close to the true Jacobian (e.g. SDIRK
            # stage reuse or consecutive same-h steps).
            if self.globalization == 'linesearch' and J_local is not None:
                phi0 = 0.5 * errF * errF
                if sp.issparse(J_local):
                    grad_phi = np.asarray(J_local.T @ F_in, dtype=float).reshape(-1)
                else:
                    grad_phi = np.asarray(J_local.T @ F_in, dtype=float).reshape(-1)
                grad_dir = float(np.dot(grad_phi, delta))

                alpha = 1.0
                accepted = False

                if np.isfinite(grad_dir) and grad_dir < 0.0:
                    for _ in range(self.max_backtracks):
                        y_trial = y + alpha * delta
                        F_trial = np.asarray(func(y_trial), dtype=float).reshape(-1)
                        phi_trial = 0.5 * float(np.dot(F_trial, F_trial))
                        if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir:
                            y = y_trial
                            accepted = True
                            break
                        alpha *= self.ls_beta
                        if alpha < self.ls_min_alpha:
                            break

                if not accepted:
                    # Steepest-descent fallback
                    nrm_g = float(np.linalg.norm(grad_phi))
                    if nrm_g == 0.0:
                        return (y, F_in, errF, False, iteration)
                    delta_g = -grad_phi
                    grad_dir_g = -nrm_g * nrm_g
                    alpha = 1.0
                    for _ in range(self.max_backtracks):
                        y_trial = y + alpha * delta_g
                        F_trial = np.asarray(func(y_trial), dtype=float).reshape(-1)
                        phi_trial = 0.5 * float(np.dot(F_trial, F_trial))
                        if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir_g:
                            y = y_trial
                            accepted = True
                            break
                        alpha *= self.ls_beta
                    if not accepted:
                        return (y, F_in, errF, False, iteration)
            else:
                np.add(y, delta, out=y)

        # Max iterations exhausted
        F_in = np.asarray(func(y), dtype=float).reshape(-1)
        self.last_Fk_val = F_in
        errF = self._errf_metric(F_in, y)
        return (y, F_in, errF, False, self.max_iter)

    def _solve_vi_identity(self, func, y0):
        """Fast VI for IdentityProjection — pure Richardson iteration.

        For identity projection the fixed-point map is simply
        ``y_{k+1} = y_k - rho * F(y_k)`` with no projection calls.
        The natural residual is ``rho * F(y)`` and the error metric
        matches the relative block-L2 norm used by the general VI path.

        .. note::

           Richardson iteration converges only linearly.  For stiff PDE
           systems the semismooth Newton path (``method='semismooth_newton'``)
           will converge in far fewer iterations.
        """
        y = y0.copy()
        n_vi = y.size
        slices = self.component_slices

        # --- rho initialisation (mirrors general VI) ---
        if slices is not None and len(slices) > 0:
            last = getattr(self, 'rho_last', self.rho0)
            m = len(slices)
            if np.isscalar(last):
                rho_blk = np.full(m, float(last), dtype=float)
            else:
                arr = np.asarray(last, dtype=float).reshape(-1)
                rho_blk = arr.copy() if arr.size == m else np.full(m, float(np.mean(arr)), dtype=float)
            np.clip(rho_blk, self.rho_min, self.rho_max, out=rho_blk)
            rho_vec = np.empty(n_vi, dtype=float)
            for v, s in zip(rho_blk, slices):
                rho_vec[s] = float(v)
        else:
            rho_scalar = float(getattr(self, 'rho_last', self.rho0))
            if not (np.isfinite(rho_scalar) and rho_scalar > 0):
                rho_scalar = 1.0
            rho_scalar = float(np.clip(rho_scalar, self.rho_min, self.rho_max))
            rho_vec = np.full(n_vi, rho_scalar, dtype=float)
            rho_blk = rho_scalar  # kept for _set_rho_last

        # --- helper: block-relative error (same metric as general VI) ---
        # When per-DOF weighted norm is active, use it instead.
        _use_wrms = self._use_weighted_norm

        def _err(r_vec, y_vec):
            if _use_wrms:
                return self._wrms(r_vec, y_vec)
            if slices is not None and len(slices) > 0:
                vals = []
                for s in slices:
                    rs, ys = r_vec[s], y_vec[s]
                    nn = max(1, rs.size)
                    nr = float(np.linalg.norm(rs)) / math.sqrt(nn)
                    ny = float(np.linalg.norm(ys)) / math.sqrt(nn)
                    vals.append(nr / (1.0 + ny))
                return max(vals) if vals else 0.0
            nn = max(1, r_vec.size)
            return (float(np.linalg.norm(r_vec)) / math.sqrt(nn)
                    / (1.0 + float(np.linalg.norm(y_vec)) / math.sqrt(nn)))

        # Convergence threshold: wrms <= 1 or legacy err <= tol
        _conv_thresh = 1.0 if _use_wrms else self.tol

        # --- buffers ---
        _r_buf = np.empty(n_vi, dtype=float)

        Fk = func(y)
        self.last_Fk_val = Fk
        np.multiply(rho_vec, Fk, out=_r_buf)          # residual = rho * F
        err = _err(_r_buf, y)

        k = 0
        while err > _conv_thresh and k < self.max_iter:
            # Richardson step: y_new = y - rho * F(y)
            y_new = y - rho_vec * Fk
            Fk_new = func(y_new)

            # --- rho adaptation (scalar or per-block) ---
            if slices is not None and len(slices) > 0:
                for i, s in enumerate(slices):
                    den = float(np.linalg.norm(y_new[s] - y[s]))
                    stuck = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + float(np.linalg.norm(y[s])))
                    if den < stuck:
                        continue
                    rk_i = rho_blk[i] * float(np.linalg.norm(Fk_new[s] - Fk[s])) / den
                    if rk_i > self.L:
                        rho_blk[i] *= self.nu
                    elif rk_i < self.Lmin:
                        rho_blk[i] /= self.nu
                np.clip(rho_blk, self.rho_min, self.rho_max, out=rho_blk)
                for v, s in zip(rho_blk, slices):
                    rho_vec[s] = float(v)
            else:
                den = float(np.linalg.norm(y_new - y))
                stuck = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + float(np.linalg.norm(y)))
                if den >= stuck:
                    rk = rho_vec[0] * float(np.linalg.norm(Fk_new - Fk)) / den
                    if rk > self.L:
                        rho_vec *= self.nu
                    elif rk < self.Lmin:
                        rho_vec /= self.nu
                    np.clip(rho_vec, self.rho_min, self.rho_max, out=rho_vec)
                    rho_blk = float(rho_vec[0])

            y = y_new
            Fk = Fk_new
            self.last_Fk_val = Fk

            np.multiply(rho_vec, Fk, out=_r_buf)
            err = _err(_r_buf, y)
            k += 1

        success = (err <= _conv_thresh)
        self._set_rho_last(rho_blk, update_default=True)
        return (y, Fk, err, success, k)

    # ---------------- Semismooth Newton ----------------
    def _phi(self, y):
        Fk_val = self.func(y)
        lam = self.lam
        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)
        proj_val = self._project(y, y - lam * Fk_val, lam, tcur, Fk_val, prev)
        F = y - proj_val
        return 0.5 * np.dot(F, F)

    def _solve_with_semismooth_newton(self, func, y0):
        y = y0.copy()
        lam = self.lam
        n = len(y)

        # Decide sparse path once (dimension doesn't change within a solve)
        sparse_active = self._sparse_active(n)
        if sparse_active:
            I = self._I_cache.get(("csr", n))
            if I is None:
                I = sp.eye(n, format='csr')
                self._I_cache[("csr", n)] = I
        else:
            I = self._I_cache.get(n)
            if I is None:
                I = np.eye(n)
                self._I_cache[n] = I

        # Buffers
        candidate = np.empty_like(y)
        proj_z = np.empty_like(y)
        F_buf = np.empty_like(y)

        # Reset Broyden state
        if self.use_broyden:
            self._B = None
            self._y_prev_broyden = None
            self._F_prev_broyden = None

        # --- Cache batch projection detection (done once, not every iteration) ---
        _use_batch = False
        _batch_row_slices = None
        P = self.proj
        try:
            _has_batch = hasattr(P, 'project_batch') and callable(P.project_batch)
            _ci = getattr(P, 'constraint_indices', None)
            if _has_batch and _ci is not None and np.size(_ci) > 0:
                _ci = np.asarray(_ci)
                _ci_sorted = np.sort(_ci)
                _diffs = np.diff(_ci_sorted)
                _boundaries = np.where(_diffs > 1)[0] + 1
                _runs = np.split(_ci_sorted, _boundaries)
                _block_len = None
                _ok = True
                _batch_row_slices = []
                for _r in _runs:
                    if _r.size == 0:
                        continue
                    _start, _stop = int(_r[0]), int(_r[-1] + 1)
                    if _block_len is None:
                        _block_len = _stop - _start
                    elif _block_len != (_stop - _start):
                        _ok = False
                        break
                    _batch_row_slices.append(slice(_start, _stop))
                _use_batch = _ok and (len(_batch_row_slices) > 0)
                if _use_batch:
                    _batch_dim = _batch_row_slices[0].stop - _batch_row_slices[0].start
                    _batch_rows = len(_batch_row_slices)
                    _Yv = np.empty((_batch_rows, _batch_dim), dtype=y.dtype)
                    _Cv = np.empty_like(_Yv)
        except Exception:
            _use_batch = False

        # Track whether we have a cached F_in from a previous _update_rho call
        _cached_F_in = None

        # --- Active-set locking for gap-based projections ---
        # Evaluate gap at the *predicted* candidate (y0 − λ F(y0)) to
        # decide which contacts are active for the entire solve.  This
        # prevents active-set chattering that destroys Newton convergence.
        _has_lock = hasattr(P, 'lock_active_set') and hasattr(P, 'unlock_active_set')
        _has_relock = _has_lock and hasattr(P, 'reset_branch_cache')
        if _has_lock:
            tcur_init = getattr(self, 'current_time', None)
            _F0 = func(y)
            # Lock active set at the *initial state* y (previous step's
            # converged solution).  The old predictor  y − λ F(y)  is a
            # crude forward-Euler step of the natural map; for stiff or
            # DAE-like systems the position rows of F carry an O(1/h)
            # factor, causing the predictor to massively overshoot
            # position variables and produce spurious contact activation
            # when gap is position-based (e.g. gap = q_y).
            # Locking at y is physically meaningful: if the ball is
            # clearly above the ground (gap > 0) at the start of the
            # step, contact is inactive.  Should the iterate cross
            # gap = 0 during the Newton solve, Proposal 3 (monotone
            # relocking below) will activate the contact.
            P.lock_active_set(y, t=tcur_init)
            _cached_F_in = _F0            # reuse for first iteration

        def _unlock():
            if _has_lock:
                P.unlock_active_set()

        # Merit tracking for Proposal 3 (complementarity-based relocking).
        # After each accepted Newton step, if the natural-map merit
        # Ψ = ½‖r‖² improved, re-evaluate the gap at the new candidate
        # and re-lock.  Prevents mis-classification at the predictor from
        # persisting, while still preventing within-iteration chatter.
        _prev_merit = float('inf')
        _lam_readapt = False  # set True after Proposal 3b relock

        for iteration in range(1, self.max_iter + 1):
            # cache context once per iteration
            tcur = getattr(self, 'current_time', None)
            prev = getattr(self, 'prev_state', None)

            # Optional adaptive lam — pass cached F_in to avoid redundant func eval.
            # Skip when the projection is rho-independent (e.g. algebraic
            # constraints): lam has no effect on the projection output and
            # lam=1 gives the optimal standard Newton system.
            # Also skip after the first iteration: for SSN the Newton
            # direction already accounts for lam, so re-adapting every
            # iteration destroys quadratic convergence and wastes evals.
            # Exception: re-adapt once after a Proposal 3b active-set
            # relock, since the Lipschitz constant changes.
            if (self.adaptive_lam
                    and self.lam_update_strategy == 'vi'
                    and not self._is_rho_independent
                    and (iteration <= 1 or _lam_readapt)
                    and np.ndim(lam) == 0):
                try:
                    lam = self._update_rho(func, y, lam, Fk_val=_cached_F_in)
                    self.lam = lam
                    _lam_readapt = False
                except Exception:
                    _lam_readapt = False

            F_in = func(y)
            self.last_Fk_val = F_in  # cheap attribute write
            _cached_F_in = F_in  # cache for next iteration's _update_rho

            # candidate = y - lam F(y)
            np.subtract(y, lam * F_in, out=candidate)

            # projection (fastpath) — batch detection already cached
            proj_val = None
            try:
                if _use_batch:
                    for i, sl in enumerate(_batch_row_slices):
                        _Yv[i] = y[sl]
                        _Cv[i] = candidate[sl]
                    Pv = P.project_batch(_Yv, _Cv, rhok=lam, t=tcur, Fk_val=F_in)
                    proj_z[:] = candidate
                    for i, sl in enumerate(_batch_row_slices):
                        proj_z[sl] = Pv[i]
                    proj_val = proj_z
                else:
                    proj_val = self._project(y, candidate, lam, tcur, F_in, prev)
            except Exception:
                proj_val = self._project(y, candidate, lam, tcur, F_in, prev)
            proj_z[:] = proj_val

            # F_buf = y - proj_z
            np.subtract(y, proj_z, out=F_buf)
            converged, errF = self._converged_with_metric(F_buf, y)

            if converged:
                # --- Post-convergence active-set relock (Proposal 3b) ---
                # Before accepting, check if any previously-inactive
                # contacts have closed (gap ≤ 0) at the converged state.
                # If so, add them to the active set (monotone union) and
                # continue iterating.  This closes the gap in Proposal 3
                # which sits *after* the convergence check and thus never
                # fires when convergence is reached with the wrong
                # (all-inactive) active set.
                if _has_relock:
                    _old_mask_cv = P._locked_active
                    if (_old_mask_cv is not None
                            and not _old_mask_cv.all()):
                        _gap_func_cv = getattr(P, 'gap_func', None)
                        if _gap_func_cv is not None:
                            _tcur_cv = getattr(self, 'current_time', None)
                            _gn = getattr(P, '_gap_nargs', None)
                            if _gn is not None and _gn <= 1:
                                _gaps_cv = np.atleast_1d(
                                    _gap_func_cv(y))
                            else:
                                _gaps_cv = np.atleast_1d(
                                    _gap_func_cv(y, _tcur_cv))
                            _new_cv = _gaps_cv <= P.gap_tol
                            _union_cv = _old_mask_cv | _new_cv
                            if not np.array_equal(_union_cv,
                                                  _old_mask_cv):
                                P._locked_active = _union_cv
                                _lam_readapt = True
                                continue  # re-enter loop with updated set

                y[:] = proj_z
                F_y = func(y)
                _unlock()
                return (y.copy(), F_y, errF, True, iteration)

            # Inner Jacobian
            used_broyden = False
            if self.use_broyden and not sparse_active:
                if self._B is not None and self._y_prev_broyden is not None and self._F_prev_broyden is not None:
                    s_vec = y - self._y_prev_broyden
                    y_vec = F_in - self._F_prev_broyden
                    denom = float(np.dot(s_vec, s_vec))
                    if np.isfinite(denom) and denom > 0.0:
                        Bs = self._B @ s_vec
                        corr = (y_vec - Bs) / denom
                        self._B = self._B + np.outer(corr, s_vec)
                if self._B is None:
                    if self.jacobian is not None:
                        B0 = self.jacobian(y)
                    else:
                        B0 = self._numerical_jacobian(func, y, sparse=False)
                    if sp.issparse(B0):
                        B0 = B0.toarray()
                    self._B = B0
                J_in = self._B
                used_broyden = True
            else:
                # Use optimized Jacobian computation with caching
                J_in = self._compute_jacobian_csr(func, y, sparse_active)

            # Eisenstat–Walker tol for GMRES if enabled
            rtol_dyn = self.gmres_tol
            if self.linear_solver == 'gmres' and self.linear_tol_strategy != 'fixed':
                eta = min(0.5, self.eisenstat_c * (errF ** self.eisenstat_exp))
                rtol_dyn = max(self.gmres_tol, eta)

            if sparse_active:
                # Sparse path: either use matrix-free GMRES (default) or explicit matrix + SPLU/PETSc.
                J_in = self._to_csr(J_in)
                D_out = self._compute_tangent_csr(candidate, y, lam, tcur, F_in, prev, n)
                if isinstance(D_out, tuple):
                    Dproj, Dstate = D_out
                else:
                    Dproj, Dstate = D_out, None
                rhs = -F_buf

                # Detect diagonal D for fast assembly (avoid expensive D @ J matmul).
                # D is diagonal when it has exactly n nonzeros and they sit on the diagonal.
                _D_is_diag = False
                _D_diag_vals = None
                if Dstate is None and sp.issparse(Dproj):
                    # Fast structural diagonal case first; then fall back to a
                    # numerical diagonal check that tolerates stored zeros in a
                    # fixed sparse pattern (e.g. full SOC blocks with zero
                    # off-diagonal entries in interior / polar regions).
                    _dptr = Dproj.indptr
                    if Dproj.nnz <= n and np.all(np.diff(_dptr) == 1):
                        _didx = Dproj.indices
                        if np.array_equal(_didx, np.arange(n)):
                            _D_is_diag = True
                            _D_diag_vals = Dproj.data  # length-n diagonal
                    if not _D_is_diag:
                        _D_is_diag, _D_diag_vals = self._extract_sparse_numeric_diagonal(Dproj, n)

                if self.linear_solver in ('splu', 'petsc'):
                    if _D_is_diag:
                        # Fast diagonal assembly:  J = I - D + lam * D @ J_in
                        # = diag(1 - d) + lam * diag(d) @ J_in
                        # = diag(1 - d) + J_in scaled row-wise by lam*d
                        d = _D_diag_vals
                        J_mat = self._assemble_diag_newton_csr(J_in, d, lam)
                        if J_mat is None:
                            # Fallback when J_in lacks a structural diagonal.
                            _scale = lam * d
                            J_mat = J_in.multiply(_scale[:, None]) if hasattr(J_in, 'multiply') else sp.diags(_scale) @ J_in
                            _one_minus_d = 1.0 - d
                            J_mat = J_mat + sp.diags(_one_minus_d, format='csr')
                    else:
                        # General: J = I - D + D @ diag(lam) @ J_in
                        # For scalar lam the commutation is trivial;
                        # for vector lam we must row-scale J_in first.
                        if np.ndim(lam) >= 1:
                            _J_lam = J_in.multiply(lam[:, None]) if hasattr(J_in, 'multiply') else sp.diags(lam) @ J_in
                        else:
                            _J_lam = lam * J_in
                        if Dstate is None:
                            J_mat = I - Dproj + Dproj @ _J_lam
                        else:
                            J_mat = I - Dproj - Dstate + Dproj @ _J_lam
                    if not sp.issparse(J_mat):
                        J_mat = self._to_csr(J_mat, n)
                    # Ensure every diagonal position is structurally present.
                    # Sparse arithmetic (e.g. I - D when D=I) can prune
                    # numerical zeros, leaving MUMPS/SuperLU without a pivot
                    # entry.  Reading and writing back the diagonal is a
                    # no-op on values but forces structural allocation.
                    J_mat = J_mat.tocsr()
                    _diag = J_mat.diagonal()
                    J_mat.setdiag(_diag)
                    delta, ok = self._solve_linear_sparse(J_mat, rhs, rtol=rtol_dyn, pattern_hint=None)
                else:
                    if _D_is_diag:
                        d = _D_diag_vals
                        _scale = lam * d
                        _one_minus_d = 1.0 - d
                        def _matvec(v, _s=_scale, _omd=_one_minus_d, _J=J_in):
                            return _omd * v + _s * (_J @ v)
                        def _rmatvec(w, _s=_scale, _omd=_one_minus_d, _J=J_in):
                            return _omd * w + _J.T @ (_s * w)
                    else:
                        # D @ diag(lam) @ J — reorder so lam scales
                        # the Jv product *before* D acts on it.
                        # Works identically for scalar and vector lam.
                        if Dstate is None:
                            def _matvec(v, _D=Dproj, _J=J_in, _lam=lam):
                                return (v - _D @ v) + _D @ (_lam * (_J @ v))
                            def _rmatvec(w, _D=Dproj, _J=J_in, _lam=lam):
                                Dt = _D.T @ w
                                return (w - Dt) + _J.T @ (_lam * Dt)
                        else:
                            def _matvec(v, _D=Dproj, _B=Dstate, _J=J_in, _lam=lam):
                                return (v - _D @ v - _B @ v) + _D @ (_lam * (_J @ v))
                            def _rmatvec(w, _D=Dproj, _B=Dstate, _J=J_in, _lam=lam):
                                Dt = _D.T @ w
                                return (w - Dt - _B.T @ w) + _J.T @ (_lam * Dt)

                    J = spla.LinearOperator((n, n), matvec=_matvec, rmatvec=_rmatvec)
                    delta, ok = self._solve_linear_sparse(J, rhs, rtol=rtol_dyn, pattern_hint=None)

                if not ok:
                    _unlock()
                    return (y, F_in, errF, False, iteration)
            else:
                P = getattr(self, 'proj', None)
                if (P is not None and hasattr(P, 'tangent_cone_split')
                        and callable(getattr(P, 'tangent_cone_split'))):
                    Dproj, Dstate = P.tangent_cone_split(
                        candidate, y, rhok=lam, t=tcur, Fk_val=F_in, prev_state=prev)
                else:
                    Dproj, Dstate = self._tangent(candidate, y, lam, tcur, F_in, prev), None

                if sp.issparse(Dproj):
                    Dproj = Dproj.toarray()
                if Dstate is not None and sp.issparse(Dstate):
                    Dstate = Dstate.toarray()
                if sp.issparse(J_in):
                    J_in = J_in.toarray()
                # Dense: J = I - D + D @ diag(lam) @ J
                if np.ndim(lam) >= 1:
                    _J_lam = lam[np.newaxis, :] * J_in   # row-scale J_in
                else:
                    _J_lam = lam * J_in
                if Dstate is None:
                    J = I - Dproj + Dproj @ _J_lam
                else:
                    J = I - Dproj - Dstate + Dproj @ _J_lam
                try:
                    delta = np.linalg.solve(np.asarray(J), -F_buf)
                except np.linalg.LinAlgError:
                    _unlock()
                    return (y, F_in, errF, False, iteration)

            # Globalization (optional)
            if self.globalization == 'linesearch':
                phi0 = 0.5 * errF * errF

                if sparse_active and _D_is_diag:
                    d = _D_diag_vals
                    _omd = 1.0 - d
                    _s_ls = lam * d
                    def _apply_JT_local(v):
                        return _omd * v + J_in.T @ (_s_ls * v)
                else:
                    def _apply_JT_local(v, _lam=lam):
                        Dt = Dproj.T @ v
                        if Dstate is None:
                            return v - Dt + J_in.T @ (_lam * Dt)
                        return v - Dt - (Dstate.T @ v) + J_in.T @ (_lam * Dt)

                grad_phi = _apply_JT_local(F_buf)
                grad_dir = float(np.dot(grad_phi, delta))

                # Inline _phi: phi(y') = 0.5 * ||y' - proj(y', y' - lam*F(y'))||^2
                def _phi_inline(y_t):
                    Fk_t = func(y_t)
                    cand_t = y_t - lam * Fk_t
                    proj_t = self._project(y_t, cand_t, lam, tcur, Fk_t, prev)
                    r_t = y_t - proj_t
                    return 0.5 * float(np.dot(r_t, r_t))

                alpha = 1.0
                backtracks = 0
                accepted = False

                if np.isfinite(grad_dir) and grad_dir < 0.0:
                    y_trial = y + alpha * delta
                    phi_trial = _phi_inline(y_trial)
                    while (phi_trial > phi0 + self.ls_c1 * alpha * grad_dir
                           and backtracks < self.max_backtracks
                           and alpha > self.ls_min_alpha):
                        alpha *= self.ls_beta
                        y_trial = y + alpha * delta
                        phi_trial = _phi_inline(y_trial)
                        backtracks += 1

                    if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir:
                        if self.use_broyden and not sparse_active:
                            self._y_prev_broyden = y.copy()
                            self._F_prev_broyden = F_in.copy()
                        y = y_trial
                        accepted = True

                if not accepted:
                    nrm_g = np.linalg.norm(grad_phi)
                    if nrm_g == 0.0:
                        _unlock()
                        return (y, F_in, errF, False, iteration)
                    delta_g = -grad_phi
                    grad_dir = -nrm_g * nrm_g

                    alpha = 1.0
                    backtracks = 0
                    y_trial = y + alpha * delta_g
                    phi_trial = _phi_inline(y_trial)

                    while (phi_trial > phi0 + self.ls_c1 * alpha * grad_dir
                           and backtracks < self.max_backtracks
                           and alpha > self.ls_min_alpha):
                        alpha *= self.ls_beta
                        y_trial = y + alpha * delta_g
                        phi_trial = _phi_inline(y_trial)
                        backtracks += 1

                    if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir:
                        if self.use_broyden and not sparse_active:
                            self._y_prev_broyden = y.copy()
                            self._F_prev_broyden = F_in.copy()
                        y = y_trial
                    else:
                        _unlock()
                        return (y, F_in, errF, False, iteration)
            else:
                if self.use_broyden and not sparse_active:
                    self._y_prev_broyden = y.copy()
                    self._F_prev_broyden = F_in.copy()
                np.add(y, delta, out=y)

            # --- Proposal 3: monotone active-set relocking ---
            # After each accepted Newton step, re-evaluate the gap and
            # take the UNION of the current lock with newly-detected
            # contacts.  Contacts are never *removed* during a solve —
            # only added.  This corrects predictor mis-classification
            # (a contact the predictor missed) without triggering the
            # false-deactivation problem (a contact that appears
            # resolved at an intermediate iterate but is still needed).
            #
            # Only attempt when:
            #   (a) merit improved (don't waste evals on diverging iters)
            #   (b) not all blocks are already active (nothing to add)
            #   (c) iteration ≥ 2 (first iteration is the big correction)
            if _has_relock and iteration >= 2:
                _cur_merit = 0.5 * float(np.dot(F_buf, F_buf))
                _old_mask = P._locked_active
                _all_active = (_old_mask is not None
                               and _old_mask.all())
                if _cur_merit < _prev_merit and not _all_active:
                    # Evaluate gap at current iterate directly (cheap).
                    _gap_nargs = getattr(P, '_gap_nargs', None)
                    _gap_func = getattr(P, 'gap_func', None)
                    if _gap_func is not None:
                        _tcur_rl = getattr(self, 'current_time', None)
                        if _gap_nargs is not None and _gap_nargs <= 1:
                            _gaps_rl = np.atleast_1d(_gap_func(y))
                        else:
                            _gaps_rl = np.atleast_1d(
                                _gap_func(y, _tcur_rl))
                        _new_active = _gaps_rl <= P.gap_tol
                        if _old_mask is not None:
                            _union = _old_mask | _new_active
                            if not np.array_equal(_union, _old_mask):
                                # Actually gained a new contact
                                P._locked_active = _union
                _prev_merit = _cur_merit

        # Out of iterations
        F_in = func(y)
        self.last_Fk_val = F_in
        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)
        F_resid = y - self._project(y, y - lam * F_in, lam, tcur, F_in, prev)
        errF = self._errf_metric(F_resid, y)
        _unlock()
        return (y, F_in, errF, False, self.max_iter)



    def _solve_with_VI(self, func, y0):
        def _sanitize_rho(rho_in, *, context="init"):
            """Ensure rho (scalar or array) is finite, positive, and within [rho_floor, rho_ceil].
            If non-finite values are found, reset them to a safe default (self.rho0 or 1.0) then clip.
            Returns sanitized rho with same type/shape semantics (scalar or ndarray).
            """
            debug_local = bool(getattr(self, 'debug_vi', False))
            if np.isscalar(rho_in):
                r = float(rho_in)
                if not np.isfinite(r) or r <= 0.0:
                    base = self.rho0 if (np.isscalar(self.rho0) and np.isfinite(self.rho0) and self.rho0 > 0) else 1.0
                    if debug_local:
                        print(f"[VI] rho sanitize ({context}): scalar reset from {r} to {base}")
                    r = float(base)
                r = float(np.clip(r, self.rho_min, self.rho_max))
                return r
            else:
                arr = np.asarray(rho_in, dtype=float)
                reset_mask = ~np.isfinite(arr) | (arr <= 0.0)
                if np.any(reset_mask):
                    base = self.rho0
                    if not (np.isscalar(base) and np.isfinite(base) and base > 0):
                        base = 1.0
                    if debug_local:
                        bad_vals = arr[reset_mask]
                        print(f"[VI] rho sanitize ({context}): resetting {bad_vals.size} entries (e.g., {bad_vals[:3]}) to {base}")
                    arr[reset_mask] = float(base)
                # clip to bounds
                np.clip(arr, self.rho_min, self.rho_max, out=arr)
                return arr

        # Helper: block-wise relative L2 of natural residual r = (y - P(y - rho F(y)))
        # When per-DOF weighted norm is active, use it instead.
        _use_wrms = self._use_weighted_norm

        def _rel_block_l2(r, y, slices):
            if _use_wrms:
                return self._wrms(r, y)
            if slices is not None:
                vals = []
                for s in slices:
                    rs, ys = r[s], y[s]
                    n  = max(1, rs.size)
                    nr = np.linalg.norm(rs) / math.sqrt(n)   # RMS of residual
                    ny = np.linalg.norm(ys) / math.sqrt(n)   # RMS of state
                    vals.append(nr / (1.0 + ny))
                return max(vals) if vals else 0.0
            else:
                n  = max(1, r.size)
                nr = np.linalg.norm(r) / math.sqrt(n)
                ny = np.linalg.norm(y) / math.sqrt(n)
                return nr / (1.0 + ny)


        # Expand block-wise rho (length = number of component_slices) to a per-index vector (length = n)
        # Pre-allocated buffer for rho expansion (filled in-place)
        _rho_vec_buf = None

        def _expand_rho_to_vec(rho_in, n, slices):
            nonlocal _rho_vec_buf
            # Allocate or resize the cached buffer once
            if _rho_vec_buf is None or _rho_vec_buf.size != n:
                _rho_vec_buf = np.empty(n, dtype=float)
            buf = _rho_vec_buf
            if np.isscalar(rho_in):
                buf[:] = float(rho_in)
                return buf
            arr = np.asarray(rho_in, dtype=float)
            if arr.ndim == 0:
                buf[:] = float(arr)
                return buf
            if arr.size == n:
                buf[:] = arr
                return buf
            if slices is not None and arr.size == len(slices):
                for v, s in zip(arr, slices):
                    buf[s] = float(v)
                return buf
            # Fallback: broadcast mean
            buf[:] = float(np.mean(arr))
            return buf

        # Initialize block rho from last solve (if available). If scalar, broadcast to blocks when slices exist.
        def _init_block_rho():
            last = getattr(self, 'rho_last', self.rho0)
            slices = self.component_slices
            if slices is None:
                return float(last) if np.isscalar(last) else float(np.mean(np.asarray(last, dtype=float)))
            m = len(slices)
            if np.isscalar(last):
                return np.full(m, float(last), dtype=float)
            arr = np.asarray(last, dtype=float).reshape(-1)
            if arr.size == m:
                return arr.copy()
            return np.full(m, float(np.mean(arr)), dtype=float)

        k = 0
        yk = y0.copy()
        n_vi = yk.size
        # Pre-allocate reusable buffers for the VI iteration hot loop
        _candidate_buf = np.empty(n_vi, dtype=float)
        _proj_cand_buf = np.empty(n_vi, dtype=float)
        _r_buf = np.empty(n_vi, dtype=float)
        debug = bool(getattr(self, 'debug_vi', False))

        # Use per-block rho when component_slices is defined; otherwise scalar
        if self.component_slices is not None and len(self.component_slices) > 0:
            rho = _init_block_rho()  # shape (n_blocks,)
        else:
            rho = float(getattr(self, 'rho_last', self.rho0))
        # Sanitize initial rho and persist immediately so bad values don't leak into next solves
        rho = _sanitize_rho(rho, context="init")
        self._set_rho_last(rho)
        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)
        if debug:
            if self.component_slices is not None and len(self.component_slices) > 0:
                print(f"[VI] init: blocks={len(self.component_slices)} rho={rho}")
            else:
                print(f"[VI] init: scalar rho={rho:.3e}")

        Fk_val = func(yk)
        self.last_Fk_val = Fk_val
        # Candidate uses per-index scaling; projector must receive the same per-index rho
        rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
        # Guard against non-finite rho_vec
        if not np.all(np.isfinite(rho_vec)):
            rho = _sanitize_rho(rho, context="expand-init")
            rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
        np.subtract(yk, rho_vec * Fk_val, out=_candidate_buf)
        if not np.all(np.isfinite(_candidate_buf)):
            # Reduce rho and try once more
            rho = _sanitize_rho(rho * 0.1 if np.isscalar(rho) else rho * 0.1, context="candidate-init")
            rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
            np.subtract(yk, rho_vec * Fk_val, out=_candidate_buf)
        y_proj = self._project(yk, _candidate_buf, rho_vec, tcur, Fk_val, prev)

        # Block-wise L2 natural residual at yk
        np.subtract(yk, y_proj, out=_r_buf)
        err = _rel_block_l2(_r_buf, yk, self.component_slices)
        if not np.isfinite(err):
            # If projection resulted in non-finite error, reset rho to safe value and recompute once
            rho = _sanitize_rho(self.rho0 if np.isfinite(self.rho0) else 1.0, context="err-init-reset")
            rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
            np.subtract(yk, rho_vec * Fk_val, out=_candidate_buf)
            y_proj = self._project(yk, _candidate_buf, rho_vec, tcur, Fk_val, prev)
            np.subtract(yk, y_proj, out=_r_buf)
            err = _rel_block_l2(_r_buf, yk, self.component_slices)
        if debug:
            print(f"[VI] k={k} err={err:.3e}")

        # Convergence threshold: wrms <= 1 or legacy err <= tol
        _conv_thresh = 1.0 if _use_wrms else self.tol

        while err > _conv_thresh and k < self.max_iter:
            # Project with current rho
            tcur = getattr(self, 'current_time', None)
            prev = getattr(self, 'prev_state', None)

            # Fk_val is carried forward from previous iteration (or init);
            # no redundant func(yk) call needed.
            self.last_Fk_val = Fk_val
            rho = _sanitize_rho(rho, context="iter-pre")
            rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
            np.subtract(yk, rho_vec * Fk_val, out=_candidate_buf)
            if not np.all(np.isfinite(_candidate_buf)):
                rho = _sanitize_rho(rho * 0.5 if np.isscalar(rho) else rho * 0.5, context="iter-candidate")
                rho_vec = _expand_rho_to_vec(rho, n_vi, self.component_slices)
                np.subtract(yk, rho_vec * Fk_val, out=_candidate_buf)
            yk1 = self._project(yk, _candidate_buf, rho_vec, tcur, Fk_val, prev)

            # Evaluate at new point and compute residual for error
            Fk_val_1 = func(yk1)
            rho = _sanitize_rho(rho, context="iter-post-proj")
            rho_vec1 = _expand_rho_to_vec(rho, n_vi, self.component_slices)
            np.subtract(yk1, rho_vec1 * Fk_val_1, out=_proj_cand_buf)
            proj_yk1 = self._project(yk1, _proj_cand_buf, rho_vec1, tcur, Fk_val_1, prev)

            # Block-wise L2 natural residual at yk1
            np.subtract(yk1, proj_yk1, out=_r_buf)
            err = _rel_block_l2(_r_buf, yk1, self.component_slices)
            if not np.isfinite(err):
                if debug:
                    print(f"[VI] non-finite err encountered; shrinking rho and retrying one step")
                rho = _sanitize_rho(rho * 0.5 if np.isscalar(rho) else rho * 0.5, context="iter-err-reset")
                rho_vec1 = _expand_rho_to_vec(rho, n_vi, self.component_slices)
                np.subtract(yk1, rho_vec1 * Fk_val_1, out=_proj_cand_buf)
                proj_yk1 = self._project(yk1, _proj_cand_buf, rho_vec1, tcur, Fk_val_1, prev)
                np.subtract(yk1, proj_yk1, out=_r_buf)
                err = _rel_block_l2(_r_buf, yk1, self.component_slices)

            # Update rho per block
            if self.component_slices is not None and len(self.component_slices) > 0:
                rb = np.asarray(rho, dtype=float).copy()
                if self.vi_strict_block_lipschitz:
                    # Strict component-wise Lipschitz enforcement with re-projections
                    yk_current = yk1.copy()
                    Fk_current = Fk_val_1.copy()
                    for i, s in enumerate(self.component_slices):
                        # Stuck detection for this block relative to yk
                        den_initial = np.linalg.norm(yk_current[s] - yk[s])
                        stuck_thresh = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + np.linalg.norm(yk[s]))
                        if den_initial < stuck_thresh:
                            continue

                        iter_count = 0
                        rk_i = np.inf
                        # Increase rho[i] until Lipschitz satisfied or max iters
                        while iter_count < self.vi_max_block_adjust_iters:
                            rho_vec_rb = _expand_rho_to_vec(rb, n_vi, self.component_slices)
                            candidate = yk - rho_vec_rb * Fk_val
                            yk_temp = self._project(yk, candidate, rho_vec_rb, tcur, Fk_val, prev)
                            Fk_temp = func(yk_temp)

                            den = np.linalg.norm(yk_temp[s] - yk[s])
                            if den < stuck_thresh:
                                # Component stuck; stop adjusting this block
                                break
                            num = rb[i] * np.linalg.norm(Fk_temp[s] - Fk_val[s])
                            rk_i = num / den if den != 0.0 else 0.0
                            if rk_i > self.L:
                                rb[i] = self.nu * rb[i]
                                iter_count += 1
                            else:
                                # Lipschitz satisfied
                                yk_current = yk_temp
                                Fk_current = Fk_temp
                                break

                        # Optional single decrease if too small (mirror scalar logic)
                        if np.isfinite(rk_i) and rk_i < self.Lmin:
                            rb[i] = (1.0 / self.nu) * rb[i]
                            # We do not re-check after decrease (to match scalar path semantics)

                        # Recompute current state after this block's rho change
                        rho_vec_rb = _expand_rho_to_vec(rb, n_vi, self.component_slices)
                        candidate = yk - rho_vec_rb * Fk_val
                        yk_current = self._project(yk, candidate, rho_vec_rb, tcur, Fk_val, prev)
                        Fk_current = func(yk_current)

                    # After all components adjusted, update outputs with current state
                    yk1 = yk_current
                    Fk_val_1 = Fk_current
                    # Ensure positivity and clamp
                    rb = np.maximum(rb, np.finfo(float).tiny)
                    np.clip(rb, self.rho_min, self.rho_max, out=rb)
                    rho = _sanitize_rho(rb, context="iter-update-strict")
                else:
                    # Fast per-block update without extra projections
                    for i, s in enumerate(self.component_slices):
                        num = rb[i] * np.linalg.norm(Fk_val_1[s] - Fk_val[s])
                        den = np.linalg.norm(yk1[s] - yk[s])
                        # Detect stuck components: absolute + relative threshold
                        stuck_thresh = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + np.linalg.norm(yk[s]))
                        if den < stuck_thresh:
                            # Skip update for stuck block
                            continue
                        rk_i = num / den
                        if rk_i > self.L:
                            rb[i] = self.nu * rb[i]
                        elif rk_i < self.Lmin:  # strict < to avoid growth bias at boundary
                            rb[i] = (1.0 / self.nu) * rb[i]
                    # Ensure positivity and clamp
                    rb = np.maximum(rb, np.finfo(float).tiny)
                    np.clip(rb, self.rho_min, self.rho_max, out=rb)
                    # Sanitize and clip per-block
                    rho = _sanitize_rho(rb, context="iter-update")
            else:
                # Scalar update with stuck detection
                rhos = float(rho)
                num = rhos * np.linalg.norm(Fk_val_1 - Fk_val)
                den = np.linalg.norm(yk1 - yk)
                stuck_thresh = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + np.linalg.norm(yk))
                if den >= stuck_thresh:
                    rk = num / den
                    if rk > self.L:
                        rhos = self.nu * rhos
                    elif rk < self.Lmin:
                        rhos = (1.0 / self.nu) * rhos
                # sanitize and clamp
                rhos = float(np.clip(rhos, self.rho_min, self.rho_max))
                rho = _sanitize_rho(rhos, context="iter-update-scalar")

            if debug:
                if isinstance(rho, np.ndarray):
                    print(f"[VI] k={k+1} err={err:.3e} rho={rho}")
                else:
                    print(f"[VI] k={k+1} err={err:.3e} rho={rho:.3e}")

            yk = yk1
            # Carry forward Fk_val_1 as next iteration's Fk_val (avoids redundant func call)
            Fk_val = Fk_val_1
            k += 1

        success = (err <= _conv_thresh)
        # Persist last rho for subsequent solves (both cached and default field)
        rho = _sanitize_rho(rho, context="final")
        self._set_rho_last(rho, update_default=True)
        # Fk_val is already up-to-date from the last iteration (or init if 0 iterations)
        F_final = Fk_val
        self.last_Fk_val = F_final
        if debug:
            if isinstance(rho, np.ndarray):
                print(f"[VI] done: success={success} iters={k} final_err={err:.3e} rho={rho}")
            else:
                print(f"[VI] done: success={success} iters={k} final_err={err:.3e} rho={rho:.3e}")
        return (yk, F_final, err, success, k)


    # # ---------------- VI (projected fixed-point) ----------------
    # def _solve_with_VI(self, func, y0):
    #     k = 0
    #     yk = y0.copy()
    #     n=yk.size

    #     rho = self.rho0
    #     tcur = getattr(self, 'current_time', None)
    #     prev = getattr(self, 'prev_state', None)

    #     Fk_val = func(yk)
    #     self.last_Fk_val = Fk_val
    #     y_proj = self._project(yk, yk - rho * Fk_val, rho, tcur, Fk_val, prev)
    #     err = np.linalg.norm(yk - y_proj)/math.sqrt(n)

    #     while err > self.tol and k < self.max_iter:
    #         rho = self._update_rho(func, yk, rho)
    #         tcur = getattr(self, 'current_time', None)
    #         prev = getattr(self, 'prev_state', None)

    #         Fk_val = func(yk)
    #         self.last_Fk_val = Fk_val
    #         yk1 = self._project(yk, yk - rho * Fk_val, rho, tcur, Fk_val, prev)

    #         Fk_val_1 = func(yk1)
    #         err = np.linalg.norm(yk1 - self._project(yk1, yk1 - rho * Fk_val_1, rho, tcur, Fk_val_1, prev))/math.sqrt(n)
    #         yk = yk1
    #         k += 1

    #     success = (err <= self.tol)
    #     return (yk, func(yk), err, success, k)
    
    # # ---- VI stepsize update (unchanged math, but uses fast projection) ----
    # def _update_rho(self, func, yk, rho):
    #     # guard against bad rho
    #     if not np.isscalar(rho) or not np.isfinite(rho) or rho <= 0:
    #         base = self.rho0 if (np.isscalar(self.rho0) and np.isfinite(self.rho0) and self.rho0 > 0) else 1.0
    #         rho = float(base)

    #     tcur = getattr(self, 'current_time', None)
    #     prev = getattr(self, 'prev_state', None)

    #     Fk_val = func(yk)
    #     self.last_Fk_val = Fk_val
    #     # Use per-index rho_vec consistently for projection
    #     slices = getattr(self, 'component_slices', None)
    #     if slices is not None and len(slices) > 0:
    #         rho_vec = np.empty_like(yk, dtype=float)
    #         # expand scalar rho to blocks then to vector
    #         rb = np.full(len(slices), float(rho), dtype=float)
    #         for v, s in zip(rb, slices):
    #             rho_vec[s] = v
    #     else:
    #         rho_vec = float(rho) * np.ones_like(yk, dtype=float)
    #     yk1 = self._project(yk, yk - rho_vec * Fk_val, rho_vec, tcur, Fk_val, prev)
    #     # Stuck detection for scalar path
    #     den = np.linalg.norm(yk1 - yk)
    #     stuck_thresh = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + np.linalg.norm(yk))
    #     if den >= stuck_thresh:
    #         rk = self._get_rk(func, yk1, yk, rho)
    #         while rk > self.L:
    #             rho = self.nu * rho
    #             # refresh rho_vec
    #             if slices is not None and len(slices) > 0:
    #                 rb = np.full(len(slices), float(rho), dtype=float)
    #                 for v, s in zip(rb, slices):
    #                     rho_vec[s] = v
    #             else:
    #                 rho_vec.fill(float(rho))
    #             yk1 = self._project(yk, yk - rho_vec * Fk_val, rho_vec, tcur, Fk_val, prev)
    #             den = np.linalg.norm(yk1 - yk)
    #             if den < stuck_thresh:
    #                 break
    #             rk = self._get_rk(func, yk1, yk, rho)
    #         if rk < self.Lmin:
    #             rho = (1.0 / self.nu) * rho
    #     # Clamp
    #     rho = float(np.clip(rho, self.rho_min, self.rho_max))
    #     return rho

    # def _get_rk(self, func, yk1, yk, rho):
    #     num = rho * np.linalg.norm(func(yk1) - func(yk))
    #     den = np.linalg.norm(yk1 - yk)
    #     return 0.0 if den == 0.0 else (num / den)



    # ---- VI stepsize update (unchanged math, but uses fast projection) ----
    def _update_rho(self, func, yk, rho, Fk_val=None):
        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)

        if Fk_val is None:
            Fk_val = func(yk)
            self.last_Fk_val = Fk_val
        yk1 = self._project(yk, yk - rho * Fk_val, rho, tcur, Fk_val, prev)
        rk = self._get_rk(func, yk1, yk, rho)
        while rk > self.L:
            rho = self.nu * rho
            if rho < self._lam_floor:
                rho = self._lam_floor
                break
            yk1 = self._project(yk, yk - rho * Fk_val, rho, tcur, Fk_val, prev)
            rk = self._get_rk(func, yk1, yk, rho)
        if rk <= self.Lmin:
            rho = (1.0 / self.nu) * rho
        return rho

    def _get_rk(self, func, yk1, yk, rho):
        num = rho * np.linalg.norm(func(yk1) - func(yk))
        den = np.linalg.norm(yk1 - yk)
        return 0.0 if den == 0.0 else (num / den)

    # ---------- Numerical Jacobian ----------
    def _numerical_jacobian(self, func, y, eps: float | None = None, sparse: bool | None = None, mode: str = 'fd'):
        n = len(y)
        use_sparse = self._sparse_active(n) if sparse is None else bool(sparse)

        if (mode or 'fd').lower() == 'cs':
            try:
                h = 1e-30
                # Vectorized complex-step: perturb all columns at once
                Y_cs = np.tile(y.astype(complex), (n, 1))  # (n, n)
                Y_cs[np.arange(n), np.arange(n)] += 1j * h
                # Evaluate all perturbed points
                J = np.empty((n, n), dtype=float)
                for i in range(n):
                    J[:, i] = np.imag(func(Y_cs[i])) / h
                return J if not use_sparse else sp.csr_matrix(J)
            except Exception:
                pass

        # --- Coloring-based sparse Jacobian when sparsity pattern is known ---
        sparsity = self.jacobian_sparsity
        if sparsity is not None and use_sparse:
            if sparsity.shape != (n, n):
                raise ValueError(
                    f"jacobian_sparsity has shape {sparsity.shape}, expected {(n, n)}"
                )
            try:
                from scipy.optimize._numdiff import approx_derivative
                J_csr = approx_derivative(
                    func, y, method='2-point',
                    sparsity=sparsity,
                    rel_step=eps,
                )
                if sp.issparse(J_csr):
                    return J_csr.tocsr()
                return sp.csr_matrix(J_csr)
            except Exception:
                pass  # fall through to vectorized FD

        # --- Vectorized finite differences ---
        F0 = func(y)
        base = np.sqrt(np.finfo(float).eps) if eps is None else float(eps)
        h_vec = base * np.maximum(1.0, np.abs(y))  # (n,) per-column step sizes

        # Build all perturbed states at once: Y_pert[i,:] = y with y[i] += h_vec[i]
        Y_pert = np.tile(y, (n, 1))  # (n, n)
        Y_pert[np.arange(n), np.arange(n)] += h_vec

        # Evaluate all perturbed points (n func calls, but avoid per-call y.copy())
        J = np.empty((n, n), dtype=y.dtype)
        for i in range(n):
            J[:, i] = (func(Y_pert[i]) - F0) / h_vec[i]

        return J if not use_sparse else sp.csr_matrix(J)

    def _compute_jacobian_csr(self, func, y, sparse_active):
        """Compute Jacobian and return as CSR, using caching to avoid dense-to-sparse overhead."""
        # 1. Compute raw Jacobian (dense or sparse)
        if self.jacobian is not None:
            J_raw = self.jacobian(y)
        else:
            J_raw = self._numerical_jacobian(func, y, sparse=sparse_active)

        if not sparse_active:
            return J_raw

        # Check if cache needs reset due to shape change
        if self._J_cached is not None and self._J_cached.shape != (len(y), len(y)):
            self._J_cached = None
            self._J_rows = None

        # 2. Convert/Cache for Sparse
        if self._J_cached is not None:
            if sp.issparse(J_raw):
                # Assume constant pattern: copy data
                if self._J_cached.data.size == J_raw.data.size:
                     self._J_cached.data[:] = J_raw.data
                     return self._J_cached
                else:
                     # Pattern changed? Rebuild
                     self._J_cached = self._to_csr(J_raw)
                     self._J_rows = None
                     return self._J_cached
            elif isinstance(J_raw, np.ndarray):
                # Dense update into sparse cache
                if self._J_rows is None:
                     n = self._J_cached.shape[0]
                     self._J_rows = np.repeat(np.arange(n), np.diff(self._J_cached.indptr))
                
                # Fast update using fancy indexing
                try:
                    self._J_cached.data[:] = J_raw[self._J_rows, self._J_cached.indices]
                    return self._J_cached
                except Exception:
                    pass

        # 3. Build Cache (First time or fallback)
        J_csr = self._to_csr(J_raw)
        self._J_cached = J_csr
        
        # If J_raw was dense, prepare rows for future fast updates
        if isinstance(J_raw, np.ndarray) and not sp.issparse(J_raw):
             n = J_csr.shape[0]
             self._J_rows = np.repeat(np.arange(n), np.diff(J_csr.indptr))
        
        return J_csr

    def _compute_tangent_csr(self, candidate, y, lam, tcur, F_in, prev, n):
        """Compute Tangent and return as CSR, using caching to avoid dense-to-sparse overhead.

        Returns either a single CSR matrix ``D`` (legacy) or a tuple
        ``(D_cand, D_state)`` when the projector exposes
        ``tangent_cone_split``, signalling that the projection depends on
        ``current_state`` (e.g. Moreau De Saxcé augmentation).
        """
        # 1. Check for split tangent (state-dependent projector)
        P = getattr(self, 'proj', None)
        if (P is not None and hasattr(P, 'tangent_cone_split')
                and callable(getattr(P, 'tangent_cone_split'))):
            # Split derivative: (dP/dcandidate, dP/dcurrent_state)
            D_cand_raw, D_state_raw = P.tangent_cone_split(
                candidate, y, rhok=lam, t=tcur, Fk_val=F_in, prev_state=prev)
            return (self._to_csr(D_cand_raw, n), self._to_csr(D_state_raw, n))

        # 2. Legacy single-tangent path
        D_raw = self._tangent(candidate, y, lam, tcur, F_in, prev)

        # Check if cache needs reset due to shape change
        if self._D_cached is not None and self._D_cached.shape != (n, n):
            self._D_cached = None
            self._D_rows = None

        # 2. Convert/Cache for Sparse
        if self._D_cached is not None:
            if sp.issparse(D_raw):
                # Assume constant pattern: copy data
                if self._D_cached.data.size == D_raw.data.size:
                     self._D_cached.data[:] = D_raw.data
                     return self._D_cached
                else:
                     # Pattern changed? Rebuild
                     self._D_cached = self._to_csr(D_raw, n)
                     self._D_rows = None
                     return self._D_cached
            elif isinstance(D_raw, np.ndarray):
                # Dense update into sparse cache
                if self._D_rows is None:
                     # If D_raw is 1D (diagonal), handle separately or treat as dense diagonal
                     if D_raw.ndim == 1:
                         # 1D array -> diagonal matrix. 
                         # If cached structure matches diagonal, we can update data directly?
                         # Usually sp.diags is fast, but let's see if we can update in place.
                         # For diagonal CSR, indices are 0..n-1, indptr is 0..n.
                         # It's safer/easier to just use sp.diags unless we really want to cache structure.
                         # Given profiling showed _to_csr cost, let's cache the diagonal structure too.
                         pass
                     else:
                         self._D_rows = np.repeat(np.arange(n), np.diff(self._D_cached.indptr))
                
                # Fast update using fancy indexing
                try:
                    if D_raw.ndim == 1:
                        # Special case: D_raw is diagonal vector, D_cached is CSR diagonal
                        # CSR diagonal data is just the vector itself if constructed via sp.diags(v).tocsr()
                        # But let's be safe:
                        self._D_cached.data[:] = D_raw
                    else:
                        self._D_cached.data[:] = D_raw[self._D_rows, self._D_cached.indices]
                    return self._D_cached
                except Exception:
                    pass

        # 3. Build Cache (First time or fallback)
        D_csr = self._to_csr(D_raw, n)
        self._D_cached = D_csr
        
        # If D_raw was dense, prepare rows for future fast updates
        if isinstance(D_raw, np.ndarray) and not sp.issparse(D_raw):
             if D_raw.ndim == 2:
                 self._D_rows = np.repeat(np.arange(n), np.diff(D_csr.indptr))
             # For 1D, we don't need rows, we just copy 1:1 if structure matches
        
        return D_csr

    # ---------- Sparse helpers ----------
    def _to_csr(self, A, n=None):
        if sp.issparse(A):
            return A if isinstance(A, sp.csr_matrix) else A.tocsr()
        if isinstance(A, np.ndarray):
            if A.ndim == 1:
                return sp.diags(A, format='csr')
            return sp.csr_matrix(A)
        if n is not None:
            try:
                return sp.csr_matrix(A, shape=(n, n))
            except Exception:
                pass
        return sp.csr_matrix(A)

    def _extract_sparse_numeric_diagonal(self, D_csr, n):
        """Detect numerically diagonal sparse matrices even with stored zeros.

        Returns ``(True, diag)`` when every nonzero entry lies on the diagonal
        and there is at most one diagonal entry per row. This lets the Newton
        assembly fast path remain active for fixed sparse patterns that keep
        structural zero off-diagonals (for example full SOC blocks whose
        interior/polar values are numerically diagonal).
        """
        D_csr = self._to_csr(D_csr, n)
        if D_csr.shape != (n, n):
            return (False, None)

        data = D_csr.data
        if data.size == 0:
            return (True, np.zeros(n, dtype=float))

        nz_mask = data != 0
        if not np.any(nz_mask):
            return (True, np.zeros(n, dtype=data.dtype))

        row_idx = np.repeat(np.arange(n), np.diff(D_csr.indptr))
        nz_rows = row_idx[nz_mask]
        nz_cols = D_csr.indices[nz_mask]
        if np.any(nz_cols != nz_rows):
            return (False, None)

        counts = np.bincount(nz_rows, minlength=n)
        if np.any(counts > 1):
            return (False, None)

        diag = np.zeros(n, dtype=data.dtype)
        diag[nz_rows] = data[nz_mask]
        return (True, diag)

    def _prepare_diag_newton_template(self, J_csr):
        """Cache row metadata for exact diagonal-tangent sparse assembly."""
        key = (J_csr.shape, J_csr.nnz, id(J_csr.indptr), id(J_csr.indices))
        if self._diag_newton_key == key and self._diag_newton_out is not None:
            return True

        n = J_csr.shape[0]
        row_idx = np.repeat(np.arange(n), np.diff(J_csr.indptr))
        diag_pos = np.full(n, -1, dtype=np.int64)
        has_full_diag = True
        for i in range(n):
            start = J_csr.indptr[i]
            stop = J_csr.indptr[i + 1]
            row_cols = J_csr.indices[start:stop]
            hits = np.flatnonzero(row_cols == i)
            if hits.size == 0:
                has_full_diag = False
                break
            diag_pos[i] = start + hits[0]

        self._diag_newton_key = key
        self._diag_newton_row_idx = row_idx
        self._diag_newton_diag_pos = diag_pos if has_full_diag else None
        self._diag_newton_out = None

        if not has_full_diag:
            return False

        self._diag_newton_out = sp.csr_matrix(
            (np.empty_like(J_csr.data), J_csr.indices.copy(), J_csr.indptr.copy()),
            shape=J_csr.shape,
        )
        return True

    def _assemble_diag_newton_csr(self, J_csr, d_diag, lam):
        """Assemble ``diag(1-d) + diag(d*lam) @ J`` without sparse temporaries."""
        if not self._prepare_diag_newton_template(J_csr):
            return None

        out = self._diag_newton_out
        row_idx = self._diag_newton_row_idx
        diag_pos = self._diag_newton_diag_pos

        d_arr = np.asarray(d_diag)
        if np.ndim(lam) >= 1:
            scale = d_arr * np.asarray(lam)
        else:
            scale = d_arr * float(lam)

        np.multiply(J_csr.data, scale[row_idx], out=out.data)
        out.data[diag_pos] += (1.0 - d_arr)
        return out

    def _solve_linear_sparse(self, J, rhs, rtol=None, pattern_hint=None):
        n = J.shape[0]
        b = rhs if (isinstance(rhs, np.ndarray) and rhs.ndim == 1) else np.asarray(rhs).reshape(n)

        # Matrix-free / LinearOperator path: use GMRES without ILU/SPLU
        if isinstance(J, spla.LinearOperator):
            x, info = self._gmres(
                J, b,
                rtol=(self.gmres_tol if rtol is None else rtol),
                maxiter=self.gmres_maxiter,
                restart=self.gmres_restart,
            )
            return (x, info == 0)

        # PETSc path (optional, requires petsc4py)
        if self.linear_solver == 'petsc':
            if not PETSC_AVAILABLE:
                warnings.warn(
                    "PETSc not available. Install with: pip install solve_nivp[petsc] "
                    "or conda install -c conda-forge petsc4py. Falling back to GMRES+ILU.",
                    UserWarning,
                )
                # Fall through to ILU path below
            else:
                return self._solve_with_petsc(J, b, rtol=rtol)

        if self.linear_solver == 'umfpack':
            if not UMFPACK_AVAILABLE:
                warnings.warn(
                    "UMFPACK not available. Install with: pip install scikit-umfpack. "
                    "Falling back to splu.",
                    UserWarning, stacklevel=2,
                )
                self.linear_solver = 'splu'  # fall through below
            else:
                nnz = getattr(J, 'nnz', None)
                pattern_key = (J.shape, nnz, pattern_hint)
                reuse_budget = int(self.precond_reuse_steps)

                need_rebuild = (
                    self._lu is None
                    or self._lu_shape != J.shape
                    or self._lu_pattern != pattern_key
                    or reuse_budget <= 0
                    or self._lu_use_count >= reuse_budget
                )

                if need_rebuild:
                    try:
                        matrix = J
                        if not sp.issparse(matrix):
                            matrix = sp.csc_matrix(matrix)
                        elif not sp.isspmatrix_csc(matrix):
                            matrix = matrix.tocsc()
                        # --- UMFPACK symbolic/numeric separation ---
                        # The UmfpackContext lives in _umf_ctx (NOT _lu)
                        # so that SDIRK2's h-change invalidation (which
                        # sets _lu = None) does NOT destroy the cached
                        # symbolic analysis.  Symbolic depends only on
                        # the sparsity pattern; numeric depends on values.
                        sym_key = (J.shape, nnz)
                        if (self._umf_ctx is not None
                                and self._umf_symbolic_key == sym_key):
                            # Same pattern → reuse symbolic, redo numeric only
                            self._umf_ctx.numeric(matrix)
                        else:
                            # New pattern → full symbolic + numeric
                            umf = UmfpackContext("di")
                            umf.symbolic(matrix)
                            umf.numeric(matrix)
                            self._umf_ctx = umf
                            self._umf_symbolic_key = sym_key
                        self._lu = self._umf_ctx   # alias for solve + reuse check
                        self._lu_matrix = matrix
                        self._lu_use_count = 0
                        self._lu_pattern = pattern_key
                        self._lu_shape = J.shape
                    except Exception:
                        self._lu = None
                        self._lu_matrix = None
                        self._lu_pattern = None
                        self._lu_shape = None
                        self._umf_ctx = None
                        self._umf_symbolic_key = None

                if self._lu is not None:
                    try:
                        x = self._lu.solve(
                            _UMFPACK_A, self._lu_matrix, b,
                            autoTranspose=True,
                        )
                        self._lu_use_count += 1
                        return x, True
                    except Exception:
                        self._lu = None
                        self._lu_matrix = None
                        self._lu_pattern = None
                        self._lu_shape = None
                        self._umf_ctx = None
                        self._umf_symbolic_key = None

                # Fallback to GMRES
                x, info = self._gmres(
                    J, b,
                    rtol=(self.gmres_tol if rtol is None else rtol),
                    maxiter=self.gmres_maxiter,
                    restart=self.gmres_restart,
                )
                return (x, info == 0)

        if self.linear_solver == 'splu':
            nnz = getattr(J, 'nnz', None)
            pattern_key = (J.shape, nnz, pattern_hint)
            reuse_budget = int(self.precond_reuse_steps)

            def _build_lu():
                matrix = J
                if not sp.issparse(matrix):
                    matrix = sp.csc_matrix(matrix)
                elif not sp.isspmatrix_csc(matrix):
                    matrix = matrix.tocsc()
                self._lu = spla.splu(matrix, permc_spec=self.splu_permc_spec)
                self._lu_use_count = 0
                self._lu_pattern = pattern_key
                self._lu_shape = J.shape

            need_rebuild = (
                self._lu is None
                or self._lu_shape != J.shape
                or self._lu_pattern != pattern_key
                or reuse_budget <= 0
                or self._lu_use_count >= reuse_budget
            )

            if need_rebuild:
                try:
                    _build_lu()
                except Exception:
                    self._lu = None
                    self._lu_pattern = None
                    self._lu_shape = None

            if self._lu is not None:
                try:
                    x = self._lu.solve(b)
                    self._lu_use_count += 1
                    return x, True
                except Exception:
                    # Drop stale factorization and try a single rebuild before falling back
                    self._lu = None
                    self._lu_pattern = None
                    self._lu_shape = None
                    try:
                        _build_lu()
                    except Exception:
                        self._lu = None
                        self._lu_pattern = None
                        self._lu_shape = None
                    else:
                        x = self._lu.solve(b)
                        self._lu_use_count += 1
                        return x, True

            # Fallback to GMRES if SPLU failed
            x, info = self._gmres(
                J, b,
                rtol=(self.gmres_tol if rtol is None else rtol),
                maxiter=self.gmres_maxiter,
                restart=self.gmres_restart,
            )
            return (x, info == 0)
        else:
            M = None
            need_rebuild = (
                self._ilu is None or self._last_shape != J.shape or self._ilu_steps_since_build >= self.precond_reuse_steps
            )
            pattern_key = (J.shape, J.nnz, pattern_hint)
            if need_rebuild or self._last_pattern != pattern_key:
                try:
                    ilu = spla.spilu(
                        J.tocsc(),
                        drop_tol=self.ilu_drop_tol,
                        fill_factor=self.ilu_fill_factor,
                        permc_spec=self.ilu_permc_spec,
                    )
                    self._ilu = ilu
                    self._ilu_steps_since_build = 0
                    self._last_shape = J.shape
                    self._last_pattern = pattern_key
                except Exception:
                    self._ilu = None
            if self._ilu is not None:
                ilu = self._ilu
                M = spla.LinearOperator(J.shape, matvec=lambda v: ilu.solve(v))
                self._ilu_steps_since_build += 1
            x, info = self._gmres(
                J, b, M=M,
                rtol=(self.gmres_tol if rtol is None else rtol),
                maxiter=self.gmres_maxiter,
                restart=self.gmres_restart,
            )
            return (x, info == 0)

    def _gmres(self, A, b, M=None, rtol=None, maxiter=None, restart=None):
        kwargs = {'M': M, 'maxiter': maxiter, 'restart': restart}
        try:
            return spla.gmres(A, b, rtol=(rtol if rtol is not None else self.gmres_tol), atol=0.0, **kwargs)
        except TypeError:
            return spla.gmres(A, b, tol=(rtol if rtol is not None else self.gmres_tol), **kwargs)

    def _sparse_active(self, n: int) -> bool:
        if isinstance(self.sparse, str):
            if self.sparse.lower() == 'auto':
                return n >= self.sparse_threshold
            return True
        return bool(self.sparse)

    def _petsc_comm_size(self, comm):
        if comm is None:
            return None
        for name in ('getSize', 'Get_size'):
            fn = getattr(comm, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
        size = getattr(comm, 'size', None)
        if size is not None:
            try:
                return int(size)
            except Exception:
                pass
        return None

    def _resolve_petsc_comm(self):
        """Resolve the communicator used for PETSc objects.

        Only single-rank PETSc solves are currently supported. ``petsc_comm``
        may still be ``'world'`` provided the effective communicator size is 1.
        """
        spec = self.petsc_comm
        if spec is None or spec == 'self':
            comm = PETSc.COMM_SELF
        elif isinstance(spec, str):
            key = spec.lower()
            if key == 'self':
                comm = PETSc.COMM_SELF
            elif key == 'world':
                comm = getattr(PETSc, 'COMM_WORLD', PETSc.COMM_SELF)
            else:
                raise ValueError("petsc_comm must be 'self', 'world', or a PETSc/mpi4py communicator object.")
        else:
            comm = spec

        return comm

    def _petsc_comm_rank(self, comm):
        if comm is None:
            return 0
        for name in ('getRank', 'Get_rank'):
            fn = getattr(comm, name, None)
            if callable(fn):
                try:
                    return int(fn())
                except Exception:
                    pass
        rank = getattr(comm, 'rank', None)
        if rank is not None:
            try:
                return int(rank)
            except Exception:
                pass
        return 0

    def _petsc_owned_range(self, n: int, comm):
        """Return the contiguous row ownership range for this rank."""
        size = self._petsc_comm_size(comm) or 1
        rank = self._petsc_comm_rank(comm)
        base, extra = divmod(int(n), int(size))
        start = rank * base + min(rank, extra)
        stop = start + base + (1 if rank < extra else 0)
        return int(start), int(stop)

    def _petsc_type_supported(self, kind, type_name):
        """Return True when the current PETSc build accepts the requested type."""
        if not PETSC_AVAILABLE or not type_name:
            return False

        key = (kind, str(type_name))
        cached = _PETSC_TYPE_SUPPORT_CACHE.get(key)
        if cached is not None:
            return cached

        ok = False
        obj = None
        try:
            if kind == 'mat':
                obj = PETSc.Mat().create(comm=PETSc.COMM_SELF)
                obj.setSizes((1, 1))
                obj.setType(type_name)
            elif kind == 'vec':
                obj = PETSc.Vec().create(comm=PETSc.COMM_SELF)
                obj.setSizes(1)
                obj.setType(type_name)
            else:
                raise ValueError(f"Unknown PETSc object kind '{kind}'")
            ok = True
        except Exception:
            ok = False
        finally:
            if obj is not None:
                try:
                    obj.destroy()
                except Exception:
                    pass

        _PETSC_TYPE_SUPPORT_CACHE[key] = ok
        return ok

    def _resolve_petsc_backend(self, opts):
        """Determine the effective PETSc backend after capability checks."""
        requested_mat_type = opts.get('mat_type')
        requested_vec_type = opts.get('vec_type')
        requested_gpu = (
            requested_mat_type in _PETSC_GPU_MAT_TYPES
            or requested_vec_type in _PETSC_GPU_VEC_TYPES
        )

        effective_mat_type = requested_mat_type
        effective_vec_type = requested_vec_type
        reasons = []

        if effective_mat_type in _PETSC_GPU_MAT_TYPES and not self._petsc_type_supported('mat', effective_mat_type):
            reasons.append(f"matrix type '{effective_mat_type}'")
            effective_mat_type = None
        if effective_vec_type in _PETSC_GPU_VEC_TYPES and not self._petsc_type_supported('vec', effective_vec_type):
            reasons.append(f"vector type '{effective_vec_type}'")
            effective_vec_type = None

        if effective_mat_type in _PETSC_GPU_MAT_TYPES and effective_vec_type is None:
            paired_vec = _PETSC_GPU_PAIR_FOR_MAT.get(effective_mat_type)
            if paired_vec and self._petsc_type_supported('vec', paired_vec):
                effective_vec_type = paired_vec
        if effective_vec_type in _PETSC_GPU_VEC_TYPES and effective_mat_type is None:
            paired_mat = _PETSC_GPU_PAIR_FOR_VEC.get(effective_vec_type)
            if paired_mat and self._petsc_type_supported('mat', paired_mat):
                effective_mat_type = paired_mat

        use_gpu = (
            effective_mat_type in _PETSC_GPU_MAT_TYPES
            and effective_vec_type in _PETSC_GPU_VEC_TYPES
        )

        if requested_gpu and not use_gpu:
            warn_key = (requested_mat_type, requested_vec_type)
            if warn_key not in self._petsc_gpu_warned:
                detail = ", ".join(reasons) if reasons else "the requested PETSc GPU backend"
                warnings.warn(
                    f"Requested PETSc GPU backend is unavailable on this PETSc build ({detail}). "
                    "Falling back to CPU PETSc objects.",
                    RuntimeWarning,
                )
                self._petsc_gpu_warned.add(warn_key)
            effective_mat_type = None
            effective_vec_type = None

        return effective_mat_type, effective_vec_type, use_gpu

    def _solve_with_petsc(self, J, b, rtol=None):
        """Solve linear system using PETSc with configurable Krylov solver and preconditioner.

        Parameters
        ----------
        J : scipy.sparse matrix or numpy.ndarray
            The system matrix.
        b : numpy.ndarray
            Right-hand side vector.
        rtol : float, optional
            Relative tolerance for the iterative solver.

        Returns
        -------
        x : numpy.ndarray
            Solution vector.
        success : bool
            True if the solver converged.
        
        Notes
        -----
        For GPU acceleration, set in petsc_options:
            'mat_type': 'aijcusparse',  # GPU sparse matrix
            'vec_type': 'cuda',          # GPU vectors
        And set environment variable before importing:
            os.environ['PETSC_OPTIONS'] = '-use_gpu_aware_mpi 0'
        """
        n = J.shape[0]
        opts = dict(self.petsc_options)
        comm = self._resolve_petsc_comm()
        comm_size = self._petsc_comm_size(comm) or 1
        distributed = comm_size > 1
        effective_mat_type, effective_vec_type, use_gpu = self._resolve_petsc_backend(opts)

        if distributed:
            raise NotImplementedError(
                "Distributed PETSc communicators are not supported yet."
            )

        # Convert J to CSR if needed
        if not sp.issparse(J):
            J_csr = sp.csr_matrix(J)
        elif not sp.isspmatrix_csr(J):
            J_csr = J.tocsr()
        else:
            J_csr = J

        if distributed and J_csr.shape[0] != J_csr.shape[1]:
            raise NotImplementedError(
                "Distributed PETSc prototype currently assumes square matrices."
            )

        row_start, row_stop = self._petsc_owned_range(J_csr.shape[0], comm)
        local_n = row_stop - row_start

        # Determine if we need to rebuild the KSP
        is_direct_solver = opts.get('ksp_type') == 'preonly' and opts.get('pc_type') in ('lu', 'cholesky')
        reuse_budget = max(1, self.petsc_reuse_steps)
        need_rebuild = (
            self._petsc_ksp is None
            or self._petsc_shape != J.shape
            or self._petsc_comm_obj is not comm
            or self._petsc_effective_mat_type != effective_mat_type
            or self._petsc_effective_vec_type != effective_vec_type
            or (is_direct_solver and self._petsc_build_count >= reuse_budget)
        )

        # For direct solvers, we want to reuse the factorization like SPLU does.
        # The key insight: SPLU reuses stale factorizations for precond_reuse_steps solves.
        # This works because Newton iteration is self-correcting.
        #
        # However, when Newton explicitly recomputes the Jacobian (convergence
        # is slow), the fresh matrix MUST be factorised — otherwise the Newton
        # step is based on a stale system and convergence stalls or fails.
        if getattr(self, '_petsc_needs_matrix_update', False):
            if is_direct_solver:
                need_rebuild = True
            else:
                self._petsc_pc_needs_update = True
            self._petsc_needs_matrix_update = False

        if is_direct_solver and not need_rebuild:
            # Reuse existing factorization without updating matrix
            # This is the key optimization that makes MUMPS competitive with SPLU
            pass  # Skip to solve phase
        elif need_rebuild:
            # Destroy old objects if they exist
            if self._petsc_mat is not None:
                try:
                    self._petsc_mat.destroy()
                except Exception:
                    pass
            if self._petsc_ksp is not None:
                try:
                    self._petsc_ksp.destroy()
                except Exception:
                    pass

            # Create PETSc matrix from scipy CSR
            if use_gpu:
                # Create GPU matrix with the effective PETSc matrix type.
                # Ensure indices are the correct type (int32 for most PETSc builds)
                J_local = J_csr[row_start:row_stop].tocsr() if distributed else J_csr
                indptr = J_local.indptr.astype(PETSc.IntType, copy=False)
                indices = J_local.indices.astype(PETSc.IntType, copy=False)
                data = np.ascontiguousarray(J_local.data, dtype=PETSc.ScalarType)

                self._petsc_mat = PETSc.Mat().create(comm=comm)
                self._petsc_mat.setType(effective_mat_type)
                if distributed:
                    self._petsc_mat.setSizes(((local_n, J_csr.shape[0]), (local_n, J_csr.shape[1])))
                else:
                    self._petsc_mat.setSizes(J_csr.shape)
                self._petsc_mat.setPreallocationCSR((indptr, indices))
                self._petsc_mat.setUp()
                self._petsc_mat.setValuesCSR(indptr, indices, data)
            else:
                # Standard CPU matrix. For multi-rank runs, each rank hands PETSc
                # only its owned contiguous row block while the outer solver still
                # retains the full SciPy matrix for residual/Jacobian evaluation.
                J_local = J_csr[row_start:row_stop].tocsr() if distributed else J_csr
                size_spec = ((local_n, J_csr.shape[0]), (local_n, J_csr.shape[1])) if distributed else J_csr.shape
                self._petsc_mat = PETSc.Mat().createAIJ(
                    size=size_spec,
                    csr=(J_local.indptr.astype(PETSc.IntType, copy=False),
                         J_local.indices.astype(PETSc.IntType, copy=False),
                         J_local.data),
                    comm=comm,
                )
            self._petsc_mat.assemble()

            self._petsc_ksp = PETSc.KSP().create(comm=comm)
            self._petsc_ksp.setOperators(self._petsc_mat)

            # Apply user options
            opts = self.petsc_options
            if 'ksp_type' in opts:
                self._petsc_ksp.setType(opts['ksp_type'])
            
            pc = self._petsc_ksp.getPC()
            if 'pc_type' in opts:
                pc.setType(opts['pc_type'])
            if opts.get('pc_type') == 'hypre' and 'pc_hypre_type' in opts:
                pc.setHYPREType(opts['pc_hypre_type'])
            
            # For direct solvers, set the solver type (MUMPS, SuperLU_dist, etc.)
            if opts.get('pc_type') in ('lu', 'cholesky') and 'pc_factor_mat_solver_type' in opts:
                pc.setFactorSolverType(opts['pc_factor_mat_solver_type'])
                # CRITICAL: Tell PETSc to reuse the factorization on subsequent solves!
                pc.setReusePreconditioner(True)

            # Set tolerances - prefer petsc_options values, fall back to class defaults
            effective_rtol = opts.get('ksp_rtol', rtol if rtol is not None else self.gmres_tol)
            effective_maxiter = opts.get('ksp_max_it', self.gmres_maxiter or 1000)
            self._petsc_ksp.setTolerances(rtol=effective_rtol, atol=1e-50, max_it=effective_maxiter)
            
            # Set GMRES restart if specified
            if opts.get('ksp_type') == 'gmres' and 'ksp_gmres_restart' in opts:
                self._petsc_ksp.setGMRESRestart(opts['ksp_gmres_restart'])

            # =========================================================================
            # Push ALL user options into PETSc options database with prefix FIRST
            # This MUST happen before setFieldSplitIS so sub-solver options are visible
            # =========================================================================
            prefix = 'nivp_'
            self._petsc_ksp.setOptionsPrefix(prefix)
            petsc_opts_db = PETSc.Options()
            for k, v in opts.items():
                # Skip keys we've already handled manually
                if k in ('ksp_type', 'pc_type', 'pc_hypre_type', 'pc_factor_mat_solver_type',
                         'ksp_rtol', 'ksp_max_it', 'ksp_gmres_restart',
                         'mat_type', 'vec_type'):
                    continue
                # Handle flag options (value is None or True)
                if v is None or v is True:
                    petsc_opts_db[prefix + k] = ''
                else:
                    petsc_opts_db[prefix + k] = str(v)
            
            # Field-split preconditioner for saddle-point systems
            # MUST be configured AFTER options are in database so sub-solver options work
            if opts.get('pc_type') == 'fieldsplit':
                if self.component_slices is not None and len(self.component_slices) >= 2:
                    # Create index sets for each field
                    # CRITICAL: Use numeric names '0', '1' to match option keys 'fieldsplit_0_*'
                    self._petsc_field_is = []
                    for i, sl in enumerate(self.component_slices):
                        indices = self._component_indices(sl, n=J.shape[0], dtype=PETSc.IntType)
                        is_field = PETSc.IS().createGeneral(indices, comm=comm)
                        self._petsc_field_is.append(is_field)
                        pc.setFieldSplitIS((str(i), is_field))  # '0', '1' not 'field0', 'field1'
                    
                    # Set fieldsplit type
                    fs_type = opts.get('pc_fieldsplit_type', 'schur')
                    pc.setFieldSplitType(getattr(PETSc.PC.CompositeType, fs_type.upper(), 
                                                  PETSc.PC.CompositeType.SCHUR))
                else:
                    warnings.warn("fieldsplit PC requires component_slices to define fields")

            # Allow command-line overrides and finalize setup
            self._petsc_ksp.setFromOptions()

            self._petsc_shape = J.shape
            self._petsc_build_count = 0
            self._petsc_use_gpu = use_gpu
            self._petsc_comm_obj = comm
            self._petsc_effective_mat_type = effective_mat_type
            self._petsc_effective_vec_type = effective_vec_type
            self._petsc_owned_rows = (row_start, row_stop)
            self._petsc_pc_needs_update = True  # First solve needs PC setup
        else:
            # Reuse existing KSP - only update matrix values if needed
            # For direct solvers, skip matrix update entirely (reuse factorization)
            # For iterative solvers, update matrix but reuse preconditioner
            if not is_direct_solver:
                try:
                    J_local = J_csr[row_start:row_stop].tocsr() if distributed else J_csr
                    self._petsc_mat.setValuesCSR(
                        J_local.indptr.astype(PETSc.IntType, copy=False),
                        J_local.indices.astype(PETSc.IntType, copy=False),
                        J_local.data,
                    )
                    self._petsc_mat.assemble()
                except Exception:
                    # If in-place update fails, rebuild
                    self._petsc_build_count = reuse_budget
                    return self._solve_with_petsc(J, b, rtol=rtol)

                # Refresh the preconditioner on demand, but keep the operator current.
                if self._petsc_build_count % reuse_budget == 0:
                    self._petsc_pc_needs_update = True

                if getattr(self, '_petsc_pc_needs_update', True):
                    pc = self._petsc_ksp.getPC()
                    pc.setUp()
                    self._petsc_pc_needs_update = False

        self._petsc_build_count += 1

        # Create PETSc vectors (GPU or CPU)
        if distributed:
            local_b = np.ascontiguousarray(b[row_start:row_stop], dtype=np.float64)
            b_petsc = PETSc.Vec().createMPI((local_n, n), comm=comm)
            b_petsc.setArray(local_b)
            x_petsc = self._petsc_mat.createVecRight()
        elif getattr(self, '_petsc_use_gpu', False) and effective_vec_type in _PETSC_GPU_VEC_TYPES:
            b_arr = np.ascontiguousarray(b, dtype=np.float64)

            b_petsc = PETSc.Vec().create(comm=comm)
            b_petsc.setType(effective_vec_type)
            b_petsc.setSizes(n)
            b_petsc.setUp()
            b_petsc.setArray(b_arr)

            x_petsc = PETSc.Vec().create(comm=comm)
            x_petsc.setType(effective_vec_type)
            x_petsc.setSizes(n)
            x_petsc.setUp()
        else:
            b_petsc = PETSc.Vec().createWithArray(b.copy(), comm=comm)
            x_petsc = self._petsc_mat.createVecRight()

        # Solve
        self._petsc_ksp.solve(b_petsc, x_petsc)

        # Extract solution (copy from GPU if needed)
        if distributed:
            scatter = None
            x_all = None
            try:
                scatter, x_all = PETSc.Scatter.toAll(x_petsc)
                scatter.scatter(
                    x_petsc,
                    x_all,
                    addv=PETSc.InsertMode.INSERT_VALUES,
                    mode=PETSc.ScatterMode.FORWARD,
                )
                x = x_all.getArray().copy()
            finally:
                if x_all is not None:
                    x_all.destroy()
                if scatter is not None:
                    scatter.destroy()
        else:
            x = x_petsc.getArray().copy()

        # Check convergence
        reason = self._petsc_ksp.getConvergedReason()
        iters = self._petsc_ksp.getIterationNumber()
        success = reason > 0  # Positive values indicate convergence

        # Log divergence reason for debugging
        if not success and reason < 0:
            reason_names = {
                -2: 'NULL', -3: 'MAX_ITERATIONS', -4: 'DTOL', -5: 'BREAKDOWN',
                -6: 'BREAKDOWN_BICG', -7: 'NONSYMMETRIC', -8: 'INDEFINITE_PC',
                -9: 'NANORINF', -10: 'INDEFINITE_MAT'
            }
            reason_str = reason_names.get(reason, f'UNKNOWN({reason})')
            warnings.warn(
                f"PETSc solver diverged: {reason_str} after {iters} iterations. "
                f"For indefinite/saddle-point matrices, use ksp_type='gmres' instead of 'cg'.",
                RuntimeWarning
            )

        # Cleanup vectors (but keep KSP and Mat for reuse)
        b_petsc.destroy()
        x_petsc.destroy()

        return (x, success)
