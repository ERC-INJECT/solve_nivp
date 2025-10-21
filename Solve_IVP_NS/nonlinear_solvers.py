

# nonlinear_solvers.py  (fast-path projector dispatch)

from __future__ import annotations

import inspect
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
import math

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
        use_autodiff: bool = False,
        autodiff_mode: str = 'numerical',
        L: float = 0.9,
        Lmin: float = 0.3,
        nu: float = 0.66,
        lam: float = 1.0,
        sparse: bool | str = 'auto',
        sparse_threshold: int = 200,
        linear_solver: str = 'gmres',
        precond_reuse_steps: int = 5,
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
    ) -> None:
        if method not in ['VI', 'semismooth_newton']:
            raise ValueError("Unsupported solver method. Use 'VI' or 'semismooth_newton'.")
        self.method = method
        self.jacobian = jacobian
        self.tol = tol
        self.max_iter = max_iter
        self.proj = proj
        self.rho0 = rho0
        self.delta = delta
        self.component_slices = component_slices
        self.use_autodiff = False
        self.autodiff_mode = 'numerical'
        self.L = L
        self.Lmin = Lmin
        self.nu = nu
        self.lam = lam

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
        self.lam_update_strategy = lam_update_strategy or 'vi'
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

        # Identity caches
        self._I_cache = {}

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
        # --- detect what the projector exposes ---
        has_prev_p = _supports(P.project, 'prev_state')
        has_step_p = _supports(P.project, 'step_size')
        has_rhok_p = _supports(P.project, 'rhok')          # keyword form
        has_rho_p  = _supports(P.project, 'rho')           # positional/keyword form
        has_t_p    = _supports(P.project, 't')
        has_Fk_p   = _supports(P.project, 'Fk_val')

        # ---- PROJECT BINDER ----
        if has_prev_p:
            def _project(cur, cand, rho, t, Fk, prev):
                # Only pass parameters the projector actually accepts
                kw = {}
                if has_t_p:   kw['t'] = t
                if has_Fk_p:  kw['Fk_val'] = Fk
                if has_step_p:
                    kw['step_size'] = getattr(self, 'prev_step', None)
                kw['prev_state'] = prev

                if has_rhok_p:
                    # projector expects keyword rhok
                    kw['rhok'] = rho
                    return P.project(cur, cand, **kw)
                elif has_rho_p:
                    # projector expects a 'rho' parameter (positional-or-keyword) — pass positionally
                    return P.project(cur, cand, rho, **kw)
                else:
                    # projector doesn't want any stepsize param
                    return P.project(cur, cand, **kw)
        else:
            def _project(cur, cand, rho, t, Fk, prev):
                kw = {}
                if has_t_p:   kw['t'] = t
                if has_Fk_p:  kw['Fk_val'] = Fk
                if has_step_p:
                    kw['step_size'] = getattr(self, 'prev_step', None)
                if has_rhok_p:
                    kw['rhok'] = rho
                    return P.project(cur, cand, **kw)
                elif has_rho_p:
                    return P.project(cur, cand, rho, **kw)
                else:
                    return P.project(cur, cand, **kw)

        self._project = _project

        # ---- TANGENT BINDER (unchanged except we accept both rhok/rho too) ----
        has_prev_t = _supports(P.tangent_cone, 'prev_state')
        has_step_t = _supports(P.tangent_cone, 'step_size')
        has_rhok_t = _supports(P.tangent_cone, 'rhok')
        has_rho_t  = _supports(P.tangent_cone, 'rho')
        has_t_t    = _supports(P.tangent_cone, 't')
        has_Fk_t   = _supports(P.tangent_cone, 'Fk_val')

        if has_prev_t:
            def _tangent(cand, cur, rho, t, Fk, prev):
                kw = {}
                if has_t_t:   kw['t'] = t
                if has_Fk_t:  kw['Fk_val'] = Fk
                if has_step_t:
                    kw['step_size'] = getattr(self, 'prev_step', None)
                kw['prev_state'] = prev
                if has_rhok_t:
                    kw['rhok'] = rho
                    return P.tangent_cone(cand, cur, **kw)
                elif has_rho_t:
                    return P.tangent_cone(cand, cur, rho, **kw)
                else:
                    return P.tangent_cone(cand, cur, **kw)
        else:
            def _tangent(cand, cur, rho, t, Fk, prev):
                kw = {}
                if has_t_t:   kw['t'] = t
                if has_Fk_t:  kw['Fk_val'] = Fk
                if has_step_t:
                    kw['step_size'] = getattr(self, 'prev_step', None)
                if has_rhok_t:
                    kw['rhok'] = rho
                    return P.tangent_cone(cand, cur, **kw)
                elif has_rho_t:
                    return P.tangent_cone(cand, cur, rho, **kw)
                else:
                    return P.tangent_cone(cand, cur, **kw)

        self._tangent = _tangent


    # ---------- Public API ----------
    def _func_wrapper(self, y):
        return self.func(y)

    def set_func(self, func):
        self.func = func

    def set_projection(self, proj):
        """Allow swapping projector at runtime, with fastpath rebind."""
        self.proj = proj
        self._bind_projector_fastpaths()

    def solve(self, func, y0):
        self.set_func(func)
        if self.method == 'VI':
            return self._solve_with_VI(func, y0)
        else:
            return self._solve_with_semismooth_newton(func, y0)

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

        for iteration in range(1, self.max_iter + 1):
            # cache context once per iteration
            tcur = getattr(self, 'current_time', None)
            prev = getattr(self, 'prev_state', None)

            # Optional adaptive lam
            if self.adaptive_lam and self.lam_update_strategy == 'vi':
                try:
                    lam = self._update_rho(func, y, lam)
                    self.lam = lam
                except Exception:
                    pass

            F_in = func(y)
            self.last_Fk_val = F_in  # cheap attribute write

            # candidate = y - lam F(y)
            np.subtract(y, lam * F_in, out=candidate)

            # projection (fastpath)
            proj_val = None
            P = self.proj
            # Use batched path only when projector exposes a project_batch and indices are contiguous blocks
            try:
                has_batch = hasattr(P, 'project_batch') and callable(P.project_batch)
                ci = getattr(P, 'constraint_indices', None)
                # decide if indices form contiguous ranges we can view as rows
                use_batch = False
                row_slices = None
                if has_batch and ci is not None and np.size(ci) > 0:
                    ci = np.asarray(ci)
                    ci_sorted = np.sort(ci)
                    # detect consecutive runs
                    diffs = np.diff(ci_sorted)
                    # form run boundaries where diff > 1
                    boundaries = np.where(diffs > 1)[0] + 1
                    runs = np.split(ci_sorted, boundaries)
                    # batch only if runs are identical-length blocks (heuristic for [n, t1..tk] layout)
                    block_len = None
                    ok = True
                    row_slices = []
                    for r in runs:
                        if r.size == 0:
                            continue
                        start, stop = int(r[0]), int(r[-1] + 1)
                        if block_len is None:
                            block_len = stop - start
                        elif block_len != (stop - start):
                            ok = False
                            break
                        row_slices.append(slice(start, stop))
                    use_batch = ok and (len(row_slices) > 0)
                if use_batch:
                    # Create a view of constrained blocks stacked as rows
                    rows = len(row_slices)
                    dim = row_slices[0].stop - row_slices[0].start
                    Yv = np.empty((rows, dim), dtype=candidate.dtype)
                    Cv = np.empty_like(Yv)
                    for i, sl in enumerate(row_slices):
                        Yv[i] = y[sl]
                        Cv[i] = candidate[sl]
                    # Call batched projection once
                    Pv = P.project_batch(Yv, Cv, rhok=lam, t=tcur, Fk_val=F_in)
                    # Write back into proj_val buffer
                    proj_val = candidate.copy()
                    for i, sl in enumerate(row_slices):
                        proj_val[sl] = Pv[i]
                    # For unconstrained entries, the identity projection applies (candidate unchanged)
                else:
                    proj_val = self._project(y, candidate, lam, tcur, F_in, prev)
            except Exception:
                # Fallback to scalar path on any unexpected condition
                proj_val = self._project(y, candidate, lam, tcur, F_in, prev)
            proj_z[:] = proj_val

            # F_buf = y - proj_z
            np.subtract(y, proj_z, out=F_buf)
            errF = np.linalg.norm(F_buf)
            if errF < self.tol:
                y[:] = proj_z
                F_y = func(y)
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
                        mode = 'cs' if getattr(self, 'autodiff_mode', 'numerical') == 'cs' else 'fd'
                        B0 = self._numerical_jacobian(func, y, sparse=False, mode=mode)
                    if sp.issparse(B0):
                        B0 = B0.toarray()
                    self._B = B0
                J_in = self._B
                used_broyden = True
            else:
                if self.jacobian is not None:
                    J_in = self.jacobian(y)
                else:
                    mode = 'cs' if getattr(self, 'autodiff_mode', 'numerical') == 'cs' else 'fd'
                    J_in = self._numerical_jacobian(func, y, sparse=sparse_active, mode=mode)

            # Eisenstat–Walker tol for GMRES if enabled
            rtol_dyn = self.gmres_tol
            if self.linear_solver == 'gmres' and self.linear_tol_strategy != 'fixed':
                eta = min(0.5, self.eisenstat_c * (errF ** self.eisenstat_exp))
                rtol_dyn = max(self.gmres_tol, eta)

            if sparse_active:
                # Use matrix-free linear operator to avoid forming (I - D) + lam * D @ J_in
                J_in = self._to_csr(J_in)
                Dproj = self._to_csr(self._tangent(candidate, y, lam, tcur, F_in, prev), n)
                rhs = -F_buf

                # Define matvec and rmatvec for LinearOperator: J = I - D + lam * D @ J_in
                def _matvec(v, _D=Dproj, _J=J_in, _lam=lam):
                    return (v - _D @ v) + _lam * (_D @ (_J @ v))

                def _rmatvec(w, _D=Dproj, _J=J_in, _lam=lam):
                    # J^T = I - D^T + lam * J_in^T @ D^T
                    return (w - _D.T @ w) + _lam * (_J.T @ (_D.T @ w))

                J = spla.LinearOperator((n, n), matvec=_matvec, rmatvec=_rmatvec)
                delta, ok = self._solve_linear_sparse(J, rhs, rtol=rtol_dyn, pattern_hint=None)
                if not ok:
                    return (y, F_in, errF, False, iteration)
            else:
                Dproj = self._tangent(candidate, y, lam, tcur, F_in, prev)
                if sp.issparse(Dproj):
                    Dproj = Dproj.toarray()
                if sp.issparse(J_in):
                    J_in = J_in.toarray()
                J = I - Dproj + lam * (Dproj @ J_in)
                try:
                    delta = np.linalg.solve(J, -F_buf)
                except np.linalg.LinAlgError:
                    return (y, F_in, errF, False, iteration)

            # Globalization (optional)
            if self.globalization == 'linesearch':
                phi0 = 0.5 * errF * errF
                grad_dir = -errF * errF
                alpha = 1.0
                backtracks = 0

                y_trial = y + alpha * delta
                phi_trial = self._phi(y_trial)
                while (phi_trial > phi0 + self.ls_c1 * alpha * grad_dir
                       and backtracks < self.max_backtracks
                       and alpha > self.ls_min_alpha):
                    alpha *= self.ls_beta
                    y_trial = y + alpha * delta
                    phi_trial = self._phi(y_trial)
                    backtracks += 1

                if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir:
                    if self.use_broyden and not sparse_active:
                        self._y_prev_broyden = y.copy()
                        self._F_prev_broyden = F_in.copy()
                    y = y_trial
                else:
                    # Steepest descent fallback
                    if sp.issparse(J):
                        grad_phi = J.T.dot(F_buf)
                    else:
                        grad_phi = J.T @ F_buf
                    nrm_g = np.linalg.norm(grad_phi)
                    if nrm_g == 0.0:
                        return (y, F_in, errF, False, iteration)
                    delta_g = -grad_phi
                    grad_dir = -nrm_g * nrm_g

                    alpha = 1.0
                    backtracks = 0
                    y_trial = y + alpha * delta_g
                    phi_trial = self._phi(y_trial)

                    while (phi_trial > phi0 + self.ls_c1 * alpha * grad_dir
                           and backtracks < self.max_backtracks
                           and alpha > self.ls_min_alpha):
                        alpha *= self.ls_beta
                        y_trial = y + alpha * delta_g
                        phi_trial = self._phi(y_trial)
                        backtracks += 1

                    if phi_trial <= phi0 + self.ls_c1 * alpha * grad_dir:
                        if self.use_broyden and not sparse_active:
                            self._y_prev_broyden = y.copy()
                            self._F_prev_broyden = F_in.copy()
                        y = y_trial
                    else:
                        return (y, F_in, errF, False, iteration)
            else:
                if self.use_broyden and not sparse_active:
                    self._y_prev_broyden = y.copy()
                    self._F_prev_broyden = F_in.copy()
                np.add(y, delta, out=y)

        # Out of iterations
        F_in = func(y)
        self.last_Fk_val = F_in
        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)
        errF = np.linalg.norm(y - self._project(y, y - lam * F_in, lam, tcur, F_in, prev))
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
        def _rel_block_l2(r, y, slices):
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
        def _expand_rho_to_vec(rho_in, n, slices):
            # scalar -> full vector
            if np.isscalar(rho_in):
                return float(rho_in) * np.ones(n, dtype=float)
            arr = np.asarray(rho_in, dtype=float)
            if arr.ndim == 0:
                return float(arr) * np.ones(n, dtype=float)
            if arr.size == n:
                return arr.astype(float, copy=False)
            if slices is not None and arr.size == len(slices):
                vec = np.empty(n, dtype=float)
                for v, s in zip(arr, slices):
                    vec[s] = float(v)
                return vec
            # Fallback: broadcast mean
            return float(np.mean(arr)) * np.ones(n, dtype=float)

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
        debug = bool(getattr(self, 'debug_vi', False))

        # Use per-block rho when component_slices is defined; otherwise scalar
        if self.component_slices is not None and len(self.component_slices) > 0:
            rho = _init_block_rho()  # shape (n_blocks,)
        else:
            rho = float(getattr(self, 'rho_last', self.rho0))
        # Sanitize initial rho and persist immediately so bad values don't leak into next solves
        rho = _sanitize_rho(rho, context="init")
        self.rho_last = rho
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
        rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
        # Guard against non-finite rho_vec
        if not np.all(np.isfinite(rho_vec)):
            rho = _sanitize_rho(rho, context="expand-init")
            rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
        candidate = yk - rho_vec * Fk_val
        if not np.all(np.isfinite(candidate)):
            # Reduce rho and try once more
            rho = _sanitize_rho(rho * 0.1 if np.isscalar(rho) else rho * 0.1, context="candidate-init")
            rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
            candidate = yk - rho_vec * Fk_val
        y_proj = self._project(yk, candidate, rho_vec, tcur, Fk_val, prev)

        # Block-wise L2 natural residual at yk
        r0 = (yk - y_proj)
        err = _rel_block_l2(r0, yk, self.component_slices)
        if not np.isfinite(err):
            # If projection resulted in non-finite error, reset rho to safe value and recompute once
            rho = _sanitize_rho(self.rho0 if np.isfinite(self.rho0) else 1.0, context="err-init-reset")
            rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
            candidate = yk - rho_vec * Fk_val
            y_proj = self._project(yk, candidate, rho_vec, tcur, Fk_val, prev)
            r0 = (yk - y_proj)
            err = _rel_block_l2(r0, yk, self.component_slices)
        if debug:
            print(f"[VI] k={k} err={err:.3e}")

        while err > self.tol and k < self.max_iter:
            # Project with current rho
            tcur = getattr(self, 'current_time', None)
            prev = getattr(self, 'prev_state', None)

            Fk_val = func(yk)
            self.last_Fk_val = Fk_val
            rho = _sanitize_rho(rho, context="iter-pre")
            rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
            candidate = yk - rho_vec * Fk_val
            if not np.all(np.isfinite(candidate)):
                rho = _sanitize_rho(rho * 0.5 if np.isscalar(rho) else rho * 0.5, context="iter-candidate")
                rho_vec = _expand_rho_to_vec(rho, len(yk), self.component_slices)
                candidate = yk - rho_vec * Fk_val
            yk1 = self._project(yk, candidate, rho_vec, tcur, Fk_val, prev)

            # Evaluate at new point and compute residual for error
            Fk_val_1 = func(yk1)
            rho = _sanitize_rho(rho, context="iter-post-proj")
            rho_vec1 = _expand_rho_to_vec(rho, len(yk1), self.component_slices)
            proj_candidate = yk1 - rho_vec1 * Fk_val_1
            proj_yk1 = self._project(yk1, proj_candidate, rho_vec1, tcur, Fk_val_1, prev)

            # Block-wise L2 natural residual at yk1
            r1 = (yk1 - proj_yk1)
            err = _rel_block_l2(r1, yk1, self.component_slices)
            if not np.isfinite(err):
                if debug:
                    print(f"[VI] non-finite err encountered; shrinking rho and retrying one step")
                rho = _sanitize_rho(rho * 0.5 if np.isscalar(rho) else rho * 0.5, context="iter-err-reset")
                rho_vec1 = _expand_rho_to_vec(rho, len(yk1), self.component_slices)
                proj_candidate = yk1 - rho_vec1 * Fk_val_1
                proj_yk1 = self._project(yk1, proj_candidate, rho_vec1, tcur, Fk_val_1, prev)
                r1 = (yk1 - proj_yk1)
                err = _rel_block_l2(r1, yk1, self.component_slices)

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
                            rho_vec_rb = _expand_rho_to_vec(rb, len(yk), self.component_slices)
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
                        rho_vec_rb = _expand_rho_to_vec(rb, len(yk), self.component_slices)
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
            k += 1

        success = (err <= self.tol)
        # Persist last rho for subsequent solves (both cached and default field)
        rho = _sanitize_rho(rho, context="final")
        self.rho0 = rho
        self.rho_last = rho
        F_final = func(yk)
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
    #     err = np.linalg.norm(yk - y_proj)

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
    
    # ---- VI stepsize update (unchanged math, but uses fast projection) ----
    def _update_rho(self, func, yk, rho):
        # guard against bad rho
        if not np.isscalar(rho) or not np.isfinite(rho) or rho <= 0:
            base = self.rho0 if (np.isscalar(self.rho0) and np.isfinite(self.rho0) and self.rho0 > 0) else 1.0
            rho = float(base)

        tcur = getattr(self, 'current_time', None)
        prev = getattr(self, 'prev_state', None)

        Fk_val = func(yk)
        self.last_Fk_val = Fk_val
        # Use per-index rho_vec consistently for projection
        slices = getattr(self, 'component_slices', None)
        if slices is not None and len(slices) > 0:
            rho_vec = np.empty_like(yk, dtype=float)
            # expand scalar rho to blocks then to vector
            rb = np.full(len(slices), float(rho), dtype=float)
            for v, s in zip(rb, slices):
                rho_vec[s] = v
        else:
            rho_vec = float(rho) * np.ones_like(yk, dtype=float)
        yk1 = self._project(yk, yk - rho_vec * Fk_val, rho_vec, tcur, Fk_val, prev)
        # Stuck detection for scalar path
        den = np.linalg.norm(yk1 - yk)
        stuck_thresh = self.stuck_eps_abs + self.stuck_eps_rel * (1.0 + np.linalg.norm(yk))
        if den >= stuck_thresh:
            rk = self._get_rk(func, yk1, yk, rho)
            while rk > self.L:
                rho = self.nu * rho
                # refresh rho_vec
                if slices is not None and len(slices) > 0:
                    rb = np.full(len(slices), float(rho), dtype=float)
                    for v, s in zip(rb, slices):
                        rho_vec[s] = v
                else:
                    rho_vec.fill(float(rho))
                yk1 = self._project(yk, yk - rho_vec * Fk_val, rho_vec, tcur, Fk_val, prev)
                den = np.linalg.norm(yk1 - yk)
                if den < stuck_thresh:
                    break
                rk = self._get_rk(func, yk1, yk, rho)
            if rk < self.Lmin:
                rho = (1.0 / self.nu) * rho
        # Clamp
        rho = float(np.clip(rho, self.rho_min, self.rho_max))
        return rho

    def _get_rk(self, func, yk1, yk, rho):
        num = rho * np.linalg.norm(func(yk1) - func(yk))
        den = np.linalg.norm(yk1 - yk)
        return 0.0 if den == 0.0 else (num / den)

    # ---------- Numerical Jacobian ----------
    def _numerical_jacobian(self, func, y, eps: float | None = None, sparse: bool | None = None, mode: str = 'fd'):
        n = len(y)
        J = np.empty((n, n), dtype=y.dtype)

        if (mode or 'fd').lower() == 'cs':
            try:
                h = 1e-30
                y_cs = y.astype(complex)
                for i in range(n):
                    y_cs_i = y_cs.copy()
                    y_cs_i[i] += 1j * h
                    Fi = func(y_cs_i)
                    J[:, i] = np.imag(Fi) / h
                use_sparse = self._sparse_active(n) if sparse is None else bool(sparse)
                return J if not use_sparse else sp.csr_matrix(J)
            except Exception:
                pass

        F0 = func(y)
        base = np.sqrt(np.finfo(float).eps) if eps is None else float(eps)
        for i in range(n):
            h = base * max(1.0, abs(y[i]))
            y_eps = y.copy()
            y_eps[i] += h
            F_eps = func(y_eps)
            J[:, i] = (F_eps - F0) / h
        use_sparse = self._sparse_active(n) if sparse is None else bool(sparse)
        return J if not use_sparse else sp.csr_matrix(J)

    # ---------- Sparse helpers ----------
    def _to_csr(self, A, n=None):
        if sp.issparse(A):
            return A.tocsr()
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

        if self.linear_solver == 'splu':
            try:
                lu = spla.splu(J.tocsc(), permc_spec=self.splu_permc_spec)
                x = lu.solve(b)
                return x, True
            except Exception:
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
