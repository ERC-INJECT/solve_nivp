from __future__ import annotations

import numpy as np
import math

class AdaptiveStepping:
    """
    Adaptive step-size controller with Richardson error estimation.

    Two operating modes controlled by `mode`:

    1. mode="classic":
       - Acceptance rule: accept step if E_curr <= 1.0 (standard LTE criterion).
       - Step-size update: classical P / PI controller
             h_next = h_curr * safety * E_curr^{-alpha_PI} * E_prev^{-beta_PI}
         (if E_prev is available); otherwise just proportional.
       - Growth/shrink clamped by h_up, h_down each step, and [h_min,h_max] globally.

    2. mode="ratio":
       - Uses Gustafsson / Söderlind-style digital filter on the step-size ratio.
       - Computes rho_prop from a multi-step PI-like recurrence involving
         c_n = 1/E_n, c_{n-1}, and the previous ratio.
       - Acceptance is based on the proposed ratio rho_prop compared to
         [r_min, r_max]. If rho_prop is too small, the step is *rejected*
         and we only shrink h; if within band or above band, the step is
         *accepted*, potentially clamping rho_prop to r_max.
       - Supports controller presets:
            'elementary', 'PI3040', 'PI3333', 'PI4020', 'H211PI', 'H211b'
         The H211b case uses b_param to tune beta1,beta2,alpha_ctrl=1/b_param.

    Common behavior:
    - We estimate local error via Richardson extrapolation from
         one full step of size h
         two half steps of size h/2
      and compute a scaled RMS relative error E_curr.
    - The lower-accuracy full step (y_full) is used only for error estimation;
      the accepted state is always the high-accuracy y_hi (two half steps).

    step(...) returns:
        (y_new, fk_new, h_next, E_curr, success, solver_error, iterations)

    Parameters
    ----------
    integrator : object
        Must implement .step(fun, t, y, h) -> (y_new, fk, solver_err, ok, iters)
        where `ok` is True iff nonlinear solve converged.

    component_slices : list[slice] or None
        Used to compute blockwise error if you only want subsets in the RMS.
        skip_error_indices can exclude some blocks.

    mode : {"classic","ratio"}
        Selects acceptance/update logic as described above.
    """

    def __init__(
        self,
        integrator,
        component_slices=None,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        h0: float | str | None = 1e-2,
        h_min: float = 1e-10,
        h_max: float = 1e3,
        h_up: float = 2.0,
        h_down: float = 0.6,
        method_order: int | None = None,
        # --- classic PI settings ---
        safety: float = 0.9,
        use_PI: bool = True,
        # --- ratio/digital-filter settings ---
        controller: str = "PI3040",
        b_param: float = 2.0,
        r_min: float = 0.8,
        r_max: float = 1.2,
        reject_reboot_thresh: int = 3,
        # --- global flags ---
        mode: str = "classic",   # "classic" or "ratio"
        verbose: bool = False,
        skip_error_indices=None,
        record_attempts: bool = False,
        # --- DAE-aware error weighting ---
        dae_var_weight: str = "auto",   # "auto", "exclude", "include"
        # --- Active-set filter for nonsmooth contact ---
        active_set_filter: bool = False,
    ) -> None:

        self.integrator = integrator
        self.component_slices = component_slices

        # ---- Per-DOF tolerance vectors (SUNDIALS / scipy convention) ----
        # Accept scalar, per-slice list/tuple, or full per-DOF array.
        # Internally we store 1-D numpy arrays (or a scalar float when the
        # system size is not yet known so that the first call lazily expands).
        self._atol_raw = atol
        self._rtol_raw = rtol
        self._atol_vec: np.ndarray | None = None   # lazily built on first use
        self._rtol_vec: np.ndarray | None = None

        # ---- Automatic h₀ (Hairer-Wanner) ----
        # When h0 is None or "auto", defer estimation to the first call to
        # step(), where we have access to fun, t₀, y₀.
        if h0 is None or (isinstance(h0, str) and h0.lower() == 'auto'):
            self._auto_h0 = True
            self.h = float(h_min)     # placeholder until estimated
        else:
            self._auto_h0 = False
            self.h = float(h0)

        # global hard clamps on h
        self.h_min = float(h_min)
        self.h_max = float(h_max)

        # ---- DAE-aware error weighting ----
        # "auto"   : detect algebraic DOFs from mass matrix on first use
        # "exclude": same as auto (kept for clarity)
        # "include": traditional behaviour, weight all DOFs equally
        self._dae_var_weight_mode = dae_var_weight.lower().strip()
        self._dae_mask: np.ndarray | None = None    # lazily built

        # ---- Active-set filter (nonsmooth contact) ----
        # When enabled, DOFs whose contact regime changed during a step
        # (e.g. stick↔slip, contact↔separation) are suppressed in the
        # error norm.  This prevents the embedded error estimator from
        # overreacting to discontinuous constraint-force jumps.
        self.active_set_filter = bool(active_set_filter)
        self._transition_mask: np.ndarray | None = None  # set per step

        # per-step up/down clamp factors (classic mode)
        self.h_up = float(h_up)
        self.h_down = float(h_down)

        self.verbose = bool(verbose)
        self.skip_error_indices = set(skip_error_indices or [])
        self.record_attempts = bool(record_attempts)

        # numerical order p of the base integrator
        self.p = int(method_order) if method_order is not None else self._infer_method_order(integrator)

        # ----- classic PI controller parameters -----
        self.safety = float(safety)
        self.use_PI = bool(use_PI)
        # Gustafsson-style exponents for PI controller in classic mode
        # alpha_PI, beta_PI correspond to E_curr^{-alpha} * E_prev^{-beta}
        self._alpha_PI = 0.7 / (self.p + 1.0)
        self._beta_PI  = 0.4 / (self.p + 1.0)

        # ----- ratio / digital-filter controller parameters -----
        self.controller = controller.lower()
        self.b_param = float(b_param)
        # ratio-band logic
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.reject_reboot_thresh = int(reject_reboot_thresh)
        self._reject_streak = 0  # consecutive rejects

        # controller memory shared by both modes
        # previous step's normalized error
        self._E_prev = None
        # previous step-size ratio rho_n = h_n / h_{n-1} (for ratio mode)
        self._rho_prev = 1.0

        # for scratch allocation in _scaled_error
        self._etol_buf = None
        self._err_buf = None

        # optional logging of attempted steps (accepted + rejected)
        self._attempt_t = None
        self._attempt_h = None
        self._attempt_accept = None
        self._attempt_error = None
        self._attempt_status = None
        self.reset_attempt_log()

        # --- Nonlinear failure recovery tracking ---
        # Detects persistent fail→succeed cycles (the "death spiral" where
        # every proposed h triggers a NL failure, the retry at h*h_down
        # succeeds with tiny error, the controller grows h, and the next
        # attempt fails again).
        self._nl_fail_recovery_count = 0   # consecutive fail→succeed pairs
        self._nl_success_no_fail = 0       # consecutive clean successes
        self._in_nl_recovery = False        # previous step() had NL failure
        self._consecutive_nl_fails = 0     # consecutive NL failures (no success between)

        # How many consecutive NL failures before a full solver reset
        # (destroy PETSc KSP, SPLU, ILU, all caches — force fresh Jacobian
        # and fresh factorisation on the next attempt).
        self._NL_RESCUE_THRESH = 5

        # operating mode
        m = mode.lower().strip()
        if m not in ("classic", "ratio"):
            raise ValueError("mode must be 'classic' or 'ratio'")
        self.mode = m

    # ------------------------------------------------------------------
    # Tolerance helpers  (per-DOF weighted-norm à la SUNDIALS / scipy)
    # ------------------------------------------------------------------

    @property
    def atol(self):
        """Return atol — scalar float or 1-D ndarray."""
        if self._atol_vec is not None:
            return self._atol_vec
        return self._atol_raw

    @atol.setter
    def atol(self, value):
        """Allow ``stepper.atol = X`` from outside (e.g. solve_ivp_ns tuning)."""
        self._atol_raw = value
        self._atol_vec = None          # invalidate cache so next call re-expands

    @property
    def rtol(self):
        if self._rtol_vec is not None:
            return self._rtol_vec
        return self._rtol_raw

    @rtol.setter
    def rtol(self, value):
        self._rtol_raw = value
        self._rtol_vec = None

    def _ensure_tol_vectors(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(atol_vec, rtol_vec)`` of length *n*.

        Lazily expands the raw tolerances provided at construction:

        * **scalar** → broadcast to length *n*.
        * **per-slice sequence** (length == ``len(component_slices)``) →
          expanded to per-DOF via the slice mapping.
        * **per-DOF array** (length == *n*) → used directly.
        """
        if self._atol_vec is not None and self._atol_vec.shape == (n,):
            return self._atol_vec, self._rtol_vec

        self._atol_vec = self._expand_tol(self._atol_raw, n, 'atol')
        self._rtol_vec = self._expand_tol(self._rtol_raw, n, 'rtol')
        return self._atol_vec, self._rtol_vec

    def _expand_tol(self, raw, n: int, name: str) -> np.ndarray:
        """Expand a raw tolerance value to a length-*n* 1-D float array."""
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            # scalar → broadcast
            return np.full(n, float(arr))
        if arr.shape == (n,):
            return arr.copy()
        # per-slice?
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

    # ------------------------------------------------------------------
    # Active-set filter helpers
    # ------------------------------------------------------------------

    def _get_projection(self):
        """Return the projection object from the integrator's solver, or None."""
        solver = getattr(self.integrator, 'solver', None)
        if solver is None:
            return None
        proj = getattr(solver, 'proj', None)
        return proj

    # ------------------------------------------------------------------
    # DAE-aware error weighting
    # ------------------------------------------------------------------

    def _ensure_dae_mask(self, n: int) -> np.ndarray:
        """Return a 1-D array of shape *(n,)* with per-DOF error weights.

        * **differential** DOF → weight 1.0
        * **algebraic** DOF (zero mass-matrix row) → weight 0.0

        The mask is built lazily on the first call by inspecting the mass
        matrix ``A`` stored on the integrator.  If ``dae_var_weight='include'``
        or no mass matrix is available, a uniform-ones vector is returned.
        """
        if self._dae_mask is not None and self._dae_mask.shape == (n,):
            return self._dae_mask

        if self._dae_var_weight_mode == 'include':
            self._dae_mask = np.ones(n, dtype=float)
            return self._dae_mask

        # Try to get the mass matrix from the integrator
        A = getattr(self.integrator, 'A', None)
        if A is None or getattr(self.integrator, 'use_identity', True):
            # No mass matrix (identity) → all differential
            self._dae_mask = np.ones(n, dtype=float)
            return self._dae_mask

        self._dae_mask = self._detect_algebraic_dofs(A, n)

        n_alg = int(np.sum(self._dae_mask == 0.0))
        if self.verbose and n_alg > 0:
            print(f"[adaptive/DAE] detected {n_alg} algebraic DOFs "
                  f"(of {n} total) — excluded from error norm")

        return self._dae_mask

    @staticmethod
    def _detect_algebraic_dofs(A, n: int) -> np.ndarray:
        """Build a 0/1 mask from mass matrix *A*.

        A row whose absolute-value norm is below a small tolerance is
        classified as algebraic (weight = 0).  Works for both dense and
        sparse matrices.

        Uses fully vectorised operations — no Python row loop.
        """
        import scipy.sparse as sp
        mask = np.ones(n, dtype=float)
        m = min(n, A.shape[0])

        if sp.issparse(A):
            A_csr = sp.csr_matrix(A)
            # Vectorised: compute max |value| per row via abs() then
            # group-max using np.maximum.reduceat on the CSR data array.
            if A_csr.nnz == 0:
                mask[:m] = 0.0
            else:
                abs_data = np.abs(A_csr.data)
                # Rows that have at least one entry
                row_nnz = np.diff(A_csr.indptr[:m + 1])
                nonempty = row_nnz > 0

                # reduceat only over non-empty rows (avoids 0-length segments)
                starts = A_csr.indptr[:m][nonempty]
                row_max = np.maximum.reduceat(abs_data, starts)
                # Only keep the first element of each segment
                # (reduceat already does this for us since segments are contiguous)
                alg_nonempty = row_max < 1e-14

                # Empty rows are always algebraic
                alg_mask = np.ones(m, dtype=bool)
                alg_mask[nonempty] = alg_nonempty
                # Rows with zero nnz stay True (algebraic)

                mask[:m] = np.where(alg_mask, 0.0, 1.0)
        else:
            A_arr = np.asarray(A)
            row_max = np.max(np.abs(A_arr[:m, :]), axis=1)
            mask[:m] = np.where(row_max < 1e-14, 0.0, 1.0)

        return mask

    # ------------------------------------------------------------------
    # Hairer-Wanner automatic h₀ estimation
    # ------------------------------------------------------------------

    def _estimate_h0_hairer(self, fun, t0: float, y0: np.ndarray) -> float:
        """Estimate a good initial step size using the Hairer-Wanner algorithm.

        Reference: Hairer, Nørsett & Wanner, *Solving Ordinary Differential
        Equations I*, § II.4, algorithm on p. 169.

        Steps
        -----
        1. Compute weighted norms d₀ = ‖y₀‖ and d₁ = ‖f₀‖.
        2. First guess  h₀ = 0.01 · d₀ / d₁   (or 10⁻⁶ if too small).
        3. Euler step   y₁ = y₀ + h₀ · f₀.
        4. Second derivative estimate  d₂ = ‖f(t₀+h₀, y₁) - f₀‖ / h₀.
        5. Refine  h₁ = (0.01 / max(d₁,d₂))^{1/(p+1)}.
        6. Return  min(100·h₀, h₁), clamped to [h_min, h_max].
        """
        p = self.p
        n = y0.size
        atol_v, rtol_v = self._ensure_tol_vectors(n)

        def _wnorm(v):
            """Weighted RMS norm with tol = atol + rtol*|y0|."""
            sc = atol_v + rtol_v * np.abs(y0)
            return math.sqrt(np.mean((v / sc) ** 2))

        f0 = np.asarray(fun(t0, y0), dtype=float).ravel()

        d0 = _wnorm(y0)
        d1 = _wnorm(f0)

        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        # Explicit Euler probe
        y1 = y0 + h0 * f0
        f1 = np.asarray(fun(t0 + h0, y1), dtype=float).ravel()

        d2 = _wnorm(f1 - f0) / h0

        if max(d1, d2) <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1.0 / (p + 1))

        h_est = min(100.0 * h0, h1)
        h_est = min(self.h_max, max(self.h_min, h_est))

        if self.verbose:
            print(f"[adaptive/auto_h0] d0={d0:.3e}, d1={d1:.3e}, d2={d2:.3e}, "
                  f"h0_probe={h0:.3e}, h1={h1:.3e} → h_est={h_est:.3e}")

        return h_est

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_method_order(self, integrator) -> int:
        p = getattr(integrator, 'order', None)
        if isinstance(p, (int, float)) and p > 0:
            return int(p)

        name = integrator.__class__.__name__.lower()
        if 'trapezoidal' in name or 'embeddedbetr' in name:
            return 2
        if 'thetamethod' in name:
            theta = getattr(integrator, 'theta', None)
            return 2 if (theta is not None and abs(theta - 0.5) < 1e-12) else 1
        # default conservative guess
        return 1

    def _filter_coeffs(self):
        """
        Map controller name -> (beta1, beta2, alpha_ctrl, k)
        used in ratio mode for the digital filter.

        k = p+1, assuming LTE ~ h^(p+1).
        """
        k = self.p + 1.0
        name = self.controller

        if name == "elementary":
            beta1, beta2, alpha_ctrl = 1.0, 0.0, 0.0
        elif name == "pi3040":
            beta1, beta2, alpha_ctrl = 7.0/10.0, -4.0/10.0, 0.0
        elif name == "pi3333":
            beta1, beta2, alpha_ctrl = 2.0/3.0, -1.0/3.0, 0.0
        elif name == "pi4020":
            beta1, beta2, alpha_ctrl = 3.0/5.0, -1.0/5.0, 0.0
        elif name == "h211pi":
            beta1, beta2, alpha_ctrl = 1.0/6.0, 1.0/6.0, 0.0
        elif name == "h211b":
            # tunable 1/b scheme
            b = self.b_param
            beta1 = 1.0 / b
            beta2 = 1.0 / b
            alpha_ctrl = 1.0 / b
        else:
            raise ValueError(f"Unknown controller '{self.controller}'")

        return beta1, beta2, alpha_ctrl, k

    def _scaled_error(self, y_prev, y_lo, y_hi) -> float:
        """
        Compute normalized error E via Richardson step-doubling:
            raw_err = (y_lo - y_hi) / (2^p - 1)
        scale each component by atol_i + rtol_i * max(|.|),
        then RMS over included components.

        E <= 1   => "good enough" in classic sense.

        Tolerances may be per-DOF vectors (SUNDIALS convention).

        When DAE-aware weighting is active, algebraic DOFs (zero mass-
        matrix rows) are excluded from the norm (IDA convention).
        """
        denom = max(1e-14, (2.0 ** self.p) - 1.0)
        accum = 0.0
        count = 0

        n = y_hi.size
        atol_v, rtol_v = self._ensure_tol_vectors(n)
        dae_w = self._ensure_dae_mask(n)

        if self.component_slices is None:
            if self._err_buf is None or self._err_buf.shape != y_hi.shape:
                self._err_buf = np.empty_like(y_hi)
            if self._etol_buf is None or self._etol_buf.shape != y_hi.shape:
                self._etol_buf = np.empty_like(y_hi)

            # raw_err
            np.subtract(y_lo, y_hi, out=self._err_buf)
            self._err_buf /= denom

            # tol scaling (per-DOF)
            np.maximum(np.abs(y_lo), np.abs(y_hi), out=self._etol_buf)
            self._etol_buf *= rtol_v
            self._etol_buf += atol_v

            # scaled error per component
            np.divide(self._err_buf, self._etol_buf, out=self._err_buf)

            # Apply DAE mask — zero out algebraic DOFs
            self._err_buf *= dae_w

            # Apply active-set transition mask (suppress regime-changing DOFs)
            if self._transition_mask is not None:
                self._err_buf *= self._transition_mask

            accum = float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
            # Count only differential DOFs in the denominator
            w_eff = dae_w if self._transition_mask is None else dae_w * self._transition_mask
            count = int(np.sum(w_eff > 0.0))
        else:
            for i, sl in enumerate(self.component_slices):
                if i in self.skip_error_indices:
                    continue
                lo = y_lo[sl]
                hi = y_hi[sl]
                a_blk = atol_v[sl]
                r_blk = rtol_v[sl]
                w_blk = dae_w[sl]

                # (re)allocate to match current block
                if self._err_buf is None or self._err_buf.shape != hi.shape:
                    self._err_buf = np.empty_like(hi)
                if self._etol_buf is None or self._etol_buf.shape != hi.shape:
                    self._etol_buf = np.empty_like(hi)

                # raw_err block
                np.subtract(lo, hi, out=self._err_buf)
                self._err_buf /= denom

                np.maximum(np.abs(lo), np.abs(hi), out=self._etol_buf)
                self._etol_buf *= r_blk
                self._etol_buf += a_blk

                np.divide(self._err_buf, self._etol_buf, out=self._err_buf)

                # Apply DAE mask
                self._err_buf *= w_blk

                # Apply active-set transition mask (suppress regime-changing DOFs)
                if self._transition_mask is not None:
                    self._err_buf *= self._transition_mask[sl]

                w_eff_blk = w_blk if self._transition_mask is None else w_blk * self._transition_mask[sl]
                accum += float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
                count += int(np.sum(w_eff_blk > 0.0))

        return 0.0 if count == 0 else math.sqrt(accum / count)

    def _scaled_error_embedded(self, y_new, err_vec) -> float:
        """Compute normalised RMS error from an embedded error vector.

        Unlike :meth:`_scaled_error` (which builds the raw error from two
        Richardson solutions), this method takes the element-wise error
        estimate directly from the integrator (e.g. SDIRK2) and scales it
        by the same tolerance formula::

            tol_i = atol_i + rtol_i * |y_new_i|
            E     = rms( err_i / tol_i )

        Tolerances may be per-DOF vectors (SUNDIALS convention).
        When DAE-aware weighting is active, algebraic DOFs are excluded.

        Returns ``E`` with the same semantics: ``E <= 1`` means acceptable.
        """
        accum = 0.0
        count = 0

        n = y_new.size
        atol_v, rtol_v = self._ensure_tol_vectors(n)
        dae_w = self._ensure_dae_mask(n)

        if self.component_slices is None:
            if self._err_buf is None or self._err_buf.shape != y_new.shape:
                self._err_buf = np.empty_like(y_new)
            if self._etol_buf is None or self._etol_buf.shape != y_new.shape:
                self._etol_buf = np.empty_like(y_new)

            # tol scaling: atol_i + rtol_i * |y_new_i|
            np.abs(y_new, out=self._etol_buf)
            self._etol_buf *= rtol_v
            self._etol_buf += atol_v

            # scaled error
            np.divide(err_vec, self._etol_buf, out=self._err_buf)

            # Apply DAE mask — zero out algebraic DOFs
            self._err_buf *= dae_w

            # Apply active-set transition mask (suppress regime-changing DOFs)
            if self._transition_mask is not None:
                self._err_buf *= self._transition_mask

            accum = float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
            w_eff = dae_w if self._transition_mask is None else dae_w * self._transition_mask
            count = int(np.sum(w_eff > 0.0))
        else:
            for i, sl in enumerate(self.component_slices):
                if i in self.skip_error_indices:
                    continue
                yn_blk = y_new[sl]
                er_blk = err_vec[sl]
                a_blk = atol_v[sl]
                r_blk = rtol_v[sl]
                w_blk = dae_w[sl]

                if self._err_buf is None or self._err_buf.shape != yn_blk.shape:
                    self._err_buf = np.empty_like(yn_blk)
                if self._etol_buf is None or self._etol_buf.shape != yn_blk.shape:
                    self._etol_buf = np.empty_like(yn_blk)

                np.abs(yn_blk, out=self._etol_buf)
                self._etol_buf *= r_blk
                self._etol_buf += a_blk

                np.divide(er_blk, self._etol_buf, out=self._err_buf)

                # Apply DAE mask
                self._err_buf *= w_blk

                # Apply active-set transition mask (suppress regime-changing DOFs)
                if self._transition_mask is not None:
                    self._err_buf *= self._transition_mask[sl]

                w_eff_blk = w_blk if self._transition_mask is None else w_blk * self._transition_mask[sl]
                accum += float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
                count += int(np.sum(w_eff_blk > 0.0))

        return 0.0 if count == 0 else math.sqrt(accum / count)

    # ------------------------------------------------------------------
    # Attempt logging helpers (optional)
    # ------------------------------------------------------------------
    def reset_attempt_log(self):
        """Clear stored attempt information if logging is enabled."""
        if not self.record_attempts:
            self._attempt_t = None
            self._attempt_h = None
            self._attempt_accept = None
            self._attempt_error = None
            self._attempt_status = None
            return

        self._attempt_t = []
        self._attempt_h = []
        self._attempt_accept = []
        self._attempt_error = []
        self._attempt_status = []

    def _finalize_return(self, t, h_attempt, y_out, fk_out, h_next,
                         E_curr, success, solver_error, iterations, status):
        if self.record_attempts:
            if self._attempt_t is None:
                self.reset_attempt_log()
            self._attempt_t.append(float(t))
            self._attempt_h.append(float(h_attempt))
            self._attempt_accept.append(bool(success))
            if E_curr is None:
                self._attempt_error.append(np.nan)
            else:
                try:
                    self._attempt_error.append(float(E_curr))
                except Exception:
                    self._attempt_error.append(np.nan)
            self._attempt_status.append(status)

        return y_out, fk_out, h_next, E_curr, success, solver_error, iterations

    def get_attempt_log(self):
        """Return recorded attempt arrays or None if logging disabled."""
        if not self.record_attempts or self._attempt_t is None:
            return None

        return {
            "t": np.asarray(self._attempt_t, dtype=float),
            "dt": np.asarray(self._attempt_h, dtype=float),
            "accepted": np.asarray(self._attempt_accept, dtype=bool),
            "error": np.asarray(self._attempt_error, dtype=float),
            "status": np.asarray(self._attempt_status, dtype=object),
        }

    # ------------------------------------------------------------------
    # Nonlinear-failure recovery cap
    # ------------------------------------------------------------------
    _NL_PERSIST_THRESH = 3     # fail→succeed pairs before capping kicks in
    _NL_RECOVERY_GROWTH = 1.05 # max per-step growth factor under the cap
    _NL_RELAX_SUCCESSES = 3    # clean successes needed to relax one streak level

    def _apply_nl_recovery_cap(self, h_used: float, h_next: float) -> float:
        """Cap *h_next* when a persistent nonlinear-failure pattern is detected.

        **Problem**: when the nonlinear solver repeatedly fails at "normal"
        step sizes (due to poor conditioning on coarse meshes, extreme
        stiffness, etc.), the adaptive loop enters a death spiral:

        1. Controller proposes ``h``.
        2. NL solver fails → retry at ``h * h_down``.
        3. Retry succeeds with tiny error → PI proposes ``h_next ≈ h_up * h * h_down``.
        4. ``h_up * h_down ≈ 1`` so ``h_next ≈ h`` → fails again → repeat.

        Because growth never overcomes the shrink, ``h`` monotonically decays
        to ``h_min`` and the integration terminates prematurely.

        **Fix**: track consecutive fail→succeed pairs.  After
        ``_NL_PERSIST_THRESH`` such pairs, limit growth to a conservative
        factor (``_NL_RECOVERY_GROWTH``, default 5 %) per step.  This
        stabilises ``h`` near the solver's actual convergence threshold
        instead of letting it spiral to zero.

        The cap is gradually relaxed after consecutive successes *without*
        any preceding NL failure  (``_NL_RELAX_SUCCESSES``).

        Called from both the embedded-error and Richardson acceptance paths.

        Parameters
        ----------
        h_used : float
            The step size that was actually *accepted* (after any NL-failure
            shrink).
        h_next : float
            The step size proposed by the PI / digital-filter controller.

        Returns
        -------
        h_next : float
            Possibly capped step size.
        """
        if self._in_nl_recovery:
            # We just successfully recovered from a NL failure.
            self._nl_fail_recovery_count += 1
            self._in_nl_recovery = False
            self._nl_success_no_fail = 0
        else:
            # Clean success (no preceding NL failure).
            self._nl_success_no_fail += 1
            if self._nl_success_no_fail >= self._NL_RELAX_SUCCESSES:
                self._nl_fail_recovery_count = max(
                    0, self._nl_fail_recovery_count - 1
                )
                self._nl_success_no_fail = 0  # reset after decrement

        if self._nl_fail_recovery_count >= self._NL_PERSIST_THRESH:
            h_capped = h_used * self._NL_RECOVERY_GROWTH
            if h_next > h_capped:
                if self.verbose:
                    print(
                        f"[adaptive] NL-recovery cap: h_next {h_next:.3e} "
                        f"-> {h_capped:.3e} "
                        f"(streak={self._nl_fail_recovery_count})"
                    )
                h_next = h_capped

        return h_next

    # ---------------------------
    # CLASSIC controller proposal
    # ---------------------------
    def _propose_h_classic(self, h_curr: float, E_curr: float) -> float:
        """
        Classic next-step proposal (P or PI), with acceptance test
        E_curr <= 1 handled outside this function.

        Matches your second code block.
        """
        tiny = 1e-16

        if (not np.isfinite(E_curr)) or (E_curr <= 0.0):
            # weird / zero error => allow growth up to h_up
            g = self.h_up
        else:
            if self.use_PI and self._E_prev is not None and self._E_prev > tiny:
                # PI controller
                g = (self.safety
                     * (E_curr      ** (-self._alpha_PI))
                     * (self._E_prev ** (-self._beta_PI)))
            else:
                # proportional-only
                g = self.safety * (E_curr ** (-1.0 / (self.p + 1.0)))

            # clamp growth/shrink per step
            g = min(self.h_up, max(self.h_down, g))

        h_next = g * h_curr
        # clamp globally
        h_next = min(self.h_max, max(self.h_min, h_next))
        return h_next

    # ---------------------------
    # Error-predictive rejection shrink
    # ---------------------------
    _REJECT_FLOOR = 0.1   # minimum ratio h_new/h on rejection (SUNDIALS ≈ 0.25, scipy ≈ 0.2)

    def _rejection_shrink(self, h: float, E_curr: float) -> float:
        """Compute a reduced step size after an *error* rejection.

        Instead of the blind ``h * h_down`` fall-back, use the standard
        optimal-control formula

            h_new = h · clamp(safety · E^{−1/(p+1)},  _REJECT_FLOOR, 1)

        This mirrors SUNDIALS CVODE and scipy.integrate, which apply the
        elementary controller even on rejection, but with a more aggressive
        minimum ratio than the acceptance-path ``h_down`` (typically 0.1–0.25
        instead of 0.6).  For very large errors the step can therefore shrink
        by up to 10× in a single rejection, avoiding a long cascade of blind
        0.6× halvings.
        """
        if E_curr > 0.0 and np.isfinite(E_curr):
            g = self.safety * (E_curr ** (-1.0 / (self.p + 1.0)))
            g = max(self._REJECT_FLOOR, min(g, 1.0))   # clamp to [floor, 1]
        else:
            g = self.h_down          # non-finite / zero error → blind fall-back
        h_new = g * h
        return max(self.h_min, h_new)

    # ---------------------------
    # RATIO / DIGITAL-FILTER proposal
    # ---------------------------
    def _propose_h_ratio(self, h_curr: float, E_curr: float):
        """
        Core Gustafsson/Söderlind digital filter update for step-size ratio.

        Returns:
            h_prop  = raw proposed next step (unclamped to band)
            rho_prop = h_prop / h_curr
        """
        beta1, beta2, alpha_ctrl, k = self._filter_coeffs()
        tiny = 1e-16

        # c_n = 1/E_n
        if (not np.isfinite(E_curr)) or (E_curr <= tiny):
            c_n = 1.0 / tiny
        else:
            c_n = 1.0 / E_curr

        if (self._E_prev is not None) and np.isfinite(self._E_prev) and self._E_prev > tiny:
            c_nm1 = 1.0 / self._E_prev
        else:
            c_nm1 = 1.0  # neutral

        rho_prev = self._rho_prev
        if (rho_prev is None) or (not np.isfinite(rho_prev)) or (rho_prev <= tiny):
            rho_prev = 1.0

        # digital filter for ratio
        rho_next = ((c_n   ** (beta1 / k))
                    * (c_nm1 ** (beta2 / k))
                    * (rho_prev ** (-alpha_ctrl)))

        h_prop = h_curr * rho_next
        rho_prop = rho_next  # = h_prop / h_curr if h_curr>0
        return h_prop, rho_prop

    def _apply_ratio_acceptance(self, t, h_curr, h_prop, rho_prop, E_curr, fk_full, it_full, solver_err):
        """
        Implements the ratio-band accept/reject logic from your first code block.
        Returns the standard step(...) tuple.
        """

        # If the proposed ratio is too *small*, treat as reject.
        if rho_prop < self.r_min:
            # reject
            if self.verbose:
                print(f"[adaptive] REJECT @ t={t:.6g}: r={rho_prop:.3f} < r_min={self.r_min}, "
                      f"E={E_curr:.3e}, h={h_curr:.3e}")

            # mild shrink instead of following rho_prop literally
            h_next = self.r_min * h_curr
            h_next = min(self.h_max, max(self.h_min, h_next))

            # track rejects
            self._reject_streak += 1
            if self._reject_streak >= self.reject_reboot_thresh:
                if self.verbose:
                    print(f"[adaptive] reboot PI after {self._reject_streak} rejects")
                self._E_prev   = None
                self._rho_prev = 1.0
                self._reject_streak = 0

            # no memory update of E_prev, rho_prev on reject
            return (
                # state does NOT advance on reject
                None,          # y_new  (caller will replace with y)
                fk_full,       # fk_new (from the failed full step)
                h_next,        # h_new proposal for retry
                E_curr,        # E_curr is still informative
                False,         # success=False
                solver_err,    # solver_error
                it_full        # iterations
            )

        # ACCEPT
        if rho_prop > self.r_max:
            rho_actual = self.r_max  # clamp big jump
            if self.verbose:
                print(f"[adaptive] ACCEPT(clamp) @ t={t:.6g}: r={rho_prop:.3f} -> {rho_actual:.3f}, "
                      f"E={E_curr:.3e}")
        else:
            rho_actual = rho_prop
            if self.verbose:
                print(f"[adaptive] ACCEPT @ t={t:.6g}: r={rho_prop:.3f}, E={E_curr:.3e}")

        h_next = rho_actual * h_curr
        h_next = min(self.h_max, max(self.h_min, h_next))

        # reset reject streak on success
        self._reject_streak = 0
        # update memory
        self._E_prev = E_curr
        self._rho_prev = rho_actual if (np.isfinite(rho_actual) and rho_actual > 0) else 1.0

        # We'll return a marker and let caller supply y_hi etc. after calling this.
        return (
            "ACCEPT",
            None,        # fk_new placeholder, caller will override
            h_next,
            E_curr,
            True,
            solver_err,
            it_full
        )

    # ------------------------------------------------------------------
    # Public step()
    # ------------------------------------------------------------------
    def step(self, fun, t, y, h):
        """
        Attempt one adaptive step of size h starting from (t,y).

        If ``h0`` was set to ``None`` or ``'auto'`` at construction time
        the first call triggers the Hairer-Wanner initial step-size
        estimator and *h* is replaced by the computed value.

        If the integrator exposes ``has_embedded_error = True``, a single
        call to ``integrator.step()`` is made and the element-wise error
        vector it returns is used directly for step-size control—no
        Richardson extrapolation (step-doubling) is performed.  This cuts
        the number of implicit solves from 3× to 1× per attempt.

        Returns
        -------
        (y_new, fk_new, h_next, E_curr, success, solver_error, iterations)
        where:
          - y_new is the *accepted* state (high-accuracy y_hi if accepted),
            or y (unchanged) if rejected.
          - fk_new is the residual from the accepted solve (y_hi),
            or from the single full step if rejected.
          - h_next is the next step size suggestion.
          - E_curr is the normalized local error estimate for this attempt.
          - success is True if we accept and advance, else False.
        """

        # ==============================================================
        # Auto h₀: Hairer-Wanner estimation on the very first call
        # ==============================================================
        if self._auto_h0:
            h = self._estimate_h0_hairer(fun, t, y)
            self._auto_h0 = False          # only once

        # ==============================================================
        # Fast path: integrator provides an embedded error estimate
        # (e.g. SDIRK2).  Only ONE call to integrator.step() needed.
        # ==============================================================
        if getattr(self.integrator, 'has_embedded_error', False):
            return self._step_embedded(fun, t, y, h)

        # ==============================================================
        # Default path: Richardson extrapolation (step-doubling)
        # ==============================================================
        return self._step_richardson(fun, t, y, h)

    # ------------------------------------------------------------------
    # Embedded-error path  (SDIRK2, etc.)
    # ------------------------------------------------------------------
    def _step_embedded(self, fun, t, y, h):
        """Single-step adaptive attempt using the integrator's own error."""
        # ── Active-set filter: snapshot regime before step ──
        regime_before = None
        if self.active_set_filter:
            proj = self._get_projection()
            if proj is not None:
                regime_before = proj.regime_snapshot()

        try:
            y_new, fk_new, err_vec, ok, it = self.integrator.step(fun, t, y, h)
        except RuntimeError as e:
            if self.verbose:
                print(f"[adaptive/emb] runtime error @ t={t:.6g}: {e}")
            self._transition_mask = None
            h_retry = max(self.h_min, self.h_down * h)
            return self._finalize_return(
                t, h, y, None, h_retry, np.inf, False, np.inf, 0,
                "embedded_step_runtime_error",
            )

        # ── Active-set filter: build transition mask ──
        self._transition_mask = None
        if self.active_set_filter and regime_before is not None:
            proj = self._get_projection()
            if proj is not None:
                mask = proj.regime_changed_mask(regime_before, y.shape[0])
                if mask is not None:
                    self._transition_mask = mask
                    if self.verbose:
                        n_supp = int(np.sum(mask == 0.0))
                        if n_supp > 0:
                            print(f"[adaptive/asf] suppressing {n_supp} DOFs "
                                  f"at t={t:.6g} due to regime transition")

        if not ok:
            self._consecutive_nl_fails += 1
            if self.verbose:
                print(f"[adaptive/emb] nonlinear fail @ t={t:.6g}: shrinking")
            # Mark that we are in NL-failure recovery (for death-spiral cap)
            self._in_nl_recovery = True
            self._nl_success_no_fail = 0
            # Invalidate the solver's cached LU factorisation so that the
            # retry with a smaller h (and therefore a different iteration
            # matrix M/(γh_new) − J) does not blindly reuse a stale
            # factorisation from the failed attempt.
            solver = getattr(self.integrator, 'solver', None)
            if solver is not None:
                solver._lu = None
                solver._lu_shape = None
                solver._J_cross_call = None
                solver._petsc_needs_matrix_update = True
                # After several consecutive failures, do a full solver reset:
                # destroy PETSc KSP (stale MUMPS factorisation), ILU,
                # equilibration cache — everything.  This forces a completely
                # fresh Jacobian + factorisation on the next attempt.
                if self._consecutive_nl_fails >= self._NL_RESCUE_THRESH:
                    if hasattr(solver, 'invalidate_all_caches'):
                        solver.invalidate_all_caches()
                    if self.verbose:
                        print(
                            f"[adaptive] full solver reset after "
                            f"{self._consecutive_nl_fails} consecutive NL failures"
                        )
            # Also clear the integrator's step-size LU cache
            if hasattr(self.integrator, '_lu_h_cache'):
                self.integrator._lu_h_cache.clear()
            h_retry = max(self.h_min, self.h_down * h)
            return self._finalize_return(
                t, h, y, None, h_retry, np.inf, False, np.inf, it,
                "embedded_step_nonlinear_fail",
            )

        # Normalised RMS error from embedded estimate
        # NL solve succeeded — reset consecutive failure counter
        self._consecutive_nl_fails = 0
        solver_err = 0.0  # no separate "solver residual" to report
        if isinstance(err_vec, np.ndarray) and err_vec.shape == y_new.shape:
            E_curr = self._scaled_error_embedded(y_new, err_vec)
        else:
            # Fallback: treat scalar err_vec as pre-scaled
            try:
                E_curr = float(err_vec)
            except (TypeError, ValueError):
                E_curr = 0.0

        # ---- controller logic (identical to Richardson path) ----------
        if self.mode == "classic":
            success = (E_curr <= 1.0)
            if success:
                h_next = self._propose_h_classic(h, E_curr)
                # Apply NL-recovery cap to prevent death spiral
                h_next = self._apply_nl_recovery_cap(h, h_next)
            else:
                h_reject = self._rejection_shrink(h, E_curr)
                if self.verbose:
                    print(f"[adaptive/emb] reject @ t={t:.6g}, E={E_curr:.3e}, "
                          f"h={h:.3e} -> h_next={h_reject:.3e}")
                return self._finalize_return(
                    t, h, y, fk_new, h_reject, E_curr, False,
                    solver_err, it, "embedded_classic_reject",
                )

            if self.verbose:
                print(f"[adaptive/emb] accept @ t={t:.6g} -> t+{h:.3e}, "
                      f"E={E_curr:.3e}, h_next={h_next:.3e}")

            self._E_prev = E_curr
            self._rho_prev = (h_next / h) if h > 0.0 else 1.0

            return self._finalize_return(
                t, h, y_new, fk_new, h_next, E_curr, True,
                solver_err, it, "embedded_classic_accept",
            )

        else:
            # ratio / digital-filter mode
            h_prop, rho_prop = self._propose_h_ratio(h, E_curr)
            decision, _, h_next, E_out, success, solver_error_out, it_used = \
                self._apply_ratio_acceptance(
                    t, h, h_prop, rho_prop, E_curr, fk_new, it, solver_err,
                )

            if not success:
                return self._finalize_return(
                    t, h, y, fk_new, h_next, E_out, False,
                    solver_error_out, it_used, "embedded_ratio_reject",
                )

            # Apply NL-recovery cap to prevent death spiral
            h_next = self._apply_nl_recovery_cap(h, h_next)

            return self._finalize_return(
                t, h, y_new, fk_new, h_next, E_out, True,
                solver_error_out, it_used, "embedded_ratio_accept",
            )

    # ------------------------------------------------------------------
    # Richardson extrapolation path  (BackwardEuler, TR, Composite, …)
    # ------------------------------------------------------------------
    def _step_richardson(self, fun, t, y, h):

        # ── Active-set filter: snapshot regime before step ──
        regime_before = None
        if self.active_set_filter:
            proj = self._get_projection()
            if proj is not None:
                regime_before = proj.regime_snapshot()

        # ------------------------------------------------------
        # 1. Take one full step of size h
        # ------------------------------------------------------
        try:
            y_full, fk_full, solver_err, ok_full, it_full = \
                self.integrator.step(fun, t, y, h)
        except RuntimeError as e:
            if self.verbose:
                print(f"[adaptive] error in full step @ t={t:.6g}: {e}")
            # catastrophic failure: we shrink and do not advance
            h_retry = max(self.h_min, self.h_down * h)
            return self._finalize_return(
                t,
                h,
                y,
                None,
                h_retry,
                np.inf,
                False,
                np.inf,
                0,
                "full_step_runtime_error",
            )

        if not ok_full:
            self._consecutive_nl_fails += 1
            if self.verbose:
                print(f"[adaptive] nonlinear fail @ t={t:.6g}: shrinking")
            # Mark NL-failure recovery (for death-spiral cap)
            self._in_nl_recovery = True
            self._nl_success_no_fail = 0
            # Invalidate cached LU — h is about to change
            solver = getattr(self.integrator, 'solver', None)
            if solver is not None:
                solver._lu = None
                solver._lu_shape = None
                solver._J_cross_call = None
                solver._petsc_needs_matrix_update = True
                if self._consecutive_nl_fails >= self._NL_RESCUE_THRESH:
                    if hasattr(solver, 'invalidate_all_caches'):
                        solver.invalidate_all_caches()
                    if self.verbose:
                        print(
                            f"[adaptive] full solver reset after "
                            f"{self._consecutive_nl_fails} consecutive NL failures"
                        )
            # Also clear the integrator's step-size LU cache
            if hasattr(self.integrator, '_lu_h_cache'):
                self.integrator._lu_h_cache.clear()
            h_retry = max(self.h_min, self.h_down * h)
            return self._finalize_return(
                t,
                h,
                y,
                None,
                h_retry,
                np.inf,
                False,
                solver_err,
                it_full,
                "full_step_nonlinear_fail",
            )

        # ------------------------------------------------------
        # 2. Take two half steps (h/2 each) for the higher-accuracy solution
        # ------------------------------------------------------
        h2 = 0.5 * h
        try:
            y_half, _, _, ok_h1, _ = self.integrator.step(fun, t, y, h2)
            if not ok_h1:
                self._consecutive_nl_fails += 1
                if self.verbose:
                    print(f"[adaptive] half-step fail #1 @ t={t:.6g}")
                self._in_nl_recovery = True
                self._nl_success_no_fail = 0
                h_retry = max(self.h_min, self.h_down * h)
                return self._finalize_return(
                    t,
                    h,
                    y,
                    None,
                    h_retry,
                    np.inf,
                    False,
                    np.inf,
                    0,
                    "half_step_fail_1",
                )

            y_hi, fk_hi, _, ok_h2, _ = self.integrator.step(fun, t + h2, y_half, h2)
            if not ok_h2:
                self._consecutive_nl_fails += 1
                if self.verbose:
                    print(f"[adaptive] half-step fail #2 @ t={t:.6g}")
                self._in_nl_recovery = True
                self._nl_success_no_fail = 0
                h_retry = max(self.h_min, self.h_down * h)
                return self._finalize_return(
                    t,
                    h,
                    y,
                    None,
                    h_retry,
                    np.inf,
                    False,
                    np.inf,
                    0,
                    "half_step_fail_2",
                )

        except RuntimeError as e:
            if self.verbose:
                print(f"[adaptive] error in half steps @ t={t:.6g}: {e}")
            h_retry = max(self.h_min, self.h_down * h)
            return self._finalize_return(
                t,
                h,
                y,
                None,
                h_retry,
                np.inf,
                False,
                np.inf,
                0,
                "half_step_runtime_error",
            )

        # ------------------------------------------------------
        # 3. Compute local normalized error E_curr
        # ------------------------------------------------------
        # All three solves (full + 2 half) succeeded — reset counter
        self._consecutive_nl_fails = 0

        # ── Active-set filter: build transition mask ──
        self._transition_mask = None
        if self.active_set_filter and regime_before is not None:
            proj = self._get_projection()
            if proj is not None:
                mask = proj.regime_changed_mask(regime_before, y.shape[0])
                if mask is not None:
                    self._transition_mask = mask
                    if self.verbose:
                        n_supp = int(np.sum(mask == 0.0))
                        if n_supp > 0:
                            print(f"[adaptive/asf] suppressing {n_supp} DOFs "
                                  f"at t={t:.6g} due to regime transition")

        E_curr = self._scaled_error(y, y_full, y_hi)

        # ------------------------------------------------------
        # 4. Branch on controller mode
        # ------------------------------------------------------

        if self.mode == "classic":
            # --- CLASSIC MODE ---
            # Acceptance: E_curr <= 1.0
            success = (E_curr <= 1.0)
            if E_curr <= 1:
                # Suggest next step size via classic P/PI rule
                h_next = self._propose_h_classic(h, E_curr)
                # Apply NL-recovery cap to prevent death spiral
                h_next = self._apply_nl_recovery_cap(h, h_next)

            if not success:
                # reject: do not advance state
                h_reject = self._rejection_shrink(h, E_curr)
                if self.verbose:
                    print(f"[adaptive] reject @ t={t:.6g}, E={E_curr:.3e}, "
                          f"h_curr={h:.3e} -> h_next={h_reject:.3e}")
                # do NOT update _E_prev on reject (optional; classical codes often don't)
                return self._finalize_return(
                    t,
                    h,
                    y,
                    fk_full,
                    h_reject,
                    E_curr,
                    False,
                    solver_err,
                    it_full,
                    "classic_reject",
                )

            # accept
            if self.verbose:
                print(f"[adaptive] accept @ t={t:.6g} -> t+{h:.3e}, "
                      f"E={E_curr:.3e}, h_next={h_next:.3e}")

            # update memory AFTER success
            self._E_prev = E_curr
            if h > 0.0:
                self._rho_prev = h_next / h
            else:
                self._rho_prev = 1.0

            return self._finalize_return(
                t,
                h,
                y_hi,
                fk_hi,
                h_next,
                E_curr,
                True,
                solver_err,
                it_full,
                "classic_accept",
            )

        else:
            # --- RATIO MODE ---
            # We do *not* accept/reject based on E_curr<=1.
            # Instead we build a ratio proposal rho_prop and
            # decide based on [r_min, r_max].
            # print("yay we are newage")
            h_prop, rho_prop = self._propose_h_ratio(h, E_curr)

            # apply band logic
            decision, fk_new_tmp, h_next, E_out, success, solver_error_out, it_used = \
                self._apply_ratio_acceptance(
                    t, h, h_prop, rho_prop, E_curr, fk_full, it_full, solver_err
                )

            if not success:
                # REJECT: stay at y, use fk_full, don't advance time
                return self._finalize_return(
                    t,
                    h,
                    y,
                    fk_full,
                    h_next,
                    E_out,
                    False,
                    solver_error_out,
                    it_used,
                    "ratio_reject",
                )

            # ACCEPT path from ratio mode:
            # Apply NL-recovery cap to prevent death spiral
            h_next = self._apply_nl_recovery_cap(h, h_next)

            # decision == "ACCEPT" here.
            # we already updated _E_prev, _rho_prev, _reject_streak inside _apply_ratio_acceptance

            return self._finalize_return(
                t,
                h,
                y_hi,
                fk_hi,
                h_next,
                E_out,
                True,
                solver_error_out,
                it_used,
                "ratio_accept",
            )
