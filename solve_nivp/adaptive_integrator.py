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
        h0: float = 1e-2,
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
    ) -> None:

        self.integrator = integrator
        self.component_slices = component_slices
        self.atol = float(atol)
        self.rtol = float(rtol)

        # current step size h_n
        self.h = float(h0)

        # global hard clamps on h
        self.h_min = float(h_min)
        self.h_max = float(h_max)

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

        # operating mode
        m = mode.lower().strip()
        if m not in ("classic", "ratio"):
            raise ValueError("mode must be 'classic' or 'ratio'")
        self.mode = m

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
        scale each component by atol + rtol * max(|.|),
        then RMS over included components.

        E <= 1   => "good enough" in classic sense.
        """
        denom = max(1e-14, (2.0 ** self.p) - 1.0)
        accum = 0.0
        count = 0

        if self.component_slices is None:
            if self._err_buf is None or self._err_buf.shape != y_hi.shape:
                self._err_buf = np.empty_like(y_hi)
            if self._etol_buf is None or self._etol_buf.shape != y_hi.shape:
                self._etol_buf = np.empty_like(y_hi)

            # raw_err
            np.subtract(y_lo, y_hi, out=self._err_buf)
            self._err_buf /= denom

            # tol scaling
            np.maximum(np.abs(y_lo), np.abs(y_hi), out=self._etol_buf)
            self._etol_buf *= self.rtol
            self._etol_buf += self.atol

            # scaled error per component
            np.divide(self._err_buf, self._etol_buf, out=self._err_buf)

            accum = float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
            count = self._err_buf.size
        else:
            for i, sl in enumerate(self.component_slices):
                if i in self.skip_error_indices:
                    continue
                lo = y_lo[sl]
                hi = y_hi[sl]

                # (re)allocate to match current block
                if self._err_buf is None or self._err_buf.shape != hi.shape:
                    self._err_buf = np.empty_like(hi)
                if self._etol_buf is None or self._etol_buf.shape != hi.shape:
                    self._etol_buf = np.empty_like(hi)

                # raw_err block
                np.subtract(lo, hi, out=self._err_buf)
                self._err_buf /= denom

                np.maximum(np.abs(lo), np.abs(hi), out=self._etol_buf)
                self._etol_buf *= self.rtol
                self._etol_buf += self.atol

                np.divide(self._err_buf, self._etol_buf, out=self._err_buf)

                accum += float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
                count += self._err_buf.size
        # print(f"error is: {math.sqrt(accum / count) if count>0 else 0.0}")
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
            if self.verbose:
                print(f"[adaptive] nonlinear fail @ t={t:.6g}: shrinking")
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
                if self.verbose:
                    print(f"[adaptive] half-step fail #1 @ t={t:.6g}")
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
                if self.verbose:
                    print(f"[adaptive] half-step fail #2 @ t={t:.6g}")
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

            if not success:
                # reject: do not advance state
                if self.verbose:
                    print(f"[adaptive] reject @ t={t:.6g}, E={E_curr:.3e}, "
                          f"h_curr={h:.3e} -> h_next={h*self.h_down:.3e}")
                # do NOT update _E_prev on reject (optional; classical codes often don't)
                return self._finalize_return(
                    t,
                    h,
                    y,
                    fk_full,
                    h * self.h_down,
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
