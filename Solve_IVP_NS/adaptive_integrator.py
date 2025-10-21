import numpy as np
import math


class AdaptiveStepping:
    """Richardson extrapolation based adaptive step-size controller.

    Strategy
    --------
    Performs one full step of size ``h`` and two half steps of size ``h/2``.
    A Richardson LTE estimate is formed from the difference of the two solutions::

        E_raw = y_full - y_hi
        E = RMS( (y_full - y_hi) / (2^p - 1) / (atol + rtol*max(|y_prev|,|y_hi|)) )

    The step is accepted if ``E <= 1`` and the accepted state is the *high*
    accuracy solution from the two half steps (``y_hi``). On rejection the
    candidate is discarded and only the step-size is shrunk.

    Controller
    ----------
    Proposed new step size (PI variant when previous error available)::

        h_new = h * safety * E^{-alpha} * E_prev^{-beta}

    with clamps ``h_down <= factor <= h_up``. Exponents default to Gustafsson
    style values ``alpha = 0.7/(p+1)``, ``beta = 0.4/(p+1)``.

    Parameters
    ----------
    integrator : IntegrationMethod
        Underlying implicit integrator implementing ``step`` returning the
        standard 5‑tuple.
    component_slices : list[slice], optional
        State partition for block-wise error computation.
    atol, rtol : float
        Absolute / relative tolerances.
    h0, h_min, h_max : float
        Initial, minimum, maximum step sizes.
    h_up, h_down : float
        Growth / shrinkage clamps on multiplicative factor.
    method_order : int, optional
        Explicit order ``p`` (auto inferred from integrator if omitted).
    safety : float
        Multiplicative safety factor (<1) for cautious adaptation.
    use_PI : bool
        Enable PI controller; otherwise falls back to simple proportional.
    skip_error_indices : iterable[int], optional
        Indices (w.r.t. ``component_slices``) excluded from LTE norm (e.g. purely
        algebraic or projection-only variables).

    Returns (from ``step``)
    -----------------------
    ``(y_new, fk_new, h_new, E, success, solver_error, iterations)`` where
    ``fk_new`` is the residual returned by the accepted underlying step (HI variant).
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
        # Optional explicit method order; if None, infer from integrator
        method_order: int | None = None,
        safety: float = 0.9,
        use_PI: bool = True,
        verbose: bool = False,
        skip_error_indices=None,
    ) -> None:
        self.integrator = integrator
        self.component_slices = component_slices
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.h = float(h0)
        self.h_min = float(h_min)
        self.h_max = float(h_max)
        self.h_up = float(h_up)
        self.h_down = float(h_down)
        self.verbose = bool(verbose)
        self.skip_error_indices = set(skip_error_indices or [])

        # Determine method order p (1 for BE/Composite-1, 2 for TR, etc.)
        self.p = int(method_order) if method_order is not None else self._infer_method_order(integrator)

        # PI controller parameters (Gustafsson-like exponents)
        self.safety = float(safety)
        self.use_PI = bool(use_PI)
        self._alpha = 0.7 / (self.p + 1.0)
        self._beta = 0.4 / (self.p + 1.0)
        self._E_prev = None

        # Small buffers to reduce allocations in error computation
        self._etol_buf = None
        self._err_buf = None

    def _infer_method_order(self, integrator) -> int:
        # Respect explicit attribute if present
        p = getattr(integrator, 'order', None)
        if isinstance(p, (int, float)) and p > 0:
            return int(p)
        name = integrator.__class__.__name__.lower()
        # Common cases
        if 'trapezoidal' in name or 'embeddedbetr' in name:
            return 2
        if 'thetamethod' in name:
            theta = getattr(integrator, 'theta', None)
            return 2 if theta is not None and abs(theta - 0.5) < 1e-12 else 1
        # Default conservative choice
        return 1

    def _scaled_error(self, y_prev, y_lo, y_hi) -> float:
        """Global RMS scaled error using Richardson with denom max-clamped.

        Optimized to reuse shared buffers and avoid temporary allocations.
        """
        denom = max(1e-14, (2.0 ** self.p) - 1.0)
        accum = 0.0
        count = 0

        if self.component_slices is None:
            # Allocate or resize buffers
            if self._err_buf is None or self._err_buf.shape != y_hi.shape:
                self._err_buf = np.empty_like(y_hi)
            if self._etol_buf is None or self._etol_buf.shape != y_hi.shape:
                self._etol_buf = np.empty_like(y_hi)
            # err = (y_lo - y_hi) / denom  (in-place)
            np.subtract(y_lo, y_hi, out=self._err_buf)
            self._err_buf /= denom
            # etol = atol + rtol*max(|y_hi|, |y_lo|)  (in-place)
            np.maximum(np.abs(y_hi), np.abs(y_lo), out=self._etol_buf)
            self._etol_buf *= self.rtol
            self._etol_buf += self.atol
            # se = err / etol  (reuse err buffer)
            np.divide(self._err_buf, self._etol_buf, out=self._err_buf)
            accum = float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
            count = self._err_buf.size
        else:
            for i, sl in enumerate(self.component_slices):
                if i in self.skip_error_indices:
                    continue
                lo = y_lo[sl]
                hi = y_hi[sl]
                # Allocate or resize buffers for this block
                if self._err_buf is None or self._err_buf.shape != hi.shape:
                    self._err_buf = np.empty_like(hi)
                if self._etol_buf is None or self._etol_buf.shape != hi.shape:
                    self._etol_buf = np.empty_like(hi)
                # err = (lo - hi) / denom
                np.subtract(lo, hi, out=self._err_buf)
                self._err_buf /= denom
                # etol = atol + rtol*max(|hi|, |lo|)
                np.maximum(np.abs(hi), np.abs(lo), out=self._etol_buf)
                self._etol_buf *= self.rtol
                self._etol_buf += self.atol
                # se = err / etol
                np.divide(self._err_buf, self._etol_buf, out=self._err_buf)
                accum += float(np.dot(self._err_buf.ravel(), self._err_buf.ravel()))
                count += self._err_buf.size

        return 0.0 if count == 0 else math.sqrt(accum / count)

    def _propose_h(self, h: float, E: float) -> float:
        if E <= 0.0 or not np.isfinite(E):
            g = self.h_up
        else:
            if self.use_PI and self._E_prev is not None and self._E_prev > 0.0:
                g = self.safety * (E ** (-self._alpha)) * (self._E_prev ** (-self._beta))
            else:
                g = self.safety * (E ** (-1.0 / (self.p + 1.0)))
            g = min(self.h_up, max(self.h_down, g))
        return min(self.h_max, max(self.h_min, g * h))

    def step(self, fun, t, y, h):
        """Perform one adaptive step; returns (y_new, fk_new, h_new, E, success, solver_error, iterations)."""
        # Full step
        try:
            y_full, fk_full, solver_err, ok_full, it_full = self.integrator.step(fun, t, y, h)
        except RuntimeError as e:
            if self.verbose:
                print(f"[adaptive] error in full step @ t={t:.6g}: {e}")
            return y, None, h, np.inf, False, np.inf, 0
        if not ok_full:
            # Shrink and retry outside
            if self.verbose:
                print(f" convergent reject @ t={t:.6g},h -> {max(self.h_min, self.h_down * h):.3e}")
            return y, None, max(self.h_min, self.h_down * h), np.inf, False, solver_err, it_full

        # Two half-steps
        h2 = 0.5 * h
        try:
            y_half, _, _, ok_h1, _ = self.integrator.step(fun, t, y, h2)
            if not ok_h1:
                if self.verbose:
                    print(f" convergent reject @ t={t:.6g},h -> {max(self.h_min, self.h_down * h):.3e}")
                return y, None, max(self.h_min, self.h_down * h), np.inf, False, np.inf, 0
            y_hi, fk_hi, _, ok_h2, _ = self.integrator.step(fun, t + h2, y_half, h2)
            if not ok_h2:
                if self.verbose:
                    print(f" convergent reject @ t={t:.6g},h -> {max(self.h_min, self.h_down * h):.3e}")
                return y, None, max(self.h_min, self.h_down * h), np.inf, False, np.inf, 0
        except RuntimeError as e:
            if self.verbose:
                print(f"[adaptive] error in half steps @ t={t:.6g}: {e}")
            return y, None, h, np.inf, False, np.inf, 0

        # Scaled RMS error
        E = self._scaled_error(y, y_full, y_hi)
        success = (E <= 1.0)
        h_new = self._propose_h(h, E)
        self._E_prev = E

        if not success:
            if self.verbose:
                print(f"[adaptive] reject @ t={t:.6g}, E={E:.3e} ⇒ h -> {h_new:.3e}")
            # reject: keep y, try smaller h
            return y, fk_full, h_new, E, False, solver_err, it_full

        # accept: use HI solution and residual
        return y_hi, fk_hi, h_new, E, True, solver_err, it_full
