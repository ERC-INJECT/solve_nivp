"""General-purpose adaptive stepping Gym environment."""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # pragma: no cover - import guard
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except ImportError as exc:  # pragma: no cover - raised at import time
    raise ImportError(
        "AdaptiveStepperEnv requires 'gymnasium'. Install Solve_IVP_NS[rl] or add gymnasium manually."
    ) from exc

# Type aliases for readability
RewardFn = Callable[[Sequence[float], float, np.ndarray, "AdaptiveStepperEnv"], float]
ObservationFn = Callable[[float | None, int | None, np.ndarray, Sequence[float] | None, "AdaptiveStepperEnv"], np.ndarray]

__all__ = ["AdaptiveStepperEnv", "RewardFn", "ObservationFn"]


class AdaptiveStepperEnv(gym.Env):
    """Gym-compatible environment that wraps an adaptive ODE integrator.

    Parameters
    ----------
    system : callable
        Right-hand side function passed to the integrator.
    dt0, t0, x0, tnmax, dt_min, dt_max : float, array_like
        Baseline time-stepping parameters and initial state.
    nparams : tuple
        Auxiliary parameters forwarded to the integrator (e.g., tolerances).
    integrator : object
        Integrator exposing ``step(system, t, x, h)`` returning the tuple
        ``(x_new, fk_new, error, success, iterations)``.
    component_slices : list[slice] | None
        Optional state partition for block-wise error monitoring.
    reward_fn : callable
        User-supplied reward function of signature ``reward_fn(solver_perf, dt_attempt, xk, env)``.
    obs_fn : callable
        User-supplied observation builder ``obs_fn(dt_attempt, converged, xk, solver_perf, env)``.
    obs_space : gymnasium.Space
        Observation space describing the output of ``obs_fn``.
    q, atol, rtol : float
        Richardson extrapolation and scaling parameters for LTE estimation.
    verbose : bool
        When true, prints per-component scaled error diagnostics.
    skip_error_indices : Iterable[int] | None
        Indices (referencing ``component_slices``) to exclude from the LTE norm.
    alpha, nu, kappa, lam, eps : float
        Reward shaping parameters stored on the environment for use by custom
        reward functions (not consumed internally).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        system: Callable[[float, np.ndarray], np.ndarray],
        dt0: float,
        t0: float,
        x0: np.ndarray,
        tnmax: float,
        dt_min: float,
        dt_max: float,
        nparams: tuple[float, int],
        integrator,
        component_slices: list[slice] | None,
        reward_fn: RewardFn,
        obs_fn: ObservationFn,
        obs_space: gym.Space,
        *,
        q: float = 0.5,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        verbose: bool = False,
        skip_error_indices: Iterable[int] | None = None,
        alpha: float = 2.0,
        nu: float = 2.0,
        kappa: float = 0.01,
        lam: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.system = system
        self.integrator = integrator
        self.tnmax = float(tnmax)
        self.dt_max = float(dt_max)
        self.dt_min = float(dt_min)

        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float64)
        self.observation_space = obs_space

        self.initial_dt = float(dt0)
        self.initial_t = float(t0)
        self.initial_x = np.array(x0, copy=True)
        self.fk = None
        self.nparams = nparams

        self.computational_time = 0.0

        self.error = 0.0
        self.iter_error = 0.0
        self.finished = False
        self.terminate = False

        self.component_slices = component_slices
        self.reward_fn = reward_fn
        self.obs_fn = obs_fn

        self.q = float(q)
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.verbose = bool(verbose)
        self.skip_error_indices = list(skip_error_indices) if skip_error_indices is not None else []

        self.alpha = float(alpha)
        self.nu = float(nu)
        self.kappa = float(kappa)
        self.lam = float(lam)
        self.eps = float(eps)

        self.prev_h: float | None = None
        self._rt_hist: deque[float] = deque(maxlen=512)
        self._rt_q = (0.05, 0.95)
        self.rt_min_est: float | None = None
        self.rt_max_est: float | None = None
        # Cached range for dt computation to avoid per-step recompute
        self._dt_range = float(self.dt_max - self.dt_min)

        # Small buffer reused in error computation to minimize allocations
        self._etol_buf = None  # type: ignore[assignment]

        self.reset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _update_runtime_bounds(self, runtime: float) -> None:
        self._rt_hist.append(float(runtime))
        if len(self._rt_hist) >= 20:
            arr = np.fromiter(self._rt_hist, dtype=float)
            qlo, qhi = np.quantile(arr, self._rt_q)
            span = max(1e-6, qhi - qlo)
            self.rt_min_est = max(0.0, qlo - 0.1 * span)
            self.rt_max_est = qhi + 0.1 * span

    def _get_E(self, y_prev: np.ndarray, y_full: np.ndarray, y_half_full: np.ndarray) -> float:
        """Compute global RMS scaled error using Richardson, mirroring AdaptiveStepping.

        Uses denom = max(1e-14, 2**p - 1), with p derived from q via p = 1/q - 1.
        Applies block-wise exclusion via skip_error_indices when component_slices is provided.
        Allocation is minimized by reusing an internal buffer for etol.
        """
        # Determine Richardson denominator (clamped for safety)
        p = 1.0 / self.q - 1.0
        denom = max(1e-14, (2.0 ** p) - 1.0)

        accum = 0.0
        count = 0

        if self.component_slices is None:
            # Vectorized path for full state
            err = (y_full - y_half_full) / denom
            # Reuse buffer for etol
            if self._etol_buf is None or self._etol_buf.shape != y_half_full.shape:
                self._etol_buf = np.empty_like(y_half_full)
            # etol = atol + rtol*max(|y_half|, |y_full|)
            np.maximum(np.abs(y_half_full), np.abs(y_full), out=self._etol_buf)
            self._etol_buf *= self.rtol
            self._etol_buf += self.atol
            se = err / self._etol_buf
            # sum of squares
            accum = float(np.dot(se.ravel(), se.ravel()))
            count = se.size
        else:
            # Block-wise path with optional exclusions
            if not isinstance(self.component_slices, list):
                raise ValueError("component_slices must be a list of slice objects or indices.")
            for idx, sl in enumerate(self.component_slices):
                if idx in self.skip_error_indices:
                    continue
                prev = y_prev[sl]
                lo = y_full[sl]
                hi = y_half_full[sl]
                err = (lo - hi) / denom
                # Reuse buffer, resizing only when needed
                if self._etol_buf is None or self._etol_buf.shape != hi.shape:
                    self._etol_buf = np.empty_like(hi)
                np.maximum(np.abs(hi), np.abs(lo), out=self._etol_buf)
                self._etol_buf *= self.rtol
                self._etol_buf += self.atol
                se = err / self._etol_buf
                accum += float(np.dot(se.ravel(), se.ravel()))
                count += se.size

                if self.verbose:
                    # Summarized diagnostic to avoid huge prints
                    rms_comp = float(np.sqrt(np.mean((se.ravel()) ** 2))) if se.size else 0.0
                    print(f"Component {idx + 1}: RMS Scaled Error = {rms_comp:.3e}")

        return 0.0 if count == 0 else float(np.sqrt(accum / count))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def step(self, action):  # type: ignore[override]
        action = np.asarray(action, dtype=float)
        dt_norm = float(action[0])
        # Clamp and compute dt using cached range to reduce per-step overhead
        if dt_norm < 0.0:
            dt_norm = 0.0
        elif dt_norm > 1.0:
            dt_norm = 1.0
        dt = dt_norm * self._dt_range + self.dt_min
        dt_attempt = dt

        solver_perf, _, self.t_k, self.xk, self.fk = self.increment_env(self.t_k, self.xk, self.fk, dt, self.nparams)
        runtime_inc, dts, error_LO, error_lil1, error_HI, error, \
            success_LO, success_lil1, success_HI, kiter_LO, iter_lil1, kiter_HI = solver_perf

        self._update_runtime_bounds(runtime_inc)

        reward = self.reward_fn(solver_perf, dt_attempt, self.xk, self)
        done = self.t_k >= self.tnmax
        self.iter_error = error_LO + error_lil1 + error_HI
        converged = 1 if dts > 0 else 0
        obs = self.obs_fn(dt_attempt, converged, self.xk, solver_perf,self.fk, self)
        info = {"t_k1": self.t_k, "xk": self.xk, "residuals": self.fk, "Sim_time": self.computational_time}

        if dts > 0:
            self.prev_h = dts
        return obs, reward, done, self.terminate, info

    def reset(self, **kwargs):  # type: ignore[override]
        seed = kwargs.get("seed")
        if seed is not None:
            np.random.seed(seed)
        self.terminate = False
        self.t_k = self.initial_t
        self.computational_time = 0.0
        self.error = 2.0
        self.dt = 2.0
        self.xk = self.initial_x.copy()
        self.fk = None
        self.iter_error = 2.0
        self.prev_h = None
        self._rt_hist.clear()
        self.rt_min_est = None
        self.rt_max_est = None
        obs = self.obs_fn(None, None, self.xk, None, self.fk, self)
        return obs, {}

    # ------------------------------------------------------------------
    # Integration wrapper
    # ------------------------------------------------------------------
    def increment_env(self, t, xks, fks, h, nparams):
        tol, max_iter = nparams
        start_time = time.time()

        FK1 = fks
        xk1 = xks
        t_k1 = t
        E = 2.0

        error_lil1 = 2.0
        error_HI = 2.0
        kiter_HI = 100.0
        iter_lil1 = 100.0
        success_HI = False
        success_lil1 = False

        xk_LO, FK_LO, error_LO, success_LO, kiter_LO = self.integrator.step(self.system, t, xks, h)
        if success_LO:
            yk1_lil1, FK1_lil1, error_lil1, success_lil1, iter_lil1 = self.integrator.step(self.system, t, xks, h / 2.0)
            if success_lil1:
                xK_HI, FK_HI, error_HI, success_HI, kiter_HI = self.integrator.step(
                    self.system, t + h / 2.0, yk1_lil1, h / 2.0
                )

            if success_HI:
                E = self._get_E(xks, xk_LO, xK_HI)
                xk1 = xK_HI
                FK1 = FK_HI
                t_k1 = t + h

        end_time = time.time()
        run_time = end_time - start_time
        solver_perf = [
            run_time,
            t_k1 - t,
            error_LO,
            error_lil1,
            error_HI,
            E,
            success_LO,
            success_lil1,
            success_HI,
            kiter_LO,
            iter_lil1,
            kiter_HI,
        ]
        self.computational_time += run_time
        return solver_perf, None, t_k1, xk1, FK1
