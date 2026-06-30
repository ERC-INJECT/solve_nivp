"""MJF as a first-class integrator under the solve_nivp framework.

The Descriptor-Moreau-Jean-Fremond stepper is wrapped as an
:class:`~solve_nivp.integrations.IntegrationMethod` so the generic
:class:`~solve_nivp.ODESolver` drives it -- fixed-step is simply
``ODESolver(adaptive=False)``.  This replaces the bespoke fixed-step loop
(formerly the notebook-local ``mjf_driver.run_mjf_fixed_step``) with the
package's own driver, while the contact solve (petsc_ssn / MUMPS) inside the
stepper is left untouched.

Public API
----------
MJFIntegrationMethod  : the integrator adapter
solve_mjf_fixed_step  : thin ODESolver-based driver returning the same
                        ``(result, reaction_history)`` tuple the old helper did
MJFContactView        : reaction-history view for existing post-processing
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np

from .integrations import IntegrationMethod


class MJFContactView:
    """Stand-in for the Radau projected-contact object in post-processing.

    MJF keeps reactions in per-step ``info``; the driver collects them and this
    view replays them through the ``reaction_history`` interface used by the
    notebooks' ``_contact_history``.
    """

    def __init__(self, A, y0, component_slices, reaction_state_indices,
                 reaction_state_to_reported_scale):
        self.A = A
        self.y0 = y0
        self.component_slices = component_slices
        self.reaction_state_indices = reaction_state_indices
        self.reaction_state_to_reported_scale = reaction_state_to_reported_scale
        self.projected_radau_contact = None
        self._reaction_history = None

    def reaction_history(self, y_hist):
        hist = np.asarray(self._reaction_history, dtype=float)
        y = np.asarray(y_hist, dtype=float)
        if y.ndim == 1:
            return hist[-1]
        if hist.shape[0] != y.shape[0]:
            raise RuntimeError(
                f"MJF reaction history has {hist.shape[0]} rows; "
                f"expected {y.shape[0]}")
        return hist


def _copy_aux(aux: dict) -> dict:
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in aux.items()}


class MJFIntegrationMethod(IntegrationMethod):
    """Adapt a ``DescriptorMoreauJeanFremondStepper`` to the integrator API.

    ``step(fun, t, y, h)`` ignores ``fun`` (MJF carries its own descriptor
    dynamics), advances one fixed theta-step, carries the MJF ``aux`` state
    internally, and accumulates per-step reactions in ``reaction_history``
    (initial warm reaction + one per step).  Two friction modes:

    * constant Coulomb -- leave ``slip_slice=None``
    * slip-weakening   -- pass ``slip_slice``/``mu_from_slip``/``vel_t_extract``
      /``n_phys`` and the contact law uses ``mu(y_{k+theta})`` via an outer
      fixed point each step.
    """

    def __init__(self, stepper, aux0: dict, *, n_c: int, reaction_scale: float,
                 warm_r: Optional[np.ndarray] = None,
                 slip_slice: Optional[slice] = None,
                 mu_from_slip: Optional[Callable] = None,
                 vel_t_extract=None, n_phys: Optional[int] = None,
                 mu_fp_tol: float = 1.0e-10, mu_fp_max_iter: int = 30):
        self.stepper = stepper
        self.theta = float(stepper.theta)
        self.aux = _copy_aux(aux0)
        self.n_c = int(n_c)
        self.reaction_scale = float(reaction_scale)
        r0 = (np.zeros(self.stepper.n_react) if warm_r is None
              else np.asarray(warm_r, float).copy())
        self.reaction_history = [r0]
        self.info_history: list = []
        self.A = None                       # ODESystem may assign; unused by MJF
        self.slip_slice = slip_slice
        self.mu_from_slip = mu_from_slip
        self.vel_t_extract = vel_t_extract
        self.n_phys = n_phys
        self.mu_fp_tol = float(mu_fp_tol)
        self.mu_fp_max_iter = int(mu_fp_max_iter)

    def _step_weakening(self, t, y, h):
        assert self.mu_from_slip is not None and self.vel_t_extract is not None
        s_k = np.asarray(y[self.slip_slice], float).ravel()
        mu_guess = np.asarray(self.mu_from_slip(s_k), float).ravel()
        v_t_theta = np.zeros(self.n_c)
        y1 = aux1 = info1 = None
        last_resid = np.inf
        n_iter = 0
        for n_iter in range(1, self.mu_fp_max_iter + 1):
            aux_in = _copy_aux(self.aux)
            aux_in["mu"] = mu_guess.copy()
            y1, aux1, info1 = self.stepper.step(t, y, aux_in, h)
            y_theta = y + self.theta * (y1 - y)
            v_t_theta = np.asarray(self.vel_t_extract @ y_theta[:self.n_phys], float).ravel()
            s_theta = s_k + self.theta * h * np.abs(v_t_theta)
            mu_theta = np.asarray(self.mu_from_slip(s_theta), float).ravel()
            last_resid = float(np.linalg.norm(mu_theta - mu_guess, ord=np.inf))
            mu_guess = mu_theta
            if last_resid <= self.mu_fp_tol:
                break
        s_next = s_k + h * np.abs(v_t_theta)
        y1 = np.asarray(y1, float).copy()
        y1[self.slip_slice] = s_next
        aux1 = _copy_aux(aux1)
        aux1["mu"] = np.asarray(self.mu_from_slip(s_next), float).ravel()
        aux1["cum_slip"] = s_next.copy()
        # record mu fixed-point convergence so the caller can verify friction
        info1 = dict(info1)
        info1["mu_fp_resid"] = last_resid
        info1["mu_fp_iters"] = int(n_iter)
        info1["mu_fp_converged"] = bool(last_resid <= self.mu_fp_tol)
        return y1, aux1, info1

    def step(self, fun, t, y, h):
        if self.slip_slice is None:
            y1, aux1, info = self.stepper.step(t, y, self.aux, h)
        else:
            y1, aux1, info = self._step_weakening(t, y, h)
        self.aux = aux1
        p_contact = np.asarray(info.get("p_contact", np.zeros(self.stepper.n_react)), float)
        self.reaction_history.append(p_contact * self.reaction_scale / h)
        self.info_history.append(dict(info))
        success = bool(
            np.all(np.isfinite(y1))
            and info.get("soccp_converged", True)
            and info.get("mu_fp_converged", True)
        )
        err = float(info.get("soccp_residual", 0.0) or 0.0)
        iters = int(info.get("soccp_outer_iters", 0) or 0)
        return y1, None, err, success, iters


def solve_mjf_fixed_step(stepper, y0, aux0, tmax, h_fixed, n_c, reaction_scale,
                         *, slip_slice=None, mu_from_slip=None, vel_t_extract=None,
                         n_phys=None, warm_r=None, mu_fp_tol=1.0e-10,
                         mu_fp_max_iter=30, max_steps=200000, label="MJF", verbose=True):
    """Drive ``stepper`` to ``tmax`` with fixed step ``h_fixed`` via ``ODESolver``.

    Drop-in for the old ``run_mjf_fixed_step``: returns the same 6-tuple
    ``(t, y, h, None, info_list, attempts)`` plus the reaction history.
    """
    from .ODESystem import ODESystem
    from .ODESolver import ODESolver

    method = MJFIntegrationMethod(
        stepper, aux0, n_c=n_c, reaction_scale=reaction_scale, warm_r=warm_r,
        slip_slice=slip_slice, mu_from_slip=mu_from_slip,
        vel_t_extract=vel_t_extract, n_phys=n_phys,
        mu_fp_tol=mu_fp_tol, mu_fp_max_iter=mu_fp_max_iter)
    system = ODESystem(fun=lambda _t, _y: np.zeros_like(_y),
                       y0=np.asarray(y0, float).copy(), method=method, adaptive=False)
    t, y, _h, _fk, _errs = ODESolver(system, (0.0, float(tmax)), h=float(h_fixed)).solve()
    n_steps = len(t) - 1
    h_hist = list(np.diff(t)) if n_steps > 0 else []
    info_hist = []
    for k, info in enumerate(method.info_history):
        ia = dict(info)
        ia["fixed_h"] = h_hist[k] if k < len(h_hist) else float(h_fixed)
        info_hist.append(ia)
    attempts = {"accepted": [True] * n_steps,
                "records": [{"accepted": True, "t": float(t[k]), "h": h_hist[k]}
                            for k in range(n_steps)]}
    if verbose:
        print(f"{label}: done {len(t)} states, t_final={t[-1]:.4e}, "
              f"steps={n_steps}, h={h_fixed:.3e} (via ODESolver)")
        _resids = [i["mu_fp_resid"] for i in method.info_history if "mu_fp_resid" in i]
        if _resids:
            _its = [i.get("mu_fp_iters", 0) for i in method.info_history if "mu_fp_resid" in i]
            _bad = sum(1 for i in method.info_history if i.get("mu_fp_converged") is False)
            ok = "CONVERGED" if _bad == 0 else f"NOT CONVERGED ({_bad} steps)"
            print(f"{label}: mu fixed-point -- max resid={max(_resids):.2e} "
                  f"(tol {mu_fp_tol:.0e}), max iters={max(_its)}/{mu_fp_max_iter} -> {ok}")
        _soc = [float(i.get("soccp_residual", 0.0) or 0.0) for i in method.info_history]
        if _soc:
            print(f"{label}: contact SSN -- max soccp_residual={max(_soc):.2e}")
    result = (np.asarray(t, float), np.asarray(y, float),
              np.asarray(h_hist, float), None, info_hist, attempts)
    return result, np.asarray(method.reaction_history, float)
