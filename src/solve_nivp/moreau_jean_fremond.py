"""theta-Moreau-Jean-Fremond integrator for nonsmooth contact dynamics.

A first/second-order single-stage scheme for unilateral contact with Coulomb
friction, valid for elastodynamics, Biot poroelasticity, and slip-rate /
state-dependent friction (slip-weakening, Dieterich-Ruina aging or slip).

Per step at constant ``theta in [1/2, 1/(1+e_max)]`` the scheme solves a
single block-augmented linear system for the predictor plus a SOCCP for the
contact impulses, with the contact law evaluated on the theta-weighted
average velocity ``u_{k+theta} = (1-theta) u_k + theta u_{k+1}``.  At
``theta = 1/2`` the formulation collapses to the Fremond average-velocity
shift (Acary-Collins-Craft 2025), which is unconditionally energy-dissipative
across stick / slip / take-off / impact regardless of restitution and
contact coupling -- the Kane-pathology fix.

Compared with the projected Radau IIA scheme this is lower-order (1 or 2
vs 3) on the smooth part but gives provable discrete energy bounds.  Use it
as a reference for energy-conservation testing and for problems where
restitution-dependent impacts make Newton-Coulomb at the endpoint
energetically unsafe.

State layout (flat array form):

    y = [q (n_solid), v (n_solid), p_pore (n_fluid)]

Auxiliary state (slip / rate-state) is carried separately as a dict.

Reference
---------
V. Acary, F. Collins-Craft (2025).  "A second-order Moreau-Jean scheme with
the Fremond impact law for the Newton-Lagrange formulation of frictional
contact dynamics."  HAL-04230941.
"""

from __future__ import annotations

import collections
import gc
import inspect as _inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mjf_run_control import MJFRunMonitor, resolve_mjf_checkpoint

from .soccp_pgs import (
    soccp_pgs,
    fremond_shift_factory,
    desaxce_shift_factory,
    SocppPgsInfo,
    _shift_jacobian_fd,
    _classify_regime,
)
from .soc import _soc_fb_phi_and_jac
from .projections import IdentityProjection, MuScaledSOCProjection

try:
    from ._numba_accel import (
        NUMBA_AVAILABLE as _NUMBA_OK,
        soc_fb_ssn_assemble as _soc_fb_assemble_nb,
        soc_fb_ssn_assemble_state_mu as _soc_fb_assemble_state_mu_nb,
    )
except Exception:  # pragma: no cover - numba is an optional dependency
    _NUMBA_OK = False
    _soc_fb_assemble_nb = None
    _soc_fb_assemble_state_mu_nb = None
from .solvers.nonlinear_solvers import ImplicitEquationSolver, PETSC_AVAILABLE


# -----------------------------------------------------------------------------
# Built-in aux-state laws for slip-rate friction
# -----------------------------------------------------------------------------


def _constant_mu_update(aux: dict, u_kt: np.ndarray,
                         block_slices, h: float, params: dict) -> dict:
    """Constant friction: mu does not evolve.  Default for plain Coulomb."""
    return {"mu": np.asarray(aux["mu"], dtype=float).copy()}


def _slip_weakening_update(aux: dict, u_kt: np.ndarray,
                           block_slices, h: float, params: dict) -> dict:
    """Linear slip-weakening: mu = mu_s - (mu_s - mu_d) * min(delta/D_c, 1).

    aux: {'cum_slip': (n_contacts,), 'mu': (n_contacts,)}
    """
    cum = np.asarray(aux["cum_slip"], dtype=float).copy()
    for k, sl in enumerate(block_slices):
        if sl.stop - sl.start > 1:
            cum[k] += float(h) * float(np.linalg.norm(u_kt[sl][1:]))
    D_c = float(params.get("D_c", 1.0))
    mu_s = float(params.get("mu_s", 0.6))
    mu_d = float(params.get("mu_d", 0.5))
    mu_new = mu_s - (mu_s - mu_d) * np.minimum(cum / max(D_c, 1.0e-30), 1.0)
    return {"cum_slip": cum, "mu": mu_new}


def _rate_state_aging_update(aux: dict, u_kt: np.ndarray,
                              block_slices, h: float, params: dict) -> dict:
    """Dieterich aging law: theta_dot = 1 - V theta / L.

    mu = mu_0 + a ln(V/V_0) + b ln(V_0 theta / L)
    """
    state = np.asarray(aux["state"], dtype=float).copy()
    L = float(params.get("L", 1.0e-3))
    a = float(params.get("a", 0.01))
    b = float(params.get("b", 0.014))
    V_0 = float(params.get("V_0", 1.0e-6))
    mu_0 = float(params.get("mu_0", 0.6))

    V_arr = np.zeros(len(block_slices), dtype=float)
    for k, sl in enumerate(block_slices):
        if sl.stop - sl.start > 1:
            V_arr[k] = float(np.linalg.norm(u_kt[sl][1:]))
    V_eff = np.maximum(V_arr, 1.0e-30)

    # Implicit aging-law update for stability:
    # state_new = state_k + h (1 - V state_new / L)
    # state_new (1 + h V/L) = state_k + h
    state_new = (state + float(h)) / (1.0 + float(h) * V_eff / L)

    safe_state = np.maximum(state_new, 1.0e-30)
    mu_new = mu_0 + a * np.log(V_eff / V_0) + b * np.log(V_0 * safe_state / L)
    # Guard against negative friction from extreme parameters.
    mu_new = np.maximum(mu_new, 0.0)
    return {"state": state_new, "mu": mu_new}


def _rate_state_slip_update(aux: dict, u_kt: np.ndarray,
                             block_slices, h: float, params: dict) -> dict:
    """Ruina slip law: theta_dot = -(V theta / L) ln(V theta / L)."""
    state = np.asarray(aux["state"], dtype=float).copy()
    L = float(params.get("L", 1.0e-3))
    a = float(params.get("a", 0.01))
    b = float(params.get("b", 0.014))
    V_0 = float(params.get("V_0", 1.0e-6))
    mu_0 = float(params.get("mu_0", 0.6))

    V_arr = np.zeros(len(block_slices), dtype=float)
    for k, sl in enumerate(block_slices):
        if sl.stop - sl.start > 1:
            V_arr[k] = float(np.linalg.norm(u_kt[sl][1:]))
    V_eff = np.maximum(V_arr, 1.0e-30)

    # Explicit slip-law update; for stiff regimes use sub-stepping outside.
    Vtheta_L = V_eff * np.maximum(state, 1.0e-30) / L
    log_term = np.log(np.maximum(Vtheta_L, 1.0e-30))
    state_new = state - float(h) * Vtheta_L * log_term
    state_new = np.maximum(state_new, 1.0e-30)

    mu_new = mu_0 + a * np.log(V_eff / V_0) + b * np.log(V_0 * state_new / L)
    mu_new = np.maximum(mu_new, 0.0)
    return {"state": state_new, "mu": mu_new}


_AUX_LAWS = {
    "constant": _constant_mu_update,
    "slip_weakening": _slip_weakening_update,
    "rate_state_aging": _rate_state_aging_update,
    "rate_state_slip": _rate_state_slip_update,
}


_CONTACT_RESIDUAL_ALIASES = {
    "fb": "soc_fb",
    "soc_fb": "soc_fb",
    "socfb": "soc_fb",
    "fischer_burmeister": "soc_fb",
    "soc_fischer_burmeister": "soc_fb",
    "soc": "soc_projection",
    "projection": "soc_projection",
    "soc_projection": "soc_projection",
    "natural_map": "soc_projection",
    "soc_natural_map": "soc_projection",
    "desaxce": "soc_projection",
}


def _normalize_contact_linear_solver(name: Any) -> str:
    key = str(name or "auto").strip().lower()
    if key not in ("auto", "dense", "petsc"):
        raise ValueError(
            f"contact_linear_solver must be 'auto', 'dense', or 'petsc'; got {name!r}"
        )
    return key


# Below this size the contact Newton system is solved with LAPACK directly;
# the PETSc round trip costs more than the factorization itself.
_CONTACT_DENSE_SOLVE_MAX = 512


def _normalize_contact_residual(name: Any) -> str:
    key = str(name or "soc_fb").strip().lower().replace("-", "_")
    try:
        return _CONTACT_RESIDUAL_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(_CONTACT_RESIDUAL_ALIASES))
        raise ValueError(
            "contact_residual must be a SOC residual alias "
            f"({valid}); got {name!r}"
        ) from exc


# -----------------------------------------------------------------------------
# Stepper
# -----------------------------------------------------------------------------


@dataclass
class MoreauJeanFremondStepper:
    """Single-step theta-Moreau-Jean-Fremond integrator with porodynamics.

    Parameters
    ----------
    M, K, C : (n_solid, n_solid) solid mass / stiffness / damping.
    S, D : (n_fluid, n_fluid) fluid storage / permeability, optional.
    B_biot : (n_fluid, n_solid) Biot coupling, optional.
    H_callable : ``q -> H`` returning the (n_react, n_solid) local-to-global
        contact frame map.  Pass a constant matrix for small-deformation
        problems; pass a callable for finite-rotation contacts.
    F_callable : ``t -> (n_solid,)`` applied solid force.
    source_callable : ``t -> (n_fluid,)`` applied fluid source, optional.
    block_slices : per-contact slices into the reaction vector.
    e_N_vec : (n_contacts,) normal restitution per contact.
    theta : theta-method parameter, default 0.5 (Fremond average-velocity).
    """

    M: Any
    K: Any
    C: Any
    block_slices: list
    e_N_vec: np.ndarray
    H_callable: Any
    F_callable: Callable[[float], np.ndarray]
    S: Any = None
    D: Any = None
    B_biot: Any = None
    source_callable: Optional[Callable[[float], np.ndarray]] = None
    theta: float = 0.5
    aux_law: Any = "constant"
    aux_law_params: dict = field(default_factory=dict)
    soccp_tol_outer: float = 1.0e-10
    soccp_max_outer: int = 300
    soccp_max_inner: int = 30
    soccp_sor_omega: float = 1.0
    contact_solver: str = "pgs"
    contact_residual: str = "soc_fb"
    contact_soc_rho: Any = "auto"
    contact_soc_rho_scale: float = 1.0
    contact_soc_rho_min: float = 1.0e-12
    contact_soc_rho_max: float = 1.0e12
    contact_petsc_options: Optional[dict] = None
    contact_petsc_reuse_steps: int = 1
    contact_linear_solver: str = "auto"
    contact_ssn_tol: float = 1.0e-10
    # Cap on semismooth-Newton iterations for the cone solve.  Near a marginal
    # stick/slip boundary the SOCCP is only weakly semismooth and convergence to
    # contact_ssn_tol can need ~50-100 iterations; a cap of 30 stalls there just
    # short of tol, which the adaptive driver sees as a step failure -> spurious
    # rejection -> h collapse.  Easy steps break early on the residual, so the
    # higher cap only does extra work where it prevents that collapse.
    contact_ssn_max_iter: int = 100
    contact_ssn_line_search: bool = True
    contact_allow_nonconverged: bool = False

    def __post_init__(self):
        self.M = _to_dense_or_sparse(self.M)
        self.K = _to_dense_or_sparse(self.K)
        self.C = _to_dense_or_sparse(self.C)
        self.n_solid = int(self.M.shape[0])
        if self.M.shape != (self.n_solid, self.n_solid):
            raise ValueError(f"M must be square; got {self.M.shape}")
        if self.K.shape != (self.n_solid, self.n_solid):
            raise ValueError(f"K must be ({self.n_solid}, {self.n_solid})")
        if self.C.shape != (self.n_solid, self.n_solid):
            raise ValueError(f"C must be ({self.n_solid}, {self.n_solid})")

        if self.S is not None or self.D is not None or self.B_biot is not None:
            if self.S is None or self.D is None or self.B_biot is None:
                raise ValueError(
                    "S, D, B_biot must all be provided together for poro mode"
                )
            self.S = _to_dense_or_sparse(self.S)
            self.D = _to_dense_or_sparse(self.D)
            self.B_biot = _to_dense_or_sparse(self.B_biot)
            self.n_fluid = int(self.S.shape[0])
            if self.S.shape != (self.n_fluid, self.n_fluid):
                raise ValueError(f"S must be square; got {self.S.shape}")
            if self.D.shape != (self.n_fluid, self.n_fluid):
                raise ValueError(f"D must be ({self.n_fluid}, {self.n_fluid})")
            if self.B_biot.shape != (self.n_fluid, self.n_solid):
                raise ValueError(
                    f"B_biot must be ({self.n_fluid}, {self.n_solid}); "
                    f"got {self.B_biot.shape}"
                )
        else:
            self.n_fluid = 0

        self.block_slices = list(self.block_slices)
        self.n_contacts = len(self.block_slices)
        self.n_react = sum(sl.stop - sl.start for sl in self.block_slices)
        self.e_N_vec = np.asarray(self.e_N_vec, dtype=float).ravel()
        if self.e_N_vec.size != self.n_contacts:
            raise ValueError(
                f"e_N_vec must have {self.n_contacts} entries; "
                f"got {self.e_N_vec.size}"
            )

        e_max = float(np.max(self.e_N_vec)) if self.n_contacts else 0.0
        theta = float(self.theta)
        if not (0.5 - 1.0e-12 <= theta <= (1.0 / (1.0 + e_max)) + 1.0e-12):
            raise ValueError(
                f"theta = {theta} must lie in [1/2, 1/(1+e_max)] = "
                f"[0.5, {1.0/(1.0+e_max):.6f}] for energy stability "
                f"(Acary-Collins-Craft Prop. 1)"
            )
        self.theta = theta

        if self.aux_law in _AUX_LAWS:
            self._aux_update_fn = _AUX_LAWS[self.aux_law]
        elif callable(self.aux_law):
            self._aux_update_fn = self.aux_law
        else:
            raise ValueError(
                f"aux_law must be one of {list(_AUX_LAWS)} or a callable; "
                f"got {self.aux_law!r}"
            )

        self.contact_solver = str(self.contact_solver or "pgs").lower()
        if self.contact_solver not in ("pgs", "petsc_ssn"):
            raise ValueError(
                "contact_solver must be 'pgs' or 'petsc_ssn'; "
                f"got {self.contact_solver!r}"
            )
        self.contact_residual = _normalize_contact_residual(self.contact_residual)
        self.contact_soc_rho_scale = float(self.contact_soc_rho_scale)
        self.contact_soc_rho_min = float(self.contact_soc_rho_min)
        self.contact_soc_rho_max = float(self.contact_soc_rho_max)
        if self.contact_soc_rho_min <= 0.0:
            raise ValueError("contact_soc_rho_min must be positive")
        if self.contact_soc_rho_max < self.contact_soc_rho_min:
            raise ValueError("contact_soc_rho_max must be >= contact_soc_rho_min")
        self.contact_petsc_options = dict(self.contact_petsc_options or {
            "ksp_type": "preonly",
            "pc_type": "lu",
        })
        self.contact_petsc_reuse_steps = max(1, int(self.contact_petsc_reuse_steps))
        self.contact_linear_solver = _normalize_contact_linear_solver(
            self.contact_linear_solver
        )
        self.contact_ssn_tol = float(self.contact_ssn_tol)
        self.contact_ssn_max_iter = int(self.contact_ssn_max_iter)
        self.contact_ssn_line_search = bool(self.contact_ssn_line_search)
        self.contact_allow_nonconverged = bool(self.contact_allow_nonconverged)
        self._contact_petsc_solver = None

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def state_size(self) -> int:
        return 2 * self.n_solid + self.n_fluid

    def _split_state(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=float).ravel()
        if y.size != self.state_size:
            raise ValueError(
                f"state has {y.size} entries; expected {self.state_size}"
            )
        ns = self.n_solid
        nf = self.n_fluid
        q = y[:ns].copy()
        v = y[ns:2 * ns].copy()
        p_pore = y[2 * ns:2 * ns + nf].copy()
        return q, v, p_pore

    def _pack_state(self, q: np.ndarray, v: np.ndarray,
                    p_pore: np.ndarray) -> np.ndarray:
        return np.concatenate([q, v, p_pore])

    # ------------------------------------------------------------------
    # Augmented operator and its solves
    # ------------------------------------------------------------------

    def _build_aug_operator(self, h: float):
        """Assemble M_hat, S_hat and the augmented operator factor.

        M_hat = M + h*theta*C + (h*theta)^2 K
        S_hat = S + h*theta*D
        Aug   = [[M_hat,         h*theta*B_biot^T],
                 [h*theta*B_biot, S_hat            ]]
        """
        h = float(h)
        ht = h * self.theta
        ht2 = ht * ht
        M_hat = _add(self.M, _scale(self.C, ht))
        M_hat = _add(M_hat, _scale(self.K, ht2))
        if self.n_fluid:
            S_hat = _add(self.S, _scale(self.D, ht))
            B = self.B_biot
            B_T_scaled = _scale(_transpose(B), ht)
            B_scaled = _scale(B, ht)
            aug = _block2x2(M_hat, B_T_scaled, B_scaled, S_hat)
        else:
            aug = M_hat
        return aug, M_hat

    def _solve_aug(self, aug, rhs: np.ndarray) -> np.ndarray:
        """Solve ``aug @ x = rhs`` using sparse or dense backend."""
        if sp.issparse(aug):
            return spla.spsolve(aug.tocsc(), np.asarray(rhs))
        return np.linalg.solve(np.asarray(aug, dtype=float), np.asarray(rhs))

    def _solve_aug_multi(self, aug, RHS: np.ndarray) -> np.ndarray:
        """Solve ``aug @ X = RHS`` for many right-hand sides."""
        if sp.issparse(aug):
            X = spla.spsolve(aug.tocsc(), sp.csc_matrix(RHS))
            if sp.issparse(X):
                X = np.asarray(X.toarray())
            return np.asarray(X)
        return np.linalg.solve(np.asarray(aug, dtype=float), np.asarray(RHS))

    # ------------------------------------------------------------------
    # One step
    # ------------------------------------------------------------------


    def step(
        self,
        t: float,
        y: np.ndarray,
        aux: dict,
        h: float,
        *,
        return_diagnostic: bool = False,
    ) -> tuple[np.ndarray, dict, dict]:
        """Advance (q, v, p_pore, aux) by one step of size ``h``.

        Parameters
        ----------
        t : current time.
        y : flat state at t (q, v, p_pore concatenated).
        aux : auxiliary state at t (per-contact slip / rate-state).
        h : step size (>0).

        Returns
        -------
        y_new : flat state at t + h.
        aux_new : updated auxiliary state.
        info : per-step diagnostic (regimes, iterations, residuals,
            energy slack if return_diagnostic=True).
        """
        if h <= 0.0:
            raise ValueError("step size must be positive")

        q, v, p_pore = self._split_state(y)
        ns = self.n_solid
        nf = self.n_fluid
        ht = float(h) * self.theta
        n_react = self.n_react

        # Build augmented operator and predictor RHS.
        aug, M_hat = self._build_aug_operator(h)
        F_kt = np.asarray(self.F_callable(t + ht), dtype=float).ravel()
        if F_kt.size != ns:
            raise ValueError(f"F(t) returned size {F_kt.size}; expected {ns}")
        rhs_v = _matvec(self.M, v) - ht * _matvec(self.K, q) + ht * F_kt
        rhs_aug = np.zeros(ns + nf, dtype=float)
        rhs_aug[:ns] = rhs_v
        if nf:
            if self.source_callable is not None:
                s_kt = np.asarray(self.source_callable(t + ht), dtype=float).ravel()
            else:
                s_kt = np.zeros(nf, dtype=float)
            rhs_p = _matvec(self.S, p_pore) + ht * s_kt
            rhs_aug[ns:] = rhs_p

        # Predictor (no contact reaction).
        z_pred = self._solve_aug(aug, rhs_aug)
        v_kt_pred = z_pred[:ns]
        p_kt_pore_pred = z_pred[ns:] if nf else np.zeros(0)

        # Compute H and the predictor contact velocity.
        H = self.H_callable(q) if callable(self.H_callable) else self.H_callable
        if sp.issparse(H):
            H_dense = np.asarray(H.toarray(), dtype=float)
        else:
            H_dense = np.asarray(H, dtype=float)
        if H_dense.shape != (n_react, ns):
            raise ValueError(
                f"H must have shape ({n_react}, {ns}); got {H_dense.shape}"
            )

        # Old contact velocity for the Fremond shift.
        u_old = H_dense @ v
        u_kt_pred = H_dense @ v_kt_pred

        if n_react > 0:
            # Build Delassus W = theta * H @ aug_inv_solid_block @ H^T.
            # Solve aug_op @ X = [H^T; 0] for the augmented system.
            RHS_X = np.zeros((ns + nf, n_react), dtype=float)
            RHS_X[:ns, :] = H_dense.T
            X = self._solve_aug_multi(aug, RHS_X)
            X_solid = np.asarray(X[:ns, :], dtype=float)
            W = self.theta * (H_dense @ X_solid)

            # Update mu from current aux.
            mu_vec = np.asarray(aux["mu"], dtype=float).ravel()
            if mu_vec.size != self.n_contacts:
                raise ValueError(
                    f"aux['mu'] has {mu_vec.size} entries; "
                    f"expected {self.n_contacts}"
                )

            # Build Fremond shift using u_old as u_N_old for each contact.
            u_N_old_per_contact = np.array([
                u_old[sl.start] for sl in self.block_slices
            ])
            shift_fn = fremond_shift_factory(
                mu_vec, self.e_N_vec, u_N_old_per_contact, theta=self.theta,
            )

            # SOCCP affine offset b = u_kt_pred (contact velocity at predictor).
            # Inactive contacts are detected post-solve from p[k] ~ 0.
            b_soccp = u_kt_pred.copy()

            # Warm-start from previous step's reaction if cached.
            p0 = aux.get("p_contact_prev", None)
            if p0 is None or len(p0) != n_react:
                p0 = np.zeros(n_react, dtype=float)
            else:
                p0 = np.asarray(p0, dtype=float).copy()

            if self.contact_solver == "pgs":
                p_contact, soccp_info = soccp_pgs(
                    W, b_soccp, self.block_slices, mu_vec,
                    shift_fn=shift_fn, p0=p0,
                    max_outer=self.soccp_max_outer,
                    max_inner=self.soccp_max_inner,
                    tol_outer=self.soccp_tol_outer,
                    sor_omega=self.soccp_sor_omega,
                    return_info=True,
                )
                contact_diag = {}
            else:
                p_contact, soccp_info, contact_diag = self._solve_contact_petsc_ssn(
                    W, b_soccp, mu_vec, shift_fn, p0,
                )

            # Recover full state with contact reaction.
            rhs_aug_full = rhs_aug.copy()
            rhs_aug_full[:ns] = rhs_v + self.theta * (H_dense.T @ p_contact)
            z_full = self._solve_aug(aug, rhs_aug_full)
        else:
            p_contact = np.zeros(0)
            soccp_info = SocppPgsInfo(converged=True)
            z_full = z_pred
            X_solid = None

        v_kt = z_full[:ns]
        p_kt_pore = z_full[ns:] if nf else np.zeros(0)

        # Recover endpoint quantities.
        v_new = v + (v_kt - v) / self.theta
        q_new = q + h * v_kt
        p_pore_new = (
            p_pore + (p_kt_pore - p_pore) / self.theta if nf else np.zeros(0)
        )

        u_kt = H_dense @ v_kt if n_react else np.zeros(0)

        # Update aux state.
        aux_new = self._aux_update_fn(
            aux, u_kt, self.block_slices, h, self.aux_law_params,
        )
        aux_new["p_contact_prev"] = p_contact.copy()
        aux_new["u_old_endpoint"] = (H_dense @ v_new).copy() if n_react else np.zeros(0)

        y_new = self._pack_state(q_new, v_new, p_pore_new)

        info = {
            "soccp_outer_iters": soccp_info.outer_iters,
            "soccp_inner_iters": soccp_info.inner_iters,
            "soccp_converged": soccp_info.converged,
            "soccp_residual": soccp_info.outer_residual,
            "regime": list(soccp_info.regime),
            "p_contact": p_contact.copy(),
            "u_kt": u_kt.copy(),
        }
        if n_react > 0 and self.contact_solver == "petsc_ssn":
            info.update(contact_diag)
        if return_diagnostic:
            info.update(self._energy_terms(
                q, v, p_pore, q_new, v_new, p_pore_new,
                u_kt, p_contact, t, h,
            ))

        return y_new, aux_new, info


    # ------------------------------------------------------------------
    # PETSc contact solve
    # ------------------------------------------------------------------

    def _contact_linear_solve(self, J, rhs: np.ndarray) -> tuple[np.ndarray, str]:
        """Solve one contact Newton system, returning (solution, backend)."""
        n = int(J.shape[0])
        if self.contact_linear_solver == "dense" or (
            self.contact_linear_solver == "auto" and n <= _CONTACT_DENSE_SOLVE_MAX
        ):
            return (
                np.linalg.solve(
                    np.asarray(J, dtype=float), np.asarray(rhs, dtype=float).ravel()
                ),
                "dense",
            )
        return self._contact_linear_solve_petsc(J, rhs), "petsc"

    def _contact_linear_solve_petsc(self, J, rhs: np.ndarray) -> np.ndarray:
        """Solve a contact Newton system with the shared PETSc backend."""
        if not PETSC_AVAILABLE:
            raise RuntimeError(
                "contact_solver='petsc_ssn' requires petsc4py/PETSc"
            )
        if self._contact_petsc_solver is None:
            self._contact_petsc_solver = ImplicitEquationSolver(
                method="semismooth_newton",
                proj=IdentityProjection(),
                linear_solver="petsc",
                petsc_options=dict(self.contact_petsc_options),
                petsc_reuse_steps=self.contact_petsc_reuse_steps,
            )
        x, ok = self._contact_petsc_solver._solve_with_petsc(
            sp.csr_matrix(J), np.asarray(rhs, dtype=float).ravel(),
        )
        if not ok:
            raise RuntimeError("PETSc contact Newton linear solve failed")
        return np.asarray(x, dtype=float).ravel()

    @staticmethod
    def _scalar_fb_and_jac(a: float, b: float) -> tuple[float, float, float]:
        norm = float(np.hypot(a, b))
        if norm > 1.0e-14:
            return norm - a - b, a / norm - 1.0, b / norm - 1.0
        # Any Clarke element is valid at the origin; this conservative choice
        # keeps the Newton matrix nonsingular for the normal NCP rows.
        return 0.0, -1.0, -1.0

    def _contact_soc_rho_value(self, W_block: np.ndarray, d: int) -> float:
        rho = self.contact_soc_rho
        if isinstance(rho, str):
            rho_name = rho.strip().lower().replace("-", "_")
            if rho_name != "auto":
                raise ValueError(
                    "contact_soc_rho must be 'auto' or a positive scalar; "
                    f"got {rho!r}"
                )
            W_block = np.asarray(W_block, dtype=float)
            if d <= 1:
                scale = abs(float(W_block[0, 0])) if W_block.size else 0.0
            else:
                W_tt = W_block[1:, 1:]
                try:
                    eigs = np.linalg.eigvals(W_tt)
                    scale = float(np.max(np.abs(eigs))) if eigs.size else 0.0
                except np.linalg.LinAlgError:
                    diag = np.diag(W_tt) if W_tt.ndim == 2 else np.asarray(W_tt)
                    scale = float(np.max(np.abs(diag))) if diag.size else 0.0
                if not np.isfinite(scale) or scale <= 0.0:
                    scale = abs(float(W_block[0, 0])) if W_block.size else 0.0
            val = self.contact_soc_rho_scale / max(scale, self.contact_soc_rho_min)
        else:
            val = float(rho)
        if val <= 0.0 or not np.isfinite(val):
            raise ValueError(f"contact_soc_rho must be finite and positive (got {val})")
        return float(np.clip(val, self.contact_soc_rho_min, self.contact_soc_rho_max))

    def _contact_block_transforms(self, mu_vec: np.ndarray, block_slices=None) -> list:
        """Per-block mu-scaled cone transforms (t_x, t_y, t_y_inv).

        ``None`` marks blocks handled by the scalar branch (d == 1 or
        vanishing friction).  Hoisted out of the residual loop so one SSN
        solve builds them once instead of per Newton/line-search evaluation.

        The self-dual rescalings ``T_x = diag(1, mu I)``, ``T_y = diag(mu, I)``
        are diagonal, so each is stored as its diagonal vector and applied by
        broadcasting -- this avoids the per-block 2x2/3x3 matmul overhead in
        the residual/Jacobian loop.

        ``block_slices`` overrides ``self.block_slices`` so an active subset of
        contacts (gap index set) can be solved as a self-contained reduced
        problem; it defaults to all blocks.
        """
        if block_slices is None:
            block_slices = self.block_slices
        transforms = []
        for k, sl in enumerate(block_slices):
            d = sl.stop - sl.start
            mu = float(mu_vec[k])
            if d == 1 or mu <= 1.0e-14:
                transforms.append(None)
                continue
            t_x = np.ones(d, dtype=float)
            t_x[1:] = mu
            t_y = np.ones(d, dtype=float)
            t_y[0] = mu
            t_y_inv = np.ones(d, dtype=float)
            t_y_inv[0] = 1.0 / mu
            transforms.append((t_x, t_y, t_y_inv))
        return transforms

    def _contact_ssn_residual_jacobian(
        self,
        p: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        mu_vec: np.ndarray,
        shift_fn,
        *,
        want_jacobian: bool,
        transforms: Optional[list] = None,
        block_slices=None,
    ):
        if block_slices is None:
            block_slices = self.block_slices
        n_react = b.size
        p = np.asarray(p, dtype=float).ravel()
        use_soc_natural_map = self.contact_residual == "soc_projection"

        # Fast path: fused numba kernel for the soc_fb residual mode with a
        # built-in Frémond/De Saxce shift and uniform block dimension.  Falls
        # back to the pure-NumPy loop below for the soc_projection mode, custom
        # shifts (no _soc_shift_params), mixed block dims, or no numba.
        shift_params = getattr(shift_fn, "_soc_shift_params", None)
        if (
            _NUMBA_OK
            and not use_soc_natural_map
            and shift_params is not None
            and n_react > 0
            and block_slices
        ):
            dims = [sl.stop - sl.start for sl in block_slices]
            if all(dd == dims[0] for dd in dims):
                W_c = np.ascontiguousarray(
                    W.toarray() if sp.issparse(W) else W, dtype=float
                )
                res_nb, jac_nb, u_nb = _soc_fb_assemble_nb(
                    W_c,
                    np.ascontiguousarray(b, dtype=float),
                    p,
                    np.ascontiguousarray(mu_vec, dtype=float),
                    np.ascontiguousarray(shift_params["alpha"], dtype=float),
                    np.ascontiguousarray(shift_params["rest"], dtype=float),
                    int(dims[0]),
                    bool(want_jacobian),
                    1.0e-14,
                )
                if want_jacobian:
                    return res_nb, jac_nb, u_nb
                return res_nb, None, u_nb

        u_full = W @ p + b
        residual = np.zeros(n_react, dtype=float)
        jac = np.zeros((n_react, n_react), dtype=float) if want_jacobian else None
        if transforms is None:
            transforms = self._contact_block_transforms(mu_vec, block_slices)
        shift_jacobian = getattr(shift_fn, "jacobian", None)

        for k, sl in enumerate(block_slices):
            u = np.asarray(u_full[sl], dtype=float)
            p_block = np.asarray(p[sl], dtype=float)
            d = p_block.size
            mu = float(mu_vec[k])
            u_hat = shift_fn(u, k) if shift_fn is not None else u
            W_rows = W[sl, :]

            d_uhat_du = None
            if want_jacobian:
                if shift_fn is None:
                    d_uhat_du = np.eye(d)
                elif shift_jacobian is not None:
                    d_uhat_du = shift_jacobian(u, k)
                else:
                    d_uhat_du = _shift_jacobian_fd(shift_fn, u, k)

            if transforms[k] is None:
                if use_soc_natural_map:
                    rho = self._contact_soc_rho_value(W[sl, sl], d)
                    z_n = float(p_block[0]) - rho * float(u_hat[0])
                    proj_n = max(z_n, 0.0)
                    J_n = 1.0 if z_n >= 0.0 else 0.0
                    residual[sl.start] = float(p_block[0]) - proj_n
                    if want_jacobian:
                        jac[sl.start, :] = J_n * rho * (d_uhat_du[0, :] @ W_rows)
                        jac[sl.start, sl.start] += 1.0 - J_n
                    if d > 1:
                        residual[sl.start + 1:sl.stop] = p_block[1:]
                        if want_jacobian:
                            jac[sl.start + 1:sl.stop, sl.start + 1:sl.stop] = np.eye(d - 1)
                else:
                    phi, dphi_du, dphi_dp = self._scalar_fb_and_jac(
                        float(u_hat[0]), float(p_block[0]),
                    )
                    residual[sl.start] = phi
                    if want_jacobian:
                        jac[sl.start, :] = dphi_du * (d_uhat_du[0, :] @ W_rows)
                        jac[sl.start, sl.start] += dphi_dp
                    if d > 1:
                        residual[sl.start + 1:sl.stop] = p_block[1:]
                        if want_jacobian:
                            jac[sl.start + 1:sl.stop, sl.start + 1:sl.stop] = np.eye(d - 1)
                continue

            t_x, t_y, t_y_inv = transforms[k]

            x_sd = t_x * u_hat
            y_sd = t_y * p_block

            if use_soc_natural_map:
                rho = self._contact_soc_rho_value(W[sl, sl], d)
                z = y_sd - rho * x_sd
                if want_jacobian:
                    proj_z, J_proj = MuScaledSOCProjection._proj_mu_scaled_soc(
                        z, 1.0, return_jacobian=True,
                    )
                else:
                    proj_z = MuScaledSOCProjection._proj_mu_scaled_soc(
                        z, 1.0, return_jacobian=False,
                    )
                    J_proj = None
                residual[sl] = t_y_inv * (y_sd - proj_z)
                if want_jacobian:
                    # T_y_inv @ (J_proj @ (rho * T_x @ d_uhat_du @ W_rows));
                    # diagonal T_x / T_y_inv applied as row scalings.
                    M_row = t_x[:, None] * (d_uhat_du @ W_rows)
                    jac[sl, :] = t_y_inv[:, None] * (J_proj @ (rho * M_row))
                    jac[sl, sl] += t_y_inv[:, None] * (
                        (np.eye(d, dtype=float) - J_proj) * t_y[None, :]
                    )
                continue

            phi_sd, dphi_dx, dphi_dy = _soc_fb_phi_and_jac(x_sd, y_sd)
            residual[sl] = t_y_inv * phi_sd

            if want_jacobian:
                # T_y_inv @ (dphi_dx @ T_x @ d_uhat_du @ W_rows) and
                # T_y_inv @ (dphi_dy @ T_y), with diagonal T_* as broadcasts.
                M_row = t_x[:, None] * (d_uhat_du @ W_rows)
                jac[sl, :] = t_y_inv[:, None] * (dphi_dx @ M_row)
                jac[sl, sl] += t_y_inv[:, None] * (dphi_dy * t_y[None, :])

        if want_jacobian:
            return residual, jac, u_full
        return residual, None, u_full

    def _solve_contact_petsc_ssn(
        self,
        W,
        b: np.ndarray,
        mu_vec: np.ndarray,
        shift_fn,
        p0: np.ndarray,
        block_slices=None,
    ) -> tuple[np.ndarray, SocppPgsInfo, dict]:
        """Global reduced SOCCP semismooth Newton with PETSc linear solves.

        ``block_slices`` (default all) lets the caller pass only the active
        contacts (gap index set) as a self-contained reduced problem.
        """
        if block_slices is None:
            block_slices = self.block_slices
        if sp.issparse(W):
            W_dense = np.asarray(W.toarray(), dtype=float)
        else:
            W_dense = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        p = np.asarray(p0, dtype=float).copy()
        transforms = self._contact_block_transforms(mu_vec, block_slices)
        linear_backend = (
            "dense"
            if self.contact_linear_solver == "dense"
            or (self.contact_linear_solver == "auto"
                and b.size <= _CONTACT_DENSE_SOLVE_MAX)
            else "petsc"
        )

        converged = False
        residual_norm = float("inf")
        iters = 0
        for it in range(max(1, self.contact_ssn_max_iter)):
            F, J, _u = self._contact_ssn_residual_jacobian(
                p, W_dense, b, mu_vec, shift_fn, want_jacobian=True,
                transforms=transforms, block_slices=block_slices,
            )
            residual_norm = float(np.linalg.norm(F))
            iters = it + 1
            if residual_norm < self.contact_ssn_tol:
                converged = True
                break

            delta, linear_backend = self._contact_linear_solve(J, -F)
            if self.contact_ssn_line_search:
                alpha = 1.0
                accepted = False
                while alpha >= 1.0e-8:
                    p_trial = p + alpha * delta
                    F_trial, _, _ = self._contact_ssn_residual_jacobian(
                        p_trial, W_dense, b, mu_vec, shift_fn,
                        want_jacobian=False, transforms=transforms,
                        block_slices=block_slices,
                    )
                    trial_norm = float(np.linalg.norm(F_trial))
                    if np.isfinite(trial_norm) and trial_norm <= (1.0 - 1.0e-4 * alpha) * residual_norm:
                        p = p_trial
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    p_trial = p + delta
                    F_trial, _, _ = self._contact_ssn_residual_jacobian(
                        p_trial, W_dense, b, mu_vec, shift_fn,
                        want_jacobian=False, transforms=transforms,
                        block_slices=block_slices,
                    )
                    if np.all(np.isfinite(F_trial)):
                        p = p_trial
                    else:
                        break
            else:
                p = p + delta

        F_final, _, u_full = self._contact_ssn_residual_jacobian(
            p, W_dense, b, mu_vec, shift_fn, want_jacobian=False,
            transforms=transforms, block_slices=block_slices,
        )
        residual_norm = float(np.linalg.norm(F_final))
        converged = converged or residual_norm < self.contact_ssn_tol
        if not converged and not self.contact_allow_nonconverged:
            raise RuntimeError(
                "PETSc contact SSN failed to converge: "
                f"iters={iters}, residual={residual_norm:.3e}"
            )

        info = SocppPgsInfo(
            outer_iters=iters,
            inner_iters=iters,
            converged=converged,
            outer_residual=residual_norm,
            local_residual_max=residual_norm,
        )
        info.regime = [
            _classify_regime(p[sl], u_full[sl], float(mu_vec[k]), tol=self.contact_ssn_tol)
            for k, sl in enumerate(block_slices)
        ]
        diag = {
            "contact_solver": "petsc_ssn",
            "contact_residual": self.contact_residual,
            "contact_ssn_iters": iters,
            "contact_ssn_converged": converged,
            "contact_ssn_residual": residual_norm,
            "contact_linear_solver": linear_backend,
        }
        return p, info, diag

    # ------------------------------------------------------------------
    # Energy diagnostic
    # ------------------------------------------------------------------

    def _energy_terms(
        self, q_k, v_k, p_pore_k,
        q_kp1, v_kp1, p_pore_kp1,
        u_kt, p_contact, t, h,
    ) -> dict:
        """Per-step energy decomposition, after Acary-Collins-Craft Prop 1.

        Returns dict with:
            dE_mech, W_ext, W_damp, W_contact, slack
        where slack = (W_ext + W_damp) - dE_mech + W_contact.  For
        theta in [1/2, 1/(1+e_max)] we expect slack >= 0 to round-off
        for the mechanical energy budget; with damping or contact
        dissipation the inequality is strict.
        """
        ht = float(h) * self.theta
        v_kt = (1.0 - self.theta) * v_k + self.theta * v_kp1

        # Mechanical energy.
        E_k = 0.5 * v_k @ _matvec(self.M, v_k) + 0.5 * q_k @ _matvec(self.K, q_k)
        E_kp1 = (
            0.5 * v_kp1 @ _matvec(self.M, v_kp1)
            + 0.5 * q_kp1 @ _matvec(self.K, q_kp1)
        )
        dE_mech = E_kp1 - E_k

        F_kt = np.asarray(self.F_callable(t + ht), dtype=float).ravel()
        W_ext = float(h) * (v_kt @ F_kt)
        W_damp = -float(h) * (v_kt @ _matvec(self.C, v_kt))
        W_contact = float(u_kt @ p_contact) if u_kt.size and p_contact.size else 0.0

        # Theta-method extra dissipation: (1/2 - theta) [||v_{k+1}-v_k||_M^2 + ||q_{k+1}-q_k||_K^2]
        dv = v_kp1 - v_k
        dq = q_kp1 - q_k
        dE_theta = (0.5 - self.theta) * (
            dv @ _matvec(self.M, dv) + dq @ _matvec(self.K, dq)
        )

        # Discrete dissipation slack (Acary-Collins-Craft Prop 1).
        slack = W_ext + W_damp - dE_mech + dE_theta + W_contact

        return {
            "dE_mech": float(dE_mech),
            "W_ext": float(W_ext),
            "W_damp": float(W_damp),
            "W_contact": float(W_contact),
            "dE_theta": float(dE_theta),
            "slack": float(slack),
        }


_GMRES_HAS_RTOL = "rtol" in _inspect.signature(spla.gmres).parameters


def _gmres_compat(A, b, *, M=None, rtol=1.0e-10, atol=0.0, restart=100, maxiter=1000):
    """``scipy.sparse.linalg.gmres`` across the rtol/tol kwarg rename (>=1.12)."""
    if _GMRES_HAS_RTOL:
        return spla.gmres(A, b, M=M, rtol=rtol, atol=atol,
                          restart=restart, maxiter=maxiter)
    return spla.gmres(A, b, M=M, tol=rtol, atol=atol,
                      restart=restart, maxiter=maxiter)


def _build_dense_row_woodbury(op_csr, rows):
    """Split ``op = op_sparse + U V^T`` for an operator that is sparse except for
    a few dense rows, and prefactor it for a Sherman-Morrison-Woodbury solve.

    A descriptor with full-state feedback (e.g. a dense LQR/observer gain fed
    back through one algebraic row) has a theta operator that is otherwise
    banded but for those gain rows; a single dense row fills the sparse LU and
    dominates the factorization.  Each such row ``r`` is rank-1: ``U`` columns
    are the unit vectors ``e_r`` and ``V`` columns hold the row's off-diagonal
    content, so ``op_sparse`` keeps only the diagonal on those rows and is
    genuinely sparse.  Factor ``op_sparse`` once, then correct with the rank-k
    capacitance ``C = I + V^T op_sparse^{-1} U``.  Exact (an identity), so it
    does not perturb the Newton/adaptive behaviour the way a truncated gain
    would.  Returns ``None`` (caller falls back to a plain LU) if the structure
    does not apply -- e.g. a peeled row has a zero diagonal.
    """
    n = op_csr.shape[0]
    rows = np.unique(np.asarray(rows, dtype=int).ravel())
    rows = rows[(rows >= 0) & (rows < n)]
    if rows.size == 0:
        return None
    k = int(rows.size)
    indptr, indices, data = op_csr.indptr, op_csr.indices, op_csr.data
    V = np.zeros((n, k))
    op = op_csr.tolil()
    for j, r in enumerate(rows):
        s, e = indptr[r], indptr[r + 1]
        rowvec = np.zeros(n)
        rowvec[indices[s:e]] = data[s:e]
        diag = float(rowvec[r])
        if abs(diag) < 1.0e-300:
            return None
        rowvec[r] = 0.0
        V[:, j] = rowvec
        op.rows[int(r)] = [int(r)]
        op.data[int(r)] = [diag]
    try:
        lu = spla.splu(op.tocsc())
        U = np.zeros((n, k))
        U[rows, np.arange(k)] = 1.0
        W = np.asarray(lu.solve(U)).reshape(n, k)
        Cinv = np.linalg.inv(np.eye(k) + V.T @ W)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    return {"lu": lu, "W": W, "V": V, "Cinv": Cinv}


def _build_lowrank_woodbury_from_solves(solve_multi_fn, U, V, coef):
    """Capacitance data for ``op = op_sparse + coef U V^T`` given a raw multi-RHS
    solve against ``op_sparse``.

    Backend-agnostic: the caller supplies ``solve_multi_fn`` (a scipy ``splu``
    back-substitution or a PETSc/MUMPS ``matSolve``) so the exact Sherman-
    Morrison-Woodbury update works under either direct backend.  ``op_sparse`` is
    the operator built from the *sparse* Jacobian (the caller arranges for
    ``rhs_jac`` to omit the ``-U V^T`` term); the correction is exact.  Returns
    ``None`` (caller falls back to plain LU) if ``op_sparse`` is singular or the
    capacitance ``I + V^T op_sparse^{-1} (coef U)`` is.
    """
    U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float)
    if U.ndim == 1: U = U[:, None]
    if V.ndim == 1: V = V[:, None]
    k = U.shape[1]
    try:
        W = np.asarray(solve_multi_fn(float(coef) * U)).reshape(U.shape[0], k)
        Cinv = np.linalg.inv(np.eye(k) + V.T @ W)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    return {"W": W, "V": V, "Cinv": Cinv}


def _solve_unilateral_lcp(M, q, *, tol=1.0e-14, max_iter=200):
    """Projected Gauss-Seidel for the symmetric LCP ``0 <= q + M nu _|_ nu >= 0``.

    Used only for the small (n_contacts x n_contacts) *position projection* problem
    of the combined scheme -- a pure unilateral (no friction) complementarity over
    the normal gaps, so ``M = G B^{-1} G^T`` is symmetric PSD and PGS converges.
    Non-penetrating contacts (``q_i >= 0``) get ``nu_i = 0`` by complementarity, so
    passing the broad index set (all contacts) is safe.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = q.size
    nu = np.zeros(n, dtype=float)
    if n == 0:
        return nu
    M = np.asarray(M, dtype=float)
    d = np.diag(M).copy()
    d[d <= 0.0] = 1.0
    for _ in range(max_iter):
        err = 0.0
        for i in range(n):
            ri = q[i] + float(M[i] @ nu) - M[i, i] * nu[i]
            ni = -ri / d[i]
            ni = ni if ni > 0.0 else 0.0
            err = max(err, abs(ni - nu[i]))
            nu[i] = ni
        if err < tol:
            break
    return nu


class _LRCPythonCtx:
    """MATSHELL context ``y = A_sparse x + coef * U (V^T x)``.

    Fallback used by the ``petsc_shell`` backend only when ``Mat.createLRC`` /
    ``MatDenseCUDA`` is unavailable on the build: ``U``, ``V`` stay host-side, so
    each mult adds the rank-k correction on the CPU (a small device<->host
    round-trip when ``A_sparse`` lives on the GPU).  Implements only ``mult``,
    which is all a right-preconditioned GMRES/FGMRES needs.
    """

    def __init__(self, A_sparse, U, V, coef):
        self._A = A_sparse
        self._U = np.ascontiguousarray(np.asarray(U, dtype=float))
        self._V = np.ascontiguousarray(np.asarray(V, dtype=float))
        self._coef = float(coef)

    def mult(self, mat, x, y):
        self._A.mult(x, y)                                    # y = A_sparse x
        xv = np.asarray(x.getArray(readonly=True))
        yv = np.asarray(y.getArray(readonly=True))
        y.setArray(yv + self._coef * (self._U @ (self._V.T @ xv)))


class _ThetaFactorization:
    """One factorization (or preconditioned iterative solve) of the theta operator.

    A single ``step()`` needs ``n_react + 2`` solves against the same matrix
    (predictor, Delassus columns, corrector); this object factorizes once and
    back-substitutes, batching the Delassus block as a multi-RHS solve.  Its
    lifetime is tied to one ``(t, y, h)`` triple, so no staleness is possible
    regardless of step rejections or exceptions.

    Backends:

    * ``"scipy"`` -- direct sparse LU (``splu``).
    * ``"petsc"`` -- PETSc KSP; direct (``preonly``/``lu`` + MUMPS) or
      iterative (e.g. ``gmres`` + ``hypre``/``fieldsplit``) purely via
      ``petsc_options`` -- no code change needed to switch.
    * ``"iterative"`` -- PETSc-free GMRES + ILU preconditioner (SciPy), for
      low-memory solves at dense meshes where direct-LU fill-in dominates and
      PETSc is unavailable.  Options: ``drop_tol``, ``fill_factor``, ``rtol``,
      ``atol``, ``restart``, ``maxiter``, ``strict``.
    """

    def __init__(self, op, backend: str, petsc_options: Optional[dict] = None,
                 dense_rows=None, lowrank=None):
        self.backend = str(backend or "scipy").lower()
        op_csr = op.tocsr() if sp.issparse(op) else sp.csr_matrix(op)
        self.shape = op_csr.shape
        # Sherman-Morrison-Woodbury fast path for a sparse operator plus a
        # low-rank feedback update; direct backends only.  An explicit
        # ``lowrank=(U, V, coef)`` (op = op_sparse + coef U V^T) costs only the
        # true rank; ``dense_rows`` peels dense rows (rank = #rows).
        self._wb = None
        if lowrank is not None and self.backend == "iterative":
            # The iterative backend has no exact inner solve, so it cannot form
            # the Woodbury capacitance ``I + V^T op_sparse^{-1} (coef U)``; a
            # plain solve of ``op_sparse`` would silently drop the feedback.
            raise ValueError(
                "theta_lowrank_jac requires a direct backend "
                "('scipy' or 'petsc'); the iterative backend has no exact "
                "inner solve for the Woodbury capacitance")
        # ``dense_rows`` peeling stays scipy-only and is built up front (it uses
        # its own peeled-operator LU, kept in ``self._wb['lu']``).  The generic
        # ``lowrank`` update is built AFTER backend setup (below), so its raw
        # inner solve can run against the just-created scipy LU or PETSc KSP.
        if (self.backend == "scipy" and lowrank is None
                and dense_rows is not None
                and np.atleast_1d(dense_rows).size > 0):
            self._wb = _build_dense_row_woodbury(op_csr, dense_rows)
        if self.backend == "iterative":
            opts = dict(petsc_options or {})
            self._op = op_csr.tocsc()
            self._rtol = float(opts.get("rtol", 1.0e-10))
            self._atol = float(opts.get("atol", 0.0))
            self._restart = int(opts.get("restart", 100))
            self._maxiter = int(opts.get("maxiter", 1000))
            self._strict = bool(opts.get("strict", True))
            drop_tol = float(opts.get("drop_tol", 1.0e-4))
            fill_factor = float(opts.get("fill_factor", 10.0))
            try:
                ilu = spla.spilu(self._op, drop_tol=drop_tol,
                                 fill_factor=fill_factor)
                self._M = spla.LinearOperator(self.shape, ilu.solve)
            except (RuntimeError, ValueError):
                # ILU can fail on strongly singular/saddle blocks; fall back to
                # unpreconditioned GMRES rather than crashing.
                self._M = None
        elif self.backend == "petsc":
            if not PETSC_AVAILABLE:
                raise RuntimeError(
                    "theta_linear_solver='petsc' requires petsc4py/PETSc"
                )
            from petsc4py import PETSc
            self._PETSc = PETSc
            n = op_csr.shape[0]
            self._mat = PETSc.Mat().createAIJ(
                size=(n, n), csr=(op_csr.indptr, op_csr.indices, op_csr.data),
            )
            self._mat.assemble()
            opts = dict(petsc_options or {})
            ksp = PETSc.KSP().create(PETSc.COMM_SELF)
            ksp.setOperators(self._mat)
            ksp.setType(opts.get("ksp_type", "preonly"))
            pc = ksp.getPC()
            pc.setType(opts.get("pc_type", "lu"))
            factor_type = opts.get("pc_factor_mat_solver_type")
            if factor_type:
                pc.setFactorSolverType(str(factor_type))
            ksp.setUp()
            self._ksp = ksp
            self._x = self._mat.createVecLeft()
            self._b = self._mat.createVecRight()
        elif self.backend == "petsc_shell":
            # PETSc KSP whose SYSTEM operator is the TRUE low-rank operator
            # ``op_sparse + coef U V^T`` (a MATLRC applied inside the Krylov
            # matvec) while the PRECONDITIONER matrix is ``op_sparse`` alone -- so
            # the rank-k crack-pressure feedback is absorbed by the iterations and
            # the converged residual is a true-operator residual (no inexact
            # Woodbury capacitance).  Never sets ``self._lu`` and never touches the
            # Woodbury path (see the ``!= 'petsc_shell'`` guard below), so
            # ``self._wb`` stays ``None``.  Purely additive: no other backend is
            # reached through this branch.
            if not PETSC_AVAILABLE:
                raise RuntimeError(
                    "theta_linear_solver='petsc_shell' requires petsc4py/PETSc"
                )
            if (dense_rows is not None
                    and np.atleast_1d(dense_rows).size > 0):
                raise NotImplementedError(
                    "theta_linear_solver='petsc_shell' does not support "
                    "theta_dense_rows (dense-row peeling is scipy-only); pass a "
                    "low-rank (U, V) feedback pair via theta_lowrank_jac instead"
                )
            self._init_petsc_shell(op_csr, petsc_options, lowrank)
        elif self._wb is None:
            self._lu = spla.splu(op_csr.tocsc())
        # Generic low-rank Woodbury update, built now that the backend's raw
        # inner solve (``self._lu`` for scipy, ``self._ksp`` for petsc) exists.
        # The ``petsc_shell`` backend is EXCLUDED: it applies the low-rank term
        # directly inside its Krylov matvec (MATLRC), so forming a Woodbury
        # capacitance from its INEXACT KSP solves would silently corrupt every
        # subsequent solve.  ``self._wb`` therefore stays ``None`` for it.
        if lowrank is not None and self.backend != "petsc_shell":
            U_lr, V_lr, coef_lr = lowrank
            self._wb = _build_lowrank_woodbury_from_solves(
                self._raw_solve_multi, U_lr, V_lr, coef_lr)
            if self._wb is None:
                # op here is op_sparse, so a plain solve would silently solve the
                # WRONG operator (feedback dropped) -- fail loudly instead.
                raise RuntimeError(
                    "theta_lowrank_jac: the sparse theta operator is singular; "
                    "cannot form the Woodbury update")

    def _petsc_solve_vec(self, rhs):
        """Raw single-RHS PETSc KSP solve against ``op_sparse`` (no Woodbury)."""
        self._b.setArray(np.asarray(rhs, dtype=float).ravel())
        self._ksp.solve(self._b, self._x)
        return np.array(self._x.getArray(readonly=True), copy=True)

    def _petsc_mat_solve(self, RHS):
        """Raw multi-RHS PETSc solve against ``op_sparse`` via ``matSolve``.

        Falls back to a per-column raw KSP solve (never through ``self.solve``,
        which would re-enter the Woodbury path) if the dense ``matSolve`` route
        is unavailable.
        """
        PETSc = self._PETSc
        RHS = np.asarray(RHS, dtype=float)
        try:
            B_mat = PETSc.Mat().createDense(
                size=(RHS.shape[0], RHS.shape[1]),
                array=np.asfortranarray(RHS), comm=PETSc.COMM_SELF,
            )
            B_mat.assemble()
            X_mat = PETSc.Mat().createDense(
                size=(RHS.shape[0], RHS.shape[1]), comm=PETSc.COMM_SELF,
            )
            X_mat.assemble()
            self._ksp.matSolve(B_mat, X_mat)
            X = np.array(X_mat.getDenseArray(), copy=True)
            B_mat.destroy()
            X_mat.destroy()
            return X
        except Exception:
            return np.column_stack(
                [self._petsc_solve_vec(RHS[:, j]) for j in range(RHS.shape[1])]
            )

    #: option keys accepted by the ``petsc_shell`` backend (unknown keys raise).
    _PETSC_SHELL_OPTION_KEYS = frozenset((
        "mat_type", "ksp_type", "pc_type", "pc_factor_mat_solver_type",
        "rtol", "atol", "max_it", "restart", "pc_side",
        "options_prefix", "extra_options", "warm_start",
    ))

    def _init_petsc_shell(self, op_csr, petsc_options, lowrank):
        """Build the ``petsc_shell`` KSP: system operator = MATLRC
        (``op_sparse + coef U V^T``), preconditioner matrix = ``op_sparse``."""
        import time
        from petsc4py import PETSc
        self._PETSc = PETSc
        opts = dict(petsc_options or {})
        unknown = set(opts) - self._PETSC_SHELL_OPTION_KEYS
        if unknown:
            raise ValueError(
                "theta_linear_solver='petsc_shell' got unknown option key(s) "
                f"{sorted(unknown)}; allowed keys are "
                f"{sorted(self._PETSC_SHELL_OPTION_KEYS)}"
            )
        mat_type = str(opts.get("mat_type", "aij")).lower()
        gpu = mat_type == "aijcusparse"
        ksp_type = str(opts.get("ksp_type", "gmres"))
        pc_type = str(opts.get("pc_type", "ilu"))
        factor_type = opts.get("pc_factor_mat_solver_type")
        rtol = float(opts.get("rtol", 1.0e-10))
        atol = float(opts.get("atol", 1.0e-50))
        max_it = int(opts.get("max_it", 1000))
        restart = opts.get("restart")
        pc_side = str(opts.get("pc_side", "right")).lower()
        self._warm_start = bool(opts.get("warm_start", False))
        self.instrumentation = []
        self.used_shell_fallback = False
        self._U_m = None
        self._V_m = None
        self._c = None
        self._shell_ctx = None

        n = op_csr.shape[0]
        # Preconditioner matrix: op_sparse only (a real assembled AIJ/AIJCUSPARSE
        # matrix the PC can factor).  Built via createAIJ (the production 'petsc'
        # idiom) and converted to aijcusparse on the GPU path.
        Amat = PETSc.Mat().createAIJ(
            size=(n, n), csr=(op_csr.indptr, op_csr.indices, op_csr.data),
        )
        Amat.assemble()
        if gpu:
            Amat_gpu = Amat.convert("aijcusparse")
            if Amat_gpu is not Amat:
                Amat.destroy()
            Amat = Amat_gpu
        self._Amat = Amat

        # System operator: op_sparse (+ low-rank feedback if present).
        if lowrank is not None:
            U_lr, V_lr, coef_lr = lowrank
            U = np.asarray(U_lr, dtype=float)
            V = np.asarray(V_lr, dtype=float)
            if U.ndim == 1:
                U = U[:, None]
            if V.ndim == 1:
                V = V[:, None]
            self._sys = self._build_lrc_operator(Amat, U, V, float(coef_lr), gpu)
        else:
            self._sys = Amat

        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        # Amat_sys = true LRC operator (Krylov matvec); Pmat = op_sparse (factored
        # by the PC).  The rank-k perturbation is absorbed by the iterations.
        ksp.setOperators(self._sys, Amat)
        ksp.setType(ksp_type)
        if restart is not None and ksp_type in (
                "gmres", "fgmres", "lgmres", "dgmres", "pipefgmres", "pgmres"):
            ksp.setGMRESRestart(int(restart))
        pc = ksp.getPC()
        pc.setType(pc_type)
        if factor_type:
            pc.setFactorSolverType(str(factor_type))
        ksp.setTolerances(rtol=rtol, atol=atol, max_it=max_it)
        # UNPRECONDITIONED norm so ``rtol`` means the same thing across variants.
        ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
        # Right preconditioning gives a deterministic residual meaning for
        # gmres; fgmres is right-preconditioned by construction (leave its side).
        if ksp_type != "fgmres":
            ksp.setPCSide(PETSc.PC.Side.RIGHT if pc_side == "right"
                          else PETSc.PC.Side.LEFT)
        if self._warm_start:
            ksp.setInitialGuessNonzero(True)
        prefix = opts.get("options_prefix")
        if prefix:
            ksp.setOptionsPrefix(str(prefix))
        extra = opts.get("extra_options")
        if extra:
            optDB = PETSc.Options()
            for kk, vv in dict(extra).items():
                optDB[str(kk)] = vv
            ksp.setFromOptions()
        t0 = time.perf_counter()
        ksp.setUp()
        self.setup_time = float(time.perf_counter() - t0)
        self._ksp = ksp
        # Vectors from the concrete Amat so their device type matches the PC.
        self._x = Amat.createVecRight()
        self._b = Amat.createVecRight()
        self._x.set(0.0)

    def _build_lrc_operator(self, Amat, U, V, coef, gpu):
        """MATLRC ``Amat + U diag(coef) V^T`` (instance-method createLRC; the
        coefficient is a mandatory Vec).

        On the GPU path the dense factors ``U``, ``V`` are built as HOST dense
        mats (``createDense``) and moved to the device with
        ``.convert('densecuda')`` -- calling ``createDenseCUDA`` directly is a
        native PETSc/cuSPARSE SEGV on this build (round-5 diagnosis: the first
        ``createDenseCUDA`` crashes uncatchably in-process), whereas the
        host-then-convert idiom is probe-verified (device LRC matvec rel_err
        ~1.7e-16, scratchpad ``gpu_lrc_fix_probes.py`` case
        ``fix_convert_densecuda_UV``).  The coefficient Vec is a device
        (``'cuda'``) Vec so its VecType matches the device U/V -- createLRC
        rejects a mismatched coefficient VecType.

        Falls back to a Python MATSHELL computing the same matvec with host
        U, V if createLRC / MatDenseCUDA is unavailable on this build (flagged
        via ``used_shell_fallback``)."""
        PETSc = self._PETSc
        n, k = U.shape
        U_m = V_m = c = None
        U_host = V_host = None
        try:
            if gpu:
                # NEVER createDenseCUDA here: the first such call is a native
                # SEGV on this PETSc/cuSPARSE build.  Build host dense mats and
                # move them to the device via convert -- the probe-verified fix.
                U_host = PETSc.Mat().createDense(
                    size=(n, k), array=np.asfortranarray(U),
                    comm=PETSc.COMM_SELF)
                V_host = PETSc.Mat().createDense(
                    size=(n, k), array=np.asfortranarray(V),
                    comm=PETSc.COMM_SELF)
                U_host.assemble()
                V_host.assemble()
                U_m = U_host.convert("densecuda")
                V_m = V_host.convert("densecuda")
                # convert returns fresh device mats; release the host originals.
                for hm in (U_host, V_host):
                    if hm is not None and hm is not U_m and hm is not V_m:
                        try:
                            hm.destroy()
                        except Exception:
                            pass
                U_host = V_host = None
            else:
                U_m = PETSc.Mat().createDense(
                    size=(n, k), array=np.asfortranarray(U),
                    comm=PETSc.COMM_SELF)
                V_m = PETSc.Mat().createDense(
                    size=(n, k), array=np.asfortranarray(V),
                    comm=PETSc.COMM_SELF)
                U_m.assemble()
                V_m.assemble()
            c_arr = coef * np.ones(k, dtype=float)
            if gpu:
                c = PETSc.Vec().create(comm=PETSc.COMM_SELF)
                c.setSizes(k)
                c.setType("cuda")
                c.setUp()
                c.setArray(c_arr)
                c.assemble()
            else:
                c = PETSc.Vec().createWithArray(c_arr, comm=PETSc.COMM_SELF)
            self._c_arr = c_arr           # keep the buffer alive for createWithArray
            A_full = PETSc.Mat().createLRC(Amat, U_m, c, V_m)
            self._U_m, self._V_m, self._c = U_m, V_m, c
            return A_full
        except Exception:
            for m in (U_m, V_m, c, U_host, V_host):
                try:
                    if m is not None:
                        m.destroy()
                except Exception:
                    pass
            self._U_m = self._V_m = self._c = None
            self.used_shell_fallback = True
            ctx = _LRCPythonCtx(Amat, U, V, coef)
            shell = PETSc.Mat().createPython((n, n), ctx, comm=PETSc.COMM_SELF)
            shell.setUp()
            self._shell_ctx = ctx
            return shell

    def _petsc_shell_solve_vec(self, rhs, warm=False):
        """Single-RHS KSP solve of the true LRC operator; records
        (iterations, wall, converged_reason) and raises on divergence."""
        import time
        self._b.setArray(np.asarray(rhs, dtype=float).ravel())
        if warm and self._warm_start:
            self._ksp.setInitialGuessNonzero(True)   # reuse the previous solution
        else:
            self._ksp.setInitialGuessNonzero(False)
            self._x.set(0.0)
        t0 = time.perf_counter()
        self._ksp.solve(self._b, self._x)
        wall = float(time.perf_counter() - t0)
        reason = int(self._ksp.getConvergedReason())
        iters = int(self._ksp.getIterationNumber())
        self.instrumentation.append((iters, wall, reason))
        if reason < 0:
            raise RuntimeError(
                "theta petsc_shell KSP failed to converge "
                f"(KSPConvergedReason={reason}, iters={iters}); loosen rtol or "
                "strengthen the preconditioner"
            )
        return np.array(self._x.getArray(readonly=True), copy=True)

    def _raw_solve_multi(self, B):
        """Multi-RHS solve of the *sparse* operator only (bypasses Woodbury).

        Used to build the Woodbury capacitance and inside :meth:`_wb_apply`.
        Always returns a 2-D array of shape ``(n, ncols)``.
        """
        B = np.asarray(B, dtype=float)
        if B.ndim == 1:
            B = B[:, None]
        if self.backend == "petsc":
            return np.asarray(self._petsc_mat_solve(B))
        return np.asarray(self._lu.solve(B))

    def _wb_apply(self, B):
        """Woodbury solve: ``op^{-1} B = x0 - W C^{-1} (V^T x0)``, ``x0 = op_sparse^{-1} B``.

        ``dense_rows`` factorizations carry a self-contained peeled-operator LU
        under ``wb['lu']``; the generic low-rank path has no stored factor and
        routes through the backend's raw solve.
        """
        wb = self._wb
        lu = wb.get("lu")
        if lu is not None:
            x0 = lu.solve(B)
        else:
            x0 = self._raw_solve_multi(B)
            if np.asarray(B).ndim == 1:
                x0 = x0[:, 0]
        return x0 - wb["W"] @ (wb["Cinv"] @ (wb["V"].T @ x0))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float).ravel()
        if self._wb is not None:
            return np.asarray(self._wb_apply(rhs))
        if self.backend == "iterative":
            x, info = _gmres_compat(
                self._op, rhs, M=self._M, rtol=self._rtol, atol=self._atol,
                restart=self._restart, maxiter=self._maxiter,
            )
            if info != 0 and self._strict:
                raise RuntimeError(
                    "theta iterative solve did not converge "
                    f"(gmres info={info}); loosen rtol or strengthen the "
                    "preconditioner (drop_tol/fill_factor)"
                )
            return np.asarray(x)
        if self.backend == "petsc":
            return self._petsc_solve_vec(rhs)
        if self.backend == "petsc_shell":
            # z_pred / corrector single solve: warm-startable (the B-column loop
            # in solve_multi stays cold).
            return self._petsc_shell_solve_vec(rhs, warm=True)
        return np.asarray(self._lu.solve(rhs))

    def solve_multi(self, RHS: np.ndarray) -> np.ndarray:
        RHS = np.asarray(RHS, dtype=float)
        if RHS.ndim == 1:
            RHS = RHS[:, None]
        if self._wb is not None:
            return np.asarray(self._wb_apply(RHS))
        if self.backend == "iterative":
            return np.column_stack(
                [self.solve(RHS[:, j]) for j in range(RHS.shape[1])]
            )
        if self.backend == "petsc":
            return np.asarray(self._petsc_mat_solve(RHS))
        if self.backend == "petsc_shell":
            # NEVER ksp.matSolve for an iterative KSP: loop columns, cold start
            # (B-columns must not inherit a warm guess).
            return np.column_stack(
                [self._petsc_shell_solve_vec(RHS[:, j], warm=False)
                 for j in range(RHS.shape[1])]
            )
        return np.asarray(self._lu.solve(RHS))

    def destroy(self):
        self._wb = None
        if self.backend == "petsc":
            for attr in ("_x", "_b", "_ksp", "_mat"):
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.destroy()
                    setattr(self, attr, None)
        elif self.backend == "petsc_shell":
            # ``_sys`` may alias ``_Amat`` (lowrank=None): destroy each unique
            # PETSc object once.
            seen = set()
            for attr in ("_x", "_b", "_ksp", "_sys", "_Amat",
                         "_U_m", "_V_m", "_c"):
                obj = getattr(self, attr, None)
                if obj is not None and id(obj) not in seen:
                    seen.add(id(obj))
                    try:
                        obj.destroy()
                    except Exception:
                        pass
                setattr(self, attr, None)


class DescriptorMoreauJeanFremondStepper(MoreauJeanFremondStepper):
    """Contact-reduced theta-MJF stepper for first-order descriptor DAEs.

    This variant works directly with systems of the form

        A ydot = rhs(t, y) + B p_delta,

    using the theta state ``z = y_{k+theta}`` as the eliminated physical
    unknown.  Algebraic projection-style constraints ``q = g(y)`` are enforced
    by replacing rows in the theta operator with the corresponding linearized
    constraint rows.  For linear constraints this is exact.
    """

    def __init__(
        self,
        *,
        A,
        rhs_callable,
        rhs_jac_callable,
        D_extract,
        B,
        contacts: list[dict],
        constraints: Optional[list[dict]] = None,
        contact_offset_force: Optional[Any] = None,
        mu_state_callback: Optional[Any] = None,
        dmu_dstate_callback: Optional[Any] = None,
        gap_callable: Optional[Any] = None,
        gap_tol: float = 0.0,
        gap_jac: Optional[Any] = None,
        combined_projection: bool = False,
        projection_metric_inv: Optional[Any] = None,
        projection_tol: float = 1.0e-12,
        combined_projection_max_iter: int = 3,
        reaction_state_to_reported_scale: Any = 1.0,
        theta: float = 0.5,
        aux_law: Any = "constant",
        aux_law_params: Optional[dict] = None,
        contact_solver: str = "petsc_ssn",
        contact_residual: str = "soc_fb",
        contact_soc_rho: Any = "auto",
        contact_soc_rho_scale: float = 1.0,
        contact_soc_rho_min: float = 1.0e-12,
        contact_soc_rho_max: float = 1.0e12,
        contact_petsc_options: Optional[dict] = None,
        contact_petsc_reuse_steps: int = 1,
        contact_linear_solver: str = "auto",
        theta_linear_solver: str = "scipy",
        theta_petsc_options: Optional[dict] = None,
        theta_iterative_options: Optional[dict] = None,
        theta_petsc_reuse_steps: int = 1,
        theta_dense_rows: Optional[Any] = None,
        theta_lowrank_jac: Optional[Any] = None,
        theta_cache_size: int = 4,
        contact_ssn_tol: float = 1.0e-10,
        contact_ssn_max_iter: int = 100,
        contact_ssn_line_search: bool = True,
        contact_allow_nonconverged: bool = False,
    ):
        self.A = _to_dense_or_sparse(A)
        self.rhs_callable = rhs_callable
        self.rhs_jac_callable = rhs_jac_callable
        self.D_extract = _to_dense_or_sparse(D_extract)
        self.B = _to_dense_or_sparse(B)
        self.constraints = list(constraints or [])
        self.contact_offset_force = contact_offset_force
        self.mu_state_callback = mu_state_callback
        self.dmu_dstate_callback = dmu_dstate_callback
        # Geometric activation (gap index set): gap_callable(q_k) -> g_N per contact
        # keeps contact alpha in the cone problem only while g_alpha(q_k) <= gap_tol,
        # so the cone can open/close contacts (cf. Acary & Collins-Craft 2025, Eq. 19,
        # less the u_{N,k}<=0 term, which would flicker a persistent contact whose
        # normal velocity vibrates around 0).  The gap must be scaled consistently
        # with D_extract, i.e. d g_alpha/dt = u_{N,alpha}.  For a PERSISTENT contact
        # held exactly at g=0 (e.g. a prestressed fault, gap ~ 1e-10 numerical noise),
        # set gap_tol to a small margin above that noise (e.g. 1e-6) so it stays
        # active; for a genuinely separating contact (a bouncing ball, gap ~ O(1))
        # use gap_tol=0.  Default gap_callable=None => persistent contact, unchanged.
        self.gap_callable = gap_callable
        self.gap_tol = float(gap_tol)
        if gap_callable is not None and mu_state_callback is not None:
            raise NotImplementedError(
                "gap_callable is not yet supported together with mu_state_callback "
                "(the state-mu coupled SSN path); use the outer mu fixed point "
                "(mu_from_state in solve_mjf_adaptive) instead."
            )
        # Combined projection (GGL position admissibility, cf. Acary 2013
        # "Projected event-capturing time-stepping schemes" + Siconos
        # MoreauJeanCombinedProjectionOSi).  This is a thin wrapper *around* the
        # Fremond velocity law, NOT a replacement: the velocity SOCCP still gives
        # the physical contact impulse p; an additional projection then nudges the
        # END-of-step position so g(q_{k+1}) >= 0, using a separate projection
        # multiplier nu that is NEVER treated as a contact impulse and NEVER turned
        # into a rebound velocity.  This fixes gap drift / penetration without the
        # spurious kinetic energy that the naive  u_N + g/h  correction injects
        # (which made the inelastic ball bounce).  Opt-in; default off => the path
        # below is byte-for-byte the validated velocity-only scheme.
        self.combined_projection = bool(combined_projection)
        self.gap_jac = gap_jac
        self.projection_metric_inv = projection_metric_inv
        self.projection_tol = float(projection_tol)
        self.combined_projection_max_iter = max(1, int(combined_projection_max_iter))
        if self.combined_projection:
            if self.gap_callable is None:
                raise ValueError(
                    "combined_projection=True requires gap_callable (the normal gap "
                    "g(q) per contact, scaled consistently with D_extract so that "
                    "dg/dt = u_N)."
                )
            if self.gap_jac is None:
                raise ValueError(
                    "combined_projection=True requires gap_jac: the gap Jacobian "
                    "dg/dy, an (n_contacts x n_state) array/sparse matrix or a "
                    "callable(y, t) returning one.  For a linear gap g = G y it is "
                    "the constant matrix G."
                )
            if mu_state_callback is not None:
                raise NotImplementedError(
                    "combined_projection is not yet supported with mu_state_callback; "
                    "use the outer mu fixed point (mu_from_state in solve_mjf_adaptive)."
                )
        self.theta = float(theta)
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        self.aux_law = aux_law
        self.aux_law_params = dict(aux_law_params or {})
        if self.aux_law in _AUX_LAWS:
            self._aux_update_fn = _AUX_LAWS[self.aux_law]
        elif callable(self.aux_law):
            self._aux_update_fn = self.aux_law
        else:
            raise ValueError(
                f"aux_law must be one of {list(_AUX_LAWS)} or a callable; "
                f"got {self.aux_law!r}"
            )

        self.n_state = int(self.A.shape[0])
        if self.A.shape != (self.n_state, self.n_state):
            raise ValueError(f"A must be square; got {self.A.shape}")
        if self.D_extract.shape[1] != self.n_state:
            raise ValueError(
                f"D_extract must have {self.n_state} columns; got {self.D_extract.shape}"
            )
        if self.B.shape[0] != self.n_state:
            raise ValueError(f"B must have {self.n_state} rows; got {self.B.shape}")

        self.contacts = list(contacts)
        self.block_slices = []
        self.contact_velocity_rows = []
        self.mu_init = np.zeros(len(self.contacts), dtype=float)
        self.e_N_vec = np.zeros(len(self.contacts), dtype=float)
        cursor = 0
        for k, c in enumerate(self.contacts):
            rows = [int(c["vel_normal_idx"])]
            rows.extend(int(i) for i in np.atleast_1d(c.get("vel_tangential_idx", [])))
            d = len(rows)
            self.block_slices.append(slice(cursor, cursor + d))
            self.contact_velocity_rows.extend(rows)
            cursor += d
            mu_val = c.get("mu_init", c.get("mu", 0.0))
            self.mu_init[k] = 0.0 if callable(mu_val) else float(mu_val)
            self.e_N_vec[k] = float(c.get("e", 0.0))
        self.n_contacts = len(self.block_slices)
        self.n_react = cursor
        if self.B.shape[1] != self.n_react:
            raise ValueError(
                f"B must have {self.n_react} columns for the contact blocks; got {self.B.shape}"
            )
        scale = np.asarray(reaction_state_to_reported_scale, dtype=float)
        if scale.ndim == 0:
            scale = np.full(self.n_react, float(scale), dtype=float)
        else:
            scale = scale.ravel().astype(float, copy=True)
        if scale.size != self.n_react:
            raise ValueError(
                "reaction_state_to_reported_scale must be scalar or have one "
                f"entry per reaction DOF; got {scale.size}, expected {self.n_react}"
            )
        if np.any(scale <= 0.0):
            raise ValueError("reaction_state_to_reported_scale entries must be positive")
        for sl in self.block_slices:
            if sl.stop - sl.start > 1 and not np.allclose(scale[sl], scale[sl.start]):
                raise ValueError(
                    "reaction_state_to_reported_scale must be uniform within each "
                    "contact block for Coulomb cone scaling"
                )
        self.reaction_state_to_reported_scale = scale
        if self.contact_velocity_rows:
            self.D_contact = self.D_extract[self.contact_velocity_rows, :]
        else:
            self.D_contact = sp.csr_matrix((0, self.n_state))

        self.contact_solver = str(contact_solver or "petsc_ssn").lower()
        if self.contact_solver not in ("pgs", "petsc_ssn"):
            raise ValueError(
                "contact_solver must be 'pgs' or 'petsc_ssn'; "
                f"got {self.contact_solver!r}"
            )
        self.contact_residual = _normalize_contact_residual(contact_residual)
        self.contact_soc_rho = contact_soc_rho
        self.contact_soc_rho_scale = float(contact_soc_rho_scale)
        self.contact_soc_rho_min = float(contact_soc_rho_min)
        self.contact_soc_rho_max = float(contact_soc_rho_max)
        if self.contact_soc_rho_min <= 0.0:
            raise ValueError("contact_soc_rho_min must be positive")
        if self.contact_soc_rho_max < self.contact_soc_rho_min:
            raise ValueError("contact_soc_rho_max must be >= contact_soc_rho_min")
        self.contact_petsc_options = dict(contact_petsc_options or {
            "ksp_type": "preonly",
            "pc_type": "lu",
        })
        self.contact_petsc_reuse_steps = max(1, int(contact_petsc_reuse_steps))
        self.contact_linear_solver = _normalize_contact_linear_solver(
            contact_linear_solver
        )
        self.theta_linear_solver = str(theta_linear_solver or "scipy").lower()
        if self.theta_linear_solver not in (
                "scipy", "petsc", "iterative", "petsc_shell"):
            raise ValueError(
                "theta_linear_solver must be 'scipy', 'petsc', 'iterative', or "
                f"'petsc_shell'; got {self.theta_linear_solver!r}"
            )
        self.theta_petsc_options = dict(theta_petsc_options or self.contact_petsc_options)
        self.theta_iterative_options = dict(theta_iterative_options or {})
        self.theta_petsc_reuse_steps = max(1, int(theta_petsc_reuse_steps))
        if theta_dense_rows is None:
            self.theta_dense_rows = None
        else:
            rows = np.unique(np.asarray(theta_dense_rows, dtype=int).ravel())
            self.theta_dense_rows = rows if rows.size else None
        # Explicit low-rank Jacobian term: the true Jacobian is
        # ``rhs_jac(t, y) - U @ V.T`` (rhs_jac returns the genuinely sparse part).
        # Since ``op = (1/theta) A - h J``, the operator is ``op_sparse + h U V^T``,
        # a rank-k Woodbury update -- exact, and far cheaper than dense feedback
        # rows when a dense gain (e.g. full-state LQR/observer feedback fed through
        # a sparse input column) is globally low rank.  Mutually exclusive with
        # theta_dense_rows (this takes precedence).
        if theta_lowrank_jac is None:
            self.theta_lowrank_jac = None
        else:
            U, V = theta_lowrank_jac
            U = np.asarray(U, dtype=float); V = np.asarray(V, dtype=float)
            if U.ndim == 1: U = U[:, None]
            if V.ndim == 1: V = V[:, None]
            if U.shape != V.shape:
                raise ValueError(
                    f"theta_lowrank_jac U and V must have the same shape; got {U.shape} vs {V.shape}"
                )
            self.theta_lowrank_jac = (U, V)
            self.theta_dense_rows = None
        self.contact_ssn_tol = float(contact_ssn_tol)
        self.contact_ssn_max_iter = int(contact_ssn_max_iter)
        self.contact_ssn_line_search = bool(contact_ssn_line_search)
        self.contact_allow_nonconverged = bool(contact_allow_nonconverged)
        self._contact_petsc_solver = None
        # Multi-level theta-factorization cache: op(h) = (1/theta)A - hJ is
        # affine and byte-identical at a fixed h, so each distinct h needs
        # exactly one factorization.  Keyed by float(h) with an LRU bound so
        # h-revisits under an adaptive driver are free.
        self.theta_cache_size = max(1, int(theta_cache_size))
        self._theta_cache = collections.OrderedDict()
        self._theta_factorizations = 0     # permanent instrumentation counter
        self._theta_cache_evictions = 0
        self._B_dense = None
        # last converged effective contact impulse + its (t, y, h) key, reused
        # as the contact-SSN warm start on mu fixed-point re-entries (identical
        # (t, y, h), only mu changed) when the caller supplies no nonzero
        # p_contact_prev.  Restricted to re-entries on purpose: warm-starting a
        # semismooth Newton across steps can shift the active set into a worse
        # basin and slow/stall convergence.  Initial guess only.
        self._last_p_eff = None
        self._last_call_key = None

    @property
    def state_size(self) -> int:
        return self.n_state

    @staticmethod
    def _index_array(indexer, n: int) -> np.ndarray:
        if isinstance(indexer, slice):
            return np.arange(*indexer.indices(n), dtype=int)
        arr = np.asarray(indexer, dtype=int).ravel()
        arr[arr < 0] += n
        return arr

    @staticmethod
    def _call_constraint_fn(fn, y_sub, t):
        if fn is None:
            return None
        if not callable(fn):
            return fn
        for args in ((y_sub, t, None), (y_sub, t), (y_sub,)):
            try:
                return fn(*args)
            except TypeError:
                continue
        return fn(y_sub)

    def _rhs_and_jac_affine(self, t: float, y_ref: np.ndarray):
        f_ref = np.asarray(self.rhs_callable(t, y_ref), dtype=float).ravel()
        J = self.rhs_jac_callable(t, y_ref)
        J_sp = J.tocsr() if sp.issparse(J) else sp.csr_matrix(np.asarray(J, dtype=float))
        f_affine = f_ref - np.asarray(J_sp @ y_ref).ravel()
        return f_affine, J_sp

    def _build_theta_system(self, t: float, y: np.ndarray, h: float):
        y = np.asarray(y, dtype=float).ravel()
        if y.size != self.n_state:
            raise ValueError(f"state has {y.size} entries; expected {self.n_state}")
        t_theta = float(t) + self.theta * float(h)
        f_affine, J = self._rhs_and_jac_affine(t_theta, y)
        A_sp = self.A.tocsr() if sp.issparse(self.A) else sp.csr_matrix(self.A)
        op = ((1.0 / self.theta) * A_sp - float(h) * J).tocsr()
        rhs = np.asarray((1.0 / self.theta) * (A_sp @ y) + float(h) * f_affine).ravel()

        if self.constraints:
            patch_rows = []
            patch_cols = []
            patch_data = []
            q_all = []
            for c in self.constraints:
                y_sl = c["y_slice"]
                q_sl = c["q_slice"]
                y_idx = self._index_array(y_sl, self.n_state)
                q_idx = self._index_array(q_sl, self.n_state)
                y_sub = y[y_idx]
                g_val = np.asarray(
                    self._call_constraint_fn(c.get("g"), y_sub, t_theta),
                    dtype=float,
                ).ravel()
                Jg_raw = self._call_constraint_fn(c.get("dg_dy"), y_sub, t_theta)
                if Jg_raw is None:
                    raise ValueError("Descriptor MJF constraints require dg_dy")
                Jg = (
                    Jg_raw.tocoo() if sp.issparse(Jg_raw)
                    else sp.coo_matrix(np.atleast_2d(np.asarray(Jg_raw, dtype=float)))
                )
                if q_idx.size != g_val.size or Jg.shape != (q_idx.size, y_idx.size):
                    raise ValueError(
                        "constraint shape mismatch: "
                        f"g={g_val.shape}, dg={Jg.shape}, q={q_idx.size}, y={y_idx.size}"
                    )
                rhs[q_idx] = g_val - Jg @ y_sub
                q_all.append(q_idx)
                patch_rows.append(q_idx)
                patch_cols.append(q_idx)
                patch_data.append(np.ones(q_idx.size))
                if Jg.nnz:
                    patch_rows.append(q_idx[Jg.row])
                    patch_cols.append(y_idx[Jg.col])
                    patch_data.append(-Jg.data)

            q_all = np.concatenate(q_all)
            if np.unique(q_all).size != q_all.size:
                raise ValueError("constraints assign overlapping q_slice rows")
            zero_mask = np.zeros(self.n_state, dtype=bool)
            zero_mask[q_all] = True
            op.data[np.repeat(zero_mask, np.diff(op.indptr))] = 0.0
            patch = sp.coo_matrix(
                (np.concatenate(patch_data),
                 (np.concatenate(patch_rows), np.concatenate(patch_cols))),
                shape=op.shape,
            )
            op = (op + patch.tocsr()).tocsr()

        return op, rhs

    def _contact_offset_impulse(self, y: np.ndarray, t: float, h: float) -> np.ndarray:
        if self.n_react == 0 or self.contact_offset_force is None:
            return np.zeros(self.n_react, dtype=float)
        source = self.contact_offset_force
        if callable(source):
            for args in ((y, t), (y,), ()):  # constant callbacks in examples use (y, t)
                try:
                    raw = source(*args)
                    break
                except TypeError:
                    continue
            else:
                raw = source(y, t)
        else:
            raw = source
        force = np.asarray(raw, dtype=float).ravel()
        if force.size == 1 and self.n_react != 1:
            force = np.full(self.n_react, float(force[0]), dtype=float)
        if force.size != self.n_react:
            raise ValueError(
                "contact_offset_force must return one value per reaction DOF; "
                f"got {force.size}, expected {self.n_react}"
            )
        return float(h) * force / self.reaction_state_to_reported_scale

    def _theta_system_cached(self, t: float, y: np.ndarray, h: float) -> dict:
        """Factorized theta system for ``(t, y, h)``, memoized per step size.

        The theta operator ``op(h) = (1/theta)A - hJ`` is affine and, at a fixed
        h, byte-identical across steps; only the predictor RHS depends on
        ``(t, y)``.  Entries are cached in an LRU-bounded ``OrderedDict`` keyed by
        ``float(h)`` (cap ``theta_cache_size``), so a lenient adaptive driver that
        revisits earlier step sizes reuses their factorization for free.

        Within a fixed h three cases arise:

        * exact ``(t, y, h)`` re-entry (aux/mu fixed-point iterations re-enter
          ``step()`` with identical arguments): return the cached entry as is --
          operator, factorization, predictor and Delassus block are unchanged.
        * byte-identical operator, new ``(t, y)``: reuse the factorization and
          Delassus block, recompute only the back-substitution.
        * a genuine operator change (time-dependent or nonlinear J) at this h is
          detected by the byte fingerprint and triggers a full rebuild -- no
          cross-step staleness.
        """
        cache = self._theta_cache
        key = float(h)
        entry = cache.get(key)
        # exact (t, y, h) hit: aux fixed-point re-entry, nothing changed at all
        if (
            entry is not None
            and entry["t"] == float(t)
            and np.array_equal(entry["y"], y)
        ):
            cache.move_to_end(key)
            return entry

        op, rhs_base = self._build_theta_system(t, y, h)
        op = op.tocsr() if sp.issparse(op) else sp.csr_matrix(op)

        # Explicit low-rank Jacobian: rhs_jac (hence ``op`` and the affine RHS
        # built in _build_theta_system) omits the ``-U V^T`` feedback term.  The
        # Woodbury solve restores it in the operator; restore it in the predictor
        # RHS too, so both linearize about the SAME J: the affine term needs
        # ``f - J_full y`` rather than ``f - J_sparse y``, i.e. ``+ h U V^T y``.
        if self.theta_lowrank_jac is not None:
            U_lr, V_lr = self.theta_lowrank_jac
            yv = np.asarray(y, dtype=float).ravel()
            rhs_base = rhs_base + float(h) * (U_lr @ (V_lr.T @ yv))

        # Cross-step reuse at this h: reuse the factorization + Delassus block
        # (its per-rung Woodbury rides inside fac) whenever the operator matrix is
        # byte-identical; recompute only the back-substitution.  A genuine
        # operator change at this h fails the fingerprint and rebuilds below.
        reuse = (
            entry is not None
            and entry.get("fac") is not None
            and entry.get("op_shape") == op.shape
            and entry.get("op_nnz") == op.nnz
            and np.array_equal(entry["op_indptr"], op.indptr)
            and np.array_equal(entry["op_indices"], op.indices)
            and np.array_equal(entry["op_data"], op.data)
        )
        if reuse:
            fac = entry["fac"]
            z_pred = fac.solve(rhs_base)
            new_entry = dict(entry)                  # shares fac, X_contact, W
            new_entry.update(t=float(t), y=np.array(y, copy=True),
                             rhs=rhs_base, z_pred=z_pred)
            if self.n_react > 0:
                new_entry["b_soccp"] = np.asarray(self.D_contact @ z_pred, dtype=float).ravel()
                new_entry["u_old"] = np.asarray(self.D_contact @ y, dtype=float).ravel()
            cache[key] = new_entry
            cache.move_to_end(key)
            return new_entry

        # operator changed / cold rung -> (re)factorize and rebuild the Delassus
        # block, dropping this rung's stale factorization first.  Pop the stale
        # entry from the cache BEFORE destroying its factorization: if the
        # rebuild below raises (e.g. _ThetaFactorization construction fails),
        # no cache entry may remain reachable that references a destroyed fac.
        if entry is not None and entry.get("fac") is not None:
            cache.pop(key, None)
            entry["fac"].destroy()
        lowrank = None
        if self.theta_lowrank_jac is not None:
            U_lr, V_lr = self.theta_lowrank_jac
            lowrank = (U_lr, V_lr, float(h))     # op = op_sparse + h U V^T
        self._theta_factorizations += 1
        if self.theta_linear_solver == "iterative":
            fac = _ThetaFactorization(op, "iterative", self.theta_iterative_options,
                                      dense_rows=self.theta_dense_rows, lowrank=lowrank)
        elif self.theta_linear_solver == "petsc_shell":
            # Explicit elif (never the ``else`` fallback): a missing branch would
            # silently route petsc_shell to scipy-splu and corrupt every solve
            # while all gates pass.  The low-rank (U, V, h) term is applied INSIDE
            # the Krylov matvec (MATLRC), so no Woodbury capacitance is formed.
            fac = _ThetaFactorization(op, "petsc_shell", self.theta_petsc_options,
                                      dense_rows=self.theta_dense_rows, lowrank=lowrank)
        else:
            backend = "petsc" if self.theta_linear_solver == "petsc" else "scipy"
            fac = _ThetaFactorization(op, backend, self.theta_petsc_options,
                                      dense_rows=self.theta_dense_rows, lowrank=lowrank)
        z_pred = fac.solve(rhs_base)
        new_entry = {
            "t": float(t),
            "h": float(h),
            "y": np.array(y, copy=True),
            "rhs": rhs_base,
            "fac": fac,
            "z_pred": z_pred,
            "op_shape": op.shape,
            "op_nnz": op.nnz,
            "op_indptr": op.indptr.copy(),
            "op_indices": op.indices.copy(),
            "op_data": op.data.copy(),
        }
        if self.n_react > 0:
            if self._B_dense is None:
                B = self.B.tocsr() if sp.issparse(self.B) else sp.csr_matrix(self.B)
                self._B_dense = np.asarray(B.toarray(), dtype=float)
            X = fac.solve_multi(self._B_dense)
            new_entry["X_contact"] = np.asarray(X, dtype=float)
            new_entry["W"] = np.asarray(self.D_contact @ X, dtype=float)
            new_entry["b_soccp"] = np.asarray(self.D_contact @ z_pred, dtype=float).ravel()
            new_entry["u_old"] = np.asarray(self.D_contact @ y, dtype=float).ravel()
        cache[key] = new_entry
        cache.move_to_end(key)
        # LRU eviction: the current rung is most-recently-used (moved to the
        # end), so popitem(last=False) never evicts it.
        while len(cache) > self.theta_cache_size:
            _evicted_key, evicted = cache.popitem(last=False)
            if evicted.get("fac") is not None:
                evicted["fac"].destroy()
            self._theta_cache_evictions += 1
        return new_entry


    @staticmethod
    def _call_optional_state_fn(fn, z_theta: np.ndarray, t_theta: float):
        if fn is None:
            return None
        if not callable(fn):
            return fn
        for args in ((z_theta, t_theta), (z_theta,), ()): 
            try:
                return fn(*args)
            except TypeError:
                continue
        return fn(z_theta, t_theta)

    def _mu_from_theta_state(self, z_theta: np.ndarray, t_theta: float) -> np.ndarray:
        raw = self._call_optional_state_fn(self.mu_state_callback, z_theta, t_theta)
        mu = np.asarray(raw, dtype=float).ravel()
        if mu.size == 1 and self.n_contacts != 1:
            mu = np.full(self.n_contacts, float(mu[0]), dtype=float)
        if mu.size != self.n_contacts:
            raise ValueError(
                "mu_state_callback must return one value per contact; "
                f"got {mu.size}, expected {self.n_contacts}"
            )
        if np.any(mu < 0.0) or not np.all(np.isfinite(mu)):
            raise ValueError("mu_state_callback returned non-finite or negative friction")
        return mu

    def _solve_contact_petsc_ssn_state_mu(
        self,
        W,
        b: np.ndarray,
        p0: np.ndarray,
        *,
        z_pred: np.ndarray,
        X_contact: np.ndarray,
        offset_impulse: np.ndarray,
        t_theta: float,
        u_N_old: np.ndarray,
    ) -> tuple[np.ndarray, SocppPgsInfo, dict, np.ndarray]:
        """Reduced contact SSN with mu evaluated from z_theta(p).

        ``u_N_old`` is the PRE-STEP normal contact velocity ``(D @ y)[normal]``
        per contact -- the ``u_{N,k}`` of the Fremond restitution offset
        ``(theta(1+e)-1) u_{N,k}``, same datum as the standard (aux-mu) route.
        It must NOT be approximated by the free predictor
        ``D (z_pred - X offset)``: that is the velocity with the entire contact
        reaction (including the prestress hold) removed, which for a
        prestressed resting contact is a large fictitious closing velocity --
        the restitution offset it induces forces spurious slip through the
        De Saxce coupling (a critically prestressed fault in exact equilibrium
        ruptures in one step).
        """
        if sp.issparse(W):
            W_dense = np.asarray(W.toarray(), dtype=float)
        else:
            W_dense = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        p = np.asarray(p0, dtype=float).copy()
        z_pred = np.asarray(z_pred, dtype=float).ravel()
        X_contact = np.asarray(X_contact, dtype=float)
        offset_impulse = np.asarray(offset_impulse, dtype=float).ravel()
        u_N_old = np.asarray(u_N_old, dtype=float).ravel()

        z_free = z_pred - X_contact @ offset_impulse

        def state_from_effective_impulse(p_eff: np.ndarray) -> np.ndarray:
            return z_free + X_contact @ np.asarray(p_eff, dtype=float).ravel()

        # Hoisted fast path for the residual.  mu is NOT hoisted: it is
        # re-evaluated from the trial theta-state at EVERY residual/Jacobian
        # evaluation (that per-evaluation update IS the coupled mu closure).
        # What is hoisted are the genuine per-solve constants -- the
        # restitution offset rest = (theta(1+e)-1) u_N_old (a pre-step datum
        # of the impact law) and the contiguous W/b copies -- so the fused
        # numba kernel is called directly with the fresh mu instead of
        # rebuilding a shift closure + re-copying W/b on every line-search
        # trial (the residual is evaluated O(iters x trials) times per solve).
        _dims = [sl.stop - sl.start for sl in self.block_slices]
        _rest_c = np.ascontiguousarray(
            (self.theta * (1.0 + self.e_N_vec) - 1.0) * u_N_old, dtype=float,
        )
        _use_nb = (
            _NUMBA_OK
            and _soc_fb_assemble_nb is not None
            and self.contact_residual == "soc_fb"
            and b.size > 0
            and bool(_dims)
            and all(dd == _dims[0] for dd in _dims)
        )
        _W_c = np.ascontiguousarray(W_dense) if _use_nb else None
        _b_c = np.ascontiguousarray(b) if _use_nb else None

        def residual_for(p_eff: np.ndarray):
            z_theta = state_from_effective_impulse(p_eff)
            mu_vec = self._mu_from_theta_state(z_theta, t_theta)
            if _use_nb:
                mu_c = np.ascontiguousarray(mu_vec, dtype=float)
                F, _, u_full = _soc_fb_assemble_nb(
                    _W_c, _b_c, np.ascontiguousarray(p_eff, dtype=float),
                    mu_c, mu_c, _rest_c, int(_dims[0]), False, 1.0e-14,
                )
                return F, u_full, mu_vec
            shift_fn = fremond_shift_factory(
                mu_vec, self.e_N_vec, u_N_old, theta=self.theta,
            )
            F, _, u_full = self._contact_ssn_residual_jacobian(
                p_eff, W_dense, b, mu_vec, shift_fn, want_jacobian=False,
            )
            return F, u_full, mu_vec

        def dmu_dp_for(z_theta: np.ndarray) -> np.ndarray:
            if self.dmu_dstate_callback is None:
                return np.zeros((0, p.size), dtype=float)
            raw = self._call_optional_state_fn(
                self.dmu_dstate_callback, z_theta, t_theta,
            )
            grad = np.asarray(raw, dtype=float)
            if grad.ndim == 1 and self.n_contacts == 1:
                grad = grad.reshape(1, -1)
            if grad.shape != (self.n_contacts, self.n_state):
                raise ValueError(
                    "dmu_dstate_callback must return shape "
                    f"({self.n_contacts}, {self.n_state}); got {grad.shape}"
                )
            return np.asarray(grad @ X_contact, dtype=float)

        def residual_and_jacobian_fd(p_eff: np.ndarray):
            F, _u, mu_vec = residual_for(p_eff)
            J = np.empty((F.size, p_eff.size), dtype=float)
            for j in range(p_eff.size):
                eps = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(p_eff[j])))
                p_step = np.array(p_eff, copy=True)
                p_step[j] += eps
                F_step, _, _ = residual_for(p_step)
                J[:, j] = (F_step - F) / eps
            return F, J, mu_vec

        def residual_and_jacobian(p_eff: np.ndarray):
            if self.dmu_dstate_callback is None or self.contact_residual != "soc_fb":
                return residual_and_jacobian_fd(p_eff)
            if _use_nb and _soc_fb_assemble_state_mu_nb is not None:
                # Fused kernel: residual + mu-coupled Jacobian in one pass
                # (identical formulas to the python loop below).
                z_theta = state_from_effective_impulse(p_eff)
                mu_vec = self._mu_from_theta_state(z_theta, t_theta)
                dmu_dp = dmu_dp_for(z_theta)
                mu_c = np.ascontiguousarray(mu_vec, dtype=float)
                F, J, _u = _soc_fb_assemble_state_mu_nb(
                    _W_c, _b_c, np.ascontiguousarray(p_eff, dtype=float),
                    mu_c, mu_c, _rest_c,
                    np.ascontiguousarray(dmu_dp, dtype=float),
                    int(_dims[0]), 1.0e-14,
                )
                return F, J, mu_vec
            F, u_full, mu_vec = residual_for(p_eff)
            z_theta = state_from_effective_impulse(p_eff)
            dmu_dp = dmu_dp_for(z_theta)
            J = np.zeros((F.size, p_eff.size), dtype=float)
            eye_react = np.eye(p_eff.size, dtype=float)
            for k, sl in enumerate(self.block_slices):
                d = sl.stop - sl.start
                mu = float(mu_vec[k])
                if d == 1 or mu <= 1.0e-14:
                    shift_fn = fremond_shift_factory(
                        mu_vec, self.e_N_vec, u_N_old, theta=self.theta,
                    )
                    _, J_const, _ = self._contact_ssn_residual_jacobian(
                        p_eff, W_dense, b, mu_vec, shift_fn,
                        want_jacobian=True,
                    )
                    J[sl, :] = J_const[sl, :]
                    continue

                u = np.asarray(u_full[sl], dtype=float)
                p_block = np.asarray(p_eff[sl], dtype=float)
                W_rows = W_dense[sl, :]
                u_t = u[1:]
                norm_t = float(np.linalg.norm(u_t))
                rest = (self.theta * (1.0 + float(self.e_N_vec[k])) - 1.0) * float(u_N_old[k])
                uhat = u.copy()
                uhat[0] += mu * norm_t + rest

                J_shift = np.eye(d, dtype=float)
                if norm_t > 0.0:
                    J_shift[0, 1:] = mu * u_t / norm_t
                else:
                    J_shift[0, 1:] = mu
                duhat_dmu = np.zeros(d, dtype=float)
                duhat_dmu[0] = norm_t

                T_x = np.eye(d, dtype=float)
                T_x[1:, 1:] *= mu
                T_y = np.eye(d, dtype=float)
                T_y[0, 0] = mu
                T_y_inv = np.eye(d, dtype=float)
                T_y_inv[0, 0] = 1.0 / mu
                dT_x = np.zeros((d, d), dtype=float)
                dT_x[1:, 1:] = np.eye(d - 1)
                dT_y = np.zeros((d, d), dtype=float)
                dT_y[0, 0] = 1.0
                dT_y_inv = np.zeros((d, d), dtype=float)
                dT_y_inv[0, 0] = -1.0 / (mu * mu)

                x_sd = T_x @ uhat
                y_sd = T_y @ p_block
                phi_sd, dphi_dx, dphi_dy = _soc_fb_phi_and_jac(x_sd, y_sd)

                dmu_row = dmu_dp[k]
                E_block = eye_react[sl, :]
                duhat_dp = J_shift @ W_rows + np.outer(duhat_dmu, dmu_row)
                dx_dp = T_x @ duhat_dp + np.outer(dT_x @ uhat, dmu_row)
                dy_dp = T_y @ E_block + np.outer(dT_y @ p_block, dmu_row)
                J[sl, :] = T_y_inv @ (dphi_dx @ dx_dp + dphi_dy @ dy_dp)
                J[sl, :] += np.outer(dT_y_inv @ phi_sd, dmu_row)
            return F, J, mu_vec

        linear_backend = (
            "dense"
            if self.contact_linear_solver == "dense"
            or (self.contact_linear_solver == "auto"
                and b.size <= _CONTACT_DENSE_SOLVE_MAX)
            else "petsc"
        )
        converged = False
        residual_norm = float("inf")
        iters = 0
        for it in range(max(1, self.contact_ssn_max_iter)):
            F, J, mu_vec = residual_and_jacobian(p)
            residual_norm = float(np.linalg.norm(F))
            iters = it + 1
            if residual_norm < self.contact_ssn_tol:
                converged = True
                break
            delta, linear_backend = self._contact_linear_solve(J, -F)
            if self.contact_ssn_line_search:
                alpha = 1.0
                accepted = False
                while alpha >= 1.0e-8:
                    p_trial = p + alpha * delta
                    F_trial, _, _ = residual_for(p_trial)
                    trial_norm = float(np.linalg.norm(F_trial))
                    if np.isfinite(trial_norm) and trial_norm <= (1.0 - 1.0e-4 * alpha) * residual_norm:
                        p = p_trial
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    p_trial = p + delta
                    F_trial, _, _ = residual_for(p_trial)
                    if np.all(np.isfinite(F_trial)):
                        p = p_trial
                    else:
                        break
            else:
                p = p + delta

        F_final, u_full, mu_vec = residual_for(p)
        residual_norm = float(np.linalg.norm(F_final))
        converged = converged or residual_norm < self.contact_ssn_tol
        if not converged and not self.contact_allow_nonconverged:
            raise RuntimeError(
                "PETSc contact SSN with state-dependent mu failed to converge: "
                f"iters={iters}, residual={residual_norm:.3e}"
            )
        info = SocppPgsInfo(
            outer_iters=iters,
            inner_iters=iters,
            converged=converged,
            outer_residual=residual_norm,
            local_residual_max=residual_norm,
        )
        info.regime = [
            _classify_regime(p[sl], u_full[sl], float(mu_vec[k]), tol=self.contact_ssn_tol)
            for k, sl in enumerate(self.block_slices)
        ]
        diag = {
            "contact_solver": "petsc_ssn",
            "contact_residual": self.contact_residual,
            "contact_ssn_iters": iters,
            "contact_ssn_converged": converged,
            "contact_ssn_residual": residual_norm,
            "contact_linear_solver": linear_backend,
            "mu_state_coupled": True,
        }
        return p, info, diag, mu_vec.copy()

    def _solve_contact_active_subset(
        self, active_blocks, W, b_soccp, offset_impulse, p0_eff, mu_vec, u_N_old,
    ):
        """Solve the SOCCP over only the active contacts (gap index set).

        ``active_blocks`` is the list of active contact indices.  The reduced
        problem is built by slicing the cached full Delassus ``W`` and the
        predictor velocity, re-basing the block slices contiguously.  Open (gap-deactivated)
        contacts keep ``p_effective = 0`` and stay out of the cone problem,
        but their prestress release couples into the active rows through the
        Delassus off-diagonal via the full-offset reduced rhs
        ``b_a = (b_soccp - W @ offset)[active]``.  Returns a FULL-length
        ``p_effective`` plus the SOCCP info/diag.
        """
        ab = np.asarray(active_blocks, dtype=int)
        act_idx = np.concatenate([
            np.arange(self.block_slices[k].start, self.block_slices[k].stop)
            for k in ab
        ]).astype(int)
        red_slices = []
        off = 0
        for k in ab:
            d = self.block_slices[k].stop - self.block_slices[k].start
            red_slices.append(slice(off, off + d))
            off += d
        Wm = W.toarray() if sp.issparse(W) else np.asarray(W, dtype=float)
        # Build the reduced rhs from the FULL prestress offset so the active
        # (closed) rows feel the RELEASE of the gap-deactivated rows through the
        # Delassus off-diagonal.  A geometrically-open contact carries zero
        # TOTAL reaction (p_effective = 0, booked below) and hands the bulk the
        # increment p_contact = -offset; through W this perturbs a coupled,
        # still-closed contact's velocity by W[closed, open] @ (-offset[open]).
        # The correct reduced affine term is therefore the full-set rhs
        #   b_eff = b_soccp - W @ offset_impulse    (see the all-active path
        #   in step())
        # restricted to the active rows.  The open rows stay OUT of the cone
        # problem (they are not in act_idx) -- only their prestress *release*
        # couples in.  Zeroing the open rows' offset here would drop that cross
        # term and let a closed contact adjacent to an open prestressed one
        # drift into the wall at an O(h) normal velocity (interpenetration).
        offset_full = np.asarray(offset_impulse, dtype=float)
        b_eff_full = np.asarray(b_soccp, dtype=float) - Wm @ offset_full
        W_a = Wm[np.ix_(act_idx, act_idx)]
        b_a = b_eff_full[act_idx]
        mu_a = np.asarray(mu_vec, dtype=float)[ab]
        eN_a = np.asarray(self.e_N_vec, dtype=float)[ab]
        uN_a = np.asarray(u_N_old, dtype=float)[ab]
        p0_a = np.asarray(p0_eff, dtype=float)[act_idx]
        shift_a = fremond_shift_factory(mu_a, eN_a, uN_a, theta=self.theta)
        if self.contact_solver == "pgs":
            p_a, soccp_info = soccp_pgs(
                W_a, b_a, red_slices, mu_a, shift_fn=shift_a, p0=p0_a,
                max_outer=300, max_inner=30, tol_outer=self.contact_ssn_tol,
                return_info=True,
            )
            contact_diag = {}
        else:
            p_a, soccp_info, contact_diag = self._solve_contact_petsc_ssn(
                W_a, b_a, mu_a, shift_a, p0_a, block_slices=red_slices,
            )
        # Gap-deactivated (open) contacts carry zero TOTAL reaction: p_effective
        # = 0 on their rows, so the increment handed to the bulk is p_contact =
        # -offset (the prestress release), NOT 0.  Only the active rows carry the
        # solved reaction ``p_a``.
        p_effective = np.zeros_like(np.asarray(offset_impulse, dtype=float))
        p_effective[act_idx] = p_a
        # report the regime full-length: inactive contacts -> "inactive"
        red_regime = getattr(soccp_info, "regime", None)
        full_regime = ["inactive"] * self.n_contacts
        if red_regime is not None:
            for j, k in enumerate(ab):
                if j < len(red_regime):
                    full_regime[int(k)] = red_regime[j]
        soccp_info.regime = full_regime
        contact_diag = dict(contact_diag)
        contact_diag["active_contacts"] = [int(k) for k in ab]
        return p_effective, soccp_info, contact_diag

    # ------------------------------------------------------------------
    # Velocity solve / geometric activation / position projection helpers
    # (factored so the combined-projection outer loop can re-run the
    #  velocity solve under an updated active set; with combined_projection
    #  off these are each called exactly once and reproduce the original
    #  single-pass path byte-for-byte.)
    # ------------------------------------------------------------------
    def _gap_value(self, y_state: np.ndarray, t_gap: float) -> np.ndarray:
        g = np.asarray(
            self._call_optional_state_fn(self.gap_callable, y_state, t_gap),
            dtype=float,
        ).ravel()
        if g.size != self.n_contacts:
            raise ValueError(
                f"gap_callable returned {g.size} gaps; expected {self.n_contacts}"
            )
        return g

    def _geometric_active_blocks(self, y_state: np.ndarray, t_gap: float):
        """Gap index set Ibar1_k at ``y_state``: None (all active) or the active
        indices.  Pure geometric activation g(q) <= gap_tol (Eq. 19 less the
        u_{N,k}<=0 term -- see __init__)."""
        if self.gap_callable is None:
            return None
        active_mask = (self._gap_value(y_state, t_gap) <= self.gap_tol)
        if bool(active_mask.all()):
            return None
        return np.flatnonzero(active_mask)

    def _velocity_solve_given_active(
        self, active_blocks, W, b_soccp, b_eff, offset_impulse, p0_eff,
        mu_vec, u_N_old,
    ):
        """Fremond velocity SOCCP for a given active set (None => all contacts)."""
        if active_blocks is None:
            shift_fn = fremond_shift_factory(
                mu_vec, self.e_N_vec, u_N_old, theta=self.theta,
            )
            if self.contact_solver == "pgs":
                p_effective, soccp_info = soccp_pgs(
                    W, b_eff, self.block_slices, mu_vec,
                    shift_fn=shift_fn, p0=p0_eff,
                    max_outer=300, max_inner=30, tol_outer=self.contact_ssn_tol,
                    return_info=True,
                )
                contact_diag = {}
            else:
                p_effective, soccp_info, contact_diag = self._solve_contact_petsc_ssn(
                    W, b_eff, mu_vec, shift_fn, p0_eff,
                )
        elif active_blocks.size == 0:
            # Every contact is geometrically open -> zero total reaction on all
            # rows; p_contact = p_effective - offset = -offset releases the
            # prestress everywhere.
            p_effective = np.zeros_like(np.asarray(offset_impulse, dtype=float))
            soccp_info = SocppPgsInfo(converged=True)
            soccp_info.regime = ["inactive"] * self.n_contacts
            contact_diag = {"active_contacts": []}
        else:
            p_effective, soccp_info, contact_diag = self._solve_contact_active_subset(
                active_blocks, W, b_soccp, offset_impulse, p0_eff, mu_vec, u_N_old,
            )
        return p_effective, soccp_info, contact_diag

    def _gap_jacobian_at(self, y_state: np.ndarray, t_gap: float) -> np.ndarray:
        """The normal-gap Jacobian G = dg/dy as a dense (n_contacts x n_state) array."""
        G = self.gap_jac(y_state, t_gap) if callable(self.gap_jac) else self.gap_jac
        G = G.toarray() if sp.issparse(G) else np.asarray(G, dtype=float)
        if G.shape != (self.n_contacts, self.n_state):
            raise ValueError(
                f"gap_jac must be ({self.n_contacts}, {self.n_state}); got {G.shape}"
            )
        return G

    def _apply_projection_metric_inv(self, M: np.ndarray) -> np.ndarray:
        """Apply B^{-1} (the projection metric inverse) to the columns of ``M``.

        None => identity, so the correction lives in span(G^T): for a gap that
        depends only on position DOFs this moves ONLY those DOFs and leaves the
        velocity DOFs untouched -> no rebound, no kinetic-energy injection."""
        Minv = self.projection_metric_inv
        if Minv is None:
            return M
        if callable(Minv):
            return np.asarray(Minv(M), dtype=float)
        return np.asarray(Minv @ M, dtype=float)

    def _project_position(self, y_new: np.ndarray, t_end: float):
        """GGL position projection: nudge ``y_new`` so g(q_{k+1}) >= 0.

        Solves the unilateral problem  min 1/2 ||dy||_B^2  s.t.  g + G dy >= 0,
        i.e.  dy = B^{-1} G^T nu  with  0 <= g + (G B^{-1} G^T) nu _|_ nu >= 0.
        The multiplier ``nu`` is a *position* projection multiplier, kept entirely
        separate from the contact impulse ``p`` -- it is never used as a reaction
        or turned into a velocity.  Re-linearizes for a (possibly) nonlinear gap;
        for a linear gap it converges in one pass.  Returns (y_proj, diag)."""
        y_proj = np.asarray(y_new, dtype=float).copy()
        pen_before = float(min(0.0, self._gap_value(y_proj, t_end).min()))
        nu_total = np.zeros(self.n_contacts, dtype=float)
        iters = 0
        for it in range(self.combined_projection_max_iter):
            g = self._gap_value(y_proj, t_end)
            if g.min() >= -self.projection_tol:
                break
            G = self._gap_jacobian_at(y_proj, t_end)
            MinvGT = self._apply_projection_metric_inv(G.T)           # (n_state, n_c)
            Delassus = G @ MinvGT                                     # (n_c, n_c) SPD
            nu = _solve_unilateral_lcp(Delassus, g, tol=self.projection_tol)
            y_proj = y_proj + MinvGT @ nu
            nu_total += nu
            iters = it + 1
        pen_after = float(min(0.0, self._gap_value(y_proj, t_end).min()))
        diag = {
            "proj_iters": iters,
            "proj_penetration_before": pen_before,
            "proj_penetration_after": pen_after,
            "proj_multiplier": nu_total,
            "proj_correction_norm": float(np.linalg.norm(y_proj - y_new)),
        }
        return y_proj, diag

    @staticmethod
    def _same_active_blocks(a, b) -> bool:
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        a = np.asarray(a, dtype=int).ravel()
        b = np.asarray(b, dtype=int).ravel()
        return a.size == b.size and bool(np.all(a == b))

    def step(
        self,
        t: float,
        y: np.ndarray,
        aux: dict,
        h: float,
        *,
        return_diagnostic: bool = False,
    ) -> tuple[np.ndarray, dict, dict]:
        if h <= 0.0:
            raise ValueError("step size must be positive")
        y = np.asarray(y, dtype=float).ravel()
        theta_sys = self._theta_system_cached(t, y, h)
        rhs_base = theta_sys["rhs"]
        z_pred = theta_sys["z_pred"]

        if self.n_react > 0:
            W = theta_sys["W"]
            u_old = theta_sys["u_old"]
            b_soccp = theta_sys["b_soccp"]
            p0 = aux.get("p_contact_prev", None)
            p0 = (np.zeros(self.n_react, dtype=float)
                  if p0 is None or len(p0) != self.n_react
                  else np.asarray(p0, dtype=float))
            offset_impulse = self._contact_offset_impulse(
                z_pred, float(t) + self.theta * float(h), h,
            )
            b_eff = b_soccp - W @ offset_impulse
            # Warm start (initial guess only; converged solution unchanged):
            # Warm-start priority (initial guess only; the converged solution is
            # set by the SOCCP, not the guess):
            #   1. mu fixed-point re-entry (identical (t, y, h)): the previous
            #      sweep's converged impulse -- exact h, near-final mu, so the
            #      semismooth Newton starts inside its quadratic basin.
            #   2. previous step's impulse, RESCALED to the current step size:
            #      contact impulses are h*force, so carrying p_contact_prev
            #      across the adaptive full/half sub-steps unscaled leaves the
            #      guess ~2x off in magnitude -- far outside the Newton basin
            #      (globalization then costs ~3x the iterations).
            #   3. the offset impulse (prestress hold).
            reentry = (
                self._last_call_key is not None
                and self._last_call_key[0] == float(t)
                and self._last_call_key[2] == float(h)
                and np.array_equal(self._last_call_key[1], y)
            )
            if (reentry and self._last_p_eff is not None
                    and self._last_p_eff.size == self.n_react):
                p0_eff = self._last_p_eff.copy()
            elif np.any(p0):
                h_prev = aux.get("p_contact_prev_h")
                p0_scale = (
                    float(h) / float(h_prev)
                    if h_prev is not None and float(h_prev) > 0.0 else 1.0
                )
                p0_eff = p0 * p0_scale + offset_impulse
            else:
                p0_eff = offset_impulse.copy()
            t_theta = float(t) + self.theta * float(h)
            X_contact = np.asarray(theta_sys["X_contact"], dtype=float)
            proj_diag = {}
            # Pre-step normal contact velocity u_{N,k} = (D y)[normal] -- the
            # Fremond restitution datum, shared by BOTH mu routes below.
            u_N_old = np.array([u_old[sl.start] for sl in self.block_slices])
            if self.mu_state_callback is not None:
                if self.contact_solver != "petsc_ssn":
                    raise ValueError("state-dependent mu currently requires contact_solver='petsc_ssn'")
                p_effective, soccp_info, contact_diag, mu_vec = self._solve_contact_petsc_ssn_state_mu(
                    W, b_eff, p0_eff,
                    z_pred=z_pred,
                    X_contact=theta_sys["X_contact"],
                    offset_impulse=offset_impulse,
                    t_theta=t_theta,
                    u_N_old=u_N_old,
                )
                p_contact = p_effective - offset_impulse
                # z_theta = A^{-1}(rhs_base + B p_contact) = z_pred + X_contact p_contact
                # by linearity, since z_pred = A^{-1} rhs_base and X_contact = A^{-1} B
                # are already cached -- avoids a full theta-operator back-substitution
                # on every (mu fixed-point) step() call.
                z_theta = z_pred + X_contact @ p_contact
                y_new = y + (z_theta - y) / self.theta
                mu_vec = self._mu_from_theta_state(z_theta, t_theta)
            else:
                mu_vec = np.asarray(aux.get("mu", self.mu_init), dtype=float).ravel()
                if mu_vec.size != self.n_contacts:
                    raise ValueError(
                        f"aux['mu'] has {mu_vec.size} entries; expected {self.n_contacts}"
                    )
                # Geometric activation: gap index set Ibar1_k (Eq. 19), decided at
                # the OLD position q_k = y.  gap_callable is None -> every contact
                # active (persistent contact).  When combined_projection is OFF the
                # loop runs exactly once and is byte-for-byte the original path.
                active_blocks = self._geometric_active_blocks(y, t_theta)
                cp_iters = 0
                cp_converged = not self.combined_projection
                for _cp_it in range(self.combined_projection_max_iter):
                    cp_iters = _cp_it + 1
                    p_effective, soccp_info, contact_diag = self._velocity_solve_given_active(
                        active_blocks, W, b_soccp, b_eff, offset_impulse, p0_eff,
                        mu_vec, u_N_old,
                    )
                    p_contact = p_effective - offset_impulse
                    z_theta = z_pred + X_contact @ p_contact
                    y_new = y + (z_theta - y) / self.theta
                    if not self.combined_projection:
                        break
                    # GGL position projection at the step endpoint, then re-check
                    # the gap index set there.  Re-solve the velocity problem only
                    # if the projected position changed which contacts are active
                    # (the "combined" coupling); otherwise accept the projected y.
                    y_proj, proj_diag = self._project_position(
                        y_new, float(t) + float(h),
                    )
                    new_blocks = self._geometric_active_blocks(
                        y_proj, float(t) + float(h),
                    )
                    y_new = y_proj
                    if self._same_active_blocks(new_blocks, active_blocks):
                        cp_converged = True
                        break
                    active_blocks = new_blocks
                if self.combined_projection:
                    proj_diag = dict(proj_diag)
                    proj_diag["combined_proj_iters"] = cp_iters
                    proj_diag["combined_proj_converged"] = bool(cp_converged)
            self._last_p_eff = np.asarray(p_effective, dtype=float).copy()
            self._last_call_key = (float(t), y.copy(), float(h))
        else:
            p_contact = np.zeros(0, dtype=float)
            p_effective = p_contact.copy()
            offset_impulse = p_contact.copy()
            soccp_info = SocppPgsInfo(converged=True)
            contact_diag = {}
            z_theta = z_pred
            proj_diag = {}
            y_new = y + (z_theta - y) / self.theta
        u_theta = (
            np.asarray(self.D_contact @ z_theta, dtype=float).ravel()
            if self.n_react else np.zeros(0, dtype=float)
        )
        aux_new = self._aux_update_fn(
            aux if "mu" in aux else {**aux, "mu": self.mu_init},
            u_theta, self.block_slices, h, self.aux_law_params,
        )
        aux_new["p_contact_prev"] = p_contact.copy()
        aux_new["p_contact_prev_h"] = float(h)   # impulse scale for warm-start rescaling
        aux_new["u_old_endpoint"] = (
            np.asarray(self.D_contact @ y_new, dtype=float).ravel()
            if self.n_react else np.zeros(0, dtype=float)
        )
        info = {
            "soccp_outer_iters": soccp_info.outer_iters,
            "soccp_inner_iters": soccp_info.inner_iters,
            "soccp_converged": soccp_info.converged,
            "soccp_residual": soccp_info.outer_residual,
            "regime": list(soccp_info.regime),
            "p_contact": p_contact.copy(),
            "p_contact_effective": p_effective.copy(),
            "p_contact_offset": offset_impulse.copy(),
            "u_kt": u_theta.copy(),
            "mu_used": (
                np.asarray(mu_vec, dtype=float).copy()
                if self.n_react else np.zeros(0, dtype=float)
            ),
        }
        info.update(contact_diag)
        if proj_diag:
            info.update(proj_diag)
        return y_new, aux_new, info


# -----------------------------------------------------------------------------
# Adaptive driver
# -----------------------------------------------------------------------------


def solve_mjf_adaptive(
    stepper,
    t_span,
    y0: np.ndarray,
    aux0: Optional[dict] = None,
    *,
    rtol: Any = 1.0e-3,
    atol: Any = 1.0e-6,
    h0: Optional[float] = None,
    h_min: float = 1.0e-10,
    h_max: float = np.inf,
    safety: float = 0.9,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
    error_order: int = 2,
    error_mask: Optional[np.ndarray] = None,
    mu_from_state: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    mu_fixed_point_tol: float = 1.0e-8,
    mu_fixed_point_max_iter: int = 12,
    aux_sync: Optional[Callable[[dict, np.ndarray], dict]] = None,
    max_attempts: int = 100000,
    label: str = "MJF-adaptive",
    on_step: Optional[Callable[..., None]] = None,
    progress_interval_s: Optional[float] = None,
    checkpoint_path=None,
    checkpoint_every_steps: Optional[int] = None,
    checkpoint_every_walltime_s: Optional[float] = None,
    thin_output: int = 1,
    gc_interval: int = 0,
    resume_from=None,
):
    """Adaptive step-doubling driver for Moreau-Jean-Fremond steppers.

    Each attempt takes one full step and two half steps; the weighted RMS
    difference between the refined and coarse endpoint drives a standard
    I-controller, and the refined solution is kept on acceptance (local
    extrapolation).  Works with any object exposing
    ``step(t, y, aux, h) -> (y_new, aux_new, info)`` and a ``theta``
    attribute, in particular :class:`DescriptorMoreauJeanFremondStepper`.

    Parameters
    ----------
    mu_from_state : callable, optional
        ``y -> mu_vec`` evaluating state-dependent friction (e.g.
        slip-weakening on a cumulative-slip block).  When given, every
        (sub)step solves a fixed point so the contact law uses
        ``mu(y_{k+theta})`` rather than a stale coefficient; the converged
        value is reported as ``info['mu_law']``.
    error_mask : (n,) bool array, optional
        State entries included in the error norm.  Use this to exclude
        constrained rows (reaction storage, multiplier blocks) whose
        Richardson difference is not a discretization error.
    aux_sync : callable, optional
        ``(aux, y) -> aux`` applied after each accepted step, for aux entries
        mirrored from the DAE state (e.g. ``aux['cum_slip']``).
    label : str
        Run label used in progress lines and checkpoints.
    on_step : callable, optional
        ``on_step(t, y, aux, info)`` invoked after every accepted step (never
        for rejected attempts).  Exceptions propagate and abort the march.
    progress_interval_s : float, optional
        When set, print a one-line status (accepted/rejected counts,
        simulation time, percent of horizon, step size, elapsed wall time,
        ETA) at most every this many wall-clock seconds, plus a final line.
    checkpoint_path : str or os.PathLike, optional
        Destination ``.npz`` for restart checkpoints (written atomically;
        each write replaces the previous file).  A final checkpoint is
        written when the march completes.
    checkpoint_every_steps : int, optional
        Checkpoint every N accepted steps.
    checkpoint_every_walltime_s : float, optional
        Checkpoint whenever this much wall time passed since the last one
        (combinable with ``checkpoint_every_steps``; first trigger wins).
    thin_output : int, default 1
        Store only every N-th accepted step in the returned ``t``/``y``/``h``
        /``info`` histories (the final state is always stored); the march
        itself is unchanged.  With ``thin_output > 1`` the per-attempt
        ``attempts['records']`` keep only the stored accepted steps --
        use ``attempts['n_accepted']`` / ``attempts['n_rejected']`` (always
        present) for exact counts.
    gc_interval : int, default 0
        Run ``gc.collect()`` every N accepted steps (0 disables).
    resume_from : str, os.PathLike or dict, optional
        Checkpoint path (or dict from
        :func:`~solve_nivp.mjf_run_control.load_mjf_checkpoint`) to restart
        from: the march starts at the checkpointed ``(t, y, aux)`` and runs
        to ``t_span[1]``; ``y0``/``aux0`` are ignored, and ``h0`` defaults to
        the checkpointed step-size proposal.  The returned histories cover
        only this segment.  The step controller's error-history memory is not
        checkpointed (it re-converges within a few steps).

    Returns
    -------
    t : (m+1,) accepted times
    y : (m+1, n) accepted states (refined solutions)
    h : (m,) accepted step sizes
    info : list of m per-step dicts from the refined half step, augmented
        with ``adaptive_error``, ``adaptive_h``, ``mu_law`` (when
        ``mu_from_state`` is given), and ``p_contact_force`` /
        ``p_contact_effective_force`` (impulse over the half step converted
        to reported force units).
    attempts : dict with ``accepted`` flags, per-attempt ``records``, and
        exact ``n_accepted`` / ``n_rejected`` counters.
    """
    t0, t_end = float(t_span[0]), float(t_span[1])
    if t_end <= t0:
        raise ValueError("t_span must be increasing")
    thin_output = max(1, int(thin_output))
    step_index0 = 0
    if resume_from is not None:
        _ckpt = resolve_mjf_checkpoint(resume_from)
        t0 = float(_ckpt["t"])
        y0 = _ckpt["y"]
        aux0 = _ckpt["aux"]
        step_index0 = int(_ckpt.get("step_index", 0))
        if h0 is None:
            _h_next = _ckpt.get("extras", {}).get("h_next")
            h0 = float(_h_next) if _h_next is not None else float(_ckpt["h"])
    y_k = np.asarray(y0, dtype=float).ravel().copy()
    n_state = y_k.size
    aux_k = dict(aux0 or {})

    monitor = None
    if (on_step is not None or progress_interval_s is not None
            or checkpoint_path is not None
            or checkpoint_every_steps is not None
            or checkpoint_every_walltime_s is not None):
        monitor = MJFRunMonitor(
            t_start=t0, t_end=t_end, label=label, driver="adaptive",
            progress_interval_s=progress_interval_s, on_step=on_step,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_every_walltime_s=checkpoint_every_walltime_s,
            step_index0=step_index0)

    # A resumed checkpoint already at (or past) the horizon: return the
    # trivial single-state segment (fresh calls with a bad horizon raised
    # above, unchanged).
    if resume_from is not None:
        _arrived_eps = 10.0 * np.finfo(float).eps * max(abs(t_end), 1.0)
        if t0 >= t_end - _arrived_eps:
            if monitor is not None:
                monitor.finish(t0, y_k, aux_k, h=float(h0),
                               extras={"h_next": float(h0)})
            return (
                np.asarray([t0], dtype=float),
                np.asarray([y_k.copy()], dtype=float),
                np.zeros(0),
                [],
                {"accepted": [], "records": [],
                 "n_accepted": 0, "n_rejected": 0,
                 "thin_output": thin_output},
            )

    atol_arr = np.broadcast_to(np.asarray(atol, dtype=float), (n_state,))
    rtol_arr = np.broadcast_to(np.asarray(rtol, dtype=float), (n_state,))
    if error_mask is None:
        mask = np.ones(n_state, dtype=bool)
    else:
        mask = np.asarray(error_mask, dtype=bool).ravel()
        if mask.size != n_state:
            raise ValueError(
                f"error_mask has {mask.size} entries; expected {n_state}"
            )

    theta = float(getattr(stepper, "theta", 0.5))
    reported_scale = np.asarray(
        getattr(stepper, "reaction_state_to_reported_scale", 1.0), dtype=float,
    )

    def _copy_aux(aux: dict) -> dict:
        return {
            key: (val.copy() if hasattr(val, "copy") else val)
            for key, val in aux.items()
        }

    def _weighted_error(y_ref: np.ndarray, y_coarse: np.ndarray) -> float:
        diff = y_ref - y_coarse
        scale = atol_arr + rtol_arr * np.maximum(np.abs(y_ref), np.abs(y_coarse))
        vals = diff[mask] / np.maximum(scale[mask], 1.0e-300)
        if vals.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(vals * vals)))

    def _step_with_mu(t_loc: float, y_loc: np.ndarray, aux_in: dict, h_step: float):
        if mu_from_state is None:
            y1, aux1, info1 = stepper.step(t_loc, y_loc, aux_in, h_step)
            return y1, aux1, dict(info1)
        mu_guess = np.asarray(mu_from_state(y_loc), dtype=float).ravel()
        err_mu = 0.0
        mu_iter = 0
        y1 = aux1 = info1 = None
        for mu_iter in range(1, max(1, int(mu_fixed_point_max_iter)) + 1):
            aux_try = _copy_aux(aux_in)
            aux_try["mu"] = mu_guess.copy()
            y1, aux1, info1 = stepper.step(t_loc, y_loc, aux_try, h_step)
            y_theta = y_loc + theta * (y1 - y_loc)
            mu_theta = np.asarray(mu_from_state(y_theta), dtype=float).ravel()
            err_mu = (
                float(np.max(np.abs(mu_theta - mu_guess))) if mu_guess.size else 0.0
            )
            mu_guess = mu_theta
            if err_mu <= float(mu_fixed_point_tol):
                break
        info1 = dict(info1)
        info1["mu_law"] = mu_guess.copy()
        info1["mu_fixed_point_iters"] = mu_iter
        info1["mu_fixed_point_error"] = err_mu
        return y1, aux1, info1

    t_hist = [t0]
    y_hist = [y_k.copy()]
    h_hist = []
    info_hist = []
    attempt_records = []
    n_accepted = 0
    n_rejected = 0
    # Thinning bookkeeping: index of the last accepted step actually stored,
    # and a one-slot buffer of the most recent accepted step so the final
    # state can be stored even when it falls off the thinning grid.  The
    # buffer (with its state copy) is only maintained when thinning is on.
    _last_stored_step = 0
    _last_entry = None
    expo = -1.0 / (float(error_order) + 1.0)
    if h0 is None:
        h0 = min(float(h_max), (t_end - t0) / 100.0)
    h_try = min(float(h0), float(h_max), t_end - t0)
    attempts = 0
    last_exc = None
    # Scaled by |t_end| so accumulated t drift near a large horizon still counts
    # as "arrived" (an absolute 10*eps is far below one ULP of t_end ~ 1e2).
    _end_eps = 10.0 * np.finfo(float).eps * max(abs(t_end), 1.0)
    t_cur = t0
    while t_cur < t_end - _end_eps:
        if attempts >= int(max_attempts):
            raise RuntimeError(
                f"adaptive MJF loop exceeded {max_attempts} attempts"
            )
        attempts += 1
        t_k = float(t_cur)
        h_k = min(float(h_try), t_end - t_k, float(h_max))
        h_k = max(h_k, min(float(h_min), t_end - t_k))
        # Horizon-snap: if a full step would leave only a sub-step sliver before
        # t_end (float drift), fold it into this step so the next iteration does
        # not take a degenerate micro-step.  MJF works in impulses (h*force), so a
        # micro final step (h ~ 1e-13) makes the reported reaction ~ impulse/h
        # blow up (garbage final row) and can stall the cone solve.
        _rem = (t_end - t_k) - h_k
        if 0.0 < _rem < 1.0e-3 * h_k:
            h_k = t_end - t_k
        aux_start = _copy_aux(aux_k)
        try:
            y_full, _aux_full, _info_full = _step_with_mu(t_k, y_k, aux_start, h_k)
            h_half = 0.5 * h_k
            y_mid, aux_mid, _info_mid = _step_with_mu(t_k, y_k, aux_start, h_half)
            y_ref, aux_ref, info_ref = _step_with_mu(
                t_k + h_half, y_mid, aux_mid, h_half,
            )
            err = _weighted_error(y_ref, y_full)
            finite = np.isfinite(err) and bool(np.all(np.isfinite(y_ref)))
        except Exception as exc:  # noqa: BLE001 — converted into a rejection
            y_ref = aux_ref = info_ref = None
            err = float("inf")
            finite = False
            last_exc = exc
        accepted = finite and (err <= 1.0 or h_k <= float(h_min) * (1.0 + 1e-12))
        _rec = {"accepted": bool(accepted), "t": t_k, "h": h_k, "error": float(err)}
        if accepted:
            n_accepted += 1
            t_cur = t_k + h_k
            y_k = y_ref
            aux_k = aux_sync(aux_ref, y_ref) if aux_sync is not None else aux_ref
            info_aug = dict(info_ref)
            info_aug["adaptive_error"] = float(err)
            info_aug["adaptive_h"] = float(h_k)
            p_c = np.asarray(info_aug.get("p_contact", np.zeros(0)), dtype=float)
            p_e = np.asarray(
                info_aug.get("p_contact_effective", np.zeros(0)), dtype=float,
            )
            if p_c.size:
                info_aug["p_contact_force"] = p_c * reported_scale / h_half
                info_aug["p_contact_effective_force"] = p_e * reported_scale / h_half
            _store = (n_accepted % thin_output == 0)
            if _store:
                t_hist.append(t_cur)
                y_hist.append(y_k.copy())
                h_hist.append(h_k)
                info_hist.append(info_aug)
                _last_stored_step = n_accepted
            if thin_output == 1 or _store:
                attempt_records.append(_rec)
            if thin_output > 1:
                _last_entry = (n_accepted, t_cur, y_k.copy(), h_k, info_aug, _rec)
            if err <= 1.0e-16:
                factor = float(max_factor)
            else:
                factor = float(safety) * err ** expo
                factor = min(float(max_factor), max(float(min_factor), factor))
            h_try = min(float(h_max), max(float(h_min), h_k * factor))
            if monitor is not None:
                monitor.after_step(t_cur, y_k, aux_k, info_aug, h=h_k,
                                   extras={"h_next": float(h_try)})
            if gc_interval > 0 and n_accepted % gc_interval == 0:
                gc.collect()
        else:
            n_rejected += 1
            if thin_output == 1:
                attempt_records.append(_rec)
            if not finite and h_k <= float(h_min) * (1.0 + 1e-12):
                raise RuntimeError(
                    f"MJF step failed at h_min={h_k}"
                ) from last_exc
            if not np.isfinite(err) or err <= 0.0:
                factor = float(min_factor)
            else:
                factor = float(safety) * err ** expo
                factor = min(0.8, max(float(min_factor), factor))
            h_try = max(float(h_min), h_k * factor)
            if monitor is not None:
                monitor.after_reject(t_k, h_k)

    # With thinning on, the final accepted state may fall off the storage
    # grid; append it (and its attempt record) from the buffer so the
    # trajectory always ends at t_end.
    if _last_entry is not None and _last_entry[0] != _last_stored_step:
        _, _t_last, _y_last, _h_last, _info_last, _rec_last = _last_entry
        t_hist.append(_t_last)
        y_hist.append(_y_last)
        h_hist.append(_h_last)
        info_hist.append(_info_last)
        attempt_records.append(_rec_last)
    if monitor is not None:
        monitor.finish(t_cur, y_k, aux_k, h=float(h_try),
                       extras={"h_next": float(h_try)})
    return (
        np.asarray(t_hist, dtype=float),
        np.asarray(y_hist, dtype=float),
        np.asarray(h_hist, dtype=float),
        info_hist,
        {
            "accepted": [rec["accepted"] for rec in attempt_records],
            "records": attempt_records,
            "n_accepted": n_accepted,
            "n_rejected": n_rejected,
            "thin_output": thin_output,
        },
    )


# -----------------------------------------------------------------------------
# Ratio-mode adaptive driver: reuse AdaptiveStepping instead of hand-rolling a
# controller.  A thin adapter presents the MJF stepper under the integrator
# contract .step(fun, t, y, h) -> (y_new, fk, solver_err, ok, iters); the
# generic AdaptiveStepping(mode="ratio") owns step-size control (Gustafsson /
# Soderlind digital filter with the LENIENT ratio-band acceptance), DAE-aware
# error weighting, and the active-set filter.  A HOLD layer above the
# controller keeps the committed step size FIXED while the controller's
# proposal drifts inside a band, so the per-h theta factorization cache
# is reused: factorizations ~= number of distinct committed h << steps.
# -----------------------------------------------------------------------------


def _mass_is_identity(A) -> bool:
    """True iff mass matrix ``A`` is the identity (all DOFs differential).

    Sparse-safe: never densifies (an ``A.toarray()`` is n^2 memory and OOMs at
    production sizes).  A false negative only routes through
    ``_detect_algebraic_dofs``, which then finds no zero rows and returns the
    same all-differential mask, so the exact (rather than allclose) comparison
    is safe.
    """
    A_csr = A.tocsr() if sp.issparse(A) else sp.csr_matrix(np.asarray(A, dtype=float))
    n, m = A_csr.shape
    if n != m:
        return False
    diff = (A_csr - sp.identity(n, format="csr")).tocsr()
    diff.eliminate_zeros()
    return diff.nnz == 0


def _mjf_velocity_dof_map(stepper) -> list:
    """Per-contact-block map to the coupled *state* velocity DOF indices.

    Uses ``stepper.D_contact``, whose rows are already permuted into reaction
    (block-slice) order via ``contact_velocity_rows`` — indexing ``D_extract``
    directly by block-slice positions is WRONG whenever the contact velocity
    rows interleave (e.g. crack plants with rows ``[k, n_c + k]`` for contact
    ``k``): the filter would silently suppress OTHER contacts' DOFs.  Row
    access is CSR-native (no densification).
    """
    D = stepper.D_contact
    D_csr = D.tocsr() if sp.issparse(D) else sp.csr_matrix(np.asarray(D, dtype=float))
    vmap = []
    for sl in stepper.block_slices:
        idxs: set[int] = set()
        for r in range(sl.start, sl.stop):
            if r < D_csr.shape[0]:
                lo, hi = D_csr.indptr[r], D_csr.indptr[r + 1]
                cols = D_csr.indices[lo:hi]
                vals = D_csr.data[lo:hi]
                idxs.update(int(c) for c, v in zip(cols, vals) if v != 0.0)
        vmap.append(np.array(sorted(idxs), dtype=int))
    return vmap


class _MJFRegimeProjection:
    """Projection-shaped regime tracker for AdaptiveStepping's active-set filter.

    The MJF SOCCP reports a per-contact regime label ('separation'|'stick'|
    'slip') in each step's ``info['regime']``.  The adapter pushes the latest
    label list here; :meth:`regime_snapshot` / :meth:`regime_changed_mask` then
    mirror :class:`MuScaledSOCProjection`'s active-set-filter contract so the
    error estimator can suppress velocity DOFs whose contact regime flipped
    across a step (the discontinuous constraint-force jump is not a
    discretization error and must not force a rejection).
    """

    def __init__(self, velocity_dof_map: list):
        self._velocity_dof_map = velocity_dof_map
        self._regime = None  # latest list[str] or None until first step

    def update_regime(self, regime) -> None:
        if regime is not None:
            self._regime = list(regime)

    def regime_snapshot(self):
        return None if self._regime is None else list(self._regime)

    def regime_changed_mask(self, prev_snapshot, n_dof):
        if prev_snapshot is None or self._regime is None:
            return None
        cur = self._regime
        n_blk = min(len(prev_snapshot), len(cur))
        changed = [k for k in range(n_blk) if prev_snapshot[k] != cur[k]]
        if not changed:
            return None
        mask = np.ones(int(n_dof), dtype=float)
        for k in changed:
            if k < len(self._velocity_dof_map):
                for idx in self._velocity_dof_map[k]:
                    if 0 <= idx < n_dof:
                        mask[idx] = 0.0
        return mask


class _MJFSolverShim:
    """Minimal ``solver`` object so AdaptiveStepping can reach ``.proj``.

    ``AdaptiveStepping`` reads ``integrator.solver.proj`` for the active-set
    filter and pokes a few cache attributes on nonlinear failure; those pokes
    are inert for MJF (the theta cache is keyed by h, so a retry at a different
    h never reuses a stale factorization).
    """

    def __init__(self, proj):
        self.proj = proj
        self._lu = None
        self._lu_shape = None
        self._J_cross_call = None
        self._petsc_needs_matrix_update = False


class _MJFRatioAdapter:
    """Adapt a ``DescriptorMoreauJeanFremondStepper`` to the integrator contract.

    ``step(fun, t, y, h) -> (y_new, info, solver_err, ok, iters)`` ignores
    ``fun`` (MJF carries its own descriptor dynamics), runs the contact solve
    (plus an optional outer mu fixed point, exactly as ``solve_mjf_adaptive``),
    and reports ``ok`` from the SSN/SOCCP outcome so the controller can shrink
    on a failed solve.  ``info`` rides back as the ``fk`` slot of the contract
    so the driver recovers the accepted step's reactions.

    The controller's Richardson estimator calls ``step`` three times per
    attempt (one full + two half steps).  The MJF ``aux`` state (cumulative
    slip, warm-start impulse, friction) must thread through those sub-steps:
    the full and first-half sub-steps start from the committed aux, and the
    second-half sub-step must start from the aux the first half produced.  This
    is done with a per-attempt ``{(t, y) -> aux}`` map seeded at the committed
    state; each completed sub-step records its endpoint aux, so the second
    half's start ``(t + h/2, y_half)`` resolves to the first half's output
    (a byte-exact key match, since the controller forwards the same array).
    """

    has_embedded_error = False  # force the Richardson (step-doubling) path

    def __init__(self, stepper, *, mu_from_state=None, slip_weakening=None,
                 mu_fixed_point_tol: float = 1.0e-10,
                 mu_fixed_point_max_iter: int = 30):
        self.stepper = stepper
        self.theta = float(getattr(stepper, "theta", 0.5))
        if mu_from_state is not None and slip_weakening is not None:
            raise ValueError(
                "pass at most one of mu_from_state / slip_weakening"
            )
        self.mu_from_state = mu_from_state
        # slip_weakening: explicit-slip route for the causal DRIVER build, where
        # the stepper's slip rows are identity/rhs=0 so a mu_from_state(y) that
        # reads a state slip row would FREEZE mu.  Here the slip is integrated
        # explicitly from the tangential velocity (s += h|v_t,theta|) and written
        # back into the slip rows, reproducing MJFIntegrationMethod._step_weakening.
        self.slip_weakening = dict(slip_weakening) if slip_weakening else None
        self.mu_fixed_point_tol = float(mu_fixed_point_tol)
        self.mu_fixed_point_max_iter = int(mu_fixed_point_max_iter)

        # DAE-aware error weighting reads these off the integrator.
        self.A = stepper.A
        self.use_identity = _mass_is_identity(stepper.A)

        # Active-set filter reaches ``.solver.proj``.
        self._proj = _MJFRegimeProjection(_mjf_velocity_dof_map(stepper))
        self.solver = _MJFSolverShim(self._proj)

        # aux threading across the 3 Richardson sub-steps of one attempt
        self._committed_aux: dict = {}
        self._aux_map: dict = {}
        self._last_out_y = None
        self._last_out_aux = None
        self._last_out_info = None
        self._last_exc = None

    # -- aux threading -------------------------------------------------------
    @staticmethod
    def _key(t, y):
        return (float(t), np.ascontiguousarray(y, dtype=float).tobytes())

    def reset_committed(self, t, y, aux) -> None:
        """Reset the transient aux map to a single committed ``(t, y) -> aux``.

        Called by the driver between controller steps: on acceptance with the
        newly advanced state, on rejection with the unchanged state.
        """
        self._committed_aux = _copy_aux_dict(aux)
        self._aux_map = {self._key(t, y): _copy_aux_dict(aux)}

    def accepted_aux(self):
        """Aux of the accepted refined solution (the last sub-step's output)."""
        return _copy_aux_dict(self._last_out_aux)

    # -- mu fixed point (mirrors solve_mjf_adaptive._step_with_mu) -----------
    def _solve_with_mu(self, t, y, aux_in, h):
        if self.mu_from_state is None:
            y1, aux1, info1 = self.stepper.step(t, y, _copy_aux_dict(aux_in), h)
            return y1, aux1, dict(info1)
        mu_guess = np.asarray(self.mu_from_state(y), dtype=float).ravel()
        err_mu = 0.0
        mu_iter = 0
        y1 = aux1 = info1 = None
        for mu_iter in range(1, max(1, self.mu_fixed_point_max_iter) + 1):
            aux_try = _copy_aux_dict(aux_in)
            aux_try["mu"] = mu_guess.copy()
            y1, aux1, info1 = self.stepper.step(t, y, aux_try, h)
            y_theta = y + self.theta * (y1 - y)
            mu_theta = np.asarray(self.mu_from_state(y_theta), dtype=float).ravel()
            err_mu = (
                float(np.max(np.abs(mu_theta - mu_guess))) if mu_guess.size else 0.0
            )
            mu_guess = mu_theta
            if err_mu <= self.mu_fixed_point_tol:
                break
        info1 = dict(info1)
        info1["mu_law"] = mu_guess.copy()
        info1["mu_fixed_point_iters"] = mu_iter
        info1["mu_fixed_point_error"] = err_mu
        info1["mu_fp_converged"] = bool(err_mu <= self.mu_fixed_point_tol)
        return y1, aux1, info1

    # -- slip-weakening (explicit slip, mirrors MJFIntegrationMethod._step_weakening)
    def _solve_slip_weakening(self, t, y, aux_in, h):
        sw = self.slip_weakening
        slip_slice = sw["slip_slice"]
        mu_from_slip = sw["mu_from_slip"]
        vel_t_extract = sw["vel_t_extract"]
        n_phys = int(sw["n_phys"])
        s_k = np.asarray(y[slip_slice], dtype=float).ravel()
        mu_guess = np.asarray(mu_from_slip(s_k), dtype=float).ravel()
        v_t_theta = np.zeros_like(mu_guess)
        y1 = aux1 = info1 = None
        last_resid = np.inf
        n_iter = 0
        for n_iter in range(1, max(1, self.mu_fixed_point_max_iter) + 1):
            aux_try = _copy_aux_dict(aux_in)
            aux_try["mu"] = mu_guess.copy()
            y1, aux1, info1 = self.stepper.step(t, y, aux_try, h)
            y_theta = y + self.theta * (y1 - y)
            v_t_theta = np.asarray(
                vel_t_extract @ y_theta[:n_phys], dtype=float,
            ).ravel()
            s_theta = s_k + self.theta * h * np.abs(v_t_theta)
            mu_theta = np.asarray(mu_from_slip(s_theta), dtype=float).ravel()
            last_resid = (
                float(np.linalg.norm(mu_theta - mu_guess, ord=np.inf))
                if mu_guess.size else 0.0
            )
            mu_guess = mu_theta
            if last_resid <= self.mu_fixed_point_tol:
                break
        # explicit slip integral, written back into the (identity/rhs=0) slip rows
        s_next = s_k + h * np.abs(v_t_theta)
        y1 = np.asarray(y1, dtype=float).ravel().copy()
        y1[slip_slice] = s_next
        aux1 = _copy_aux_dict(aux1)
        mu_next = np.asarray(mu_from_slip(s_next), dtype=float).ravel()
        aux1["mu"] = mu_next
        aux1["cum_slip"] = s_next.copy()
        info1 = dict(info1)
        info1["mu_law"] = mu_next
        info1["mu_fixed_point_iters"] = int(n_iter)
        info1["mu_fixed_point_error"] = last_resid
        info1["mu_fp_converged"] = bool(last_resid <= self.mu_fixed_point_tol)
        return y1, aux1, info1

    # -- integrator contract -------------------------------------------------
    def step(self, fun, t, y, h):
        y = np.asarray(y, dtype=float).ravel()
        aux_in = self._aux_map.get(self._key(t, y), self._committed_aux)
        try:
            if self.slip_weakening is not None:
                y1, aux1, info = self._solve_slip_weakening(
                    float(t), y, aux_in, float(h))
            else:
                y1, aux1, info = self._solve_with_mu(float(t), y, aux_in, float(h))
        except Exception as exc:  # noqa: BLE001 — surfaced as a rejection
            self._last_exc = exc
            return y, None, float("inf"), False, 0

        y1 = np.asarray(y1, dtype=float).ravel()
        finite = bool(np.all(np.isfinite(y1)))
        soccp_ok = bool(info.get("soccp_converged", True))
        ssn_ok = bool(info.get("contact_ssn_converged", True))
        # mu-gate: production MJFIntegrationMethod.step fails the step when the
        # friction fixed point has not converged; mirror that here so the
        # controller shrinks rather than accepting a stale-mu state.
        mu_ok = bool(info.get("mu_fp_converged", True))
        ok = finite and soccp_ok and ssn_ok and mu_ok
        solver_err = float(info.get("soccp_residual", 0.0) or 0.0)
        iters = int(info.get("soccp_outer_iters", 0) or 0)

        # record this sub-step's endpoint aux for the next sub-step, and remember
        # the refined output so the driver can commit it on acceptance.
        self._aux_map[self._key(float(t) + float(h), y1)] = _copy_aux_dict(aux1)
        self._last_out_y = y1.copy()
        self._last_out_aux = _copy_aux_dict(aux1)
        self._last_out_info = dict(info)
        self._proj.update_regime(info.get("regime"))
        return y1, info, solver_err, ok, iters


def _copy_aux_dict(aux: dict) -> dict:
    return {
        key: (val.copy() if hasattr(val, "copy") else val)
        for key, val in (aux or {}).items()
    }


def solve_mjf_adaptive_ratio(
    stepper,
    t_span,
    y0: np.ndarray,
    aux0: Optional[dict] = None,
    *,
    rtol: Any = 1.0e-3,
    atol: Any = 1.0e-6,
    h0: Optional[float] = None,
    h_min: float = 1.0e-10,
    h_max: float = np.inf,
    error_order: int = 2,
    error_mask: Optional[np.ndarray] = None,
    mu_from_state: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    slip_weakening: Optional[dict] = None,
    mu_fixed_point_tol: float = 1.0e-10,
    mu_fixed_point_max_iter: int = 30,
    aux_sync: Optional[Callable[[dict, np.ndarray], dict]] = None,
    max_attempts: int = 100000,
    # --- ratio controller / hold layer ---
    controller: str = "H211b",
    r_min: float = 0.8,
    r_max: float = 1.2,
    b_param: float = 2.0,
    dae_var_weight: str = "auto",
    active_set_filter: bool = True,
    hold_threshold: float = 0.2,
    quiescence: Optional[Callable[[float, np.ndarray], bool]] = None,
    record_attempts: bool = False,
    label: str = "MJF-ratio",
    on_step: Optional[Callable[..., None]] = None,
    progress_interval_s: Optional[float] = None,
    checkpoint_path=None,
    checkpoint_every_steps: Optional[int] = None,
    checkpoint_every_walltime_s: Optional[float] = None,
    thin_output: int = 1,
    gc_interval: int = 0,
    resume_from=None,
):
    """Ratio-mode adaptive driver for Moreau-Jean-Fremond steppers.

    Delegates step-size control to :class:`~solve_nivp.adaptive_integrator.AdaptiveStepping`
    in ``mode="ratio"`` (Gustafsson / Soderlind digital filter with lenient
    ratio-band acceptance, DAE-aware error weighting, active-set filter).  A
    HOLD layer above the controller keeps the *committed* step size fixed while
    the controller's proposal drifts within ``hold_threshold`` of it, so the
    per-h theta factorization cache is reused and the number of factorizations
    tracks the number of *distinct* committed step sizes rather than the number
    of steps.

    Additive to :func:`solve_mjf_adaptive` (unchanged): same plant, aux, and
    mu-fixed-point contract, plus the reaction / horizon-landing bookkeeping.

    Parameters mirror :func:`solve_mjf_adaptive`; extras:

    controller : digital-filter preset (default ``"H211b"``).
    r_min, r_max : ratio-band acceptance bounds (``rho_prop`` inside is accepted).
    hold_threshold : commit a new step size only when the ACCUMULATED proposal
        drift since the last commit, ``prod_k(h_next_k / h_held)``, leaves the
        band ``|drift - 1| <= hold_threshold`` (default 0.2 = 20%); between
        commits the held factorization is reused.  Accumulation matters: the
        controller clamps each accepted-step ratio to ``[r_min, r_max]``, so a
        single-step test against a threshold >= ``r_max - 1`` would never fire
        and h would be stuck at h0.
    mu_fixed_point_tol, mu_fixed_point_max_iter : friction fixed-point control;
        defaults (1e-10, 30) align with the production ``MJFIntegrationMethod``
        (stricter than ``solve_mjf_adaptive``'s 1e-8/12).  Non-convergence
        FAILS the attempt (mu-gate), as in ``MJFIntegrationMethod.step``.
    slip_weakening : optional dict ``{slip_slice, mu_from_slip, vel_t_extract,
        n_phys}`` selecting the explicit-slip friction route (mutually exclusive
        with ``mu_from_state``).  Use this for the causal DRIVER build where the
        stepper's slip rows are identity/rhs=0: a ``mu_from_state(y)`` reading a
        state slip row would FREEZE mu, whereas this route integrates the slip
        explicitly from the tangential velocity (``s += h|v_t,theta|``) and writes
        it back, so ``mu(s)`` actually evolves (reproduces
        ``MJFIntegrationMethod._step_weakening``).
    dae_var_weight : ``"auto"`` excludes zero-mass (algebraic) DOFs from the
        error norm.
    active_set_filter : suppress regime-transition velocity DOFs from the norm.
    quiescence : optional ``(t, y) -> bool``; when True the attempt proposes
        ``h_max`` (the controller and ratio band still gate acceptance).
    label, on_step, progress_interval_s, checkpoint_path,
    checkpoint_every_steps, checkpoint_every_walltime_s, thin_output,
    gc_interval, resume_from : run-control options with the same semantics as
        :func:`solve_mjf_adaptive` (progress heartbeat, per-accepted-step user
        callback, periodic atomic restart checkpoints, history thinning with
        the final state always stored, periodic ``gc.collect()``, and restart
        from a checkpoint).  Ratio-specific notes: the checkpoint stores the
        *held/committed* step size, so a resumed march continues at the held
        ``h`` (``h0`` overrides it); the digital filter's internal memory and
        the hold-layer drift accumulator are not checkpointed and re-converge
        within a few steps after resume.

    Returns ``(t, y, h, info, attempts)`` as :func:`solve_mjf_adaptive`, with
    ``attempts`` additionally carrying ``n_factorizations`` (theta refactor
    count over the march), ``distinct_committed_h``, ``n_accepted`` /
    ``n_rejected`` exact counters, and, when ``record_attempts``, the
    controller ``attempt_log``.
    """
    from .adaptive_integrator import AdaptiveStepping

    t0, t_end = float(t_span[0]), float(t_span[1])
    if t_end <= t0:
        raise ValueError("t_span must be increasing")
    thin_output = max(1, int(thin_output))
    step_index0 = 0
    if resume_from is not None:
        _ckpt = resolve_mjf_checkpoint(resume_from)
        t0 = float(_ckpt["t"])
        y0 = _ckpt["y"]
        aux0 = _ckpt["aux"]
        step_index0 = int(_ckpt.get("step_index", 0))
        if h0 is None:
            _h_held = _ckpt.get("extras", {}).get("h_held")
            h0 = float(_h_held) if _h_held is not None else float(_ckpt["h"])
    y_k = np.asarray(y0, dtype=float).ravel().copy()
    n_state = y_k.size
    aux_k = dict(aux0 or {})

    monitor = None
    if (on_step is not None or progress_interval_s is not None
            or checkpoint_path is not None
            or checkpoint_every_steps is not None
            or checkpoint_every_walltime_s is not None):
        monitor = MJFRunMonitor(
            t_start=t0, t_end=t_end, label=label, driver="ratio",
            progress_interval_s=progress_interval_s, on_step=on_step,
            checkpoint_path=checkpoint_path,
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_every_walltime_s=checkpoint_every_walltime_s,
            step_index0=step_index0)

    # A resumed checkpoint already at (or past) the horizon: return the
    # trivial single-state segment (fresh calls with a bad horizon raised
    # above, unchanged).
    if resume_from is not None:
        _arrived_eps = 10.0 * np.finfo(float).eps * max(abs(t_end), 1.0)
        if t0 >= t_end - _arrived_eps:
            if monitor is not None:
                monitor.finish(t0, y_k, aux_k, h=float(h0),
                               extras={"h_held": float(h0)})
            attempts_out = {
                "accepted": [], "records": [], "n_factorizations": 0,
                "distinct_committed_h": [], "n_attempts": 0,
                "n_accepted": 0, "n_rejected": 0,
                "thin_output": thin_output,
            }
            if record_attempts:
                attempts_out["attempt_log"] = None
            return (
                np.asarray([t0], dtype=float),
                np.asarray([y_k.copy()], dtype=float),
                np.zeros(0),
                [],
                attempts_out,
            )

    reported_scale = np.asarray(
        getattr(stepper, "reaction_state_to_reported_scale", 1.0), dtype=float,
    )

    adapter = _MJFRatioAdapter(
        stepper,
        mu_from_state=mu_from_state,
        slip_weakening=slip_weakening,
        mu_fixed_point_tol=mu_fixed_point_tol,
        mu_fixed_point_max_iter=mu_fixed_point_max_iter,
    )

    if h0 is None:
        h0 = min(float(h_max), (t_end - t0) / 100.0)
    h0 = min(float(h0), float(h_max), t_end - t0)

    ctrl = AdaptiveStepping(
        integrator=adapter,
        atol=atol,
        rtol=rtol,
        h0=float(h0),
        h_min=float(h_min),
        h_max=float(h_max),
        method_order=int(error_order),
        mode="ratio",
        controller=controller,
        b_param=float(b_param),
        r_min=float(r_min),
        r_max=float(r_max),
        dae_var_weight=dae_var_weight,
        active_set_filter=bool(active_set_filter),
        record_attempts=bool(record_attempts),
    )

    # Honour an explicit error_mask by folding it into the (auto) DAE weight:
    # excluded rows get weight 0 in BOTH the error accumulation and the DOF
    # count, exactly as solve_mjf_adaptive's boolean mask does.
    if error_mask is not None:
        mask_f = np.asarray(error_mask, dtype=bool).ravel().astype(float)
        if mask_f.size != n_state:
            raise ValueError(
                f"error_mask has {mask_f.size} entries; expected {n_state}"
            )
        auto = ctrl._ensure_dae_mask(n_state).copy()
        ctrl._dae_mask = auto * mask_f

    t_hist = [t0]
    y_hist = [y_k.copy()]
    h_hist = []
    info_hist = []
    attempt_records = []
    n_accepted = 0
    n_rejected = 0
    # Thinning bookkeeping (same contract as solve_mjf_adaptive): last stored
    # accepted-step index plus a one-slot buffer of the newest accepted step,
    # maintained only when thinning is on.
    _last_stored_step = 0
    _last_entry = None
    distinct_h: set[float] = set()
    n_fac_start = int(getattr(stepper, "_theta_factorizations", 0))

    t = t0
    h_held = float(h0)                 # committed / reused step size
    drift = 1.0                        # accumulated proposal drift since last commit
    adapter.reset_committed(t, y_k, aux_k)
    _end_eps = 10.0 * np.finfo(float).eps * max(abs(t_end), 1.0)
    _snap_frac = 1.0e-3
    attempts = 0

    while t < t_end - _end_eps:
        if attempts >= int(max_attempts):
            raise RuntimeError(
                f"adaptive-ratio MJF loop exceeded {max_attempts} attempts"
            )
        attempts += 1

        # committed step, clamped to the horizon; quiescence proposes h_max.
        h_step = min(h_held, float(h_max), t_end - t)
        quiesced = quiescence is not None and bool(quiescence(t, y_k))
        if quiesced:
            h_step = min(float(h_max), t_end - t)
        h_step = max(h_step, min(float(h_min), t_end - t))
        # horizon snap: fold a sub-step sliver into this step (no micro-steps).
        _rem = (t_end - t) - h_step
        if 0.0 < _rem < _snap_frac * h_step:
            h_step = t_end - t

        distinct_h.add(float(h_step))
        y_new, info, h_next, E_curr, success, solver_err, iters = ctrl.step(
            stepper.rhs_callable, t, y_k, h_step,
        )
        accepted = bool(success) and bool(np.all(np.isfinite(y_new)))
        _rec = {"accepted": accepted, "t": float(t), "h": float(h_step),
                "h_committed": float(h_held), "error": float(E_curr)}

        if accepted:
            n_accepted += 1
            aux_ref = adapter.accepted_aux()
            if aux_sync is not None:
                aux_ref = aux_sync(aux_ref, y_new)
            info_aug = dict(info)
            info_aug["adaptive_error"] = float(E_curr)
            info_aug["adaptive_h"] = float(h_step)
            info_aug["adaptive_h_committed"] = float(h_held)
            p_c = np.asarray(info_aug.get("p_contact", np.zeros(0)), dtype=float)
            p_e = np.asarray(
                info_aug.get("p_contact_effective", np.zeros(0)), dtype=float,
            )
            if p_c.size:
                h_half = 0.5 * h_step
                info_aug["p_contact_force"] = p_c * reported_scale / h_half
                info_aug["p_contact_effective_force"] = p_e * reported_scale / h_half

            t = t + h_step
            y_k = np.asarray(y_new, dtype=float).ravel().copy()
            aux_k = aux_ref
            _store = (n_accepted % thin_output == 0)
            if _store:
                t_hist.append(t)
                y_hist.append(y_k.copy())
                h_hist.append(float(h_step))
                info_hist.append(info_aug)
                _last_stored_step = n_accepted
            if thin_output == 1 or _store:
                attempt_records.append(_rec)
            if thin_output > 1:
                _last_entry = (n_accepted, t, y_k.copy(), float(h_step),
                               info_aug, _rec)
            adapter.reset_committed(t, y_k, aux_k)

            # HOLD layer: ratio mode drifts h every accepted step (h_next =
            # rho*h_step with rho clamped to [r_min, r_max]), so a SINGLE-step
            # ratio test against hold_threshold >= r_max-1 is unsatisfiable (a
            # dead band that degenerates the march to fixed-step at h0).
            # Instead ACCUMULATE the per-step proposal ratio across held steps
            # and commit once the accumulated drift leaves the hold band; this
            # lets h grow/shrink geometrically (~r_max per step across commits)
            # while the theta factorization is still reused between commits.
            if quiesced:
                # quiescence boosted this attempt to h_max; adopt the
                # controller's proposal for the next committed step directly.
                h_held = min(float(h_max), max(float(h_min), float(h_next)))
                drift = 1.0
            elif h_step == h_held:
                drift *= (h_next / h_step) if h_step > 0.0 else 1.0
                if abs(drift - 1.0) > hold_threshold:
                    h_held = min(float(h_max), max(float(h_min), h_held * drift))
                    drift = 1.0
            # else: horizon-clamped final sliver -- keep h_held and drift.
            if monitor is not None:
                monitor.after_step(t, y_k, aux_k, info_aug, h=float(h_step),
                                   extras={"h_held": float(h_held)})
            if gc_interval > 0 and n_accepted % gc_interval == 0:
                gc.collect()
        else:
            # reject: state unchanged; a failed step must change h, so follow
            # the controller's (shrunk) proposal and reset the drift memory.
            n_rejected += 1
            if thin_output == 1:
                attempt_records.append(_rec)
            adapter.reset_committed(t, y_k, aux_k)
            h_new = min(float(h_max), max(float(h_min), float(h_next)))
            if h_new <= float(h_min) * (1.0 + 1.0e-12) and h_held <= float(h_min) * (1.0 + 1.0e-12):
                raise RuntimeError(
                    f"adaptive-ratio MJF step failed at h_min={h_new}"
                )
            h_held = h_new
            drift = 1.0
            if monitor is not None:
                monitor.after_reject(float(t), float(h_step))

    # With thinning on, the final accepted state may fall off the storage
    # grid; append it (and its attempt record) from the buffer so the
    # trajectory always ends at t_end.
    if _last_entry is not None and _last_entry[0] != _last_stored_step:
        _, _t_last, _y_last, _h_last, _info_last, _rec_last = _last_entry
        t_hist.append(_t_last)
        y_hist.append(_y_last)
        h_hist.append(_h_last)
        info_hist.append(_info_last)
        attempt_records.append(_rec_last)
    if monitor is not None:
        monitor.finish(float(t), y_k, aux_k, h=float(h_held),
                       extras={"h_held": float(h_held)})
    n_factorizations = int(getattr(stepper, "_theta_factorizations", 0)) - n_fac_start
    attempts_out = {
        "accepted": [rec["accepted"] for rec in attempt_records],
        "records": attempt_records,
        "n_factorizations": n_factorizations,
        "distinct_committed_h": sorted(distinct_h),
        "n_attempts": attempts,
        "n_accepted": n_accepted,
        "n_rejected": n_rejected,
        "thin_output": thin_output,
    }
    if record_attempts:
        attempts_out["attempt_log"] = ctrl.get_attempt_log()
    return (
        np.asarray(t_hist, dtype=float),
        np.asarray(y_hist, dtype=float),
        np.asarray(h_hist, dtype=float),
        info_hist,
        attempts_out,
    )


# -----------------------------------------------------------------------------
# Builder helpers
# -----------------------------------------------------------------------------


def build_moreau_jean_fremond(
    M, K, C,
    contacts: list[dict],
    H_callable,
    F_callable,
    *,
    S=None, D=None, B_biot=None,
    source_callable=None,
    aux_state_init: Optional[dict] = None,
    aux_law: Any = "constant",
    aux_law_params: Optional[dict] = None,
    e_N: Any = 0.0,
    theta: float = 0.5,
    soccp_tol_outer: float = 1.0e-10,
    soccp_max_outer: int = 300,
    soccp_max_inner: int = 30,
    soccp_sor_omega: float = 1.0,
    contact_solver: str = "pgs",
    contact_residual: str = "soc_fb",
    contact_soc_rho: Any = "auto",
    contact_soc_rho_scale: float = 1.0,
    contact_soc_rho_min: float = 1.0e-12,
    contact_soc_rho_max: float = 1.0e12,
    contact_petsc_options: Optional[dict] = None,
    contact_petsc_reuse_steps: int = 1,
    contact_linear_solver: str = "auto",
    contact_ssn_tol: float = 1.0e-10,
    contact_ssn_max_iter: int = 100,
    contact_ssn_line_search: bool = True,
    contact_allow_nonconverged: bool = False,
) -> tuple[MoreauJeanFremondStepper, dict]:
    """Construct a Moreau-Jean-Fremond stepper and its initial aux state.

    Parameters
    ----------
    M, K, C : (n_solid, n_solid) solid blocks.
    contacts : list of dicts with keys 'block_size' (1, 2, or 3) and
        optionally 'mu_init' (initial friction coefficient).
    H_callable : ``q -> H(q)`` or a constant matrix.
    F_callable : ``t -> applied solid force``.
    S, D, B_biot : optional fluid blocks for porodynamics.
    aux_state_init : optional override of the per-contact auxiliary state.
        For ``slip_weakening``, default is {'cum_slip': 0, 'mu': mu_init}.
        For rate-state laws, default is {'state': L/V_0, 'mu': mu_init}.
    aux_law : 'slip_weakening' | 'rate_state_aging' | 'rate_state_slip' | callable
    aux_law_params : dict of law-specific parameters (D_c, mu_s, mu_d for
        slip-weakening; a, b, V_0, L for rate-state).
    e_N : scalar or per-contact restitution (default 0).
    theta : theta-method parameter, default 1/2 (Fremond, energy-dissipative).

    Returns
    -------
    stepper : MoreauJeanFremondStepper
    aux_init : dict with the initial auxiliary state.
    """
    n_contacts = len(contacts)
    block_slices = []
    cursor = 0
    mu_init = np.zeros(n_contacts, dtype=float)
    for k, c in enumerate(contacts):
        d = int(c.get("block_size", 1))
        if d not in (1, 2, 3):
            raise ValueError(f"contact {k}: block_size must be 1, 2 or 3")
        block_slices.append(slice(cursor, cursor + d))
        cursor += d
        mu_init[k] = float(c.get("mu_init", 0.0))

    if np.ndim(e_N) == 0:
        e_N_vec = np.full(n_contacts, float(e_N), dtype=float)
    else:
        e_N_vec = np.asarray(e_N, dtype=float).ravel()
        if e_N_vec.size != n_contacts:
            raise ValueError("e_N must be a scalar or have one entry per contact")

    aux_law_params = dict(aux_law_params or {})

    if aux_state_init is None:
        if aux_law == "slip_weakening":
            aux_state_init = {"cum_slip": np.zeros(n_contacts), "mu": mu_init.copy()}
            # If user did not pass mu_s explicitly, inherit it from mu_init so
            # that the slip-weakening law starts at the user's chosen mu and
            # does not snap to the literal default.
            aux_law_params.setdefault("mu_s", float(mu_init[0]) if n_contacts else 0.6)
            aux_law_params.setdefault("mu_d", aux_law_params["mu_s"])
        elif aux_law in ("rate_state_aging", "rate_state_slip"):
            L = float(aux_law_params.get("L", 1.0e-3))
            V_0 = float(aux_law_params.get("V_0", 1.0e-6))
            aux_state_init = {
                "state": np.full(n_contacts, L / V_0, dtype=float),
                "mu": mu_init.copy(),
            }
        else:
            aux_state_init = {"mu": mu_init.copy()}

    stepper = MoreauJeanFremondStepper(
        M=M, K=K, C=C,
        S=S, D=D, B_biot=B_biot,
        block_slices=block_slices,
        e_N_vec=e_N_vec,
        H_callable=H_callable,
        F_callable=F_callable,
        source_callable=source_callable,
        theta=theta,
        aux_law=aux_law,
        aux_law_params=aux_law_params,
        soccp_tol_outer=soccp_tol_outer,
        soccp_max_outer=soccp_max_outer,
        soccp_max_inner=soccp_max_inner,
        soccp_sor_omega=soccp_sor_omega,
        contact_solver=contact_solver,
        contact_residual=contact_residual,
        contact_soc_rho=contact_soc_rho,
        contact_soc_rho_scale=contact_soc_rho_scale,
        contact_soc_rho_min=contact_soc_rho_min,
        contact_soc_rho_max=contact_soc_rho_max,
        contact_petsc_options=contact_petsc_options,
        contact_petsc_reuse_steps=contact_petsc_reuse_steps,
        contact_linear_solver=contact_linear_solver,
        contact_ssn_tol=contact_ssn_tol,
        contact_ssn_max_iter=contact_ssn_max_iter,
        contact_ssn_line_search=contact_ssn_line_search,
        contact_allow_nonconverged=contact_allow_nonconverged,
    )
    return stepper, aux_state_init


# -----------------------------------------------------------------------------
# Block / sparse helpers
# -----------------------------------------------------------------------------


def _to_dense_or_sparse(M):
    if sp.issparse(M):
        return M.tocsr()
    return np.asarray(M, dtype=float)


def _matvec(M, x):
    if sp.issparse(M):
        return np.asarray(M @ x).ravel()
    return np.asarray(M, dtype=float) @ np.asarray(x).ravel()


def _add(A, B):
    if sp.issparse(A) or sp.issparse(B):
        A_sp = sp.csr_matrix(A) if not sp.issparse(A) else A
        B_sp = sp.csr_matrix(B) if not sp.issparse(B) else B
        return (A_sp + B_sp).tocsr()
    return np.asarray(A, dtype=float) + np.asarray(B, dtype=float)


def _scale(A, c):
    if sp.issparse(A):
        return (float(c) * A).tocsr()
    return float(c) * np.asarray(A, dtype=float)


def _transpose(A):
    if sp.issparse(A):
        return A.T.tocsr()
    return np.asarray(A, dtype=float).T


def _block2x2(A11, A12, A21, A22):
    """Build a 2x2 block matrix preserving sparsity if any block is sparse."""
    any_sparse = any(sp.issparse(M) for M in (A11, A12, A21, A22))
    if any_sparse:
        return sp.bmat([[A11, A12], [A21, A22]], format="csr")
    A11 = np.asarray(A11, dtype=float)
    A12 = np.asarray(A12, dtype=float)
    A21 = np.asarray(A21, dtype=float)
    A22 = np.asarray(A22, dtype=float)
    n1 = A11.shape[0]
    n2 = A22.shape[0]
    out = np.zeros((n1 + n2, n1 + n2), dtype=float)
    out[:n1, :n1] = A11
    out[:n1, n1:] = A12
    out[n1:, :n1] = A21
    out[n1:, n1:] = A22
    return out
