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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .soccp_pgs import (
    soccp_pgs,
    fremond_shift_factory,
    desaxce_shift_factory,
    SocppPgsInfo,
    _shift_jacobian_fd,
    _classify_regime,
)
from .projected_radau_contact import _soc_fb_phi_and_jac
from .projections import IdentityProjection, MuScaledSOCProjection
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
    contact_ssn_max_iter: int = 30
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

    def _contact_block_transforms(self, mu_vec: np.ndarray) -> list:
        """Per-block mu-scaled cone transforms (T_x, T_y, T_y_inv).

        ``None`` marks blocks handled by the scalar branch (d == 1 or
        vanishing friction).  Hoisted out of the residual loop so one SSN
        solve builds them once instead of per Newton/line-search evaluation.
        """
        transforms = []
        for k, sl in enumerate(self.block_slices):
            d = sl.stop - sl.start
            mu = float(mu_vec[k])
            if d == 1 or mu <= 1.0e-14:
                transforms.append(None)
                continue
            T_x = np.eye(d, dtype=float)
            T_x[1:, 1:] *= mu
            T_y = np.eye(d, dtype=float)
            T_y[0, 0] = mu
            T_y_inv = np.eye(d, dtype=float)
            T_y_inv[0, 0] = 1.0 / mu
            transforms.append((T_x, T_y, T_y_inv))
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
    ):
        n_react = b.size
        p = np.asarray(p, dtype=float).ravel()
        u_full = W @ p + b
        residual = np.zeros(n_react, dtype=float)
        jac = np.zeros((n_react, n_react), dtype=float) if want_jacobian else None
        use_soc_natural_map = self.contact_residual == "soc_projection"
        if transforms is None:
            transforms = self._contact_block_transforms(mu_vec)
        shift_jacobian = getattr(shift_fn, "jacobian", None)

        for k, sl in enumerate(self.block_slices):
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

            T_x, T_y, T_y_inv = transforms[k]

            x_sd = T_x @ u_hat
            y_sd = T_y @ p_block

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
                residual[sl] = T_y_inv @ (y_sd - proj_z)
                if want_jacobian:
                    jac[sl, :] = T_y_inv @ (
                        J_proj @ (rho * T_x @ d_uhat_du @ W_rows)
                    )
                    jac[sl, sl] += T_y_inv @ (
                        (np.eye(d, dtype=float) - J_proj) @ T_y
                    )
                continue

            phi_sd, dphi_dx, dphi_dy = _soc_fb_phi_and_jac(x_sd, y_sd)
            residual[sl] = T_y_inv @ phi_sd

            if want_jacobian:
                jac[sl, :] = T_y_inv @ (dphi_dx @ T_x @ d_uhat_du @ W_rows)
                jac[sl, sl] += T_y_inv @ (dphi_dy @ T_y)

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
    ) -> tuple[np.ndarray, SocppPgsInfo, dict]:
        """Global reduced SOCCP semismooth Newton with PETSc linear solves."""
        if sp.issparse(W):
            W_dense = np.asarray(W.toarray(), dtype=float)
        else:
            W_dense = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        p = np.asarray(p0, dtype=float).copy()
        transforms = self._contact_block_transforms(mu_vec)
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
                transforms=transforms,
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
                    )
                    if np.all(np.isfinite(F_trial)):
                        p = p_trial
                    else:
                        break
            else:
                p = p + delta

        F_final, _, u_full = self._contact_ssn_residual_jacobian(
            p, W_dense, b, mu_vec, shift_fn, want_jacobian=False,
            transforms=transforms,
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
            for k, sl in enumerate(self.block_slices)
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


class _ThetaFactorization:
    """One LU factorization of the theta operator.

    A single ``step()`` needs ``n_react + 2`` solves against the same matrix
    (predictor, Delassus columns, corrector); this object factorizes once and
    back-substitutes, batching the Delassus block as a multi-RHS solve.  Its
    lifetime is tied to one ``(t, y, h)`` triple, so no staleness is possible
    regardless of step rejections or exceptions.
    """

    def __init__(self, op, backend: str, petsc_options: Optional[dict] = None):
        self.backend = str(backend or "scipy").lower()
        op_csr = op.tocsr() if sp.issparse(op) else sp.csr_matrix(op)
        self.shape = op_csr.shape
        if self.backend == "petsc":
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
        else:
            self._lu = spla.splu(op_csr.tocsc())

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=float).ravel()
        if self.backend == "petsc":
            self._b.setArray(rhs)
            self._ksp.solve(self._b, self._x)
            return np.array(self._x.getArray(readonly=True), copy=True)
        return np.asarray(self._lu.solve(rhs))

    def solve_multi(self, RHS: np.ndarray) -> np.ndarray:
        RHS = np.asarray(RHS, dtype=float)
        if RHS.ndim == 1:
            RHS = RHS[:, None]
        if self.backend == "petsc":
            PETSc = self._PETSc
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
                    [self.solve(RHS[:, j]) for j in range(RHS.shape[1])]
                )
        return np.asarray(self._lu.solve(RHS))

    def destroy(self):
        if self.backend == "petsc":
            for attr in ("_x", "_b", "_ksp", "_mat"):
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.destroy()
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
        theta_petsc_reuse_steps: int = 1,
        contact_ssn_tol: float = 1.0e-10,
        contact_ssn_max_iter: int = 30,
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
        if self.theta_linear_solver not in ("scipy", "petsc"):
            raise ValueError(
                "theta_linear_solver must be 'scipy' or 'petsc'; "
                f"got {self.theta_linear_solver!r}"
            )
        self.theta_petsc_options = dict(theta_petsc_options or self.contact_petsc_options)
        self.theta_petsc_reuse_steps = max(1, int(theta_petsc_reuse_steps))
        self.contact_ssn_tol = float(contact_ssn_tol)
        self.contact_ssn_max_iter = int(contact_ssn_max_iter)
        self.contact_ssn_line_search = bool(contact_ssn_line_search)
        self.contact_allow_nonconverged = bool(contact_allow_nonconverged)
        self._contact_petsc_solver = None
        self._theta_cache = None
        self._B_dense = None

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
        """Factorized theta system for ``(t, y, h)``, memoized one-deep.

        Outer fixed-point iterations on aux state (e.g. mu(s) consistency at
        the theta point) re-enter ``step()`` with identical ``(t, y, h)``;
        the operator, factorization, predictor, and Delassus block do not
        depend on aux, so they are reused.  Any change of arguments rebuilds
        from scratch — there is no cross-step staleness.
        """
        cache = self._theta_cache
        # exact (t, y, h) hit: aux fixed-point re-entry, nothing changed at all
        if (
            cache is not None
            and cache["t"] == float(t)
            and cache["h"] == float(h)
            and np.array_equal(cache["y"], y)
        ):
            return cache

        op, rhs_base = self._build_theta_system(t, y, h)
        op = op.tocsr() if sp.issparse(op) else sp.csr_matrix(op)

        # Cross-step reuse: for a fixed-step, constant (affine) operator the
        # theta-matrix is byte-identical across steps -- only the predictor RHS
        # depends on y/t.  Reuse the factorization + Delassus block whenever the
        # operator matrix is unchanged; recompute only the back-substitution.
        # Any genuine operator change (h, time-dependent or nonlinear J) is
        # detected here and triggers a full rebuild -- no staleness.
        reuse = (
            cache is not None
            and cache["h"] == float(h)
            and cache.get("fac") is not None
            and cache.get("op_shape") == op.shape
            and cache.get("op_nnz") == op.nnz
            and np.array_equal(cache["op_indptr"], op.indptr)
            and np.array_equal(cache["op_indices"], op.indices)
            and np.array_equal(cache["op_data"], op.data)
        )
        if reuse:
            fac = cache["fac"]
            z_pred = fac.solve(rhs_base)
            entry = dict(cache)                      # shares fac, X_contact, W
            entry.update(t=float(t), y=np.array(y, copy=True),
                         rhs=rhs_base, z_pred=z_pred)
            if self.n_react > 0:
                entry["b_soccp"] = np.asarray(self.D_contact @ z_pred, dtype=float).ravel()
                entry["u_old"] = np.asarray(self.D_contact @ y, dtype=float).ravel()
            self._theta_cache = entry
            return entry

        # operator changed -> (re)factorize and rebuild the Delassus block
        if cache is not None and cache.get("fac") is not None:
            cache["fac"].destroy()
        backend = "petsc" if self.theta_linear_solver == "petsc" else "scipy"
        fac = _ThetaFactorization(op, backend, self.theta_petsc_options)
        z_pred = fac.solve(rhs_base)
        entry = {
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
            entry["X_contact"] = np.asarray(X, dtype=float)
            entry["W"] = np.asarray(self.D_contact @ X, dtype=float)
            entry["b_soccp"] = np.asarray(self.D_contact @ z_pred, dtype=float).ravel()
            entry["u_old"] = np.asarray(self.D_contact @ y, dtype=float).ravel()
        self._theta_cache = entry
        return entry


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
    ) -> tuple[np.ndarray, SocppPgsInfo, dict, np.ndarray]:
        """Reduced contact SSN with mu evaluated from z_theta(p)."""
        if sp.issparse(W):
            W_dense = np.asarray(W.toarray(), dtype=float)
        else:
            W_dense = np.asarray(W, dtype=float)
        b = np.asarray(b, dtype=float).ravel()
        p = np.asarray(p0, dtype=float).copy()
        z_pred = np.asarray(z_pred, dtype=float).ravel()
        X_contact = np.asarray(X_contact, dtype=float)
        offset_impulse = np.asarray(offset_impulse, dtype=float).ravel()
        u_N_old = np.array([
            (self.D_contact @ (z_pred - X_contact @ offset_impulse))[sl.start]
            for sl in self.block_slices
        ])

        def state_from_effective_impulse(p_eff: np.ndarray) -> np.ndarray:
            return z_pred + X_contact @ (np.asarray(p_eff, dtype=float).ravel() - offset_impulse)

        def residual_for(p_eff: np.ndarray):
            z_theta = state_from_effective_impulse(p_eff)
            mu_vec = self._mu_from_theta_state(z_theta, t_theta)
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
            F, u_full, mu_vec = residual_for(p_eff)
            if self.dmu_dstate_callback is None or self.contact_residual != "soc_fb":
                return residual_and_jacobian_fd(p_eff)
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
        mu_vec = self._mu_from_theta_state(state_from_effective_impulse(p), t_theta)
        u_full = np.asarray(W_dense @ p + b, dtype=float).ravel()
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
        fac = theta_sys["fac"]
        z_pred = theta_sys["z_pred"]

        if self.n_react > 0:
            B = self.B.tocsr() if sp.issparse(self.B) else sp.csr_matrix(self.B)
            W = theta_sys["W"]
            u_old = theta_sys["u_old"]
            b_soccp = theta_sys["b_soccp"]
            p0 = aux.get("p_contact_prev", np.zeros(self.n_react, dtype=float))
            if len(p0) != self.n_react:
                p0 = np.zeros(self.n_react, dtype=float)
            offset_impulse = self._contact_offset_impulse(
                z_pred, float(t) + self.theta * float(h), h,
            )
            b_eff = b_soccp - W @ offset_impulse
            p0_eff = np.asarray(p0, dtype=float) + offset_impulse
            t_theta = float(t) + self.theta * float(h)
            if self.mu_state_callback is not None:
                if self.contact_solver != "petsc_ssn":
                    raise ValueError("state-dependent mu currently requires contact_solver='petsc_ssn'")
                p_effective, soccp_info, contact_diag, mu_vec = self._solve_contact_petsc_ssn_state_mu(
                    W, b_eff, p0_eff,
                    z_pred=z_pred,
                    X_contact=theta_sys["X_contact"],
                    offset_impulse=offset_impulse,
                    t_theta=t_theta,
                )
            else:
                mu_vec = np.asarray(aux.get("mu", self.mu_init), dtype=float).ravel()
                if mu_vec.size != self.n_contacts:
                    raise ValueError(
                        f"aux['mu'] has {mu_vec.size} entries; expected {self.n_contacts}"
                    )
                u_N_old = np.array([u_old[sl.start] for sl in self.block_slices])
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
            p_contact = p_effective - offset_impulse
            z_theta = fac.solve(rhs_base + np.asarray(B @ p_contact).ravel())
            if self.mu_state_callback is not None:
                mu_vec = self._mu_from_theta_state(z_theta, t_theta)
        else:
            p_contact = np.zeros(0, dtype=float)
            p_effective = p_contact.copy()
            offset_impulse = p_contact.copy()
            soccp_info = SocppPgsInfo(converged=True)
            contact_diag = {}
            z_theta = z_pred

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
    attempts : dict with ``accepted`` flags and per-attempt ``records``.
    """
    t0, t_end = float(t_span[0]), float(t_span[1])
    if t_end <= t0:
        raise ValueError("t_span must be increasing")
    y_k = np.asarray(y0, dtype=float).ravel().copy()
    n_state = y_k.size
    aux_k = dict(aux0 or {})

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
    expo = -1.0 / (float(error_order) + 1.0)
    if h0 is None:
        h0 = min(float(h_max), (t_end - t0) / 100.0)
    h_try = min(float(h0), float(h_max), t_end - t0)
    attempts = 0
    last_exc = None
    while t_hist[-1] < t_end - 10.0 * np.finfo(float).eps:
        if attempts >= int(max_attempts):
            raise RuntimeError(
                f"adaptive MJF loop exceeded {max_attempts} attempts"
            )
        attempts += 1
        t_k = float(t_hist[-1])
        h_k = min(float(h_try), t_end - t_k, float(h_max))
        h_k = max(h_k, min(float(h_min), t_end - t_k))
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
        attempt_records.append(
            {"accepted": bool(accepted), "t": t_k, "h": h_k, "error": float(err)}
        )
        if accepted:
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
            t_hist.append(t_k + h_k)
            y_hist.append(y_k.copy())
            h_hist.append(h_k)
            info_hist.append(info_aug)
            if err <= 1.0e-16:
                factor = float(max_factor)
            else:
                factor = float(safety) * err ** expo
                factor = min(float(max_factor), max(float(min_factor), factor))
            h_try = min(float(h_max), max(float(h_min), h_k * factor))
        else:
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

    return (
        np.asarray(t_hist, dtype=float),
        np.asarray(y_hist, dtype=float),
        np.asarray(h_hist, dtype=float),
        info_hist,
        {
            "accepted": [rec["accepted"] for rec in attempt_records],
            "records": attempt_records,
        },
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
    contact_ssn_max_iter: int = 30,
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
