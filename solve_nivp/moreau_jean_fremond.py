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
)


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

            p_contact, soccp_info = soccp_pgs(
                W, b_soccp, self.block_slices, mu_vec,
                shift_fn=shift_fn, p0=p0,
                max_outer=self.soccp_max_outer,
                max_inner=self.soccp_max_inner,
                tol_outer=self.soccp_tol_outer,
                sor_omega=self.soccp_sor_omega,
                return_info=True,
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
        if return_diagnostic:
            info.update(self._energy_terms(
                q, v, p_pore, q_new, v_new, p_pore_new,
                u_kt, p_contact, t, h,
            ))

        return y_new, aux_new, info

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
