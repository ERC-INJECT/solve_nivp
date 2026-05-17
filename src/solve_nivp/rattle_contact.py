"""True nonsmooth RATTLE integrator (Breuling et al. 2024).

This module implements the nonsmooth RATTLE method for constrained mechanical
systems following the structure of Cardillo (cardilloproject/cardillo) and the
paper:

    Breuling et al. (2024), "A nonsmooth RATTLE algorithm for mechanical
    systems with frictional unilateral constraints", Nonlinear Analysis:
    Hybrid Systems, DOI:10.1016/j.nahs.2024.101469

The key design principle is a strict separation between configuration ``q`` and
velocity ``u``, connected by a kinematic map ``q_dot = B(t,q) @ u + beta(t,q)``.
The integrator is a two-stage Lobatto IIIA--IIIB method:

* **Stage 1** (position level): Nonlinear solve for ``(q_{n+1}, u_{1/2})`` with
  proximal maps enforcing Signorini contact at the gap level and Coulomb friction
  at the slip-velocity level.

* **Stage 2** (velocity level): Linear saddle-point solve for ``u_{n+1}`` with
  proximal maps using Newton's restitution law.

When no contacts, bilaterals, or algebraic constraints are present the method
degrades to the symplectic Lobatto IIIA--IIIB integrator (2nd order,
time-reversible for conservative systems).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
#  Utility helpers (kept from original)
# ---------------------------------------------------------------------------

def _asvec(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _matvec(mat: Any, vec: np.ndarray) -> np.ndarray:
    if sp.issparse(mat):
        return np.asarray(mat @ vec, dtype=float).ravel()
    return np.asarray(mat, dtype=float) @ np.asarray(vec, dtype=float)


def _project_negative_orthant(arg: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(arg, dtype=float), 0.0)


def _project_ball(arg: np.ndarray, radius: float) -> np.ndarray:
    arg = _asvec(arg)
    radius = max(float(radius), 0.0)
    nrm = float(np.linalg.norm(arg))
    if nrm <= radius or nrm <= 0.0:
        return arg.copy()
    return (radius / nrm) * arg


def _solve_linear(mat: Any, rhs: np.ndarray) -> np.ndarray:
    rhs = _asvec(rhs)
    if sp.issparse(mat):
        return _asvec(spla.spsolve(mat.tocsc(), rhs))
    return _asvec(np.linalg.solve(np.asarray(mat, dtype=float), rhs))


def _eval_callable_or_const(obj, *args):
    """If *obj* is callable, call it with *args*; otherwise return it as-is."""
    if callable(obj):
        return obj(*args)
    return obj


def _as_2d(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    return v


def _wrms_norm(delta: np.ndarray, atol: float, rtol: float,
               ref: np.ndarray) -> float:
    """Weighted root-mean-square norm (Hairer convention)."""
    sc = atol + np.maximum(np.abs(ref), np.abs(ref + delta)) * rtol
    return float(np.sqrt(np.mean((delta / sc) ** 2)))


# ---------------------------------------------------------------------------
#  Data model
# ---------------------------------------------------------------------------

@dataclass
class RattleMechanicalSystem:
    """Cardillo-style mechanical system description for RATTLE.

    State: q in R^nq (configuration), u in R^nu (velocity).
    Kinematic equation: q_dot = B(t,q) @ u + beta(t,q)
    Momentum balance:   M(t,q) @ u_dot = h(t,q,u) + sum(W_i @ lambda_i)
    """
    nq: int
    nu: int
    q0: np.ndarray
    u0: np.ndarray

    # M(t, q) -> (nu, nu) or constant matrix.  None => identity.
    M: Any = None
    # h(t, q, u) -> (nu,) total smooth force vector.
    h_force: Callable[..., np.ndarray] = None  # type: ignore[assignment]
    # B(t, q) -> (nq, nu) kinematic map.  None => identity (requires nq == nu).
    B_kin: Any = None
    # beta(t, q) -> (nq,) kinematic drift.  None => zero.
    beta: Any = None

    # Optional analytic Jacobians for Newton efficiency.
    dh_dq: Optional[Callable] = None
    dh_du: Optional[Callable] = None
    dB_dq: Optional[Callable] = None

    def __post_init__(self) -> None:
        self.q0 = _asvec(self.q0)
        self.u0 = _asvec(self.u0)
        if self.B_kin is None and self.nq != self.nu:
            raise ValueError(
                "When B_kin is None (identity kinematic map), nq must equal nu; "
                f"got nq={self.nq}, nu={self.nu}.")
        if self.h_force is None:
            self.h_force = lambda t, q, u: np.zeros(self.nu, dtype=float)

    # -- evaluation helpers --------------------------------------------------

    def eval_M(self, t: float, q: np.ndarray) -> Any:
        if self.M is None:
            return sp.eye(self.nu, format="csc")
        return _eval_callable_or_const(self.M, t, q)

    def eval_h(self, t: float, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        return _asvec(self.h_force(t, q, u))

    def eval_B(self, t: float, q: np.ndarray) -> Any:
        if self.B_kin is None:
            return sp.eye(self.nq, format="csc")
        return _eval_callable_or_const(self.B_kin, t, q)

    def eval_beta(self, t: float, q: np.ndarray) -> np.ndarray:
        if self.beta is None:
            return np.zeros(self.nq, dtype=float)
        return _asvec(_eval_callable_or_const(self.beta, t, q))

    def eval_qdot(self, t: float, q: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
        """q_dot = B(t,q) @ u + beta(t,q)."""
        return _asvec(_matvec(self.eval_B(t, q), u) + self.eval_beta(t, q))


@dataclass
class RattleContactSpec:
    """One unilateral frictional contact point."""
    g_N: Callable[..., float]           # gap: g_N(t, q) -> scalar
    W_N: Any                            # (nu,) or callable(t,q)->(nu,)
    gamma_F: Callable[..., np.ndarray]  # slip vel: gamma_F(t,q,u) -> (n_F,)
    W_F: Any                            # (nu, n_F) or callable
    mu: Any = 0.0                       # Coulomb coefficient (float or callable)
    e: float = 0.0                      # Newton restitution coefficient
    n_F: int = 1                        # friction directions

    def eval_g_N(self, t: float, q: np.ndarray) -> float:
        return float(self.g_N(t, q))

    def eval_W_N(self, t: float, q: np.ndarray) -> np.ndarray:
        return _asvec(_eval_callable_or_const(self.W_N, t, q))

    def eval_gamma_F(self, t: float, q: np.ndarray,
                     u: np.ndarray) -> np.ndarray:
        return _asvec(self.gamma_F(t, q, u))

    def eval_W_F(self, t: float, q: np.ndarray) -> Any:
        W = _eval_callable_or_const(self.W_F, t, q)
        return _as_2d(W) if not sp.issparse(W) else W

    def eval_mu(self, *args) -> float:
        if callable(self.mu):
            return float(self.mu(*args))
        return float(self.mu)

    def eval_g_N_dot(self, t: float, q: np.ndarray,
                     u: np.ndarray) -> float:
        """Normal velocity: g_N_dot = W_N^T @ u."""
        return float(self.eval_W_N(t, q) @ u)


@dataclass
class RattleBilateralSpec:
    """Holonomic bilateral constraint: g(t,q) = 0."""
    g: Callable[..., np.ndarray]        # g(t, q) -> (n_g,)
    W_g: Any                            # (nu, n_g) or callable
    n_g: int = 1
    gamma: Optional[Callable] = None    # velocity-level: gamma(t,q,u) -> (n_gamma,)
    W_gamma: Any = None                 # (nu, n_gamma) or callable
    n_gamma: int = 0
    dg_dq: Optional[Callable] = None    # Jacobian dg/dq
    dgamma_dq: Optional[Callable] = None
    dgamma_du: Optional[Callable] = None

    def eval_g(self, t: float, q: np.ndarray) -> np.ndarray:
        return _asvec(self.g(t, q))

    def eval_W_g(self, t: float, q: np.ndarray) -> Any:
        return _as_2d(_eval_callable_or_const(self.W_g, t, q))

    def eval_gamma(self, t: float, q: np.ndarray,
                   u: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            return np.zeros(0, dtype=float)
        return _asvec(self.gamma(t, q, u))

    def eval_W_gamma(self, t: float, q: np.ndarray) -> Any:
        if self.W_gamma is None:
            return np.zeros((0, 0), dtype=float)
        return _as_2d(_eval_callable_or_const(self.W_gamma, t, q))

    def eval_g_dot_u(self, t: float, q: np.ndarray) -> Any:
        """Velocity-level Jacobian of position constraint: dg/dt partial through u."""
        W = self.eval_W_g(t, q)
        if sp.issparse(W):
            return W.T
        return np.asarray(W, dtype=float).T

    def eval_gamma_u(self, t: float, q: np.ndarray) -> Any:
        """Velocity-level Jacobian of velocity constraint w.r.t. u."""
        W = self.eval_W_gamma(t, q)
        if sp.issparse(W):
            return W.T
        return np.asarray(W, dtype=float).T

    def eval_chi_g(self, t: float, q: np.ndarray) -> np.ndarray:
        """Residual of g_dot when u=0: g_dot(t,q,0)."""
        if self.dg_dq is not None:
            return np.zeros(self.n_g, dtype=float)
        return np.zeros(self.n_g, dtype=float)

    def eval_chi_gamma(self, t: float, q: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            return np.zeros(0, dtype=float)
        return _asvec(self.gamma(t, q, np.zeros(0)))


@dataclass
class RattleAlgebraicSpec:
    """Compliance / algebraic constraint: c(t, q, u, la_c) = 0."""
    c: Callable[..., np.ndarray]        # c(t, q, u, la_c) -> (n_c,)
    W_c: Any                            # (nu, n_c) or callable
    n_c: int = 1
    dc_du: Optional[Callable] = None
    dc_dlac: Optional[Callable] = None

    def eval_c(self, t: float, q: np.ndarray, u: np.ndarray,
               la_c: np.ndarray) -> np.ndarray:
        return _asvec(self.c(t, q, u, la_c))

    def eval_W_c(self, t: float, q: np.ndarray) -> Any:
        return _as_2d(_eval_callable_or_const(self.W_c, t, q))


@dataclass
class RattleContactSystem:
    """Assembled RATTLE system with all constraints."""
    mech: RattleMechanicalSystem
    contacts: list[RattleContactSpec] = field(default_factory=list)
    bilaterals: list[RattleBilateralSpec] = field(default_factory=list)
    algebraics: list[RattleAlgebraicSpec] = field(default_factory=list)
    prox_alpha: float = 0.5
    prox_r_min: float = 1.0e-8
    prox_r_max: float = 1.0e8
    gap_tol: float = 0.0
    initial_normal_forces: Optional[np.ndarray] = None
    initial_friction_forces: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.n_contacts = len(self.contacts)
        self.nla_N = self.n_contacts
        self.nla_F = sum(c.n_F for c in self.contacts)
        self.n_bilat_g = sum(b.n_g for b in self.bilaterals)
        self.n_bilat_gamma = sum(b.n_gamma for b in self.bilaterals)
        self.n_algebraic = sum(a.n_c for a in self.algebraics)
        self.has_contact = self.n_contacts > 0
        self.has_bilateral = self.n_bilat_g > 0 or self.n_bilat_gamma > 0
        self.has_algebraic = self.n_algebraic > 0
        self.has_constraints = (self.has_contact or self.has_bilateral
                                or self.has_algebraic)
        if self.initial_normal_forces is None:
            self.initial_normal_forces = np.zeros(self.nla_N, dtype=float)
        else:
            self.initial_normal_forces = _asvec(self.initial_normal_forces)
            if self.initial_normal_forces.size != self.nla_N:
                raise ValueError(
                    "initial_normal_forces must have length "
                    f"{self.nla_N}, got {self.initial_normal_forces.size}."
                )
        if self.initial_friction_forces is None:
            self.initial_friction_forces = np.zeros(self.nla_F, dtype=float)
        else:
            self.initial_friction_forces = _asvec(self.initial_friction_forces)
            if self.initial_friction_forces.size != self.nla_F:
                raise ValueError(
                    "initial_friction_forces must have length "
                    f"{self.nla_F}, got {self.initial_friction_forces.size}."
                )


@dataclass
class RattleSolveResult:
    """Result container for RATTLE integration."""
    times: np.ndarray
    states: np.ndarray              # shape (m, nq+nu), stacked [q, u]
    step_sizes: np.ndarray
    reaction_force_history: np.ndarray
    reaction_impulse_history: np.ndarray
    step_success: np.ndarray
    step_iterations: np.ndarray
    step_solver_error: np.ndarray
    step_true_residual_inf: np.ndarray
    step_true_residual_rms: np.ndarray
    stage1_fixed_point_iterations: np.ndarray
    stage2_fixed_point_iterations: np.ndarray
    bilateral_impulse_history: Optional[np.ndarray] = None
    failure: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
#  Core solver
# ---------------------------------------------------------------------------

class RattleSolver:
    """Two-stage nonsmooth RATTLE integrator."""

    def __init__(
        self,
        system: RattleContactSystem,
        *,
        newton_tol: float = 1.0e-10,
        newton_max_iter: int = 30,
        fixed_point_tol: float = 1.0e-8,
        fixed_point_max_iter: int = 50,
        linesearch_max_iter: int = 8,
        state_rtol: float = 1.0e-8,
        state_atol: float = 1.0e-10,
        reuse_factorization: bool = True,
        stage1_method: str = "fixed_point",
    ) -> None:
        self.sys = system
        self.mech = system.mech
        self.newton_tol = float(newton_tol)
        self.newton_max_iter = int(newton_max_iter)
        self.fp_tol = float(fixed_point_tol)
        self.fp_max = int(fixed_point_max_iter)
        self.ls_max = int(linesearch_max_iter)
        self.state_rtol = float(state_rtol)
        self.state_atol = float(state_atol)
        self.reuse_fac = bool(reuse_factorization)
        if stage1_method not in ("fixed_point", "semismooth_newton"):
            raise ValueError(
                f"stage1_method must be 'fixed_point' or 'semismooth_newton', "
                f"got {stage1_method!r}."
            )
        self.stage1_method = str(stage1_method)

        # Proximal parameters (allocated once, updated per step).
        self.prox_r_N = np.ones(system.nla_N, dtype=float)
        self.prox_r_F = np.ones(system.nla_F, dtype=float)

        # Dimension helpers for stage-1 unknown layout.
        nq, nu = self.mech.nq, self.mech.nu
        n_alg = system.n_algebraic
        n_bg = system.n_bilat_g
        n_bv = system.n_bilat_gamma
        # x1 = [q_{n+1}(nq), u_{1/2}(nu), la_c(n_alg), P_g1(n_bg), P_gamma1(n_bv)]
        self._x1_splits = np.cumsum([nq, nu, n_alg, n_bg])[:-1] if (n_alg + n_bg + n_bv) > 0 \
            else np.array([nq], dtype=int)
        self._nx1 = nq + nu + n_alg + n_bg + n_bv
        # Smooth state size for convergence check (q, u_{1/2}, la_c).
        self._n_smooth_state = nq + nu + n_alg

    # ------------------------------------------------------------------
    #  Delassus proximal parameter estimation
    # ------------------------------------------------------------------

    def _compute_prox_params(self, t: float, q: np.ndarray) -> None:
        if not self.sys.has_contact:
            return
        M = self.mech.eval_M(t, q)
        alpha = self.sys.prox_alpha
        r_min = self.sys.prox_r_min
        r_max = self.sys.prox_r_max

        # Factor M for solving M x = w.
        if sp.issparse(M):
            M_lu = spla.splu(M.tocsc())
            solve = M_lu.solve
        else:
            M_dense = np.asarray(M, dtype=float)
            M_lu = None
            solve = lambda b: np.linalg.solve(M_dense, b)

        # Normal contacts: rho_N^k = alpha / G_NN,kk
        if self.sys.nla_N > 0:
            w_n_cols = [c.eval_W_N(t, q) for c in self.sys.contacts]
            W_N = np.column_stack(w_n_cols)
            M_inv_WN = np.asarray(solve(W_N), dtype=float)
            G_NN_diag = np.sum(W_N * M_inv_WN, axis=0)
            self.prox_r_N[:] = np.clip(
                alpha / np.maximum(G_NN_diag, 1.0e-30),
                r_min,
                r_max,
            )

        # Friction: one scale per contact block, using the block diagonal scale
        # advocated in the thesis discussion of G_FF.
        f_offset = 0
        for c in self.sys.contacts:
            if c.n_F <= 0:
                continue
            W_F = c.eval_W_F(t, q)
            W_F_dense = W_F.toarray() if sp.issparse(W_F) else np.asarray(W_F, dtype=float)
            M_inv_WF = np.asarray(solve(W_F_dense), dtype=float)
            G_FF_diag = np.sum(W_F_dense * M_inv_WF, axis=0)
            g_ref = float(np.min(np.maximum(G_FF_diag, 1.0e-30)))
            rho_f = float(np.clip(alpha / g_ref, r_min, r_max))
            self.prox_r_F[f_offset : f_offset + c.n_F] = rho_f
            f_offset += c.n_F

    # ------------------------------------------------------------------
    #  Stage 1: residual, Jacobian, Newton, proximal maps
    # ------------------------------------------------------------------

    def _split_x1(self, x: np.ndarray):
        """Split the Stage 1 smooth unknowns into components."""
        nq, nu = self.mech.nq, self.mech.nu
        n_alg = self.sys.n_algebraic
        n_bg = self.sys.n_bilat_g
        q_new = x[:nq]
        u_half = x[nq:nq + nu]
        la_c = x[nq + nu:nq + nu + n_alg] if n_alg > 0 else np.zeros(0)
        P_g1 = x[nq + nu + n_alg:nq + nu + n_alg + n_bg] if n_bg > 0 else np.zeros(0)
        P_gam1 = x[nq + nu + n_alg + n_bg:] if self.sys.n_bilat_gamma > 0 else np.zeros(0)
        return q_new, u_half, la_c, P_g1, P_gam1

    def _stage1_residual(self, x: np.ndarray, P_N: np.ndarray,
                         P_F: np.ndarray, *, t_n: float, h: float,
                         q_n: np.ndarray, u_n: np.ndarray,
                         B_n: Any, beta_n: np.ndarray,
                         M_n: Any) -> np.ndarray:
        q_new, u_half, la_c, P_g1, P_gam1 = self._split_x1(x)
        t_new = t_n + h

        # Block 1: kinematic equation.
        qdot_new = self.mech.eval_qdot(t_new, q_new, u_half)
        qdot_n = _asvec(_matvec(B_n, u_half) + beta_n)
        r_kin = q_new - q_n - 0.5 * h * (qdot_n + qdot_new)

        # Block 2: half-step momentum balance.
        h_smooth = self.mech.eval_h(t_n, q_n, u_half)
        r_mom = _matvec(M_n, u_half - u_n) - 0.5 * h * h_smooth

        # Contact impulses.
        if self.sys.has_contact:
            f_off = 0
            for k, c in enumerate(self.sys.contacts):
                W_N = c.eval_W_N(t_n, q_n)
                W_F = c.eval_W_F(t_n, q_n)
                r_mom = r_mom - W_N * P_N[k]
                for j in range(c.n_F):
                    w_j = _asvec(W_F[:, j]) if not sp.issparse(W_F) \
                        else np.asarray(W_F[:, j].toarray(), dtype=float).ravel()
                    r_mom = r_mom - w_j * P_F[f_off + j]
                f_off += c.n_F

        # Algebraic compliance force.
        if self.sys.has_algebraic:
            off = 0
            for a in self.sys.algebraics:
                W_c = a.eval_W_c(t_n, q_n)
                la_block = la_c[off:off + a.n_c]
                r_mom = r_mom - 0.5 * h * _matvec(W_c, la_block)
                off += a.n_c

        # Bilateral impulses.
        if self.sys.has_bilateral:
            g_off, gam_off = 0, 0
            for b in self.sys.bilaterals:
                W_g = b.eval_W_g(t_n, q_n)
                P_g_block = P_g1[g_off:g_off + b.n_g]
                r_mom = r_mom - _matvec(W_g, P_g_block)
                g_off += b.n_g
                if b.n_gamma > 0:
                    W_gam = b.eval_W_gamma(t_n, q_n)
                    P_gam_block = P_gam1[gam_off:gam_off + b.n_gamma]
                    r_mom = r_mom - _matvec(W_gam, P_gam_block)
                    gam_off += b.n_gamma

        blocks = [r_kin, r_mom]

        # Block 3: compliance equations.
        if self.sys.has_algebraic:
            off = 0
            for a in self.sys.algebraics:
                la_block = la_c[off:off + a.n_c]
                blocks.append(a.eval_c(t_n, q_n, u_half, la_block))
                off += a.n_c

        # Block 4: position-level bilateral g(t_{n+1}, q_{n+1}) = 0.
        if self.sys.n_bilat_g > 0:
            off = 0
            for b in self.sys.bilaterals:
                blocks.append(b.eval_g(t_new, q_new))
                off += b.n_g

        # Block 5: velocity-level bilateral gamma(t_{n+1}, q_{n+1}, u_{1/2}) = 0.
        if self.sys.n_bilat_gamma > 0:
            for b in self.sys.bilaterals:
                if b.n_gamma > 0:
                    blocks.append(b.eval_gamma(t_new, q_new, u_half))

        return np.concatenate(blocks)

    def _stage1_jacobian(self, x: np.ndarray, *, t_n: float, h: float,
                         q_n: np.ndarray, u_n: np.ndarray,
                         B_n: Any, M_n: Any) -> Any:
        """Assemble the analytic Jacobian for the Stage 1 smooth system."""
        q_new, u_half, la_c, P_g1, P_gam1 = self._split_x1(x)
        t_new = t_n + h
        nq, nu = self.mech.nq, self.mech.nu
        n_alg = self.sys.n_algebraic
        n_bg = self.sys.n_bilat_g
        n_bv = self.sys.n_bilat_gamma
        N = self._nx1

        # Dense assembly (switch to sparse for large systems later).
        J = np.zeros((N, N), dtype=float)

        # --- Block (kin, q): I - 0.5*h * d(qdot_new)/d(q_new) ---
        J[:nq, :nq] = np.eye(nq)
        if self.mech.dB_dq is not None:
            pass  # TODO: configuration-dependent B Jacobian
        # For B=I case, d(qdot_new)/d(q_new) = 0, so J[kin,q] = I.

        # --- Block (kin, u): -0.5*h * (B_n + B_new) ---
        B_new = self.mech.eval_B(t_new, q_new)
        B_n_dense = B_n.toarray() if sp.issparse(B_n) else np.asarray(B_n, dtype=float)
        B_new_dense = B_new.toarray() if sp.issparse(B_new) else np.asarray(B_new, dtype=float)
        J[:nq, nq:nq + nu] = -0.5 * h * (B_n_dense + B_new_dense)

        # --- Block (mom, u): M_n - 0.5*h * dh/du ---
        M_n_dense = M_n.toarray() if sp.issparse(M_n) else np.asarray(M_n, dtype=float)
        J[nq:nq + nu, nq:nq + nu] = M_n_dense
        if self.mech.dh_du is not None:
            dh_du_val = self.mech.dh_du(t_n, q_n, u_half)
            if sp.issparse(dh_du_val):
                dh_du_val = dh_du_val.toarray()
            J[nq:nq + nu, nq:nq + nu] -= 0.5 * h * np.asarray(dh_du_val, dtype=float)

        # Block (mom, q_new) is zero: the Stage-1 force h(t_n, q_n, u_half)
        # is evaluated at the fixed q_n per Lobatto IIIA-IIIB, so it carries
        # no dependence on q_new. The dh_dq hook is still consumed by the
        # Stage-2 half-step (t_new, q_new, u_half) assembly elsewhere.

        col = nq + nu  # column offset for remaining blocks

        # --- Block (mom, la_c): -0.5*h * W_c ---
        if n_alg > 0:
            off = 0
            for a in self.sys.algebraics:
                W_c = a.eval_W_c(t_n, q_n)
                W_c_dense = W_c.toarray() if sp.issparse(W_c) else np.asarray(W_c, dtype=float)
                J[nq:nq + nu, col + off:col + off + a.n_c] = -0.5 * h * W_c_dense
                off += a.n_c

        # --- Block (mom, P_g1): -W_g ---
        col_bg = nq + nu + n_alg
        if n_bg > 0:
            off = 0
            for b in self.sys.bilaterals:
                W_g = b.eval_W_g(t_n, q_n)
                W_g_dense = W_g.toarray() if sp.issparse(W_g) else np.asarray(W_g, dtype=float)
                J[nq:nq + nu, col_bg + off:col_bg + off + b.n_g] = -W_g_dense
                off += b.n_g

        # --- Block (mom, P_gamma1): -W_gamma ---
        col_bv = col_bg + n_bg
        if n_bv > 0:
            off = 0
            for b in self.sys.bilaterals:
                if b.n_gamma > 0:
                    W_gam = b.eval_W_gamma(t_n, q_n)
                    W_gam_dense = W_gam.toarray() if sp.issparse(W_gam) \
                        else np.asarray(W_gam, dtype=float)
                    J[nq:nq + nu, col_bv + off:col_bv + off + b.n_gamma] = -W_gam_dense
                    off += b.n_gamma

        # --- Block (compliance, u): dc/du,  Block (compliance, la_c): dc/dlac ---
        row = nq + nu
        if n_alg > 0:
            off_c = 0
            for a in self.sys.algebraics:
                if a.dc_du is not None:
                    dc_du_val = a.dc_du(t_n, q_n, u_half, la_c[off_c:off_c + a.n_c])
                    if sp.issparse(dc_du_val):
                        dc_du_val = dc_du_val.toarray()
                    J[row:row + a.n_c, nq:nq + nu] = np.asarray(dc_du_val, dtype=float)
                if a.dc_dlac is not None:
                    dc_dl_val = a.dc_dlac(t_n, q_n, u_half, la_c[off_c:off_c + a.n_c])
                    if sp.issparse(dc_dl_val):
                        dc_dl_val = dc_dl_val.toarray()
                    J[row:row + a.n_c, nq + nu + off_c:nq + nu + off_c + a.n_c] = \
                        np.asarray(dc_dl_val, dtype=float)
                row += a.n_c
                off_c += a.n_c

        # --- Block (g, q): dg/dq ---
        if n_bg > 0:
            off = 0
            for b in self.sys.bilaterals:
                if b.dg_dq is not None:
                    dg = b.dg_dq(t_new, q_new)
                    if sp.issparse(dg):
                        dg = dg.toarray()
                    J[row:row + b.n_g, :nq] = np.asarray(dg, dtype=float)
                else:
                    # Fall back to W_g^T as approximation (exact when g = W_g^T q + ...).
                    W_g = b.eval_W_g(t_new, q_new)
                    W_g_dense = W_g.toarray() if sp.issparse(W_g) else np.asarray(W_g, dtype=float)
                    J[row:row + b.n_g, :nq] = W_g_dense.T[:b.n_g, :nq]
                row += b.n_g
                off += b.n_g

        # --- Block (gamma, q) and (gamma, u) ---
        if n_bv > 0:
            for b in self.sys.bilaterals:
                if b.n_gamma > 0:
                    if b.dgamma_du is not None:
                        dgu = b.dgamma_du(t_new, q_new, u_half)
                        if sp.issparse(dgu):
                            dgu = dgu.toarray()
                        J[row:row + b.n_gamma, nq:nq + nu] = np.asarray(dgu, dtype=float)
                    else:
                        W_gam = b.eval_W_gamma(t_new, q_new)
                        W_gam_dense = W_gam.toarray() if sp.issparse(W_gam) \
                            else np.asarray(W_gam, dtype=float)
                        J[row:row + b.n_gamma, nq:nq + nu] = W_gam_dense.T[:b.n_gamma, :]
                    if b.dgamma_dq is not None:
                        dgq = b.dgamma_dq(t_new, q_new, u_half)
                        if sp.issparse(dgq):
                            dgq = dgq.toarray()
                        J[row:row + b.n_gamma, :nq] = np.asarray(dgq, dtype=float)
                    row += b.n_gamma

        return J

    def _newton_stage1(self, x0: np.ndarray, P_N: np.ndarray,
                       P_F: np.ndarray, *, t_n: float, h: float,
                       q_n: np.ndarray, u_n: np.ndarray,
                       B_n: Any, beta_n: np.ndarray,
                       M_n: Any) -> tuple[np.ndarray, int, float]:
        """Newton solve for the Stage 1 smooth unknowns."""
        x = x0.copy()
        res = self._stage1_residual(x, P_N, P_F, t_n=t_n, h=h,
                                    q_n=q_n, u_n=u_n, B_n=B_n,
                                    beta_n=beta_n, M_n=M_n)
        res_norm = float(np.max(np.abs(res)))
        if res_norm <= self.newton_tol:
            return x, 0, res_norm

        _cached_lu = None
        for it in range(1, self.newton_max_iter + 1):
            if _cached_lu is None or not self.reuse_fac:
                jac = self._stage1_jacobian(x, t_n=t_n, h=h,
                                            q_n=q_n, u_n=u_n,
                                            B_n=B_n, M_n=M_n)
                _cached_lu = jac  # for dense use directly

            step = _solve_linear(_cached_lu, -res)

            # Linesearch.
            alpha_ls = 1.0
            x_trial = x + step
            res_trial = self._stage1_residual(x_trial, P_N, P_F, t_n=t_n,
                                              h=h, q_n=q_n, u_n=u_n,
                                              B_n=B_n, beta_n=beta_n,
                                              M_n=M_n)
            res_trial_norm = float(np.max(np.abs(res_trial)))
            ls_it = 0
            while (res_trial_norm > res_norm and ls_it < self.ls_max
                   and alpha_ls > 1e-4):
                alpha_ls *= 0.5
                x_trial = x + alpha_ls * step
                res_trial = self._stage1_residual(
                    x_trial, P_N, P_F, t_n=t_n, h=h,
                    q_n=q_n, u_n=u_n, B_n=B_n, beta_n=beta_n, M_n=M_n)
                res_trial_norm = float(np.max(np.abs(res_trial)))
                ls_it += 1

            x = x_trial
            res = res_trial
            res_norm = res_trial_norm
            if res_norm <= self.newton_tol:
                return x, it, res_norm

        return x, self.newton_max_iter, res_norm

    # ------------------------------------------------------------------
    #  Stage 1 proximal maps
    # ------------------------------------------------------------------

    def _prox_stage1(self, q_new: np.ndarray, u_half: np.ndarray,
                     P_N_old: np.ndarray, P_F_old: np.ndarray,
                     t_new: float, h: float
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Position-level proximal maps for normal contact + friction.

        A hard `gap > gap_tol` cut-off that zeroes ``P_N`` is not used here.
        The Moreau prox ``max(0, P_N_old − (r_N/h) g_N)`` already collapses
        to zero for clearly open contacts (``(r_N/h) g_N > P_N_old``) and, on
        dense Delassus couplings with a non-trivial ``P_N_old``, smoothly
        relaxes the impulse as the gap crosses zero. A hard switch here
        creates a bang-bang discontinuity that defeats the fixed-point when
        Newton momentarily nudges ``g_N`` slightly positive while ``P_N_old``
        is still the correct closed-contact impulse.
        """
        P_N_new = np.zeros_like(P_N_old)
        P_F_new = np.zeros_like(P_F_old)
        active = np.zeros(self.sys.n_contacts, dtype=bool)
        gap_tol = float(self.sys.gap_tol)
        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            gap = c.eval_g_N(t_new, q_new)

            r_N = self.prox_r_N[k]
            prox_arg_N = r_N / h * gap - P_N_old[k]
            P_N_new[k] = max(0.0, -prox_arg_N)
            # Hard release: contact is definitely open and carries no impulse.
            if gap > gap_tol and P_N_new[k] == 0.0:
                f_off += c.n_F
                continue
            active[k] = (prox_arg_N <= 0.0) or (P_N_new[k] > 0.0)

            # Friction.
            mu = c.eval_mu(q_new) if callable(c.mu) else float(c.mu)
            gamma_F_val = c.eval_gamma_F(t_new, q_new, u_half)
            for j in range(c.n_F):
                r_F = self.prox_r_F[f_off + j]
                tang_arg = r_F * gamma_F_val[j:j + 1] - P_F_old[f_off + j:f_off + j + 1]
                # Scalar 2D friction: ball projection with radius mu * P_N.
                proj = _project_ball(tang_arg, mu * P_N_new[k])
                P_F_new[f_off + j] = -float(proj[0])

            f_off += c.n_F
        return P_N_new, P_F_new, active

    # ------------------------------------------------------------------
    #  Stage 1 semismooth-Newton variant (opt-in)
    # ------------------------------------------------------------------
    #
    # The fixed-point variant iterates Newton(smooth) ↔ prox(P_N, P_F). Its
    # precision floor is the prox contraction tolerance, so |g_N| stalls near
    # ``fp_tol``. The semismooth-Newton variant instead stacks the contact
    # impulses into the unknown vector and assembles a single extended
    # residual that includes NCP rows:
    #
    #     y = [x1_smooth, P_N, P_F]
    #     F(y) = [ r_kin(x1)
    #              r_mom(x1) − W_N P_N − W_F P_F
    #              NCP_N(q_new, P_N)
    #              NCP_F(u_half, P_N, P_F) ]
    #
    # with a scaled minimum-map for the normal NCP and an active-set
    # (stick / slip±) form for the scalar friction NCP. The Clarke derivative
    # selects one of two or three branches per contact per iterate, giving
    # Newton-rate convergence and driving ``g_N`` to roundoff.
    #
    # Restrictions: only ``n_F ∈ {0, 1}`` friction directions are currently
    # supported — multi-direction (SOC) friction still needs the fixed-point
    # variant.

    def _ssn_pack(self, x1: np.ndarray, P_N: np.ndarray, P_F: np.ndarray
                  ) -> np.ndarray:
        return np.concatenate([x1, P_N, P_F])

    def _ssn_split(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F
        x1 = y[: self._nx1]
        P_N = y[self._nx1 : self._nx1 + nla_N]
        P_F = y[self._nx1 + nla_N : self._nx1 + nla_N + nla_F]
        return x1, P_N, P_F

    def _ssn_decide_active_sets(self, q_new: np.ndarray, u_half: np.ndarray,
                                P_N: np.ndarray, P_F: np.ndarray,
                                t_new: float, h: float
                                ) -> tuple[np.ndarray, np.ndarray]:
        """Pick NCP branches at the current iterate.

        Returns
        -------
        active_N : (nla_N,) bool
            ``True`` when the normal complementarity row enforces ``g_N = 0``
            (contact closed); ``False`` when it enforces ``P_N = 0``.
        fric_branch : (nla_F,) int
            0 = stick (``gamma_F = 0`` row), +1 = slip plus
            (``P_F − μ P_N = 0``), −1 = slip minus (``P_F + μ P_N = 0``).
        """
        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F
        active_N = np.zeros(nla_N, dtype=bool)
        fric_branch = np.zeros(nla_F, dtype=np.int8)

        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            r_N = self.prox_r_N[k]
            gap = c.eval_g_N(t_new, q_new)
            # Scaled minimum-map pivot: active ⇔ P_N > (r_N/h) g_N.
            active_N[k] = P_N[k] > (r_N / h) * gap

            if c.n_F == 0:
                continue
            if c.n_F > 1:
                raise NotImplementedError(
                    "stage1_method='semismooth_newton' currently supports only "
                    "scalar friction (n_F ∈ {0, 1}); multi-direction SOC friction "
                    "must use the fixed-point variant."
                )
            mu = c.eval_mu(q_new) if callable(c.mu) else float(c.mu)
            r_F = self.prox_r_F[f_off]
            gamma = float(c.eval_gamma_F(t_new, q_new, u_half)[0])
            limit = mu * max(P_N[k], 0.0)
            clip_arg = P_F[f_off] - r_F * gamma
            if not active_N[k] or limit <= 0.0:
                # No normal impulse → friction is pinned to zero (degenerate).
                fric_branch[f_off] = 2  # sentinel: force P_F = 0 row
            elif clip_arg > limit:
                fric_branch[f_off] = +1
            elif clip_arg < -limit:
                fric_branch[f_off] = -1
            else:
                fric_branch[f_off] = 0
            f_off += c.n_F

        return active_N, fric_branch

    def _ssn_residual(self, y: np.ndarray, *, t_n: float, h: float,
                      q_n: np.ndarray, u_n: np.ndarray,
                      B_n: Any, beta_n: np.ndarray, M_n: Any,
                      active_N: np.ndarray, fric_branch: np.ndarray
                      ) -> np.ndarray:
        x1, P_N, P_F = self._ssn_split(y)
        r_smooth = self._stage1_residual(
            x1, P_N, P_F, t_n=t_n, h=h, q_n=q_n, u_n=u_n,
            B_n=B_n, beta_n=beta_n, M_n=M_n,
        )
        nq, nu = self.mech.nq, self.mech.nu
        q_new = x1[:nq]
        u_half = x1[nq:nq + nu]
        t_new = t_n + h

        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F
        r_N_block = np.zeros(nla_N, dtype=float)
        r_F_block = np.zeros(nla_F, dtype=float)

        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            r_N = self.prox_r_N[k]
            gap = c.eval_g_N(t_new, q_new)
            if active_N[k]:
                r_N_block[k] = (r_N / h) * gap
            else:
                r_N_block[k] = P_N[k]

            if c.n_F == 0:
                continue
            mu = c.eval_mu(q_new) if callable(c.mu) else float(c.mu)
            r_F = self.prox_r_F[f_off]
            gamma = float(c.eval_gamma_F(t_new, q_new, u_half)[0])
            branch = int(fric_branch[f_off])
            if branch == 0:       # stick: gamma_F = 0
                r_F_block[f_off] = r_F * gamma
            elif branch == +1:    # slip +: P_F − μ P_N = 0
                r_F_block[f_off] = P_F[f_off] - mu * P_N[k]
            elif branch == -1:    # slip −: P_F + μ P_N = 0
                r_F_block[f_off] = P_F[f_off] + mu * P_N[k]
            else:                 # sentinel 2: P_F = 0 (no normal impulse)
                r_F_block[f_off] = P_F[f_off]
            f_off += c.n_F

        return np.concatenate([r_smooth, r_N_block, r_F_block])

    def _ssn_jacobian(self, y: np.ndarray, *, t_n: float, h: float,
                      q_n: np.ndarray, u_n: np.ndarray,
                      B_n: Any, M_n: Any,
                      active_N: np.ndarray, fric_branch: np.ndarray
                      ) -> np.ndarray:
        x1, P_N, P_F = self._ssn_split(y)
        nq, nu = self.mech.nq, self.mech.nu
        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F
        n_smooth = self._nx1
        N_full = n_smooth + nla_N + nla_F

        J = np.zeros((N_full, N_full), dtype=float)
        J_smooth = self._stage1_jacobian(
            x1, t_n=t_n, h=h, q_n=q_n, u_n=u_n, B_n=B_n, M_n=M_n,
        )
        J[:n_smooth, :n_smooth] = np.asarray(J_smooth, dtype=float)

        # ── d r_mom / d P_N = −W_N,  d r_mom / d P_F = −W_F ────────────────
        col_PN = n_smooth
        col_PF = n_smooth + nla_N
        mom_row0 = nq
        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            W_N = c.eval_W_N(t_n, q_n)
            J[mom_row0:mom_row0 + nu, col_PN + k] = -np.asarray(W_N, dtype=float)
            if c.n_F > 0:
                W_F = c.eval_W_F(t_n, q_n)
                W_F_dense = W_F.toarray() if sp.issparse(W_F) else np.asarray(W_F, dtype=float)
                for j in range(c.n_F):
                    J[mom_row0:mom_row0 + nu, col_PF + f_off + j] = -W_F_dense[:, j]
            f_off += c.n_F

        # ── Normal NCP rows ────────────────────────────────────────────────
        q_new = x1[:nq]
        u_half = x1[nq:nq + nu]
        t_new = t_n + h
        row_N = n_smooth
        for k, c in enumerate(self.sys.contacts):
            r_N = self.prox_r_N[k]
            if active_N[k]:
                # (r_N/h) * g_N(q_new) row:  d/dq_new ≈ (r_N/h) W_N^T
                # (exact when g_N(q) = W_N^T q + const with B = I).
                W_N_new = c.eval_W_N(t_new, q_new)
                J[row_N + k, :nq] = (r_N / h) * np.asarray(W_N_new, dtype=float)
            else:
                J[row_N + k, col_PN + k] = 1.0

        # ── Friction NCP rows ──────────────────────────────────────────────
        row_F = n_smooth + nla_N
        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            if c.n_F == 0:
                continue
            mu = c.eval_mu(q_new) if callable(c.mu) else float(c.mu)
            r_F = self.prox_r_F[f_off]
            branch = int(fric_branch[f_off])
            if branch == 0:
                # r_F * gamma_F row: d/du_half ≈ r_F W_F^T.
                W_F_new = c.eval_W_F(t_new, q_new)
                W_F_new_dense = W_F_new.toarray() if sp.issparse(W_F_new) \
                    else np.asarray(W_F_new, dtype=float)
                J[row_F + f_off, nq:nq + nu] = r_F * W_F_new_dense[:, 0]
            elif branch == +1:
                J[row_F + f_off, col_PF + f_off] = 1.0
                J[row_F + f_off, col_PN + k] = -mu
            elif branch == -1:
                J[row_F + f_off, col_PF + f_off] = 1.0
                J[row_F + f_off, col_PN + k] = +mu
            else:  # sentinel: P_F = 0
                J[row_F + f_off, col_PF + f_off] = 1.0
            f_off += c.n_F

        return J

    def _ssn_stage1(self, q_n: np.ndarray, u_n: np.ndarray, P_N0: np.ndarray,
                    P_F0: np.ndarray, *, t_n: float, h: float,
                    B_n: Any, beta_n: np.ndarray, M_n: Any, x1_guess: Optional[np.ndarray]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
        """Stage 1 semismooth-Newton solve.

        Returns ``(x1, P_N, P_F, active, iterations, residual, converged)``.
        ``active`` is the stage-1 normal active set (for Stage 2).
        """
        nq, nu = self.mech.nq, self.mech.nu
        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F

        if x1_guess is None:
            x1 = np.zeros(self._nx1, dtype=float)
            x1[:nq] = q_n
            x1[nq:nq + nu] = u_n
        else:
            x1 = np.asarray(x1_guess, dtype=float).copy()
            x1[:nq] = q_n

        y = self._ssn_pack(x1, P_N0.copy(), P_F0.copy())

        q_new = y[:nq]
        u_half = y[nq:nq + nu]
        _, P_N_cur, P_F_cur = self._ssn_split(y)
        active_N, fric_branch = self._ssn_decide_active_sets(
            q_new, u_half, P_N_cur, P_F_cur, t_n + h, h,
        )

        res = self._ssn_residual(
            y, t_n=t_n, h=h, q_n=q_n, u_n=u_n, B_n=B_n, beta_n=beta_n, M_n=M_n,
            active_N=active_N, fric_branch=fric_branch,
        )
        res_norm = float(np.max(np.abs(res)))
        if res_norm <= self.newton_tol:
            return (y[:self._nx1], y[self._nx1:self._nx1 + nla_N],
                    y[self._nx1 + nla_N:], active_N, 0, res_norm, True)

        prev_sets: Optional[tuple[bytes, bytes]] = None
        for it in range(1, self.newton_max_iter + 1):
            J = self._ssn_jacobian(
                y, t_n=t_n, h=h, q_n=q_n, u_n=u_n, B_n=B_n, M_n=M_n,
                active_N=active_N, fric_branch=fric_branch,
            )
            try:
                step = np.linalg.solve(J, -res)
            except np.linalg.LinAlgError:
                return (y[:self._nx1], y[self._nx1:self._nx1 + nla_N],
                        y[self._nx1 + nla_N:], active_N, it, res_norm, False)

            alpha_ls = 1.0
            ls_it = 0
            y_trial = y + step
            q_trial = y_trial[:nq]
            u_trial = y_trial[nq:nq + nu]
            _, P_N_trial, P_F_trial = self._ssn_split(y_trial)
            active_trial, branch_trial = self._ssn_decide_active_sets(
                q_trial, u_trial, P_N_trial, P_F_trial, t_n + h, h,
            )
            res_trial = self._ssn_residual(
                y_trial, t_n=t_n, h=h, q_n=q_n, u_n=u_n,
                B_n=B_n, beta_n=beta_n, M_n=M_n,
                active_N=active_trial, fric_branch=branch_trial,
            )
            res_trial_norm = float(np.max(np.abs(res_trial)))
            while (res_trial_norm > res_norm and ls_it < self.ls_max
                   and alpha_ls > 1e-4):
                alpha_ls *= 0.5
                y_trial = y + alpha_ls * step
                q_trial = y_trial[:nq]
                u_trial = y_trial[nq:nq + nu]
                _, P_N_trial, P_F_trial = self._ssn_split(y_trial)
                active_trial, branch_trial = self._ssn_decide_active_sets(
                    q_trial, u_trial, P_N_trial, P_F_trial, t_n + h, h,
                )
                res_trial = self._ssn_residual(
                    y_trial, t_n=t_n, h=h, q_n=q_n, u_n=u_n,
                    B_n=B_n, beta_n=beta_n, M_n=M_n,
                    active_N=active_trial, fric_branch=branch_trial,
                )
                res_trial_norm = float(np.max(np.abs(res_trial)))
                ls_it += 1

            y = y_trial
            res = res_trial
            res_norm = res_trial_norm
            active_N = active_trial
            fric_branch = branch_trial
            # Detect active-set cycles: if the branches and residual are
            # unchanged for two iterations and ``res_norm`` is above tol, the
            # Clarke branch selection is oscillating on a boundary point.
            sets_sig = (active_N.tobytes(), fric_branch.tobytes())
            if res_norm <= self.newton_tol:
                converged = True
                return (y[:self._nx1], y[self._nx1:self._nx1 + nla_N],
                        y[self._nx1 + nla_N:], active_N, it, res_norm, converged)
            if prev_sets == sets_sig and ls_it == self.ls_max:
                break
            prev_sets = sets_sig

        return (y[:self._nx1], y[self._nx1:self._nx1 + nla_N],
                y[self._nx1 + nla_N:], active_N, self.newton_max_iter, res_norm, False)

    # ------------------------------------------------------------------
    #  Stage 2: saddle-point solve + velocity proximal maps
    # ------------------------------------------------------------------

    def _solve_stage2(self, t_new: float, h: float,
                      q_new: np.ndarray, u_half: np.ndarray,
                      u_n: np.ndarray, q_n: np.ndarray,
                      P_N2: np.ndarray, P_F2: np.ndarray,
                      active: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 2 linear saddle-point solve for u_{n+1}."""
        nu = self.mech.nu
        n_bg = self.sys.n_bilat_g
        n_bv = self.sys.n_bilat_gamma
        n_total = nu + n_bg + n_bv

        M_new = self.mech.eval_M(t_new, q_new)
        M_dense = M_new.toarray() if sp.issparse(M_new) else np.asarray(M_new, dtype=float)

        # RHS: momentum part.
        h_new = self.mech.eval_h(t_new, q_new, u_half)
        rhs_mom = _matvec(M_new, u_half) + 0.5 * h * h_new

        # Add contact impulses to RHS.
        if self.sys.has_contact:
            f_off = 0
            for k, c in enumerate(self.sys.contacts):
                W_N = c.eval_W_N(t_new, q_new)
                rhs_mom = rhs_mom + W_N * P_N2[k]
                W_F = c.eval_W_F(t_new, q_new)
                for j in range(c.n_F):
                    w_j = _asvec(W_F[:, j]) if not sp.issparse(W_F) \
                        else np.asarray(W_F[:, j].toarray(), dtype=float).ravel()
                    rhs_mom = rhs_mom + w_j * P_F2[f_off + j]
                f_off += c.n_F

        # Add algebraic compliance forces.
        if self.sys.has_algebraic:
            for a in self.sys.algebraics:
                W_c = a.eval_W_c(t_new, q_new)
                la_c2 = np.zeros(a.n_c, dtype=float)
                rhs_mom = rhs_mom + 0.5 * h * _matvec(W_c, la_c2)

        if n_bg == 0 and n_bv == 0:
            # No bilateral constraints: simple linear solve.
            u_new = _solve_linear(M_new, rhs_mom)
            return u_new, np.zeros(n_bg), np.zeros(n_bv)

        # Assemble saddle-point matrix.
        K = np.zeros((n_total, n_total), dtype=float)
        K[:nu, :nu] = M_dense
        rhs_full = np.zeros(n_total, dtype=float)
        rhs_full[:nu] = rhs_mom

        row = nu
        col_bg = nu
        col_bv = nu + n_bg
        g_off, gam_off = 0, 0
        for b in self.sys.bilaterals:
            # Position-level bilateral.
            W_g = b.eval_W_g(t_new, q_new)
            W_g_dense = W_g.toarray() if sp.issparse(W_g) else np.asarray(W_g, dtype=float)
            # K[mom, P_g2] = -W_g
            K[:nu, col_bg + g_off:col_bg + g_off + b.n_g] = -W_g_dense
            # K[g_eq, u] = g_dot_u = W_g^T
            g_dot_u = b.eval_g_dot_u(t_new, q_new)
            g_dot_u_dense = g_dot_u.toarray() if sp.issparse(g_dot_u) \
                else np.asarray(g_dot_u, dtype=float)
            K[row:row + b.n_g, :nu] = g_dot_u_dense
            # RHS: -chi_g
            rhs_full[row:row + b.n_g] = -b.eval_chi_g(t_new, q_new)
            row += b.n_g
            g_off += b.n_g

            # Velocity-level bilateral.
            if b.n_gamma > 0:
                W_gam = b.eval_W_gamma(t_new, q_new)
                W_gam_dense = W_gam.toarray() if sp.issparse(W_gam) \
                    else np.asarray(W_gam, dtype=float)
                K[:nu, col_bv + gam_off:col_bv + gam_off + b.n_gamma] = -W_gam_dense
                gamma_u = b.eval_gamma_u(t_new, q_new)
                gamma_u_dense = gamma_u.toarray() if sp.issparse(gamma_u) \
                    else np.asarray(gamma_u, dtype=float)
                K[row:row + b.n_gamma, :nu] = gamma_u_dense
                rhs_full[row:row + b.n_gamma] = -b.eval_chi_gamma(t_new, q_new)
                row += b.n_gamma
                gam_off += b.n_gamma

        sol = _solve_linear(K, rhs_full)
        u_new = sol[:nu]
        P_g2 = sol[nu:nu + n_bg]
        P_gam2 = sol[nu + n_bg:]
        return u_new, P_g2, P_gam2

    def _prox_stage2(self, t_new: float, q_new: np.ndarray,
                     u_new: np.ndarray, u_n: np.ndarray,
                     t_n: float, q_n: np.ndarray,
                     P_N_total: np.ndarray, P_F_total: np.ndarray,
                     active: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
        """Velocity-level proximal maps with Newton restitution."""
        P_N_new = np.zeros_like(P_N_total)
        P_F_new = np.zeros_like(P_F_total)
        f_off = 0
        for k, c in enumerate(self.sys.contacts):
            if not active[k]:
                f_off += c.n_F
                continue

            # Newton restitution: xi_N = g_N_dot(u_new) + e * g_N_dot(u_n).
            g_N_dot_new = c.eval_g_N_dot(t_new, q_new, u_new)
            g_N_dot_old = c.eval_g_N_dot(t_n, q_n, u_n)
            xi_N = g_N_dot_new + c.e * g_N_dot_old

            r_N = self.prox_r_N[k]
            prox_arg_N = r_N * xi_N - P_N_total[k]
            P_N_new[k] = max(0.0, -prox_arg_N)

            # Friction on velocity level.
            mu = c.eval_mu(q_new) if callable(c.mu) else float(c.mu)
            xi_F = c.eval_gamma_F(t_new, q_new, u_new)
            for j in range(c.n_F):
                r_F = self.prox_r_F[f_off + j]
                tang_arg = r_F * xi_F[j:j + 1] - P_F_total[f_off + j:f_off + j + 1]
                proj = _project_ball(tang_arg, mu * P_N_new[k])
                P_F_new[f_off + j] = -float(proj[0])
            f_off += c.n_F
        return P_N_new, P_F_new

    # ------------------------------------------------------------------
    #  Full step
    # ------------------------------------------------------------------

    def step(self, *, t_n: float, q_n: np.ndarray, u_n: np.ndarray,
             h: float,
             P_N_guess: Optional[np.ndarray] = None,
             P_F_guess: Optional[np.ndarray] = None,
             x1_guess: Optional[np.ndarray] = None,
             P_N2_guess: Optional[np.ndarray] = None,
             P_F2_guess: Optional[np.ndarray] = None,
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """One RATTLE time step.

        Returns
        -------
        q_new, u_new : ndarray
            Configuration and velocity at t_{n+1}.
        P_N_total, P_F_total : ndarray
            Total contact impulses (Stage 1 + Stage 2).
        info : dict
            Diagnostics.
        """
        nq, nu = self.mech.nq, self.mech.nu
        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F

        P_N = _asvec(P_N_guess) if P_N_guess is not None else np.zeros(nla_N)
        P_F = _asvec(P_F_guess) if P_F_guess is not None else np.zeros(nla_F)

        # Pre-compute operators at (t_n, q_n).
        B_n = self.mech.eval_B(t_n, q_n)
        beta_n = self.mech.eval_beta(t_n, q_n)
        M_n = self.mech.eval_M(t_n, q_n)

        # Delassus prox parameter estimation.
        self._compute_prox_params(t_n, q_n)

        # Warm-start Stage 1 from the previously converged stage state when
        # available. Cardillo's reference implementation carries x1 forward
        # between time steps, which is especially helpful on load ramps and
        # threshold stick/slip transitions.
        if x1_guess is None:
            x1 = np.zeros(self._nx1, dtype=float)
            x1[:nq] = q_n
            x1[nq:nq + nu] = u_n
        else:
            x1 = _asvec(x1_guess).copy()
            if x1.shape != (self._nx1,):
                raise ValueError(
                    f"x1_guess must have shape {(self._nx1,)}, got {x1.shape}."
                )
            x1[:nq] = q_n

        # ---- STAGE 1 ----
        if not self.sys.has_contact:
            # No contact: single Newton solve, no fixed-point.
            x1, newton_its, newton_err = self._newton_stage1(
                x1, P_N, P_F, t_n=t_n, h=h, q_n=q_n, u_n=u_n,
                B_n=B_n, beta_n=beta_n, M_n=M_n)
            if newton_err > self.newton_tol:
                return (q_n.copy(), u_n.copy(), P_N * 0, P_F * 0,
                        {"success": False, "status": "stage1_newton_failed",
                         "iterations": newton_its, "solver_error": newton_err,
                         "stage1_fixed_point_iterations": 0,
                         "stage2_fixed_point_iterations": 0})
            q_new, u_half = x1[:nq], x1[nq:nq + nu]
            active = np.zeros(self.sys.n_contacts, dtype=bool)
            s1_fp_its = 0
        elif self.stage1_method == "semismooth_newton":
            # Stacked semismooth-Newton Stage 1: drives g_N to roundoff via
            # minimum-map + active-set friction inside a single Newton solve.
            x1_ssn, P_N_new, P_F_new, active, ssn_its, ssn_err, ssn_ok = \
                self._ssn_stage1(
                    q_n, u_n, P_N, P_F, t_n=t_n, h=h,
                    B_n=B_n, beta_n=beta_n, M_n=M_n,
                    x1_guess=x1 if x1_guess is not None else None,
                )
            if not ssn_ok:
                return (q_n.copy(), u_n.copy(), P_N * 0, P_F * 0,
                        {"success": False,
                         "status": "stage1_semismooth_newton_failed",
                         "iterations": ssn_its,
                         "solver_error": float(ssn_err),
                         "stage1_fixed_point_iterations": ssn_its,
                         "stage2_fixed_point_iterations": 0})
            x1 = x1_ssn
            P_N = P_N_new
            P_F = P_F_new
            q_new = x1[:nq]
            u_half = x1[nq:nq + nu]
            newton_err = ssn_err
            s1_fp_its = ssn_its
        else:
            # Fixed-point iteration: Newton <-> Prox.
            active = np.zeros(self.sys.n_contacts, dtype=bool)
            s1_fp_its = 0
            stage1_err = np.inf
            newton_err_last = np.inf
            total_newton_its = 0
            for it1 in range(1, self.fp_max + 1):
                x1_new, n_its, n_err = self._newton_stage1(
                    x1, P_N, P_F, t_n=t_n, h=h, q_n=q_n, u_n=u_n,
                    B_n=B_n, beta_n=beta_n, M_n=M_n)
                total_newton_its += n_its
                newton_err_last = n_err
                q_cand = x1_new[:nq]
                u_half_cand = x1_new[nq:nq + nu]
                t_new = t_n + h

                P_N_new, P_F_new, active = self._prox_stage1(
                    q_cand, u_half_cand, P_N, P_F, t_new, h)

                # Convergence on smooth state.
                state_slice = x1_new[:self._n_smooth_state]
                state_slice_old = x1[:self._n_smooth_state]
                stage1_err = _wrms_norm(
                    state_slice - state_slice_old,
                    self.state_atol, self.state_rtol, state_slice_old)
                s1_fp_its = it1

                if stage1_err <= 1.0 and n_err <= self.newton_tol:
                    x1 = x1_new
                    P_N = P_N_new
                    P_F = P_F_new
                    break
                x1 = x1_new
                P_N = P_N_new
                P_F = P_F_new
            else:
                return (q_n.copy(), u_n.copy(), P_N * 0, P_F * 0,
                        {"success": False,
                         "status": "stage1_fixed_point_failed",
                         "iterations": total_newton_its,
                         "solver_error": float(stage1_err),
                         "stage1_fixed_point_iterations": s1_fp_its,
                         "stage2_fixed_point_iterations": 0})

            q_new = x1[:nq]
            u_half = x1[nq:nq + nu]
            newton_err = newton_err_last

        # ---- STAGE 2 ----
        P_N1 = P_N.copy()
        P_F1 = P_F.copy()
        t_new = t_n + h

        if not self.sys.has_contact:
            # No contact: single saddle-point solve.
            u_new, P_g2, P_gam2 = self._solve_stage2(
                t_new, h, q_new, u_half, u_n, q_n,
                np.zeros(nla_N), np.zeros(nla_F), active)
            s2_fp_its = 0
        else:
            # Fixed-point iteration: linear solve <-> velocity prox.
            P_N2 = _asvec(P_N2_guess).copy() if P_N2_guess is not None else np.zeros(nla_N, dtype=float)
            P_F2 = _asvec(P_F2_guess).copy() if P_F2_guess is not None else np.zeros(nla_F, dtype=float)
            if P_N2.shape != (nla_N,) or P_F2.shape != (nla_F,):
                raise ValueError(
                    f"Stage-2 impulse guesses must have shapes {(nla_N,)} and {(nla_F,)}, "
                    f"got {P_N2.shape} and {P_F2.shape}."
                )
            s2_fp_its = 0
            for it2 in range(1, self.fp_max + 1):
                u_new, P_g2, P_gam2 = self._solve_stage2(
                    t_new, h, q_new, u_half, u_n, q_n,
                    P_N2, P_F2, active)

                P_N_total = P_N1 + P_N2
                P_F_total = P_F1 + P_F2
                P_N_total_new, P_F_total_new = self._prox_stage2(
                    t_new, q_new, u_new, u_n, t_n, q_n,
                    P_N_total, P_F_total, active)

                P_N2_new = P_N_total_new - P_N1
                P_F2_new = P_F_total_new - P_F1

                # Convergence on velocity.
                s2_fp_its = it2
                err_N = float(np.max(np.abs(P_N2_new - P_N2))) if nla_N > 0 else 0.0
                err_F = float(np.max(np.abs(P_F2_new - P_F2))) if nla_F > 0 else 0.0
                if max(err_N, err_F) <= self.fp_tol:
                    P_N2 = P_N2_new
                    P_F2 = P_F2_new
                    break
                P_N2 = P_N2_new
                P_F2 = P_F2_new
            else:
                return (q_n.copy(), u_n.copy(), P_N * 0, P_F * 0,
                        {"success": False,
                         "status": "stage2_fixed_point_failed",
                         "iterations": s2_fp_its,
                         "solver_error": max(err_N, err_F),
                         "stage1_fixed_point_iterations": s1_fp_its,
                         "stage2_fixed_point_iterations": s2_fp_its})

            # Final velocity with converged impulses.
            u_new, P_g2, P_gam2 = self._solve_stage2(
                t_new, h, q_new, u_half, u_n, q_n,
                P_N2, P_F2, active)

        P_N_total = P_N1 + (P_N2 if self.sys.has_contact else np.zeros(nla_N))
        P_F_total = P_F1 + (P_F2 if self.sys.has_contact else np.zeros(nla_F))

        solver_err = float(newton_err)
        return q_new, u_new, P_N_total, P_F_total, {
            "success": True,
            "status": "ok",
            "iterations": s1_fp_its + s2_fp_its,
            "solver_error": solver_err,
            "stage1_fixed_point_iterations": s1_fp_its,
            "stage2_fixed_point_iterations": s2_fp_its,
            "step_true_residual_inf": solver_err,
            "step_true_residual_rms": solver_err,
            "x1_state": x1.copy(),
            "P_N1_state": P_N1.copy(),
            "P_F1_state": P_F1.copy(),
            "P_N2_state": P_N2.copy() if self.sys.has_contact else np.zeros(nla_N, dtype=float),
            "P_F2_state": P_F2.copy() if self.sys.has_contact else np.zeros(nla_F, dtype=float),
        }

    # ------------------------------------------------------------------
    #  Time loop
    # ------------------------------------------------------------------

    def solve(self, t_span: tuple[float, float], *, n_steps: int
              ) -> RattleSolveResult:
        if int(n_steps) < 1:
            raise ValueError("n_steps must be at least 1.")
        t0, tf = float(t_span[0]), float(t_span[1])
        h = (tf - t0) / float(n_steps)
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(
                f"Invalid step size from t_span={t_span!r}, n_steps={n_steps!r}.")

        nla_N = self.sys.nla_N
        nla_F = self.sys.nla_F
        nq, nu = self.mech.nq, self.mech.nu
        n_react = nla_N + nla_F
        initial_force = np.concatenate(
            [
                np.asarray(self.sys.initial_normal_forces, dtype=float),
                np.asarray(self.sys.initial_friction_forces, dtype=float),
            ]
        )
        initial_impulse = initial_force * h

        times = [t0]
        states = [np.concatenate([self.mech.q0, self.mech.u0])]
        react_impulses = [initial_impulse.copy()]
        react_forces = [initial_force.copy()]
        step_success_list: list[bool] = []
        step_iters: list[int] = []
        step_errs: list[float] = []
        step_res_inf: list[float] = []
        step_res_rms: list[float] = []
        s1_its: list[int] = []
        s2_its: list[int] = []
        failure = None

        q_n = self.mech.q0.copy()
        u_n = self.mech.u0.copy()
        t_n = t0
        P_N_prev: Optional[np.ndarray] = initial_impulse[:nla_N].copy()
        P_F_prev: Optional[np.ndarray] = initial_impulse[nla_N:].copy()
        x1_prev: Optional[np.ndarray] = None
        P_N2_prev = np.zeros(nla_N, dtype=float)
        P_F2_prev = np.zeros(nla_F, dtype=float)

        for step_idx in range(int(n_steps)):
            q_new, u_new, P_N_tot, P_F_tot, info = self.step(
                t_n=t_n, q_n=q_n, u_n=u_n, h=h,
                P_N_guess=P_N_prev, P_F_guess=P_F_prev,
                x1_guess=x1_prev,
                P_N2_guess=P_N2_prev,
                P_F2_guess=P_F2_prev)
            success = bool(info.get("success", False))
            step_success_list.append(success)
            step_iters.append(int(info.get("iterations", 0)))
            step_errs.append(float(info.get("solver_error", np.nan)))
            step_res_inf.append(float(info.get("step_true_residual_inf", np.nan)))
            step_res_rms.append(float(info.get("step_true_residual_rms", np.nan)))
            s1_its.append(int(info.get("stage1_fixed_point_iterations", 0)))
            s2_its.append(int(info.get("stage2_fixed_point_iterations", 0)))

            if not success:
                failure = {"step_index": step_idx,
                           "status": info.get("status"),
                           "iterations": info.get("iterations"),
                           "solver_error": info.get("solver_error")}
                break

            t_n = t_n + h
            q_n = q_new
            u_n = u_new
            P_N_prev = np.asarray(info.get("P_N1_state"), dtype=float).copy()
            P_F_prev = np.asarray(info.get("P_F1_state"), dtype=float).copy()
            x1_prev = np.asarray(info.get("x1_state"), dtype=float).copy()
            P_N2_prev = np.asarray(info.get("P_N2_state"), dtype=float).copy()
            P_F2_prev = np.asarray(info.get("P_F2_state"), dtype=float).copy()
            p_total = np.concatenate([P_N_tot, P_F_tot])
            times.append(t_n)
            states.append(np.concatenate([q_n, u_n]))
            react_impulses.append(p_total.copy())
            react_forces.append(p_total / h)

        return RattleSolveResult(
            times=np.asarray(times, dtype=float),
            states=np.asarray(states, dtype=float),
            step_sizes=np.full(max(len(times) - 1, 0), h, dtype=float),
            reaction_force_history=np.asarray(react_forces, dtype=float),
            reaction_impulse_history=np.asarray(react_impulses, dtype=float),
            step_success=np.asarray(step_success_list, dtype=bool),
            step_iterations=np.asarray(step_iters, dtype=int),
            step_solver_error=np.asarray(step_errs, dtype=float),
            step_true_residual_inf=np.asarray(step_res_inf, dtype=float),
            step_true_residual_rms=np.asarray(step_res_rms, dtype=float),
            stage1_fixed_point_iterations=np.asarray(s1_its, dtype=int),
            stage2_fixed_point_iterations=np.asarray(s2_its, dtype=int),
            failure=failure,
        )


# ---------------------------------------------------------------------------
#  Factory: Cardillo-style native interface
# ---------------------------------------------------------------------------

def build_rattle_system(
    mech: RattleMechanicalSystem,
    *,
    contacts: Optional[list[RattleContactSpec]] = None,
    bilaterals: Optional[list[RattleBilateralSpec]] = None,
    algebraics: Optional[list[RattleAlgebraicSpec]] = None,
    prox_alpha: float = 0.5,
    prox_r_min: float = 1.0e-8,
    prox_r_max: float = 1.0e8,
    gap_tol: float = 0.0,
    initial_normal_forces: Optional[np.ndarray] = None,
    initial_friction_forces: Optional[np.ndarray] = None,
) -> RattleContactSystem:
    """Build a RATTLE system from Cardillo-style mechanical description."""
    return RattleContactSystem(
        mech=mech,
        contacts=contacts or [],
        bilaterals=bilaterals or [],
        algebraics=algebraics or [],
        prox_alpha=float(prox_alpha),
        prox_r_min=float(prox_r_min),
        prox_r_max=float(prox_r_max),
        gap_tol=float(gap_tol),
        initial_normal_forces=initial_normal_forces,
        initial_friction_forces=initial_friction_forces,
    )


# ---------------------------------------------------------------------------
#  Legacy adapter: first-order DAE interface -> Cardillo RATTLE
# ---------------------------------------------------------------------------

def build_dynamic_rattle_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    *,
    B,
    gap_extract,
    vel_extract,
    n_base: int,
    velocity_slice: slice,
    component_slices=None,
    rhs_jac=None,
    gap_tol: float = 0.0,
    prox_scaling: float = 1.0,
    prox_r_min: float = 1.0e-8,
    prox_r_max: float = 1.0e8,
) -> RattleContactSystem:
    """Create a RATTLE system from the legacy first-order DAE interface.

    Accepts the same arguments as the old ``build_dynamic_rattle_contact``
    and converts to the Cardillo-style ``(M, h, B, q, u)`` formulation.
    """
    y0 = _asvec(y0)
    n_phys = y0.size
    nq = int(n_base)
    vs = slice(int(velocity_slice.start), int(velocity_slice.stop))
    nu = vs.stop - vs.start

    q0 = y0[:nq]
    u0 = y0[vs]

    A_mat = A
    if sp.issparse(A_mat):
        A_mat = A_mat.tocsr()

    # Extract M from momentum rows of A.
    mom_rows = np.arange(vs.start, vs.stop, dtype=int)
    if sp.issparse(A_mat):
        M_const = A_mat[mom_rows, :][:, vs].tocsc()
    else:
        A_dense = np.asarray(A_mat, dtype=float)
        M_const = A_dense[np.ix_(mom_rows, np.arange(vs.start, vs.stop))]

    # Build h_force(t, q, u) from momentum rows of rhs_smooth(t, y).
    def h_force(t, q, u):
        y = np.zeros(n_phys, dtype=float)
        y[:nq] = q
        y[vs] = u
        rhs_val = _asvec(rhs_smooth(t, y))
        return rhs_val[mom_rows]

    # Optional Jacobians.
    dh_dq_fn = None
    dh_du_fn = None
    if rhs_jac is not None:
        def dh_du_fn(t, q, u):
            y = np.zeros(n_phys, dtype=float)
            y[:nq] = q
            y[vs] = u
            J_full = rhs_jac(t, y)
            if sp.issparse(J_full):
                return J_full.tocsr()[mom_rows, :][:, vs]
            return np.asarray(J_full, dtype=float)[np.ix_(mom_rows, np.arange(vs.start, vs.stop))]

        def dh_dq_fn(t, q, u):
            y = np.zeros(n_phys, dtype=float)
            y[:nq] = q
            y[vs] = u
            J_full = rhs_jac(t, y)
            if sp.issparse(J_full):
                return J_full.tocsr()[mom_rows, :][:, :nq]
            return np.asarray(J_full, dtype=float)[np.ix_(mom_rows, np.arange(nq))]

    mech = RattleMechanicalSystem(
        nq=nq, nu=nu, q0=q0, u0=u0,
        M=M_const, h_force=h_force,
        B_kin=None,   # identity kinematic map for this interface
        beta=None,
        dh_dq=dh_dq_fn, dh_du=dh_du_fn,
    )

    # Convert contact dicts to RattleContactSpec.
    if sp.issparse(gap_extract):
        gap_extract_csr = gap_extract.tocsr()
    else:
        gap_extract_csr = np.asarray(gap_extract, dtype=float)
    if sp.issparse(vel_extract):
        vel_extract_csr = vel_extract.tocsr()
    else:
        vel_extract_csr = np.asarray(vel_extract, dtype=float)

    B_mat = B
    if sp.issparse(B_mat):
        B_csc = B_mat.tocsc()
    else:
        B_csc = np.asarray(B_mat, dtype=float)

    contact_specs = []
    col = 0
    for cd in contacts:
        normal_idx = int(cd["vel_normal_idx"])
        tang_idx = np.asarray(cd["vel_tangential_idx"], dtype=int).ravel()
        mu_val = cd.get("mu", 0.0)
        e_val = cd.get("e", 0.0)
        n_F = int(tang_idx.size)
        block_dim = 1 + n_F

        # Gap function: g_N(t, q) -- extract from gap_extract restricted to config columns.
        row_n = normal_idx

        def _make_gap(r=row_n):
            def gap_fn(t, q):
                y = np.zeros(n_phys, dtype=float)
                y[:nq] = q
                gap_raw = _matvec(gap_extract_csr, y)
                return -float(gap_raw[r])
            return gap_fn

        # W_N: column of B corresponding to normal impulse.
        b_col_n = col

        def _make_W_N(c=b_col_n):
            if sp.issparse(B_csc):
                w = np.asarray(B_csc[:, c].toarray(), dtype=float).ravel()
            else:
                w = B_csc[:, c].copy()
            # Restrict to velocity DOFs since RATTLE W_N is (nu,).
            return w[vs]

        # gamma_F: tangential slip velocity.
        def _make_gamma_F(rows_t=tang_idx.copy()):
            def gamma_F_fn(t, q, u):
                y = np.zeros(n_phys, dtype=float)
                y[:nq] = q
                y[vs] = u
                vel_raw = _matvec(vel_extract_csr, y)
                return vel_raw[rows_t]
            return gamma_F_fn

        # W_F: columns of B corresponding to tangential impulses.
        b_cols_t = list(range(col + 1, col + block_dim))

        def _make_W_F(cols=b_cols_t):
            if sp.issparse(B_csc):
                w = np.asarray(B_csc[:, cols].toarray(), dtype=float)
            else:
                w = B_csc[:, cols].copy()
            return w[vs, :]

        spec = RattleContactSpec(
            g_N=_make_gap(),
            W_N=_make_W_N(),
            gamma_F=_make_gamma_F(),
            W_F=_make_W_F(),
            mu=mu_val,
            e=e_val,
            n_F=n_F,
        )
        contact_specs.append(spec)
        col += block_dim

    return RattleContactSystem(
        mech=mech,
        contacts=contact_specs,
        bilaterals=[],
        algebraics=[],
        prox_alpha=float(prox_scaling),
        prox_r_min=float(prox_r_min),
        prox_r_max=float(prox_r_max),
        gap_tol=float(gap_tol),
    )


def solve_dynamic_rattle_contact(
    system: RattleContactSystem,
    t_span: tuple[float, float],
    *,
    n_steps: int,
    solver_opts: Optional[dict[str, Any]] = None,
) -> RattleSolveResult:
    """Solve one fixed-step trajectory with the RATTLE backend."""
    opts = {} if solver_opts is None else dict(solver_opts)
    solver = RattleSolver(system, **opts)
    return solver.solve(t_span, n_steps=int(n_steps))
