"""Breuling-style projected Radau contact helpers.

This module implements an opt-in contact model for the two-stage Radau IIA
integrator.  The stage contact laws are evaluated on accumulated stage
percussions ``dmu = A @ dPi`` rather than on local stage reaction rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .alart_curnier_contact import (
    _call_state_time_fk,
    _call_with_time_state_fk,
    _count_required_args,
    _dense_or_sparse,
    _eval_s0,
    _eval_w0,
    _vectorize_mu,
)
from .contact import ContactSystem
from .ncp_contact import (
    _contact_block_residual_and_jac,
    _eval_contact_scalar_field,
    _normalize_ncp_name,
)
from .projections import (
    AlgebraicConstraintProjection,
    IdentityProjection,
    MuScaledSOCProjection,
)


def _sparse_index_array(indexer, n: int) -> np.ndarray:
    """Return integer indices for a row/column indexer without touching data."""
    if isinstance(indexer, slice):
        start, stop, step = indexer.indices(n)
        return np.arange(start, stop, step, dtype=int)
    arr = np.asarray(indexer, dtype=int).ravel()
    if arr.size and np.any(arr < 0):
        arr = arr.copy()
        arr[arr < 0] += n
    return arr


def _zero_sparse_columns(M, cols) -> sp.csr_matrix:
    """Zero selected sparse columns without converting the whole matrix to LIL."""
    M_csc = M.tocsc(copy=True)
    cols = np.asarray(cols, dtype=int).ravel()
    if cols.size == 0:
        return M_csc.tocsr()
    for col in cols:
        start, stop = M_csc.indptr[int(col)], M_csc.indptr[int(col) + 1]
        M_csc.data[start:stop] = 0.0
    M_csc.eliminate_zeros()
    return M_csc.tocsr()


def _replace_sparse_rows(M, rows, replacement) -> sp.csr_matrix:
    """Return M with selected rows replaced by rows from replacement.

    This avoids CSR->LIL conversion for the common case where only a few
    algebraic/contact rows must be overwritten in a large sparse Jacobian.
    """
    M_csr = M.tocsr(copy=True)
    repl = replacement.tocsr() if sp.issparse(replacement) else sp.csr_matrix(replacement)
    if M_csr.shape != repl.shape:
        raise ValueError(
            f"replacement shape {repl.shape} does not match matrix shape {M_csr.shape}"
        )

    row_arrays = []
    for item in rows:
        row_idx = _sparse_index_array(item, M_csr.shape[0])
        if row_idx.size:
            row_arrays.append(row_idx)
    if not row_arrays:
        return M_csr
    row_idx = np.unique(np.concatenate(row_arrays))

    indptr = M_csr.indptr
    for row in row_idx:
        M_csr.data[indptr[row]:indptr[row + 1]] = 0.0
    M_csr.eliminate_zeros()

    repl_sel = repl[row_idx, :].tocoo(copy=False)
    if repl_sel.nnz:
        patch = sp.csr_matrix(
            (repl_sel.data, (row_idx[repl_sel.row], repl_sel.col)),
            shape=M_csr.shape,
        )
        M_csr = M_csr + patch
        M_csr.eliminate_zeros()
    return M_csr.tocsr()


class ProjectedRadauContactLaw:
    """Small interface for normal-cone contact laws used by projected Radau.

    Attributes
    ----------
    expects_velocity_normal : bool
        If True, the law is a coupled Lorentz-cone formulation that requires
        the normal kinematic input to share units with the tangential block
        (both velocities).  The stage residual then uses a Moreau viability
        gate (Acary-Brogliato sec. 5.2 / 11): if ``gap > gap_tol`` the contact
        is treated as inactive and the law is bypassed; otherwise the law is
        evaluated with ``normal_quantity`` set to the velocity-level normal
        component of ``contact_velocity``.  Default ``False`` preserves the
        Breuling product-cone stage formulation used by NCP scalar laws.
    """

    expects_velocity_normal: bool = False

    def residual_and_jac(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def residual(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> np.ndarray:
        return self.residual_and_jac(
            normal_quantity,
            contact_velocity,
            percussion,
            mu,
            normal_scale,
            friction_scale,
        )[0]


class NCPNormalConeLaw(ProjectedRadauContactLaw):
    """NCP realization of the Breuling normal-cone inclusions."""

    def __init__(
        self,
        *,
        ncp_type: str = "fischer_burmeister",
        normal_ncp_type: Optional[str] = None,
        friction_ncp_type: Optional[str] = None,
        friction_law: str = "compliance",
    ) -> None:
        ncp_type = _normalize_ncp_name(ncp_type, label="contact_law")
        self.normal_ncp_type = _normalize_ncp_name(
            ncp_type if normal_ncp_type is None else normal_ncp_type,
            label="normal_ncp_type",
        )
        self.friction_ncp_type = _normalize_ncp_name(
            ncp_type if friction_ncp_type is None else friction_ncp_type,
            label="friction_ncp_type",
        )
        self.friction_law = str(friction_law).strip().lower().replace("-", "_")
        if self.friction_law not in {"compliance", "natural_map"}:
            raise ValueError(
                "friction_law must be 'compliance' or 'natural_map' "
                f"(got {friction_law!r})"
            )

    def residual_and_jac(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return _contact_block_residual_and_jac(
            normal_quantity,
            contact_velocity,
            percussion,
            mu,
            self.normal_ncp_type,
            self.friction_ncp_type,
            normal_scale,
            friction_scale,
            self.friction_law,
        )


def _soc_fb_phi(x: np.ndarray, y: np.ndarray, *, tie_tol: float = 1.0e-14) -> np.ndarray:
    r"""Second-order-cone Fischer-Burmeister function.

    Implements eq. (76)/(77) of Acary-Bremond-Huber (Jordan-algebra form):

        Φ_FB(x, y) = x + y − (x² + y²)^{1/2}

    where products and square root are taken in the Jordan algebra
    associated with the second-order cone.  Zeros of this map are
    exactly the SOCCP solutions  K ∋ x ⊥ y ∈ K.

    The Jordan product on ℝ × ℝ^(d-1) is
        x · y = (x^T y, y_N x_T + x_N y_T),
    so x² = (‖x‖², 2 x_N x_T).  For w ∈ K, the spectral decomposition
    yields  w^{1/2} = √λ_1 u_1 + √λ_2 u_2  with
        λ_i = w_N + (-1)^i ‖w_T‖,
        u_i = ½ (1, (-1)^i w_T/‖w_T‖).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    d = x.size
    if y.size != d:
        raise ValueError(f"x, y must have the same size (got {x.size}, {y.size})")

    w_N = float(x @ x + y @ y)
    if d == 1:
        w_T = np.zeros(0, dtype=float)
    else:
        w_T = 2.0 * (x[0] * x[1:] + y[0] * y[1:])
    w_T_norm = float(np.linalg.norm(w_T)) if d > 1 else 0.0

    lam1 = max(w_N - w_T_norm, 0.0)
    lam2 = max(w_N + w_T_norm, 0.0)
    sqrt_lam1 = np.sqrt(lam1)
    sqrt_lam2 = np.sqrt(lam2)
    s_N = 0.5 * (sqrt_lam1 + sqrt_lam2)

    phi = np.empty(d, dtype=float)
    phi[0] = x[0] + y[0] - s_N
    if d > 1:
        if w_T_norm > tie_tol:
            w_T_hat = w_T / w_T_norm
        else:
            w_T_hat = np.zeros_like(w_T)
        s_T = 0.5 * (sqrt_lam2 - sqrt_lam1) * w_T_hat
        phi[1:] = x[1:] + y[1:] - s_T
    return phi


def _soc_jordan_multiplication(x: np.ndarray) -> np.ndarray:
    r"""Jordan multiplication matrix on ℝ × ℝ^(d-1):

        L_x = [[ x_N        x_T^T          ],
               [ x_T        x_N · I_{d-1}  ]].

    The Jordan product is then  x · y = L_x y .
    """
    x = np.asarray(x, dtype=float).ravel()
    d = x.size
    L = np.zeros((d, d), dtype=float)
    L[0, 0] = x[0]
    if d > 1:
        L[0, 1:] = x[1:]
        L[1:, 0] = x[1:]
        L[1:, 1:] = x[0] * np.eye(d - 1)
    return L


def _soc_fb_phi_and_jac_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tie_tol: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Closed-form d = 2 specialization of :func:`_soc_fb_phi_and_jac`.

    On ℝ × ℝ the Jordan-multiplication matrices are arrow matrices
    ``L_u = [[u0, u1], [u1, u0]]`` with eigenpairs ``(u0 ± u1, (1, ±1)/√2)``,
    so ``L_s⁻¹`` and the Moore-Penrose pseudoinverse used at the cone
    boundary both have two-term closed forms.  Scalar arithmetic replaces
    the generic spectral machinery; eigenvalue thresholding mirrors
    ``np.linalg.pinv(L_s, rcond=tie_tol)`` so degenerate-case Clarke
    elements coincide with the generic path.
    """
    x0, x1 = float(x[0]), float(x[1])
    y0, y1 = float(y[0]), float(y[1])

    w0 = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1
    w1 = 2.0 * (x0 * x1 + y0 * y1)
    abs_w1 = abs(w1)
    sqrt_lam1 = np.sqrt(max(w0 - abs_w1, 0.0))
    sqrt_lam2 = np.sqrt(max(w0 + abs_w1, 0.0))
    s0 = 0.5 * (sqrt_lam1 + sqrt_lam2)
    if abs_w1 > tie_tol:
        s1 = 0.5 * (sqrt_lam2 - sqrt_lam1) * (1.0 if w1 >= 0.0 else -1.0)
    else:
        s1 = 0.0
    phi = np.array([x0 + y0 - s0, x1 + y1 - s1])

    interior = (s0 > abs(s1) + tie_tol) and (s0 > tie_tol)
    if interior:
        det = s0 * s0 - s1 * s1
        i00 = s0 / det
        i01 = -s1 / det
    else:
        # Eigenvalues of L_s are s0 ± s1 >= 0 (s lies in the cone);
        # threshold against rcond * lambda_max exactly as np.linalg.pinv.
        lam_p = s0 + s1
        lam_m = s0 - s1
        cutoff = tie_tol * max(lam_p, lam_m)
        c_p = 1.0 / lam_p if lam_p > cutoff else 0.0
        c_m = 1.0 / lam_m if lam_m > cutoff else 0.0
        i00 = 0.5 * (c_p + c_m)
        i01 = 0.5 * (c_p - c_m)

    df_dx = np.array([
        [1.0 - (i00 * x0 + i01 * x1), -(i00 * x1 + i01 * x0)],
        [-(i01 * x0 + i00 * x1), 1.0 - (i01 * x1 + i00 * x0)],
    ])
    df_dy = np.array([
        [1.0 - (i00 * y0 + i01 * y1), -(i00 * y1 + i01 * y0)],
        [-(i01 * y0 + i00 * y1), 1.0 - (i01 * y1 + i00 * y0)],
    ])
    return phi, df_dx, df_dy


def _soc_fb_phi_and_jac(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tie_tol: float = 1.0e-14,
    fd_eps: float = 1.0e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""SOC FB residual + analytical Jordan-algebra Jacobians.

    Φ(x, y) = x + y − s    where s = (x² + y²)^{1/2}.
    Differentiating  s² = w  gives  2 s · ds = dw,  hence

        ∂s/∂x = L_s⁻¹ L_x,    ∂s/∂y = L_s⁻¹ L_y,

    with L_x the Jordan multiplication matrix.  Therefore

        ∂Φ/∂x = I − L_s⁻¹ L_x,    ∂Φ/∂y = I − L_s⁻¹ L_y.

    L_s is invertible iff s ∈ int(K) (s_N > ‖s_T‖); the only degenerate
    case is s = 0  ⇔  w = 0  ⇔  x = y = 0, where we fall back to a finite
    difference (the Clarke subdifferential there is the convex hull of
    {0, I}, see Acary-Bremond-Huber eq. 154 — FD picks one element of it).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    d = x.size
    if d == 2:
        return _soc_fb_phi_and_jac_2d(x, y, tie_tol=tie_tol)

    # Residual + s, computed once.
    phi = _soc_fb_phi(x, y, tie_tol=tie_tol)
    s = (x + y) - phi  # equals (x² + y²)^{1/2}

    L_s = _soc_jordan_multiplication(s)
    L_x = _soc_jordan_multiplication(x)
    L_y = _soc_jordan_multiplication(y)
    eye = np.eye(d, dtype=float)

    s_N = float(s[0])
    s_T_norm = float(np.linalg.norm(s[1:])) if d > 1 else 0.0
    interior = (s_N > s_T_norm + tie_tol) and (s_N > tie_tol)

    if interior:
        # Smooth case: L_s is invertible, ∂s/∂w = (2 L_s)⁻¹  ⇒
        # ∂Φ/∂x = I − L_s⁻¹ L_x,  ∂Φ/∂y = I − L_s⁻¹ L_y.
        Ls_inv_Lx = np.linalg.solve(L_s, L_x)
        Ls_inv_Ly = np.linalg.solve(L_s, L_y)
    else:
        # Degenerate cases:
        #   (a) boundary  s ∈ ∂K  (λ_1(w) = 0, λ_2(w) > 0):
        #       L_s has a 1D null space along the cone-tangent direction.
        #       The Moore-Penrose pseudoinverse picks the minimum-norm
        #       Clarke subgradient — a specific, deterministic element of
        #       the boundary subdifferential analogous to eq. (154).
        #   (b) origin   s = 0   (λ_1(w) = λ_2(w) = 0):
        #       L_s = 0  ⇒  L_s⁺ = 0  ⇒  ∂Φ/∂x = I − 0 = I, which is the
        #       upper-bound element of co{0, I} from the (0, 0) row of (154).
        Ls_pinv = np.linalg.pinv(L_s, rcond=tie_tol)
        Ls_inv_Lx = Ls_pinv @ L_x
        Ls_inv_Ly = Ls_pinv @ L_y

    df_dx = eye - Ls_inv_Lx
    df_dy = eye - Ls_inv_Ly
    return phi, df_dx, df_dy


class SOCFischerBurmeisterLaw(ProjectedRadauContactLaw):
    """Acary-Bremond-Huber eq. (79): SOC Fischer-Burmeister with De Saxce coupling.

    Implements

        F_FB(u, r) = Φ_FB( T_x · û , T_y · r ) = 0,

    where the De Saxce bipotential correction is baked into  û  via
        û_N = u_N + α‖u_T‖,    û_T = u_T,
    the self-dual rescaling  T_x = diag(1, μI),  T_y = diag(μ, I)
    sends the friction cone to the symmetric Lorentz cone, and Φ_FB is
    the Jordan-algebra SOC Fischer-Burmeister function.  Unlike the
    natural map  R = P_K(R − ρ·û)  used by ``DeSaxceProjectedConeLaw``,
    Φ_FB is ρ-free and smooth on its domain (sharp only at x = y = 0),
    so Newton iterates do not stall on the set-valued kink at u_T = 0.

    The Jordan-algebra products in Φ_FB require ``û`` to be a homogeneous
    kinematic vector, so the stage and endpoint residuals are evaluated at
    velocity level (``expects_velocity_normal = True``).  Position-level
    feasibility is recovered between steps via Moreau viability: when
    ``gap > gap_tol`` the contact is inactive and the law is not invoked.
    """

    expects_velocity_normal = True

    def __init__(
        self,
        *,
        alpha: Optional[float] = None,
        tie_tol: float = 1.0e-14,
        fd_eps: float = 1.0e-7,
    ) -> None:
        self.alpha = None if alpha is None else float(alpha)
        self.tie_tol = float(tie_tol)
        self.fd_eps = float(fd_eps)

    def residual_and_jac(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_contact = np.asarray(contact_velocity, dtype=float).ravel()
        r_blk = np.asarray(percussion, dtype=float).ravel()
        d = r_blk.size
        if d == 0:
            raise ValueError("contact block must contain at least one normal row")
        if u_contact.size != d:
            raise ValueError(
                "contact_velocity and percussion must have the same block size "
                f"(got {u_contact.size} and {d})"
            )

        mu = float(mu)
        if mu < 0.0:
            raise ValueError(f"mu must be nonnegative for SOC FB contact (got {mu})")
        alpha = mu if self.alpha is None else float(self.alpha)
        if alpha < 0.0:
            raise ValueError(
                f"alpha must be nonnegative for SOC FB contact (got {alpha})"
            )

        # --- Build u_hat with De Saxce correction (eq. 67/79). ------------
        u_hat = np.zeros(d, dtype=float)
        u_hat[0] = float(normal_quantity)
        if d > 1:
            u_hat[1:] = u_contact[1:]

        D_uhat = np.eye(d, dtype=float)
        if d > 1 and alpha > self.tie_tol:
            u_t = u_hat[1:]
            speed = float(np.linalg.norm(u_t))
            u_hat[0] += alpha * speed
            if speed > self.tie_tol:
                D_uhat[0, 1:] = alpha * (u_t / speed)
            else:
                r_t = r_blk[1:]
                r_t_norm = float(np.linalg.norm(r_t))
                if r_t_norm > self.tie_tol:
                    D_uhat[0, 1:] = -alpha * (r_t / r_t_norm)

        # --- Frictionless / scalar normal-only fallback. ------------------
        if d == 1 or mu <= self.tie_tol:
            # SOC FB collapses to scalar FB on (u_hat, r) for the normal
            # complementarity 0 ≤ u_hat_N ⊥ r_N ≥ 0.
            a = float(r_blk[0])
            b = float(u_hat[0])
            rad = float(np.hypot(a, b))
            phi = a + b - rad
            f_blk = r_blk.copy()
            f_blk[0] = phi
            if d > 1:
                f_blk[1:] = r_blk[1:]
            df_dnormal = np.zeros(d, dtype=float)
            df_du = np.zeros((d, d), dtype=float)
            df_dr = np.eye(d, dtype=float)
            if rad > self.tie_tol:
                dphi_da = 1.0 - a / rad
                dphi_db = 1.0 - b / rad
            else:
                dphi_da, dphi_db = 0.0, 0.0
            df_dr[0, 0] = dphi_da
            df_dnormal[0] = dphi_db
            return f_blk, df_dnormal, df_du, df_dr

        # --- Self-dual transformation (eq. 68). ---------------------------
        T_x = np.eye(d, dtype=float)
        T_x[1:, 1:] *= mu
        T_y = np.eye(d, dtype=float)
        T_y[0, 0] = mu
        T_y_inv = np.eye(d, dtype=float)
        T_y_inv[0, 0] = 1.0 / mu

        x_sd = T_x @ u_hat
        y_sd = T_y @ r_blk

        # --- SOC Fischer-Burmeister residual + Jacobians. -----------------
        phi_sd, dphi_dx, dphi_dy = _soc_fb_phi_and_jac(
            x_sd, y_sd, tie_tol=self.tie_tol, fd_eps=self.fd_eps,
        )

        # --- Pull back to physical (untransformed) reaction frame. --------
        # f_blk = T_y_inv · Φ( T_x · û(u, normal), T_y · r )
        # df_blk/d(normal_quantity) = T_y_inv · dphi_dx · T_x · ∂û/∂normal
        # df_blk/du                  = T_y_inv · dphi_dx · T_x · ∂û/∂u
        # df_blk/dr                  = T_y_inv · dphi_dy · T_y
        chain_x = T_y_inv @ dphi_dx @ T_x        # (d, d)
        df_blk_duhat = chain_x @ D_uhat          # ∂f / ∂(u_hat in physical frame)
        df_dnormal = df_blk_duhat[:, 0].copy()
        df_du = np.zeros((d, d), dtype=float)
        if d > 1:
            df_du[:, 1:] = df_blk_duhat[:, 1:]
        df_dr = T_y_inv @ dphi_dy @ T_y
        f_blk = T_y_inv @ phi_sd
        return f_blk, df_dnormal, df_du, df_dr

    def residual(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> np.ndarray:
        u_contact = np.asarray(contact_velocity, dtype=float).ravel()
        r_blk = np.asarray(percussion, dtype=float).ravel()
        d = r_blk.size
        if d == 0:
            raise ValueError("contact block must contain at least one normal row")
        if u_contact.size != d:
            raise ValueError(
                "contact_velocity and percussion must have the same block size "
                f"(got {u_contact.size} and {d})"
            )

        mu = float(mu)
        if mu < 0.0:
            raise ValueError(f"mu must be nonnegative for SOC FB contact (got {mu})")
        alpha = mu if self.alpha is None else float(self.alpha)
        if alpha < 0.0:
            raise ValueError(
                f"alpha must be nonnegative for SOC FB contact (got {alpha})"
            )

        u_hat = np.zeros(d, dtype=float)
        u_hat[0] = float(normal_quantity)
        if d > 1:
            u_hat[1:] = u_contact[1:]
            if alpha > self.tie_tol:
                u_hat[0] += alpha * float(np.linalg.norm(u_hat[1:]))

        if d == 1 or mu <= self.tie_tol:
            a = float(r_blk[0])
            b = float(u_hat[0])
            f_blk = r_blk.copy()
            f_blk[0] = a + b - float(np.hypot(a, b))
            return f_blk

        x_sd = u_hat.copy()
        x_sd[1:] *= mu
        y_sd = r_blk.copy()
        y_sd[0] *= mu
        phi_sd = _soc_fb_phi(x_sd, y_sd, tie_tol=self.tie_tol)
        phi_sd[0] /= mu
        return phi_sd


class DeSaxceProjectedConeLaw(ProjectedRadauContactLaw):
    """Self-dual De Saxce natural-map realization of the cone law.

    Algebraic form:  R = P_{K_μ}(R − ρ · û)  on the rescaled self-dual
    variables ``x = (u_hat_N, mu*u_hat_T)`` and ``y = (mu*r_N, r_T)``.
    The natural-map projection couples normal and tangential components
    through the Coulomb cone, so ``û`` must be a homogeneous kinematic
    vector — the stage and endpoint residuals are evaluated at velocity
    level (``expects_velocity_normal = True``) and Moreau viability gates
    inactive contacts (``gap > gap_tol``) before the law is invoked.
    """

    expects_velocity_normal = True

    def __init__(
        self,
        *,
        alpha: Optional[float] = None,
        rho: Any = "auto",
        rho_scale: float = 1.0,
        rho_min: float = 1.0e-12,
        rho_max: float = 1.0e12,
        tie_tol: float = 1.0e-14,
    ) -> None:
        self.alpha = None if alpha is None else float(alpha)
        self.rho = rho
        self.rho_scale = float(rho_scale)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.tie_tol = float(tie_tol)

    def _rho_value(self, normal_scale: float, friction_scale: float, d: int) -> float:
        if isinstance(self.rho, str):
            rho_name = self.rho.strip().lower().replace("-", "_")
            if rho_name != "auto":
                raise ValueError(
                    "desaxce rho must be 'auto' or a positive scalar "
                    f"(got {self.rho!r})"
                )
            scale = normal_scale if d == 1 else friction_scale
            val = self.rho_scale / max(float(scale), self.rho_min)
        else:
            val = float(self.rho)
        if val <= 0.0 or not np.isfinite(val):
            raise ValueError(f"desaxce rho must be finite and positive (got {val})")
        return float(np.clip(val, self.rho_min, self.rho_max))

    def residual_and_jac(
        self,
        normal_quantity: float,
        contact_velocity: np.ndarray,
        percussion: np.ndarray,
        mu: float,
        normal_scale: float,
        friction_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        u_contact = np.asarray(contact_velocity, dtype=float).ravel()
        r_blk = np.asarray(percussion, dtype=float).ravel()
        d = r_blk.size
        if d == 0:
            raise ValueError("contact block must contain at least one normal row")
        if u_contact.size != d:
            raise ValueError(
                "contact_velocity and percussion must have the same block size "
                f"(got {u_contact.size} and {d})"
            )

        mu = float(mu)
        if mu < 0.0:
            raise ValueError(f"mu must be nonnegative for De Saxce contact (got {mu})")
        alpha = mu if self.alpha is None else float(self.alpha)
        if alpha < 0.0:
            raise ValueError(
                f"alpha must be nonnegative for De Saxce contact (got {alpha})"
            )

        u_hat = np.zeros(d, dtype=float)
        u_hat[0] = float(normal_quantity)
        if d > 1:
            u_hat[1:] = u_contact[1:]

        D_uhat = np.eye(d, dtype=float)
        if d > 1 and alpha > self.tie_tol:
            u_t = u_hat[1:]
            speed = float(np.linalg.norm(u_t))
            u_hat[0] += alpha * speed
            if speed > self.tie_tol:
                D_uhat[0, 1:] = alpha * (u_t / speed)
            else:
                r_t = r_blk[1:]
                r_t_norm = float(np.linalg.norm(r_t))
                if r_t_norm > self.tie_tol:
                    D_uhat[0, 1:] = -alpha * (r_t / r_t_norm)

        normal_scale = float(normal_scale)
        friction_scale = float(friction_scale)
        if normal_scale <= 0.0:
            raise ValueError(
                f"normal_scale must be positive for De Saxce contact (got {normal_scale})"
            )
        if d > 1 and friction_scale <= 0.0:
            raise ValueError(
                "friction_scale must be positive for De Saxce contact "
                f"(got {friction_scale})"
            )

        if d == 1 or mu <= self.tie_tol:
            rho = self._rho_value(normal_scale, friction_scale, d)
            z_n = r_blk[0] - rho * u_hat[0]
            p_n = max(z_n, 0.0)
            J_n = 1.0 if z_n >= 0.0 else 0.0
            f_blk = r_blk.copy()
            f_blk[0] = r_blk[0] - p_n
            df_dnormal = np.zeros(d, dtype=float)
            df_dnormal[0] = J_n * rho
            df_du = np.zeros((d, d), dtype=float)
            df_dr = np.eye(d, dtype=float)
            df_dr[0, 0] = 1.0 - J_n
            return f_blk, df_dnormal, df_du, df_dr

        rho = self._rho_value(normal_scale, friction_scale, d)
        T_x = np.eye(d, dtype=float)
        T_x[1:, 1:] *= mu
        T_y = np.eye(d, dtype=float)
        T_y[0, 0] = mu
        T_y_inv = np.eye(d, dtype=float)
        T_y_inv[0, 0] = 1.0 / mu

        x_sd = T_x @ u_hat
        y_sd = T_y @ r_blk
        z = y_sd - rho * x_sd
        proj_z, J_proj = MuScaledSOCProjection._proj_mu_scaled_soc(
            z, 1.0, return_jacobian=True
        )

        f_blk = T_y_inv @ (y_sd - proj_z)
        dphi_duhat = T_y_inv @ (J_proj @ (rho * T_x @ D_uhat))
        df_dnormal = dphi_duhat[:, 0].copy()
        df_du = np.zeros((d, d), dtype=float)
        if d > 1:
            df_du[:, 1:] = dphi_duhat[:, 1:]
        df_dr = T_y_inv @ ((np.eye(d, dtype=float) - J_proj) @ T_y)
        return f_blk, df_dnormal, df_du, df_dr


@dataclass
class ProjectedRadauContactModel:
    """Stateful model consumed by :class:`solve_nivp.integrations.RadauIIA`."""

    A_phys: Any
    rhs_smooth: Callable
    y0_phys: np.ndarray
    contacts: list[dict]
    gap_func: Optional[Callable]
    B: Any
    C_extract: Any
    D_extract: Any
    reaction_extract_rows: Any
    normal_rows: Any
    U_contact: Any
    rhs_jac: Optional[Callable]
    algebraic_projection: Optional[AlgebraicConstraintProjection]
    constraint_q_slices: list
    contact_law: ProjectedRadauContactLaw
    component_slices: list
    projection_indices: Optional[np.ndarray]
    restitution_normal: float
    restitution_tangential: float
    reported_reaction_units: str
    get_s0: Optional[Callable]
    get_w0: Optional[Callable]
    constant_contact_offsets: bool
    normal_r: Any
    friction_r: Any
    gap_tol: float
    endpoint_inactive_handling: str
    reaction_state_indices: Optional[np.ndarray] = None
    reaction_state_to_reported_scale: Any = 1.0
    mask_reaction_state_in_smooth_rhs: bool = False
    auto_rho_strategy: str = "h_scaled"
    effective_jacobian_fn: Optional[Callable] = None
    delassus_cache_log_tol: float = 0.5
    endpoint_law: Optional[ProjectedRadauContactLaw] = None

    def __post_init__(self) -> None:
        self.y0_phys = np.asarray(self.y0_phys, dtype=float).ravel()
        self.n_phys = self.y0_phys.size
        self.A_phys = _dense_or_sparse(self.A_phys)
        self.B = _dense_or_sparse(self.B)
        if self.B.shape[0] != self.n_phys:
            raise ValueError(
                f"B has {self.B.shape[0]} rows but n_phys = {self.n_phys}"
            )
        self.n_react = int(self.B.shape[1])
        self.constraint_q_slices = list(self.constraint_q_slices or [])
        self._B_coupling = self._zero_constraint_rows(self.B)
        self.reaction_extract_rows = np.asarray(
            self.reaction_extract_rows, dtype=int
        ).ravel()
        if self.reaction_extract_rows.size != self.n_react:
            raise ValueError(
                "reaction_extract_rows must have one entry per reaction "
                f"DOF (got {self.reaction_extract_rows.size}, expected {self.n_react})"
            )
        self.normal_rows = np.asarray(self.normal_rows, dtype=int).ravel()
        if self.normal_rows.size != len(self.contacts):
            raise ValueError(
                "normal_rows must have one entry per contact "
                f"(got {self.normal_rows.size}, expected {len(self.contacts)})"
            )
        if self.U_contact is not None:
            self.U_contact = _dense_or_sparse(self.U_contact)
            if self.U_contact.shape != (self.n_react, self.n_phys):
                raise ValueError(
                    "U_contact must map the physical state to the reaction "
                    f"block shape {(self.n_react, self.n_phys)}; got {self.U_contact.shape}"
                )
        if self.projection_indices is not None:
            self.projection_indices = np.asarray(self.projection_indices, dtype=int).ravel()
        self.reported_reaction_units = str(self.reported_reaction_units).strip().lower()
        if self.reported_reaction_units not in {"force", "impulse"}:
            raise ValueError("reported_reaction_units must be 'force' or 'impulse'")
        if self.reaction_state_indices is not None:
            idx = np.asarray(self.reaction_state_indices, dtype=int).ravel()
            if idx.size != self.n_react:
                raise ValueError(
                    "reaction_state_indices must contain one state entry per "
                    f"reaction DOF (got {idx.size}, expected {self.n_react})"
                )
            if np.any(idx < 0) or np.any(idx >= self.n_phys):
                raise ValueError("reaction_state_indices contains out-of-range entries")
            scale = np.asarray(self.reaction_state_to_reported_scale, dtype=float)
            if scale.ndim == 0:
                scale = np.full(self.n_react, float(scale), dtype=float)
            else:
                scale = scale.ravel()
            if scale.size != self.n_react:
                raise ValueError(
                    "reaction_state_to_reported_scale must be scalar or have one "
                    f"entry per reaction DOF (got {scale.size}, expected {self.n_react})"
                )
            if np.any(scale == 0.0):
                raise ValueError("reaction_state_to_reported_scale entries must be nonzero")
            self.reaction_state_indices = idx
            self.reaction_state_to_reported_scale = scale
            self._reaction_storage_to_reported_scale = scale
            self.mask_reaction_state_in_smooth_rhs = bool(
                self.mask_reaction_state_in_smooth_rhs
            )
        else:
            self.reaction_state_to_reported_scale = None
            self._reaction_storage_to_reported_scale = np.ones(
                self.n_react, dtype=float
            )
            self.mask_reaction_state_in_smooth_rhs = False
        self.endpoint_inactive_handling = (
            str(self.endpoint_inactive_handling).strip().lower().replace("-", "_")
        )
        if self.endpoint_inactive_handling not in {"gap", "natural_map"}:
            raise ValueError(
                "endpoint_inactive_handling must be 'gap' or 'natural_map' "
                f"(got {self.endpoint_inactive_handling!r})"
            )

        self._s0_nargs = _count_required_args(self.get_s0)
        self._w0_nargs = _count_required_args(self.get_w0)
        self._normal_r_nargs = _count_required_args(self.normal_r) if callable(self.normal_r) else None
        self._friction_r_nargs = _count_required_args(self.friction_r) if callable(self.friction_r) else None
        self._use_auto_normal_r = isinstance(self.normal_r, str) and self.normal_r.strip().lower() == "auto"
        self._use_auto_friction_r = isinstance(self.friction_r, str) and self.friction_r.strip().lower() == "auto"
        if self._use_auto_normal_r:
            self.normal_r = 1.0
        if self._use_auto_friction_r:
            self.friction_r = 1.0
        self._auto_normal_r_base, self._auto_friction_r_base = self._auto_scale_bases()

        strategy = str(self.auto_rho_strategy).strip().lower().replace("-", "_")
        if strategy not in {"h_scaled", "delassus"}:
            raise ValueError(
                "auto_rho_strategy must be 'h_scaled' or 'delassus' "
                f"(got {self.auto_rho_strategy!r})"
            )
        self.auto_rho_strategy = strategy
        self._delassus_cache: dict = {"h": None, "rho_N": None, "rho_T": None}

        self.last_stage_dpi = None
        self.last_stage_dmu = None
        self.last_delta_pi = None
        self.last_total_pi = None
        self.last_total_effective_pi = None
        self.last_reported_reaction = None
        self.last_reported_storage_reaction = None
        self._projection_map_cache = None
        self._last_stage_delta_y = None
        self._last_stage_dpi_guess = None
        self._last_stage_guess_h = None

        if self.endpoint_law is None:
            self.endpoint_law = self.contact_law

    def _reported_from_impulse(self, impulse, h):
        out = np.asarray(impulse, dtype=float).copy()
        if self.reported_reaction_units == "force":
            out = out / float(h)
        return out

    def _reported_derivative_from_impulse(self, h):
        if self.reported_reaction_units == "force":
            return 1.0 / float(h)
        return 1.0

    def _reaction_state_to_reported(self, y):
        if self.reaction_state_indices is None:
            return np.zeros(self.n_react, dtype=float)
        vals = np.asarray(y, dtype=float).ravel()[self.reaction_state_indices]
        return vals * self.reaction_state_to_reported_scale

    def _storage_to_reported(self, reaction, sl=None):
        scale = self._reaction_storage_to_reported_scale
        if sl is not None:
            scale = scale[sl]
        reaction = np.asarray(reaction, dtype=float).ravel()
        if reaction.size != np.asarray(scale).size:
            raise ValueError(
                "reaction storage block has incompatible size "
                f"{reaction.size}; expected {np.asarray(scale).size}"
            )
        return (
            reaction
            * scale
        )

    def _storage_jacobian_scale(self, sl):
        return self._reaction_storage_to_reported_scale[sl]

    def _write_storage_to_reaction_state(self, y, reaction_storage):
        if self.reaction_state_indices is None:
            return
        y[self.reaction_state_indices] = np.asarray(
            reaction_storage, dtype=float
        ).ravel()

    def reaction_state_history(self, y_hist):
        if self.reaction_state_indices is None:
            raise RuntimeError("reaction_state_indices are not configured")
        y_hist = np.asarray(y_hist, dtype=float)
        if y_hist.ndim == 1:
            return self._reaction_state_to_reported(y_hist)
        return (
            y_hist[:, self.reaction_state_indices]
            * self.reaction_state_to_reported_scale[None, :]
        )

    def _callback_augmented_state(self, y):
        y_aug = np.zeros(self.n_phys + self.n_react, dtype=float)
        y_aug[: self.n_phys] = np.asarray(y, dtype=float).ravel()
        if self.reaction_state_indices is not None:
            y_aug[self.n_phys:] = self._reaction_state_to_reported(y)
        return y_aug

    def _smooth_state(self, y):
        y = np.asarray(y, dtype=float).ravel()
        if (
            self.reaction_state_indices is None
            or not self.mask_reaction_state_in_smooth_rhs
        ):
            return y
        out = y.copy()
        out[self.reaction_state_indices] = 0.0
        return out

    def _compress_stage_residual(self, F_stacked):
        out_size = (
            self.n_phys
            if self.reaction_state_indices is not None
            else self.n_phys + self.n_react
        )
        out = np.zeros(out_size, dtype=float)
        if F_stacked is None:
            return out
        F_stacked = np.asarray(F_stacked, dtype=float).ravel()
        n = self.n_phys
        r = self.n_react
        if F_stacked.size >= 2 * n + 2 * r:
            out[:n] = F_stacked[n:2 * n]
            if self.reaction_state_indices is None:
                out[n:] = F_stacked[2 * n + r:2 * n + 2 * r]
            else:
                out[self.reaction_state_indices] = F_stacked[
                    2 * n + r:2 * n + 2 * r
                ]
        else:
            m = min(out.size, F_stacked.size)
            out[:m] = F_stacked[:m]
        return out

    # ------------------------------------------------------------------
    # Basic evaluations
    # ------------------------------------------------------------------
    def _matvec(self, M, x):
        return np.asarray(M @ x, dtype=float).ravel()

    def _zero_constraint_rows(self, M):
        if not self.constraint_q_slices:
            return M
        if sp.issparse(M):
            out = M.tolil(copy=True)
            for qs in self.constraint_q_slices:
                out[qs, :] = 0.0
            return out.tocsr()
        out = np.array(M, dtype=float, copy=True)
        for qs in self.constraint_q_slices:
            out[qs, :] = 0.0
        return out

    def _smooth_rhs(self, t, y, *, h=None, prev_state=None):
        y_eval = self._smooth_state(y)
        out = np.asarray(
            _call_with_time_state_fk(self.rhs_smooth, t, y_eval, None), dtype=float
        ).ravel().copy()
        if self.algebraic_projection is None:
            return out
        c_res = self.algebraic_projection.constraint_residual(
            y, t=t, Fk_val=None, step_size=h, prev_state=prev_state
        )
        for qs in self.constraint_q_slices:
            out[qs] = -c_res[qs]
        return out

    def _smooth_jac(self, t, y, *, h=None, prev_state=None):
        if self.rhs_jac is not None:
            J = _dense_or_sparse(
                _call_with_time_state_fk(self.rhs_jac, t, self._smooth_state(y), None)
            )
            if (
                self.reaction_state_indices is not None
                and self.mask_reaction_state_in_smooth_rhs
            ):
                if sp.issparse(J):
                    J = _zero_sparse_columns(J, self.reaction_state_indices)
                else:
                    J = np.asarray(J, dtype=float).copy()
                    J[:, self.reaction_state_indices] = 0.0
            if self.algebraic_projection is None:
                return J
            patch = self.algebraic_projection.build_constraint_patch(
                y, self.n_phys, t=t, Fk_val=None, step_size=h, prev_state=prev_state
            ).tocsr()
            if sp.issparse(J):
                return _replace_sparse_rows(J, self.constraint_q_slices, -patch)
            J_arr = np.asarray(J, dtype=float).copy()
            patch_arr = patch.toarray()
            for qs in self.constraint_q_slices:
                J_arr[qs, :] = -patch_arr[qs, :]
            return sp.csr_matrix(J_arr)
        f0 = self._smooth_rhs(t, y, h=h, prev_state=prev_state)
        n = self.n_phys
        eps = np.sqrt(np.finfo(float).eps)
        h_vec = eps * np.maximum(1.0, np.abs(y))
        J = np.empty((n, n), dtype=float)
        for j in range(n):
            yp = y.copy()
            yp[j] += h_vec[j]
            J[:, j] = (
                self._smooth_rhs(t, yp, h=h, prev_state=prev_state) - f0
            ) / h_vec[j]
        return J

    def gap(self, y, t):
        if self.gap_func is not None:
            return np.atleast_1d(np.asarray(self.gap_func(y, t), dtype=float)).ravel()
        vals = self.C_extract @ y
        vals = np.asarray(vals, dtype=float).ravel()
        return vals[self.normal_rows]

    def gap_jacobian(self, y, t):
        if self.gap_func is None and self.C_extract is not None:
            if sp.issparse(self.C_extract):
                return self.C_extract[self.normal_rows, :].tocsr()
            return np.asarray(self.C_extract[self.normal_rows, :], dtype=float)
        g0 = self.gap(y, t)
        n = self.n_phys
        eps = np.sqrt(np.finfo(float).eps)
        h_vec = eps * np.maximum(1.0, np.abs(y))
        J = np.empty((g0.size, n), dtype=float)
        for j in range(n):
            yp = y.copy()
            yp[j] += h_vec[j]
            J[:, j] = (self.gap(yp, t) - g0) / h_vec[j]
        return J

    def contact_velocity(self, y):
        if self.U_contact is not None:
            return self._matvec(self.U_contact, y)
        return y[self.reaction_extract_rows]

    def _static_gap_jacobian_dense(self, y, t):
        cached = getattr(self, "_gap_jacobian_dense_static", None)
        if cached is not None:
            return cached
        if self.gap_func is None and self.C_extract is not None:
            if sp.issparse(self.C_extract):
                gap_jac = self.C_extract[self.normal_rows, :].tocsr()
                cached = gap_jac.toarray()
            else:
                cached = np.asarray(self.C_extract[self.normal_rows, :], dtype=float)
            self._gap_jacobian_dense_static = cached
            return cached
        gap_jac = self.gap_jacobian(y, t)
        return (
            gap_jac.toarray()
            if sp.issparse(gap_jac)
            else np.asarray(gap_jac, dtype=float)
        )

    def _static_contact_operator_dense(self):
        cached = getattr(self, "_U_contact_dense_static", None)
        if cached is not None:
            return cached
        if self.U_contact is not None:
            cached = (
                self.U_contact.toarray()
                if sp.issparse(self.U_contact)
                else np.asarray(self.U_contact, dtype=float)
            )
        else:
            cached = np.zeros((self.n_react, self.n_phys), dtype=float)
            cached[np.arange(self.n_react), self.reaction_extract_rows] = 1.0
        self._U_contact_dense_static = cached
        return cached

    def _evaluate_offset_force(self, y, t):
        out = np.zeros(self.n_react, dtype=float)
        if self.get_s0 is None and self.get_w0 is None:
            return out
        y_aug = self._callback_augmented_state(y)
        s0 = _eval_s0(
            self.get_s0,
            self._s0_nargs,
            len(self.contacts),
            y_aug,
            t=t,
            Fk_val=None,
        )
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            out[sl.start] = float(s0[k])
            m = sl.stop - sl.start - 1
            if m > 0:
                out[sl.start + 1 : sl.stop] = _eval_w0(
                    self.get_w0, self._w0_nargs, y_aug, k, m, t=t, Fk_val=None
                )
        return out

    def _offset_force(self, y, t):
        if not bool(self.constant_contact_offsets):
            return self._evaluate_offset_force(y, t)
        cached = getattr(self, "_constant_offset_force_cache", None)
        if cached is None:
            cached = self._evaluate_offset_force(y, t)
            self._constant_offset_force_cache = cached
        return cached.copy()

    def _scale_arrays(self, y, t, h):
        n_blocks = len(self.contacts)
        use_delassus = (
            self.auto_rho_strategy == "delassus"
            and (self._use_auto_normal_r or self._use_auto_friction_r)
        )
        if use_delassus:
            try:
                rho_N, rho_T = self._compute_delassus_rho(y, t, h)
            except Exception:
                rho_N, rho_T = None, None
            if rho_N is not None:
                if self._use_auto_normal_r:
                    normal = rho_N
                else:
                    normal = _eval_contact_scalar_field(
                        self.normal_r, self._normal_r_nargs, n_blocks, "normal_r",
                        self._callback_augmented_state(y), t=t, Fk_val=None,
                    )
                if self._use_auto_friction_r:
                    friction = rho_T
                else:
                    friction = _eval_contact_scalar_field(
                        self.friction_r, self._friction_r_nargs, n_blocks, "friction_r",
                        self._callback_augmented_state(y), t=t, Fk_val=None,
                    )
                return normal, friction

        if self._use_auto_normal_r:
            normal = self._auto_normal_r_base * float(abs(h))
        else:
            normal = _eval_contact_scalar_field(
                self.normal_r, self._normal_r_nargs, n_blocks, "normal_r",
                self._callback_augmented_state(y), t=t, Fk_val=None,
            )
        if self._use_auto_friction_r:
            friction = self._auto_friction_r_base.copy()
        else:
            friction = _eval_contact_scalar_field(
                self.friction_r, self._friction_r_nargs, n_blocks, "friction_r",
                self._callback_augmented_state(y), t=t, Fk_val=None,
            )
        return normal, friction

    def _build_effective_jacobian(self, y, t, h):
        """Effective Newton-step Jacobian M_eff ≈ (1/h)·A − ∂rhs/∂y.

        Used to estimate the per-contact Delassus W = D·M_eff⁻¹·B used by
        rule (111) of Acary–Brogliato:  ρ_N = 1/W_NN,  ρ_T = 1/λ_max(W_TT).
        Integrator-agnostic: a custom builder may be supplied via
        ``effective_jacobian_fn`` to fold a stage coefficient γ into the form
        (1/(γh))·A − J_rhs, but for the cone radius the leading h-scaling and
        the symmetry of the normal/tangent treatment are what matters.
        """
        h = float(h)
        if h == 0.0:
            raise ValueError("Delassus ρ requires h != 0")
        if self.effective_jacobian_fn is not None:
            return self.effective_jacobian_fn(np.asarray(y, dtype=float), float(t), h)
        if self.rhs_jac is None:
            raise RuntimeError(
                "Delassus ρ needs rhs_jac or effective_jacobian_fn to build M_eff."
            )
        J = self._smooth_jac(t, np.asarray(y, dtype=float), h=h, prev_state=y)
        A = self.A_phys
        if sp.issparse(A) or sp.issparse(J):
            A_csc = A.tocsc() if sp.issparse(A) else sp.csc_matrix(A)
            J_csc = J.tocsc() if sp.issparse(J) else sp.csc_matrix(J)
            return (1.0 / h) * A_csc - J_csc
        return (1.0 / h) * np.asarray(A, dtype=float) - np.asarray(J, dtype=float)

    def _compute_delassus_rho(self, y, t, h):
        """Solve M_eff·X = B per contact column and return rule (111) ρ.

        Cached on h within ±exp(delassus_cache_log_tol).
        """
        h = float(h)
        cache = self._delassus_cache
        h_old = cache.get("h")
        if h_old is not None and h_old != 0.0:
            ratio = h / h_old
            if ratio > 0 and abs(np.log(ratio)) < self.delassus_cache_log_tol:
                return cache["rho_N"], cache["rho_T"]

        M_eff = self._build_effective_jacobian(y, t, h)

        if sp.issparse(M_eff):
            # Stripped multiplier rows leave M_eff structurally rank-deficient.
            # Stamp identity into zero rows so the multiplier subspace becomes
            # a trivial zero-solve and splu produces a finite factorization
            # without affecting the bulk Delassus result.
            M_eff_csr = M_eff.tocsr()
            row_max = np.abs(M_eff_csr).max(axis=1).toarray().ravel()
            zero_rows = np.where(row_max == 0.0)[0]
            if zero_rows.size:
                M_eff_lil = M_eff_csr.tolil()
                for r in zero_rows:
                    M_eff_lil[r, r] = 1.0
                M_eff_csr = M_eff_lil.tocsr()
            M_eff_csc = M_eff_csr.tocsc()
            n = M_eff_csc.shape[0]
            try:
                lu = sp.linalg.splu(M_eff_csc)
            except Exception:
                return None, None
            probe = lu.solve(np.zeros(n))
            if not np.all(np.isfinite(probe)):
                return None, None

            def solve_one(rhs):
                rhs = np.asarray(rhs, dtype=float).ravel()
                out = lu.solve(rhs)
                if not np.all(np.isfinite(out)):
                    raise RuntimeError("singular M_eff in Delassus solve")
                return out
        else:
            from scipy.linalg import lu_factor, lu_solve

            try:
                lu_p = lu_factor(np.asarray(M_eff, dtype=float))
            except Exception:
                return None, None

            def solve_one(rhs):
                rhs = np.asarray(rhs, dtype=float).ravel()
                out = lu_solve(lu_p, rhs)
                if not np.all(np.isfinite(out)):
                    raise RuntimeError("singular M_eff in Delassus solve")
                return out

        B = self._B_coupling
        if sp.issparse(B):
            B_dense = B.toarray()
        else:
            B_dense = np.asarray(B, dtype=float)

        D = self.U_contact
        if D is None:
            n_react = self.n_react
            n_phys = self.n_phys
            D_dense = np.zeros((n_react, n_phys), dtype=float)
            for col, row in enumerate(self.reaction_extract_rows):
                D_dense[col, row] = 1.0
        elif sp.issparse(D):
            D_dense = D.toarray()
        else:
            D_dense = np.asarray(D, dtype=float)

        n_c = len(self.contacts)
        rho_N = np.empty(n_c, dtype=float)
        rho_T = np.empty(n_c, dtype=float)
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            n_n = 1
            n_t = sl.stop - sl.start - n_n

            x_n = solve_one(B_dense[:, sl.start])
            W_NN = float(D_dense[sl.start, :] @ x_n)
            rho_N[k] = 1.0 / max(abs(W_NN), 1e-30)

            if n_t <= 0:
                rho_T[k] = rho_N[k]
                continue

            cols = np.arange(sl.start + 1, sl.stop)
            W_TT = np.empty((n_t, n_t), dtype=float)
            for j, c in enumerate(cols):
                x_t = solve_one(B_dense[:, c])
                W_TT[:, j] = D_dense[cols, :] @ x_t
            try:
                eigs = np.linalg.eigvals(W_TT)
                lam_max = float(np.max(np.abs(eigs)))
            except np.linalg.LinAlgError:
                lam_max = float(np.max(np.abs(np.diag(W_TT))))
            rho_T[k] = 1.0 / max(lam_max, 1e-30)

        cache["h"] = h
        cache["rho_N"] = rho_N
        cache["rho_T"] = rho_T
        return rho_N, rho_T

    def _auto_scale_bases(self):
        if sp.issparse(self.A_phys):
            diag = np.abs(np.asarray(self.A_phys.diagonal()).ravel())
        else:
            diag = np.abs(np.diag(np.asarray(self.A_phys, dtype=float)))
        pos = diag > 0
        diag = np.where(pos, diag, (diag[pos].min() if pos.any() else 1.0))

        if self.U_contact is not None:
            D_t = self.U_contact.T
            D_dense = D_t.toarray() if sp.issparse(D_t) else np.asarray(D_t, dtype=float)
        else:
            D_dense = np.zeros((self.n_phys, self.n_react), dtype=float)
            for col, row in enumerate(self.reaction_extract_rows):
                D_dense[row, col] = 1.0
        B_dense = (
            self._B_coupling.toarray()
            if sp.issparse(self._B_coupling)
            else np.asarray(self._B_coupling, dtype=float)
        )

        def coupling_scale(col):
            d = D_dense[:, col]
            b = B_dense[:, col]
            val = float(abs(np.sum(d * b / diag)))
            if not np.isfinite(val) or val <= 0.0:
                val = float(np.sum(d * d / diag))
            return val if np.isfinite(val) and val > 0.0 else 1.0

        normal = np.zeros(len(self.contacts), dtype=float)
        friction = np.zeros(len(self.contacts), dtype=float)
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            normal[k] = coupling_scale(sl.start)
            if sl.stop - sl.start > 1:
                vals = []
                for col in range(sl.start + 1, sl.stop):
                    vals.append(coupling_scale(col))
                friction[k] = float(np.mean(vals))
            else:
                friction[k] = normal[k]
        return np.where(normal > 0, normal, 1.0), np.where(friction > 0, friction, 1.0)

    def _mu_state_jacobian(self, y, t, mu_vals=None):
        out = np.zeros((len(self.contacts), self.n_phys), dtype=float)
        if not self.contacts:
            return out
        y = np.asarray(y, dtype=float).ravel()
        if mu_vals is None:
            mu_vals = _vectorize_mu(self.contacts, y, t=t, Fk_val=None)
        eps_base = np.sqrt(np.finfo(float).eps)
        for k, ci in enumerate(self.contacts):
            dmu_dy = ci.get("dmu_dy")
            if dmu_dy is not None:
                raw = _call_state_time_fk(
                    dmu_dy,
                    ci.get("dmu_dy_nargs"),
                    y,
                    t,
                    None,
                )
                arr = np.asarray(raw, dtype=float).ravel()
                if arr.size != self.n_phys:
                    raise ValueError(
                        "dmu_dy must return one derivative per physical state "
                        f"DOF (got {arr.size}, expected {self.n_phys})"
                    )
                out[k, :] = arr
                continue
            if ci.get("mu_is_const", False):
                continue
            get_mu = ci["get_mu"]
            nargs = ci["mu_nargs"]
            mu0 = float(mu_vals[k])
            for j in range(self.n_phys):
                eps = eps_base * max(1.0, abs(float(y[j])))
                yp = y.copy()
                yp[j] += eps
                mup = float(_call_state_time_fk(get_mu, nargs, yp, t, None))
                out[k, j] = (mup - mu0) / eps
        return out

    def _law_residual_and_jac(
        self,
        normal_quantity,
        contact_velocity,
        percussion,
        mu,
        normal_scale,
        friction_scale,
        *,
        need_mu_derivative=False,
        law=None,
    ):
        if law is None:
            law = self.contact_law
        f_blk, df_dnormal, df_du, df_dr = law.residual_and_jac(
            normal_quantity,
            contact_velocity,
            percussion,
            mu,
            normal_scale,
            friction_scale,
        )
        df_dmu = None
        if need_mu_derivative:
            eps = np.sqrt(np.finfo(float).eps) * max(1.0, abs(float(mu)))
            if float(mu) > eps:
                fp = law.residual_and_jac(
                    normal_quantity,
                    contact_velocity,
                    percussion,
                    float(mu) + eps,
                    normal_scale,
                    friction_scale,
                )[0]
                fm = law.residual_and_jac(
                    normal_quantity,
                    contact_velocity,
                    percussion,
                    float(mu) - eps,
                    normal_scale,
                    friction_scale,
                )[0]
                df_dmu = (fp - fm) / (2.0 * eps)
            else:
                fp = law.residual_and_jac(
                    normal_quantity,
                    contact_velocity,
                    percussion,
                    float(mu) + eps,
                    normal_scale,
                    friction_scale,
                )[0]
                df_dmu = (fp - f_blk) / eps
        return f_blk, df_dnormal, df_du, df_dr, df_dmu

    # ------------------------------------------------------------------
    # Stage residual and Jacobian
    # ------------------------------------------------------------------
    def unpack(self, Z):
        n = self.n_phys
        r = self.n_react
        Y1 = Z[:n]
        Y2 = Z[n:2 * n]
        dPi1 = Z[2 * n:2 * n + r]
        dPi2 = Z[2 * n + r:2 * n + 2 * r]
        return [Y1, Y2], [dPi1, dPi2]

    def pack(self, Y, dPi):
        return np.concatenate([Y[0], Y[1], dPi[0], dPi[1]])

    def _reaction_storage_from_state(self, y_aug, y_prev):
        if self.n_react == 0:
            return np.zeros(0, dtype=float)
        if self.reaction_state_indices is not None:
            return np.asarray(y_prev, dtype=float).ravel()[
                self.reaction_state_indices
            ].copy()
        if y_aug.size >= self.n_phys + self.n_react:
            return np.asarray(
                y_aug[self.n_phys:self.n_phys + self.n_react],
                dtype=float,
            ).copy()
        return np.zeros(self.n_react, dtype=float)

    def _stage_initial_guess(self, y_aug, y_prev, h):
        prev_storage = self._reaction_storage_from_state(y_aug, y_prev)
        if self.reported_reaction_units == "force":
            warm_pi = float(h) * prev_storage
        else:
            warm_pi = prev_storage.copy()

        Y0 = [y_prev.copy(), y_prev.copy()]
        dPi0 = [warm_pi.copy(), warm_pi.copy()]

        last_delta = self._last_stage_delta_y
        last_dpi = self._last_stage_dpi_guess
        last_h = self._last_stage_guess_h
        if (
            self.reaction_state_indices is not None
            and last_delta is not None
            and last_dpi is not None
            and last_h is not None
            and float(last_h) > 0.0
        ):
            scale = float(h) / float(last_h)
            if np.isfinite(scale) and 0.05 <= scale <= 20.0:
                try:
                    Y_trial = [
                        y_prev + scale * np.asarray(last_delta[i], dtype=float)
                        for i in range(2)
                    ]
                    dPi_trial = [
                        scale * np.asarray(last_dpi[i], dtype=float)
                        for i in range(2)
                    ]
                    if (
                        Y_trial[0].shape == y_prev.shape
                        and Y_trial[1].shape == y_prev.shape
                        and dPi_trial[0].shape == (self.n_react,)
                        and dPi_trial[1].shape == (self.n_react,)
                    ):
                        Y0 = [Y_trial[0].copy(), Y_trial[1].copy()]
                        dPi0 = [dPi_trial[0].copy(), dPi_trial[1].copy()]
                except (TypeError, ValueError, FloatingPointError):
                    pass

        if self.reaction_state_indices is not None:
            idx = self.reaction_state_indices
            for i in range(2):
                Y0[i] = np.asarray(Y0[i], dtype=float).copy()
                Y0[i][idx] = self._reported_from_impulse(dPi0[i], h)
        return self.pack(Y0, dPi0)

    def _remember_stage_guess(self, Y, dPi, y_prev, h):
        self._last_stage_delta_y = [
            np.asarray(Y[i], dtype=float).copy() - y_prev for i in range(2)
        ]
        self._last_stage_dpi_guess = [
            np.asarray(dPi[i], dtype=float).copy() for i in range(2)
        ]
        self._last_stage_guess_h = float(h)

    def stage_quantities(self, Z, t, h, rk_A, rk_c):
        Y, dPi = self.unpack(Z)
        dmu = []
        offsets = []
        for i in range(2):
            dmu_i = rk_A[i, 0] * dPi[0] + rk_A[i, 1] * dPi[1]
            offsets.append(float(rk_c[i] * h) * self._offset_force(Y[i], t + rk_c[i] * h))
            dmu.append(dmu_i)
        return Y, dPi, dmu, offsets

    def residual(self, Z, t, y_prev_phys, h, rk_A, rk_b, rk_c):
        Y, dPi, dmu, offsets = self.stage_quantities(Z, t, h, rk_A, rk_c)
        F = np.zeros_like(Z, dtype=float)
        n = self.n_phys
        r = self.n_react
        f = [
            self._smooth_rhs(t + rk_c[i] * h, Y[i], h=h, prev_state=y_prev_phys)
            for i in range(2)
        ]

        for i in range(2):
            phys = self._matvec(self.A_phys, Y[i] - y_prev_phys)
            for j in range(2):
                phys -= rk_A[i, j] * (
                    h * f[j] + self._matvec(self._B_coupling, dPi[j])
                )
            F[i * n:(i + 1) * n] = phys
            if self.reaction_state_indices is not None:
                idx = self.reaction_state_indices
                F[i * n + idx] = Y[i][idx] - self._reported_from_impulse(dPi[i], h)

        for i in range(2):
            F[2 * n + i * r:2 * n + (i + 1) * r] = self._contact_residual(
                Y[i], t + rk_c[i] * h, dmu[i], offsets[i], h,
                endpoint=False,
            )
        return F

    def jacobian(self, Z, t, y_prev_phys, h, rk_A, rk_b, rk_c):
        Y, dPi, dmu, offsets = self.stage_quantities(Z, t, h, rk_A, rk_c)
        n = self.n_phys
        r = self.n_react
        A_sp = self.A_phys.tocsr() if sp.issparse(self.A_phys) else sp.csr_matrix(self.A_phys)
        B_sp = (
            self._B_coupling.tocsr()
            if sp.issparse(self._B_coupling)
            else sp.csr_matrix(self._B_coupling)
        )
        J_s = []
        for i in range(2):
            Ji = self._smooth_jac(
                t + rk_c[i] * h, Y[i], h=h, prev_state=y_prev_phys
            )
            J_s.append(Ji.tocsr() if sp.issparse(Ji) else sp.csr_matrix(Ji))

        rows = []
        for i in range(2):
            row = []
            for j in range(2):
                block = -float(rk_A[i, j] * h) * J_s[j]
                if i == j:
                    block = block + A_sp
                row.append(block.tocsr())
            for j in range(2):
                row.append((-float(rk_A[i, j]) * B_sp).tocsr())
            rows.append(sp.hstack(row, format="csr"))

        for i in range(2):
            Jy, Jr = self._contact_jacobian(
                Y[i], t + rk_c[i] * h, dmu[i], offsets[i], h,
                endpoint=False,
            )
            row = []
            for j in range(2):
                row.append(Jy if i == j else sp.csr_matrix((r, n)))
            for j in range(2):
                row.append((float(rk_A[i, j]) * Jr).tocsr())
            rows.append(sp.hstack(row, format="csr"))
        J_full = sp.vstack(rows, format="csr")
        if self.reaction_state_indices is not None:
            idx = self.reaction_state_indices
            d_reported = self._reported_derivative_from_impulse(h)
            repl_rows = []
            repl_cols = []
            repl_data = []
            for i in range(2):
                row_base = i * n
                y_col_base = i * n
                dpi_col_base = 2 * n + i * r
                for local_col, state_idx in enumerate(idx):
                    row_idx = row_base + int(state_idx)
                    repl_rows.extend([row_idx, row_idx])
                    repl_cols.extend([
                        y_col_base + int(state_idx),
                        dpi_col_base + local_col,
                    ])
                    repl_data.extend([1.0, -d_reported])
            replacement = sp.csr_matrix(
                (repl_data, (repl_rows, repl_cols)), shape=J_full.shape
            )
            J_full = _replace_sparse_rows(J_full, [np.asarray(repl_rows, dtype=int)], replacement)
        return J_full.tocsr()

    def _contact_residual(self, y, t, percussion, offset_measure, h, *, endpoint):
        gaps = self.gap(y, t)
        u_contact = self.contact_velocity(y)
        mu = _vectorize_mu(self.contacts, y, t=t, Fk_val=None)
        normal_r, friction_r = self._scale_arrays(y, t, h)
        out = np.zeros(self.n_react, dtype=float)
        effective = percussion + offset_measure
        velocity_normal = bool(
            getattr(self.contact_law, "expects_velocity_normal", False)
        )
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            effective_blk = self._storage_to_reported(percussion[sl], sl) + offset_measure[sl]
            if velocity_normal:
                gap_or_xi = float(u_contact[sl.start])
            else:
                gap_or_xi = float(gaps[k] - (0.0 if endpoint else self.gap_tol))
            f_blk = self.contact_law.residual(
                gap_or_xi,
                u_contact[sl],
                effective_blk,
                mu[k],
                normal_r[k],
                friction_r[k],
            )
            out[sl] = f_blk
        return out

    def _contact_jacobian(self, y, t, percussion, offset_measure, h, *, endpoint):
        gaps = self.gap(y, t)
        gap_dense = self._static_gap_jacobian_dense(y, t)
        U_dense = self._static_contact_operator_dense()
        if self.U_contact is not None:
            u_contact = U_dense @ y
        else:
            u_contact = y[self.reaction_extract_rows]
        mu = _vectorize_mu(self.contacts, y, t=t, Fk_val=None)
        mu_y = self._mu_state_jacobian(y, t, mu)
        normal_r, friction_r = self._scale_arrays(y, t, h)
        effective = percussion + offset_measure
        Jy = np.zeros((self.n_react, self.n_phys), dtype=float)
        Jr = np.zeros((self.n_react, self.n_react), dtype=float)
        velocity_normal = bool(
            getattr(self.contact_law, "expects_velocity_normal", False)
        )
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            effective_blk = self._storage_to_reported(percussion[sl], sl) + offset_measure[sl]
            if velocity_normal:
                gap_or_xi = float(u_contact[sl.start])
                normal_row = U_dense[sl.start, :]
            else:
                gap_or_xi = float(gaps[k] - (0.0 if endpoint else self.gap_tol))
                normal_row = gap_dense[k, :]
            need_mu = bool(np.any(np.abs(mu_y[k, :]) > 0.0))
            _, df_dgap, df_du, df_dr, df_dmu = self._law_residual_and_jac(
                gap_or_xi,
                u_contact[sl],
                effective_blk,
                mu[k],
                normal_r[k],
                friction_r[k],
                need_mu_derivative=need_mu,
            )
            D_blk = U_dense[sl, :]
            Jy[sl, :] = np.outer(df_dgap, normal_row) + df_du @ D_blk
            if need_mu:
                Jy[sl, :] += np.outer(df_dmu, mu_y[k, :])
            Jr[sl, sl] = df_dr @ np.diag(self._storage_jacobian_scale(sl))
        return sp.csr_matrix(Jy), sp.csr_matrix(Jr)

    # ------------------------------------------------------------------
    # Endpoint projection
    # ------------------------------------------------------------------
    def _projection_map(self):
        if self._projection_map_cache is not None:
            return self._projection_map_cache
        if self.n_react == 0:
            self._projection_map_cache = (
                np.array([], dtype=int),
                np.zeros((self.n_phys, self.n_react), dtype=float),
            )
            return self._projection_map_cache

        idx = None
        if self.projection_indices is None:
            if sp.issparse(self.A_phys):
                diag = np.abs(np.asarray(self.A_phys.diagonal()).ravel())
            else:
                diag = np.abs(np.diag(np.asarray(self.A_phys, dtype=float)))
            diag_scale = float(diag.max()) if diag.size else 0.0
            diag_tol = 100.0 * np.finfo(float).eps * max(1.0, diag_scale)
            auto_idx = np.flatnonzero(diag > diag_tol)
            # Descriptor systems carry algebraic rows/cols in the nullspace of
            # A_phys. Endpoint projection must solve only on the dynamic block.
            if 0 < auto_idx.size < self.n_phys:
                idx = auto_idx
        else:
            idx = np.asarray(self.projection_indices, dtype=int).ravel()

        if idx is None:
            B_rhs = (
                self._B_coupling.toarray()
                if sp.issparse(self._B_coupling)
                else np.asarray(self._B_coupling, dtype=float)
            )
            if sp.issparse(self.A_phys):
                P = spla.spsolve(self.A_phys.tocsc(), B_rhs)
            else:
                P = np.linalg.solve(np.asarray(self.A_phys, dtype=float), B_rhs)
            P = np.asarray(P, dtype=float)
            if P.ndim == 1:
                P = P.reshape(-1, 1)
            self._projection_map_cache = (np.arange(self.n_phys, dtype=int), P)
            return self._projection_map_cache

        if idx.size == 0:
            self._projection_map_cache = (
                idx,
                np.zeros((self.n_phys, self.n_react), dtype=float),
            )
            return self._projection_map_cache

        if sp.issparse(self.A_phys):
            A_red = self.A_phys[idx, :][:, idx].tocsc()
        else:
            A_red = np.asarray(self.A_phys, dtype=float)[np.ix_(idx, idx)]
        if sp.issparse(self._B_coupling):
            B_red = self._B_coupling[idx, :].toarray()
        else:
            B_red = np.asarray(self._B_coupling, dtype=float)[idx, :]
        if sp.issparse(A_red):
            P_red = spla.spsolve(A_red, B_red)
        else:
            P_red = np.linalg.solve(A_red, B_red)
        P = np.zeros((self.n_phys, self.n_react), dtype=float)
        P[idx, :] = np.asarray(P_red, dtype=float)
        self._projection_map_cache = (idx, P)
        return self._projection_map_cache

    def project_endpoint(self, y_stage, y_prev_phys, stage_pi, t_new, h):
        _, P = self._projection_map()
        offset = float(h) * self._offset_force(y_stage, t_new)
        total_stage_eff = self._storage_to_reported(stage_pi) + offset

        def residual(delta_pi):
            y_plus = y_stage + P @ delta_pi
            total_eff = total_stage_eff + self._storage_to_reported(delta_pi)
            return self._endpoint_contact_residual(
                y_plus, y_prev_phys, total_eff, t_new, h,
            )

        def jac(delta_pi):
            y_plus = y_stage + P @ delta_pi
            total_eff = total_stage_eff + self._storage_to_reported(delta_pi)
            return self._endpoint_contact_jacobian(
                y_plus, y_prev_phys, total_eff, t_new, h, P,
            )

        delta = np.zeros(self.n_react, dtype=float)
        ok = True
        err = np.inf
        # Scale the endpoint Newton tolerance by the natural problem magnitude
        # (||total_stage_eff||): a fixed 1e-10 absolute floor sits below the
        # double-precision round-off floor cond(J)*eps*||r|| once accumulated
        # impulses grow with h, so Newton stalls at machine precision and
        # exhausts its iteration budget.
        ref_scale = max(1.0, float(np.linalg.norm(total_stage_eff)))
        endpoint_tol = 1.0e-10 * ref_scale
        for _ in range(20):
            F = residual(delta)
            err = float(np.linalg.norm(F))
            if err < endpoint_tol:
                break
            J = jac(delta)
            try:
                step = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(J, -F, rcond=None)[0]
            alpha = 1.0
            base = err
            accepted = False
            while alpha >= 1.0e-8:
                trial = delta + alpha * step
                trial_err = float(np.linalg.norm(residual(trial)))
                if np.isfinite(trial_err) and trial_err <= (1.0 - 1.0e-4 * alpha) * base:
                    delta = trial
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                delta = delta + step
        else:
            ok = False

        y_plus = y_stage + P @ delta
        total_pert = stage_pi + delta
        total_eff = self._storage_to_reported(total_pert) + offset
        reported_storage = self._reported_from_impulse(total_pert, h)
        reported = self._storage_to_reported(reported_storage)
        self.last_reported_storage_reaction = reported_storage.copy()
        return y_plus, delta, total_pert, total_eff, reported, ok, err

    def _endpoint_contact_residual(self, y_plus, y_prev_phys, total_eff, t_new, h):
        # Endpoint dispatch is always velocity-level (Breuling Stage 2): xi[0] is
        # the restitution-shifted normal velocity, used both as the De Saxce /
        # SOC-FB kinematic input and as the velocity-level Signorini argument
        # for scalar NCP laws.  The position-level admissibility check happens
        # earlier via the endpoint_inactive_handling=='gap' gate.
        gaps = self.gap(y_plus, t_new)
        u_new = self.contact_velocity(y_plus)
        u_old = self.contact_velocity(y_prev_phys)
        mu = _vectorize_mu(self.contacts, y_plus, t=t_new, Fk_val=None)
        normal_r, friction_r = self._scale_arrays(y_plus, t_new, h)
        out = np.zeros(self.n_react, dtype=float)
        endpoint_law = self.endpoint_law
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            if (
                self.endpoint_inactive_handling == "gap"
                and gaps[k] > self.gap_tol
            ):
                out[sl] = total_eff[sl]
                continue
            xi = u_new[sl].copy()
            xi[0] += self.restitution_normal * u_old[sl.start]
            if sl.stop - sl.start > 1:
                xi[1:] += self.restitution_tangential * u_old[sl.start + 1:sl.stop]
            f_blk = endpoint_law.residual(
                float(xi[0]), xi, total_eff[sl], mu[k], normal_r[k], friction_r[k]
            )
            out[sl] = f_blk
        return out

    def _endpoint_contact_jacobian(self, y_plus, y_prev_phys, total_eff, t_new, h, P):
        gaps = self.gap(y_plus, t_new)
        U = self.U_contact
        if U is not None:
            U_dense = U.toarray() if sp.issparse(U) else np.asarray(U, dtype=float)
        else:
            U_dense = np.zeros((self.n_react, self.n_phys), dtype=float)
            U_dense[np.arange(self.n_react), self.reaction_extract_rows] = 1.0
        u_new = U_dense @ y_plus
        u_old = self.contact_velocity(y_prev_phys)
        DP = U_dense @ P
        mu = _vectorize_mu(self.contacts, y_plus, t=t_new, Fk_val=None)
        mu_y = self._mu_state_jacobian(y_plus, t_new, mu)
        normal_r, friction_r = self._scale_arrays(y_plus, t_new, h)
        J = np.zeros((self.n_react, self.n_react), dtype=float)
        for k, ci in enumerate(self.contacts):
            sl = ci["block_slice"]
            if (
                self.endpoint_inactive_handling == "gap"
                and gaps[k] > self.gap_tol
            ):
                J[sl, sl] = np.diag(self._storage_jacobian_scale(sl))
                continue
            xi = u_new[sl].copy()
            xi[0] += self.restitution_normal * u_old[sl.start]
            if sl.stop - sl.start > 1:
                xi[1:] += self.restitution_tangential * u_old[sl.start + 1:sl.stop]
            need_mu = bool(np.any(np.abs(mu_y[k, :]) > 0.0))
            _, df_dgap, df_du, df_dr, df_dmu = self._law_residual_and_jac(
                float(xi[0]),
                xi,
                total_eff[sl],
                mu[k],
                normal_r[k],
                friction_r[k],
                need_mu_derivative=need_mu,
                law=self.endpoint_law,
            )
            DPi_blk = DP[sl, :]
            storage_scale = np.diag(self._storage_jacobian_scale(sl))
            J[sl, :] = np.outer(df_dgap, DPi_blk[0, :]) + df_du @ DPi_blk
            if need_mu:
                J[sl, :] += np.outer(df_dmu, mu_y[k, :] @ P)
            J[sl, :] += (df_dr @ storage_scale) @ np.eye(self.n_react)[sl, :]
        return J

    # ------------------------------------------------------------------
    # Integrator entry point
    # ------------------------------------------------------------------
    def step(self, integrator, t, y_aug, h):
        if h <= 0.0:
            raise ValueError("Projected Radau contact requires positive h")
        y_aug = np.asarray(y_aug, dtype=float).ravel()
        y_prev = y_aug[: self.n_phys]
        Z0 = self._stage_initial_guess(y_aug, y_prev, h)

        rk_A = integrator._rk_A
        rk_b = integrator._rk_b
        rk_c = integrator._rk_c

        def F(Z):
            return self.residual(Z, t, y_prev, h, rk_A, rk_b, rk_c)

        def J(Z):
            return self.jacobian(Z, t, y_prev, h, rk_A, rk_b, rk_c)

        solver = integrator.solver
        saved_jac = solver.jacobian
        saved_cold = solver._cold_start_slices
        saved_atol_vec = solver._nl_atol_vec
        saved_rtol_vec = solver._nl_rtol_vec
        saved_lam = solver.lam
        try:
            solver.jacobian = J
            solver._cold_start_slices = None
            if solver._use_weighted_norm:
                base_atol, base_rtol = solver._ensure_nl_tol_vectors(y_aug.size)
                atol_phys = base_atol[: self.n_phys]
                rtol_phys = base_rtol[: self.n_phys]
                if self.reaction_state_indices is not None:
                    idx = self.reaction_state_indices
                    # dPi has units of (storage state) * h, while base_atol[idx]
                    # is supplied for the storage block; scale by h so the
                    # absolute tolerance lives in the same units as the dPi
                    # unknown. rtol multiplies |dPi| and self-scales already.
                    atol_react = base_atol[idx] * float(h)
                    rtol_react = base_rtol[idx]
                else:
                    atol_react = base_atol[self.n_phys:self.n_phys + self.n_react] * float(h)
                    rtol_react = base_rtol[self.n_phys:self.n_phys + self.n_react]
                solver._nl_atol_vec = np.concatenate([atol_phys, atol_phys, atol_react, atol_react])
                solver._nl_rtol_vec = np.concatenate([rtol_phys, rtol_phys, rtol_react, rtol_react])
            solver.lam = np.ones_like(Z0)
            Z_sol, F_sol, err, ok, iters = solver.solve(F, Z0)
        finally:
            solver.jacobian = saved_jac
            solver._cold_start_slices = saved_cold
            solver._nl_atol_vec = saved_atol_vec
            solver._nl_rtol_vec = saved_rtol_vec
            solver.lam = saved_lam

        F_aug = self._compress_stage_residual(F_sol)
        if not ok:
            try:
                import os as _os
                _diag_path = _os.environ.get(
                    "PR_CONTACT_FAILDIAG", "/tmp/sw_step_diag.log"
                )
                n = self.n_phys
                r = self.n_react
                F1 = np.asarray(F_sol[:n], dtype=float)
                F2 = np.asarray(F_sol[n:2 * n], dtype=float)
                F3 = np.asarray(F_sol[2 * n:2 * n + r], dtype=float)
                F4 = np.asarray(F_sol[2 * n + r:2 * n + 2 * r], dtype=float)
                Z1 = np.asarray(Z_sol[:n], dtype=float)
                Z3 = np.asarray(Z_sol[2 * n:2 * n + r], dtype=float)
                with open(_diag_path, "a") as _fh:
                    _fh.write(
                        f"FAIL t={float(t):.6f} h={float(h):.6e} iters={int(iters)}\n"
                    )
                    _fh.write(
                        f"  ||F||  Y1={np.linalg.norm(F1):.3e} Y2={np.linalg.norm(F2):.3e} "
                        f"dPi1={np.linalg.norm(F3):.3e} dPi2={np.linalg.norm(F4):.3e}\n"
                    )
                    _fh.write(
                        f"  max|F| Y1={np.max(np.abs(F1)):.3e}@{int(np.argmax(np.abs(F1)))} "
                        f"Y2={np.max(np.abs(F2)):.3e}@{int(np.argmax(np.abs(F2)))} "
                        f"dPi1={np.max(np.abs(F3)):.3e}@{int(np.argmax(np.abs(F3)))} "
                        f"dPi2={np.max(np.abs(F4)):.3e}@{int(np.argmax(np.abs(F4)))}\n"
                    )
                    _fh.write(
                        f"  ||Z||  Y1={np.linalg.norm(Z1):.3e} dPi1={np.linalg.norm(Z3):.3e}\n"
                    )
                    if self.reaction_state_indices is not None:
                        idx = self.reaction_state_indices
                        F1_react = F1[idx]
                        F1_phys = F1.copy()
                        F1_phys[idx] = 0.0
                        _fh.write(
                            f"  |F1@react_idx|={np.linalg.norm(F1_react):.3e} "
                            f"|F1@non_react|={np.linalg.norm(F1_phys):.3e}\n"
                        )
                    if self.constraint_q_slices:
                        for k_q, qs in enumerate(self.constraint_q_slices):
                            _fh.write(
                                f"  |F1@constr{k_q}|={np.linalg.norm(F1[qs]):.3e}\n"
                            )
            except Exception as _e:
                pass
            return y_aug, F_aug, np.zeros_like(y_aug), False, iters

        Y, dPi, dmu, _offsets = self.stage_quantities(Z_sol, t, h, rk_A, rk_c)
        self._remember_stage_guess(Y, dPi, y_prev, h)
        stage_pi = rk_b[0] * dPi[0] + rk_b[1] * dPi[1]
        y_plus, delta_pi, total_pi, total_eff, reported, proj_ok, proj_err = self.project_endpoint(
            Y[-1], y_prev, stage_pi, t + h, h
        )
        reported_storage = self._reported_from_impulse(total_pi, h)

        if self.reaction_state_indices is not None:
            y_new = np.asarray(y_plus, dtype=float).copy()
            self._write_storage_to_reaction_state(y_new, reported_storage)
        else:
            y_new = np.zeros(self.n_phys + self.n_react, dtype=float)
            y_new[: self.n_phys] = y_plus
            y_new[self.n_phys:] = reported

        err_embed = np.zeros_like(y_new)
        coeffs = getattr(integrator, "_err_coeffs", np.zeros(2))
        if len(coeffs) >= 2:
            err_embed[: self.n_phys] = coeffs[0] * (Y[0] - y_prev) + coeffs[1] * (Y[1] - y_prev)
            if self.reaction_state_indices is not None:
                err_embed[self.reaction_state_indices] = 0.0

        self.last_stage_dpi = np.vstack(dPi)
        self.last_stage_dmu = np.vstack(dmu)
        self.last_delta_pi = delta_pi.copy()
        self.last_total_pi = total_pi.copy()
        self.last_total_effective_pi = total_eff.copy()
        self.last_reported_reaction = reported.copy()
        stage_reported = self._reported_from_impulse(stage_pi, h)
        if self.reaction_state_indices is not None:
            stage_state = np.asarray(Y[-1], dtype=float).copy()
            self._write_storage_to_reaction_state(stage_state, stage_reported)
            integrator.last_post_step_delta = y_new - stage_state
        else:
            integrator.last_post_step_delta = y_new - np.concatenate([Y[-1], stage_reported])

        endpoint_residual = self._endpoint_contact_residual(
            y_plus, y_prev, total_eff, t + h, h
        )
        if self.reaction_state_indices is not None:
            F_aug[self.reaction_state_indices] = endpoint_residual
        else:
            F_aug[self.n_phys:] = endpoint_residual
        if not proj_ok:
            try:
                import os as _os
                _diag_path = _os.environ.get(
                    "PR_CONTACT_FAILDIAG", "/tmp/sw_step_diag.log"
                )
                ep_res = np.asarray(endpoint_residual, dtype=float)
                d_pi = np.asarray(delta_pi, dtype=float)
                t_pi = np.asarray(total_pi, dtype=float)
                t_eff = np.asarray(total_eff, dtype=float)
                gaps_at_end = np.asarray(self.gap(y_plus, t + h), dtype=float)
                u_end = np.asarray(self.contact_velocity(y_plus), dtype=float)
                with open(_diag_path, "a") as _fh:
                    _fh.write(
                        f"PROJ_FAIL t={float(t):.6f} h={float(h):.6e} proj_err={float(proj_err):.3e}\n"
                    )
                    _fh.write(
                        f"  ||ep_res||={np.linalg.norm(ep_res):.3e} max={np.max(np.abs(ep_res)):.3e}@{int(np.argmax(np.abs(ep_res)))}\n"
                    )
                    _fh.write(
                        f"  ||delta_pi||={np.linalg.norm(d_pi):.3e} ||total_pi||={np.linalg.norm(t_pi):.3e} ||total_eff||={np.linalg.norm(t_eff):.3e}\n"
                    )
                    _fh.write(
                        f"  gap min={float(np.min(gaps_at_end)):.3e} max={float(np.max(gaps_at_end)):.3e}\n"
                    )
                    _fh.write(
                        f"  |u_contact| max={float(np.max(np.abs(u_end))):.3e}\n"
                    )
                    for k_c, ci in enumerate(self.contacts):
                        sl = ci["block_slice"]
                        _fh.write(
                            f"  c{k_c}: ep_res[sl]={np.linalg.norm(ep_res[sl]):.3e} "
                            f"r_eff_N={float(t_eff[sl.start]):.3e} "
                            f"r_eff_T_norm={float(np.linalg.norm(t_eff[sl.start+1:sl.stop])):.3e} "
                            f"u_N={float(u_end[sl.start]):.3e} "
                            f"u_T_norm={float(np.linalg.norm(u_end[sl.start+1:sl.stop])):.3e}\n"
                        )
            except Exception:
                pass
        return y_new, F_aug, err_embed, bool(proj_ok), int(iters)


def build_projected_radau_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    component_slices=None,
    C_extract=None,
    D_extract=None,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    contact_law="fischer_burmeister",
    endpoint_law=None,
    normal_ncp_type=None,
    friction_ncp_type=None,
    friction_law="compliance",
    desaxce_alpha=None,
    desaxce_rho="auto",
    desaxce_rho_scale=1.0,
    desaxce_rho_min=1.0e-12,
    desaxce_rho_max=1.0e12,
    desaxce_tie_tol=1.0e-14,
    get_s0=None,
    get_w0=None,
    constant_contact_offsets=False,
    normal_r="auto",
    friction_r="auto",
    auto_rho_strategy="h_scaled",
    effective_jacobian_fn=None,
    delassus_cache_log_tol=0.5,
    projection_indices=None,
    restitution_normal=0.0,
    restitution_tangential=0.0,
    reported_reaction_units="force",
    reaction_state_indices=None,
    reaction_state_to_reported_scale=1.0,
    mask_reaction_state_in_smooth_rhs=True,
    endpoint_inactive_handling="gap",
):
    """Build an opt-in Breuling projected-Radau contact system."""
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size
    if C_extract is not None:
        C_extract = _dense_or_sparse(C_extract)
        if C_extract.shape[1] != n_phys:
            raise ValueError(f"C_extract has {C_extract.shape[1]} columns but n_phys = {n_phys}")
    if D_extract is not None:
        D_extract = _dense_or_sparse(D_extract)
        if D_extract.shape[1] != n_phys:
            raise ValueError(f"D_extract has {D_extract.shape[1]} columns but n_phys = {n_phys}")
    elif C_extract is not None:
        D_extract = C_extract
    if gap_func is None and C_extract is None:
        raise ValueError("gap_func must be provided when C_extract is None")

    norm_contacts = []
    reaction_extract_rows = []
    reaction_idx = 0
    for c in contacts:
        v_n = int(c["vel_normal_idx"])
        v_t = list(np.atleast_1d(c.get("vel_tangential_idx", [])).astype(int))
        mu_val = c.get("mu", 0.0)
        mu_is_const = not callable(mu_val)
        if callable(mu_val):
            get_mu = mu_val
        else:
            mu_const = float(mu_val)

            def get_mu(y, t=None, Fk_val=None, _m=mu_const):
                return _m

        dmu_dy = c.get("dmu_dy", c.get("mu_jac", None))

        rows = [v_n] + v_t
        reaction_extract_rows.extend(rows)
        norm_contacts.append({
            "vN": v_n,
            "vT": v_t,
            "block_slice": slice(reaction_idx, reaction_idx + len(rows)),
            "get_mu": get_mu,
            "mu_nargs": _count_required_args(get_mu),
            "mu_is_const": mu_is_const,
            "dmu_dy": dmu_dy,
            "dmu_dy_nargs": _count_required_args(dmu_dy),
        })
        reaction_idx += len(rows)

    n_react = reaction_idx
    if B is None and D_extract is not None:
        B_mat = D_extract[reaction_extract_rows, :].T.tocsr() if sp.issparse(D_extract) else np.asarray(D_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None and C_extract is not None:
        B_mat = C_extract[reaction_extract_rows, :].T.tocsr() if sp.issparse(C_extract) else np.asarray(C_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None:
        B_mat = np.zeros((n_phys, n_react), dtype=float)
        for col, row in enumerate(reaction_extract_rows):
            B_mat[row, col] = 1.0
    else:
        B_mat = _dense_or_sparse(B)
        if B_mat.shape != (n_phys, n_react):
            raise ValueError(f"B shape {B_mat.shape} doesn't match {(n_phys, n_react)}")

    reaction_state_indices_arr = None
    if reaction_state_indices is not None:
        reaction_state_indices_arr = np.asarray(reaction_state_indices, dtype=int).ravel()
        if reaction_state_indices_arr.size != n_react:
            raise ValueError(
                "reaction_state_indices must have one entry per reaction "
                f"DOF (got {reaction_state_indices_arr.size}, expected {n_react})"
            )
        if np.any(reaction_state_indices_arr < 0) or np.any(reaction_state_indices_arr >= n_phys):
            raise ValueError("reaction_state_indices contains out-of-range entries")
    use_inplace_reaction_state = reaction_state_indices_arr is not None
    n_aug = n_phys if use_inplace_reaction_state else n_phys + n_react

    if D_extract is not None:
        U_contact = D_extract[reaction_extract_rows, :].tocsr() if sp.issparse(D_extract) else np.asarray(D_extract[reaction_extract_rows, :], dtype=float)
    else:
        U_contact = None

    law_obj = contact_law
    endpoint_law_obj = endpoint_law
    if isinstance(contact_law, str):
        law_name = str(contact_law).strip().lower().replace("-", "_").replace(" ", "_")
        if law_name in {"desaxce", "de_saxce"}:
            law_obj = DeSaxceProjectedConeLaw(
                alpha=desaxce_alpha,
                rho=desaxce_rho,
                rho_scale=desaxce_rho_scale,
                rho_min=desaxce_rho_min,
                rho_max=desaxce_rho_max,
                tie_tol=desaxce_tie_tol,
            )
        elif law_name in {
            "soc_fb", "socfb", "soc_fischer_burmeister",
            "fischer_burmeister_soc", "fb_soc", "abh_soc_fb",
        }:
            # Breuling product cone at internal stages (position-level
            # Signorini scalar FB on the normal + velocity-level Coulomb
            # compliance on the tangential), De Saxce / SOC-FB bipotential
            # at the endpoint Stage 2 where velocity-cone duality is the
            # variationally correct setting.  See projected_radau_contact
            # design notes 2026-05-11.
            law_obj = NCPNormalConeLaw(
                ncp_type="fischer_burmeister",
                normal_ncp_type=normal_ncp_type,
                friction_ncp_type=friction_ncp_type,
                friction_law=friction_law,
            )
            if endpoint_law_obj is None:
                endpoint_law_obj = SOCFischerBurmeisterLaw(
                    alpha=desaxce_alpha,
                    tie_tol=desaxce_tie_tol,
                )
        elif law_name in {"soc_fb_uniform", "socfb_uniform"}:
            # Legacy single-law dispatch: SOC-FB at both stages.  Retained
            # for regression comparison with pre-split behavior.  Stage 1
            # SOC-FB is not Breuling-correct; prefer 'soc_fb' for new work.
            law_obj = SOCFischerBurmeisterLaw(
                alpha=desaxce_alpha,
                tie_tol=desaxce_tie_tol,
            )
        else:
            law_obj = NCPNormalConeLaw(
                ncp_type=contact_law,
                normal_ncp_type=normal_ncp_type,
                friction_ncp_type=friction_ncp_type,
                friction_law=friction_law,
            )
    if not isinstance(law_obj, ProjectedRadauContactLaw):
        if not hasattr(law_obj, "residual_and_jac"):
            raise TypeError("contact_law must be a string or expose residual_and_jac")
    if endpoint_law_obj is not None and not isinstance(
        endpoint_law_obj, ProjectedRadauContactLaw
    ):
        if not hasattr(endpoint_law_obj, "residual_and_jac"):
            raise TypeError("endpoint_law must expose residual_and_jac")

    algebraic_projection = None
    constraint_q_slices = []
    if constraints is not None:
        algebraic_projection = AlgebraicConstraintProjection(constraints=constraints)
        constraint_q_slices = list(algebraic_projection.constraint_q_slices)

    if use_inplace_reaction_state:
        A_aug = A.tocsr() if sp.issparse(A) else np.asarray(A, dtype=float).copy()
        y0_aug = y0.copy()
        if component_slices is not None:
            cs_aug = list(component_slices)
        else:
            cs_aug = [slice(0, n_phys)]
    else:
        if sp.issparse(A):
            A_aug = sp.block_diag([A, sp.csr_matrix((n_react, n_react))], format="csr")
        else:
            A_aug = np.zeros((n_aug, n_aug), dtype=float)
            A_aug[:n_phys, :n_phys] = np.asarray(A, dtype=float)
        y0_aug = np.zeros(n_aug, dtype=float)
        y0_aug[:n_phys] = y0

        if component_slices is not None:
            cs_aug = list(component_slices)
            cs_aug.append(slice(n_phys, n_aug))
        else:
            cs_aug = [slice(0, n_phys), slice(n_phys, n_aug)]

    model = ProjectedRadauContactModel(
        A_phys=A,
        rhs_smooth=rhs_smooth,
        y0_phys=y0,
        contacts=norm_contacts,
        gap_func=gap_func,
        B=B_mat,
        C_extract=C_extract,
        D_extract=D_extract,
        reaction_extract_rows=np.asarray(reaction_extract_rows, dtype=int),
        normal_rows=np.asarray([ci["vN"] for ci in norm_contacts], dtype=int),
        U_contact=U_contact,
        rhs_jac=rhs_jac,
        algebraic_projection=algebraic_projection,
        constraint_q_slices=constraint_q_slices,
        contact_law=law_obj,
        component_slices=cs_aug,
        projection_indices=projection_indices,
        restitution_normal=float(restitution_normal),
        restitution_tangential=float(restitution_tangential),
        reported_reaction_units=reported_reaction_units,
        get_s0=get_s0,
        get_w0=get_w0,
        constant_contact_offsets=bool(constant_contact_offsets),
        normal_r=normal_r,
        friction_r=friction_r,
        gap_tol=float(gap_tol),
        endpoint_inactive_handling=endpoint_inactive_handling,
        reaction_state_indices=reaction_state_indices_arr,
        reaction_state_to_reported_scale=reaction_state_to_reported_scale,
        mask_reaction_state_in_smooth_rhs=(
            bool(mask_reaction_state_in_smooth_rhs)
            if use_inplace_reaction_state else False
        ),
        auto_rho_strategy=auto_rho_strategy,
        effective_jacobian_fn=effective_jacobian_fn,
        delassus_cache_log_tol=float(delassus_cache_log_tol),
        endpoint_law=endpoint_law_obj,
    )

    if use_inplace_reaction_state:
        def rhs_aug(t, y, *extra):
            return model._smooth_rhs(t, np.asarray(y)[:n_phys])

        def jac_aug(t, y, *extra):
            J = model._smooth_jac(t, np.asarray(y)[:n_phys])
            return J.tocsr() if sp.issparse(J) else sp.csr_matrix(J)
    else:
        def rhs_aug(t, y, *extra):
            out = np.zeros(n_aug, dtype=float)
            out[:n_phys] = model._smooth_rhs(t, np.asarray(y)[:n_phys])
            return out

        def jac_aug(t, y, *extra):
            J = model._smooth_jac(t, np.asarray(y)[:n_phys])
            J = J.tocsr() if sp.issparse(J) else sp.csr_matrix(J)
            return sp.bmat(
                [[J, sp.csr_matrix((n_phys, n_react))],
                 [sp.csr_matrix((n_react, n_phys)), sp.csr_matrix((n_react, n_react))]],
                format="csr",
            )

    cs = ContactSystem(
        A=A_aug,
        rhs=rhs_aug,
        y0=y0_aug,
        projection=IdentityProjection(component_slices=cs_aug),
        component_slices=cs_aug,
        integrator_opts={
            "stages": 2,
            "pass_prev_state": True,
            "pass_step_size": True,
            "projected_radau_contact": model,
        },
        n_phys=n_phys,
        B=B_mat,
        rhs_jac=jac_aug,
        solver_opts={"rhs_jac": jac_aug},
    )
    cs.projected_radau_contact = model
    if use_inplace_reaction_state:
        cs.reaction_state_indices = reaction_state_indices_arr
        cs.reaction_state_to_reported_scale = model.reaction_state_to_reported_scale
        cs.reaction_history = lambda y_hist, t_hist=None: model.reaction_state_history(y_hist)
    else:
        cs.reaction_history = lambda y_hist, t_hist=None: np.asarray(y_hist)[:, n_phys:]
    return cs
