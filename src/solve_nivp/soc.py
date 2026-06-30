"""Second-order-cone convex-analysis primitives.

This module is the single home for the low-level second-order-cone (SOC)
operations used across the package.  It intentionally depends only on
NumPy so that integrators, contact laws, and SOCCP solvers can share these
kernels without importing one another.

Two distinct families live here:

* **Fischer-Burmeister complementarity** -- :func:`soc_fb_phi` and
  :func:`soc_fb_phi_and_jac` implement the Jordan-algebra SOC FB function
  ``Phi(x, y) = x + y - (x**2 + y**2)**(1/2)`` whose zeros are exactly the
  SOCCP solutions ``K ∋ x ⊥ y ∈ K``.  These are *complementarity functions*
  (two arguments, a velocity-like ``x`` and a force-like ``y``) consumed by
  semismooth Newton solvers.

* **Euclidean projection** -- :func:`proj_mu_scaled_soc` returns the
  nearest point of the mu-scaled friction cone ``K_mu`` to a single point
  ``z = (s, w)``, via the spectral (eigenvalue) decomposition.  This is the
  *projection operator* used by VI / projection fixed-point schemes and as
  the Clarke Jacobian source for the natural-residual map.

Both encode the same cone complementarity but are different maps; keep the
distinction in mind when choosing one.
"""
from __future__ import annotations

import math

import numpy as np


# -----------------------------------------------------------------------------
# Fischer-Burmeister complementarity (Jordan algebra)
# -----------------------------------------------------------------------------


def soc_fb_phi(x: np.ndarray, y: np.ndarray, *, tie_tol: float = 1.0e-14) -> np.ndarray:
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

    # Scalar fast path for the dominant 2-D contact case (one normal + one
    # tangential DOF).  Identical arithmetic to the generic branch below, but
    # avoids the per-call overhead of np.linalg.norm / np.empty on tiny arrays
    # -- this kernel is called O(contacts * outer * newton) times per solve.
    if d == 2:
        x0 = float(x[0]); x1 = float(x[1])
        y0 = float(y[0]); y1 = float(y[1])
        w_N = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1
        w_T = 2.0 * (x0 * x1 + y0 * y1)
        w_T_norm = abs(w_T)
        sqrt_lam1 = math.sqrt(max(w_N - w_T_norm, 0.0))
        sqrt_lam2 = math.sqrt(max(w_N + w_T_norm, 0.0))
        s_N = 0.5 * (sqrt_lam1 + sqrt_lam2)
        if w_T_norm > tie_tol:
            s_T = 0.5 * (sqrt_lam2 - sqrt_lam1) * (1.0 if w_T >= 0.0 else -1.0)
        else:
            s_T = 0.0
        return np.array([x0 + y0 - s_N, x1 + y1 - s_T])

    w_N = float(x @ x + y @ y)
    if d == 1:
        w_T = np.zeros(0, dtype=float)
        w_T_norm = 0.0
    else:
        w_T = 2.0 * (x[0] * x[1:] + y[0] * y[1:])
        w_T_norm = math.sqrt(float(w_T @ w_T))

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


def soc_jordan_multiplication(x: np.ndarray) -> np.ndarray:
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


def soc_fb_phi_and_jac_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tie_tol: float = 1.0e-14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Closed-form d = 2 specialization of :func:`soc_fb_phi_and_jac`.

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


def soc_fb_phi_and_jac(
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
        return soc_fb_phi_and_jac_2d(x, y, tie_tol=tie_tol)

    # Residual + s, computed once.
    phi = soc_fb_phi(x, y, tie_tol=tie_tol)
    s = (x + y) - phi  # equals (x² + y²)^{1/2}

    L_s = soc_jordan_multiplication(s)
    L_x = soc_jordan_multiplication(x)
    L_y = soc_jordan_multiplication(y)
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


# -----------------------------------------------------------------------------
# Euclidean projection onto the mu-scaled second-order cone
# -----------------------------------------------------------------------------


def proj_mu_scaled_soc(z, mu, return_jacobian=False, eps=1e-30):
    r"""Project ``z = (s, w)`` onto :math:`K_\mu` via spectral eigenvalues.

    Spectral eigenvalues of the μ-scaled SOC:

    .. math::
        \lambda_+ = s + \mu\,r, \qquad
        \lambda_- = s - r/\mu \quad (\mu > 0)

    where :math:`r = \|w\|`.  Region classification:

    * **Interior**: :math:`\lambda_- \ge 0` and :math:`s \ge 0`
    * **Polar**: :math:`\lambda_+ \le 0`
    * **Boundary**: :math:`\lambda_+ > 0` and :math:`\lambda_- < 0`

    Projection via positive-part of eigenvalues:

    .. math::
        \Pi_{K_\mu}(z) = (\lambda_+)_+\, c_+ + (\lambda_-)_+\, c_-

    No value regularisation is needed: when :math:`r = 0`,
    :math:`\lambda_+ = \lambda_- = s`, so the point is interior
    (:math:`s \ge 0`) or polar (:math:`s \le 0`), never boundary.

    Parameters
    ----------
    z : ndarray, shape (1+m,)
        The vector ``[s, w_1, ..., w_m]``.
    mu : float
        Friction coefficient (cone opening).
    return_jacobian : bool
        If ``True`` return ``(projection, jacobian)`` where
        *jacobian* is the ``(1+m, 1+m)`` Clarke sub-differential.
    eps : float
        Kept for API compatibility; only used to stabilise the
        angular-stiffness ratio :math:`\lambda_+/r` in the Jacobian
        when :math:`r \to 0`.

    Returns
    -------
    p : ndarray, shape (1+m,)
        or ``(p, J)`` when ``return_jacobian=True``.
    """
    z = np.asarray(z, dtype=float)
    s = float(z[0])
    w = z[1:].copy()
    m = w.size
    d = 1 + m                       # block dimension
    r = float(np.linalg.norm(w))     # ||w||

    # ---- μ = 0: degenerate cone K_0 = {(s, 0) : s ≥ 0} ----
    if mu <= 0.0:
        p = np.zeros(d)
        p[0] = max(s, 0.0)
        if not return_jacobian:
            return p
        J = np.zeros((d, d))
        if s >= 0.0:
            J[0, 0] = 1.0       # Clarke selection at s = 0
        return p, J

    # ---- Spectral eigenvalues ----
    lam_plus  = s + mu * r       # λ₊ = s + μ‖w‖
    lam_minus = s - r / mu       # λ₋ = s − ‖w‖/μ

    # ---- Region 1: interior (λ₋ ≥ 0 and s ≥ 0) ----
    if lam_minus >= 0.0 and s >= 0.0:
        if return_jacobian:
            return z.copy(), np.eye(d)
        return z.copy()

    # ---- Region 2: polar (λ₊ ≤ 0) ----
    if lam_plus <= 0.0:
        if return_jacobian:
            return np.zeros(d), np.zeros((d, d))
        return np.zeros(d)

    # ---- Region 3: boundary (λ₊ > 0, λ₋ < 0) ----
    # r > 0 is guaranteed here: λ₊ > 0 and λ₋ < 0 with μ > 0
    # implies r > 0 (if r = 0, then λ₊ = λ₋ = s, same sign).
    alpha = 1.0 / (1.0 + mu * mu)
    w_hat = w / r                  # exact, no regularisation

    p = np.empty(d)
    p[0] = alpha * lam_plus
    p[1:] = (alpha * mu * lam_plus) * w_hat

    if not return_jacobian:
        return p

    # Clarke sub-differential on the boundary:
    #   J = α [ 1          μ ŵᵀ                         ]
    #         [ μ ŵ        μ² ŵŵᵀ + μ(λ₊/r)(I − ŵŵᵀ)  ]
    wwT = np.outer(w_hat, w_hat)
    r_jac = max(r, eps)            # stabilise λ₊/r near apex
    J = np.empty((d, d))
    J[0, 0] = alpha
    J[0, 1:] = alpha * mu * w_hat
    J[1:, 0] = alpha * mu * w_hat
    J[1:, 1:] = alpha * (
        mu * mu * wwT
        + mu * (lam_plus / r_jac) * (np.eye(m) - wwT)
    )

    return p, J


# -----------------------------------------------------------------------------
# Backward-compatible private aliases
#
# These names were previously defined in ``projected_radau_contact`` and are
# re-exported here (and from that module) so existing imports keep working.
# -----------------------------------------------------------------------------

_soc_fb_phi = soc_fb_phi
_soc_jordan_multiplication = soc_jordan_multiplication
_soc_fb_phi_and_jac_2d = soc_fb_phi_and_jac_2d
_soc_fb_phi_and_jac = soc_fb_phi_and_jac
