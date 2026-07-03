"""Optional numba-accelerated kernels for tight loops.

These functions are imported opportunistically; if numba is not available,
the module exposes NUMBA_AVAILABLE=False and no-accel fallbacks are used.
"""
from __future__ import annotations

import numpy as np

NUMBA_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - environment may not have numba
    def njit(*args, **kwargs):  # type: ignore
        def deco(f):
            return f
        return deco


@njit(cache=True)
def _projD_element(v: float, z: float, friction_val: float):
    """Project a single (v,z) pair given friction_val onto the monotone cone used by Coulomb.

    Returns (v_proj, z_proj).
    Matches logic of _projD_optimized but element-wise.
    """
    if friction_val == 0.0:
        return v, z
    # Regions
    az = abs(z)
    av = abs(v)
    if az <= v:
        s1 = 0.5 * (v + z)
        return s1, s1
    if av <= -z:
        return 0.0, 0.0
    if az <= -v:
        s2 = 0.5 * (-v + z)
        return -s2, s2
    # Identity region
    return v, z


@njit(cache=True)
def projD_optimized_nb(v: "float[:]", z: "float[:]", friction_vals: "float[:]"):
    n = v.shape[0]
    out_v = v.copy()
    out_z = z.copy()
    for i in range(n):
        vv, zz = _projD_element(v[i], z[i], friction_vals[i])
        out_v[i] = vv
        out_z[i] = zz
    return out_v, out_z


@njit(cache=True)
def classify_regions_nb(v_arr: "float[:]", zt_arr: "float[:]", tol: float):
    """Classify regions per constrained pair for tangent_cone selection.

    Returns int8 codes:
      0: P_zero (tip or region 2)
      1: P_I
      2: P_ray_pp
      3: P_ray_mp
      4: P_tie
    Applies the same scaled tolerance policy as Python code.
    """
    n = v_arr.shape[0]
    codes = [0] * n  # use Python list for numba simplicity; cast by caller
    for k in range(n):
        v = v_arr[k]
        zt = zt_arr[k]
        scale = 1.0 + (abs(v) if abs(v) > abs(zt) else abs(zt))
        ts = tol * scale
        if (abs(v) <= ts) and (abs(zt) <= ts):
            codes[k] = 0  # tip -> zero
            continue
        if abs(zt) < (v - ts):
            codes[k] = 2  # ray ++
            continue
        if abs(v) < (-zt - ts):
            codes[k] = 0  # region 2 -> zero
            continue
        if abs(zt) < (-v - ts):
            codes[k] = 3  # ray -+
            continue
        if (abs(abs(zt) - v) <= ts) or (abs(abs(v) + zt) <= ts):
            codes[k] = 4  # tie
            continue
        codes[k] = 1  # identity
    return codes


@njit(cache=True)
def wrms_kernel(F, y, atol_v, rtol_v):
    """Compute weighted RMS norm in a single pass — no intermediate arrays.

    Equivalent to ``sqrt(mean((F / (atol + rtol*|y|))^2))``.
    """
    n = F.shape[0]
    acc = 0.0
    for i in range(n):
        w = atol_v[i] + rtol_v[i] * abs(y[i])
        s = F[i] / w
        acc += s * s
    return (acc / n) ** 0.5


# -----------------------------------------------------------------------------
# Fused MJF contact SSN assembly (soc_fb residual + Jacobian)
#
# Compiled drop-in for the per-contact loop in
# ``DescriptorMoreauJeanFremondStepper._contact_ssn_residual_jacobian`` in the
# ``soc_fb`` residual mode with a built-in Frémond/De Saxce shift.  It fuses the
# whole per-block computation (shift, mu-scaling, SOC Fischer-Burmeister
# residual + Jordan-algebra Jacobian, dense Jacobian assembly) into one
# allocation-free pass, writing straight into the dense Jacobian.  The pure
# Python/NumPy path remains the fallback for custom shifts, mixed block
# dimensions, the soc_projection mode, or when numba is unavailable.
# -----------------------------------------------------------------------------


@njit(cache=True)
def _arrow_nb(v):
    """Jordan multiplication (arrow) matrix L_v on R x R^(d-1)."""
    d = v.shape[0]
    L = np.zeros((d, d))
    L[0, 0] = v[0]
    for j in range(1, d):
        L[0, j] = v[j]
        L[j, 0] = v[j]
        L[j, j] = v[0]
    return L


@njit(cache=True)
def _pinv_rcond_nb(A, rcond):
    """Moore-Penrose pseudoinverse matching np.linalg.pinv(A, rcond=rcond)."""
    u, s, vt = np.linalg.svd(A)
    smax = 0.0
    for i in range(s.shape[0]):
        if s[i] > smax:
            smax = s[i]
    cutoff = rcond * smax
    sinv = np.zeros(s.shape[0])
    for i in range(s.shape[0]):
        if s[i] > cutoff:
            sinv[i] = 1.0 / s[i]
    d = A.shape[0]
    P = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            acc = 0.0
            for k in range(d):
                acc += vt[k, i] * sinv[k] * u[j, k]
            P[i, j] = acc
    return P


@njit(cache=True)
def _soc_fb_phi_jac_nb(x, y, want_jac, tie):
    """SOC Fischer-Burmeister residual + Jordan-algebra Jacobians.

    Mirrors :func:`solve_nivp.soc.soc_fb_phi_and_jac` (closed form for d == 2,
    spectral form with interior-solve / boundary-pinv for d >= 3).
    """
    d = x.shape[0]
    if d == 2:
        x0 = x[0]; x1 = x[1]; y0 = y[0]; y1 = y[1]
        w0 = x0 * x0 + x1 * x1 + y0 * y0 + y1 * y1
        w1 = 2.0 * (x0 * x1 + y0 * y1)
        aw = abs(w1)
        # lam1 = w0 - |w1| cancellation-free (w0, |w1| ~ ||reaction||^2 cancel near the cone
        # boundary): (w0^2-|w1|^2)/(w0+|w1|) with the numerator as a sum of squares.  Mirrors
        # solve_nivp.soc._fb_sqrt_lam1_2d; keeps the FB root on the cone to machine precision.
        _fx = (x0 - x1) * (x0 + x1)
        _fy = (y0 - y1) * (y0 + y1)
        _c1 = x0 * y0 - x1 * y1
        _c2 = x0 * y1 - x1 * y0
        _den = w0 + aw
        _lam1 = (_fx * _fx + _fy * _fy + 2.0 * (_c1 * _c1 + _c2 * _c2)) / _den if _den > 0.0 else 0.0
        sq1 = np.sqrt(_lam1) if _lam1 > 0.0 else 0.0
        sq2 = np.sqrt(max(w0 + aw, 0.0))
        s0 = 0.5 * (sq1 + sq2)
        if aw > tie:
            s1 = 0.5 * (sq2 - sq1) * (1.0 if w1 >= 0.0 else -1.0)
        else:
            s1 = 0.0
        phi = np.empty(2)
        phi[0] = x0 + y0 - s0
        phi[1] = x1 + y1 - s1
        dX = np.zeros((2, 2))
        dY = np.zeros((2, 2))
        if want_jac:
            interior = (s0 > abs(s1) + tie) and (s0 > tie)
            if interior:
                det = s0 * s0 - s1 * s1
                i00 = s0 / det
                i01 = -s1 / det
            else:
                lam_p = s0 + s1
                lam_m = s0 - s1
                cutoff = tie * max(lam_p, lam_m)
                c_p = 1.0 / lam_p if lam_p > cutoff else 0.0
                c_m = 1.0 / lam_m if lam_m > cutoff else 0.0
                i00 = 0.5 * (c_p + c_m)
                i01 = 0.5 * (c_p - c_m)
            dX[0, 0] = 1.0 - (i00 * x0 + i01 * x1)
            dX[0, 1] = -(i00 * x1 + i01 * x0)
            dX[1, 0] = -(i01 * x0 + i00 * x1)
            dX[1, 1] = 1.0 - (i01 * x1 + i00 * x0)
            dY[0, 0] = 1.0 - (i00 * y0 + i01 * y1)
            dY[0, 1] = -(i00 * y1 + i01 * y0)
            dY[1, 0] = -(i01 * y0 + i00 * y1)
            dY[1, 1] = 1.0 - (i01 * y1 + i00 * y0)
        return phi, dX, dY

    # Generic d >= 3 (and d == 1 falls through harmlessly).
    wN = 0.0
    for j in range(d):
        wN += x[j] * x[j] + y[j] * y[j]
    wT = np.empty(d - 1)
    for j in range(d - 1):
        wT[j] = 2.0 * (x[0] * x[j + 1] + y[0] * y[j + 1])
    wTn = 0.0
    for j in range(d - 1):
        wTn += wT[j] * wT[j]
    wTn = np.sqrt(wTn)
    # lam1 = wN - ||wT|| cancellation-free (mirrors solve_nivp.soc._fb_lam1_general): sum-of-squares
    # numerator with the Cauchy-Schwarz/Gram term, over (wN + ||wT||).
    _axT = 0.0; _ayT = 0.0; _xy = 0.0
    for j in range(1, d):
        _axT += x[j] * x[j]; _ayT += y[j] * y[j]; _xy += x[j] * y[j]
    _axT = np.sqrt(_axT); _ayT = np.sqrt(_ayT)
    _fx = (x[0] - _axT) * (x[0] + _axT)
    _fy = (y[0] - _ayT) * (y[0] + _ayT)
    _c1 = x[0] * y[0] - _xy
    _c2 = 0.0
    for j in range(1, d):
        _cr = x[0] * y[j] - y[0] * x[j]; _c2 += _cr * _cr
    _gram = 0.0
    for i in range(1, d):
        for j in range(i + 1, d):
            _t = x[i] * y[j] - x[j] * y[i]; _gram += _t * _t
    _den = wN + wTn
    _lam1 = (_fx * _fx + _fy * _fy + 2.0 * (_c1 * _c1 + _c2 + _gram)) / _den if _den > 0.0 else 0.0
    sq1 = np.sqrt(_lam1) if _lam1 > 0.0 else 0.0
    sq2 = np.sqrt(max(wN + wTn, 0.0))
    sN = 0.5 * (sq1 + sq2)
    phi = np.empty(d)
    s = np.empty(d)
    phi[0] = x[0] + y[0] - sN
    s[0] = sN
    if wTn > tie:
        coef = 0.5 * (sq2 - sq1) / wTn
        for j in range(d - 1):
            sT = coef * wT[j]
            s[j + 1] = sT
            phi[j + 1] = x[j + 1] + y[j + 1] - sT
    else:
        for j in range(d - 1):
            s[j + 1] = 0.0
            phi[j + 1] = x[j + 1] + y[j + 1]
    dX = np.zeros((d, d))
    dY = np.zeros((d, d))
    if want_jac:
        L_s = _arrow_nb(s)
        L_x = _arrow_nb(x)
        L_y = _arrow_nb(y)
        eye = np.eye(d)
        s_T_norm = 0.0
        for j in range(1, d):
            s_T_norm += s[j] * s[j]
        s_T_norm = np.sqrt(s_T_norm)
        interior = (s[0] > s_T_norm + tie) and (s[0] > tie)
        if interior:
            dX = eye - np.linalg.solve(L_s, L_x)
            dY = eye - np.linalg.solve(L_s, L_y)
        else:
            L_s_pinv = _pinv_rcond_nb(L_s, tie)
            dX = eye - L_s_pinv @ L_x
            dY = eye - L_s_pinv @ L_y
    return phi, dX, dY


@njit(cache=True)
def _scalar_fb_nb(a, b):
    """Scalar Fischer-Burmeister, mirrors ``_scalar_fb_and_jac``."""
    norm = np.hypot(a, b)
    if norm > 1.0e-14:
        return norm - a - b, a / norm - 1.0, b / norm - 1.0
    return 0.0, -1.0, -1.0


@njit(cache=True)
def soc_fb_ssn_assemble(W, b, p, mu, alpha, rest, d, want_jac, tie):
    """Fused soc_fb SSN residual (+ dense Jacobian) for uniform block dim d.

    ``W`` (n, n) dense C-contiguous Delassus, ``b``/``p`` (n,), ``mu``/``alpha``/
    ``rest`` (N,) per-contact friction, shift tangential coupling, and constant
    normal shift offset.  Returns ``(residual, jac, u_full)``; ``jac`` is a 1x1
    dummy when ``want_jac`` is False.
    """
    N = mu.shape[0]
    n = N * d
    u_full = W @ p + b
    residual = np.zeros(n)
    if want_jac:
        jac = np.zeros((n, n))
    else:
        jac = np.zeros((1, 1))

    for k in range(N):
        i0 = k * d
        mk = mu[k]
        ak = alpha[k]
        u = u_full[i0:i0 + d].copy()
        pb = p[i0:i0 + d].copy()

        # Frémond/De Saxce shift  u_hat = u + (alpha*||u_T|| + rest, 0).
        nT = 0.0
        for j in range(1, d):
            nT += u[j] * u[j]
        nT = np.sqrt(nT)
        u_hat = u.copy()
        u_hat[0] += ak * nT + rest[k]

        if mk <= 1.0e-14 or d == 1:
            # Scalar normal NCP + identity tangential rows.
            phi0, dphi_du, dphi_dp = _scalar_fb_nb(u_hat[0], pb[0])
            residual[i0] = phi0
            if want_jac:
                # jac[i0, :] = dphi_du * (d_uhat_du[0, :] @ W_rows)
                for c in range(n):
                    jac[i0, c] = dphi_du * W[i0, c]
                if nT > 0.0:
                    for j in range(1, d):
                        g = ak * u[j] / nT
                        for c in range(n):
                            jac[i0, c] += dphi_du * g * W[i0 + j, c]
                elif ak > 0.0:
                    for j in range(1, d):
                        for c in range(n):
                            jac[i0, c] += dphi_du * ak * W[i0 + j, c]
                jac[i0, i0] += dphi_dp
            for j in range(1, d):
                residual[i0 + j] = pb[j]
                if want_jac:
                    jac[i0 + j, i0 + j] = 1.0
            continue

        # Cone branch.  Diagonal rescalings T_x=diag(1,mu), T_y=diag(mu,1),
        # T_y_inv=diag(1/mu,1) applied as scalings.
        x_sd = u_hat.copy()
        for j in range(1, d):
            x_sd[j] *= mk
        y_sd = pb.copy()
        y_sd[0] *= mk
        phi, dX, dY = _soc_fb_phi_jac_nb(x_sd, y_sd, want_jac, tie)
        tyi0 = 1.0 / mk
        residual[i0] = tyi0 * phi[0]
        for j in range(1, d):
            residual[i0 + j] = phi[j]

        if want_jac:
            # d_uhat_du (d, d)
            Duh = np.eye(d)
            if nT > 0.0:
                for j in range(1, d):
                    Duh[0, j] = ak * u[j] / nT
            elif ak > 0.0:
                for j in range(1, d):
                    Duh[0, j] = ak
            W_rows = np.ascontiguousarray(W[i0:i0 + d, :])
            DuW = Duh @ W_rows
            # M_row = T_x @ DuW  (scale row j>=1 by mu)
            for j in range(1, d):
                for c in range(n):
                    DuW[j, c] *= mk
            JX = dX @ DuW
            # jac[block rows, :] = T_y_inv @ JX
            for c in range(n):
                jac[i0, c] = tyi0 * JX[0, c]
            for j in range(1, d):
                for c in range(n):
                    jac[i0 + j, c] = JX[j, c]
            # jac[block, block] += T_y_inv @ (dphi_dy @ T_y);  T_y=diag(mu,1..)
            for a in range(d):
                ta = tyi0 if a == 0 else 1.0
                for bb in range(d):
                    tb = mk if bb == 0 else 1.0
                    jac[i0 + a, i0 + bb] += ta * dY[a, bb] * tb

    return residual, jac, u_full
