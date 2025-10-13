"""Optional numba-accelerated kernels for tight loops.

These functions are imported opportunistically; if numba is not available,
the module exposes NUMBA_AVAILABLE=False and no-accel fallbacks are used.
"""
from __future__ import annotations

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
