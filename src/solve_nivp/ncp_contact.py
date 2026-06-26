"""NCP-based frictional contact helpers.

This module provides a full-state contact backend that embeds minimum-map
or Fischer-Burmeister nonlinear complementarity functions directly into the
augmented residual. The structure mirrors
``solve_nivp.alart_curnier_contact`` so it can be used with the package's
existing implicit integrators and semismooth-Newton solve path.

The contact residual combines:

* a position-level normal NCP function on the gap and normal reaction, and
* a velocity-level tangential compliance law

    f_T = u_T + W(u_T, r_N, r_T) r_T

where ``W`` is the fixed-point friction compliance from Macklin et al.
("Non-Smooth Newton Methods for Deformable Multi-Body Dynamics", 2019).
Following that paper, ``W`` is treated as constant inside each Newton
linearization, yielding a quasi-Newton but integrator-compatible backend.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .alart_curnier_contact import (
    _call_state_block_time_fk,
    _call_state_time_fk,
    _call_with_time_state_fk,
    _count_required_args,
    _dense_or_sparse,
    _eval_ds0_dz,
    _eval_dw0_dz,
    _eval_s0,
    _eval_w0,
    _parse_prev_and_h,
    _project_ball_and_jac,
    _vectorize_mu,
)
from .contact import ContactSystem
from .projections import AlgebraicConstraintProjection, IdentityProjection


def _normalize_ncp_name(name: str, *, label: str) -> str:
    value = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "min": "minimum_map",
        "minimum": "minimum_map",
        "minimum_map": "minimum_map",
        "minmap": "minimum_map",
        "fb": "fischer_burmeister",
        "fischer": "fischer_burmeister",
        "burmeister": "fischer_burmeister",
        "fischer_burmeister": "fischer_burmeister",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise ValueError(
            f"{label} must be 'minimum_map' or 'fischer_burmeister' (got {name!r})"
        ) from exc


def _eval_contact_scalar_field(spec, nargs, n_blocks, name, y, t=None, Fk_val=None):
    """Evaluate a scalar or per-contact scalar specification."""
    if callable(spec):
        raw = _call_state_time_fk(spec, nargs, y, t=t, Fk_val=Fk_val)
    else:
        raw = spec
    arr = np.atleast_1d(np.asarray(raw, dtype=float))
    if arr.size == 1:
        arr = np.full(n_blocks, float(arr.flat[0]), dtype=float)
    elif arr.size != n_blocks:
        raise ValueError(
            f"{name} must be a scalar or array of length {n_blocks} (got size {arr.size})"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite (got {arr})")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive (got {arr})")
    return arr.ravel()


def _minimum_map_ncp(a, b, scale, *, tie_tol=1.0e-14):
    """Minimum-map NCP function ``min(a, scale*b)`` and one sub-derivative."""
    scaled_b = float(scale * b)
    if float(a) <= scaled_b + tie_tol:
        return float(a), 1.0, 0.0
    return scaled_b, 0.0, float(scale)


def _fischer_burmeister_ncp(a, b, scale, *, tie_tol=1.0e-14):
    """Fischer-Burmeister NCP function and one Clarke derivative."""
    a = float(a)
    scale = float(scale)
    scaled_b = float(scale * b)
    rad = float(np.hypot(a, scaled_b))
    phi = a + scaled_b - rad
    if rad <= tie_tol:
        # Macklin et al. choose alpha(0,0)=0, beta(0,0)=1.
        return phi, 0.0, scale
    dphi_da = 1.0 - (a / rad)
    dphi_db = scale * (1.0 - (scaled_b / rad))
    return phi, dphi_da, dphi_db


def _minimum_map_friction_compliance(speed, cone_gap, mu_lambda_n, scale, *, tie_tol=1.0e-14):
    """Fixed-point friction compliance ``W`` for the minimum-map NCP."""
    if float(speed) <= float(scale * cone_gap) + tie_tol:
        return 0.0
    denom = float(mu_lambda_n)
    if denom <= tie_tol:
        return 0.0
    return max(0.0, float(speed - scale * cone_gap) / denom)


def _fischer_burmeister_friction_compliance(
    speed, cone_gap, mu_lambda_n, scale, *, tie_tol=1.0e-14
):
    """Fixed-point friction compliance ``W`` for the Fischer-Burmeister NCP."""
    speed = float(speed)
    scale = float(scale)
    cone_gap = float(cone_gap)
    mu_lambda_n = float(mu_lambda_n)
    scaled_gap = scale * cone_gap
    rad = float(np.hypot(speed, scaled_gap))
    numer = rad - scaled_gap
    denom = speed + scale * mu_lambda_n - rad
    if numer <= tie_tol and abs(denom) <= tie_tol:
        return 0.0
    if denom <= tie_tol:
        return 0.0 if numer <= tie_tol else scale * numer / tie_tol
    return max(0.0, scale * numer / denom)


def _minimum_map_friction_compliance_and_jac(
    speed, cone_gap, mu_lambda_n, scale, *, tie_tol=1.0e-14
):
    """Minimum-map compliance ``W`` and its local partial derivatives."""
    speed = float(speed)
    scale = float(scale)
    cone_gap = float(cone_gap)
    mu_lambda_n = float(mu_lambda_n)
    threshold = scale * cone_gap
    if speed <= threshold + tie_tol:
        return 0.0, 0.0, 0.0, 0.0
    denom = mu_lambda_n
    if denom <= tie_tol:
        return 0.0, 0.0, 0.0, 0.0
    numer = speed - threshold
    W = max(0.0, numer / denom)
    if W <= 0.0:
        return 0.0, 0.0, 0.0, 0.0
    dW_dspeed = 1.0 / denom
    dW_dgap = -scale / denom
    dW_dmulambda = -numer / max(denom * denom, tie_tol)
    return W, dW_dspeed, dW_dgap, dW_dmulambda


def _fischer_burmeister_friction_compliance_and_jac(
    speed, cone_gap, mu_lambda_n, scale, *, tie_tol=1.0e-14
):
    """Fischer-Burmeister compliance ``W`` and its local partial derivatives."""
    speed = float(speed)
    scale = float(scale)
    cone_gap = float(cone_gap)
    mu_lambda_n = float(mu_lambda_n)
    scaled_gap = scale * cone_gap
    rad = float(np.hypot(speed, scaled_gap))
    numer = rad - scaled_gap
    denom = speed + scale * mu_lambda_n - rad
    if numer <= tie_tol and abs(denom) <= tie_tol:
        return 0.0, 0.0, 0.0, 0.0
    if denom <= tie_tol:
        if numer <= tie_tol:
            return 0.0, 0.0, 0.0, 0.0
        W = scale * numer / tie_tol
        return W, 0.0, 0.0, 0.0

    W = max(0.0, scale * numer / denom)
    if W <= 0.0:
        return 0.0, 0.0, 0.0, 0.0

    if rad <= tie_tol:
        drad_dspeed = 0.0
        drad_dscaled_gap = 0.0
    else:
        drad_dspeed = speed / rad
        drad_dscaled_gap = scaled_gap / rad

    dnumer_dspeed = drad_dspeed
    dnumer_dscaled_gap = drad_dscaled_gap - 1.0
    ddenom_dspeed = 1.0 - drad_dspeed
    ddenom_dscaled_gap = -drad_dscaled_gap
    ddenom_dmulambda = scale

    factor = scale / (denom * denom)
    dW_dspeed = factor * (dnumer_dspeed * denom - numer * ddenom_dspeed)
    dW_dscaled_gap = factor * (
        dnumer_dscaled_gap * denom - numer * ddenom_dscaled_gap
    )
    dW_dgap = dW_dscaled_gap * scale
    dW_dmulambda = factor * (-numer * ddenom_dmulambda)
    return W, dW_dspeed, dW_dgap, dW_dmulambda


def _normal_ncp_residual_and_jac(gap, r_n, ncp_type, scale, *, tie_tol=1.0e-14):
    if ncp_type == "minimum_map":
        return _minimum_map_ncp(gap, r_n, scale, tie_tol=tie_tol)
    return _fischer_burmeister_ncp(gap, r_n, scale, tie_tol=tie_tol)


def _friction_compliance(speed, cone_gap, mu_lambda_n, ncp_type, scale, *, tie_tol=1.0e-14):
    if ncp_type == "minimum_map":
        return _minimum_map_friction_compliance(
            speed, cone_gap, mu_lambda_n, scale, tie_tol=tie_tol
        )
    return _fischer_burmeister_friction_compliance(
        speed, cone_gap, mu_lambda_n, scale, tie_tol=tie_tol
    )


def _friction_compliance_and_jac(
    speed, cone_gap, mu_lambda_n, ncp_type, scale, *, tie_tol=1.0e-14
):
    if ncp_type == "minimum_map":
        return _minimum_map_friction_compliance_and_jac(
            speed, cone_gap, mu_lambda_n, scale, tie_tol=tie_tol
        )
    return _fischer_burmeister_friction_compliance_and_jac(
        speed, cone_gap, mu_lambda_n, scale, tie_tol=tie_tol
    )


def _contact_block_residual_and_jac_2d(
    gap,
    u_blk,
    r_blk,
    mu,
    normal_ncp_type,
    friction_ncp_type,
    normal_scale,
    friction_scale,
    friction_law="compliance",
    *,
    tie_tol=1.0e-14,
):
    r"""Specialized two-row NCP block residual and quasi-Newton Jacobian."""
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)

    f_blk = np.zeros(2, dtype=float)
    df_dgap = np.zeros(2, dtype=float)
    df_du = np.zeros((2, 2), dtype=float)
    df_dr = np.zeros((2, 2), dtype=float)

    r_n = float(r_blk[0])
    phi_n, dphi_dgap, dphi_drn = _normal_ncp_residual_and_jac(
        gap, r_n, normal_ncp_type, normal_scale, tie_tol=tie_tol
    )
    f_blk[0] = phi_n
    df_dgap[0] = dphi_dgap
    df_dr[0, 0] = dphi_drn

    r_t = float(r_blk[1])
    mu_lambda_n = float(mu * r_n)
    if mu_lambda_n <= tie_tol:
        f_blk[1] = r_t
        df_dr[1, 1] = 1.0
        return f_blk, df_dgap, df_du, df_dr

    u_t = float(u_blk[1])
    if friction_law == "natural_map":
        y_t = r_t - float(friction_scale) * u_t
        abs_y = abs(y_t)
        if mu_lambda_n <= 0.0:
            proj = 0.0
            dproj_ddelta = 0.0
            dproj_dy = 0.0
        elif abs_y <= mu_lambda_n + tie_tol:
            proj = y_t
            dproj_ddelta = 0.0
            dproj_dy = 1.0
        else:
            sign_y = 1.0 if y_t >= 0.0 else -1.0
            proj = mu_lambda_n * sign_y
            dproj_ddelta = sign_y
            dproj_dy = 0.0
        f_blk[1] = proj - r_t
        df_du[1, 1] = -float(friction_scale) * dproj_dy
        df_dr[1, 1] = dproj_dy - 1.0
        df_dr[1, 0] = float(mu) * dproj_ddelta
        return f_blk, df_dgap, df_du, df_dr

    speed = abs(u_t)
    r_t_norm = abs(r_t)
    cone_gap = mu_lambda_n - r_t_norm
    W, dW_dspeed, dW_dgap, dW_dmulambda = _friction_compliance_and_jac(
        speed, cone_gap, mu_lambda_n, friction_ncp_type, friction_scale,
        tie_tol=tie_tol,
    )

    f_blk[1] = u_t + W * r_t
    if speed > tie_tol:
        dspeed_du = u_t / speed
    elif r_t_norm > tie_tol:
        dspeed_du = -r_t / r_t_norm
    else:
        dspeed_du = 0.0

    drnorm_dr = r_t / r_t_norm if r_t_norm > tie_tol else 0.0
    dW_du = dW_dspeed * dspeed_du
    dW_dr_t = -dW_dgap * drnorm_dr
    dW_dr_n = (dW_dmulambda + dW_dgap) * float(mu)

    df_du[1, 1] = 1.0 + r_t * dW_du
    df_dr[1, 1] = W + r_t * dW_dr_t
    df_dr[1, 0] = r_t * dW_dr_n
    return f_blk, df_dgap, df_du, df_dr


def _contact_block_residual_and_jac(
    gap,
    u_blk,
    r_blk,
    mu,
    normal_ncp_type,
    friction_ncp_type,
    normal_scale,
    friction_scale,
    friction_law="compliance",
    *,
    tie_tol=1.0e-14,
):
    r"""Compute one NCP contact block residual and quasi-Newton Jacobian."""
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)
    d = r_blk.size
    m = d - 1
    if d == 2:
        return _contact_block_residual_and_jac_2d(
            gap,
            u_blk,
            r_blk,
            mu,
            normal_ncp_type,
            friction_ncp_type,
            normal_scale,
            friction_scale,
            friction_law,
            tie_tol=tie_tol,
        )

    f_blk = np.zeros(d, dtype=float)
    df_dgap = np.zeros(d, dtype=float)
    df_du = np.zeros((d, d), dtype=float)
    df_dr = np.zeros((d, d), dtype=float)

    phi_n, dphi_dgap, dphi_drn = _normal_ncp_residual_and_jac(
        gap, r_blk[0], normal_ncp_type, normal_scale, tie_tol=tie_tol
    )
    f_blk[0] = phi_n
    df_dgap[0] = dphi_dgap
    df_dr[0, 0] = dphi_drn

    if m == 0:
        return f_blk, df_dgap, df_du, df_dr

    mu_lambda_n = float(mu * r_blk[0])
    if mu_lambda_n <= tie_tol:
        # With no admissible cone radius the total tangential traction must
        # vanish; this is the direct analogue of AC's zero-radius branch.
        f_blk[1:] = r_blk[1:]
        df_dr[1:, 1:] = np.eye(m)
        return f_blk, df_dgap, df_du, df_dr

    if friction_law == "natural_map":
        y_vec = r_blk[1:] - float(friction_scale) * u_blk[1:]
        proj, dproj_ddelta, dproj_dy = _project_ball_and_jac(
            y_vec, mu_lambda_n, tie_tol=tie_tol
        )
        f_blk[1:] = proj - r_blk[1:]
        df_du[1:, 1:] = -float(friction_scale) * dproj_dy
        df_dr[1:, 1:] = dproj_dy - np.eye(m)
        df_dr[1:, 0] = float(mu) * dproj_ddelta
        return f_blk, df_dgap, df_du, df_dr

    u_t = u_blk[1:]
    r_t = r_blk[1:]
    speed = float(np.linalg.norm(u_t))
    r_t_norm = float(np.linalg.norm(r_t))
    cone_gap = mu_lambda_n - r_t_norm
    W, dW_dspeed, dW_dgap, dW_dmulambda = _friction_compliance_and_jac(
        speed, cone_gap, mu_lambda_n, friction_ncp_type, friction_scale, tie_tol=tie_tol
    )

    f_blk[1:] = u_t + W * r_t
    if speed > tie_tol:
        dspeed_du = u_t / speed
    elif r_t_norm > tie_tol:
        # At zero slip-rate the norm direction is not unique.  For scalar
        # / low-dimensional friction blocks a robust semismooth choice is to
        # follow the impending slip direction, i.e. opposite the current
        # tangential traction.
        dspeed_du = -r_t / r_t_norm
    else:
        dspeed_du = np.zeros(m, dtype=float)

    if r_t_norm > tie_tol:
        drnorm_dr = r_t / r_t_norm
    else:
        drnorm_dr = np.zeros(m, dtype=float)

    dW_du = dW_dspeed * dspeed_du
    dW_dr_t = -dW_dgap * drnorm_dr
    dW_dr_n = (dW_dmulambda + dW_dgap) * float(mu)

    df_du[1:, 1:] = np.eye(m) + np.outer(r_t, dW_du)
    df_dr[1:, 1:] = W * np.eye(m) + np.outer(r_t, dW_dr_t)
    df_dr[1:, 0] = r_t * dW_dr_n

    return f_blk, df_dgap, df_du, df_dr


def build_ncp_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    theta=1.0,
    coupling_theta=None,
    component_slices=None,
    C_extract=None,
    D_extract=None,
    rate_form=False,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    reaction_units="impulse",
    offset_coupling_mode="constitutive_shift",
    get_s0=None,
    get_w0=None,
    get_ds0_dz=None,
    get_dw0_dz=None,
    get_s0_load=None,
    get_w0_load=None,
    get_ds0_load_dz=None,
    get_dw0_load_dz=None,
    get_s0_ref=None,
    get_w0_ref=None,
    ncp_type="fischer_burmeister",
    normal_ncp_type=None,
    friction_ncp_type=None,
    normal_r=1.0,
    friction_r=1.0,
    inactive_handling="hard_zero",
    friction_law="compliance",
    smooth_rhs_is_affine=False,
):
    r"""Build a full-state NCP contact system.

    The augmented unknown is ``[z, r]`` where ``z`` is the physical state and
    ``r`` the reaction block. The physical rows match
    :func:`build_alart_curnier_contact`; the reaction rows use:

    * a minimum-map / Fischer-Burmeister normal NCP on ``gap(z)`` and ``r_N``
    * the fixed-point friction compliance law from Macklin et al. (2019)

    Parameters largely mirror :func:`solve_nivp.alart_curnier_contact.
    build_alart_curnier_contact`. The ``*_offset*`` arguments follow the same
    semantics and are kept so the backend can be dropped into the existing
    contact examples.

    ``ncp_type`` sets both the normal and friction NCP choices unless the
    more specific ``normal_ncp_type`` / ``friction_ncp_type`` overrides are
    supplied.

    ``normal_r`` and ``friction_r`` are the positive NCP scaling /
    preconditioning factors from the paper; each may be a scalar, length-
    ``n_contacts`` array, or state-dependent callable returning either.
    Additionally, passing the string ``"auto"`` selects the Macklin et al.
    (2019) per-contact complementarity preconditioner

    .. math::

        r_i = h^2 [B^T A^{-1} B]_{ii}

    approximated by the Jacobi diagonal ``r_i = h^2 \\sum_j B_{ji}^2 /
    A_{jj}``, where ``B`` is the reaction coupling matrix and ``A`` the
    physical system matrix.  This gives a physically-motivated, step-size-
    adaptive preconditioner that removes the dimensional mismatch between
    the gap (length) and reaction (force/impulse) residual rows.
    Requires ``B`` to be non-trivial (same condition as ``reaction_units='impulse'``).

    ``inactive_handling`` controls what happens when ``gap > gap_tol``:

    * ``"hard_zero"`` keeps the legacy benchmark behavior and enforces
      ``r = 0`` outside the active set.
    * ``"ncp"`` evaluates the full NCP residual for all gaps so that the
      complementarity law itself decides whether the contact opens or closes.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size

    if coupling_theta is None:
        coupling_theta = theta
    coupling_theta = float(coupling_theta)
    gap_tol = float(gap_tol)
    reaction_units = str(reaction_units).strip().lower()
    if reaction_units not in {"impulse", "force"}:
        raise ValueError(
            "reaction_units must be either 'impulse' or 'force' "
            f"(got {reaction_units!r})"
        )
    offset_coupling_mode = str(offset_coupling_mode).strip().lower()
    if offset_coupling_mode not in {
        "constitutive_shift",
        "incremental_reference",
        "total_traction",
        "constitutive_shift_with_load",
    }:
        raise ValueError(
            "offset_coupling_mode must be one of 'constitutive_shift', "
            "'incremental_reference', 'total_traction', or "
            f"'constitutive_shift_with_load' (got {offset_coupling_mode!r})"
        )
    ncp_type = _normalize_ncp_name(ncp_type, label="ncp_type")
    normal_ncp_type = _normalize_ncp_name(
        ncp_type if normal_ncp_type is None else normal_ncp_type,
        label="normal_ncp_type",
    )
    friction_ncp_type = _normalize_ncp_name(
        ncp_type if friction_ncp_type is None else friction_ncp_type,
        label="friction_ncp_type",
    )
    friction_law = str(friction_law).strip().lower().replace("-", "_")
    if friction_law not in {"compliance", "natural_map"}:
        raise ValueError(
            "friction_law must be 'compliance' or 'natural_map' "
            f"(got {friction_law!r})"
        )
    inactive_handling = str(inactive_handling).strip().lower().replace("-", "_")
    if inactive_handling not in {"hard_zero", "ncp"}:
        raise ValueError(
            "inactive_handling must be 'hard_zero' or 'ncp' "
            f"(got {inactive_handling!r})"
        )
    smooth_rhs_is_affine = bool(smooth_rhs_is_affine)
    use_hard_zero_gate = inactive_handling == "hard_zero"
    _incremental_offset_loading = offset_coupling_mode == "incremental_reference"
    _total_offset_loading = offset_coupling_mode == "total_traction"
    _split_load_offset = offset_coupling_mode == "constitutive_shift_with_load"

    s0_nargs = _count_required_args(get_s0)
    w0_nargs = _count_required_args(get_w0)
    ds0_nargs = _count_required_args(get_ds0_dz)
    dw0_nargs = _count_required_args(get_dw0_dz)
    load_get_s0 = get_s0_load if get_s0_load is not None else (get_s0 if _split_load_offset else get_s0)
    load_get_w0 = get_w0_load if get_w0_load is not None else (get_w0 if _split_load_offset else get_w0)
    load_get_ds0_dz = (
        get_ds0_load_dz if get_ds0_load_dz is not None else (get_ds0_dz if _split_load_offset else get_ds0_dz)
    )
    load_get_dw0_dz = (
        get_dw0_load_dz if get_dw0_load_dz is not None else (get_dw0_dz if _split_load_offset else get_dw0_dz)
    )
    load_s0_nargs = _count_required_args(load_get_s0)
    load_w0_nargs = _count_required_args(load_get_w0)
    load_ds0_nargs = _count_required_args(load_get_ds0_dz)
    load_dw0_nargs = _count_required_args(load_get_dw0_dz)
    ref_s0_nargs = _count_required_args(get_s0_ref)
    ref_w0_nargs = _count_required_args(get_w0_ref)
    # Detect "auto" Macklin preconditioner before scalar/callable checks.
    _use_auto_normal_r = isinstance(normal_r, str) and normal_r.strip().lower() == "auto"
    _use_auto_friction_r = isinstance(friction_r, str) and friction_r.strip().lower() == "auto"
    if _use_auto_normal_r:
        normal_r = 1.0   # placeholder; replaced after B_mat is available
    if _use_auto_friction_r:
        friction_r = 1.0
    normal_r_nargs = _count_required_args(normal_r) if callable(normal_r) else None
    friction_r_nargs = _count_required_args(friction_r) if callable(friction_r) else None

    if C_extract is not None:
        C_extract = _dense_or_sparse(C_extract)
        if D_extract is None:
            D_extract = C_extract
        else:
            D_extract = _dense_or_sparse(D_extract)
        if C_extract.shape[1] != n_phys:
            raise ValueError(
                f"C_extract has {C_extract.shape[1]} columns but n_phys = {n_phys}"
            )
    elif D_extract is not None:
        D_extract = _dense_or_sparse(D_extract)

    if gap_func is None and C_extract is None:
        raise ValueError("gap_func must be provided when C_extract is None")

    norm_contacts = []
    reaction_idx = 0
    reaction_extract_rows = []
    for c in contacts:
        v_n = int(c["vel_normal_idx"])
        v_t = list(np.atleast_1d(c.get("vel_tangential_idx", [])).astype(int))
        mu_val = c.get("mu", 0.0)
        if callable(mu_val):
            get_mu = mu_val
        else:
            mu_const = float(mu_val)

            def get_mu(y, t=None, Fk_val=None, _m=mu_const):  # noqa: E306
                return _m

        r_n_loc = reaction_idx
        r_t_loc = [reaction_idx + 1 + j for j in range(len(v_t))]
        block_rows = [v_n] + v_t
        reaction_extract_rows.extend(block_rows)
        norm_contacts.append(
            {
                "vN": v_n,
                "vT": v_t,
                "rN_loc": r_n_loc,
                "rT_loc": r_t_loc,
                "block_slice": slice(reaction_idx, reaction_idx + 1 + len(v_t)),
                "get_mu": get_mu,
                "mu_nargs": _count_required_args(get_mu),
            }
        )
        reaction_idx += 1 + len(v_t)

    n_react = reaction_idx
    n_aug = n_phys + n_react
    n_blocks = len(norm_contacts)

    if B is None and C_extract is not None:
        if sp.issparse(C_extract):
            B_mat = C_extract[reaction_extract_rows, :].T.tocsr()
        else:
            B_mat = np.asarray(C_extract[reaction_extract_rows, :].T, dtype=float)
    elif B is None:
        B_mat = np.zeros((n_phys, n_react), dtype=float)
        col = 0
        for ci in norm_contacts:
            B_mat[ci["vN"], col] = 1.0
            col += 1
            for vt in ci["vT"]:
                B_mat[vt, col] = 1.0
                col += 1
    else:
        B_mat = _dense_or_sparse(B)
        if B_mat.shape != (n_phys, n_react):
            raise ValueError(
                f"B shape {B_mat.shape} doesn't match (n_phys={n_phys}, n_react={n_react})"
            )

    # -----------------------------------------------------------------
    # Macklin complementarity preconditioner — r_i = h² [D^T A^{-1} D]_ii
    # Jacobi approximation: r_i = h² Σ_j (D_ji² / A_jj)
    # Uses the kinematic extraction D (not B_mat) so the Delassus
    # diagonal is independent of interface weighting in B.
    # -----------------------------------------------------------------
    _auto_normal_r_base = None
    _auto_friction_r_base = None
    _h_cell_auto = [1.0]   # mutable h carrier, updated in rhs_aug / jac_aug
    _auto_rho_deferred = _use_auto_normal_r or _use_auto_friction_r

    if sp.issparse(A):
        A_aug = sp.block_diag([A, sp.csr_matrix((n_react, n_react))], format="csr")
    else:
        A_aug = np.zeros((n_aug, n_aug), dtype=float)
        A_aug[:n_phys, :n_phys] = np.asarray(A, dtype=float)

    y0_aug = np.zeros(n_aug, dtype=float)
    y0_aug[:n_phys] = y0

    if gap_func is not None:
        def gap_aug(y, t):
            return np.atleast_1d(gap_func(y[:n_phys], t))
    else:
        normal_rows = [ci["vN"] for ci in norm_contacts]

        def gap_aug(y, t):
            vals = C_extract @ y[:n_phys]
            vals = np.asarray(vals).ravel()
            return vals[normal_rows]

    if gap_func is None and C_extract is not None:
        normal_rows = [ci["vN"] for ci in norm_contacts]
        if sp.issparse(C_extract):
            gap_jac_const = C_extract[normal_rows, :].tocsr()
        else:
            gap_jac_const = np.asarray(C_extract[normal_rows, :], dtype=float)
    else:
        gap_jac_const = None

    if D_extract is not None:
        if sp.issparse(D_extract):
            U_contact = D_extract[reaction_extract_rows, :].tocsr()
        else:
            U_contact = np.asarray(D_extract[reaction_extract_rows, :], dtype=float)
        vel_indices = None
    else:
        vel_indices = np.asarray(reaction_extract_rows, dtype=int)
        U_contact = None

    # ----- Deferred Macklin auto-rho (uses kinematic D, not B_mat) -----
    if _auto_rho_deferred:
        if sp.issparse(A):
            _A_diag_phys = np.abs(np.asarray(A.diagonal()).ravel())
        else:
            _A_diag_phys = np.abs(np.diag(np.asarray(A, dtype=float)))
        _pos = _A_diag_phys > 0
        _A_diag_phys = np.where(_pos, _A_diag_phys,
                                (_A_diag_phys[_pos].min() if _pos.any() else 1.0))

        if U_contact is not None:
            _D_T = U_contact.T
            if sp.issparse(_D_T):
                _D_T_dense = _D_T.toarray()
            else:
                _D_T_dense = np.asarray(_D_T, dtype=float)
        else:
            _D_T_dense = np.zeros((n_phys, n_react), dtype=float)
            col = 0
            for ci in norm_contacts:
                _D_T_dense[ci["vN"], col] = 1.0
                col += 1
                for vt in ci["vT"]:
                    _D_T_dense[vt, col] = 1.0
                    col += 1

        _r_base_n = np.zeros(n_blocks, dtype=float)
        _r_base_f = np.zeros(n_blocks, dtype=float)
        for _k, _ci in enumerate(norm_contacts):
            _sl = _ci["block_slice"]
            _b_n = _D_T_dense[:, _sl.start]
            _r_base_n[_k] = float(np.sum(_b_n ** 2 / _A_diag_phys))
            _d = _sl.stop - _sl.start
            if _d > 1:
                _rf = 0.0
                for _col in range(_sl.start + 1, _sl.stop):
                    _b_t = _D_T_dense[:, _col]
                    _rf += float(np.sum(_b_t ** 2 / _A_diag_phys))
                _r_base_f[_k] = _rf / (_d - 1)
            else:
                _r_base_f[_k] = _r_base_n[_k]

        _r_base_n = np.where(_r_base_n > 0, _r_base_n, 1.0)
        _r_base_f = np.where(_r_base_f > 0, _r_base_f, 1.0)
        _auto_normal_r_base = _r_base_n
        _auto_friction_r_base = _r_base_f
        del _D_T_dense

    alg_proj = None
    q_slices = []
    if constraints is not None:
        alg_proj = AlgebraicConstraintProjection(constraints=constraints)
        q_slices = list(alg_proj.constraint_q_slices)

    if component_slices is not None:
        cs_aug = []
        any_array = False
        for cs_item in component_slices:
            if isinstance(cs_item, slice):
                cs_aug.append(cs_item)
            else:
                cs_aug.append(np.asarray(cs_item, dtype=int))
                any_array = True
        cs_aug.append(np.arange(n_phys, n_aug, dtype=int) if any_array else slice(n_phys, n_aug))
    else:
        vel_set = set(reaction_extract_rows)
        vel_idx = np.array(sorted(vel_set), dtype=int)
        other_idx = np.array(sorted(set(range(n_phys)) - vel_set), dtype=int)
        react_idx = np.arange(n_phys, n_aug, dtype=int)
        cs_aug = []
        if vel_idx.size > 0:
            cs_aug.append(vel_idx)
        if other_idx.size > 0:
            cs_aug.append(other_idx)
        cs_aug.append(react_idx)

    proj = IdentityProjection(component_slices=cs_aug)

    def _assemble_offset_vector_from(get_s0_fun, s0_nargs_fun, get_w0_fun, w0_nargs_fun, y_full, *, t=None, Fk_val=None):
        offset_vec = np.zeros(n_react, dtype=float)
        if get_s0_fun is None and get_w0_fun is None:
            return offset_vec

        s0_arr = _eval_s0(get_s0_fun, s0_nargs_fun, n_blocks, y_full, t=t, Fk_val=Fk_val)
        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            offset_vec[sl.start] = float(s0_arr[k])
            m_k = sl.stop - sl.start - 1
            if m_k > 0:
                offset_vec[sl.start + 1 : sl.stop] = _eval_w0(
                    get_w0_fun, w0_nargs_fun, y_full, k, m_k, t=t, Fk_val=Fk_val
                )
        return offset_vec

    def _assemble_offset_jac_from(get_ds0_fun, ds0_nargs_fun, get_dw0_fun, dw0_nargs_fun, y_full, *, t=None, Fk_val=None):
        if get_ds0_fun is None and get_dw0_fun is None:
            return None

        jac = np.zeros((n_react, n_aug), dtype=float)
        ds0_all = _eval_ds0_dz(
            get_ds0_fun, ds0_nargs_fun, n_blocks, n_aug, y_full, t=t, Fk_val=Fk_val
        )
        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            jac[sl.start, :] = 0.0 if ds0_all is None else ds0_all[k, :]
            m_k = sl.stop - sl.start - 1
            if m_k > 0 and get_dw0_fun is not None:
                jac[sl.start + 1 : sl.stop, :] = _eval_dw0_dz(
                    get_dw0_fun, dw0_nargs_fun, y_full, k, m_k, n_aug, t=t, Fk_val=Fk_val
                )
        return jac

    def _assemble_offset_vector(y_full, *, t=None, Fk_val=None):
        return _assemble_offset_vector_from(get_s0, s0_nargs, get_w0, w0_nargs, y_full, t=t, Fk_val=Fk_val)

    def _assemble_load_offset_vector(y_full, *, t=None, Fk_val=None):
        return _assemble_offset_vector_from(
            load_get_s0,
            load_s0_nargs,
            load_get_w0,
            load_w0_nargs,
            y_full,
            t=t,
            Fk_val=Fk_val,
        )

    def _assemble_offset_jac(y_full, *, t=None, Fk_val=None):
        return _assemble_offset_jac_from(
            get_ds0_dz,
            ds0_nargs,
            get_dw0_dz,
            dw0_nargs,
            y_full,
            t=t,
            Fk_val=Fk_val,
        )

    def _assemble_load_offset_jac(y_full, *, t=None, Fk_val=None):
        return _assemble_offset_jac_from(
            load_get_ds0_dz,
            load_ds0_nargs,
            load_get_dw0_dz,
            load_dw0_nargs,
            y_full,
            t=t,
            Fk_val=Fk_val,
        )

    if get_s0_ref is not None or get_w0_ref is not None:
        _offset_ref_vec = _assemble_offset_vector_from(
            get_s0_ref,
            ref_s0_nargs,
            get_w0_ref,
            ref_w0_nargs,
            y0_aug,
            t=0.0,
            Fk_val=None,
        )
    else:
        _offset_ref_vec = _assemble_load_offset_vector(y0_aug, t=0.0, Fk_val=None)

    _B_csr = sp.csr_matrix(B_mat) if not sp.issparse(B_mat) else B_mat.tocsr()
    _U_contact_csr = (
        (sp.csr_matrix(U_contact) if not sp.issparse(U_contact) else U_contact.tocsr())
        if U_contact is not None
        else None
    )
    _gap_jac_csr = (
        (sp.csr_matrix(gap_jac_const) if not sp.issparse(gap_jac_const) else gap_jac_const.tocsr())
        if gap_jac_const is not None
        else None
    )
    _jac_top_left = [None]
    _jac_top_right_key = [None]
    _jac_top_right = [None]
    _rhs_b_const = [None]
    _rhs_neg_A = [None]

    def _reaction_scale(h_val):
        if reaction_units == "force":
            return 1.0
        if h_val is None or h_val <= 0.0:
            return None
        return 1.0 / (coupling_theta * h_val)

    def _reaction_cache_key(h_val):
        if reaction_units == "force":
            return ("force", coupling_theta)
        return ("impulse", float(h_val))

    def _contact_velocity(yp, prev_state, h_val):
        if U_contact is not None:
            if rate_form:
                if prev_state is not None and h_val is not None and h_val > 0.0:
                    diff = (yp - prev_state[:n_phys]) / h_val
                    vals = U_contact @ diff
                    return np.asarray(vals).ravel()
                return np.zeros(n_react, dtype=float)
            vals = U_contact @ yp
            return np.asarray(vals).ravel()

        if rate_form:
            if prev_state is not None and h_val is not None and h_val > 0.0:
                return (yp[vel_indices] - prev_state[:n_phys][vel_indices]) / h_val
            return np.zeros(n_react, dtype=float)
        return yp[vel_indices]

    def _gap_jacobian(t, yp):
        if _gap_jac_csr is not None:
            return _gap_jac_csr

        g0 = np.atleast_1d(gap_aug(np.concatenate([yp, np.zeros(n_react)]), t))
        eps_base = 1.0e-7
        h_vec = eps_base * np.maximum(np.abs(yp), 1.0)
        J = np.empty((n_blocks, n_phys), dtype=float)
        for j in range(n_phys):
            yp_pert = yp.copy()
            yp_pert[j] += h_vec[j]
            gp = np.atleast_1d(gap_aug(np.concatenate([yp_pert, np.zeros(n_react)]), t))
            J[:, j] = (gp - g0) / h_vec[j]
        return J

    def rhs_aug(t, y, *extra):
        prev_state, Fk_val, h_val = _parse_prev_and_h(extra, y.shape)
        yp = y[:n_phys]
        r = y[n_phys:]
        out = np.zeros(n_aug, dtype=float)
        offset_vec = _assemble_offset_vector(y, t=t, Fk_val=Fk_val)
        load_offset_vec = _assemble_load_offset_vector(y, t=t, Fk_val=Fk_val)

        if smooth_rhs_is_affine and rhs_jac is not None and _rhs_neg_A[0] is not None:
            out[:n_phys] = np.asarray(_rhs_neg_A[0] @ yp).ravel() + _rhs_b_const[0]
        elif smooth_rhs_is_affine and rhs_jac is not None:
            J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
            J_s = _dense_or_sparse(J_s)
            _rhs_neg_A[0] = sp.csr_matrix(J_s) if not sp.issparse(J_s) else J_s.tocsr()
            _rhs_b_const[0] = np.asarray(
                _call_with_time_state_fk(rhs_smooth, t, np.zeros(n_phys), None)
            ).ravel()
            out[:n_phys] = np.asarray(_rhs_neg_A[0] @ yp).ravel() + _rhs_b_const[0]
        else:
            out[:n_phys] = _call_with_time_state_fk(rhs_smooth, t, yp, None)

        reaction_scale = _reaction_scale(h_val)
        if reaction_scale is not None:
            out[:n_phys] += reaction_scale * np.asarray(_B_csr @ r).ravel()
            if _incremental_offset_loading:
                out[:n_phys] += reaction_scale * np.asarray(
                    _B_csr @ (load_offset_vec - _offset_ref_vec)
                ).ravel()
            elif _total_offset_loading or _split_load_offset:
                out[:n_phys] += reaction_scale * np.asarray(_B_csr @ load_offset_vec).ravel()

        if alg_proj is not None:
            c_res = alg_proj.constraint_residual(
                yp,
                t=t,
                Fk_val=None,
                step_size=h_val,
                prev_state=(prev_state[:n_phys] if prev_state is not None else None),
            )
            for qs in q_slices:
                out[qs] = -c_res[qs]

        u_rel = _contact_velocity(yp, prev_state, h_val)
        gaps = np.atleast_1d(gap_aug(y, t))
        mu_arr = _vectorize_mu(norm_contacts, yp, t=t, Fk_val=Fk_val)
        if _use_auto_normal_r or _use_auto_friction_r:
            _h_cell_auto[0] = h_val if (h_val is not None and h_val > 0.0) else 1.0
        if _use_auto_normal_r:
            # Gap-level preconditioner: normal_r = h² * schur_base (force units)
            # or h * schur_base (impulse units, since r_imp = h * r_force absorbs one h).
            _h_eff = _h_cell_auto[0]
            _h_scale = _h_eff ** 2 if reaction_units == "force" else _h_eff
            normal_r_arr = _auto_normal_r_base * _h_scale
        else:
            normal_r_arr = _eval_contact_scalar_field(normal_r, normal_r_nargs, n_blocks, "normal_r", y, t=t, Fk_val=Fk_val)
        if _use_auto_friction_r:
            # Velocity-level preconditioner: friction_r = h * schur_base (force units)
            # or 1 * schur_base (impulse units, one h less than gap-level).
            _h_eff = _h_cell_auto[0]
            _h_scale = _h_eff if reaction_units == "force" else 1.0
            friction_r_arr = _auto_friction_r_base * _h_scale
        else:
            friction_r_arr = _eval_contact_scalar_field(friction_r, friction_r_nargs, n_blocks, "friction_r", y, t=t, Fk_val=Fk_val)

        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            blk_rows = slice(sl.start, sl.stop)
            r_blk = r[blk_rows]
            offset_blk = offset_vec[blk_rows]
            active = bool(gaps[k] <= gap_tol)
            if use_hard_zero_gate and not active:
                f_blk = r_blk.copy()
            else:
                gap_k = float(gaps[k] - gap_tol)
                u_blk = u_rel[blk_rows]
                r_eff = r_blk.copy() + offset_blk
                f_blk, _, _, _ = _contact_block_residual_and_jac(
                    gap_k,
                    u_blk,
                    r_eff,
                    mu_arr[k],
                    normal_ncp_type,
                    friction_ncp_type,
                    normal_r_arr[k],
                    friction_r_arr[k],
                    friction_law,
                )
            out[n_phys + sl.start : n_phys + sl.stop] = -f_blk

        return out

    def _fd_smooth_jac(t, yp):
        f0 = _call_with_time_state_fk(rhs_smooth, t, yp, None)
        eps_base = 1.0e-7
        h_vec = eps_base * np.maximum(np.abs(yp), 1.0)
        J = np.empty((n_phys, n_phys), dtype=float)
        for j in range(n_phys):
            yp_pert = yp.copy()
            yp_pert[j] += h_vec[j]
            fp = _call_with_time_state_fk(rhs_smooth, t, yp_pert, None)
            J[:, j] = (fp - f0) / h_vec[j]
        return J

    def jac_aug(t, y, *extra):
        prev_state, Fk_val, h_val = _parse_prev_and_h(extra, y.shape)
        if reaction_units == "impulse" and (h_val is None or h_val <= 0.0):
            h_val = 1.0
        yp = y[:n_phys]
        r = y[n_phys:]
        offset_vec = _assemble_offset_vector(y, t=t, Fk_val=Fk_val)
        offset_jac = _assemble_offset_jac(y, t=t, Fk_val=Fk_val)
        load_offset_jac = _assemble_load_offset_jac(y, t=t, Fk_val=Fk_val)

        if _jac_top_left[0] is None or not smooth_rhs_is_affine:
            if rhs_jac is not None:
                J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
            else:
                J_s = _fd_smooth_jac(t, yp)
            J_s = _dense_or_sparse(J_s)
            if not sp.issparse(J_s):
                J_s = sp.csr_matrix(J_s)
            else:
                J_s = J_s.tocsr()

            if alg_proj is not None:
                patch = alg_proj.build_constraint_patch(
                    yp,
                    n_phys,
                    t=t,
                    Fk_val=None,
                    step_size=h_val,
                    prev_state=(prev_state[:n_phys] if prev_state is not None else None),
                ).tocsr()
                J_s = J_s.tolil()
                for qs in q_slices:
                    J_s[qs, :] = (-patch[qs, :]).tolil()
                J_s = J_s.tocsr()

            if smooth_rhs_is_affine:
                _jac_top_left[0] = J_s

        top_left = _jac_top_left[0] if smooth_rhs_is_affine else J_s

        cache_key = _reaction_cache_key(h_val)
        if _jac_top_right_key[0] != cache_key:
            reaction_scale = _reaction_scale(h_val)
            if reaction_scale is None:
                reaction_scale = 0.0
            B_coup = reaction_scale * _B_csr
            if alg_proj is not None:
                B_coup = B_coup.tolil()
                for qs in q_slices:
                    B_coup[qs, :] = 0.0
                B_coup = B_coup.tocsr()
            else:
                B_coup = B_coup.tocsr()
            _jac_top_right[0] = B_coup
            _jac_top_right_key[0] = cache_key

        top_right = _jac_top_right[0]

        if (_incremental_offset_loading or _total_offset_loading or _split_load_offset) and load_offset_jac is not None:
            reaction_scale = _reaction_scale(h_val)
            if reaction_scale is None:
                reaction_scale = 0.0
            phys_corr = reaction_scale * (_B_csr @ load_offset_jac)
            phys_corr = np.asarray(phys_corr, dtype=float)
            if alg_proj is not None:
                phys_corr[q_slices, :] = 0.0
            top_left = top_left + sp.csr_matrix(phys_corr[:, :n_phys])
            top_right = top_right + sp.csr_matrix(phys_corr[:, n_phys:])

        u_rel = _contact_velocity(yp, prev_state, h_val)
        gaps = np.atleast_1d(gap_aug(y, t))
        gap_jac = _gap_jacobian(t, yp)
        if sp.issparse(gap_jac):
            gap_jac_dense = gap_jac.toarray()
        else:
            gap_jac_dense = np.asarray(gap_jac, dtype=float)
        mu_arr = _vectorize_mu(norm_contacts, yp, t=t, Fk_val=Fk_val)
        if _use_auto_normal_r or _use_auto_friction_r:
            _h_cell_auto[0] = h_val if (h_val is not None and h_val > 0.0) else 1.0
        if _use_auto_normal_r:
            _h_eff = _h_cell_auto[0]
            _h_scale = _h_eff ** 2 if reaction_units == "force" else _h_eff
            normal_r_arr = _auto_normal_r_base * _h_scale
        else:
            normal_r_arr = _eval_contact_scalar_field(normal_r, normal_r_nargs, n_blocks, "normal_r", y, t=t, Fk_val=Fk_val)
        if _use_auto_friction_r:
            # Velocity-level preconditioner: friction_r = h * schur_base (force units)
            # or 1 * schur_base (impulse units, one h less than gap-level).
            _h_eff = _h_cell_auto[0]
            _h_scale = _h_eff if reaction_units == "force" else 1.0
            friction_r_arr = _auto_friction_r_base * _h_scale
        else:
            friction_r_arr = _eval_contact_scalar_field(friction_r, friction_r_nargs, n_blocks, "friction_r", y, t=t, Fk_val=Fk_val)

        U_scaled = None
        if _U_contact_csr is not None:
            if rate_form:
                U_scaled = None if (h_val is None or h_val <= 0.0) else (_U_contact_csr / h_val)
            else:
                U_scaled = _U_contact_csr

        bl_parts = []
        br_dense = np.zeros((n_react, n_react), dtype=float)

        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            d = sl.stop - sl.start
            active = bool(gaps[k] <= gap_tol)
            offset_blk = offset_vec[sl.start:sl.stop]
            dz0_blk = None if offset_jac is None else offset_jac[sl.start:sl.stop, :]

            if use_hard_zero_gate and not active:
                block_right = np.zeros((d, n_react), dtype=float)
                block_right[:, sl.start:sl.stop] = -np.eye(d)
                bl_dense = np.zeros((d, n_phys), dtype=float)
                br_dense[sl.start:sl.stop, :] = block_right
                bl_parts.append(sp.csr_matrix(bl_dense))
                continue

            gap_k = float(gaps[k] - gap_tol)
            u_blk = u_rel[sl.start:sl.stop]
            r_blk = r[sl.start:sl.stop]
            r_eff = r_blk.copy() + offset_blk
            _, df_dgap, df_du, df_dr = _contact_block_residual_and_jac(
                gap_k,
                u_blk,
                r_eff,
                mu_arr[k],
                normal_ncp_type,
                friction_ncp_type,
                normal_r_arr[k],
                friction_r_arr[k],
                friction_law,
            )

            bl_dense = np.zeros((d, n_phys), dtype=float)
            bl_dense[0, :] += -df_dgap[0] * gap_jac_dense[k, :]

            if U_scaled is not None:
                U_blk = U_scaled[sl.start:sl.stop, :]
                bl_dense += -(df_du @ U_blk.toarray())
            elif vel_indices is not None:
                scale = (0.0 if (rate_form and (h_val is None or h_val <= 0.0))
                         else (1.0 / h_val if rate_form else 1.0))
                for li in range(d):
                    for lj in range(d):
                        bl_dense[li, vel_indices[sl.start + lj]] += -df_du[li, lj] * scale

            block_right = np.zeros((d, n_react), dtype=float)
            block_right[:, sl.start:sl.stop] = -df_dr
            if dz0_blk is not None:
                bl_dense += -(df_dr @ dz0_blk[:, :n_phys])
                block_right += -(df_dr @ dz0_blk[:, n_phys:])
            br_dense[sl.start:sl.stop, :] = block_right
            bl_parts.append(sp.csr_matrix(bl_dense))

        bottom_left = sp.vstack(bl_parts, format="csr")
        bottom_right = sp.csr_matrix(br_dense)

        return sp.bmat(
            [[top_left, top_right], [bottom_left, bottom_right]],
            format="csr",
        )

    return ContactSystem(
        A=A_aug,
        rhs=rhs_aug,
        y0=y0_aug,
        projection=proj,
        component_slices=cs_aug,
        integrator_opts={"pass_prev_state": True, "pass_step_size": True},
        n_phys=n_phys,
        B=B_mat,
        rhs_jac=jac_aug,
    )


def build_dynamic_ncp_contact(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    component_slices=None,
    gap_extract=None,
    vel_extract=None,
    constraints=None,
    rhs_jac=None,
    gap_tol=0.0,
    offset_coupling_mode="constitutive_shift",
    get_s0=None,
    get_w0=None,
    get_ds0_dz=None,
    get_dw0_dz=None,
    get_s0_load=None,
    get_w0_load=None,
    get_ds0_load_dz=None,
    get_dw0_load_dz=None,
    get_s0_ref=None,
    get_w0_ref=None,
    ncp_type="fischer_burmeister",
    normal_ncp_type=None,
    friction_ncp_type=None,
    normal_r=1.0,
    friction_r=1.0,
    inactive_handling="hard_zero",
    friction_law="compliance",
    smooth_rhs_is_affine=False,
):
    r"""Build an explicit-velocity dynamic NCP contact system."""
    return build_ncp_contact(
        A=A,
        rhs_smooth=rhs_smooth,
        y0=y0,
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        component_slices=component_slices,
        C_extract=gap_extract,
        D_extract=vel_extract,
        rate_form=False,
        constraints=constraints,
        rhs_jac=rhs_jac,
        gap_tol=gap_tol,
        reaction_units="force",
        offset_coupling_mode=offset_coupling_mode,
        get_s0=get_s0,
        get_w0=get_w0,
        get_ds0_dz=get_ds0_dz,
        get_dw0_dz=get_dw0_dz,
        get_s0_load=get_s0_load,
        get_w0_load=get_w0_load,
        get_ds0_load_dz=get_ds0_load_dz,
        get_dw0_load_dz=get_dw0_load_dz,
        get_s0_ref=get_s0_ref,
        get_w0_ref=get_w0_ref,
        ncp_type=ncp_type,
        normal_ncp_type=normal_ncp_type,
        friction_ncp_type=friction_ncp_type,
        normal_r=normal_r,
        friction_r=friction_r,
        inactive_handling=inactive_handling,
        friction_law=friction_law,
        smooth_rhs_is_affine=smooth_rhs_is_affine,
    )


class NCPBlockSystem:
    """Block-structured wrapper around NCP contact for Schur-complement solve.

    Exposes the H / J / C / residual decomposition that
    ``SchurComplementSolver`` consumes.  Internally delegates to
    ``build_ncp_contact`` for gap evaluation, NCP function computation,
    and Jacobian assembly, then extracts the 2x2 block structure.

    Parameters
    ----------
    cs : ContactSystem
        Augmented NCP contact system from ``build_ncp_contact``.
    A_phys : ndarray or sparse, (n_phys, n_phys)
        Physical mass / descriptor matrix (before augmentation).
    rhs_jac_func : callable or None
        Smooth-physics Jacobian (unused here; kept for forward compat).
    """

    def __init__(self, cs, A_phys, rhs_jac_func=None):
        self._cs = cs
        self._A_phys = A_phys
        self._rhs_jac = rhs_jac_func
        self.n_phys = int(cs.n_phys)
        self.n_react = len(cs.y0) - self.n_phys
        self._jac_aug = cs.rhs_jac

    def assemble_blocks(self, y, t, h, y_prev):
        """Return the four independent blocks of the Newton Jacobian.

        The Newton system for the augmented implicit equation is::

            [H,      B_top] [Δu]   [g]
            [B_bot,    C  ] [Δλ] = [h_c]

        where ``B_top`` and ``B_bot`` are **not** transposes of each other
        because the NCP normal constraint is position-level (gap) while the
        reaction force acts on the velocity DOFs.

        Returns
        -------
        dict
            Keys ``H``, ``B_top``, ``B_bot``, ``C``, ``g``, ``h_c``,
            ``precond_diag``.
        """
        n_p = self.n_phys
        n_r = self.n_react

        jac_full = self._jac_aug(t, y, y_prev, None, h)
        if sp.issparse(jac_full):
            jac_full = jac_full.toarray()
        jac_full = np.asarray(jac_full, dtype=float)

        A_dense = (self._A_phys.toarray() if sp.issparse(self._A_phys)
                   else np.asarray(self._A_phys, dtype=float))
        A_over_h = A_dense / h

        H = A_over_h - jac_full[:n_p, :n_p]
        B_top = -jac_full[:n_p, n_p:]
        B_bot = -jac_full[n_p:, :n_p]
        C = -jac_full[n_p:, n_p:]

        rhs_val = self._cs.rhs(t, y, y_prev, None, h)
        F = A_over_h @ (y[:n_p] - y_prev[:n_p]) - rhs_val[:n_p]
        g = -F
        h_c = rhs_val[n_p:]

        H_diag = np.diag(H)
        safe_diag = np.where(np.abs(H_diag) > 1e-30, H_diag, 1.0)
        precond_diag = np.array([
            float(np.sum(B_bot[i, :] ** 2 / safe_diag))
            for i in range(n_r)
        ])
        precond_diag = np.where(precond_diag > 1e-30, precond_diag, 1.0)

        return {
            "H": H,
            "B_top": B_top,
            "B_bot": B_bot,
            "C": C,
            "g": g,
            "h_c": h_c,
            "precond_diag": precond_diag,
        }


def build_ncp_contact_blocked(
    A,
    rhs_smooth,
    y0,
    contacts,
    gap_func=None,
    B=None,
    rhs_jac=None,
    ncp_type="fischer_burmeister",
    normal_ncp_type=None,
    friction_ncp_type=None,
    normal_r=1.0,
    friction_r=1.0,
    gap_tol=0.0,
    friction_law="compliance",
    reaction_units="force",
    **kwargs,
):
    """Build an NCP contact system with block decomposition for Schur solve.

    Parameters mirror ``build_ncp_contact``.  Returns an ``NCPBlockSystem``
    that satisfies the ``BlockStructuredSystem`` protocol.

    Parameters
    ----------
    A : ndarray or sparse, (n_phys, n_phys)
        Physical mass / descriptor matrix.
    rhs_smooth : callable
        Smooth-physics RHS ``f(t, y)``.
    y0 : ndarray, (n_phys,)
        Physical initial condition.
    contacts : list of dict
        Contact descriptors (same format as ``build_ncp_contact``).
    gap_func : callable or None
        Gap evaluation ``gap(y, t) -> (n_contacts,)``.
    B : ndarray or sparse or None
        Coupling matrix.
    rhs_jac : callable or None
        Jacobian of ``rhs_smooth``.
    ncp_type : str
        NCP function type.
    reaction_units : str
        ``"force"`` or ``"impulse"``.
    **kwargs
        Forwarded to ``build_ncp_contact``.

    Returns
    -------
    NCPBlockSystem
    """
    cs = build_ncp_contact(
        A=A,
        rhs_smooth=rhs_smooth,
        y0=y0,
        contacts=contacts,
        gap_func=gap_func,
        B=B,
        rhs_jac=rhs_jac,
        ncp_type=ncp_type,
        normal_ncp_type=normal_ncp_type,
        friction_ncp_type=friction_ncp_type,
        normal_r=normal_r,
        friction_r=friction_r,
        gap_tol=gap_tol,
        friction_law=friction_law,
        reaction_units=reaction_units,
        **kwargs,
    )
    return NCPBlockSystem(cs, A, rhs_jac_func=rhs_jac)
