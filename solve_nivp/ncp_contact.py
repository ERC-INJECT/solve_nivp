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


def _contact_block_residual_and_jac(
    gap,
    u_blk,
    r_blk,
    mu,
    normal_ncp_type,
    friction_ncp_type,
    normal_scale,
    friction_scale,
    *,
    tie_tol=1.0e-14,
):
    r"""Compute one NCP contact block residual and quasi-Newton Jacobian."""
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)
    d = r_blk.size
    m = d - 1

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

    u_t = u_blk[1:]
    r_t = r_blk[1:]
    speed = float(np.linalg.norm(u_t))
    r_t_norm = float(np.linalg.norm(r_t))
    cone_gap = mu_lambda_n - r_t_norm
    W = _friction_compliance(
        speed, cone_gap, mu_lambda_n, friction_ncp_type, friction_scale, tie_tol=tie_tol
    )

    f_blk[1:] = u_t + W * r_t
    df_du[1:, 1:] = np.eye(m)
    df_dr[1:, 1:] = W * np.eye(m)

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
    supplied. ``normal_r`` and ``friction_r`` are the positive NCP scaling /
    preconditioning factors from the paper; each may be a scalar, length-
    ``n_contacts`` array, or state-dependent callable returning either.

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
    inactive_handling = str(inactive_handling).strip().lower().replace("-", "_")
    if inactive_handling not in {"hard_zero", "ncp"}:
        raise ValueError(
            "inactive_handling must be 'hard_zero' or 'ncp' "
            f"(got {inactive_handling!r})"
        )
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

        if rhs_jac is not None and _rhs_neg_A[0] is not None:
            out[:n_phys] = np.asarray(_rhs_neg_A[0] @ yp).ravel() + _rhs_b_const[0]
        elif rhs_jac is not None:
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
        normal_r_arr = _eval_contact_scalar_field(normal_r, normal_r_nargs, n_blocks, "normal_r", y, t=t, Fk_val=Fk_val)
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

        if _jac_top_left[0] is None:
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

            _jac_top_left[0] = J_s

        top_left = _jac_top_left[0]

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
        normal_r_arr = _eval_contact_scalar_field(normal_r, normal_r_nargs, n_blocks, "normal_r", y, t=t, Fk_val=Fk_val)
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
    )
