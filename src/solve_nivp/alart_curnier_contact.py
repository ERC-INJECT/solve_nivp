"""Alart-Curnier frictional contact helpers.

This module provides a full-state benchmark implementation of the
Alart-Curnier residual discussed in::

    Bertails-Descoubes, Cadoux, Daviet, Acary (2011)
    "A Nonsmooth Newton Solver for Capturing Exact Coulomb Friction
     in Fiber Assemblies"

Unlike :mod:`solve_nivp.contact`, which augments the physical system
with reaction DOFs and enforces the Coulomb cone through an external
projection / natural-map formulation, this helper embeds the
Alart-Curnier contact law directly into the residual:

    G(z, r) = [ R_z(z, r) ;
                f_AC(u(z), r) ]

where ``R_z`` is the physical backward-Euler residual and
``f_AC`` is the exact Coulomb contact law written as a zero-finding
problem.  This is intentionally a *full-state* implementation:
we keep the physical unknowns ``z`` and reaction unknowns ``r``
coupled in one Newton solve rather than eliminating the physical
unknowns to form a Delassus operator.

The purpose of this backend is to benchmark whether the
Alart-Curnier contact residual itself yields a practical advantage
over the package's existing projection-based contact path, without
changing the time-stepping or linear-algebra architecture.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.sparse as sp

from .contact import ContactSystem
from .projections import AlgebraicConstraintProjection, IdentityProjection


def _count_required_args(fn):
    """Count declared positional arguments of *fn*.

    We intentionally count *all* positional parameters, not only the
    required ones.  These callbacks commonly advertise flexible SOC-style
    signatures such as ``(y, t=None, Fk_val=None)``; using only the number
    of required arguments would incorrectly suppress the time/state context.
    """
    if fn is None:
        return 0
    try:
        import inspect

        sig = inspect.signature(fn)
        return sum(
            1
            for p in sig.parameters.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    except (TypeError, ValueError):
        return None


def _call_with_time_state_fk(fn, t, y, fk):
    """Call a smooth RHS / Jacobian with flexible signature."""
    try:
        return fn(t, y, fk)
    except TypeError:
        try:
            return fn(t, y)
        except TypeError:
            return fn(y)


def _parse_prev_and_h(extra, y_shape):
    """Extract ``prev_state``, ``Fk_val`` and ``h`` from integrator ``*extra`` args."""
    h_val = None
    prev_state = None
    fk_val = None
    matched_arrays = []
    for a in reversed(extra):
        if a is not None and np.isscalar(a):
            h_val = float(a)
            break
    for a in extra:
        if isinstance(a, np.ndarray) and a.shape == y_shape:
            matched_arrays.append(a)
    if matched_arrays:
        prev_state = matched_arrays[0]
    if len(matched_arrays) >= 2:
        fk_val = matched_arrays[1]
    return prev_state, fk_val, h_val


def _dense_or_sparse(mat):
    """Return *mat* as CSR when sparse-like, else ndarray."""
    if sp.issparse(mat):
        return mat.tocsr()
    return np.asarray(mat, dtype=float)


def _call_state_time_fk(fn, nargs, y, t=None, Fk_val=None):
    """Call a state-dependent scalar callback with SOC-style flexible arity."""
    if fn is None:
        return None
    if nargs is None or nargs >= 3:
        try:
            return fn(y, t, Fk_val)
        except TypeError:
            return fn(y, t) if nargs != 1 else fn(y)
    if nargs == 2:
        return fn(y, t)
    return fn(y)


def _call_state_block_time_fk(fn, nargs, y, k, t=None, Fk_val=None):
    """Call a block-wise callback with SOC-style flexible arity."""
    if fn is None:
        return None
    if nargs is None or nargs >= 4:
        try:
            return fn(y, k, t, Fk_val)
        except TypeError:
            try:
                return fn(y, k, t)
            except TypeError:
                return fn(y, k)
    if nargs == 3:
        return fn(y, k, t)
    return fn(y, k)


def _vectorize_mu(contacts, y_state, t=None, Fk_val=None):
    """Evaluate per-contact friction coefficients on the physical state."""
    mu = np.empty(len(contacts), dtype=float)
    for k, ci in enumerate(contacts):
        mu[k] = float(_call_state_time_fk(ci["get_mu"], ci["mu_nargs"], y_state, t, Fk_val))
    return mu


def _eval_s0(get_s0, s0_nargs, n_blocks, y, t=None, Fk_val=None):
    """Evaluate normal reaction offsets per contact block."""
    if get_s0 is None:
        return np.zeros(n_blocks, dtype=float)
    arr = np.atleast_1d(np.asarray(_call_state_time_fk(get_s0, s0_nargs, y, t, Fk_val), dtype=float))
    if arr.size == 1:
        return np.full(n_blocks, float(arr.flat[0]), dtype=float)
    if arr.size != n_blocks:
        raise ValueError(
            f"get_s0 must return a scalar or array of length {n_blocks} (got size {arr.size})"
        )
    return arr.ravel()


def _eval_w0(get_w0, w0_nargs, y, k, m_k, t=None, Fk_val=None):
    """Evaluate tangential reaction offsets for block *k*."""
    if get_w0 is None:
        return np.zeros(m_k, dtype=float)
    arr = np.atleast_1d(
        np.asarray(_call_state_block_time_fk(get_w0, w0_nargs, y, k, t, Fk_val), dtype=float)
    )
    if arr.size != m_k:
        raise ValueError(
            f"get_w0 must return an array of length {m_k} for block {k} (got size {arr.size})"
        )
    return arr


def _eval_ds0_dz(get_ds0_dz, ds0_nargs, n_blocks, n_state, y, t=None, Fk_val=None):
    """Evaluate Jacobian of the normal offset."""
    if get_ds0_dz is None:
        return None
    arr = np.asarray(_call_state_time_fk(get_ds0_dz, ds0_nargs, y, t, Fk_val), dtype=float)
    if arr.ndim == 1:
        if arr.size != n_state:
            raise ValueError(
                f"get_ds0_dz returned length {arr.size}; expected state dimension {n_state}"
            )
        arr = np.tile(arr.reshape(1, -1), (n_blocks, 1))
    elif arr.shape == (1, n_state):
        arr = np.tile(arr, (n_blocks, 1))
    elif arr.shape != (n_blocks, n_state):
        raise ValueError(
            f"get_ds0_dz must return shape ({n_blocks}, {n_state}) or ({n_state},), got {arr.shape}"
        )
    return arr


def _eval_dw0_dz(get_dw0_dz, dw0_nargs, y, k, m_k, n_state, t=None, Fk_val=None):
    """Evaluate Jacobian of the tangential offset for block *k*."""
    if get_dw0_dz is None:
        return None
    arr = np.asarray(
        _call_state_block_time_fk(get_dw0_dz, dw0_nargs, y, k, t, Fk_val), dtype=float
    )
    if arr.shape != (m_k, n_state):
        raise ValueError(
            f"get_dw0_dz must return shape ({m_k}, {n_state}) for block {k}, got {arr.shape}"
        )
    return arr


def _project_ball_and_jac(y_vec, delta, *, tie_tol=1e-14):
    r"""Projection onto a Euclidean ball and one generalized Jacobian.

    Returns ``(proj, dproj_ddelta, dproj_dy)`` for

        g(delta, y) = Proj_{B(0, delta)}(y)

    using a deterministic Clarke selection on the boundary.
    """
    y_vec = np.asarray(y_vec, dtype=float)
    m = y_vec.size
    if delta <= 0.0:
        return np.zeros(m), np.zeros(m), np.zeros((m, m))

    norm_y = float(np.linalg.norm(y_vec))
    if norm_y <= delta + tie_tol:
        return y_vec.copy(), np.zeros(m), np.eye(m)

    if norm_y <= tie_tol:
        # Only reachable when delta ~ 0 and the point lies near the kink.
        return np.zeros(m), np.zeros(m), np.zeros((m, m))

    y_hat = y_vec / norm_y
    proj = delta * y_hat
    dproj_ddelta = y_hat
    dproj_dy = (delta / norm_y) * (np.eye(m) - np.outer(y_hat, y_hat))
    return proj, dproj_ddelta, dproj_dy


def _normal_residual_and_jac(u_n, r_n, rho_n, *, tie_tol=1e-14):
    """Normal Alart-Curnier block and one generalized derivative."""
    s = float(r_n - rho_n * u_n)
    if s >= -tie_tol:
        # Active scalar projection branch: max(s, 0) = s
        f_n = -rho_n * float(u_n)
        df_du = -rho_n
        df_dr = 0.0
    else:
        # Inactive scalar projection branch: max(s, 0) = 0
        f_n = -float(r_n)
        df_du = 0.0
        df_dr = -1.0
    return f_n, df_du, df_dr


def _contact_block_residual_and_jac(
    u_blk,
    r_blk,
    mu,
    rho_n,
    rho_t,
    tangential_damping=0.0,
):
    r"""Compute one Alart-Curnier block residual and generalized Jacobian.

    Parameters
    ----------
    u_blk : ndarray, shape (d,)
        Contact-relative velocity in local order ``[u_N, u_T...]``.
    r_blk : ndarray, shape (d,)
        Reaction vector in matching local order ``[r_N, r_T...]``.
    mu : float
        Friction coefficient.
    rho_n, rho_t : float
        Alart-Curnier scaling parameters.
    tangential_damping : float, default 0.0
        Optional fault-local tangential damping. The tangential law is
        enforced on the frictional part

        ``r_T,fric = r_T - tangential_damping * u_T``.

        This provides a small quasi-dynamic regularization for
        force-controlled sliding problems while leaving the benchmark
        unchanged when set to zero.

    Returns
    -------
    f_blk : ndarray, shape (d,)
    df_du : ndarray, shape (d, d)
    df_dr : ndarray, shape (d, d)
    """
    u_blk = np.asarray(u_blk, dtype=float)
    r_blk = np.asarray(r_blk, dtype=float)
    d = r_blk.size
    m = d - 1

    f_blk = np.zeros(d, dtype=float)
    df_du = np.zeros((d, d), dtype=float)
    df_dr = np.zeros((d, d), dtype=float)

    # Normal part
    f_n, df_n_du, df_n_dr = _normal_residual_and_jac(u_blk[0], r_blk[0], rho_n)
    f_blk[0] = f_n
    df_du[0, 0] = df_n_du
    df_dr[0, 0] = df_n_dr

    if m == 0:
        return f_blk, df_du, df_dr

    # Tangential part
    tangential_damping = float(tangential_damping)
    delta = float(mu * r_blk[0])
    r_t_fric = r_blk[1:] - tangential_damping * u_blk[1:]
    y_vec = r_t_fric - rho_t * u_blk[1:]
    proj, dproj_ddelta, dproj_dy = _project_ball_and_jac(y_vec, delta)
    f_blk[1:] = proj - r_t_fric

    df_du[1:, 1:] = -(rho_t + tangential_damping) * dproj_dy + tangential_damping * np.eye(m)
    df_dr[1:, 1:] = dproj_dy - np.eye(m)
    df_dr[1:, 0] = mu * dproj_ddelta

    return f_blk, df_du, df_dr


def build_alart_curnier_contact(
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
    rho_n=1.0,
    rho_t=1.0,
    gap_tol=0.0,
    tangential_damping=0.0,
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
    smooth_rhs_is_affine=False,
):
    r"""Build a full-state Alart-Curnier contact system.

    This helper augments the physical state with explicit reaction DOFs
    exactly like :func:`solve_nivp.contact.build_impulse_contact`, but
    replaces the external cone projection by an explicit Alart-Curnier
    residual on the reaction rows.

    The resulting implicit step solves the residual

    .. math::
        G(z, r) =
        \begin{bmatrix}
            R_z(z, r) \\
            f_{AC}(u(z), r)
        \end{bmatrix} = 0

    using the package's standard Newton path with
    :class:`~solve_nivp.IdentityProjection`.

    Parameters
    ----------
    A, rhs_smooth, y0, contacts, gap_func, B, theta, coupling_theta,
    component_slices, C_extract, D_extract, rate_form, constraints,
    rhs_jac
        Same interpretation as in
        :func:`solve_nivp.contact.build_impulse_contact`,
        except that ``e`` / restitution coefficients in ``contacts`` are
        currently ignored by this benchmark implementation.
    rho_n, rho_t : float, default 1.0
        Alart-Curnier normal and tangential scaling parameters.
    gap_tol : float, default 0.0
        Contact-activation tolerance.  A contact is active when
        ``gap <= gap_tol``.  Inactive contacts enforce ``r = 0``.
    tangential_damping : float, default 0.0
        Optional fault-local tangential damping used in the tangential
        Alart-Curnier residual. Small nonzero values regularize
        force-controlled dynamic sliding while preserving the original
        benchmark when left at zero.
    reaction_units : {"impulse", "force"}, optional
        Units carried by the reaction unknowns.

        ``"impulse"`` reproduces the original backward-Euler-oriented
        benchmark behavior: the physical rows couple reactions through

        ``B @ r / (coupling_theta * h)``.

        ``"force"`` treats reactions as force-like / traction-like
        unknowns and couples them directly through

        ``B @ r``.

        The ``"force"`` mode is the natural choice for explicit-velocity
        dynamic state vectors such as ``[u, v, p, r]`` and is not tied to
        a particular one-step integrator.
    offset_coupling_mode : {"constitutive_shift", "incremental_reference", "total_traction", "constitutive_shift_with_load"}, optional
        Controls how the optional offsets couple into the bulk equations.

        ``"constitutive_shift"`` keeps the original benchmark behavior:
        offsets only translate the local Alart-Curnier law, while the
        physical rows couple only the unknown reaction block ``r``.

        ``"incremental_reference"`` treats the offsets as a prescribed
        reference loading history.  The contact law still uses the total
        traction ``r + offset`` while the physical rows see

        ``r + (offset(t_{n+1}) - offset(t_0))``.

        This is the natural incremental formulation for prestressed
        interface loading: constant offsets act as reference prestress,
        while changes in the prescribed offset history drive the bulk.
        When separate load/reference callbacks are supplied, the physical
        rows use

        ``r + (load_offset(t_{n+1}) - ref_offset)``.

        Inactive (open) contacts enforce ``r = 0`` so that only the
        ``load - ref`` offset drives the bulk (no spurious anti-prestress).

        ``"total_traction"`` couples the full translated traction into the
        physical rows:

        ``r + offset(t_{n+1})``.

        This is useful for a static preload solve where the prescribed
        prestress should actually load the bulk so the converged state can
        be transferred into a subsequent dynamic release run.

        ``"constitutive_shift_with_load"`` splits the two roles: the
        contact law still uses the constitutive offsets ``[s0, w0]`` like
        ``"constitutive_shift"``, including ``r = 0`` on inactive
        contacts, while the physical rows also couple an optional
        prescribed load-offset field.  This is primarily useful for
        debugging or comparing against an explicit preload/release branch;
        it is not the default prestress formulation used by the helper.
    get_s0, get_w0 : callable or None, optional
        Optional SOC-style normal / tangential reaction offsets used to
        translate the Alart-Curnier reaction coordinates.  For active
        contacts the residual is evaluated on

        ``r_eff = r + [s0, w0]``.

        ``get_s0`` follows the SOC convention
        ``(y)``, ``(y, t)``, or ``(y, t, Fk_val)`` and may return a
        scalar or one value per contact block.  ``get_w0`` follows
        ``(y, k)``, ``(y, k, t)``, or ``(y, k, t, Fk_val)`` and returns
        the tangential offset vector for block ``k``.
    get_ds0_dz, get_dw0_dz : callable or None, optional
        Optional Jacobians of the offsets with respect to the full
        augmented state.  When supplied, these are included in the
        assembled generalized Jacobian exactly like the SOC helper's
        state-dependent pre-stress corrections.
    get_s0_load, get_w0_load : callable or None, optional
        Optional SOC-style offsets used in the physical-row loading terms.
        Under ``offset_coupling_mode="incremental_reference"`` these define
        the target traction history that is compared against the fixed
        reference offsets. Under
        ``offset_coupling_mode="constitutive_shift_with_load"`` they define
        the explicit load-offset field. When omitted in the latter mode,
        the constitutive offsets are reused.
    get_ds0_load_dz, get_dw0_load_dz : callable or None, optional
        Optional Jacobians of the physical-row load offsets.
    get_s0_ref, get_w0_ref : callable or None, optional
        Optional fixed reference offsets used by
        ``offset_coupling_mode="incremental_reference"``. When supplied,
        the physical rows use ``load_offset - ref_offset`` while the
        contact law still uses the constitutive offsets ``[s0, w0]``.
    """
    y0 = np.asarray(y0, dtype=float).ravel()
    n_phys = y0.size

    if coupling_theta is None:
        coupling_theta = theta
    coupling_theta = float(coupling_theta)
    rho_n = float(rho_n)
    rho_t = float(rho_t)
    gap_tol = float(gap_tol)
    tangential_damping = float(tangential_damping)
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

    # Normalize contacts and reaction ordering
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

    # Coupling matrix B
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

    n_sources_dummy = None
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

        s0_arr = _eval_s0(get_s0_fun, s0_nargs_fun, len(norm_contacts), y_full, t=t, Fk_val=Fk_val)
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
            get_ds0_fun, ds0_nargs_fun, len(norm_contacts), n_aug, y_full, t=t, Fk_val=Fk_val
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

    # ----------------------------------------------------------------
    # Performance caches — avoid recomputing constant blocks every call
    # ----------------------------------------------------------------
    _B_csr = sp.csr_matrix(B_mat) if not sp.issparse(B_mat) else B_mat.tocsr()
    _U_contact_csr = (
        (sp.csr_matrix(U_contact) if not sp.issparse(U_contact) else U_contact.tocsr())
        if U_contact is not None
        else None
    )
    _jac_top_left = [None]       # patched smooth Jacobian (CSR), computed once
    _jac_top_right_key = [None]  # cache key for top_right scaling
    _jac_top_right = [None]      # scaled B with constraint rows zeroed (CSR)
    _rhs_b_const = [None]        # constant affine part of smooth RHS
    _rhs_neg_A = [None]          # smooth Jacobian as CSR (for fast matvec)

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

        # Direct physical indexing path
        if rate_form:
            if prev_state is not None and h_val is not None and h_val > 0.0:
                return (yp[vel_indices] - prev_state[:n_phys][vel_indices]) / h_val
            return np.zeros(n_react, dtype=float)
        return yp[vel_indices]

    def rhs_aug(t, y, *extra):
        prev_state, Fk_val, h_val = _parse_prev_and_h(extra, y.shape)
        yp = y[:n_phys]
        r = y[n_phys:]
        out = np.zeros(n_aug, dtype=float)
        offset_vec = _assemble_offset_vector(y, t=t, Fk_val=Fk_val)
        load_offset_vec = _assemble_load_offset_vector(y, t=t, Fk_val=Fk_val)

        # Differential / physical rows — fast path only under an explicit
        # affine opt-in (A constant, body-force/BCs constant).  Then the RHS
        # is rhs_smooth(t, yp) = J yp + b with J = rhs_jac and b =
        # rhs_smooth(t, 0), snapshot once and reused as a sparse matvec.
        # A provided rhs_jac alone does NOT imply an affine RHS, so without
        # the flag we evaluate rhs_smooth directly each call.
        if smooth_rhs_is_affine and rhs_jac is not None and _rhs_neg_A[0] is not None:
            out[:n_phys] = np.asarray(
                _rhs_neg_A[0] @ yp
            ).ravel() + _rhs_b_const[0]
        elif smooth_rhs_is_affine and rhs_jac is not None:
            # First call: build cache
            J_s = _call_with_time_state_fk(rhs_jac, t, yp, None)
            J_s = _dense_or_sparse(J_s)
            _rhs_neg_A[0] = sp.csr_matrix(J_s) if not sp.issparse(J_s) else J_s.tocsr()
            _rhs_b_const[0] = np.asarray(
                _call_with_time_state_fk(rhs_smooth, t, np.zeros(n_phys), None)
            ).ravel()
            out[:n_phys] = np.asarray(
                _rhs_neg_A[0] @ yp
            ).ravel() + _rhs_b_const[0]
        else:
            out[:n_phys] = _call_with_time_state_fk(
                rhs_smooth, t, yp, None
            )
        reaction_scale = _reaction_scale(h_val)
        if reaction_scale is not None:
            out[:n_phys] += reaction_scale * np.asarray(_B_csr @ r).ravel()
            if _incremental_offset_loading:
                out[:n_phys] += reaction_scale * np.asarray(
                    _B_csr @ (load_offset_vec - _offset_ref_vec)
                ).ravel()
            elif _total_offset_loading or _split_load_offset:
                out[:n_phys] += reaction_scale * np.asarray(_B_csr @ load_offset_vec).ravel()

        # Algebraic physical rows are explicit residual rows:
        #   F_q = q - g(y)  ==>  rhs_q = -(q - g(y)) = g(y) - q
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

        # Contact rows
        u_rel = _contact_velocity(yp, prev_state, h_val)
        gaps = np.atleast_1d(gap_aug(y, t))
        mu_arr = _vectorize_mu(norm_contacts, yp, t=t, Fk_val=Fk_val)

        for k, ci in enumerate(norm_contacts):
            sl = ci["block_slice"]
            blk_rows = slice(sl.start, sl.stop)
            r_blk = r[blk_rows]
            offset_blk = offset_vec[blk_rows]
            active = bool(gaps[k] <= gap_tol)
            if not active:
                # Inactive (open) contacts always enforce r = 0 (no
                # reaction).  Earlier versions used r + offset = 0 under
                # incremental_reference, which injected a spurious
                # anti-prestress force into the bulk via B @ (r + load - ref)
                # and created a self-reinforcing opening instability.
                f_blk = r_blk.copy()
            else:
                u_blk = u_rel[blk_rows]
                r_eff = r_blk.copy() + offset_blk
                f_blk, _, _ = _contact_block_residual_and_jac(
                    u_blk, r_eff, mu_arr[k], rho_n, rho_t, tangential_damping
                )
            out[n_phys + sl.start : n_phys + sl.stop] = -f_blk

        return out

    def _fd_smooth_jac(t, yp):
        f0 = _call_with_time_state_fk(rhs_smooth, t, yp, None)
        n_p = len(yp)
        eps_base = 1e-7
        h_vec = eps_base * np.maximum(np.abs(yp), 1.0)
        J = np.empty((n_p, n_p), dtype=float)
        for j in range(n_p):
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

        # ---- Top-left: smooth Jacobian + constraint patches (cached) ----
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
                    yp, n_phys, t=t, Fk_val=None, step_size=h_val,
                    prev_state=(prev_state[:n_phys] if prev_state is not None else None),
                ).tocsr()
                J_s = J_s.tolil()
                for qs in q_slices:
                    J_s[qs, :] = (-patch[qs, :]).tolil()
                J_s = J_s.tocsr()

            _jac_top_left[0] = J_s

        top_left = _jac_top_left[0]

        # ---- Top-right: B/(θh) with constraint rows zeroed (cached per h) ----
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

        # ---- Bottom blocks: contact rows (rebuilt, but no LIL) ----
        u_rel = _contact_velocity(yp, prev_state, h_val)
        gaps = np.atleast_1d(gap_aug(y, t))
        mu_arr = _vectorize_mu(norm_contacts, yp, t=t, Fk_val=Fk_val)

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
            m_k = d - 1
            active = bool(gaps[k] <= gap_tol)
            offset_blk = offset_vec[sl.start:sl.stop]
            dz0_blk = None if offset_jac is None else offset_jac[sl.start:sl.stop, :]

            if not active:
                # Inactive Jacobian: df/dr = -I enforces r = 0.
                # No offset Jacobian terms — see rhs_aug inactive fix.
                block_right = np.zeros((d, n_react), dtype=float)
                block_right[:, sl.start:sl.stop] = -np.eye(d)
                bl_dense = np.zeros((d, n_phys), dtype=float)
                br_dense[sl.start:sl.stop, :] = block_right
                bl_parts.append(sp.csr_matrix(bl_dense))
                continue

            u_blk = u_rel[sl.start:sl.stop]
            r_blk = r[sl.start:sl.stop]
            r_eff = r_blk.copy() + offset_blk
            _, df_du, df_dr = _contact_block_residual_and_jac(
                u_blk, r_eff, mu_arr[k], rho_n, rho_t, tangential_damping
            )

            if U_scaled is not None:
                U_blk = U_scaled[sl.start:sl.stop, :]
                bl_dense = -(df_du @ U_blk.toarray())
            elif vel_indices is not None:
                if rate_form:
                    scale = 0.0 if (h_val is None or h_val <= 0.0) else (1.0 / h_val)
                else:
                    scale = 1.0
                bl_dense = np.zeros((d, n_phys), dtype=float)
                for li in range(d):
                    for lj in range(d):
                        bl_dense[li, vel_indices[sl.start + lj]] += -df_du[li, lj] * scale
            else:
                bl_dense = np.zeros((d, n_phys), dtype=float)

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
            [[top_left, top_right],
             [bottom_left, bottom_right]],
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


def build_dynamic_alart_curnier_contact(
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
    rho_n=1.0,
    rho_t=1.0,
    gap_tol=0.0,
    tangential_damping=0.0,
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
    smooth_rhs_is_affine=False,
):
    r"""Build an explicit-velocity dynamic Alart-Curnier contact system.

    This is a thin wrapper around :func:`build_alart_curnier_contact` for
    first-order dynamic state vectors such as ``[u, v, p]`` where the
    contact law should use the *state velocity DOFs directly* rather than a
    backward-Euler finite-difference reconstruction.

    The returned augmented system has force-like reaction unknowns, so the
    physical rows couple contact through ``B @ r`` and are therefore usable
    with the package's one-step integrators without BE-specific ``1/h``
    impulse scaling.

    Parameters
    ----------
    gap_extract : ndarray or sparse, optional
        Extraction operator used when ``gap_func`` is not supplied.
    vel_extract : ndarray or sparse, optional
        Operator mapping the physical state directly to local contact
        relative velocity blocks.  For explicit ``[u, v, p]`` states this
        should typically extract from the velocity block.
    All other arguments
        Forwarded to :func:`build_alart_curnier_contact`.
    """
    return build_alart_curnier_contact(
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
        rho_n=rho_n,
        rho_t=rho_t,
        gap_tol=gap_tol,
        tangential_damping=tangential_damping,
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
        smooth_rhs_is_affine=smooth_rhs_is_affine,
    )
