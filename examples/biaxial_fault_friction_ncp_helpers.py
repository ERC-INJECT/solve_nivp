#!/usr/bin/env python
"""Helpers for an image-style biaxial compression friction benchmark.

The benchmark is intentionally centered on the paper-faithful Macklin-style
dynamic NCP contact law:

- ``friction_law="compliance"``
- ``inactive_handling="hard_zero"``

The geometry and boundary conditions are closer to the geomechanics sketch the
user provided than the older top-shortening sliding-block benchmark:

- a square domain,
- left/bottom roller supports,
- an inclined internal fault,
- ramped compression on the top/right boundaries,
- optional fault prestress offsets to start from a closed interface.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from skfem.models.elasticity import lame_parameters

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[1]
PORO_ROOT = Path("/home/david/Documents/Poroelasticity")
NCP_NOTEBOOK_PATH = REPO_ROOT / "examples" / "prestressed_fault_dynamic_ncp.ipynb"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if PORO_ROOT.exists() and str(PORO_ROOT) not in sys.path:
    sys.path.insert(0, str(PORO_ROOT))

import solve_nivp  # noqa: E402
from poroelasticity.cgporoelastostatics import CGPoroelastostatics  # noqa: E402
from poroelasticity.mesh_builder import CrackMeshBuilder  # noqa: E402
from solve_nivp.ncp_contact import build_dynamic_ncp_contact  # noqa: E402


_NCP_NOTEBOOK_NS: dict[str, Any] | None = None


def _load_prestressed_ncp_namespace() -> dict[str, Any]:
    """Load helper functions from the existing prestressed NCP notebook."""
    global _NCP_NOTEBOOK_NS
    if _NCP_NOTEBOOK_NS is not None:
        return _NCP_NOTEBOOK_NS

    nb = json.loads(NCP_NOTEBOOK_PATH.read_text())
    ns: dict[str, Any] = {
        "__builtins__": __builtins__,
        "contextlib": contextlib,
        "io": io,
        "np": np,
        "sp": sp,
        "time": time,
        "solve_nivp": solve_nivp,
        "lame_parameters": lame_parameters,
        "CGPoroelastostatics": CGPoroelastostatics,
        "CrackMeshBuilder": CrackMeshBuilder,
        "build_dynamic_ncp_contact": build_dynamic_ncp_contact,
    }
    helper_src = "".join(nb["cells"][3]["source"])
    exec(compile(helper_src, f"{NCP_NOTEBOOK_PATH.name}:cell3", "exec"), ns)
    _NCP_NOTEBOOK_NS = ns
    return ns


def make_image_style_biaxial_bc(
    *,
    right_v1_rate: float,
    top_v2_rate: float,
    dp_rate: float = 0.0,
    left_v1: float = 0.0,
    bottom_v2: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Return roller-style BCs plus ramped compression on the loaded faces."""
    bc: dict[str, dict[str, float]] = {
        "v1": {"left": float(left_v1)},
        "v2": {"bottom": float(bottom_v2)},
        "dp_rate": {
            "left": float(dp_rate),
            "right": float(dp_rate),
            "top": float(dp_rate),
            "bottom": float(dp_rate),
        },
    }
    if abs(float(right_v1_rate)) > 0.0:
        bc["v1_rate"] = {"right": float(right_v1_rate)}
    if abs(float(top_v2_rate)) > 0.0:
        bc["v2_rate"] = {"top": float(top_v2_rate)}
    return bc


def normalized_fault_from_image() -> dict[str, float]:
    """Return the normalized version of the fault shown in the sketch."""
    x1, y1 = 0.20, 0.205
    x2, y2 = 0.80, 0.805
    dx = x2 - x1
    dy = y2 - y1
    return {
        "xmin": 0.0,
        "xmax": 1.0,
        "ymin": 0.0,
        "ymax": 1.0,
        "fault_x0": 0.5 * (x1 + x2),
        "fault_y0": 0.5 * (y1 + y2),
        "fault_length": float(np.hypot(dx, dy)),
        "fault_theta": float(np.arctan2(dy, dx)),
        "fault_x1": x1,
        "fault_y1": y1,
        "fault_x2": x2,
        "fault_y2": y2,
    }


def _reaction_nl_atol_from_offset_mismatch(
    prestress_target: dict[str, Any],
    prestress_background: dict[str, Any],
) -> float:
    """Scale the reaction tolerance from the actual target-reference mismatch."""
    delta_tau = np.asarray(
        prestress_target["tau_profile"] - prestress_background["tau_profile"],
        dtype=float,
    )
    mismatch = float(np.max(np.abs(delta_tau))) if delta_tau.size else 0.0
    if mismatch <= 0.0:
        return 1.0e-3
    return max(1.0e-14, min(1.0e-3, 1.0e-3 * mismatch))


def build_biaxial_fault_ncp_context(
    *,
    mu_friction: float = 0.6,
    n_elem: int = 18,
    element_type: str = "tri",
    xmin: float = 0.0,
    xmax: float = 1.0,
    ymin: float = 0.0,
    ymax: float = 1.0,
    fault_x0: float = 0.5,
    fault_y0: float = 0.505,
    fault_theta: float = float(np.pi / 4.0),
    fault_length: float = float(np.hypot(0.6, 0.6)),
    right_v1_rate: float = -0.02,
    top_v2_rate: float = -0.01,
    dp_rate: float = 0.0,
    normal_prestress: float = 6.6,
    background_ratio: float = 0.0,
    patch_ratio: float = 0.0,
    patch_half_width: float = 0.08,
    patch_center: float | None = None,
    dynamic_density: float = 1.101,
    dynamic_lumped_mass: bool = False,
    eta_fluid: float = 2.0e-18 / 3600.0,
    gap_tol: float = 1.0e-12,
    normal_r: float | str = 0.5,
    friction_r: float | str = 0.5,
    ncp_type: str = "fischer_burmeister",
    solver_overrides: dict[str, Any] | None = None,
    integrator_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the image-style biaxial compression benchmark context."""
    ns = _load_prestressed_ncp_namespace()
    compute_contact_plot_geometry = ns["compute_contact_plot_geometry"]
    make_fault_prestress_patch_callbacks = ns["make_fault_prestress_patch_callbacks"]

    nu = 0.25
    g_shear = 22.0e3
    beta_fluid = 8.5e-5
    k_perm = 1.0e-15 * 1.0e-6
    alpha_biot = 0.0
    scale_l = 1.0
    scale_eps = 1.0e-3
    bulk_mu_v = 0.0
    bulk_lam_v = 0.0

    e_young = 2.0 * g_shear * (1.0 + nu)
    lam, mu = lame_parameters(e_young, nu)
    params = (mu, lam, alpha_biot, beta_fluid, k_perm / float(eta_fluid))

    def body_force_fn(x, t):
        return np.array([np.zeros_like(x[0]), np.zeros_like(x[0])])

    bc = make_image_style_biaxial_bc(
        right_v1_rate=right_v1_rate,
        top_v2_rate=top_v2_rate,
        dp_rate=dp_rate,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        builder = CrackMeshBuilder(
            float(xmin),
            float(xmax),
            float(ymin),
            float(ymax),
            int(n_elem),
            crack_theta=float(fault_theta),
            crack_x0=float(fault_x0),
            crack_y0=float(fault_y0),
            crack_length=float(fault_length),
            element_type=str(element_type),
            conforming=True,
        )
        mesh, el_p, el_u, crack, _ = builder.build()

        poro = CGPoroelastostatics(
            mesh=mesh,
            element_p=el_p,
            element_u=el_u,
            params=params,
            crack=crack,
            intorder=10,
            model_params={"T": 0.0, "k_n": 0.0, "k_t": 0.0, "eta_n": None, "eta_t": None},
            scales=(scale_l, scale_eps),
            bc=bc,
            point_source_location=None,
            sources_RMS_width=0.05,
            source_type="wendland",
            P_scale=1.0,
            verbose=False,
            free_memory=True,
            enforcement_type="nodal",
            include_hessian=False,
            include_taylor=True,
            lumped_coupling="consistent",
            body_force=body_force_fn,
            bulk_viscosity={"mu_v": bulk_mu_v, "lam_v": bulk_lam_v},
            crack_law="nonsmooth",
            rotate_crack_to_nt=True,
        )
        poro.strip_multiplier_dynamics()
        _, meta = poro.build_projection()

    dyn = poro.build_first_order_dynamic_system(
        meta,
        density=float(dynamic_density),
        lumped_mass=bool(dynamic_lumped_mass),
    )

    n_orig = int(dyn["A"].shape[0])
    n_c = int(poro.n_lambda_q)
    dim = int(poro.dim)
    n_ls = dim * n_c
    field_sl = meta["field_sl"]
    lam_s_sl = meta["lam_s_sl"]

    c_s_full = meta["C_s_full"]
    r_nt = meta["R_nt"]
    n_field = int(field_sl.stop)
    n_lam_s = int(c_s_full.shape[0])

    r_block = sp.block_diag([sp.eye(n_ls), r_nt], format="csr")
    c_s_nt = r_block @ c_s_full
    c_extract = sp.hstack(
        [c_s_nt, sp.csr_matrix((n_lam_s, poro.ndofs - n_field))],
        format="csr",
    ).tocsr()

    sign_diag = np.ones(n_lam_s, dtype=float)
    sign_diag[n_ls : n_ls + n_c] = -1.0
    c_extract_contact = (sp.diags(sign_diag, format="csr") @ c_extract).tocsr()

    mu_arr = np.asarray(mu_friction, dtype=float).ravel()
    if mu_arr.size == 1:
        mu_profile = np.full(n_c, float(mu_arr[0]), dtype=float)
    elif mu_arr.size == n_c:
        mu_profile = mu_arr.astype(float, copy=True)
    else:
        raise ValueError(
            f"mu_friction must be scalar or length-{n_c}, got shape {mu_arr.shape}."
        )

    contacts = [
        {
            "vel_normal_idx": n_ls + k,
            "vel_tangential_idx": [n_ls + n_c + k],
            "mu": float(mu_profile[k]),
            "e": 0.0,
        }
        for k in range(n_c)
    ]

    jump_start = int(lam_s_sl.start) + n_ls
    jump_end = int(lam_s_sl.stop)
    b_jump_xy = (-poro.A[:, jump_start:jump_end]).tocsc()
    b_jump = (b_jump_xy @ r_nt.T).tocsc()
    perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
    b_contact = b_jump[:, perm].tocsr()

    velocity_len = int(dyn["velocity_slice"].stop - dyn["velocity_slice"].start)
    gap_extract_dyn = sp.hstack(
        [c_extract.tocsr(), sp.csr_matrix((c_extract.shape[0], velocity_len))],
        format="csr",
    ).tocsr()
    vel_extract_dyn = sp.hstack(
        [
            sp.csr_matrix((c_extract_contact.shape[0], dyn["n_base"])),
            c_extract_contact[:, dyn["u_state_indices"]].tocsr(),
        ],
        format="csr",
    ).tocsr()

    row_u = np.arange(poro.basis_p.N, poro.basis_p.N + poro.basis_u.N, dtype=int)
    b_contact_u_orig = b_contact[row_u, :].tocsr()
    dirichlet_local = np.array(
        [
            d - poro.basis_p.N
            for d in np.asarray(getattr(poro, "_dirichlet_dof_set", []), dtype=int)
            if poro.basis_p.N <= d < poro.basis_p.N + poro.basis_u.N
        ],
        dtype=int,
    )
    if dirichlet_local.size:
        b_contact_u_orig = b_contact_u_orig.tolil()
        b_contact_u_orig[dirichlet_local, :] = 0.0
        b_contact_u_orig = b_contact_u_orig.tocsr()
    b_contact_u = (dyn["T_u"] @ b_contact_u_orig).tocsr()
    b_contact_dyn = sp.vstack(
        [sp.csr_matrix((dyn["n_base"], b_contact.shape[1])), b_contact_u],
        format="csr",
    )

    flux_constraint = meta["constraints"][0]
    n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
    zero_lam_s = {
        "g": lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
        "dg_dy": lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
        "y_slice": lam_s_sl,
        "q_slice": lam_s_sl,
    }

    contact_plot = compute_contact_plot_geometry(poro, n_c)
    prestress_target = make_fault_prestress_patch_callbacks(
        contact_plot["contact_s"],
        normal_prestress=float(normal_prestress),
        mu_friction=float(mu_profile[0]) if mu_profile.size else float(mu_friction),
        background_ratio=float(background_ratio),
        patch_ratio=float(patch_ratio),
        patch_center=patch_center,
        patch_half_width=float(patch_half_width),
    )
    prestress_background = make_fault_prestress_patch_callbacks(
        contact_plot["contact_s"],
        normal_prestress=float(normal_prestress),
        mu_friction=float(mu_profile[0]) if mu_profile.size else float(mu_friction),
        background_ratio=float(background_ratio),
        patch_ratio=float(background_ratio),
        patch_center=prestress_target["patch_center"],
        patch_half_width=prestress_target["patch_half_width"],
    )

    ncp_opts = {
        "ncp_type": str(ncp_type),
        "normal_r": normal_r,
        "friction_r": friction_r,
        "gap_tol": float(gap_tol),
        "offset_coupling_mode": "incremental_reference",
        "inactive_handling": "hard_zero",
        "friction_law": "compliance",
        "get_s0": prestress_target["get_s0"],
        "get_w0": prestress_target["get_w0"],
        "get_s0_load": prestress_target["get_s0"],
        "get_w0_load": prestress_target["get_w0"],
        "get_s0_ref": prestress_background["get_s0"],
        "get_w0_ref": prestress_background["get_w0"],
    }

    y0 = np.asarray(dyn["y0"], dtype=float).copy()
    y0[lam_s_sl] = 0.0

    cs = build_dynamic_ncp_contact(
        A=dyn["A"],
        rhs_smooth=dyn["rhs"],
        rhs_jac=dyn["rhs_jac"],
        y0=y0,
        contacts=contacts,
        B=b_contact_dyn,
        gap_extract=gap_extract_dyn,
        vel_extract=vel_extract_dyn,
        constraints=[flux_constraint, zero_lam_s],
        component_slices=dyn["component_slices"],
        **ncp_opts,
    )

    solver_opts_contact = {
        "max_iter": 30,
        "tol": 1.0e-10,
        "globalization": "linesearch",
        "use_broyden": False,
        "linear_solver": "splu",
        "sparse": True,
        "precond_reuse_steps": 50,
        "rhs_jac": cs.rhs_jac,
    }
    if solver_overrides:
        solver_opts_contact.update(solver_overrides)

    integrator_opts_contact = dict(cs.integrator_opts)
    if integrator_overrides:
        integrator_opts_contact.update(integrator_overrides)

    reaction_nl_atol = _reaction_nl_atol_from_offset_mismatch(
        prestress_target,
        prestress_background,
    )
    n_react_blocks = int(len(cs.component_slices) - len(dyn["component_slices"]))
    nl_atol_contact = list(dyn["nl_atol_per_block"]) + [reaction_nl_atol] * n_react_blocks

    adaptive_opts_contact = dict(meta.get("adaptive_opts", {}))
    react_indices = list(range(len(dyn["component_slices"]), len(cs.component_slices)))
    merged_skip = list(adaptive_opts_contact.get("skip_error_indices", [])) + react_indices
    adaptive_opts_contact["skip_error_indices"] = sorted({int(idx) for idx in merged_skip})
    adaptive_opts_contact["atol"] = list(dyn["atol_per_block"]) + [1.0e-6] * n_react_blocks

    return {
        "poro": poro,
        "meta": meta,
        "dyn": dyn,
        "cs": cs,
        "contacts": contacts,
        "n_orig": n_orig,
        "n_c": n_c,
        "n_ls": n_ls,
        "c_extract": gap_extract_dyn,
        "c_extract_static": c_extract,
        "c_extract_contact_static": c_extract_contact,
        "gap_extract": gap_extract_dyn,
        "vel_extract": vel_extract_dyn,
        "B_contact_static": b_contact,
        "B_contact_dynamic": b_contact_dyn,
        "solver_opts_contact": solver_opts_contact,
        "integrator_opts_contact": integrator_opts_contact,
        "nl_atol_contact": nl_atol_contact,
        "adaptive_opts_contact": adaptive_opts_contact,
        "contact_backend_opts": ncp_opts,
        "reaction_nl_atol": reaction_nl_atol,
        "prestress_target": prestress_target,
        "prestress_background": prestress_background,
        "contact_plot": contact_plot,
        "bc": bc,
        "geometry": {
            "xmin": float(xmin),
            "xmax": float(xmax),
            "ymin": float(ymin),
            "ymax": float(ymax),
            "fault_x0": float(fault_x0),
            "fault_y0": float(fault_y0),
            "fault_theta": float(fault_theta),
            "fault_length": float(fault_length),
        },
        "loading": {
            "right_v1_rate": float(right_v1_rate),
            "top_v2_rate": float(top_v2_rate),
            "dp_rate": float(dp_rate),
        },
        "material": {
            "nu": float(nu),
            "g_shear": float(g_shear),
            "beta_fluid": float(beta_fluid),
            "k_perm": float(k_perm),
            "alpha_biot": float(alpha_biot),
        },
    }


def run_biaxial_fault_ncp_history(
    *,
    ctx: dict[str, Any],
    label: str,
    t_end_hours: float,
    n_steps: int,
    adaptive: bool = True,
    time_method: str = "backward_euler",
    h0: float | None = None,
    integrator_overrides: dict[str, Any] | None = None,
    solver_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Advance one image-style biaxial NCP context and collect diagnostics."""
    ns = _load_prestressed_ncp_namespace()
    audit_state = ns["audit_state"]
    evaluate_contact_offset_history = ns["evaluate_contact_offset_history"]
    _compute_step_residual = ns["_compute_step_residual"]
    _scalarize_solver_error = ns["_scalarize_solver_error"]

    time_scale = float(ctx["poro"].get_scales()[0])
    tmax = float(t_end_hours) / time_scale
    if h0 is None:
        h0_use = tmax if not adaptive else min(tmax / float(n_steps), 1.0e-4)
    else:
        h0_use = float(h0)

    solver_opts = dict(ctx["solver_opts_contact"])
    if solver_overrides:
        solver_opts.update(solver_overrides)

    integrator_opts = dict(ctx["integrator_opts_contact"])
    if integrator_overrides:
        integrator_opts.update(integrator_overrides)

    solve_kwargs = dict(
        fun=ctx["cs"].rhs,
        t_span=(0.0, tmax),
        y0=np.asarray(ctx["cs"].y0, dtype=float).copy(),
        method=str(time_method),
        projection=ctx["cs"].projection,
        solver="semismooth_newton",
        projection_opts={},
        solver_opts=solver_opts,
        adaptive=bool(adaptive),
        h0=h0_use,
        integrator_opts=integrator_opts,
        nl_atol=ctx["nl_atol_contact"],
        nl_rtol=1.0e-6,
        component_slices=ctx["cs"].component_slices,
        verbose=False,
        A=ctx["cs"].A,
        store_fk=False,
    )
    if adaptive:
        solve_kwargs["adaptive_opts"] = dict(ctx["adaptive_opts_contact"])
        solve_kwargs["rtol"] = 1.0
    else:
        solve_kwargs["abort_on_fixed_failure"] = False

    t0 = time.perf_counter()
    out = solve_nivp.solve_nivp(**solve_kwargs)
    solve_wall_time_s = time.perf_counter() - t0
    t_vals, y_vals, h_vals, _, error_estimates = out

    audits = [audit_state(y, ctx) for y in y_vals]
    gap_history = np.vstack([a["gap_phys"] for a in audits])
    slip_history = np.vstack([a["slip_t"] for a in audits])
    pn_history = np.vstack([a["p_n"] for a in audits])
    pt_history = np.vstack([a["p_t"] for a in audits])

    pn_offset_history = np.zeros_like(pn_history)
    pt_offset_history = np.zeros_like(pt_history)
    for i, (tt, yy) in enumerate(zip(t_vals, y_vals)):
        pn_offset_history[i], pt_offset_history[i] = evaluate_contact_offset_history(yy, tt, ctx)

    n_taken = max(len(t_vals) - 1, 0)
    step_success = np.array([bool(item[1]) for item in error_estimates[:n_taken]], dtype=bool)
    step_iterations = np.array([int(item[2]) for item in error_estimates[:n_taken]], dtype=int)
    step_solver_error = np.array(
        [_scalarize_solver_error(item[0]) for item in error_estimates[:n_taken]],
        dtype=float,
    )

    step_true_residual_inf = np.empty(n_taken, dtype=float)
    step_true_residual_rms = np.empty(n_taken, dtype=float)
    for i in range(1, len(t_vals)):
        h_step = float(t_vals[i] - t_vals[i - 1])
        res = _compute_step_residual(y_vals[i - 1], y_vals[i], t_vals[i], h_step, ctx)
        step_true_residual_inf[i - 1] = float(np.max(np.abs(res)))
        step_true_residual_rms[i - 1] = float(np.sqrt(np.mean(res * res)))

    step_sizes = np.diff(np.asarray(t_vals, dtype=float))
    reached_final_time = bool(
        abs(float(t_vals[-1]) - tmax) <= max(1.0e-12, 1.0e-9 * max(1.0, abs(tmax)))
    )
    success = reached_final_time and bool(np.all(step_success))

    return {
        "label": label,
        "success": success,
        "reached_final_time": reached_final_time,
        "adaptive": bool(adaptive),
        "time_method": str(time_method),
        "times": np.asarray(t_vals, dtype=float),
        "times_hours": np.asarray(t_vals, dtype=float) * time_scale,
        "solver_h_values": np.asarray(h_vals, dtype=float),
        "step_sizes": step_sizes,
        "gap_history": gap_history,
        "slip_history": slip_history,
        "pn_history": pn_history,
        "pt_history": pt_history,
        "pn_offset_history": pn_offset_history,
        "pt_offset_history": pt_offset_history,
        "pn_total_history": pn_history + pn_offset_history,
        "pt_total_history": pt_history + pt_offset_history,
        "step_success": step_success,
        "step_iterations": step_iterations,
        "step_solver_error": step_solver_error,
        "step_true_residual_inf": step_true_residual_inf,
        "step_true_residual_rms": step_true_residual_rms,
        "states": np.asarray(y_vals, dtype=float),
        "solve_wall_time_s": float(solve_wall_time_s),
    }


def summarize_history(ctx: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    """Return a compact set of friction-focused diagnostics."""
    gap_history = np.asarray(history["gap_history"], dtype=float)
    slip_history = np.asarray(history["slip_history"], dtype=float)
    pn_total = np.asarray(history["pn_total_history"], dtype=float)
    pt_total = np.asarray(history["pt_total_history"], dtype=float)

    mu_contact = float(ctx["contacts"][0]["mu"]) if ctx.get("contacts") else 0.0

    if mu_contact > 0.0:
        safe_denom = np.maximum(mu_contact * np.abs(pn_total), 1.0e-30)
        coulomb_ratio = np.abs(pt_total) / safe_denom
    else:
        coulomb_ratio = np.zeros_like(pt_total)
    coulomb_excess = np.maximum(np.abs(pt_total) - mu_contact * np.maximum(pn_total, 0.0), 0.0)

    final_pn = pn_total[-1] if pn_total.size else np.zeros(0, dtype=float)
    final_pt = pt_total[-1] if pt_total.size else np.zeros(0, dtype=float)

    return {
        "success": bool(history["success"]),
        "reached_final_time": bool(history["reached_final_time"]),
        "accepted_steps": int(len(history["times"]) - 1),
        "failed_steps": int((~history["step_success"]).sum()) if history["step_success"].size else 0,
        "max_penetration": float(np.max(np.maximum(-gap_history, 0.0))) if gap_history.size else 0.0,
        "max_abs_gap": float(np.max(np.abs(gap_history))) if gap_history.size else 0.0,
        "max_abs_slip": float(np.max(np.abs(slip_history))) if slip_history.size else 0.0,
        "max_abs_pn_total": float(np.max(np.abs(pn_total))) if pn_total.size else 0.0,
        "max_abs_pt_total": float(np.max(np.abs(pt_total))) if pt_total.size else 0.0,
        "max_coulomb_ratio": float(np.max(coulomb_ratio)) if coulomb_ratio.size else 0.0,
        "max_coulomb_excess": float(np.max(coulomb_excess)) if coulomb_excess.size else 0.0,
        "final_mean_pn_total": float(np.mean(final_pn)) if final_pn.size else 0.0,
        "final_mean_pt_total": float(np.mean(final_pt)) if final_pt.size else 0.0,
        "max_step_solver_error": float(np.max(history["step_solver_error"])) if history["step_solver_error"].size else 0.0,
        "max_step_residual_inf": float(np.max(history["step_true_residual_inf"])) if history["step_true_residual_inf"].size else 0.0,
        "mu_friction": mu_contact,
        "normal_prestress": float(ctx["prestress_target"]["normal_prestress"]),
        "fault_prestress_shear_background": (
            float(ctx["prestress_background"]["tau_profile"][0])
            if np.asarray(ctx["prestress_background"]["tau_profile"]).size
            else 0.0
        ),
    }
