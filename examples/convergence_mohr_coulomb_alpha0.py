"""Mesh convergence study for the sliding Mohr-Coulomb fault under prestress
at alpha = 0 (drained / no Biot coupling).  Reproduces the sliding-prestress
section of `embedded_crack_mohr_coulomb_ncp.ipynb` for a sweep of
``CrackMeshBuilder`` N_ELEM values (total target element count), measures the
final-time slip profile, and reports the L2 error against the
Crouch-Starfield / Pollard-Segall analytical reference  delta_max sqrt(1 - s^2).

Designed to run headless on an HPC node.  Usage::

    conda activate fem-env
    python examples/convergence_mohr_coulomb_alpha0.py \\
        --levels 20,40,80,160,320 --out images

Outputs (default ``images/``):
    slip_per_h.png             overlay of FE slip vs analytical for each level
    convergence_l2.png         log-log of L2 error vs h_eff
    convergence_summary.csv    per-level numbers (n_elem, h_eff, err_l2, ...)
    convergence_summary.tex    same as a LaTeX tabular fragment
"""

import argparse
import csv
import dataclasses as dc
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from skfem import ElementTriP1, MeshTri1
from skfem.models.elasticity import lame_parameters

import poroelasticity.cgporoelastostatics as _cgmod
import poroelasticity.nondimensionalise as _nondim_mod
from poroelasticity import CGPoroelastostatics, CrackMeshBuilder, MaterialParams
from poroelasticity.config import ScaleFactors as _ScaleFactors
from poroelasticity.crack_generator import RobustCrackGenerator
from poroelasticity.nondimensionalise import compute_dimensionless_params as _orig_compute_ndp_lib
from poroelasticity.cg_crack_assembler import CrackAssemblyContext, CrackInterfaceAssembler
from poroelasticity.mesh_builder import line_rotated_xy

import solve_nivp
from solve_nivp.projected_radau_contact import build_projected_radau_contact


# =============================================================================
# Wave-clock nondim override (notebook-local; mirrors the notebook's cell)
# =============================================================================
if not hasattr(_cgmod, "_pristine_compute_dimensionless_params"):
    _cgmod._pristine_compute_dimensionless_params = _nondim_mod.compute_dimensionless_params
_orig_compute_ndp = _cgmod._pristine_compute_dimensionless_params


def _wave_clock_ndp(material, scales, crack_model=None,
                    bulk_viscosity=None, dim=2, P_scale_override=None):
    if material.alpha == 0.0:
        return _orig_compute_ndp(
            material, scales, crack_model=crack_model,
            bulk_viscosity=bulk_viscosity, dim=dim, P_scale_override=P_scale_override,
        )
    mat0 = dc.replace(material, alpha=0.0)
    ndp0 = _orig_compute_ndp(
        mat0, scales, crack_model=crack_model,
        bulk_viscosity=bulk_viscosity, dim=dim,
    )
    alpha = material.alpha; beta = material.beta; C = material.C
    L = scales.L; eps = scales.eps; T_w = ndp0.T_scale; Mmod = ndp0.Mmod_scale
    P_scale_new = alpha * eps / beta
    Q_scale_new = beta * P_scale_new / T_w
    C_d_new = C * T_w / (beta * L ** 2) if C > 0 else 0.0
    if crack_model is not None and getattr(crack_model, "T", 0.0):
        T_robin_d_new = crack_model.T * T_w * P_scale_new / (Mmod * L * eps)
    else:
        T_robin_d_new = ndp0.T_robin_d
    return dc.replace(
        ndp0, alpha_d=alpha, C_d=C_d_new,
        P_scale=P_scale_new, Q_scale=Q_scale_new, T_robin_d=T_robin_d_new,
    )


_cgmod.compute_dimensionless_params = _wave_clock_ndp
_compute_ndp = _wave_clock_ndp


# =============================================================================
# USER INPUTS  (PHYSICAL units)
# =============================================================================

# --- Material -------------------------------------------------------------
NU          = 0.3
G_SHEAR     = 10e3
E_YOUNG     = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT  = 0.0                # convergence study runs alpha = 0
BETA_FLUID  = 8.5e-5
ETA_FLUID   = 2.0e-18 / 3600.0
K_PERM      = 1.0e-15 * 1.0e-6
DENSITY_PHYS = 2.7e3
BULK_DAMPING_FRACTION = 9e-1     # convergence study uses heavy damping

# --- Geometry --------------------------------------------------------------
XMIN, XMAX = 0.0, 20.0
YMIN, YMAX = 0.0, 20.0
CRACK_LENGTH = 6.0
CRACK_X0 = 0.5 * (XMIN + XMAX)
CRACK_Y0 = 0.5 * (YMIN + YMAX)

# --- Loading ---------------------------------------------------------------
SIGMA_RIGHT = 10.0
SIGMA_TOP   = 50.0
MU = 0.6

# --- Crack hydraulic interface -------------------------------------------
CRACK_T_ROBIN = 0.0
CRACK_MODEL_PARAMS = {'T': CRACK_T_ROBIN, 'tangential_flow': False}

# --- SBM / Taylor toggles -------------------------------------------------
ELEMENT_TYPE           = 'tri'
MESH_SOURCE            = 'auto'
USE_SBM               = True
CONFORMING_CRACK_MESH = not USE_SBM
INCLUDE_SBM           = False
INCLUDE_TAYLOR        = True
INCLUDE_TAYLOR_TEST   = False
INCLUDE_HESSIAN       = False
TAYLOR_METHOD         = 'nodal'
LUMPED_COUPLING       = 'consistent'

# --- Time -----------------------------------------------------------------
TMAX_PHYS = 30.0
H_PHYS    = 2.8e-1

# --- Nondim choice --------------------------------------------------------
SCALE_L   = 1.0
SCALE_EPS = 1.0e-3

# --- Adaptive integrator constants ---------------------------------------
ADAPTIVE_RTOL  = 1.0e-3
ADAPTIVE_ATOL  = 1.0e-6
ADAPTIVE_H_MIN = 1.0e-5

# --- Lysmer absorbing boundary --------------------------------------------
USE_LYSMER  = True
LYSMER_AP   = 1.0
LYSMER_AS   = 1.0

# Optional directory for per-level diagnostic sidecars.  The script's main()
# sets this; import-based runners can override it before calling run_level().
AUDIT_DIR = None


# =============================================================================
# DERIVED QUANTITIES
# =============================================================================
LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
MATERIAL = MaterialParams(
    mu=MU_LAME, lam=LAM, alpha=ALPHA_BIOT,
    beta=BETA_FLUID, C=K_PERM / ETA_FLUID, rho=DENSITY_PHYS,
)
Mmod = LAM + 2 * MU_LAME
Sigma_scale = Mmod * SCALE_EPS
T_scale = _compute_ndp(MATERIAL, _ScaleFactors(L=SCALE_L, eps=SCALE_EPS)).T_scale
BULK_MU_V  = BULK_DAMPING_FRACTION * MU_LAME * T_scale
BULK_LAM_V = BULK_DAMPING_FRACTION * LAM     * T_scale

TMAX    = TMAX_PHYS / T_scale
H_FIXED = H_PHYS    / T_scale
ADAPTIVE_H_MAX = TMAX / 10.0
H0_ADAPTIVE = H_FIXED

DISP_SCALE = SCALE_L * SCALE_EPS

S1 = max(SIGMA_RIGHT, SIGMA_TOP)
S3 = min(SIGMA_RIGHT, SIGMA_TOP)


# =============================================================================
# Mohr-Coulomb prestress and analytical reference
# =============================================================================
THETA_SLIDING = np.radians(60.0)
CRACK_THETA_SLIDING = np.pi / 2 - THETA_SLIDING


def mohr_tractions(theta):
    sigma_N = (S1 + S3) / 2 + (S1 - S3) / 2 * np.cos(2 * theta)
    tau     = (S1 - S3) / 2 * np.sin(2 * theta)
    return sigma_N, tau


sigma_N_sliding, tau_sliding_val = mohr_tractions(THETA_SLIDING)
delta_tau_mpa = abs(tau_sliding_val) - MU * sigma_N_sliding
crack_half_length = CRACK_LENGTH / 2.0
slip_max_anal = 2.0 * delta_tau_mpa * (1.0 - NU) * crack_half_length / G_SHEAR


def make_bc():
    return {'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}}


def prestress_offsets(sigma_N_phys, tau_phys):
    return sigma_N_phys, tau_phys


def coulomb_cone_warm_start(s0_arr, w0_arr, mu):
    s0_arr = np.asarray(s0_arr, dtype=float).ravel()
    w0_arr = np.asarray(w0_arr, dtype=float).ravel()
    mu = float(mu)
    inv = 1.0 / (1.0 + mu * mu)
    out = np.zeros(2 * s0_arr.size)
    for k in range(s0_arr.size):
        sN, w = float(s0_arr[k]), float(w0_arr[k])
        aw = abs(w)
        if sN >= 0.0 and aw <= mu * sN:
            r_n_p, r_t_p = sN, w
        elif mu * sN + aw <= 0.0:
            r_n_p, r_t_p = 0.0, 0.0
        else:
            r_n_p = max((sN + mu * aw) * inv, 0.0)
            r_t_p = mu * r_n_p * (1.0 if w >= 0 else -1.0)
        out[2 * k]     = r_n_p - sN
        out[2 * k + 1] = r_t_p - w
    return out


def build_lysmer_damping(poro, a_p=LYSMER_AP, a_s=LYSMER_AS):
    from skfem import FacetBasis, BilinearForm
    rho_phys = MATERIAL.rho
    Vp_phys = np.sqrt(E_YOUNG * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU) * rho_phys))
    Vs_phys = np.sqrt(E_YOUNG / (2.0 * (1.0 + NU) * rho_phys))
    T_scale_loc = _compute_ndp(MATERIAL, _ScaleFactors(L=SCALE_L, eps=SCALE_EPS)).T_scale
    Vp_d = Vp_phys * T_scale_loc / SCALE_L
    Vs_d = Vs_phys * T_scale_loc / SCALE_L
    rho_d = float(poro.rho_d)
    c_n = a_p * rho_d * Vp_d
    c_t = a_s * rho_d * Vs_d
    fb = FacetBasis(poro.mesh, poro.basis_u.elem,
                    facets=poro.mesh.boundary_facets(), intorder=poro.intorder)

    @BilinearForm
    def lysmer_form(u, v, w):
        n = w.n
        u_n = u[0] * n[0] + u[1] * n[1]
        v_n = v[0] * n[0] + v[1] * n[1]
        u_tx = u[0] - u_n * n[0]; u_ty = u[1] - u_n * n[1]
        v_tx = v[0] - v_n * n[0]; v_ty = v[1] - v_n * n[1]
        return c_n * u_n * v_n + c_t * (u_tx * v_tx + u_ty * v_ty)

    return lysmer_form.assemble(fb)


def lysmer_descriptor_block(poro, n_phys, n_base, a_p=LYSMER_AP, a_s=LYSMER_AS):
    C_l = build_lysmer_damping(poro, a_p=a_p, a_s=a_s).tocsr()
    Nu = C_l.shape[0]
    if n_phys - n_base != Nu:
        raise ValueError(f"Lysmer block sizing mismatch: Nu={Nu} vs n_phys-n_base={n_phys-n_base}")
    Z_top = sp.csr_matrix((n_base, n_phys))
    Z_left = sp.csr_matrix((Nu, n_base))
    bottom = sp.hstack([Z_left, C_l], format='csr')
    return sp.vstack([Z_top, bottom], format='csr')


def _component_major_from_xy(values_xy):
    values_xy = np.asarray(values_xy, dtype=float)
    return np.concatenate([values_xy[:, 0], values_xy[:, 1]])


def _unique_average_by_s(s, values):
    s = np.asarray(s, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    order = np.argsort(s)
    s_sorted = s[order]
    v_sorted = values[order]
    uniq, inv = np.unique(np.round(s_sorted, 14), return_inverse=True)
    if uniq.size == s_sorted.size:
        return s_sorted, v_sorted
    accum = np.zeros(uniq.size, dtype=float)
    counts = np.zeros(uniq.size, dtype=float)
    np.add.at(accum, inv, v_sorted)
    np.add.at(counts, inv, 1.0)
    return uniq, accum / np.maximum(counts, 1.0)


def _integrated_slip_error(s_param, slip_phys, *, interior=0.95):
    s, slip = _unique_average_by_s(s_param, slip_phys)
    # Add analytical tip values before clipping to the interior window; this
    # gives np.interp enough support even when the first/last contact node is
    # inside the geometric tip.
    s_aug = np.concatenate(([-1.0], s, [1.0]))
    slip_aug = np.concatenate(([0.0], slip, [0.0]))
    order = np.argsort(s_aug)
    s_aug = s_aug[order]
    slip_aug = slip_aug[order]
    s_aug, slip_aug = _unique_average_by_s(s_aug, slip_aug)

    a, b = -float(interior), float(interior)
    mask = (s_aug > a) & (s_aug < b)
    s_eval = np.concatenate(([a], s_aug[mask], [b]))
    slip_eval = np.interp(s_eval, s_aug, slip_aug)
    anal_eval = slip_max_anal * np.sqrt(np.clip(1.0 - s_eval ** 2, 0.0, None))
    err = np.abs(slip_eval) - np.abs(anal_eval)
    # ds is dimensionless; multiplying by crack_half_length converts the line
    # integral to physical arclength.  The relative norm is unchanged.
    err_l2 = float(np.sqrt(crack_half_length * np.trapezoid(err * err, s_eval)))
    ref_l2 = float(np.sqrt(crack_half_length * np.trapezoid(anal_eval * anal_eval, s_eval)))
    return err_l2, ref_l2, err_l2 / max(ref_l2, 1e-30), s_eval, slip_eval, anal_eval


def _shifted_trace_patch_error(poro, C_exp_u_plus, C_exp_u_minus, xhat_dim):
    Nu = poro.basis_u.N
    dof_xy = np.asarray(poro.basis_u.doflocs, dtype=float).T
    comp = np.arange(Nu, dtype=int) % 2

    def field_xy(xy):
        x = xy[:, 0]
        y = xy[:, 1]
        return np.column_stack([
            0.37 + 0.11 * x - 0.07 * y,
            -0.19 + 0.05 * x + 0.13 * y,
        ])

    vals = field_xy(dof_xy)
    u_old = np.where(comp == 0, vals[:, 0], vals[:, 1])
    exact_xy = _component_major_from_xy(field_xy(xhat_dim))

    plus_xy = np.asarray(C_exp_u_plus @ u_old).ravel()
    minus_signed_xy = np.asarray(C_exp_u_minus @ u_old).ravel()
    R_nt = poro._transform_info.get('R_nt')
    if R_nt is not None:
        plus_nt = np.asarray(R_nt @ plus_xy).ravel()
        minus_signed_nt = np.asarray(R_nt @ minus_signed_xy).ravel()
        exact_nt = np.asarray(R_nt @ exact_xy).ravel()
    else:
        plus_nt = plus_xy
        minus_signed_nt = minus_signed_xy
        exact_nt = exact_xy

    n_c = int(poro.n_lambda_q)
    err_plus = plus_nt - exact_nt
    err_minus = minus_signed_nt + exact_nt
    return {
        'patch_plus_n_inf': float(np.linalg.norm(err_plus[:n_c], ord=np.inf)),
        'patch_plus_t_inf': float(np.linalg.norm(err_plus[n_c:], ord=np.inf)),
        'patch_minus_n_inf': float(np.linalg.norm(err_minus[:n_c], ord=np.inf)),
        'patch_minus_t_inf': float(np.linalg.norm(err_minus[n_c:], ord=np.inf)),
        'patch_all_inf': float(max(
            np.linalg.norm(err_plus, ord=np.inf),
            np.linalg.norm(err_minus, ord=np.inf),
        )),
    }


def _interface_metric_nt(poro):
    crack_mats = getattr(poro, '_crack_matrices', {})
    if LUMPED_COUPLING is True:
        d_sigma = np.asarray(crack_mats.get('D_lambda_sigma', []), dtype=float)
        M_xy = sp.diags(d_sigma).tocsr()
    else:
        ctx = CrackAssemblyContext.from_solver_for_crack(poro, 0)
        ndp_i = poro._ndp_per_crack[0]
        assembler = CrackInterfaceAssembler(
            ctx,
            enforcement_type=poro.enforcement_type,
            taylor_method=poro.taylor_method,
            crack_law=poro.crack_law,
            tangential_flow=getattr(poro.crack_models[0], 'tangential_flow', False),
            fracture_perm=getattr(ndp_i, 'fracture_perm_d', 0.0),
            reference_aperture=getattr(ndp_i, 'reference_aperture_d', 0.0),
        )
        M_plus, M_minus = assembler.assemble_interface_mass_u()
        M_xy = (0.5 * (M_plus + M_minus)).tocsr()
    R_nt = poro._transform_info.get('R_nt')
    if R_nt is not None:
        return (R_nt @ M_xy @ R_nt.T).tocsr()
    return M_xy


def _structured_scikit_tri_mesh(n_elem):
    bnd_labels = {
        'left':   lambda x: np.isclose(x[0], XMIN),
        'right':  lambda x: np.isclose(x[0], XMAX),
        'top':    lambda x: np.isclose(x[1], YMAX),
        'bottom': lambda x: np.isclose(x[1], YMIN),
    }
    mesh = MeshTri1.init_tensor(
        np.linspace(XMIN, XMAX, n_elem + 1),
        np.linspace(YMIN, YMAX, n_elem + 1),
    ).with_boundaries(bnd_labels)
    X_crack, dXdt = line_rotated_xy(
        CRACK_THETA_SLIDING, CRACK_X0, CRACK_Y0, CRACK_LENGTH,
    )
    h = min((XMAX - XMIN) / n_elem, (YMAX - YMIN) / n_elem)
    crack = RobustCrackGenerator(
        mesh=mesh, X=X_crack, dXdt=dXdt, d2Xdt2=None,
        verbose=False, align_min=0.01, dist_mult=h,
    )
    crack.finalize()
    return mesh, ElementTriP1(), _cgmod.ElementTriP1Bubble(), crack, h


def _build_mesh_for_level(n_elem_target):
    source = str(MESH_SOURCE).strip().lower()
    if source == 'scikit' and ELEMENT_TYPE == 'tri':
        return _structured_scikit_tri_mesh(n_elem_target)

    use_gmsh = bool(source == 'gmsh' and ELEMENT_TYPE == 'quad')
    builder = CrackMeshBuilder(
        XMIN, XMAX, YMIN, YMAX, n_elem_target,
        crack_theta=CRACK_THETA_SLIDING,
        crack_x0=CRACK_X0, crack_y0=CRACK_Y0,
        crack_length=CRACK_LENGTH,
        element_type=ELEMENT_TYPE, conforming=CONFORMING_CRACK_MESH,
        use_gmsh=use_gmsh, verbose=False,
    )
    return builder.build()


def _cumtrapz(t, rate):
    t = np.asarray(t, dtype=float).ravel()
    rate = np.asarray(rate, dtype=float).ravel()
    out = np.zeros_like(rate)
    for k in range(1, rate.size):
        out[k] = out[k - 1] + 0.5 * (rate[k] + rate[k - 1]) * (t[k] - t[k - 1])
    return out


def _write_contact_time_diagnostics(
    audit_dir, n_elem_target, t_arr, y_arr, *,
    n_phys, n_base, n_c, gap_hist, vjump_hist, r_pert, r_total,
    B_c_dyn, M_cc, W_contact, A_dyn, rhs_jac_dyn_eff, cs,
):
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(t_arr, dtype=float).ravel()
    t_phys = t * T_scale
    y_phys = np.asarray(y_arr[:, :n_phys], dtype=float)
    z_hist = y_phys[:, :n_base]
    v_hist = y_phys[:, n_base:n_phys]
    gap_hist = np.asarray(gap_hist, dtype=float)
    vjump_hist = np.asarray(vjump_hist, dtype=float)
    r_pert = np.asarray(r_pert, dtype=float)
    r_total = np.asarray(r_total, dtype=float)

    r_n = r_total[:, 0::2]
    r_t = r_total[:, 1::2]
    v_n = vjump_hist[:, :n_c]
    v_t = vjump_hist[:, n_c:]
    reaction_rows_local = np.array(
        [row for j in range(n_c) for row in (j, n_c + j)], dtype=int
    )
    v_contact = vjump_hist[:, reaction_rows_local]

    cone_margin_mpa = (MU * r_n - np.abs(r_t)) * Sigma_scale
    min_cone_margin_mpa = np.min(cone_margin_mpa, axis=1)
    max_cone_violation_mpa = np.maximum(0.0, -min_cone_margin_mpa)

    friction_power_nodes = r_t * v_t
    max_positive_friction_power = np.maximum(
        0.0, np.max(friction_power_nodes, axis=1)
    )

    closed = (gap_hist <= 1.0e-8) & (r_n > 1.0e-10)
    normal_power_nodes = r_n * v_n
    max_abs_normal_power_closed = np.zeros_like(t)
    for k in range(t.size):
        if np.any(closed[k]):
            max_abs_normal_power_closed[k] = float(
                np.max(np.abs(normal_power_nodes[k, closed[k]]))
            )

    diag_cc = np.diag(np.asarray(M_cc, dtype=float))
    if diag_cc.size == 2 * n_c:
        omega_node = np.asarray(diag_cc[0::2], dtype=float)
    else:
        omega_node = np.ones(n_c, dtype=float)
    omega_node = np.maximum(omega_node, 0.0)

    B_vel = (
        B_c_dyn[n_base:n_phys, :].tocsr()
        if sp.issparse(B_c_dyn)
        else B_c_dyn[n_base:n_phys, :]
    )
    Wc = W_contact.tocsr() if sp.issparse(W_contact) else sp.csr_matrix(W_contact)
    n_idx = np.arange(0, 2 * n_c, 2)
    t_idx = np.arange(1, 2 * n_c, 2)
    W_nn = Wc[n_idx, :][:, n_idx].tocsr()
    W_tt = Wc[t_idx, :][:, t_idx].tocsr()

    p_contact = np.zeros_like(t)
    p_contact_pert = np.zeros_like(t)
    p_contact_metric = np.zeros_like(t)
    p_normal_total = np.zeros_like(t)
    p_tangent_total = np.zeros_like(t)
    p_tangent_pert = np.zeros_like(t)
    p_tangent_offset = np.zeros_like(t)
    for k in range(t.size):
        p_contact[k] = float(v_hist[k] @ (B_vel @ r_total[k]))
        p_contact_pert[k] = float(v_hist[k] @ (B_vel @ r_pert[k]))
        p_contact_metric[k] = 0.5 * float(v_contact[k] @ (Wc @ r_total[k]))
        p_normal_total[k] = 0.5 * float(v_n[k] @ (W_nn @ r_n[k]))
        p_tangent_total[k] = 0.5 * float(v_t[k] @ (W_tt @ r_t[k]))
        p_tangent_pert[k] = 0.5 * float(v_t[k] @ (W_tt @ r_pert[k, 1::2]))
        p_tangent_offset[k] = p_tangent_total[k] - p_tangent_pert[k]
    p_offset = p_contact - p_contact_pert
    p_cross_metric = p_contact_metric - p_normal_total - p_tangent_total

    # Nodal/lumped Coulomb quadrature.  This is useful as a physics diagnostic,
    # but it is not the exact discrete work conjugate of B when a consistent
    # interface mass is used.
    d_fric_lumped_rate = np.sum(
        0.5 * MU * np.maximum(r_n, 0.0) * np.abs(v_t) * omega_node[None, :],
        axis=1,
    )
    p_fric_lumped = np.sum(
        0.5 * r_t * v_t * omega_node[None, :],
        axis=1,
    )
    # Exact discrete frictional dissipation rate.  A negative value means the
    # tangential contact work is injecting energy into the mechanics.
    d_fric_rate = -p_tangent_total
    friction_injection_rate = np.maximum(0.0, p_tangent_total)

    A0 = A_dyn.tocsr() if sp.issparse(A_dyn) else sp.csr_matrix(A_dyn)
    J0_raw = rhs_jac_dyn_eff(float(t[0]), y_phys[0])
    J0 = J0_raw.tocsr() if sp.issparse(J0_raw) else sp.csr_matrix(J0_raw)
    mom_rows = slice(n_base, n_phys)
    base_cols = slice(0, n_base)
    vel_cols = slice(n_base, n_phys)
    K_op = -(J0[mom_rows, :][:, base_cols]).tocsr()
    C_op = -(J0[mom_rows, :][:, vel_cols]).tocsr()
    M_op = (A0[mom_rows, :][:, vel_cols]).tocsr()

    KE = np.zeros_like(t)
    p_elastic = np.zeros_like(t)
    d_vis_rate = np.zeros_like(t)
    for k in range(t.size):
        z_k = z_hist[k]
        v_k = v_hist[k]
        KE[k] = 0.5 * float(v_k @ (M_op @ v_k))
        p_elastic[k] = float(v_k @ (K_op @ z_k))
        d_vis_rate[k] = float(v_k @ (C_op @ v_k))

    d_fric_int = _cumtrapz(t, d_fric_rate)
    d_fric_lumped_int = _cumtrapz(t, d_fric_lumped_rate)
    w_contact = _cumtrapz(t, p_contact)
    w_contact_pert = _cumtrapz(t, p_contact_pert)
    w_offset = _cumtrapz(t, p_offset)
    w_tangent_total = _cumtrapz(t, p_tangent_total)
    w_tangent_pert = _cumtrapz(t, p_tangent_pert)
    w_tangent_offset = _cumtrapz(t, p_tangent_offset)
    w_normal_total = _cumtrapz(t, p_normal_total)
    w_cross_metric = _cumtrapz(t, p_cross_metric)
    w_el = _cumtrapz(t, p_elastic)
    d_vis_int = _cumtrapz(t, d_vis_rate)
    bulk_residual_pert = (KE - KE[0]) + w_el + d_vis_int - w_contact_pert
    bulk_residual_total = (KE - KE[0]) + w_el + d_vis_int - w_contact
    interface_residual = d_fric_int + w_tangent_total
    coulomb_lumped_residual = d_fric_lumped_int + w_tangent_total

    endpoint_res_inf = np.zeros_like(t)
    endpoint_res_l2 = np.zeros_like(t)
    endpoint_force_res_inf = np.zeros_like(t)
    law = cs.projected_radau_contact.contact_law
    prc = cs.projected_radau_contact
    for k in range(t.size):
        force_res = np.zeros(2 * n_c, dtype=float)
        final_v_contact = vjump_hist[k, reaction_rows_local]
        for j in range(n_c):
            sl = slice(2 * j, 2 * j + 2)
            v_blk = final_v_contact[sl]
            r_blk = r_total[k, sl]
            f_blk, *_ = law.residual_and_jac(
                float(v_blk[0]), v_blk, r_blk, MU, 1.0, 1.0,
            )
            force_res[sl] = f_blk
        endpoint_force_res_inf[k] = float(np.linalg.norm(force_res, ord=np.inf))

        if k == 0:
            endpoint_res_inf[k] = np.nan
            endpoint_res_l2[k] = np.nan
            continue
        h_k = float(t[k] - t[k - 1])
        if h_k <= 0.0:
            endpoint_res_inf[k] = np.nan
            endpoint_res_l2[k] = np.nan
            continue
        try:
            total_eff = h_k * r_total[k]
            f_end = prc._endpoint_contact_residual(
                y_phys[k], y_phys[k - 1], total_eff, float(t[k]), h_k,
            )
            endpoint_res_inf[k] = float(np.linalg.norm(f_end, ord=np.inf))
            endpoint_res_l2[k] = float(np.linalg.norm(f_end))
        except Exception:
            endpoint_res_inf[k] = np.nan
            endpoint_res_l2[k] = np.nan

    csv_path = audit_dir / f'contact_time_diagnostics_N{n_elem_target}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'step', 't_dless', 't_phys_s', 'dt_dless',
                'min_cone_margin_MPa', 'max_cone_violation_MPa',
                'max_positive_friction_power',
                'max_abs_normal_power_closed',
                'p_contact_total', 'p_contact_pert', 'p_offset',
                'p_contact_metric', 'p_contact_metric_mismatch',
                'p_normal_total', 'p_tangent_total', 'p_tangent_pert',
                'p_tangent_offset', 'p_cross_metric',
                'p_friction_lumped', 'd_fric_rate', 'd_fric_lumped_rate',
                'friction_injection_rate', 'D_fric', 'D_fric_lumped',
                'W_contact_total', 'W_contact_pert', 'W_offset_external',
                'W_tangent_total', 'W_tangent_pert', 'W_tangent_offset',
                'W_normal_total', 'W_cross_metric',
                'KE', 'W_elastic', 'D_vis', 'bulk_residual_pert',
                'bulk_residual_total',
                'interface_residual_D_fric_plus_W_tangent',
                'coulomb_lumped_residual',
                'endpoint_res_inf', 'endpoint_res_l2',
                'endpoint_force_res_inf',
                'max_abs_vt', 'min_gap',
            ],
        )
        writer.writeheader()
        for k in range(t.size):
            dt_k = 0.0 if k == 0 else float(t[k] - t[k - 1])
            writer.writerow({
                'step': k,
                't_dless': f'{t[k]:.16e}',
                't_phys_s': f'{t_phys[k]:.16e}',
                'dt_dless': f'{dt_k:.16e}',
                'min_cone_margin_MPa': f'{min_cone_margin_mpa[k]:.16e}',
                'max_cone_violation_MPa': f'{max_cone_violation_mpa[k]:.16e}',
                'max_positive_friction_power': f'{max_positive_friction_power[k]:.16e}',
                'max_abs_normal_power_closed': f'{max_abs_normal_power_closed[k]:.16e}',
                'p_contact_total': f'{p_contact[k]:.16e}',
                'p_contact_pert': f'{p_contact_pert[k]:.16e}',
                'p_offset': f'{p_offset[k]:.16e}',
                'p_contact_metric': f'{p_contact_metric[k]:.16e}',
                'p_contact_metric_mismatch': f'{(p_contact[k] - p_contact_metric[k]):.16e}',
                'p_normal_total': f'{p_normal_total[k]:.16e}',
                'p_tangent_total': f'{p_tangent_total[k]:.16e}',
                'p_tangent_pert': f'{p_tangent_pert[k]:.16e}',
                'p_tangent_offset': f'{p_tangent_offset[k]:.16e}',
                'p_cross_metric': f'{p_cross_metric[k]:.16e}',
                'p_friction_lumped': f'{p_fric_lumped[k]:.16e}',
                'd_fric_rate': f'{d_fric_rate[k]:.16e}',
                'd_fric_lumped_rate': f'{d_fric_lumped_rate[k]:.16e}',
                'friction_injection_rate': f'{friction_injection_rate[k]:.16e}',
                'D_fric': f'{d_fric_int[k]:.16e}',
                'D_fric_lumped': f'{d_fric_lumped_int[k]:.16e}',
                'W_contact_total': f'{w_contact[k]:.16e}',
                'W_contact_pert': f'{w_contact_pert[k]:.16e}',
                'W_offset_external': f'{w_offset[k]:.16e}',
                'W_tangent_total': f'{w_tangent_total[k]:.16e}',
                'W_tangent_pert': f'{w_tangent_pert[k]:.16e}',
                'W_tangent_offset': f'{w_tangent_offset[k]:.16e}',
                'W_normal_total': f'{w_normal_total[k]:.16e}',
                'W_cross_metric': f'{w_cross_metric[k]:.16e}',
                'KE': f'{KE[k]:.16e}',
                'W_elastic': f'{w_el[k]:.16e}',
                'D_vis': f'{d_vis_int[k]:.16e}',
                'bulk_residual_pert': f'{bulk_residual_pert[k]:.16e}',
                'bulk_residual_total': f'{bulk_residual_total[k]:.16e}',
                'interface_residual_D_fric_plus_W_tangent': f'{interface_residual[k]:.16e}',
                'coulomb_lumped_residual': f'{coulomb_lumped_residual[k]:.16e}',
                'endpoint_res_inf': f'{endpoint_res_inf[k]:.16e}',
                'endpoint_res_l2': f'{endpoint_res_l2[k]:.16e}',
                'endpoint_force_res_inf': f'{endpoint_force_res_inf[k]:.16e}',
                'max_abs_vt': f'{np.max(np.abs(v_t[k])):.16e}',
                'min_gap': f'{np.min(gap_hist[k]):.16e}',
            })

    fig, axes = plt.subplots(3, 2, figsize=(11.0, 9.0), sharex=True)
    axes = axes.ravel()
    axes[0].plot(t_phys, min_cone_margin_mpa, color='tab:blue')
    axes[0].axhline(0.0, color='0.25', lw=0.8)
    axes[0].set_ylabel('min cone margin [MPa]')
    axes[0].grid(alpha=0.3)

    eps_plot = 1.0e-300
    axes[1].semilogy(
        t_phys, np.maximum(friction_injection_rate, eps_plot),
        color='tab:red',
    )
    axes[1].set_ylabel('positive tangent power')
    axes[1].grid(alpha=0.3, which='both')

    axes[2].semilogy(
        t_phys, np.maximum(np.abs(p_normal_total), eps_plot),
        color='tab:orange',
    )
    axes[2].set_ylabel(r'$|P_\mathrm{normal}|$')
    axes[2].grid(alpha=0.3, which='both')

    axes[3].plot(t_phys, d_fric_int, color='tab:red',
                 label=r'$D_\mathrm{fric}$ exact')
    axes[3].plot(t_phys, d_fric_lumped_int, color='tab:pink', ls=':',
                 label=r'$D_\mathrm{fric}$ lumped')
    axes[3].plot(t_phys, -w_tangent_total, color='tab:purple', ls='--',
                 label=r'$-W_\mathrm{tangent,total}$')
    axes[3].plot(t_phys, -w_contact_pert, color='0.25', ls=':',
                 label=r'$-W_\mathrm{contact,pert}$')
    axes[3].set_ylabel('cumulative work')
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    axes[4].plot(t_phys, bulk_residual_pert, color='tab:blue',
                 label='bulk residual (pert)')
    axes[4].plot(t_phys, interface_residual, color='tab:green',
                 label='exact interface residual')
    axes[4].plot(t_phys, coulomb_lumped_residual, color='tab:orange', ls='--',
                 label='lumped Coulomb mismatch')
    axes[4].axhline(0.0, color='0.25', lw=0.8)
    axes[4].set_ylabel('energy residual')
    axes[4].set_xlabel('t [s]')
    axes[4].legend(fontsize=8)
    axes[4].grid(alpha=0.3)

    axes[5].semilogy(
        t_phys, np.where(
            np.isfinite(endpoint_res_inf),
            np.maximum(endpoint_res_inf, eps_plot),
            np.nan,
        ),
        color='tab:blue', label='endpoint residual',
    )
    axes[5].semilogy(
        t_phys, np.maximum(endpoint_force_res_inf, eps_plot),
        color='tab:cyan', ls='--', label='force-scale law residual',
    )
    axes[5].set_ylabel('contact residual inf-norm')
    axes[5].set_xlabel('t [s]')
    axes[5].legend(fontsize=8)
    axes[5].grid(alpha=0.3, which='both')

    fig.suptitle(f'Contact time diagnostics, N_ELEM={n_elem_target}', y=0.995)
    fig.tight_layout()
    png_path = audit_dir / f'contact_time_diagnostics_N{n_elem_target}.png'
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    finite_endpoint = endpoint_res_inf[np.isfinite(endpoint_res_inf)]
    peak_contact = max(float(np.max(np.abs(w_contact))), 1.0e-30)
    peak_pert = max(float(np.max(np.abs(w_contact_pert))), 1.0e-30)
    peak_tangent = max(float(np.max(np.abs(w_tangent_total))), 1.0e-30)
    return {
        'contact_time_diag_csv': str(csv_path),
        'contact_time_diag_png': str(png_path),
        'time_min_cone_margin_MPa': float(np.min(min_cone_margin_mpa)),
        'time_max_cone_violation_MPa': float(np.max(max_cone_violation_mpa)),
        'time_max_positive_friction_power': float(np.max(max_positive_friction_power)),
        'time_max_abs_normal_power_closed': float(np.max(max_abs_normal_power_closed)),
        'time_max_positive_tangent_power': float(np.max(friction_injection_rate)),
        'time_min_D_fric_rate': float(np.min(d_fric_rate)),
        'time_final_D_fric': float(d_fric_int[-1]),
        'time_final_D_fric_lumped': float(d_fric_lumped_int[-1]),
        'time_final_W_contact_total': float(w_contact[-1]),
        'time_final_W_contact_pert': float(w_contact_pert[-1]),
        'time_final_W_offset_external': float(w_offset[-1]),
        'time_final_W_tangent_total': float(w_tangent_total[-1]),
        'time_final_W_tangent_pert': float(w_tangent_pert[-1]),
        'time_final_W_tangent_offset': float(w_tangent_offset[-1]),
        'time_final_W_normal_total': float(w_normal_total[-1]),
        'time_final_W_cross_metric': float(w_cross_metric[-1]),
        'time_final_KE': float(KE[-1]),
        'time_final_W_elastic': float(w_el[-1]),
        'time_final_D_vis': float(d_vis_int[-1]),
        'time_final_bulk_residual_pert': float(bulk_residual_pert[-1]),
        'time_peak_bulk_residual_pert': float(np.max(np.abs(bulk_residual_pert))),
        'time_peak_bulk_residual_pert_rel': float(
            np.max(np.abs(bulk_residual_pert)) / peak_pert
        ),
        'time_final_bulk_residual_total': float(bulk_residual_total[-1]),
        'time_final_interface_residual': float(interface_residual[-1]),
        'time_peak_interface_residual': float(np.max(np.abs(interface_residual))),
        'time_peak_interface_residual_rel': float(
            np.max(np.abs(interface_residual)) / peak_tangent
        ),
        'time_final_coulomb_lumped_residual': float(coulomb_lumped_residual[-1]),
        'time_peak_coulomb_lumped_residual': float(
            np.max(np.abs(coulomb_lumped_residual))
        ),
        'time_peak_coulomb_lumped_residual_rel': float(
            np.max(np.abs(coulomb_lumped_residual)) / peak_tangent
        ),
        'time_max_contact_metric_mismatch': float(
            np.max(np.abs(p_contact - p_contact_metric))
        ),
        'time_peak_contact_total': peak_contact,
        'time_max_endpoint_res_inf': (
            float(np.max(finite_endpoint)) if finite_endpoint.size else float('nan')
        ),
        'time_max_endpoint_force_res_inf': float(np.max(endpoint_force_res_inf)),
        'time_final_endpoint_force_res_inf': float(endpoint_force_res_inf[-1]),
    }


# =============================================================================
# Per-level run
# =============================================================================

@dc.dataclass
class LevelResult:
    n_elem_target: int
    n_elem_actual: int
    n_c: int
    h_eff: float
    err_l2: float
    ref_l2: float
    rel_l2: float
    err_node_l2: float
    ref_node_l2: float
    rel_node_l2: float
    t_build: float
    t_solve: float
    t_total: float
    n_acc: int
    n_rej: int
    gap_shift_rel: float
    adjoint_rel: float
    delassus_sym_rel: float
    delassus_min_eig: float
    patch_all_inf: float
    soc_res_inf: float
    max_friction_power_pos: float
    n_open: int
    n_sticking: int
    n_sliding: int
    n_other_active: int
    min_gap: float
    final_cone_margin_abs: float
    contact_ok: bool
    s_param: np.ndarray            # (n_c,) sorted along fault
    slip_phys: np.ndarray          # (n_c,) physical metres
    s_param_with_tips: np.ndarray  # (n_c+2,)
    slip_with_tips: np.ndarray     # (n_c+2,)


def run_level(n_elem_target: int) -> LevelResult:
    print(f"\n{'='*72}\n[level] N_ELEM = {n_elem_target}\n{'='*72}", flush=True)
    t_build_start = time.perf_counter()

    mesh, el_p, el_u, crack, h_mesh = _build_mesh_for_level(n_elem_target)
    print(f"  mesh: {mesh.nelements} elements, {mesh.p.shape[1]} vertices, h~{h_mesh:.4f}",
          flush=True)

    poro = CGPoroelastostatics(
        mesh=mesh, element_p=el_p, element_u=el_u,
        params=PARAMS, material=MATERIAL,
        crack=crack, model_params=CRACK_MODEL_PARAMS,
        intorder=6, scales=(SCALE_L, SCALE_EPS), bc=make_bc(),
        P_scale=None, verbose=False, free_memory=False,
        enforcement_type='nodal', apply_transform=True,
        include_taylor=INCLUDE_TAYLOR,
        include_taylor_test=INCLUDE_TAYLOR_TEST,
        include_hessian=INCLUDE_HESSIAN, include_sbm=INCLUDE_SBM,
        taylor_method=TAYLOR_METHOD, lumped_coupling=LUMPED_COUPLING,
        crack_law='nonsmooth', rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': BULK_MU_V, 'lam_v': BULK_LAM_V},
    )
    poro.strip_multiplier_dynamics()
    projection, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta)

    A_dyn = dyn['A']; rhs_dyn = dyn['rhs']; rhs_jac_dyn = dyn['rhs_jac']
    n_base = dyn['n_base']; comp_slices = dyn['component_slices']
    Np = poro.basis_p.N; Nu = poro.basis_u.N

    info = poro._transform_info
    n_c = int(poro.n_lambda_q)
    lam_s_sl = meta['lam_s_sl']
    off_jmpu = info['off_jmpu']; n_iu = info['n_intf_u']
    off_avgls = info['off_avgls']
    off_jmpls = info['off_jmpls']; n_lsig = info['n_lam_sig']
    jmpu_n_idx = np.arange(off_jmpu, off_jmpu + n_c)
    jmpu_t_idx = np.arange(off_jmpu + n_c, off_jmpu + n_iu)
    jmpu_all = np.concatenate([jmpu_n_idx, jmpu_t_idx])
    jmpls_cols = np.arange(off_jmpls, off_jmpls + n_lsig)
    n_extract = 2 * n_c

    # SIM-consistent contact kinematics: the Coulomb law must see the shifted
    # true-crack trace u_hat = u_tilde + Delta_j u_,j, not the surrogate nodal
    # jump stored directly in the transformed [[u]] block.
    R_v = dyn.get('R_v')
    crack_mats = getattr(poro, '_crack_matrices', {})
    C_exp_u_plus = crack_mats.get('C_exp_u_plus')
    C_exp_u_minus = crack_mats.get('C_exp_u_minus')
    if C_exp_u_plus is None or C_exp_u_minus is None:
        raise RuntimeError('Missing shifted displacement trace operators C_exp_u_plus/minus')
    C_exp_u_jump_xy = (C_exp_u_plus + C_exp_u_minus).tocsr()
    if info.get('rotated_to_nt', False):
        C_exp_u_jump_nt = (info['R_nt'] @ C_exp_u_jump_xy).tocsr()
    else:
        C_exp_u_jump_nt = C_exp_u_jump_xy

    contact_coords = np.asarray(
        poro._nodal_interface_info["lambda_coords"], dtype=float,
    )[:, :n_c]
    xhat_dim, t_true, delta_dim = crack.projector.project_many(contact_coords.T * SCALE_L)
    s_true = 2.0 * np.asarray(t_true, dtype=float).ravel() - 1.0
    patch_diag = _shifted_trace_patch_error(
        poro, C_exp_u_plus, C_exp_u_minus, np.asarray(xhat_dim, dtype=float),
    )

    T_inv_u_static = poro._T_inv[Np:Np + Nu, :].tocsr()
    gap_u_hat_from_z = (C_exp_u_jump_nt @ T_inv_u_static).tocsr()
    gap_u_raw_from_z = sp.csr_matrix(
        (np.ones(n_extract), (np.arange(n_extract), jmpu_all)),
        shape=(n_extract, n_base),
    )
    gap_shift_rel = float(
        sp.linalg.norm(gap_u_hat_from_z - gap_u_raw_from_z)
        / max(1.0, sp.linalg.norm(gap_u_hat_from_z))
    )
    if R_v is not None:
        D = (C_exp_u_jump_nt @ R_v.T).tocsr()
    else:
        D = C_exp_u_jump_nt.tocsr()

    A_csr = poro.A.tocsr()
    B_u = (A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
    if R_v is not None:
        B_u = (R_v @ B_u).tocsr()

    dirichlet_local = np.array(
        [d - Np for d in np.asarray(getattr(poro, '_dirichlet_dof_set', []), dtype=int)
         if Np <= d < Np + Nu], dtype=int,
    )
    if dirichlet_local.size:
        B_u = B_u.tolil(); B_u[dirichlet_local, :] = 0.0; B_u = B_u.tocsr()

    perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
    B_u_perm = B_u[:, perm].tocsr()
    reaction_state_indices = jmpls_cols[perm]
    reaction_rows = np.array([row for k in range(n_c) for row in (k, n_c + k)], dtype=int)
    D_contact = D[reaction_rows, :].tocsr()
    W_nt = _interface_metric_nt(poro)
    W_contact = W_nt[reaction_rows, :][:, reaction_rows].tocsr()
    B_expected = (0.5 * (D_contact.T @ W_contact)).tocsr()
    adjoint_rel = float(
        sp.linalg.norm(B_u_perm - B_expected) / max(1.0, sp.linalg.norm(B_u_perm))
    )

    contacts = [
        {'vel_normal_idx': k, 'vel_tangential_idx': [n_c + k], 'mu': MU, 'e': 0.0}
        for k in range(n_c)
    ]

    n_phys = A_dyn.shape[0]
    n_aug_phys = n_phys
    n_vel = n_aug_phys - n_base
    assert gap_u_hat_from_z.shape == (n_extract, n_base)
    assert D.shape == (n_extract, n_vel)
    gap_extract_dyn = sp.hstack(
        [gap_u_hat_from_z, sp.csr_matrix((n_extract, n_vel))],
        format='csr',
    )
    vel_extract_dyn = sp.hstack(
        [sp.csr_matrix((n_extract, n_base)), D], format='csr',
    )
    B_c_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm], format='csr',
    )
    B_n_dyn = B_c_dyn[:, 0::2]
    M_cc = (D_contact @ B_u_perm).toarray()
    delassus_sym_rel = float(
        np.linalg.norm(M_cc - M_cc.T) / max(1.0, np.linalg.norm(M_cc))
    )
    try:
        delassus_min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (M_cc + M_cc.T))))
    except np.linalg.LinAlgError:
        delassus_min_eig = float('nan')

    # Taylor-shifted average pressure trace at the actual crack face.
    T_inv_p_static = poro._T_inv[0:Np, :].tocsr()
    P_old_from_dyn = sp.hstack(
        [T_inv_p_static,
         sp.csr_matrix((Np, n_aug_phys - T_inv_p_static.shape[1]))],
        format='csr',
    )
    p_gamma_extract_dyn = (
        0.5 * (poro.C_exp_p_plus + poro.C_exp_p_minus) @ P_old_from_dyn
    ).tocsr()
    pressure_normal_correction = -ALPHA_BIOT * (B_n_dyn @ p_gamma_extract_dyn)
    pressure_normal_correction.eliminate_zeros()

    if USE_LYSMER:
        C_lysmer = lysmer_descriptor_block(poro, n_phys, n_base,
                                            a_p=LYSMER_AP, a_s=LYSMER_AS)
    else:
        C_lysmer = None

    def rhs_dyn_eff(t, y, *extra):
        out = rhs_dyn(t, y, *extra) + pressure_normal_correction @ y
        if C_lysmer is not None:
            out = out - C_lysmer @ y
        return out

    def rhs_jac_dyn_eff(t, y, *extra):
        J = rhs_jac_dyn(t, y, *extra) + pressure_normal_correction
        if C_lysmer is not None:
            J = J - C_lysmer
        return J

    sN_phys, tau_phys = prestress_offsets(sigma_N_sliding, tau_sliding_val)
    s0_val = np.full(n_c, sN_phys / Sigma_scale)
    w0_val = np.full(n_c, tau_phys / Sigma_scale)

    def get_s0(_y): return s0_val
    def get_w0(_y, k): return np.array([w0_val[int(k)]])

    flux_constraint = meta['constraints'][0]
    avg_lam_s_sl = slice(off_avgls, off_avgls + n_lsig)
    zero_avg_lam_s = {
        'g':     lambda zf, *_a, _n=n_lsig: np.zeros(_n),
        'dg_dy': lambda zf, *_a, _n=n_lsig: np.zeros((_n, _n)),
        'y_slice': avg_lam_s_sl, 'q_slice': avg_lam_s_sl,
    }

    cs = build_projected_radau_contact(
        A_dyn, rhs_dyn_eff, np.zeros(A_dyn.shape[0]),
        contacts=contacts,
        C_extract=gap_extract_dyn, D_extract=vel_extract_dyn, B=B_c_dyn,
        constraints=[flux_constraint, zero_avg_lam_s],
        component_slices=comp_slices, rhs_jac=rhs_jac_dyn_eff,
        get_s0=get_s0, get_w0=get_w0,
        normal_r='auto', friction_r='auto',
        endpoint_inactive_handling='natural_map',
        contact_law='soc_fb',
        reported_reaction_units='force', auto_rho_strategy='delassus',
        reaction_state_indices=reaction_state_indices,
        reaction_state_to_reported_scale=0.5,
        mask_reaction_state_in_smooth_rhs=True,
    )

    n_aug_sliding = cs.y0.size
    nl_atol_sliding = np.full(n_aug_sliding, 1e-8)
    nl_rtol_sliding = np.full(n_aug_sliding, 1e-6)
    reaction_state_idx = getattr(cs, 'reaction_state_indices', None)
    if reaction_state_idx is not None:
        nl_atol_sliding[reaction_state_idx] = 2e-12
        nl_rtol_sliding[reaction_state_idx] = 0.0
    else:
        nl_atol_sliding[n_phys:] = 1e-12
        nl_rtol_sliding[n_phys:] = 0.0

    solver_opts = dict(cs.solver_opts)
    solver_opts.update(
        tol=1e-8, max_iter=50, rhs_jac=cs.rhs_jac,
        linear_solver='petsc',
        petsc_options={
            'ksp_type': 'preonly', 'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
        petsc_reuse_steps=50,
    )
    integrator_opts = dict(cs.integrator_opts)
    integrator_opts.update({'stages': 2, 'use_coupled_newton': True})

    t_build = time.perf_counter() - t_build_start
    print(f"  build complete: n_base={n_base}, n_c={n_c}, n_aug={n_aug_sliding}, "
          f"shift_rel={gap_shift_rel:.2e}, adjoint={adjoint_rel:.2e}, "
          f"patch={patch_diag['patch_all_inf']:.2e}, t_build={t_build:.2f}s", flush=True)

    t_solve_start = time.perf_counter()
    y0_init = cs.y0.copy()
    warm_r = coulomb_cone_warm_start(s0_val, w0_val, MU)
    if reaction_state_idx is not None:
        y0_init[reaction_state_idx] = (
            warm_r / cs.reaction_state_to_reported_scale
        )
    else:
        y0_init[-2 * n_c:] = warm_r

    skip_error_indices = [0]
    if reaction_state_idx is not None:
        skip_error_indices.extend(
            i for i in range(8, 13) if i < len(cs.component_slices)
        )
    else:
        skip_error_indices.append(len(cs.component_slices) - 1)

    (
        t_arr, y_arr, h_arr, fk_arr, info_solve, attempts
    ) = solve_nivp.solve_ivp_ns(
        fun=cs.rhs, t_span=(0.0, float(TMAX)), y0=y0_init,
        method='radau_iia', integrator_opts=integrator_opts,
        projection=cs.projection, solver='semismooth_newton',
        projection_opts={'component_slices': cs.component_slices},
        solver_opts=solver_opts,
        adaptive=True, h0=H0_ADAPTIVE,
        rtol=ADAPTIVE_RTOL, atol=ADAPTIVE_ATOL,
        skip_error_indices=skip_error_indices,
        active_set_filter=False, return_attempts=True,
        adaptive_opts=dict(
            h0=H0_ADAPTIVE, h_min=ADAPTIVE_H_MIN,
            h_max=ADAPTIVE_H_MAX, h_up=2.0, h_down=0.5, method_order=1,
        ),
        nl_atol=nl_atol_sliding, nl_rtol=nl_rtol_sliding,
        component_slices=cs.component_slices, verbose=False,
        A=cs.A, dae_var_weight='auto',
    )
    t_solve = time.perf_counter() - t_solve_start

    acc = np.asarray(attempts, dtype=bool) if attempts is not None else None
    n_acc = int(acc.sum()) if acc is not None else len(t_arr) - 1
    n_rej = int((~acc).sum()) if acc is not None else 0
    print(f"  solve complete: t_solve={t_solve:.2f}s, n_acc={n_acc}, n_rej={n_rej}, "
          f"steps={len(t_arr)}, t_final={t_arr[-1]:.4e}", flush=True)

    # --- Shifted slip extraction at final time ------------------------------
    jump_hist = np.asarray((gap_extract_dyn @ y_arr[:, :n_phys].T).T, dtype=float)
    gap_hist = jump_hist[:, :n_c]
    slip_hist = jump_hist[:, n_c:]
    vjump_hist = np.asarray((vel_extract_dyn @ y_arr[:, :n_phys].T).T, dtype=float)
    final_slip_dless = np.asarray(slip_hist[-1], dtype=float)
    slip_phys = final_slip_dless * DISP_SCALE

    r_pert = cs.reaction_history(y_arr)
    offsets = np.zeros(2 * n_c)
    offsets[0::2] = s0_val
    offsets[1::2] = w0_val
    r_total = r_pert + offsets[None, :]
    r_n = r_total[:, 0::2] * Sigma_scale
    r_t = r_total[:, 1::2] * Sigma_scale
    cone_margin = MU * r_n - np.abs(r_t)
    min_gap = float(np.min(gap_hist))
    final_cone_margin_abs = float(np.max(np.abs(cone_margin[-1])))

    final_vjump = vjump_hist[-1]
    final_v_contact = final_vjump[reaction_rows]
    final_r_total_nd = r_total[-1]
    law = cs.projected_radau_contact.contact_law
    soc_res = np.zeros(2 * n_c, dtype=float)
    classes = []
    cone_margin_nd = MU * final_r_total_nd[0::2] - np.abs(final_r_total_nd[1::2])
    friction_power = final_r_total_nd[1::2] * final_vjump[n_c:]
    gap_tol = 1.0e-8
    v_tol = 1.0e-8
    r_tol = 1.0e-10
    cone_tol_nd = 1.0e-8
    for k in range(n_c):
        sl = slice(2 * k, 2 * k + 2)
        v_blk = final_v_contact[sl]
        r_blk = final_r_total_nd[sl]
        f_blk, *_ = law.residual_and_jac(
            float(v_blk[0]), v_blk, r_blk, MU, 1.0, 1.0,
        )
        soc_res[sl] = f_blk
        gap_k = float(gap_hist[-1, k])
        rn_k = float(r_blk[0])
        rt_k = float(r_blk[1])
        vt_k = float(v_blk[1])
        if gap_k > gap_tol and rn_k <= r_tol:
            cls = 'open'
        elif abs(vt_k) > v_tol and abs(abs(rt_k) - MU * rn_k) <= cone_tol_nd:
            cls = 'sliding'
        elif abs(vt_k) <= v_tol and abs(rt_k) < MU * rn_k - cone_tol_nd:
            cls = 'sticking'
        else:
            cls = 'other_active'
        classes.append(cls)
    soc_res_inf = float(np.linalg.norm(soc_res, ord=np.inf))
    max_friction_power_pos = float(max(0.0, np.max(friction_power)))
    n_open = int(sum(c == 'open' for c in classes))
    n_sticking = int(sum(c == 'sticking' for c in classes))
    n_sliding = int(sum(c == 'sliding' for c in classes))
    n_other_active = int(sum(c == 'other_active' for c in classes))
    contact_ok = bool(
        min_gap >= -1e-8
        and final_cone_margin_abs <= 1e-2
        and soc_res_inf <= 1e-7
        and max_friction_power_pos <= 1e-9
    )

    time_diag_summary = {}
    if AUDIT_DIR is not None:
        time_diag_summary = _write_contact_time_diagnostics(
            AUDIT_DIR, n_elem_target, t_arr, y_arr,
            n_phys=n_phys, n_base=n_base, n_c=n_c,
            gap_hist=gap_hist, vjump_hist=vjump_hist,
            r_pert=r_pert, r_total=r_total,
            B_c_dyn=B_c_dyn, M_cc=M_cc, W_contact=W_contact,
            A_dyn=A_dyn, rhs_jac_dyn_eff=rhs_jac_dyn_eff, cs=cs,
        )

    sort_order = np.argsort(s_true)
    xi_phys = crack_half_length * s_true[sort_order]
    slip_phys = slip_phys[sort_order]
    s_param = s_true[sort_order]
    final_gap_sorted = gap_hist[-1, sort_order]
    final_vn_sorted = final_vjump[:n_c][sort_order]
    final_vt_sorted = final_vjump[n_c:][sort_order]
    final_rn_sorted = r_n[-1, sort_order]
    final_rt_sorted = r_t[-1, sort_order]
    cone_sorted = cone_margin[-1, sort_order]
    soc_res_node = np.sqrt(soc_res[0::2] ** 2 + soc_res[1::2] ** 2)
    soc_res_sorted = soc_res_node[sort_order]
    classes_sorted = [classes[int(i)] for i in sort_order]

    # tip augmentation: prepend / append (s = +/-1, slip = 0) per the notebook
    s_param_with_tips = np.concatenate(([-1.0], s_param, [1.0]))
    slip_with_tips = np.concatenate(([0.0], slip_phys, [0.0]))

    # --- Error in interior (|s| < 0.95) -------------------------------------
    slip_anal_phys = slip_max_anal * np.sqrt(np.clip(1.0 - s_param ** 2, 0.0, None))
    mask_interior = np.abs(s_param) < 0.95
    err_node_l2 = float(np.linalg.norm(
        np.abs(slip_phys[mask_interior]) - np.abs(slip_anal_phys[mask_interior])
    ))
    ref_node_l2 = float(np.linalg.norm(np.abs(slip_anal_phys[mask_interior])))
    rel_node_l2 = err_node_l2 / max(ref_node_l2, 1e-30)
    err_l2, ref_l2, rel_l2, _, _, _ = _integrated_slip_error(s_param, slip_phys)

    if AUDIT_DIR is not None:
        audit_dir = Path(AUDIT_DIR)
        audit_dir.mkdir(parents=True, exist_ok=True)
        with open(audit_dir / f'contact_audit_N{n_elem_target}.csv', 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'node', 's_true', 'xhat', 'yhat', 'delta_norm',
                    'gap', 'slip_phys', 'vn', 'vt', 'rn_MPa', 'rt_MPa',
                    'cone_margin_MPa', 'soc_res_norm', 'friction_power',
                    'class',
                ],
            )
            writer.writeheader()
            delta_norm_sorted = np.linalg.norm(delta_dim, axis=1)[sort_order]
            xhat_sorted = np.asarray(xhat_dim)[sort_order]
            power_sorted = friction_power[sort_order]
            for j, old_idx in enumerate(sort_order):
                writer.writerow({
                    'node': int(j),
                    's_true': f'{s_param[j]:.16e}',
                    'xhat': f'{xhat_sorted[j, 0]:.16e}',
                    'yhat': f'{xhat_sorted[j, 1]:.16e}',
                    'delta_norm': f'{delta_norm_sorted[j]:.16e}',
                    'gap': f'{final_gap_sorted[j]:.16e}',
                    'slip_phys': f'{slip_phys[j]:.16e}',
                    'vn': f'{final_vn_sorted[j]:.16e}',
                    'vt': f'{final_vt_sorted[j]:.16e}',
                    'rn_MPa': f'{final_rn_sorted[j]:.16e}',
                    'rt_MPa': f'{final_rt_sorted[j]:.16e}',
                    'cone_margin_MPa': f'{cone_sorted[j]:.16e}',
                    'soc_res_norm': f'{soc_res_sorted[j]:.16e}',
                    'friction_power': f'{power_sorted[j]:.16e}',
                    'class': classes_sorted[j],
                })
        diag_payload = {
            'N_ELEM': int(n_elem_target),
            'USE_SBM': bool(USE_SBM),
            'CONFORMING_CRACK_MESH': bool(CONFORMING_CRACK_MESH),
            'err_integrated_l2': err_l2,
            'ref_integrated_l2': ref_l2,
            'rel_integrated_l2': rel_l2,
            'err_node_l2': err_node_l2,
            'ref_node_l2': ref_node_l2,
            'rel_node_l2': rel_node_l2,
            'gap_shift_rel': gap_shift_rel,
            'adjoint_rel': adjoint_rel,
            'delassus_sym_rel': delassus_sym_rel,
            'delassus_min_eig': delassus_min_eig,
            **patch_diag,
            'soc_res_inf': soc_res_inf,
            'max_friction_power_pos': max_friction_power_pos,
            'n_open': n_open,
            'n_sticking': n_sticking,
            'n_sliding': n_sliding,
            'n_other_active': n_other_active,
            'min_gap': min_gap,
            'final_cone_margin_abs': final_cone_margin_abs,
            'contact_ok': contact_ok,
            **time_diag_summary,
        }
        (audit_dir / f'diagnostics_N{n_elem_target}.json').write_text(
            json.dumps(diag_payload, indent=2, sort_keys=True) + '\n'
        )

    domain_area = (XMAX - XMIN) * (YMAX - YMIN)
    h_eff = float(np.sqrt(domain_area / mesh.nelements))

    return LevelResult(
        n_elem_target=n_elem_target,
        n_elem_actual=int(mesh.nelements),
        n_c=int(n_c),
        h_eff=h_eff,
        err_l2=err_l2, ref_l2=ref_l2, rel_l2=rel_l2,
        err_node_l2=err_node_l2, ref_node_l2=ref_node_l2, rel_node_l2=rel_node_l2,
        t_build=t_build, t_solve=t_solve, t_total=t_build + t_solve,
        n_acc=n_acc, n_rej=n_rej,
        gap_shift_rel=gap_shift_rel, adjoint_rel=adjoint_rel,
        delassus_sym_rel=delassus_sym_rel, delassus_min_eig=delassus_min_eig,
        patch_all_inf=patch_diag['patch_all_inf'], soc_res_inf=soc_res_inf,
        max_friction_power_pos=max_friction_power_pos,
        n_open=n_open, n_sticking=n_sticking, n_sliding=n_sliding,
        n_other_active=n_other_active, min_gap=min_gap,
        final_cone_margin_abs=final_cone_margin_abs, contact_ok=contact_ok,
        s_param=s_param, slip_phys=slip_phys,
        s_param_with_tips=s_param_with_tips,
        slip_with_tips=slip_with_tips,
    )


# =============================================================================
# Plotting and output
# =============================================================================

def plot_slip_overlay(results, out_path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    s_dense = np.linspace(-1.0, 1.0, 401)
    ax.plot(s_dense, np.sqrt(1.0 - s_dense ** 2), 'k--', lw=1.6,
            label=r'analytical $\sqrt{1-s^2}$', zorder=10)
    ax.plot(s_dense, -np.sqrt(1.0 - s_dense ** 2), 'k--', lw=1.6, alpha=0.35,
            zorder=10)
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(results)))
    for color, r in zip(cmap, results):
        slip_norm = r.slip_with_tips / slip_max_anal
        ax.plot(r.s_param_with_tips, slip_norm,
                marker='o', ms=3.0, lw=1.0, color=color,
                label=f'N_ELEM={r.n_elem_target} (n_e={r.n_elem_actual}, n_c={r.n_c})')
    ax.set_xlabel(r'$s = \xi / c$')
    ax.set_ylabel(r'$[\![u_t]\!] / \delta_{\max}^{\rm anal}$')
    ax.set_title(r'Slip in parametric coordinates  ($\alpha = 0$, prestress, sliding)')
    ax.set_xlim(-1.05, 1.05)
    ax.axvline(-1.0, color='0.6', lw=0.6); ax.axvline(1.0, color='0.6', lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower center', fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_convergence(results, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    h = np.array([r.h_eff for r in results])
    err = np.array([r.err_l2 for r in results])
    ax.loglog(h, err, 'o-', ms=7, lw=1.5, color='tab:blue', label='FE (interior $|s|<0.95$)')
    if len(results) >= 2:
        slope, intercept = np.polyfit(np.log(h), np.log(err), 1)
        h_fit = np.array([h.min(), h.max()])
        ax.loglog(h_fit, np.exp(intercept) * h_fit ** slope, '--', color='tab:red',
                  lw=1.0, label=f'fit slope = {slope:.2f}')
    ax.set_xlabel(r'$h_{\rm eff} = \sqrt{A_{\rm dom} / n_{\rm elem}}$  (km)')
    ax.set_ylabel(r'$(\int |[\![u_t]\!]_h-[\![u_t]\!]_{\rm anal}|^2\,ds)^{1/2}$  (m)')
    ax.set_title(r'True-crack integrated L2 convergence vs $h_{\rm eff}$')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_csv(results, path):
    fieldnames = ['level', 'n_elem_target', 'n_elem_actual', 'n_c',
                  'h_eff', 'err_l2', 'ref_l2', 'rel_l2', 'rate',
                  'err_node_l2', 'ref_node_l2', 'rel_node_l2',
                  'gap_shift_rel', 'min_gap', 'final_cone_margin_abs',
                  'adjoint_rel', 'delassus_sym_rel', 'delassus_min_eig',
                  'patch_all_inf', 'soc_res_inf', 'max_friction_power_pos',
                  'n_open', 'n_sticking', 'n_sliding', 'n_other_active',
                  'contact_ok',
                  't_build_s', 't_solve_s', 't_total_s',
                  'n_acc', 'n_rej']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        prev = None
        for i, r in enumerate(results):
            if prev is None or r.h_eff >= prev.h_eff:
                rate = ''
            else:
                rate = f"{np.log(prev.err_l2 / r.err_l2) / np.log(prev.h_eff / r.h_eff):.3f}"
            writer.writerow({
                'level': i,
                'n_elem_target': r.n_elem_target, 'n_elem_actual': r.n_elem_actual,
                'n_c': r.n_c,
                'h_eff': f'{r.h_eff:.6e}',
                'err_l2': f'{r.err_l2:.6e}', 'ref_l2': f'{r.ref_l2:.6e}',
                'rel_l2': f'{r.rel_l2:.6f}',
                'rate': rate,
                'err_node_l2': f'{r.err_node_l2:.6e}',
                'ref_node_l2': f'{r.ref_node_l2:.6e}',
                'rel_node_l2': f'{r.rel_node_l2:.6f}',
                'gap_shift_rel': f'{r.gap_shift_rel:.6e}',
                'min_gap': f'{r.min_gap:.6e}',
                'final_cone_margin_abs': f'{r.final_cone_margin_abs:.6e}',
                'adjoint_rel': f'{r.adjoint_rel:.6e}',
                'delassus_sym_rel': f'{r.delassus_sym_rel:.6e}',
                'delassus_min_eig': f'{r.delassus_min_eig:.6e}',
                'patch_all_inf': f'{r.patch_all_inf:.6e}',
                'soc_res_inf': f'{r.soc_res_inf:.6e}',
                'max_friction_power_pos': f'{r.max_friction_power_pos:.6e}',
                'n_open': r.n_open,
                'n_sticking': r.n_sticking,
                'n_sliding': r.n_sliding,
                'n_other_active': r.n_other_active,
                'contact_ok': int(r.contact_ok),
                't_build_s': f'{r.t_build:.3f}',
                't_solve_s': f'{r.t_solve:.3f}',
                't_total_s': f'{r.t_total:.3f}',
                'n_acc': r.n_acc, 'n_rej': r.n_rej,
            })
            prev = r


def write_tex(results, path):
    lines = [
        r'\begin{tabular}{rrrrrrrrrrr}',
        r'\hline',
        r'$N_{\rm tgt}$ & $n_e$ & $n_c$ & $h_{\rm eff}$ & $\|e\|_{\ell^2}$ & rel & rate'
        r' & $t_{\rm build}$ & $t_{\rm solve}$ & $n_{\rm acc}$ & $n_{\rm rej}$ \\',
        r' & & & (km) & (m) & & & (s) & (s) & & \\',
        r'\hline',
    ]
    prev = None
    for r in results:
        if prev is None or r.h_eff >= prev.h_eff:
            rate = '--'
        else:
            rate = f"{np.log(prev.err_l2 / r.err_l2) / np.log(prev.h_eff / r.h_eff):.2f}"
        lines.append(
            f"{r.n_elem_target} & {r.n_elem_actual} & {r.n_c} & "
            f"{r.h_eff:.3e} & {r.err_l2:.3e} & {r.rel_l2:.3f} & {rate} & "
            f"{r.t_build:.1f} & {r.t_solve:.1f} & {r.n_acc} & {r.n_rej} \\\\"
        )
        prev = r
    lines += [r'\hline', r'\end{tabular}']
    Path(path).write_text('\n'.join(lines) + '\n')


# =============================================================================
# Driver
# =============================================================================

def parse_levels(arg):
    return [int(x.strip()) for x in arg.split(',') if x.strip()]


def main():
    global AUDIT_DIR, ELEMENT_TYPE, MESH_SOURCE
    ap = argparse.ArgumentParser()
    ap.add_argument('--levels', type=parse_levels,
                    default=[20, 40, 80, 160, 320],
                    help='comma-separated list of N_ELEM targets')
    ap.add_argument('--out', type=Path, default=Path('images'))
    ap.add_argument('--element-type', choices=['tri', 'quad'],
                    default=ELEMENT_TYPE,
                    help='bulk mesh topology passed to CrackMeshBuilder')
    ap.add_argument('--mesh-source', choices=['auto', 'scikit', 'gmsh'],
                    default=MESH_SOURCE,
                    help=(
                        'auto keeps existing behavior; scikit uses tensor '
                        'MeshTri1/MeshQuad1 where available'
                    ))
    args = ap.parse_args()

    ELEMENT_TYPE = args.element_type
    MESH_SOURCE = args.mesh_source
    args.out.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR = args.out / 'audits'
    print(f"output directory: {args.out.resolve()}")
    print(f"levels: {args.levels}")
    print(f"element_type: {ELEMENT_TYPE}, mesh_source={MESH_SOURCE}, "
          f"conforming={CONFORMING_CRACK_MESH}, USE_SBM={USE_SBM}")
    print(f"physics: alpha = {ALPHA_BIOT}, BULK_DAMPING_FRACTION = {BULK_DAMPING_FRACTION}, "
          f"TMAX_PHYS = {TMAX_PHYS} s (TMAX_d = {TMAX:.4g})")
    print(f"analytical slip_max = {slip_max_anal:.4e} m")

    results = []
    for n in args.levels:
        try:
            r = run_level(n)
            results.append(r)
            # Persist after each level so a partial sweep is still useful
            plot_slip_overlay(results, args.out / 'slip_per_h.png')
            if len(results) >= 2:
                plot_convergence(results, args.out / 'convergence_l2.png')
            write_csv(results, args.out / 'convergence_summary.csv')
            write_tex(results, args.out / 'convergence_summary.tex')
        except Exception as exc:
            print(f"  [level N_ELEM={n}] FAILED: {exc!r}", flush=True)
            continue

    print("\n" + "=" * 72)
    print(f"completed {len(results)} / {len(args.levels)} levels")
    if results:
        print("level summary:")
        for i, r in enumerate(results):
            print(f"  [{i}] N={r.n_elem_target:>4d}  n_e={r.n_elem_actual:>5d}  "
                  f"n_c={r.n_c:>3d}  h_eff={r.h_eff:.3e}  "
                  f"err_l2={r.err_l2:.3e}  rel={r.rel_l2:.3f}  "
                  f"node_rel={r.rel_node_l2:.3f}  shift={r.gap_shift_rel:.2e}  "
                  f"adj={r.adjoint_rel:.1e}  soc={r.soc_res_inf:.1e}  "
                  f"ok={int(r.contact_ok)}  "
                  f"t_total={r.t_total:.1f}s")


if __name__ == '__main__':
    main()
