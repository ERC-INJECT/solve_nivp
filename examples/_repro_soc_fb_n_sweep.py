"""SOC FB N-refinement reproducer for the embedded crack locked test.

Runs a stripped-down version of the locked test from
`embedded_crack_mohr_coulomb_ncp.ipynb` at a chosen N_ELEM, with
instrumentation hooks on:
  * `_compute_delassus_rho`         — logs rho_N, rho_T per call
  * `_contact_residual`             — logs gap distribution and active set
  * `_contact_jacobian`             — same
  * `ImplicitEquationSolver.solve`  — logs Newton iter count, final residual,
                                       success/failure, and exception type
  * `SOCFischerBurmeisterLaw.residual_and_jac` — logs (u_hat[0], r_blk) ranges
                                       and any LinAlgError

Usage:
    REPRO_N_ELEM=20 python examples/_repro_soc_fb_n_sweep.py
    REPRO_N_ELEM=40 python examples/_repro_soc_fb_n_sweep.py
    REPRO_N_ELEM=60 python examples/_repro_soc_fb_n_sweep.py

Diagnostics dumped to /tmp/repro_soc_fb_N{N}.json.
"""

import os
import json
import time
import traceback
import dataclasses as _dc

import numpy as np
import scipy.sparse as sp

from skfem.models.elasticity import lame_parameters

import poroelasticity.cgporoelastostatics as _cgmod
import poroelasticity.nondimensionalise as _nondim_mod
from poroelasticity import CGPoroelastostatics, CrackMeshBuilder, MaterialParams
from poroelasticity.config import ScaleFactors as _ScaleFactors

import solve_nivp
from solve_nivp.projected_radau_contact import (
    build_projected_radau_contact,
    SOCFischerBurmeisterLaw,
)
from solve_nivp.nonlinear_solvers import ImplicitEquationSolver

# ---------------------------------------------------------------------------
# CLI / env knobs
# ---------------------------------------------------------------------------
N_ELEM_ENV = int(os.environ.get("REPRO_N_ELEM", "40"))
REPRO_TMAX_FRAC = float(os.environ.get("REPRO_TMAX_FRAC", "0.005"))  # tiny window
REPRO_MODE = os.environ.get("REPRO_MODE", "locked")  # 'locked' or 'sliding'
LOG_PATH = os.environ.get(
    "REPRO_LOG_PATH",
    f"/tmp/repro_soc_fb_{REPRO_MODE}_N{N_ELEM_ENV}.json",
)
PRESTRESS_FACTOR = 2.0 if REPRO_MODE == "sliding" else 1.0

DIAG = {
    "N_ELEM": N_ELEM_ENV,
    "delassus_calls": [],
    "contact_residual_calls": [],
    "contact_jacobian_calls": [],
    "solve_calls": [],
    "soc_law_calls_summary": {"n_calls": 0, "n_pinv_fallback": 0,
                              "n_linalg_err": 0,
                              "max_uhat0_abs": 0.0, "max_r0_abs": 0.0,
                              "max_ut_norm": 0.0, "max_rt_norm": 0.0},
    "outcome": "unknown",
    "error": None,
}

# ---------------------------------------------------------------------------
# Wave-clock override (verbatim from notebook)
# ---------------------------------------------------------------------------
if not hasattr(_cgmod, "_pristine_compute_dimensionless_params"):
    _cgmod._pristine_compute_dimensionless_params = _nondim_mod.compute_dimensionless_params
_orig_compute_ndp = _cgmod._pristine_compute_dimensionless_params


def _wave_clock_ndp(material, scales, crack_model=None,
                    bulk_viscosity=None, dim=2, P_scale_override=None):
    if material.alpha == 0.0:
        return _orig_compute_ndp(
            material, scales,
            crack_model=crack_model, bulk_viscosity=bulk_viscosity,
            dim=dim, P_scale_override=P_scale_override,
        )
    mat0 = _dc.replace(material, alpha=0.0)
    ndp0 = _orig_compute_ndp(
        mat0, scales,
        crack_model=crack_model, bulk_viscosity=bulk_viscosity, dim=dim,
    )
    alpha = material.alpha
    beta = material.beta
    C = material.C
    L = scales.L
    eps = scales.eps
    T_w = ndp0.T_scale
    Mmod = ndp0.Mmod_scale
    P_scale_new = alpha * eps / beta
    Q_scale_new = beta * P_scale_new / T_w
    C_d_new = C * T_w / (beta * L ** 2) if C > 0 else 0.0
    if crack_model is not None and getattr(crack_model, "T", 0.0):
        T_robin_d_new = crack_model.T * T_w * P_scale_new / (Mmod * L * eps)
    else:
        T_robin_d_new = ndp0.T_robin_d
    return _dc.replace(
        ndp0,
        alpha_d=alpha,
        C_d=C_d_new,
        P_scale=P_scale_new,
        Q_scale=Q_scale_new,
        T_robin_d=T_robin_d_new,
    )


_cgmod.compute_dimensionless_params = _wave_clock_ndp
_compute_ndp = _wave_clock_ndp

# ---------------------------------------------------------------------------
# Material / loading constants (verbatim)
# ---------------------------------------------------------------------------
NU = 0.3
G_SHEAR = 10e3
E_YOUNG = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT = 0.3
BETA_FLUID = 8.5e-5
ETA_FLUID = 2.0e-18 / 3600.0
K_PERM = 1.0e-15 * 1.0e-6
DENSITY_PHYS = 2.7e3
BULK_DAMPING_FRACTION = 9e-1

XMIN, XMAX = 0.0, 20.0
YMIN, YMAX = 0.0, 20.0
N_ELEM = N_ELEM_ENV  # <- the swept knob
CRACK_LENGTH = 6.0
CRACK_X0 = 10.0
CRACK_Y0 = 10.0

SIGMA_RIGHT = 10.0
SIGMA_TOP = 50.0
MU = 0.6

CRACK_T_ROBIN = 0.0
CRACK_MODEL_PARAMS = {"T": CRACK_T_ROBIN, "tangential_flow": False}

USE_SBM = (os.environ.get("REPRO_USE_SBM", "1") == "1")
CONFORMING_CRACK_MESH = not USE_SBM
INCLUDE_SBM = False
INCLUDE_TAYLOR = USE_SBM
INCLUDE_TAYLOR_TEST = False
INCLUDE_HESSIAN = False

TMAX_PHYS = 30.0
H_PHYS = 2.8e-1

SCALE_L = 1.0
SCALE_EPS = 1.0e-3

LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
MATERIAL = MaterialParams(
    mu=MU_LAME, lam=LAM, alpha=ALPHA_BIOT,
    beta=BETA_FLUID, C=K_PERM / ETA_FLUID, rho=DENSITY_PHYS,
)
Mmod = LAM + 2 * MU_LAME
Sigma_scale = Mmod * SCALE_EPS
T_scale = _compute_ndp(MATERIAL, _ScaleFactors(L=SCALE_L, eps=SCALE_EPS)).T_scale
BULK_MU_V = BULK_DAMPING_FRACTION * MU_LAME * T_scale
BULK_LAM_V = BULK_DAMPING_FRACTION * LAM * T_scale

TMAX = TMAX_PHYS / T_scale
H_FIXED = H_PHYS / T_scale

# Run only a tiny slice of time: enough to trigger any Newton failure
TMAX_REPRO = TMAX * REPRO_TMAX_FRAC

ADAPTIVE_RTOL = 1.0e-3
ADAPTIVE_ATOL = 1.0e-6
ADAPTIVE_H_MIN = 1.0e-5
ADAPTIVE_H_MAX = TMAX / 10
H0_ADAPTIVE = H_FIXED

S1 = max(SIGMA_RIGHT, SIGMA_TOP)
S3 = min(SIGMA_RIGHT, SIGMA_TOP)

THETA_LOCKED = np.radians(20.0)
THETA_SLIDING = np.radians(60.0)
THETA_USED = THETA_SLIDING if REPRO_MODE == "sliding" else THETA_LOCKED
CRACK_THETA_USED = np.pi / 2 - THETA_USED


def mohr_tractions(theta):
    sigma_N = (S1 + S3) / 2 + (S1 - S3) / 2 * np.cos(2 * theta)
    tau = (S1 - S3) / 2 * np.sin(2 * theta)
    return sigma_N, tau


def make_bc():
    return {"dp_rate": {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}}


def lysmer_descriptor_block(poro, n_phys, n_base, a_p=1.0, a_s=1.0):
    from skfem import FacetBasis, BilinearForm

    rho_phys = MATERIAL.rho
    Vp_phys = np.sqrt(E_YOUNG * (1.0 - NU)
                      / ((1.0 + NU) * (1.0 - 2.0 * NU) * rho_phys))
    Vs_phys = np.sqrt(E_YOUNG / (2.0 * (1.0 + NU) * rho_phys))
    T_scale_loc = _compute_ndp(MATERIAL, _ScaleFactors(L=SCALE_L, eps=SCALE_EPS)).T_scale
    Vp_d = Vp_phys * T_scale_loc / SCALE_L
    Vs_d = Vs_phys * T_scale_loc / SCALE_L
    rho_d = float(poro.rho_d)
    c_n = a_p * rho_d * Vp_d
    c_t = a_s * rho_d * Vs_d

    fb = FacetBasis(poro.mesh, poro.basis_u.elem,
                    facets=poro.mesh.boundary_facets(),
                    intorder=poro.intorder)

    @BilinearForm
    def lysmer_form(u, v, w):
        n = w.n
        u_n = u[0] * n[0] + u[1] * n[1]
        v_n = v[0] * n[0] + v[1] * n[1]
        u_tx = u[0] - u_n * n[0]
        u_ty = u[1] - u_n * n[1]
        v_tx = v[0] - v_n * n[0]
        v_ty = v[1] - v_n * n[1]
        return c_n * u_n * v_n + c_t * (u_tx * v_tx + u_ty * v_ty)

    C_l = lysmer_form.assemble(fb).tocsr()
    Nu = C_l.shape[0]
    Z_top = sp.csr_matrix((n_base, n_phys))
    Z_left = sp.csr_matrix((Nu, n_base))
    bottom = sp.hstack([Z_left, C_l], format="csr")
    return sp.vstack([Z_top, bottom], format="csr")


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
        out[2 * k] = r_n_p - sN
        out[2 * k + 1] = r_t_p - w
    return out


# ---------------------------------------------------------------------------
# Build mesh, poro, dynamic system, contact wiring
# ---------------------------------------------------------------------------
print(f"\n=== REPRO N_ELEM = {N_ELEM} ===")
t_setup0 = time.time()

print(f"REPRO_MODE = {REPRO_MODE}  prestress_factor = {PRESTRESS_FACTOR}")
builder = CrackMeshBuilder(
    XMIN, XMAX, YMIN, YMAX, N_ELEM,
    crack_theta=CRACK_THETA_USED,
    crack_x0=CRACK_X0, crack_y0=CRACK_Y0,
    crack_length=CRACK_LENGTH,
    element_type="tri", conforming=CONFORMING_CRACK_MESH,
    verbose=False,
)
mesh, el_p, el_u, crack, h_mesh = builder.build()
print(f"h_mesh = {h_mesh:.4f},  elements = {mesh.nelements}")

bc = make_bc()
poro = CGPoroelastostatics(
    mesh=mesh, element_p=el_p, element_u=el_u,
    params=PARAMS, material=MATERIAL,
    crack=crack, model_params=CRACK_MODEL_PARAMS,
    intorder=6,
    scales=(SCALE_L, SCALE_EPS), bc=bc, P_scale=None,
    verbose=False, free_memory=False,
    enforcement_type="nodal", apply_transform=True,
    include_taylor=INCLUDE_TAYLOR,
    include_taylor_test=INCLUDE_TAYLOR_TEST,
    include_hessian=INCLUDE_HESSIAN,
    include_sbm=INCLUDE_SBM,
    taylor_method="nodal",
    lumped_coupling="consistent",
    crack_law="nonsmooth",
    rotate_crack_to_nt=True,
    bulk_viscosity={"mu_v": BULK_MU_V, "lam_v": BULK_LAM_V},
)
poro.strip_multiplier_dynamics()
projection, meta = poro.build_projection()
dyn = poro.build_first_order_dynamic_system(meta)

A_dyn = dyn["A"]
rhs_dyn = dyn["rhs"]
rhs_jac_dyn = dyn["rhs_jac"]
n_base = dyn["n_base"]
comp_slices = dyn["component_slices"]

Np = poro.basis_p.N
Nu = poro.basis_u.N
n_phys_locked = A_dyn.shape[0]

info = poro._transform_info
n_c = int(poro.n_lambda_q)
lam_s_sl = meta["lam_s_sl"]

off_jmpu = info["off_jmpu"]
n_iu = info["n_intf_u"]
off_jmpls = info["off_jmpls"]
n_lsig = info["n_lam_sig"]

jmpu_n_indices = np.arange(off_jmpu, off_jmpu + n_c)
jmpu_t_indices = np.arange(off_jmpu + n_c, off_jmpu + n_iu)
jmpu_all = np.concatenate([jmpu_n_indices, jmpu_t_indices])
jmpls_cols = np.arange(off_jmpls, off_jmpls + n_lsig)

T_u_mat = dyn["T_u"]
u_idx = dyn["u_state_indices"]
jmpu_local = np.searchsorted(u_idx, jmpu_all)
D = T_u_mat[jmpu_local, :].tocsr()

R_v = dyn.get("R_v")
A_csr = poro.A.tocsr()
B_u = (A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
if R_v is not None:
    B_u = (R_v @ B_u).tocsr()

dirichlet_local = np.array(
    [d - Np for d in np.asarray(getattr(poro, "_dirichlet_dof_set", []), dtype=int)
     if Np <= d < Np + Nu], dtype=int,
)
if dirichlet_local.size:
    B_u = B_u.tolil()
    B_u[dirichlet_local, :] = 0.0
    B_u = B_u.tocsr()

perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
B_u_perm = B_u[:, perm].tocsr()

contacts = [
    {"vel_normal_idx": k, "vel_tangential_idx": [n_c + k], "mu": MU, "e": 0.0}
    for k in range(n_c)
]

n_aug_phys = n_phys_locked
n_extract = 2 * n_c
gap_extract_dyn = sp.csr_matrix(
    (np.ones(n_extract), (np.arange(n_extract), jmpu_all)),
    shape=(n_extract, n_aug_phys),
)
vel_extract_dyn = sp.hstack(
    [sp.csr_matrix((n_extract, n_base)), D], format="csr",
)
B_c_dyn = sp.vstack(
    [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm], format="csr",
)

C_lysmer_locked = lysmer_descriptor_block(poro, n_phys_locked, n_base)


def rhs_dyn_eff(t, y, *extra):
    return rhs_dyn(t, y, *extra) - C_lysmer_locked @ y


def rhs_jac_dyn_eff(t, y, *extra):
    return rhs_jac_dyn(t, y, *extra) - C_lysmer_locked


sigma_N_used, tau_used_val = mohr_tractions(THETA_USED)
_s0_val = np.full(n_c, PRESTRESS_FACTOR * sigma_N_used / Sigma_scale)
_w0_val = np.full(n_c, PRESTRESS_FACTOR * tau_used_val / Sigma_scale)


def get_s0_locked(y):
    return _s0_val


def get_w0_locked(y, k):
    return np.array([_w0_val[int(k)]])


flux_constraint = meta["constraints"][0]
n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
zero_lam_s = {
    "g": lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
    "dg_dy": lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
    "y_slice": lam_s_sl,
    "q_slice": lam_s_sl,
}

cs = build_projected_radau_contact(
    A_dyn, rhs_dyn_eff, np.zeros(A_dyn.shape[0]),
    contacts=contacts,
    C_extract=gap_extract_dyn, D_extract=vel_extract_dyn, B=B_c_dyn,
    constraints=[flux_constraint, zero_lam_s],
    component_slices=comp_slices,
    rhs_jac=rhs_jac_dyn_eff,
    get_s0=get_s0_locked, get_w0=get_w0_locked,
    normal_r="auto", friction_r="auto",
    endpoint_inactive_handling="natural_map",
    contact_law="soc_fb",
    reported_reaction_units="force",
    auto_rho_strategy="delassus",
)

y0_locked = cs.y0.copy()
y0_locked[-2 * n_c:] = coulomb_cone_warm_start(_s0_val, _w0_val, MU)

DIAG["n_c"] = n_c
DIAG["n_phys"] = n_phys_locked
DIAG["n_aug"] = int(cs.y0.size)
DIAG["h_mesh"] = float(h_mesh)
DIAG["nelements"] = int(mesh.nelements)
DIAG["TMAX_REPRO"] = float(TMAX_REPRO)
DIAG["H0"] = float(H0_ADAPTIVE)
DIAG["setup_seconds"] = time.time() - t_setup0
print(f"setup done: n_c={n_c}, n_phys={n_phys_locked}, "
      f"n_aug={cs.y0.size}, setup={DIAG['setup_seconds']:.1f}s")

# ---------------------------------------------------------------------------
# LBB / inf-sup diagnostic (run only when LBB_ONLY=1, then exit)
# ---------------------------------------------------------------------------
if os.environ.get("LBB_ONLY") == "1":
    import sys
    import scipy.sparse.linalg as spla
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_prefix = f"/tmp/lbb_N{N_ELEM}"
    print(f"\n=== LBB diagnostic at N={N_ELEM} ===")

    B_lbb = B_u_perm.tocsr()
    D_lbb = D.tocsr()
    print(f"B shape = {B_lbb.shape}   nnz = {B_lbb.nnz}")
    print(f"D shape = {D_lbb.shape}   nnz = {D_lbb.nnz}")

    # ---------------- (a) discrete inf-sup ------------------
    BtB = (B_lbb.T @ B_lbb).tocsr()
    BBt = (B_lbb @ B_lbb.T).tocsr()
    n_lam = BtB.shape[0]
    k_eig = min(8, n_lam - 2)
    eig_low, vec_low = spla.eigsh(BtB, k=k_eig, sigma=0.0, which="LM")
    eig_low_sorted = np.sort(eig_low)
    sigma_min = float(np.sqrt(max(eig_low_sorted[0], 0.0)))
    print(f"\n(a) Lowest {k_eig} eigenvalues of B^T B (multiplier metric = I):")
    for i, ev in enumerate(eig_low_sorted):
        print(f"     λ_{i} = {ev:+.4e}    σ_{i} = {np.sqrt(max(ev,0)):+.4e}")
    print(f"     σ_min(B) = {sigma_min:.4e}")
    eig_high = spla.eigsh(BtB, k=1, which="LM",
                          return_eigenvectors=False)
    sigma_max = float(np.sqrt(max(float(eig_high[0]), 0.0)))
    print(f"     σ_max(B) = {sigma_max:.4e}")
    cond_B = sigma_max / max(sigma_min, 1e-300)
    print(f"     cond_2(B) = σ_max/σ_min = {cond_B:.3e}")

    # ---------------- (b) checkerboard signature ------------------
    ones_lam = np.ones(B_lbb.shape[1])
    ones_phys = np.ones(B_lbb.shape[0])
    B_one = np.asarray(B_lbb @ ones_lam).ravel()
    Bt_one = np.asarray(B_lbb.T @ ones_phys).ravel()
    print(f"\n(b) Uniform-multiplier and uniform-bulk responses:")
    print(f"     B @ 1_lam :  min = {B_one.min():+.3e}  "
          f"max = {B_one.max():+.3e}  std = {B_one.std():.3e}")
    print(f"     B^T @ 1   :  min = {Bt_one.min():+.3e}  "
          f"max = {Bt_one.max():+.3e}  std = {Bt_one.std():.3e}")

    # Reorder lam DOFs into (n,t) per contact: perm = [n0,t0,n1,t1,...]
    lam_n = Bt_one[0::2]
    lam_t = Bt_one[1::2]
    print(f"     B^T@1 normal block:    "
          f"sign-flips per cell = "
          f"{int(np.sum(np.sign(lam_n[1:]) * np.sign(lam_n[:-1]) < 0))}/{lam_n.size-1}")
    print(f"     B^T@1 tangential block: "
          f"sign-flips per cell = "
          f"{int(np.sum(np.sign(lam_t[1:]) * np.sign(lam_t[:-1]) < 0))}/{lam_t.size-1}")

    # ---------------- (c) B vs D^T column-norm structure ------------------
    print(f"\n(c) Column-norm pattern (per-multiplier weight) of B and D^T:")
    B_col_norm = np.sqrt(np.asarray(B_lbb.multiply(B_lbb).sum(axis=0)).ravel())
    Dt = D_lbb.T.tocsr()
    Dt_col_norm = np.sqrt(np.asarray(Dt.multiply(Dt).sum(axis=0)).ravel())
    print(f"     B col-norm   : "
          f"min = {B_col_norm.min():.3e}  max = {B_col_norm.max():.3e}  "
          f"max/min = {B_col_norm.max()/max(B_col_norm.min(),1e-300):.2e}")
    print(f"     D^T col-norm : "
          f"min = {Dt_col_norm.min():.3e}  max = {Dt_col_norm.max():.3e}  "
          f"max/min = {Dt_col_norm.max()/max(Dt_col_norm.min(),1e-300):.2e}")

    # Effective per-contact "weight" = B_col_norm / Dt_col_norm
    w_eff = B_col_norm / np.maximum(Dt_col_norm, 1e-30)
    w_eff_n = w_eff[0::2]
    w_eff_t = w_eff[1::2]
    print(f"     w_eff = ||B[:,k]|| / ||D[k,:]||  (proxy for area weight)")
    print(f"       normal block:     min={w_eff_n.min():.3e}  "
          f"max={w_eff_n.max():.3e}  ratio={w_eff_n.max()/max(w_eff_n.min(),1e-300):.2e}")
    print(f"       tangential block: min={w_eff_t.min():.3e}  "
          f"max={w_eff_t.max():.3e}  ratio={w_eff_t.max()/max(w_eff_t.min(),1e-300):.2e}")

    # ---------------- plots ------------------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    s_axis = np.linspace(-1, 1, n_c)

    axes[0, 0].plot(s_axis, w_eff_n, "o-", label="normal")
    axes[0, 0].plot(s_axis, w_eff_t, "s--", label="tangential")
    axes[0, 0].set_title("(c) per-cell effective weight  ||B[:,k]||/||D[k,:]||")
    axes[0, 0].set_xlabel("s = ξ/c"); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(s_axis, lam_n, "o-", label="(B^T 1)_n")
    axes[0, 1].plot(s_axis, lam_t, "s--", label="(B^T 1)_t")
    axes[0, 1].set_title("(b) bulk-side response to uniform multiplier")
    axes[0, 1].set_xlabel("s = ξ/c"); axes[0, 1].legend(); axes[0, 1].grid(True)

    # Plot the lowest eigenvector of B^T B  (with per-cell amplitude)
    v_low = vec_low[:, np.argmin(eig_low)]
    v_low_n = v_low[0::2]; v_low_t = v_low[1::2]
    cell_amp = np.sqrt(v_low_n**2 + v_low_t**2)
    print(f"\n     lowest-mode per-cell amplitude (||(v_n, v_t)||):")
    for k in range(n_c):
        marker = "  <-- tip" if (k == 0 or k == n_c - 1) else ""
        print(f"       cell {k:3d}  s={s_axis[k]:+.3f}   "
              f"v_n={v_low_n[k]:+.3e}  v_t={v_low_t[k]:+.3e}  "
              f"amp={cell_amp[k]:.3e}{marker}")
    interior_amp = float(np.median(cell_amp[1:-1])) if n_c > 2 else 0.0
    tip_amp = float(0.5 * (cell_amp[0] + cell_amp[-1]))
    print(f"     tip amp = {tip_amp:.3e},  interior median amp = {interior_amp:.3e},"
          f"  ratio = {tip_amp / max(interior_amp, 1e-30):.2f}")
    axes[1, 0].plot(s_axis, v_low_n, "o-", label="normal")
    axes[1, 0].plot(s_axis, v_low_t, "s--", label="tangential")
    axes[1, 0].plot(s_axis, cell_amp, "k:", label="amp", alpha=0.6)
    axes[1, 0].set_title(f"(a) lowest mode of B^T B   (λ={eig_low_sorted[0]:.2e})")
    axes[1, 0].set_xlabel("s = ξ/c"); axes[1, 0].legend(); axes[1, 0].grid(True)

    # Spectrum
    full_eig = np.sort(spla.eigsh(BtB, k=min(20, n_lam - 2),
                                  which="LM",
                                  return_eigenvectors=False))
    axes[1, 1].semilogy(np.arange(full_eig.size), full_eig, "o-",
                        label="largest")
    axes[1, 1].semilogy(np.arange(eig_low_sorted.size), eig_low_sorted, "s--",
                        label="smallest")
    axes[1, 1].set_title("eigenvalue spectrum of B^T B")
    axes[1, 1].set_xlabel("index"); axes[1, 1].legend(); axes[1, 1].grid(True)

    fig.suptitle(f"LBB diagnostic, N = {N_ELEM}", y=1.0)
    fig.tight_layout()
    fig.savefig(out_prefix + ".png", dpi=110)
    print(f"\n     plot saved -> {out_prefix}.png")
    sys.exit(0)

# ---------------------------------------------------------------------------
# Instrumentation patches (applied AFTER cs is built so cs.solver_opts etc are
# already captured)
# ---------------------------------------------------------------------------
contact_obj = cs.projected_radau_contact

_orig_delassus = contact_obj._compute_delassus_rho


def _patched_delassus(self, y, t, h):
    rho = _orig_delassus(y, t, h)
    rho_N, rho_T = rho
    rec = {"t": float(t), "h": float(h)}
    if rho_N is None:
        rec["rho_N"] = None
        rec["rho_T"] = None
    else:
        rec["rho_N_min"] = float(np.min(rho_N))
        rec["rho_N_max"] = float(np.max(rho_N))
        rec["rho_N_med"] = float(np.median(rho_N))
        rec["rho_T_min"] = float(np.min(rho_T))
        rec["rho_T_max"] = float(np.max(rho_T))
    DIAG["delassus_calls"].append(rec)
    return rho


contact_obj._compute_delassus_rho = _patched_delassus.__get__(contact_obj)

_orig_cresid = contact_obj._contact_residual
CRESID_PROGRESS = {"calls": 0, "last_print_t": time.time()}


def _patched_cresid(self, y, t, percussion, offset_measure, h, *, endpoint):
    gaps = self.gap(y, t)
    active_mask = gaps <= self.gap_tol
    rec = {
        "t": float(t), "h": float(h), "endpoint": bool(endpoint),
        "n_active": int(active_mask.sum()),
        "n_inactive": int((~active_mask).sum()),
        "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
        "gap_abs_max": float(np.abs(gaps).max()),
        "perc_max": float(np.abs(percussion).max()),
        "offset_max": float(np.abs(offset_measure).max()),
    }
    DIAG["contact_residual_calls"].append(rec)
    CRESID_PROGRESS["calls"] += 1
    if CRESID_PROGRESS["calls"] % 50 == 0:
        now = time.time()
        dt = now - CRESID_PROGRESS["last_print_t"]
        print(f"  [cresid #{CRESID_PROGRESS['calls']}] t={t:.3e} h={h:.2e} "
              f"endpoint={endpoint} n_inactive={rec['n_inactive']}/"
              f"{active_mask.size}  gap_abs_max={rec['gap_abs_max']:.2e}  "
              f"perc_max={rec['perc_max']:.2e}  dt={dt:.2f}s",
              flush=True)
        CRESID_PROGRESS["last_print_t"] = now
    return _orig_cresid(y, t, percussion, offset_measure, h, endpoint=endpoint)


contact_obj._contact_residual = _patched_cresid.__get__(contact_obj)

_orig_cjac = contact_obj._contact_jacobian


def _patched_cjac(self, y, t, percussion, offset_measure, h, *, endpoint):
    gaps = self.gap(y, t)
    active_mask = gaps <= self.gap_tol
    DIAG["contact_jacobian_calls"].append({
        "t": float(t), "h": float(h), "endpoint": bool(endpoint),
        "n_active": int(active_mask.sum()),
        "n_inactive": int((~active_mask).sum()),
        "gap_min": float(gaps.min()), "gap_max": float(gaps.max()),
    })
    return _orig_cjac(y, t, percussion, offset_measure, h, endpoint=endpoint)


contact_obj._contact_jacobian = _patched_cjac.__get__(contact_obj)


_orig_law_resjac = SOCFischerBurmeisterLaw.residual_and_jac
LAW_PRINT_EVERY = int(os.environ.get("LAW_PRINT_EVERY", "1000"))
LAW_PROGRESS = {"last_print_t": time.time(), "last_n": 0}


def _patched_law(self, normal_quantity, contact_velocity, percussion,
                 mu, normal_scale, friction_scale):
    summary = DIAG["soc_law_calls_summary"]
    summary["n_calls"] += 1
    u = np.asarray(contact_velocity, dtype=float).ravel()
    r = np.asarray(percussion, dtype=float).ravel()
    summary["max_uhat0_abs"] = max(summary["max_uhat0_abs"],
                                    float(abs(normal_quantity)))
    summary["max_r0_abs"] = max(summary["max_r0_abs"], float(abs(r[0])))
    if u.size > 1:
        summary["max_ut_norm"] = max(summary["max_ut_norm"],
                                      float(np.linalg.norm(u[1:])))
        summary["max_rt_norm"] = max(summary["max_rt_norm"],
                                      float(np.linalg.norm(r[1:])))
    n = summary["n_calls"]
    if n % LAW_PRINT_EVERY == 0:
        now = time.time()
        dt = now - LAW_PROGRESS["last_print_t"]
        dn = n - LAW_PROGRESS["last_n"]
        rate = dn / dt if dt > 0 else float("inf")
        print(f"  [law tick] n_calls={n}  rate={rate:.1f}/s  "
              f"|v_N|={abs(normal_quantity):.2e} |r_N|={abs(r[0]):.2e} "
              f"|v_T|={(np.linalg.norm(u[1:]) if u.size>1 else 0):.2e} "
              f"|r_T|={(np.linalg.norm(r[1:]) if r.size>1 else 0):.2e}",
              flush=True)
        LAW_PROGRESS["last_print_t"] = now
        LAW_PROGRESS["last_n"] = n
    try:
        return _orig_law_resjac(self, normal_quantity, contact_velocity,
                                percussion, mu, normal_scale, friction_scale)
    except np.linalg.LinAlgError as e:
        summary["n_linalg_err"] += 1
        summary.setdefault("linalg_err_msg", str(e))
        raise


SOCFischerBurmeisterLaw.residual_and_jac = _patched_law

# Optional A/B test: REPRO_LAW_FORM=position reverts the patch's velocity-form
# dispatch by toggling the flag off (so SOC FB gets gaps[k] again, like pre-patch).
if os.environ.get("REPRO_LAW_FORM", "velocity") == "position":
    SOCFischerBurmeisterLaw.expects_velocity_normal = False
    print("[A/B] REPRO_LAW_FORM=position  ->  expects_velocity_normal = False",
          flush=True)

# ImplicitEquationSolver.solve patch — capture iter count and final residual
_orig_solve = ImplicitEquationSolver.solve


def _patched_solve(self, func, y0):
    t0 = time.time()
    n_law_before = DIAG["soc_law_calls_summary"]["n_calls"]
    try:
        result = _orig_solve(self, func, y0)
        info = getattr(self, "info", {}) or {}
        n_law = DIAG["soc_law_calls_summary"]["n_calls"] - n_law_before
        wall = time.time() - t0
        rec = {
            "ok": True,
            "iters": int(info.get("iters", -1)) if isinstance(info, dict) else -1,
            "final_resid_norm": float(info.get("residual_norm", float("nan")))
                if isinstance(info, dict) else float("nan"),
            "wall": wall,
            "law_calls": n_law,
        }
        DIAG["solve_calls"].append(rec)
        print(f"  [solve #{len(DIAG['solve_calls'])}] OK  iters={rec['iters']:3d}  "
              f"law_calls={n_law:5d}  wall={wall:6.2f}s", flush=True)
        return result
    except Exception as e:
        info = getattr(self, "info", {}) or {}
        n_law = DIAG["soc_law_calls_summary"]["n_calls"] - n_law_before
        wall = time.time() - t0
        rec = {
            "ok": False,
            "exc_type": type(e).__name__,
            "exc_msg": str(e)[:300],
            "iters": int(info.get("iters", -1)) if isinstance(info, dict) else -1,
            "final_resid_norm": float(info.get("residual_norm", float("nan")))
                if isinstance(info, dict) else float("nan"),
            "wall": wall,
            "law_calls": n_law,
        }
        DIAG["solve_calls"].append(rec)
        print(f"  [solve #{len(DIAG['solve_calls'])}] FAIL "
              f"{type(e).__name__}: {str(e)[:120]}  iters={rec['iters']}  "
              f"law_calls={n_law}  wall={wall:6.2f}s", flush=True)
        raise


ImplicitEquationSolver.solve = _patched_solve

# ---------------------------------------------------------------------------
# Run integrator
# ---------------------------------------------------------------------------
n_aug_locked = cs.y0.size
nl_atol_locked = np.full(n_aug_locked, 1e-8)
nl_rtol_locked = np.full(n_aug_locked, 1e-6)
nl_atol_locked[n_phys_locked:] = 1e-10
nl_rtol_locked[n_phys_locked:] = 0.0

solver_opts_locked = dict(cs.solver_opts)
solver_opts_locked.update(tol=1e-12, max_iter=500, rhs_jac=cs.rhs_jac)

integrator_opts_locked = dict(cs.integrator_opts)
integrator_opts_locked.update({"stages": 2, "use_coupled_newton": True})

print(f"running integrator until t = {TMAX_REPRO:.4e}  (full TMAX={TMAX:.3e})")
t_run0 = time.time()

y_arr = None
try:
    out = solve_nivp.solve_ivp_ns(
        fun=cs.rhs,
        t_span=(0, TMAX_REPRO),
        y0=y0_locked,
        method="radau_iia",
        integrator_opts=integrator_opts_locked,
        projection=cs.projection,
        solver="semismooth_newton",
        projection_opts={"component_slices": cs.component_slices},
        solver_opts=solver_opts_locked,
        adaptive=True,
        h0=H0_ADAPTIVE,
        rtol=ADAPTIVE_RTOL,
        atol=ADAPTIVE_ATOL,
        skip_error_indices=[0],
        active_set_filter=False,
        return_attempts=True,
        adaptive_opts=dict(
            h0=H0_ADAPTIVE, h_min=ADAPTIVE_H_MIN, h_max=ADAPTIVE_H_MAX,
            h_up=2.0, h_down=0.5, method_order=1,
        ),
        nl_atol=nl_atol_locked,
        nl_rtol=nl_rtol_locked,
        component_slices=cs.component_slices,
        verbose=False,
        A=cs.A,
        dae_var_weight="auto",
    )
    t_arr, y_arr, h_arr, _, _, attempts = out
    DIAG["outcome"] = "ok"
    DIAG["t_final"] = float(t_arr[-1])
    DIAG["n_states"] = int(len(t_arr))
    DIAG["n_accepted"] = int(np.asarray(attempts, dtype=bool).sum())
    DIAG["n_rejected"] = int((~np.asarray(attempts, dtype=bool)).sum())
    print(f"OK: t_final={t_arr[-1]:.3e}, states={len(t_arr)}, "
          f"acc={DIAG['n_accepted']}, rej={DIAG['n_rejected']}")
except Exception as e:
    DIAG["outcome"] = "fail"
    DIAG["error"] = {
        "type": type(e).__name__,
        "msg": str(e)[:600],
        "traceback": traceback.format_exc()[-2000:],
    }
    print(f"FAIL: {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# Pollard-Segall analytical comparison (sliding mode only)
# ---------------------------------------------------------------------------
if REPRO_MODE == "sliding" and y_arr is not None:
    DISP_SCALE = SCALE_L * SCALE_EPS
    crack_half_length = CRACK_LENGTH / 2.0
    delta_tau_mpa = abs(tau_used_val) - MU * sigma_N_used
    slip_max_anal = (
        2.0 * delta_tau_mpa * (1.0 - NU) * crack_half_length / G_SHEAR
    )

    # Numerical slip from the final state, on the n_c jump-tangent DOFs.
    final_y = np.asarray(y_arr[-1], dtype=float)
    slip_dimless = final_y[jmpu_t_indices]
    slip_phys = slip_dimless * DISP_SCALE

    coords = np.asarray(
        poro._nodal_interface_info["lambda_coords"], dtype=float
    )
    centered = coords - coords.mean(axis=1, keepdims=True)
    axis = np.linalg.svd(centered, full_matrices=False)[0][:, 0]
    xi_raw = axis @ coords
    xi_phys = xi_raw - 0.5 * (xi_raw.min() + xi_raw.max())
    order = np.argsort(xi_phys)
    xi_phys = xi_phys[order]
    slip_phys = slip_phys[order]
    s_param = xi_phys / crack_half_length

    slip_anal_phys = slip_max_anal * np.sqrt(
        np.clip(1.0 - s_param ** 2, 0.0, None)
    )
    mask_interior = np.abs(s_param) < 0.95
    err_l2 = float(np.linalg.norm(
        np.abs(slip_phys[mask_interior]) - np.abs(slip_anal_phys[mask_interior])
    ))
    ref_l2 = float(np.linalg.norm(np.abs(slip_anal_phys[mask_interior])))
    rel_l2 = err_l2 / max(ref_l2, 1e-30)

    DIAG["pollard"] = {
        "slip_max_anal_m": slip_max_anal,
        "slip_max_fe_m": float(np.max(np.abs(slip_phys))),
        "err_l2_m": err_l2,
        "ref_l2_m": ref_l2,
        "rel_l2_interior": rel_l2,
        "centre_slip_m": float(slip_phys[np.argmin(np.abs(s_param))]),
    }
    print(f"\nPollard-Segall analytical comparison (interior |s|<0.95):")
    print(f"  analytical max slip   = {slip_max_anal:.4e} m")
    print(f"  FE max |slip|         = {np.max(np.abs(slip_phys)):.4e} m")
    print(f"  FE centre slip (s~0)  = {slip_phys[np.argmin(np.abs(s_param))]:.4e} m")
    print(f"  L2 err = {err_l2:.3e},  ref = {ref_l2:.3e},  rel = {rel_l2:.4f}")

DIAG["run_seconds"] = time.time() - t_run0

# ---------------------------------------------------------------------------
# Summarise
# ---------------------------------------------------------------------------
DIAG["n_solve_calls"] = len(DIAG["solve_calls"])
DIAG["n_solve_failures"] = sum(1 for c in DIAG["solve_calls"] if not c["ok"])
DIAG["n_contact_residual_calls"] = len(DIAG["contact_residual_calls"])
DIAG["n_contact_jacobian_calls"] = len(DIAG["contact_jacobian_calls"])
DIAG["n_delassus_calls"] = len(DIAG["delassus_calls"])

if DIAG["contact_residual_calls"]:
    gaps_max = [c["gap_abs_max"] for c in DIAG["contact_residual_calls"]]
    gap_max_pos = [c["gap_max"] for c in DIAG["contact_residual_calls"]]
    n_inactive = [c["n_inactive"] for c in DIAG["contact_residual_calls"]]
    DIAG["gap_abs_max_overall"] = float(max(gaps_max))
    DIAG["gap_max_pos_overall"] = float(max(gap_max_pos))
    DIAG["n_inactive_max"] = int(max(n_inactive))
    DIAG["n_inactive_total"] = int(sum(n_inactive))

if DIAG["delassus_calls"]:
    valid = [c for c in DIAG["delassus_calls"] if c.get("rho_N_max") is not None]
    if valid:
        DIAG["rho_N_max_overall"] = float(max(c["rho_N_max"] for c in valid))
        DIAG["rho_N_min_overall"] = float(min(c["rho_N_min"] for c in valid))

# Trim long lists for the JSON dump
DIAG["contact_residual_calls"] = DIAG["contact_residual_calls"][:50]
DIAG["contact_jacobian_calls"] = DIAG["contact_jacobian_calls"][:50]
DIAG["delassus_calls"] = DIAG["delassus_calls"][:50]
DIAG["solve_calls"] = DIAG["solve_calls"][:200]

with open(LOG_PATH, "w") as f:
    json.dump(DIAG, f, indent=2, default=str)

print(f"\nSUMMARY N_ELEM={N_ELEM}  outcome={DIAG['outcome']}  "
      f"solve_calls={DIAG['n_solve_calls']}  "
      f"solve_failures={DIAG['n_solve_failures']}  "
      f"law_calls={DIAG['soc_law_calls_summary']['n_calls']}  "
      f"law_pinv={DIAG['soc_law_calls_summary']['n_pinv_fallback']}  "
      f"law_linalg_err={DIAG['soc_law_calls_summary']['n_linalg_err']}")
if "gap_abs_max_overall" in DIAG:
    print(f"  gap_abs_max={DIAG['gap_abs_max_overall']:.3e}  "
          f"gap_max_pos={DIAG['gap_max_pos_overall']:.3e}  "
          f"n_inactive_max={DIAG['n_inactive_max']}/{n_c}")
if "rho_N_max_overall" in DIAG:
    print(f"  rho_N range: [{DIAG['rho_N_min_overall']:.3e}, "
          f"{DIAG['rho_N_max_overall']:.3e}]")

print(f"diagnostics dumped to {LOG_PATH}")
