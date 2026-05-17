"""Bypass-the-cone diagnostic for the embedded-crack sliding test.

The notebook delivers slip = 0.43 × Sneddon at a/L = 0.05, where the
finite-domain correction should be ≤ 5–10%.  This script isolates *where*
the deficit lives by skipping the projected-Radau contact entirely and
solving a single pure linear-elastostatic problem under a prescribed
uniform tangential surface traction Δτ_geom on the crack interface.

Three slip values get compared:
  (1) Sneddon:               2 (1−ν) Δτ a / G   (infinite medium)
  (2) Pure-elastic FE:       K · z = B · r_drive,  where r_drive is
                             a hand-rolled (r_n=0, r_t=Δτ_geom) per
                             contact node — no cone, no friction.
  (3) Notebook cone solve:   the existing projected-Radau result.

Outcome map:
  - (2) ≈ Sneddon     ⇒  gap is in the contact-law projection;
                         FE assembly is fine; PML doesn't help.
  - (2) ≈ (3)         ⇒  gap is in the FE assembly itself
                         (storage convention, rotation frame, B weights,
                          interface conventions); PML still doesn't help.
  - (2) ≈ 0.5 × Sneddon ⇒ classic factor-of-2 storage mismatch.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem.models.elasticity import lame_parameters
from poroelasticity import CGPoroelastostatics, CrackMeshBuilder, MaterialParams


# --- Notebook configuration (mirror cell 4 with BOX_SCALE = 4) ----------
NU          = 0.3
G_SHEAR     = 10.0e3
E_YOUNG     = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT  = 0.0
BETA_FLUID  = 8.5e-5
ETA_FLUID   = 2.0e-18 / 3600.0
K_PERM      = 1.0e-15 * 1.0e-6
DENSITY_PHYS = 2.7e-3
SCALE_L      = 1.0
SCALE_EPS    = 1.0e-3
BULK_MU_V    = 1.0e-1
BULK_LAM_V   = 1.0e-1

XMIN, XMAX = 0.0, 1.0
YMIN, YMAX = 0.0, 1.0
N_ELEM     = 20

CRACK_X0     = 0.5
CRACK_Y0     = 0.5
CRACK_LENGTH = 0.4

SIGMA_RIGHT = 10.0
SIGMA_TOP   = 30.0
MU          = 0.3

CRACK_T_ROBIN = 1e5
CRACK_MODEL_PARAMS = {'T': CRACK_T_ROBIN, 'tangential_flow': False}

# Match user's BOX_SCALE = 4 setup
BOX_SCALE = 4.0
XMAX *= BOX_SCALE
YMAX *= BOX_SCALE
CRACK_X0 = 0.5 * (XMIN + XMAX)
CRACK_Y0 = 0.5 * (YMIN + YMAX)
N_ELEM   = max(int(round(N_ELEM * BOX_SCALE)), N_ELEM)

LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
MATERIAL = MaterialParams(
    mu=MU_LAME, lam=LAM, alpha=ALPHA_BIOT,
    beta=BETA_FLUID, C=K_PERM / ETA_FLUID, rho=DENSITY_PHYS,
)
Mmod        = LAM + 2.0 * MU_LAME
Sigma_scale = Mmod * SCALE_EPS
DISP_SCALE  = SCALE_L * SCALE_EPS

S1 = max(SIGMA_RIGHT, SIGMA_TOP)
S3 = min(SIGMA_RIGHT, SIGMA_TOP)
THETA_SLIDING = np.radians(27.0)
CRACK_THETA_SLIDING = np.pi / 2.0 - THETA_SLIDING


def mohr_tractions(theta):
    sigma_N = (S1 + S3) / 2.0 + (S1 - S3) / 2.0 * np.cos(2.0 * theta)
    tau     = (S1 - S3) / 2.0 * np.sin(2.0 * theta)
    return sigma_N, tau


def build_sliding_poro():
    builder = CrackMeshBuilder(
        XMIN, XMAX, YMIN, YMAX, N_ELEM,
        crack_theta=CRACK_THETA_SLIDING,
        crack_x0=CRACK_X0, crack_y0=CRACK_Y0,
        crack_length=CRACK_LENGTH,
        element_type='tri', conforming=True, verbose=False,
    )
    mesh, el_p, el_u, crack, h_mesh = builder.build()
    bc = {
        'v1': {'left': 0.0},
        'v2': {'bottom': 0.0},
        'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0},
    }
    poro = CGPoroelastostatics(
        mesh=mesh, element_p=el_p, element_u=el_u,
        params=PARAMS, material=MATERIAL,
        crack=crack, model_params=CRACK_MODEL_PARAMS,
        intorder=6, scales=(SCALE_L, SCALE_EPS), bc=bc,
        P_scale=None, verbose=False, free_memory=False,
        enforcement_type='nodal', apply_transform=True,
        include_taylor=False, include_hessian=False, include_sbm=False,
        crack_law='nonsmooth', rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': BULK_MU_V, 'lam_v': BULK_LAM_V},
    )
    poro.strip_multiplier_dynamics()
    _, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta)
    return poro, meta, dyn, h_mesh


def assemble_B_and_indexing(poro, meta, dyn):
    """Replicate the cells-27/28 wiring needed to map a contact-traction
    vector (interleaved [r_n0, r_t0, r_n1, r_t1, …]) to the bulk forcing
    vector in the descriptor frame."""
    n_base = dyn['n_base']
    A_dyn  = dyn['A']
    Np = poro.basis_p.N
    Nu = poro.basis_u.N
    info = poro._transform_info
    n_c = int(poro.n_lambda_q)

    off_jmpu  = info['off_jmpu']
    n_iu      = info['n_intf_u']
    off_jmpls = info['off_jmpls']
    n_lsig    = info['n_lam_sig']

    jmpu_n_idx = np.arange(off_jmpu, off_jmpu + n_c)
    jmpu_t_idx = np.arange(off_jmpu + n_c, off_jmpu + n_iu)
    jmpu_all   = np.concatenate([jmpu_n_idx, jmpu_t_idx])
    jmpls_cols = np.arange(off_jmpls, off_jmpls + n_lsig)

    T_u = dyn['T_u']
    u_idx = dyn['u_state_indices']
    jmpu_local = np.searchsorted(u_idx, jmpu_all)
    assert np.all(u_idx[jmpu_local] == jmpu_all)

    R_v = dyn.get('R_v')
    A_csr = poro.A.tocsr()
    B_u = (A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
    if R_v is not None:
        B_u = (R_v @ B_u).tocsr()

    dirichlet_local = np.array(
        [d - Np for d in np.asarray(getattr(poro, '_dirichlet_dof_set', []), dtype=int)
         if Np <= d < Np + Nu],
        dtype=int,
    )
    if dirichlet_local.size:
        B_u = B_u.tolil()
        B_u[dirichlet_local, :] = 0.0
        B_u = B_u.tocsr()

    perm = [idx for k in range(n_c) for idx in (k, n_c + k)]
    B_u_perm = B_u[:, perm].tocsr()

    n_phys = A_dyn.shape[0]
    B_c_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm],
        format='csr',
    )
    return {
        'A_dyn': A_dyn, 'n_base': n_base, 'n_phys': n_phys,
        'Nu': Nu, 'n_c': n_c,
        'jmpu_n_idx': jmpu_n_idx, 'jmpu_t_idx': jmpu_t_idx,
        'B_c_dyn': B_c_dyn, 'u_state_indices': u_idx,
    }


def solve_static_elastic(poro, meta, dyn, idx, r_drive):
    """Solve K · z = B · r_drive on the displacement subspace.
    K is taken from the descriptor Jacobian at y=0 with the standard
    descriptor sign convention  M v̇ = -K z + ...  ⇒  K = -∂rhs/∂z."""
    n_phys = idx['n_phys']
    n_base = idx['n_base']
    A_dyn = idx['A_dyn']
    Nu = idx['Nu']

    rhs_jac_dyn = dyn['rhs_jac']
    J0 = rhs_jac_dyn(0.0, np.zeros(n_phys)).tocsr()

    mom_rows  = slice(n_base, n_phys)
    base_cols = slice(0, n_base)

    K_full = -(J0[mom_rows, :][:, base_cols]).tocsr()   # Nu × n_base
    u_idx = idx['u_state_indices']
    K_uu = K_full[:, u_idx].tocsr()

    f_full = (idx['B_c_dyn'] @ r_drive)
    f_u = np.asarray(f_full[mom_rows]).ravel()

    # Filter out Dirichlet rows (zeroed in K_uu by the descriptor strip).
    row_max = np.abs(K_uu).max(axis=1).toarray().ravel()
    free_idx = np.where(row_max > 0.0)[0]
    K_ff = K_uu[free_idx, :][:, free_idx].tocsc()
    f_free = f_u[free_idx]

    z_free = spla.spsolve(K_ff, f_free)
    if not np.all(np.isfinite(z_free)):
        raise RuntimeError("static solve produced non-finite displacements")

    z_u = np.zeros(Nu)
    z_u[free_idx] = z_free
    z = np.zeros(n_base)
    z[u_idx] = z_u
    return z, z_u, free_idx


def main():
    print("=== Bypass-the-cone Sneddon diagnostic ===")
    print(f"BOX_SCALE = {BOX_SCALE},  domain = [0,{XMAX}] × [0,{YMAX}]")
    print(f"a/L = {0.5*CRACK_LENGTH / max(XMAX, YMAX):.3f}")
    print(f"crack θ = {np.degrees(CRACK_THETA_SLIDING):.1f}° from y-axis")
    print()

    poro, meta, dyn, h_mesh = build_sliding_poro()
    idx = assemble_B_and_indexing(poro, meta, dyn)
    n_c = idx['n_c']

    sigma_N, tau = mohr_tractions(THETA_SLIDING)
    delta_tau_phys = abs(tau) - MU * sigma_N
    print(f"Mohr prestress:  σ_N = {sigma_N:.3f},  |τ| = {abs(tau):.3f},  μσ_N = {MU*sigma_N:.3f}")
    print(f"Geometric stress drop  Δτ_geom = {delta_tau_phys:.4f} MPa")
    print(f"Sigma_scale = {Sigma_scale:.2f},  DISP_SCALE = {DISP_SCALE:.1e}")
    print(f"contacts: n_c = {n_c}")
    print()

    a_half = CRACK_LENGTH / 2.0
    sneddon_phys = 2.0 * delta_tau_phys * (1.0 - NU) * a_half / G_SHEAR
    print(f"Sneddon prediction (infinite medium):  [[u_t]]_max = {sneddon_phys:.4e} m")
    print()

    delta_tau_nd = delta_tau_phys / Sigma_scale

    # ------------------------------------------------------------------
    # PROBE A — apply r_t = +Δτ_nd at every contact node (interleaved).
    # ------------------------------------------------------------------
    r_A = np.zeros(2 * n_c)
    r_A[1::2] = delta_tau_nd
    z_A, z_u_A, free_A = solve_static_elastic(poro, meta, dyn, idx, r_A)
    slip_A = z_A[idx['jmpu_t_idx']] * DISP_SCALE
    slip_A_max = float(np.max(np.abs(slip_A)))

    print(f"PROBE A:  r_t = +Δτ_nd  (Δτ_nd = {delta_tau_nd:.4e})")
    print(f"  max |[[u_t]]|_phys = {slip_A_max:.4e} m")
    print(f"  ratio vs Sneddon   = {slip_A_max/sneddon_phys:.3f}")
    print(f"  slip profile       = {slip_A * 1e6}  (µm)")
    print()

    # ------------------------------------------------------------------
    # PROBE B — sign flip:  r_t = −Δτ_nd
    # ------------------------------------------------------------------
    r_B = np.zeros(2 * n_c)
    r_B[1::2] = -delta_tau_nd
    z_B, _, _ = solve_static_elastic(poro, meta, dyn, idx, r_B)
    slip_B = z_B[idx['jmpu_t_idx']] * DISP_SCALE
    slip_B_max = float(np.max(np.abs(slip_B)))
    print(f"PROBE B:  r_t = −Δτ_nd  (sign flip)")
    print(f"  max |[[u_t]]|_phys = {slip_B_max:.4e} m")
    print(f"  ratio vs Sneddon   = {slip_B_max/sneddon_phys:.3f}")
    print()

    # ------------------------------------------------------------------
    # PROBE C — factor-of-2 storage convention test
    # ------------------------------------------------------------------
    r_C = np.zeros(2 * n_c)
    r_C[1::2] = 2.0 * delta_tau_nd     # double, in case storage is r_phys = r_stored/2
    z_C, _, _ = solve_static_elastic(poro, meta, dyn, idx, r_C)
    slip_C = z_C[idx['jmpu_t_idx']] * DISP_SCALE
    slip_C_max = float(np.max(np.abs(slip_C)))
    print(f"PROBE C:  r_t = +2·Δτ_nd  (factor-2 storage probe)")
    print(f"  max |[[u_t]]|_phys = {slip_C_max:.4e} m")
    print(f"  ratio vs Sneddon   = {slip_C_max/sneddon_phys:.3f}")
    print()

    # ------------------------------------------------------------------
    # PROBE D — also drive r_n = +Δσ_n_radial on all nodes,
    # to mimic radial cone projection of (s0, w0) onto the cone boundary.
    # ------------------------------------------------------------------
    lam_proj = (abs(tau) - MU * sigma_N) / (1.0 + MU**2)
    dn_phys = MU * lam_proj   # σ_N perturbation along radial projection
    dt_phys = -lam_proj       # τ perturbation; sign so it opposes prestress
    r_D = np.zeros(2 * n_c)
    r_D[0::2] = dn_phys / Sigma_scale
    r_D[1::2] = dt_phys / Sigma_scale
    z_D, _, _ = solve_static_elastic(poro, meta, dyn, idx, r_D)
    slip_D = z_D[idx['jmpu_t_idx']] * DISP_SCALE
    slip_D_max = float(np.max(np.abs(slip_D)))
    print(f"PROBE D:  radial-projection r_pert  (Δσ_n={dn_phys:.4f}, Δτ={dt_phys:.4f} MPa)")
    print(f"  max |[[u_t]]|_phys = {slip_D_max:.4e} m")
    print(f"  ratio vs Sneddon   = {slip_D_max/sneddon_phys:.3f}")
    print()

    # ------------------------------------------------------------------
    # B-matrix interface-weight diagnostics
    # ------------------------------------------------------------------
    B = idx['B_c_dyn']
    # Norm of B-column (per-contact) — gives the "effective traction → bulk
    # forcing" gain.  Compare normal vs tangential columns.
    col_norms_n = np.array([sp.linalg.norm(B[:, 2*k])     for k in range(n_c)])
    col_norms_t = np.array([sp.linalg.norm(B[:, 2*k + 1]) for k in range(n_c)])
    print("B column-norm diagnostics:")
    print(f"  ||B[:, n]|| range    : [{col_norms_n.min():.3e}, {col_norms_n.max():.3e}]")
    print(f"  ||B[:, t]|| range    : [{col_norms_t.min():.3e}, {col_norms_t.max():.3e}]")
    print(f"  ratio normal/tangent : {col_norms_n.mean()/max(col_norms_t.mean(),1e-30):.4f}")
    print()

    print("=== Interpretation ===")
    print(f"  Sneddon                : {sneddon_phys:.4e}")
    print(f"  Probe A (r_t=+Δτ)      : {slip_A_max:.4e}  (ratio {slip_A_max/sneddon_phys:.3f})")
    print(f"  Probe C (r_t=+2Δτ)     : {slip_C_max:.4e}  (ratio {slip_C_max/sneddon_phys:.3f})")
    print(f"  Probe D (radial proj)  : {slip_D_max:.4e}  (ratio {slip_D_max/sneddon_phys:.3f})")
    print(f"  Notebook cone (latest) : 3.9221e-06       (ratio 0.429)")
    print()
    print("If A ≈ Sneddon  ⇒  contact-law projection is the issue (fix the projection).")
    print("If C ≈ Sneddon  ⇒  factor-of-2 storage mismatch (fix conversion).")
    print("If D ≈ notebook ⇒  cone is doing its job; FE assembly is fine; the gap is")
    print("                   geometric/projection-induced and the analytical reference")
    print("                   should be Δτ_eff = λ_proj instead of Δτ_geom.")
    print("If A ≪ Sneddon  ⇒  FE assembly: B weights, rotation frame, or storage units.")


if __name__ == '__main__':
    main()
