"""Verify slip-vs-viscosity claim for embedded-crack Mohr-Coulomb sliding test.

Mirrors the sliding setup in `examples/embedded_crack_mohr_coulomb_ncp.ipynb`
with ALPHA_BIOT = 0 (pore-pressure coupling off) and sweeps BULK_MU_V across
several orders of magnitude.

The claim being verified
------------------------
For prestress placed outside the Coulomb cone, slip relaxes toward an elastic
QS limit set by the finite 1×1 biaxial geometry — emphatically NOT the
infinite-medium Sneddon value, which is invalid here.  Damping (μ_v, λ_v)
should change *only the relaxation transient*, not the QS limit.  The caveat
is that for very high viscosity the relaxation time τ_v ≈ (μ_v+λ_v)/(λ+2μ)
can exceed TMAX, leaving the run mid-transient and *appearing* to give
viscosity-dependent slip — a sampling artifact, not physics.

Per case we record:
  * slip_max         max |[[u_t]]| over the run (rotated [[u_t]] DOFs)
  * drift_inf        ||y(t_final)−y(t_final−Δt)||_inf / Δt
                     small ⇒ QS reached; large ⇒ run truncated mid-transient
  * rhs_inf          ||rhs(t_final, y_final)||_inf
  * tau_visc         (μ_v + λ_v) / (λ + 2μ)
  * zeta_v           a heuristic over-damping marker τ_v · ω_n where
                     ω_n is the modal frequency of the *constrained* slip
                     mode (a Schur reduction onto the slip jump subspace)

Two horizons:
  * TMAX_SHORT — fixed (notebook default 10).  Exposes the sampling artifact.
  * TMAX_LONG  — sized to ≥ 30·τ_v (capped) and re-run with a relaxed h_max
                 so a deeply overdamped relaxation can be resolved cheaply.

If the claim holds, slip_max at TMAX_LONG is independent of (μ_v, λ_v) to
within a few percent; slip_max at TMAX_SHORT drops monotonically as μ_v rises
past TMAX_SHORT × (λ + 2μ).
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')

import json
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from skfem.models.elasticity import lame_parameters

from poroelasticity import CGPoroelastostatics, CrackMeshBuilder

import solve_nivp
from solve_nivp.projected_radau_contact import build_projected_radau_contact


# --- Material / geometry / loading (matches notebook cell 2) ---
NU          = 0.25
G_SHEAR     = 22.0e3
E_YOUNG     = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT  = 0.0                       # pore pressure coupling OFF
BETA_FLUID  = 8.5e-5
ETA_FLUID   = 2.0e-18 / 3600.0
K_PERM      = 1.0e-15 * 1.0e-6
DENSITY     = 1.05
SCALE_L     = 1.0
SCALE_EPS   = 1.0e-3

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

LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
Mmod = LAM + 2.0 * MU_LAME
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


def build_sliding_case(bulk_mu_v, bulk_lam_v):
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
        mesh=mesh, element_p=el_p, element_u=el_u, params=PARAMS,
        crack=crack, model_params=CRACK_MODEL_PARAMS,
        intorder=6, scales=(SCALE_L, SCALE_EPS), bc=bc,
        P_scale=None, verbose=False, free_memory=False,
        enforcement_type='nodal', apply_transform=True,
        include_taylor=False, include_hessian=False, include_sbm=False,
        density=DENSITY, crack_law='nonsmooth', rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': bulk_mu_v, 'lam_v': bulk_lam_v},
    )

    poro.strip_multiplier_dynamics()
    projection, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta, density=DENSITY)

    A_dyn       = dyn['A']
    rhs_dyn     = dyn['rhs']
    rhs_jac_dyn = dyn['rhs_jac']
    n_base      = dyn['n_base']
    comp_slices = dyn['component_slices']

    Np = poro.basis_p.N
    Nu = poro.basis_u.N
    info = poro._transform_info
    n_c = int(poro.n_lambda_q)

    lam_s_sl = meta['lam_s_sl']
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
    D = T_u[jmpu_local, :].tocsr()

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

    contacts = [
        {'vel_normal_idx': k, 'vel_tangential_idx': [n_c + k], 'mu': MU, 'e': 0.0}
        for k in range(n_c)
    ]

    n_phys = A_dyn.shape[0]
    n_extract = 2 * n_c

    gap_extract_dyn = sp.csr_matrix(
        (np.ones(n_extract), (np.arange(n_extract), jmpu_all)),
        shape=(n_extract, n_phys),
    )
    vel_extract_dyn = sp.hstack(
        [sp.csr_matrix((n_extract, n_base)), D],
        format='csr',
    )
    B_c_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm],
        format='csr',
    )
    B_n_dyn = B_c_dyn[:, 0::2]

    off_avgp = info['off_avgp']
    n_intf_p = info['n_intf_p']
    assert n_intf_p == n_c
    p_gamma_extract_dyn = sp.csr_matrix(
        (np.ones(n_c), (np.arange(n_c), off_avgp + np.arange(n_c))),
        shape=(n_c, n_phys),
    )
    pressure_normal_correction = -ALPHA_BIOT * (B_n_dyn @ p_gamma_extract_dyn)
    pressure_normal_correction.eliminate_zeros()

    def rhs_dyn_eff(t, y, *extra):
        return rhs_dyn(t, y, *extra) + pressure_normal_correction @ y

    def rhs_jac_dyn_eff(t, y, *extra):
        return rhs_jac_dyn(t, y, *extra) + pressure_normal_correction

    sigma_N_sliding, tau_sliding_val = mohr_tractions(THETA_SLIDING)
    s0_val = np.full(n_c, sigma_N_sliding / Sigma_scale)
    w0_val = np.full(n_c, tau_sliding_val / Sigma_scale)

    def get_s0(y):
        return s0_val

    def get_w0(y, k):
        return np.array([w0_val[int(k)]])

    flux_constraint = meta['constraints'][0]
    n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
    zero_lam_s = {
        'g': lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
        'dg_dy': lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
        'y_slice': lam_s_sl,
        'q_slice': lam_s_sl,
    }

    cs = build_projected_radau_contact(
        A_dyn, rhs_dyn_eff, np.zeros(A_dyn.shape[0]),
        contacts=contacts,
        C_extract=gap_extract_dyn,
        D_extract=vel_extract_dyn,
        B=B_c_dyn,
        constraints=[flux_constraint, zero_lam_s],
        component_slices=comp_slices,
        rhs_jac=rhs_jac_dyn_eff,
        get_s0=get_s0, get_w0=get_w0,
        normal_r='auto', friction_r='auto',
        endpoint_inactive_handling='natural_map',
        reported_reaction_units='force',
    )

    return {
        'cs': cs,
        'A_dyn': A_dyn,
        'rhs_dyn_eff': rhs_dyn_eff,
        'rhs_jac_dyn_eff': rhs_jac_dyn_eff,
        'n_base': n_base,
        'n_phys': n_phys,
        'n_c': n_c,
        'Nu': Nu,
        'jmpu_n_idx': jmpu_n_idx,
        'jmpu_t_idx': jmpu_t_idx,
        'B_c_dyn': B_c_dyn,
        'vel_extract_dyn': vel_extract_dyn,
        'u_state_indices': u_idx,
    }


def slip_mode_compliance(case):
    """Schur-reduced slip-mode compliance: solve the elastostatic problem
    for unit uniform tangential drive *with the gap and Dirichlet rows held
    Dirichlet*, project (M, C, K) onto the resulting shape, and return the
    Rayleigh-quotient ω_n, ζ.  This is heuristic: it captures *one* slip mode
    only and ignores cone activity, but the trend ζ ∝ μ_v is rigorous.
    """
    A_dyn = case['A_dyn']
    n_base = case['n_base']
    n_phys = case['n_phys']
    n_c = case['n_c']
    rhs_jac_dyn_eff = case['rhs_jac_dyn_eff']
    B_c_dyn = case['B_c_dyn']
    u_idx = case['u_state_indices']

    J0 = rhs_jac_dyn_eff(0.0, np.zeros(n_phys)).tocsr()
    A0 = A_dyn.tocsr()

    mom_rows  = slice(n_base, n_phys)
    base_cols = slice(0, n_base)
    vel_cols  = slice(n_base, n_phys)

    K_op = -(J0[mom_rows, :][:, base_cols]).tocsr()
    C_op = -(J0[mom_rows, :][:, vel_cols]).tocsr()
    M_op = (A0[mom_rows, :][:, vel_cols]).tocsr()
    K_uu = K_op[:, u_idx].tocsr()

    row_max = np.abs(K_uu).max(axis=1).toarray().ravel()
    free_idx = np.where(row_max > 0.0)[0]
    if free_idx.size == 0:
        return {'error': 'no free DOFs'}

    K_ff = K_uu[free_idx, :][:, free_idx].tocsc()
    M_ff = M_op[free_idx, :][:, free_idx].tocsc()
    C_ff = C_op[free_idx, :][:, free_idx].tocsc()

    e_t = np.zeros(2 * n_c)
    e_t[1::2] = 1.0
    rhs_static = (B_c_dyn @ e_t)[mom_rows][free_idx]

    try:
        v = spla.spsolve(K_ff, rhs_static)
    except Exception as exc:
        return {'error': f'spsolve failed: {exc!r}'}
    if not np.all(np.isfinite(v)):
        return {'error': 'non-finite v'}

    m_mode = float(v @ (M_ff @ v))
    c_mode = float(v @ (C_ff @ v))
    k_mode = float(v @ (K_ff @ v))
    if m_mode <= 0 or k_mode <= 0:
        return {'error': f'non-positive Rayleigh (m={m_mode:.3e}, k={k_mode:.3e})'}

    omega_n = float(np.sqrt(k_mode / m_mode))
    zeta = float(c_mode / (2.0 * np.sqrt(k_mode * m_mode)))
    return {
        'm_mode': m_mode, 'c_mode': c_mode, 'k_mode': k_mode,
        'omega_n': omega_n, 'zeta_global_mode': zeta,
    }


def run_case(case, tmax, h_max=1.0e-3, h0=None):
    cs = case['cs']
    n_phys = case['n_phys']
    n_aug = cs.y0.size

    nl_atol = np.full(n_aug, 1e-8)
    nl_rtol = np.full(n_aug, 1e-6)
    nl_atol[n_phys:] = 1e-10
    nl_rtol[n_phys:] = 0.0

    solver_opts = dict(cs.solver_opts)
    solver_opts.update(
        tol=1e-8, max_iter=500, rhs_jac=cs.rhs_jac,
        linear_solver='petsc',
        petsc_options={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        },
    )
    integrator_opts = dict(cs.integrator_opts)
    integrator_opts.update({'stages': 2, 'use_coupled_newton': True})

    h0_eff = h0 if h0 is not None else min(0.01, h_max)

    t0 = time.time()
    t, y, h, fk, info, attempts = solve_nivp.solve_ivp_ns(
        fun=cs.rhs, t_span=(0, tmax), y0=cs.y0.copy(),
        method='radau_iia', integrator_opts=integrator_opts,
        projection=cs.projection, solver='semismooth_newton',
        projection_opts={'component_slices': cs.component_slices},
        solver_opts=solver_opts,
        adaptive=True, h0=h0_eff, rtol=1e-4, atol=1e-6,
        skip_error_indices=[0], active_set_filter=False, return_attempts=True,
        adaptive_opts=dict(
            h0=h0_eff, h_min=1e-8, h_max=h_max,
            h_up=2.0, h_down=0.5, method_order=1,
        ),
        nl_atol=nl_atol, nl_rtol=nl_rtol,
        component_slices=cs.component_slices,
        verbose=False, A=cs.A, dae_var_weight='auto',
    )
    elapsed = time.time() - t0

    y_arr = np.asarray(y, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    slip = y_arr[:, case['jmpu_t_idx']]
    slip_max_nondim = float(np.max(np.abs(slip)))
    slip_max_phys = slip_max_nondim * DISP_SCALE
    slip_final_nondim = float(np.max(np.abs(slip[-1])))

    rhs_at_final = cs.rhs(t_arr[-1], y_arr[-1])
    rhs_inf = float(np.max(np.abs(rhs_at_final)))

    if len(t_arr) >= 2:
        dt_last = max(t_arr[-1] - t_arr[-2], 1e-30)
        drift_inf = float(np.max(np.abs(y_arr[-1] - y_arr[-2]))) / dt_last
    else:
        drift_inf = float('nan')

    return {
        'elapsed_s': elapsed,
        't_final': float(t_arr[-1]),
        'n_states': len(t_arr),
        'h_max_used': h_max,
        'slip_max_phys': slip_max_phys,
        'slip_max_nondim': slip_max_nondim,
        'slip_final_nondim': slip_final_nondim,
        'rhs_inf': rhs_inf,
        'drift_inf': drift_inf,
    }


def main():
    # Sweep — three regimes: τ_v ≪ TMAX (fully relaxed at notebook horizon),
    # τ_v ≈ TMAX (borderline), τ_v ≫ TMAX (TMAX-truncated, exposes the artifact).
    visc_values = [1.0e2, 1.0e4, 1.0e6, 1.0e7]

    TMAX_SHORT = 10.0

    rows = []
    for mu_v in visc_values:
        lam_v = mu_v
        tau_visc = (mu_v + lam_v) / Mmod
        # Use a coarse h_max so the run finishes; adaptive controller pulls it
        # down if Newton struggles.  In QS coast-out the controller will push
        # h to h_max anyway, so we want h_max larger than the relaxation step
        # budget rather than τ_v/10.
        h_max_short = 5.0e-2
        h_max_long  = max(min(tau_visc / 4.0, 5.0), 5.0e-2)
        tmax_long = float(min(max(30.0 * tau_visc, TMAX_SHORT), 5000.0))

        print(f"\n=== μ_v = λ_v = {mu_v:.1e}    τ_v = {tau_visc:.3e}    "
              f"TMAX_long = {tmax_long:.1f}    h_max_long = {h_max_long:.1e} ===",
              flush=True)

        # Short run.
        case = build_sliding_case(mu_v, lam_v)
        print(f"  short run  TMAX={TMAX_SHORT}, h_max={h_max_short:.1e}…",
              flush=True)
        run_short = run_case(case, TMAX_SHORT, h_max=h_max_short)
        print(f"    elapsed={run_short['elapsed_s']:.1f}s, n_states={run_short['n_states']},  "
              f"slip_max={run_short['slip_max_nondim']:.4e},  "
              f"drift={run_short['drift_inf']:.2e}")

        # Long run (rebuild — projected_radau_contact mutates internal state).
        case_long = build_sliding_case(mu_v, lam_v)
        print(f"  long  run  TMAX={tmax_long}, h_max={h_max_long:.1e}…",
              flush=True)
        run_long = run_case(case_long, tmax_long, h_max=h_max_long)
        print(f"    elapsed={run_long['elapsed_s']:.1f}s, n_states={run_long['n_states']},  "
              f"slip_max={run_long['slip_max_nondim']:.4e},  "
              f"drift={run_long['drift_inf']:.2e}")

        rows.append({
            'mu_v': mu_v, 'lam_v': lam_v, 'tau_visc': tau_visc,
            'tmax_short': TMAX_SHORT, 'tmax_long': tmax_long,
            'short': run_short, 'long': run_long,
        })

        # Persist after each case so partial results survive crashes.
        with open('/tmp/damping_factor_verify.json', 'w') as f:
            json.dump(rows, f, indent=2, default=float)

    print("\n" + "=" * 100)
    print(f"{'mu_v':>9} {'tau_v':>10} {'TMAX_short/tau_v':>16} | "
          f"{'slip_short':>12} {'drift_short':>12} | "
          f"{'TMAX_long':>10} {'slip_long':>12} {'drift_long':>12}")
    print("-" * 100)
    for r in rows:
        ratio = r['tmax_short'] / max(r['tau_visc'], 1e-30)
        print(f"{r['mu_v']:>9.1e} {r['tau_visc']:>10.2e} {ratio:>16.2e} | "
              f"{r['short']['slip_max_nondim']:>12.4e} "
              f"{r['short']['drift_inf']:>12.2e} | "
              f"{r['tmax_long']:>10.1f} "
              f"{r['long']['slip_max_nondim']:>12.4e} "
              f"{r['long']['drift_inf']:>12.2e}")
    print("=" * 100)

    out_path = '/tmp/damping_factor_verify.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2, default=float)
    print(f"\nWrote sweep result rows -> {out_path}")


if __name__ == '__main__':
    main()
