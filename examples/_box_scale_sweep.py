"""BOX_SCALE convergence sweep for the embedded-crack sliding test.

Mechanics-only (ALPHA_BIOT=0).  For the same crack geometry (length=0.4) we
sweep the bulk box size (1, 2, 4, ...) keeping mesh resolution near the crack
roughly constant (N_ELEM scales with BOX_SCALE).  Theory: as a/L decreases the
slip should approach the Sneddon/Comninou infinite-medium upper bound from
below.  The user observes the opposite — debug it.

Usage:
    python _box_scale_sweep.py [BOX_SCALE...]

Outputs go to /tmp/box_sweep/.
"""
from __future__ import annotations

import os, sys, time, json
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ['MPLBACKEND'] = 'Agg'
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from skfem.models.elasticity import lame_parameters
from poroelasticity import CGPoroelastostatics, CrackMeshBuilder
import solve_nivp
from solve_nivp.projected_radau_contact import build_projected_radau_contact


# ── Configuration mirrors the notebook (ALPHA=0, sealed crack) ────────────
NU          = 0.25
G_SHEAR     = 22.0e3
E_YOUNG     = 2.0 * G_SHEAR * (1.0 + NU)
ALPHA_BIOT  = 0.0                # decoupled
BETA_FLUID  = 8.5e-5
ETA_FLUID   = 2.0e-18 / 3600.0
K_PERM      = 1.0e-15 * 1.0e-6
DENSITY     = 1.05
BULK_MU_V   = 1.0e-1
BULK_LAM_V  = 1.0e-1
SCALE_L     = 1.0
SCALE_EPS   = 1.0e-3

XMIN_BASE, XMAX_BASE = 0.0, 1.0
YMIN_BASE, YMAX_BASE = 0.0, 1.0
N_ELEM_BASE = 20
CRACK_LENGTH = 0.4
SIGMA_RIGHT = 10.0
SIGMA_TOP   = 30.0
MU = 0.3
TMAX = 5.0
H_FIXED = 0.01

THETA_SLIDING = np.radians(27.0)
CRACK_T_ROBIN = 0.0


def mohr_tractions(theta, S1, S3):
    sigma_N = (S1 + S3) / 2 + (S1 - S3) / 2 * np.cos(2 * theta)
    tau     = (S1 - S3) / 2 * np.sin(2 * theta)
    return sigma_N, tau


def _filter_surrogate_by_alignment(crack, *, align_min: float = 0.95,
                                    dmid_over_h_max: float = 0.1) -> dict:
    """Drop surrogate facets that are not well-aligned with the crack tangent.

    The library default ``align_min=0.01`` accepts almost-perpendicular
    edges, which lets in spurious "side-spur" facets at the crack tips.
    Returns counts of kept/dropped facets.
    """
    fids_full = np.asarray(crack.surrogate_facets, dtype=int)
    keep = []
    dropped = []
    for fid in fids_full:
        info = crack._facet_info[int(fid)]
        align = info['align']
        h = max(info['h'], 1e-15)
        if align >= align_min and (info['dmid'] / h) <= dmid_over_h_max:
            keep.append(int(fid))
        else:
            dropped.append((int(fid), align, info['dmid'], info['h']))
    crack._surrogate_fids = np.array(keep, dtype=int)
    return {'kept': len(keep), 'dropped': len(dropped),
            'dropped_info': dropped}


def run_box(box_scale: float, outdir: str, *, tmax: float = TMAX,
            filter_surrogate: bool = False) -> dict:
    xmax = XMAX_BASE * box_scale
    ymax = YMAX_BASE * box_scale
    n_elem = max(int(round(N_ELEM_BASE * box_scale)), N_ELEM_BASE)
    crack_x0 = 0.5 * (XMIN_BASE + xmax)
    crack_y0 = 0.5 * (YMIN_BASE + ymax)
    crack_theta_sliding = np.pi / 2 - THETA_SLIDING

    LAM, MU_LAME = lame_parameters(E_YOUNG, NU)
    PARAMS = (MU_LAME, LAM, ALPHA_BIOT, BETA_FLUID, K_PERM / ETA_FLUID)
    Mmod = LAM + 2 * MU_LAME
    Sigma_scale = Mmod * SCALE_EPS

    S1 = max(SIGMA_RIGHT, SIGMA_TOP)
    S3 = min(SIGMA_RIGHT, SIGMA_TOP)
    sigma_N_sliding, tau_sliding_val = mohr_tractions(THETA_SLIDING, S1, S3)
    delta_tau = abs(tau_sliding_val) - MU * sigma_N_sliding
    a = CRACK_LENGTH / 2.0
    sneddon_max_slip_phys = 2.0 * delta_tau * (1.0 - NU) * a / G_SHEAR

    print(f"[box={box_scale}] domain=[{XMIN_BASE},{xmax}]x[{YMIN_BASE},{ymax}], "
          f"n_elem={n_elem}, a/L={a/max(xmax-XMIN_BASE,ymax-YMIN_BASE):.4f}, "
          f"Δτ={delta_tau:.3f} MPa, Sneddon max slip = {sneddon_max_slip_phys:.3e} m")

    builder = CrackMeshBuilder(
        XMIN_BASE, xmax, YMIN_BASE, ymax, n_elem,
        crack_theta=crack_theta_sliding,
        crack_x0=crack_x0, crack_y0=crack_y0,
        crack_length=CRACK_LENGTH,
        element_type='tri', conforming=True, verbose=False,
    )
    mesh, el_p, el_u, crack, h_mesh = builder.build()

    if filter_surrogate:
        filt = _filter_surrogate_by_alignment(crack)
        print(f"[box={box_scale}] surrogate filter: kept={filt['kept']}, "
              f"dropped={filt['dropped']}")
        if filt['dropped']:
            print(f"  dropped (fid, align, dmid, h): {filt['dropped_info']}")

    poro = CGPoroelastostatics(
        mesh=mesh, element_p=el_p, element_u=el_u,
        params=PARAMS, crack=crack,
        model_params={'T': CRACK_T_ROBIN, 'tangential_flow': False},
        intorder=6, scales=(SCALE_L, SCALE_EPS),
        bc={'v1': {'left': 0.0}, 'v2': {'bottom': 0.0},
            'dp_rate': {'left': 0.0, 'right': 0.0, 'top': 0.0, 'bottom': 0.0}},
        P_scale=None, verbose=False, free_memory=False,
        enforcement_type='nodal', apply_transform=True,
        include_taylor=False, include_hessian=False, include_sbm=False,
        density=DENSITY, crack_law='nonsmooth', rotate_crack_to_nt=True,
        bulk_viscosity={'mu_v': BULK_MU_V, 'lam_v': BULK_LAM_V},
    )

    poro.strip_multiplier_dynamics()
    projection, meta = poro.build_projection()
    dyn = poro.build_first_order_dynamic_system(meta, density=DENSITY)
    A_dyn = dyn['A']
    rhs_dyn = dyn['rhs']
    rhs_jac_dyn = dyn['rhs_jac']
    n_base = dyn['n_base']
    comp_slices = dyn['component_slices']

    Np = poro.basis_p.N
    Nu = poro.basis_u.N

    info = poro._transform_info
    n_c = int(poro.n_lambda_q)
    lam_s_sl = meta['lam_s_sl']

    off_jmpu = info['off_jmpu']
    n_iu = info['n_intf_u']
    off_jmpls = info['off_jmpls']
    n_lsig = info['n_lam_sig']

    jmpu_n_idx = np.arange(off_jmpu, off_jmpu + n_c)
    jmpu_t_idx = np.arange(off_jmpu + n_c, off_jmpu + n_iu)
    jmpu_all   = np.concatenate([jmpu_n_idx, jmpu_t_idx])
    jmpls_cols = np.arange(off_jmpls, off_jmpls + n_lsig)

    T_u_mat = dyn['T_u']
    u_idx   = dyn['u_state_indices']
    jmpu_local = np.searchsorted(u_idx, jmpu_all)
    D = T_u_mat[jmpu_local, :].tocsr()
    R_v = dyn.get('R_v')
    A_csr = poro.A.tocsr()
    B_u = (A_csr[Np:Np + Nu, :][:, jmpls_cols]).tocsr()
    if R_v is not None:
        B_u = (R_v @ B_u).tocsr()
    dirichlet_local = np.array(
        [d - Np for d in np.asarray(getattr(poro, '_dirichlet_dof_set', []),
                                    dtype=int)
         if Np <= d < Np + Nu], dtype=int,
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
        [sp.csr_matrix((n_extract, n_base)), D], format='csr',
    )
    B_c_dyn = sp.vstack(
        [sp.csr_matrix((n_base, B_u_perm.shape[1])), B_u_perm], format='csr',
    )

    s0_val = np.full(n_c, sigma_N_sliding / Sigma_scale)
    w0_val = np.full(n_c, tau_sliding_val / Sigma_scale)
    def get_s0_sliding(_y): return s0_val
    def get_w0_sliding(_y, k): return np.array([w0_val[int(k)]])

    flux_constraint = meta['constraints'][0]
    n_lam_s_dim = int(lam_s_sl.stop - lam_s_sl.start)
    zero_lam_s = {
        'g':     lambda zf, *_a, _n=n_lam_s_dim: np.zeros(_n),
        'dg_dy': lambda zf, *_a, _n=n_lam_s_dim: np.zeros((_n, _n)),
        'y_slice': lam_s_sl, 'q_slice': lam_s_sl,
    }

    cs = build_projected_radau_contact(
        A_dyn, rhs_dyn, np.zeros(A_dyn.shape[0]),
        contacts=contacts,
        C_extract=gap_extract_dyn, D_extract=vel_extract_dyn, B=B_c_dyn,
        constraints=[flux_constraint, zero_lam_s],
        component_slices=comp_slices,
        rhs_jac=rhs_jac_dyn,
        get_s0=get_s0_sliding, get_w0=get_w0_sliding,
        normal_r='auto', friction_r='auto',
        endpoint_inactive_handling='natural_map',
        reported_reaction_units='force',
    )

    n_aug = cs.y0.size
    nl_atol = np.full(n_aug, 1e-8)
    nl_rtol = np.full(n_aug, 1e-6)
    nl_atol[n_phys:] = 1e-10
    nl_rtol[n_phys:] = 0.0
    solver_opts = dict(cs.solver_opts)
    solver_opts.update(
        tol=1e-8, max_iter=500, rhs_jac=cs.rhs_jac,
        linear_solver='petsc',
        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu',
                       'pc_factor_mat_solver_type': 'mumps'},
    )
    integrator_opts = dict(cs.integrator_opts)
    integrator_opts.update({'stages': 2, 'use_coupled_newton': True})

    t0 = time.time()
    (
        t_arr, y_arr, h_arr, fk_arr, info_arr, attempts
    ) = solve_nivp.solve_ivp_ns(
        fun=cs.rhs, t_span=(0, tmax), y0=cs.y0.copy(),
        method='radau_iia', integrator_opts=integrator_opts,
        projection=cs.projection, solver='semismooth_newton',
        projection_opts={'component_slices': cs.component_slices},
        solver_opts=solver_opts,
        adaptive=True, h0=H_FIXED, rtol=1e-3, atol=1e-5,
        skip_error_indices=[0], active_set_filter=False,
        return_attempts=True,
        adaptive_opts=dict(h0=H_FIXED, h_min=1e-5, h_max=tmax/2,
                           h_up=2.0, h_down=0.5, method_order=1),
        nl_atol=nl_atol, nl_rtol=nl_rtol,
        component_slices=cs.component_slices,
        verbose=False,
        A=cs.A, dae_var_weight='auto',
    )
    elapsed = time.time() - t0

    n_acc = int(np.asarray(attempts, dtype=bool).sum()) if attempts is not None else len(t_arr)-1
    n_rej = len(attempts) - n_acc if attempts is not None else 0

    DISP_SCALE = SCALE_L * SCALE_EPS
    slip_arr = np.asarray(y_arr[:, jmpu_t_idx], dtype=float)
    gap_arr  = np.asarray(y_arr[:, jmpu_n_idx], dtype=float)
    r_pert   = y_arr[:, n_phys:n_phys + 2 * n_c]
    r_n = (r_pert[:, 0::2] + s0_val[None, :]) * Sigma_scale
    r_t = (r_pert[:, 1::2] + w0_val[None, :]) * Sigma_scale
    cone_margin_final = float(np.min(MU * r_n[-1] - np.abs(r_t[-1])))

    final_slip_nondim = np.asarray(slip_arr[-1], dtype=float)
    final_slip_phys   = final_slip_nondim * DISP_SCALE
    max_slip_phys     = float(np.max(np.abs(final_slip_phys)))

    # Crack-tangent coordinates of contact nodes for slip profile
    # The crack nodes are uniformly spaced along the crack with parameter ~[-a, +a].
    s_along = np.linspace(-a, a, n_c)
    sneddon_profile = (2.0 * delta_tau * (1.0 - NU) / G_SHEAR
                       * np.sqrt(np.clip(a**2 - s_along**2, 0.0, None)))

    # rhs residual at final time
    rhs_inf = float(np.max(np.abs(cs.rhs(float(t_arr[-1]), y_arr[-1]))))

    diag = {
        'box_scale':   float(box_scale),
        'n_elem':      int(n_elem),
        'h_mesh':      float(h_mesh),
        'n_contacts':  int(n_c),
        'a_over_L':    float(a / max(xmax - XMIN_BASE, ymax - YMIN_BASE)),
        'sneddon_max_slip_phys': float(sneddon_max_slip_phys),
        'numerical_max_slip_phys': max_slip_phys,
        'ratio_num_over_sneddon': float(max_slip_phys / max(sneddon_max_slip_phys, 1e-30)),
        'min_cone_margin_final_MPa': cone_margin_final,
        'rhs_inf_at_final': rhs_inf,
        'elapsed_s':   float(elapsed),
        'n_states':    int(len(t_arr)),
        'n_accepted':  int(n_acc),
        'n_rejected':  int(n_rej),
        'h_min':       float(np.min(h_arr)),
        'h_max':       float(np.max(h_arr)),
        'h_median':    float(np.median(h_arr)),
        't_final':     float(t_arr[-1]),
    }

    np.savez_compressed(
        os.path.join(outdir, f'box{box_scale}.npz'),
        s_along=s_along,
        slip_final_phys=final_slip_phys,
        sneddon_profile=sneddon_profile,
        slip_history=slip_arr * DISP_SCALE,
        t=np.asarray(t_arr),
        h=np.asarray(h_arr),
        cone_margin_final=MU * r_n[-1] - np.abs(r_t[-1]),
    )

    print(f"[box={box_scale}] elapsed={elapsed:.1f}s, n_states={len(t_arr)}, "
          f"acc/rej={n_acc}/{n_rej}, max|slip|={max_slip_phys:.3e} m "
          f"(Sneddon={sneddon_max_slip_phys:.3e}, ratio={diag['ratio_num_over_sneddon']:.3f})")

    return diag


def main(argv):
    if len(argv) > 1:
        box_scales = [float(a) for a in argv[1:]]
    else:
        box_scales = [1.0, 2.0, 4.0]
    outdir = '/tmp/box_sweep'
    os.makedirs(outdir, exist_ok=True)
    diags = []
    for bs in box_scales:
        try:
            d = run_box(bs, outdir)
            diags.append(d)
        except Exception as exc:
            print(f"[box={bs}] FAILED: {exc!r}")
            import traceback; traceback.print_exc()

    with open(os.path.join(outdir, 'summary.json'), 'w') as f:
        json.dump(diags, f, indent=2)
    print(f"\nsummary -> {outdir}/summary.json")

    # ── Comparative plot: slip profile + ratio vs a/L ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(diags)))
    for d, c in zip(diags, colors):
        npz = np.load(os.path.join(outdir, f'box{d["box_scale"]}.npz'))
        s = npz['s_along']
        slip = npz['slip_final_phys']
        sn = npz['sneddon_profile']
        axes[0].plot(s, np.abs(slip), 'o-', color=c, lw=1.5,
                     label=f'box={d["box_scale"]}, a/L={d["a_over_L"]:.3f}')
        axes[0].plot(s, sn, '--', color=c, alpha=0.6)
    axes[0].set_xlabel('s along crack')
    axes[0].set_ylabel('|slip| at final time (m)')
    axes[0].set_title('Slip profile vs Sneddon (dashed = inf-medium upper bound)')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    aoL  = [d['a_over_L'] for d in diags]
    smax = [d['numerical_max_slip_phys'] for d in diags]
    sned = [d['sneddon_max_slip_phys']    for d in diags]
    times = [d['elapsed_s'] for d in diags]
    axes[1].plot(aoL, smax, 'rs-', lw=1.5, label='numerical max |slip|')
    axes[1].plot(aoL, sned, 'k--', label='Sneddon upper bound')
    axes[1].set_xlabel('a/L')
    axes[1].set_ylabel('max |slip| (m)')
    axes[1].set_xscale('log'); axes[1].invert_xaxis()
    axes[1].set_title('Convergence vs a/L (lower a/L = bigger box)')
    axes[1].legend(); axes[1].grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'box_scale_sweep.png'), dpi=130)
    plt.close(fig)
    print(f"plot -> {outdir}/box_scale_sweep.png")

    print('\n=== Summary table ===')
    print(f'{"box":>5}  {"a/L":>7}  {"n_elem":>6}  {"n_c":>4}  '
          f'{"max|slip| (m)":>14}  {"Sneddon (m)":>13}  {"ratio":>6}  '
          f'{"acc":>4}  {"rej":>4}  {"time (s)":>8}')
    for d in diags:
        print(f'{d["box_scale"]:5.1f}  {d["a_over_L"]:7.4f}  '
              f'{d["n_elem"]:6d}  {d["n_contacts"]:4d}  '
              f'{d["numerical_max_slip_phys"]:14.4e}  '
              f'{d["sneddon_max_slip_phys"]:13.4e}  '
              f'{d["ratio_num_over_sneddon"]:6.3f}  '
              f'{d["n_accepted"]:4d}  {d["n_rejected"]:4d}  '
              f'{d["elapsed_s"]:8.1f}')


if __name__ == '__main__':
    main(sys.argv)
